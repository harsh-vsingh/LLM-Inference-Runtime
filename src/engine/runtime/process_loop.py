import asyncio
import logging
import math
import time
from typing import Optional

from engine.memory.allocator import BlockAllocator
from engine.memory.radix_cache import RadixCache
from engine.metrics.engine_metrics import EngineMetrics
from engine.runtime.step_executor import StepExecutor
from engine.scheduling.scheduler import Scheduler
from engine.sequence import Sequence

logger = logging.getLogger(__name__)


class ProcessLoop:
    def __init__(
        self,
        scheduler: Scheduler,
        step_executor: StepExecutor,
        allocator: BlockAllocator,
        radix_cache: RadixCache,
        engine_metrics: EngineMetrics,
        block_size: int,
        chunked_prefill_enabled_fn,
    ):
        self.scheduler = scheduler
        self.step_executor = step_executor
        self.allocator = allocator
        self.radix_cache = radix_cache
        self.engine_metrics = engine_metrics
        self.block_size = block_size
        self._chunked_prefill_enabled_fn = chunked_prefill_enabled_fn

        self.wakeup_event = asyncio.Event()
        self._stopping = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run())
        return self._task

    def request_stop(self) -> None:
        self._stopping = True
        self.wakeup_event.set()

    async def cancel_and_wait(self, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    async def _run(self) -> None:
        try:
            while True:
                await self.wakeup_event.wait()

                while self.scheduler.has_unfinished_sequences():
                    if self._stopping and not self.scheduler.running:
                        break

                    now = time.time()
                    for seq in self.scheduler.running:
                        seq.request.metrics.record_admitted()

                    await self._run_one_step(now)

                self.wakeup_event.clear()

                if self._stopping and not self.scheduler.has_unfinished_sequences():
                    return
        except asyncio.CancelledError:
            raise

    async def _run_one_step(self, now: float) -> None:
        prefill_seqs, decode_seqs = self.scheduler.step()

        try:
            step_start = time.time()
            result = await asyncio.to_thread(
                self.step_executor.run_step,
                prefill_seqs,
                decode_seqs,
                self._chunked_prefill_enabled_fn(),
            )
            step_duration = time.time() - step_start

            self.engine_metrics.record_step(
                step_duration, result.is_prefill, len(result.outputs)
            )
            if result.preempted:
                for _ in result.preempted:
                    self.engine_metrics.record_preemption()

            for seq, new_text in result.outputs:
                await seq.request.put_token(new_text)

                if seq.is_finished():
                    await self._finish_sequence(seq)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unhandled error during engine step")
            await self._fail_batch(prefill_seqs, decode_seqs)
            await asyncio.sleep(0.1)

    async def _finish_sequence(self, seq: Sequence) -> None:
        seq.request.metrics.record_finished()

        full_tokens = seq.prompt_token_ids + seq.generated_token_ids
        full_block_count = math.ceil(len(full_tokens) / self.block_size)
        blocks = seq.block_table[:full_block_count]

        self.radix_cache.insert(full_tokens, blocks)
        self.allocator.decref(seq.block_table)
        seq.block_table.clear()

        self.step_executor.drop_detokenizer(seq)

        self.engine_metrics.record_finished_request(
            ttft=seq.request.metrics.ttft,
            tpot=seq.request.metrics.tpot,
            throughput=seq.request.metrics.throughput,
        )

        logger.info(
            "request finished id=%s tokens_generated=%d prompt_tokens=%d "
            "ttft=%.4fs tpot=%.4fs throughput=%.2f tok/s",
            seq.request.request_id,
            seq.request.metrics.tokens_generated,
            len(seq.prompt_token_ids),
            seq.request.metrics.ttft,
            seq.request.metrics.tpot,
            seq.request.metrics.throughput,
        )

        await seq.request.finish()

    async def _fail_batch(self, prefill_seqs, decode_seqs) -> None:
        affected = list({id(s): s for s in (prefill_seqs + decode_seqs)}.values())

        for seq in affected:
            if seq.block_table:
                self.allocator.decref(seq.block_table)
                seq.block_table.clear()

            self.step_executor.drop_detokenizer(seq)

            if seq in self.scheduler.running:
                self.scheduler.running.remove(seq)

            seq.request.is_aborted = True
            await seq.request.finish()
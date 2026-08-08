import asyncio
import logging
import os
import time
import uuid
from typing import List, Optional

from engine.config import EngineConfig
from engine.errors import AdmissionError, AdmissionRejectReason
from engine.memory.allocator import BlockAllocator
from engine.memory.prefix_match import match_prefix_for_new_run
from engine.memory.radix_cache import RadixCache
from engine.metrics.engine_metrics import EngineMetrics
from engine.metrics.snapshot import build_metrics_snapshot
from engine.request import InferenceRequest
from engine.runtime.block_admission import BlockAdmission
from engine.runtime.process_loop import ProcessLoop
from engine.runtime.sampler import sample
from engine.runtime.step_executor import StepExecutor
from engine.scheduling.scheduler import Scheduler
from engine.sequence import Sequence
from models.llama import init_flashinfer_state, patch_llama_model

logger = logging.getLogger(__name__)


class AsyncInferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        max_num_batched_tokens: int = 4096,
        max_num_seqs: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.instance_id = os.environ.get("ENGINE_INSTANCE_ID", f"engine-{uuid.uuid4().hex[:8]}")
        self.is_draining = False

        self.config = EngineConfig()

        self.block_size = 16
        self.max_chunk_size = 512

        model_config = model.config
        self.num_layers = model_config.num_hidden_layers
        self.num_qo_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads

        self.allocator = BlockAllocator(
            num_blocks=4096,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=model.dtype,
            device=model.device,
        )
        self.radix_cache = RadixCache(block_size=self.block_size, allocator=self.allocator)
        self.scheduler = Scheduler(
            max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs
        )
        self.metrics = EngineMetrics()

        patch_llama_model(model)
        init_flashinfer_state(
            kv_pool=self.allocator.kv_pool,
            page_size=self.block_size,
            device=model.device,
        )

        self._block_admission = BlockAdmission(
            allocator=self.allocator,
            radix_cache=self.radix_cache,
            scheduler=self.scheduler,
            block_size=self.block_size,
            on_preempt=lambda seq: self.metrics.record_preemption(),
        )

        self._step_executor = StepExecutor(
            model=self.model,
            tokenizer=self.tokenizer,
            allocator=self.allocator,
            radix_cache=self.radix_cache,
            block_size=self.block_size,
            max_chunk_size=self.max_chunk_size,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sampler=sample,
            allocate_blocks_fn=self._block_admission.allocate_blocks_if_needed,
            preempt_fn=self._block_admission.preempt_sequence,
        )

        self._process_loop = ProcessLoop(
            scheduler=self.scheduler,
            step_executor=self._step_executor,
            allocator=self.allocator,
            radix_cache=self.radix_cache,
            engine_metrics=self.metrics,
            block_size=self.block_size,
            chunked_prefill_enabled_fn=lambda: self.config.enable_chunked_prefill,
        )

        self._metrics_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._process_loop.start()
        self._metrics_task = asyncio.create_task(self._log_system_metrics())

    async def add_request(self, request: InferenceRequest) -> None:
        if self.is_draining:
            raise AdmissionError(
                AdmissionRejectReason.ENGINE_DRAINING,
                "engine is shutting down and is not accepting new requests",
            )

        prompt_ids = self.tokenizer(request.prompt).input_ids
        request.metrics.prompt_tokens = len(prompt_ids)

        seq = Sequence(
            request=request,
            prompt_token_ids=prompt_ids,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        matched_tokens: List[int] = []
        matched_blocks: List[int] = []

        if self.config.enable_prefix_cache:
            matched_tokens, matched_blocks = match_prefix_for_new_run(
                self.radix_cache, self.allocator, prompt_ids
            )
            request.metrics.prefix_depth_tokens = len(matched_tokens)
            request.metrics.cache_blocks_reused = len(matched_blocks)

        self.metrics.record_prompt(
            prompt_tokens=len(prompt_ids), cached_tokens=len(matched_tokens)
        )

        seq.cached_prefix_len = len(matched_tokens)
        seq.computed_len = seq.cached_prefix_len
        seq.block_table = list(matched_blocks)

        self.scheduler.add_sequence(seq)
        self.metrics.record_admitted()
        self._process_loop.wakeup_event.set()

    def get_metrics(self) -> dict:
        return build_metrics_snapshot(self)

    async def shutdown(self, timeout: float = 30.0) -> None:
        logger.info("engine shutdown requested, draining in-flight requests")
        self.is_draining = True
        self._process_loop.request_stop()

        deadline = time.time() + timeout
        while self.scheduler.has_unfinished_sequences() and time.time() < deadline:
            await asyncio.sleep(0.1)

        if self.scheduler.has_unfinished_sequences():
            logger.warning(
                "shutdown timeout reached with %d sequence(s) still in flight; aborting them",
                len(self.scheduler.running) + len(self.scheduler.waiting),
            )
            for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
                seq.request.is_aborted = True
                await seq.request.finish()
            self.scheduler.running.clear()
            self.scheduler.waiting.clear()

        await self._process_loop.cancel_and_wait(timeout=5.0)

        if self._metrics_task is not None:
            self._metrics_task.cancel()
            try:
                await asyncio.wait_for(self._metrics_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("engine shutdown complete")

    async def _log_system_metrics(self) -> None:
        try:
            while True:
                await asyncio.sleep(5)
                if not self.scheduler.has_unfinished_sequences():
                    continue

                interval = self.metrics.flush_interval()

                logger.info(
                    "[METRICS] batch=%d/%d waiting=%d cache_hit=%.1f%% "
                    "tps=%.1f avg_prefill=%.2fms avg_decode=%.2fms",
                    len(self.scheduler.running),
                    self.scheduler.max_num_seqs,
                    len(self.scheduler.waiting),
                    self.metrics.cache_hit_rate_pct,
                    self.metrics.current_tokens_per_second,
                    interval["avg_prefill_ms"],
                    interval["avg_decode_ms"],
                )
        except asyncio.CancelledError:
            pass
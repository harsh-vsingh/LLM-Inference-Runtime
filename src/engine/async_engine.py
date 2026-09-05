import asyncio
import logging
import os
import time
import uuid

from engine.config import EngineConfig
from engine.errors import AdmissionError, AdmissionRejectReason
from engine.memory.allocator import BlockAllocator
from engine.memory.capacity import estimate_num_blocks
from engine.memory.prefix_match import match_prefix_for_new_run
from engine.memory.radix_cache import RadixCache
from engine.metrics.engine_metrics import EngineMetrics
from engine.metrics.snapshot import build_metrics_snapshot
from engine.request import InferenceRequest
from engine.runtime.process_loop import ProcessLoop
from engine.runtime.sampler import sample
from engine.runtime.step_executor import StepExecutor
from engine.scheduling.admission_controller import AdmissionController
from engine.sequence import Sequence
from models.llama import init_flashinfer_state, patch_llama_model

logger = logging.getLogger(__name__)


class AsyncInferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        max_num_batched_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.instance_id = os.environ.get("ENGINE_INSTANCE_ID", f"engine-{uuid.uuid4().hex[:8]}")
        self.is_draining = False

        self.config = EngineConfig()

        self.block_size = self.config.block_size
        self.max_num_batched_tokens = max_num_batched_tokens

        model_config = model.config
        self.num_layers = model_config.num_hidden_layers
        self.num_qo_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads

        # num_blocks is hardware-derived. It depends
        # on how much GPU memory is left after model weights are loaded,
        # which varies by GPU and by model size/quantization. Falls back
        # to EngineConfig.num_blocks rather than failing.
        estimated_blocks = estimate_num_blocks(
            device=model.device,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=model.dtype,
        )
        if estimated_blocks > 0:
            self.config.num_blocks = estimated_blocks

        self.allocator = BlockAllocator(
            num_blocks=self.config.num_blocks,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=model.dtype,
            device=model.device,
        )
        self.radix_cache = RadixCache(block_size=self.block_size, allocator=self.allocator)
        self.metrics = EngineMetrics()
        self.scheduler = AdmissionController(
            config=self.config,
            allocator=self.allocator,
            radix_cache=self.radix_cache,
            block_size=self.block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            on_reject=lambda reason: self.metrics.record_rejected(),
            engine_metrics=self.metrics,
        )

        patch_llama_model(model)
        init_flashinfer_state(
            kv_pool=self.allocator.kv_pool,
            page_size=self.block_size,
            device=model.device,
        )

        self._step_executor = StepExecutor(
            model=self.model,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            max_chunk_size=self.config.max_chunk_size,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sampler=sample,
        )

        self.scheduler._drop_detokenizer_fn = self._step_executor.drop_detokenizer

        self._process_loop = ProcessLoop(
            scheduler=self.scheduler,
            step_executor=self._step_executor,
            allocator=self.allocator,
            radix_cache=self.radix_cache,
            engine_metrics=self.metrics,
            block_size=self.block_size,
        )

        self._metrics_task: asyncio.Task | None = None

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

        # NOTE: seq sits in `waiting` briefly below with an empty
        # block_table before being fully initialized. Safe only because
        # there is no `await` between add_sequence() and the block_table
        # assignment. Do not insert an await in this span without 
        # reconsidering this ordering.
        self.scheduler.add_sequence(seq)

        matched_tokens: list[int] = []
        matched_blocks: list[int] = []

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
            except (TimeoutError, asyncio.CancelledError):
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
                    "[METRICS] running=%d waiting=%d cache_hit=%.1f%% "
                    "tps=%.1f avg_prefill=%.2fms avg_decode=%.2fms",
                    len(self.scheduler.running),
                    len(self.scheduler.waiting),
                    self.metrics.cache_hit_rate_pct,
                    self.metrics.current_tokens_per_second,
                    interval["avg_prefill_ms"],
                    interval["avg_decode_ms"],
                )
        except asyncio.CancelledError:
            pass
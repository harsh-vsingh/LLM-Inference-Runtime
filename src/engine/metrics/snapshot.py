"""
Builds the public metrics snapshot returned by AsyncInferenceEngine.get_metrics().

RESTRUCTURED SHAPE - see MIGRATION notes below for the old -> new key mapping
so dashboards can be updated. This was approved as a breaking change since
nothing outside metrics_service.py depends on the exact old shape.

Design goals for the new shape:
  - `capacity` and `admission` sections are new: they expose queue depth,
    KV headroom (in tokens, not just raw block counts), and rejection
    counters - this is exactly what a future load-aware gateway needs to
    make routing decisions, so it's first-class here rather than bolted on.
  - `instance` section carries a stable identifier + draining flag, so a
    gateway can tell "don't route here, this instance is shutting down"
    from metrics/health alone, without a separate protocol.
  - Every other section keeps roughly the old grouping (scheduler,
    allocator, performance, gpu) for readability, just cleaned up.

MIGRATION (old key -> new key):
  scheduler.active_prefills          -> (removed, was always hardcoded 0)
  scheduler.preempts                 -> (removed, was always hardcoded 0)
                                         now: admission.total_preemptions
  performance.avg_decode_batch_size  -> performance.avg_decode_batch_size
                                         (now a real running average, was
                                         `len(scheduler.running)` - not an
                                         average at all, just current size)
  debug_requests[].request_id[:8]    -> debug_requests[].request_id
                                         (full id; truncation was lossy for
                                         no benefit, callers can slice)
  (new) capacity.*                   -> queue depth, KV headroom in tokens
  (new) admission.*                  -> admitted/rejected/preempted counters
  (new) instance.*                   -> instance_id, is_draining, uptime_s
"""

import time
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from engine.async_engine import AsyncInferenceEngine


def _avg_decode_batch_size(scheduler) -> float:
    """
    Average decode batch size *as experienced by sequences that have
    actually decoded at least once*, i.e. the mean of each qualifying
    running sequence's own running average (sum_decode_batch_size /
    decode_steps).

    Sequences still in prefill (decode_steps == 0 so far) are excluded
    from this average rather than included as a 0 - otherwise, under
    heavy sustained load where a large fraction of `running` sequences
    are mid-prefill at any given poll, this number gets diluted toward 0
    and misleadingly suggests "no decoding is happening" even while
    plenty of decode steps are actively running for the sequences that
    have reached that stage. This was observed directly: dashboard showed
    avg_decode_batch_size=0.0 with Running=100 and nonzero throughput.
    """
    qualifying = [
        seq for seq in scheduler.running
        if seq.request.metrics.decode_steps > 0
    ]
    if not qualifying:
        return 0.0
    return sum(seq.request.metrics.avg_decode_batch_size for seq in qualifying) / len(qualifying)


def build_metrics_snapshot(engine: "AsyncInferenceEngine") -> Dict[str, Any]:
    allocator = engine.allocator
    scheduler = engine.scheduler
    metrics = engine.metrics

    total_blocks = allocator.num_blocks
    free_blocks = allocator.get_available_blocks()
    active_blocks = total_blocks - free_blocks
    block_size = allocator.block_size

    bytes_per_token = (
        engine.num_layers * 2 * engine.num_kv_heads * engine.head_dim
        * torch.tensor([], dtype=engine.model.dtype).element_size()
    )
    block_bytes = block_size * bytes_per_token
    kv_cache_gb = (total_blocks * block_bytes) / (1024 ** 3)

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

    return {
        "instance": {
            "instance_id": engine.instance_id,
            "is_draining": engine.is_draining,
            "uptime_s": metrics.uptime_seconds,
        },

        "scheduler": {
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
        },

        "capacity": {
            # Queue-depth headroom, for admission-aware routing.
            "queue_depth": len(scheduler.waiting) + len(scheduler.running),
            "max_queue_depth": engine.config.max_queue_depth,
            # KV headroom expressed in tokens (free_blocks * block_size),
            # which is what a router actually needs to estimate whether a
            # given prompt could be admitted - raw block counts require the
            # caller to know block_size to be useful.
            "free_kv_tokens": free_blocks * block_size,
            "total_kv_tokens": total_blocks * block_size,
        },

        "admission": {
            "total_admitted": metrics.total_requests_admitted,
            "total_rejected": metrics.total_requests_rejected,
            "total_preemptions": metrics.total_preemptions,
        },

        "allocator": {
            "total_blocks": total_blocks,
            "free_blocks": free_blocks,
            "active_blocks": active_blocks,
            "utilization_pct": (active_blocks / total_blocks * 100) if total_blocks else 0.0,
        },

        "performance": {
            "tokens_per_second": metrics.current_tokens_per_second,
            "cache_hit_rate_pct": metrics.cache_hit_rate_pct,
            "avg_ttft_ms": metrics.avg_ttft_ms,
            "avg_tpot_ms": metrics.avg_tpot_ms,
            "avg_decode_batch_size": _avg_decode_batch_size(scheduler),
        },

        "gpu": {
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "kv_cache_gb": kv_cache_gb,
        },

        "debug_requests": [
            {
                "request_id": seq.request.request_id,
                "prompt_tokens": seq.original_prompt_len,
                "generated_tokens": len(seq.generated_token_ids),
                "cached_prefix": seq.cached_prefix_len,
                "status": seq.status.name,
            }
            for seq in scheduler.running[:32]
        ],
    }
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from engine.async_engine import AsyncInferenceEngine


def _avg_decode_batch_size_per_request(scheduler) -> float:
    """
    Average, across currently-running sequences, of each sequence's own
    lifetime-average decode batch size.
    """
    qualifying = [
        seq for seq in scheduler.running
        if seq.request.metrics.decode_steps > 0
    ]
    if not qualifying:
        return 0.0
    return sum(seq.request.metrics.avg_decode_batch_size for seq in qualifying) / len(qualifying)


def build_metrics_snapshot(engine: "AsyncInferenceEngine") -> dict[str, Any]:
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

    latency = metrics.latency_summary()
    step_times = metrics.step_time_summary()

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
            "queue_depth": len(scheduler.waiting) + len(scheduler.running),
            "max_queue_depth": engine.config.max_queue_depth,
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

            "ttft_ms": latency["ttft_ms"],
            "tpot_ms": latency["tpot_ms"],
            "prefill_step_ms": step_times["prefill_ms"],
            "decode_step_ms": step_times["decode_ms"],

            "current_decode_batch_size": metrics.current_decode_batch_size,
            "avg_decode_batch_size_per_request": _avg_decode_batch_size_per_request(scheduler),
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
                "queue_latency_s": seq.request.metrics.queue_latency,
                "total_time_s": seq.request.metrics.total_time,
            }

            for seq in sorted(
                scheduler.running,
                key=lambda s: s.request.metrics.total_time,
                reverse=True,
            )[:32]
        ],
    }
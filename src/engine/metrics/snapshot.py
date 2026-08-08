from typing import TYPE_CHECKING, Any, Dict
import torch

if TYPE_CHECKING:
    from engine.async_engine import AsyncInferenceEngine


def _avg_decode_batch_size(scheduler) -> float:
    running = scheduler.running
    if not running:
        return 0.0
    return sum(seq.request.metrics.avg_decode_batch_size for seq in running) / len(running)


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
            "max_num_seqs": scheduler.max_num_seqs,
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
                "prompt_tokens": len(seq.prompt_token_ids),
                "generated_tokens": len(seq.generated_token_ids),
                "cached_prefix": seq.cached_prefix_len,
                "status": seq.status.name,
            }
            for seq in scheduler.running[:32]
        ],
    }
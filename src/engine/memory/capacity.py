"""
Derives KV cache capacity from GPU memory.
"""

import logging
import torch

logger = logging.getLogger(__name__)


_KV_CACHE_MEMORY_FRACTION = 0.85
_MIN_RESERVED_BYTES = 2 * 1024 ** 3


def bytes_per_block(
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """
    Bytes needed for one KV cache block.
    (num_layers, num_blocks, 2, num_kv_heads, block_size, head_dim)
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    per_block_shape = (num_layers, 2, num_kv_heads, block_size, head_dim)
    numel = 1
    for dim in per_block_shape:
        numel *= dim
    return numel * elem_size


def estimate_num_blocks(
    device: torch.device,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    min_blocks: int = 256,
) -> int:
    """
    Computes how many KV cache blocks fit in currently-free device memory.
    Must be called after the model is loaded onto device.
    Raises RuntimeError if the resulting capacity is unusably small
    """
    if device.type != "cuda":
        logger.warning(
            "estimate_num_blocks: non-CUDA device (%s), cannot query free "
            "memory - falling back to EngineConfig.num_blocks default",
            device,
        )
        return -1

    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    reserved = max(_MIN_RESERVED_BYTES, int(free_bytes * (1 - _KV_CACHE_MEMORY_FRACTION)))
    usable_bytes = max(0, free_bytes - reserved)

    per_block = bytes_per_block(block_size, num_layers, num_kv_heads, head_dim, dtype)
    num_blocks = usable_bytes // per_block

    logger.info(
        "KV cache sizing: free=%.2fGiB reserved=%.2fGiB usable=%.2fGiB "
        "bytes/block=%d -> num_blocks=%d (%.2fGiB of KV cache)",
        free_bytes / 1024**3,
        reserved / 1024**3,
        usable_bytes / 1024**3,
        per_block,
        num_blocks,
        (num_blocks * per_block) / 1024**3,
    )

    if num_blocks < min_blocks:
        raise RuntimeError(
            f"Only {num_blocks} KV cache blocks fit in free GPU memory "
            f"({free_bytes / 1024**3:.2f}GiB free) - need at least "
            f"{min_blocks}. Model weights may be too large for this GPU, "
            f"or another process is holding memory."
        )

    return int(num_blocks)
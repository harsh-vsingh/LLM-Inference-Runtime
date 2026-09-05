"""
Engine configuration.
"""
from dataclasses import dataclass


@dataclass
class EngineConfig:
    enable_chunked_prefill: bool = True
    enable_prefix_cache: bool = True

    # Tokens per KV cache block. Fixed at engine construction time
    block_size: int = 16

    # Number of KV cache blocks the allocator owns. This default is a
    # fallback only, used when hardware-derived sizing is unavailable.
    num_blocks: int = 4096

    # Hard cap on (waiting + running) sequences. This bounds how 
    # much work can be queued 

    max_queue_depth: int = 512

    # How long a request may sit admitted-but-not-running before it's
    # failed with QUEUE_TIMEOUT. measured cumulatively from 
    # the request's original arrival time (not reset on preemption)
    max_queue_wait_seconds: float = 120.0

    # How many times a running sequence may be preempted
    max_preemption_retries: int = 3

    # Sequences past this fraction of their max_new_tokens are excluded
    # from preemption, except when every running sequence is above the threshold,
    # case protection is waived to avoid deadlocking then.
    completion_protection_threshold: float = 0.8

    # Cache only eviction fires when free blocks drop below
    # min(eviction_low_water_blocks, eviction_low_water_pct * total_blocks)
    eviction_low_water_blocks: int = 64
    eviction_low_water_pct: float = 0.1

    # Prefill chunk size cap per sequence per step.
    max_chunk_size: int = 512
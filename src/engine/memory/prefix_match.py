"""
Prefix-cache lookup helper shared by initial admission and preemption
requeueing.
"""

from typing import List, Tuple
from engine.memory.allocator import BlockAllocator
from engine.memory.radix_cache import RadixCache


def match_prefix_for_new_run(
    radix_cache: RadixCache,
    allocator: BlockAllocator,
    token_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Looks up the cached prefix for `token_ids` and returns
    (matched_tokens, matched_blocks)guaranteeing at least 
    one uncomputed token remains.
    """
    matched_tokens, matched_blocks = radix_cache.match_prefix(token_ids)

    full_hit = len(matched_tokens) == len(token_ids) and len(matched_blocks) > 0
    if full_hit:
        dropped_block = matched_blocks[-1]
        
        tokens_in_last_block = len(token_ids) % radix_cache.block_size
        if tokens_in_last_block == 0:
            tokens_in_last_block = radix_cache.block_size
            
        matched_tokens = matched_tokens[:-tokens_in_last_block]
        matched_blocks = matched_blocks[:-1]
        allocator.decref([dropped_block])

    return matched_tokens, matched_blocks
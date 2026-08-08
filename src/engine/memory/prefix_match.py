"""
Prefix-cache lookup helper shared by initial admission and preemption
requeueing.

If match_prefix() returns a FULL match (the entire prompt is already
cached), the original code deliberately drops the last matched block
before using the result, so that computed_len ends up strictly less than
len(prompt_ids) - leaving at least one uncomputed token so the engine has
something to run a forward pass on to produce the first generated token.
A sequence with computed_len == len(prompt_ids) and zero generated tokens
would have nothing left to compute and could never produce output.

This logic was duplicated verbatim in two places in the original
async_engine.py (initial admission, and _preempt_sequence). Consolidated
here as one function, with the ref-count bookkeeping tied to it made
explicit: match_prefix() increfs everything it matches (see radix_cache.py),
so if we drop the last block here we must decref that one block back out,
or it leaks a permanent phantom reference.
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
    (matched_tokens, matched_blocks) ready to seed a sequence's
    block_table, guaranteeing at least one uncomputed token remains.

    Ref-counting: the returned matched_blocks are already increfed on the
    allocator (via match_prefix) - the caller takes ownership of that
    reference by assigning it to a sequence's block_table, exactly as
    before.
    """
    matched_tokens, matched_blocks = radix_cache.match_prefix(token_ids)

    full_hit = len(matched_tokens) == len(token_ids) and len(matched_blocks) > 0
    if full_hit:
        dropped_block = matched_blocks[-1]
        matched_tokens = matched_tokens[: -radix_cache.block_size]
        matched_blocks = matched_blocks[:-1]
        # match_prefix() increfed every block it matched, including the one
        # we're now holding back - release that one reference so the cache
        # doesn't end up with a phantom extra holder for a block this
        # sequence will never actually reference from its block_table.
        allocator.decref([dropped_block])

    return matched_tokens, matched_blocks
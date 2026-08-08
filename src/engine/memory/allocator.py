"""
KV cache block allocator.

Fixes vs. the original memory.py:
  - allocate() raises BlockAllocationError (a specific, catchable type)
    instead of a bare RuntimeError. The old bare RuntimeError was caught by
    async_engine.py's broad `except Exception` around the whole step loop,
    which then force-finished EVERY running sequence in the batch, not just
    the one that failed to allocate. BlockAllocationError is meant to be
    caught narrowly, at the single call site that allocates for one
    sequence, so one sequence's OOM only affects that sequence.
  - free_all() now defaults to refusing to run if any block is still
    referenced (ref_count > 0), i.e. if any sequence is currently running
    and holding blocks. The old version unconditionally reset every
    ref-count to 0 and returned every block to the free list regardless of
    live holders - reachable in production via the admin CLEAR_CACHE
    endpoint, causing silent KV cache corruption/cross-request data
    leakage for any request that was mid-generation when it fired.
    Call with force=True to keep the old unconditional behavior
    (e.g. for tests/benchmarks where the caller has already verified
    nothing is running).
"""

from typing import List

import torch

from engine.errors import BlockAllocationError


class BlockAllocator:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.kv_pool = torch.empty(
            (
                num_layers,
                num_blocks,
                2,
                num_kv_heads,
                block_size,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )

        self.free_blocks: List[int] = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def allocate(self, num_blocks_needed: int) -> List[int]:
        if num_blocks_needed <= 0:
            return []

        if len(self.free_blocks) < num_blocks_needed:
            raise BlockAllocationError(
                f"Not enough free KV blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]

        for block in allocated:
            self.ref_counts[block] = 1

        return allocated

    def incref(self, blocks: List[int]) -> None:
        for block in blocks:
            self.ref_counts[block] += 1

    def decref(self, blocks: List[int]) -> None:
        for block in blocks:
            if self.ref_counts[block] <= 0:
                # Defensive: double-decref would silently wrap negative and
                # re-add an already-free block to free_blocks, corrupting
                # the free list with a duplicate entry. Surface it instead.
                raise BlockAllocationError(
                    f"decref on block {block} with ref_count already "
                    f"{self.ref_counts[block]} - double-free bug at call site"
                )
            self.ref_counts[block] -= 1
            if self.ref_counts[block] == 0:
                self.free_blocks.append(block)

    def get_available_blocks(self) -> int:
        return len(self.free_blocks)

    def has_live_references(self) -> bool:
        return any(rc > 0 for rc in self.ref_counts)

    def free_all(self, force: bool = False) -> None:
        """
        Forces all blocks back into the free pool.

        By default, refuses if any block still has live references (i.e.
        some sequence is currently running/holding KV state), to avoid
        corrupting in-flight generations. Pass force=True to override this
        check - only safe when the caller has independently guaranteed no
        sequences are running (e.g. a benchmark harness between runs).
        """
        if not force and self.has_live_references():
            raise BlockAllocationError(
                "free_all() refused: blocks are still referenced by "
                "running sequences. Pass force=True to override, or wait "
                "for in-flight requests to finish."
            )

        self.free_blocks = list(range(self.num_blocks))
        for i in range(self.num_blocks):
            self.ref_counts[i] = 0
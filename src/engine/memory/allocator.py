"""
KV cache block allocator.
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

    def try_allocate(self, num_blocks_needed: int, radix_cache=None) -> bool:
        """
        Ensures num_blocks_needed free blocks are available, evicting
        from radix_cache if given. Does NOT allocate the blocks.
        Call allocate()
        """
        if num_blocks_needed <= 0:
            return True

        deficit = num_blocks_needed - len(self.free_blocks)
        if deficit > 0 and radix_cache is not None:
            radix_cache.evict_lru(deficit)

        return len(self.free_blocks) >= num_blocks_needed

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
        By default, refuses if any block still has live references
        Pass force=True to override this check(unsafe).
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
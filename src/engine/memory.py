import torch
from typing import List


class BlockAllocator:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str
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
            device=device
        )

        self.free_blocks: List[int] = list(range(num_blocks))

        self.ref_counts = [0] * num_blocks

    def allocate(self, num_blocks_needed: int) -> List[int]:
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"OOM: need {num_blocks_needed}, have {len(self.free_blocks)}"
            )

        allocated = self.free_blocks[:num_blocks_needed]

        self.free_blocks = self.free_blocks[num_blocks_needed:]

        for block in allocated:
            self.ref_counts[block] = 1

        return allocated

    def incref(self, blocks: List[int]):
        for block in blocks:
            self.ref_counts[block] += 1

    def decref(self, blocks: List[int]):
        for block in blocks:
            self.ref_counts[block] -= 1

            if self.ref_counts[block] == 0:
                self.free_blocks.append(block)

    def get_available_blocks(self) -> int:
        return len(self.free_blocks)
    
    def free_all(self):
        """Forces all blocks back into the free pool."""
        self.free_blocks = list(range(self.num_blocks))
        for i in range(self.num_blocks):
            self.ref_counts[i] = 0

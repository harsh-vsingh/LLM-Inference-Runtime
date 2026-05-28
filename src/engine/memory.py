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
            (num_layers, num_blocks, 2, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device
        )
        
        self.free_blocks: List[int] = list(range(num_blocks))
        
    def allocate(self, num_blocks_needed: int) -> List[int]:
        """Reserves a given number of block indices from the free pool."""
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Out of Memory: Need {num_blocks_needed} blocks, have {len(self.free_blocks)}")
            
        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        return allocated

    def free(self, block_indices: List[int]):
        """Returns block indices back to the free pool."""
        self.free_blocks.extend(block_indices)
        
    def get_available_blocks(self) -> int:
        return len(self.free_blocks)
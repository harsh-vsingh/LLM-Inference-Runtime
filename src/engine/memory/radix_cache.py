"""
Prefix-caching radix tree over KV blocks.
match_prefix() increfs the matched blocks.
"""

import time
from typing import Dict, List, Optional, Tuple
from engine.memory.allocator import BlockAllocator


class RadixNode:
    __slots__ = ("children", "block_id", "parent", "chunk", "last_access_time")

    def __init__(self):
        self.children: Dict[Tuple[int, ...], "RadixNode"] = {}
        self.block_id: int = -1
        self.parent: Optional["RadixNode"] = None
        self.chunk: Optional[Tuple[int, ...]] = None
        self.last_access_time: float = time.time()


class RadixCache:
    def __init__(self, block_size: int, allocator: BlockAllocator):
        self.root = RadixNode()
        self.block_size = block_size
        self._allocator = allocator

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """
        Returns (matched_tokens, matched_blocks) for the longest cached
        prefix of `tokens`. The matched blocks are increfed.
        """
        node = self.root
        matched_tokens: List[int] = []
        matched_blocks: List[int] = []

        for i in range(0, len(tokens), self.block_size):
            chunk = tuple(tokens[i: i + self.block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            node = child
            node.last_access_time = time.time()
            matched_tokens.extend(chunk)
            matched_blocks.append(node.block_id)

        if matched_blocks:
            self._allocator.incref(matched_blocks)

        return matched_tokens, matched_blocks

    def reset(self) -> None:
        """
        Clears the cache tree structure only. Does not modify allocator
        ref-counts. To clear the underlying KV cache as well, call
        BlockAllocator.free_all(force=True) after this method.
        """
        self.root = RadixNode()

    def insert(self, tokens: List[int], block_table: List[int]) -> List[int]:
        """
        Inserts `tokens`/`block_table` into the cache tree.
        Newly created cache entries are increfed.
        """
        node = self.root
        newly_inserted: List[int] = []

        for i, block_id in enumerate(block_table):
            start = i * self.block_size
            end = min(start + self.block_size, len(tokens))
            chunk = tuple(tokens[start:end])

            child = node.children.get(chunk)
            if child is None:
                child = RadixNode()
                child.block_id = block_id
                child.parent = node
                child.chunk = chunk
                node.children[chunk] = child
                newly_inserted.append(block_id)

            node = node.children[chunk]
            node.last_access_time = time.time()

        if newly_inserted:
            self._allocator.incref(newly_inserted)

        return newly_inserted

    def _collect_leaf_nodes(self) -> List[RadixNode]:
        leaves: List[RadixNode] = []
        stack = list(self.root.children.values())

        while stack:
            node = stack.pop()

            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children.values())

        return leaves

    def evict_lru(self, num_blocks: int) -> List[int]:
        """
        Evicts up to num_blocks lru blocks from the
        cache tree and decrefs them on the allocator. Only 
        evicts blocks whose allocator ref_count is exactly 1
        """
        if num_blocks <= 0:
            return []

        leaves = self._collect_leaf_nodes()
        if not leaves:
            return []

        leaves.sort(key=lambda n: n.last_access_time)
        evicted: List[int] = []
        ref_counts = self._allocator.ref_counts

        for leaf in leaves:
            if len(evicted) >= num_blocks:
                break

            node = leaf
            while (
                node is not None
                and node is not self.root
                and not node.children
                and len(evicted) < num_blocks
                and ref_counts[node.block_id] == 1
            ):
                evicted.append(node.block_id)

                parent = node.parent
                if parent is not None and node.chunk is not None:
                    del parent.children[node.chunk]

                node = parent

        if evicted:
            self._allocator.decref(evicted)

        return evicted
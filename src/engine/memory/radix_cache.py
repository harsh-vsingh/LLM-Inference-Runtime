"""
Prefix-caching radix tree over KV blocks.

Fix vs. the original radix_cache.py: this class now owns an allocator
reference and manages ref-counts internally for every operation that
creates or removes a cache entry. Previously, callers were responsible for
manually calling allocator.incref()/decref() in lockstep with every
match_prefix()/insert()/evict_lru() call - repeated at 6 separate call
sites in async_engine.py. Forgetting the paired call at any one of those
sites would silently corrupt either the cache's view of what's alive or
the allocator's free list. Folding the ref-count management in here makes
that class of bug structurally impossible: there is exactly one place that
increfs on cache insert/match and decrefs on cache eviction.

match_prefix() increfs the matched blocks, because the caller is about to
attach them to a sequence's block_table (i.e. the sequence now holds a
reference). This mirrors what every call site did manually before.
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
        prefix of `tokens`. The matched blocks are increfed on the
        caller's behalf - the caller is expected to attach them to a
        sequence's block_table, i.e. take ownership of that reference.
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
        Clears the cache tree structure only. Does NOT touch allocator
        ref-counts - any blocks referenced by this tree are still owned by
        whatever sequences currently hold them in their block_table, and
        will be released normally via those sequences' own decref on
        finish. This mirrors the pre-existing benchmarking use in
        admin_service.py (CLEAR_CACHE), which pairs this with a separate,
        explicit allocator.free_all() call for the full reset - that call
        now independently refuses to run if anything is still live (see
        BlockAllocator.free_all).
        """
        self.root = RadixNode()

    def insert(self, tokens: List[int], block_table: List[int]) -> List[int]:
        """
        Inserts `tokens`/`block_table` into the cache tree. Newly-created
        cache entries are increfed on the allocator (the cache now holds a
        reference to them, in addition to whichever sequence already
        holds one from having computed them).
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

    def _collect_leaf_nodes(
        self,
        node: Optional[RadixNode] = None,
        leaves: Optional[List[RadixNode]] = None,
    ) -> List[RadixNode]:
        if node is None:
            node = self.root
        if leaves is None:
            leaves = []

        if node is not self.root and not node.children:
            leaves.append(node)

        for child in node.children.values():
            self._collect_leaf_nodes(child, leaves)

        return leaves

    def evict_lru(self, num_blocks: int) -> List[int]:
        """
        Evicts up to `num_blocks` least-recently-used blocks from the
        cache tree and decrefs them on the allocator. Only evicts blocks
        whose allocator ref_count is exactly 1, i.e. blocks held ONLY by
        the cache and not currently in use by any running sequence -
        evicting a block a live sequence still needs would corrupt that
        sequence's KV state.

        Note: no longer takes ref_counts as a parameter (the original
        signature required the caller to reach into
        `self.allocator.ref_counts` and pass it in) - this class now owns
        its allocator reference, so it reads that state itself.
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
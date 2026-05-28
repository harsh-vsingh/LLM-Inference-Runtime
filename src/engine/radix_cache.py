import time
from typing import List, Tuple, Dict, Optional


class RadixNode:
    def __init__(self):
        self.children: Dict[Tuple[int, ...], "RadixNode"] = {}
        self.block_id: int = -1

        self.parent: Optional["RadixNode"] = None
        self.chunk: Optional[Tuple[int, ...]] = None

        self.last_access_time: float = time.time()


class RadixCache:
    def __init__(self, block_size: int):
        self.root = RadixNode()
        self.block_size = block_size

    def match_prefix(
        self,
        tokens: List[int],
    ) -> Tuple[List[int], List[int]]:
        node = self.root
        matched_tokens = []
        matched_blocks = []

        for i in range(0, len(tokens), self.block_size):
            chunk = tuple(tokens[i : i + self.block_size])

            if chunk in node.children:
                node = node.children[chunk]
                node.last_access_time = time.time()
                matched_tokens.extend(chunk)
                matched_blocks.append(node.block_id)
            else:
                break

        return matched_tokens, matched_blocks

    def insert(
        self,
        tokens: List[int],
        block_table: List[int],
    ) -> List[int]:
        node = self.root
        newly_inserted = []

        for i, block_id in enumerate(block_table):
            start = i * self.block_size
            end = min(start + self.block_size, len(tokens))

            chunk = tuple(tokens[start:end])

            if chunk not in node.children:
                child = RadixNode()
                child.block_id = block_id
                child.parent = node
                child.chunk = chunk
                node.children[chunk] = child
                
                newly_inserted.append(block_id)

            node = node.children[chunk]
            node.last_access_time = time.time()
            
        return newly_inserted

    def _collect_leaf_nodes(
        self,
        node: Optional[RadixNode] = None,
        leaves: Optional[List[RadixNode]] = None,
    ):
        if node is None:
            node = self.root

        if leaves is None:
            leaves = []

        if node is not self.root and not node.children:
            leaves.append(node)

        for child in node.children.values():
            self._collect_leaf_nodes(child, leaves)

        return leaves

    def evict_lru(
        self,
        num_blocks: int,
        ref_counts: List[int],
    ) -> List[int]:
        leaves = self._collect_leaf_nodes()

        if not leaves:
            return []

        leaves.sort(key=lambda x: x.last_access_time)
        evicted = []

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

        return evicted
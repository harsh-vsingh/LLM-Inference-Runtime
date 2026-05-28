import time
from typing import List, Tuple, Dict

class RadixNode:
    def __init__(self):
        self.children: Dict[Tuple[int, ...], 'RadixNode'] = {}
        self.block_id: int = -1
        self.last_access_time: float = time.time()

class RadixCache:
    def __init__(self, block_size: int):
        self.root = RadixNode()
        self.block_size = block_size

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        node = self.root
        matched_tokens = []
        matched_blocks = []
        
        for i in range(0, len(tokens), self.block_size):
            chunk = tuple(tokens[i : i + self.block_size])
            
            if len(chunk) < self.block_size:
                break
                
            if chunk in node.children:
                node = node.children[chunk]
                node.last_access_time = time.time()
                matched_tokens.extend(chunk)
                matched_blocks.append(node.block_id)
            else:
                break
                
        return matched_tokens, matched_blocks

    def insert(self, tokens: List[int], block_table: List[int]) -> List[int]:
        node = self.root
        newly_inserted_blocks = []
        
        for i, block_id in enumerate(block_table):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            
            if end_idx > len(tokens):
                break
                
            chunk = tuple(tokens[start_idx:end_idx])
            
            if chunk not in node.children:
                new_node = RadixNode()
                new_node.block_id = block_id
                node.children[chunk] = new_node
                newly_inserted_blocks.append(block_id)
                
            node = node.children[chunk]
            node.last_access_time = time.time()
            
        return newly_inserted_blocks
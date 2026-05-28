import time
from typing import List, Tuple, Optional, Any
import torch
from transformers.cache_utils import DynamicCache

class RadixNode:
    def __init__(self):
        self.children = {} 
        self.kv_cache = None
        self.last_access_time = time.time()


class RadixCache:
    def __init__(self):
        self.cached_sequences: List[Tuple[List[int], Any]] = []

    def _slice_kv_cache(self, kv_cache: Any, slice_len: int):
        new_cache = DynamicCache()
        
        if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
            layers = zip(kv_cache.key_cache, kv_cache.value_cache)
        else:
            layers = kv_cache
            
        for layer_idx, layer in enumerate(layers):
            layer_k, layer_v = layer[0], layer[1]
            sliced_k = layer_k[:, :, :slice_len, :].clone()
            sliced_v = layer_v[:, :, :slice_len, :].clone()
            
            new_cache.update(sliced_k, sliced_v, layer_idx)
            
        return new_cache

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[Any]]:
        """
        Finds the longest common prefix in the cache.
        """
        best_match_len = 0
        best_kv_cache = None

        for cached_tokens, kv_cache in self.cached_sequences:
            match_len = 0
            for t1, t2 in zip(token_ids, cached_tokens):
                if t1 == t2:
                    match_len += 1
                else:
                    break
            
            if match_len > best_match_len:
                best_match_len = match_len
                best_kv_cache = self._slice_kv_cache(kv_cache, match_len)

        return best_match_len, best_kv_cache

    def insert(self, token_ids: List[int], kv_cache: Any):
        """
        Stores a generated sequence and its KV cache.
        For memory safety before PagedAttention, we limit cache size.
        """
        if len(self.cached_sequences) > 10:
            self.cached_sequences.pop(0)
            
        detached_cache = self._slice_kv_cache(kv_cache, len(token_ids))
        self.cached_sequences.append((list(token_ids), detached_cache))
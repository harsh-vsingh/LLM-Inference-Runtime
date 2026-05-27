class KVCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt):
        return hash(prompt)

    def get(self, prompt):
        prompt_hash = self._hash_prompt(prompt)
        if prompt_hash in self.cache:
            self.hits += 1
            return self.cache[prompt_hash]
        else:
            self.misses += 1
            return None

    def put(self, prompt, kv_data):
        prompt_hash = self._hash_prompt(prompt)
        if len(self.cache) >= self.max_size:
            self.evict()
        self.cache[prompt_hash] = kv_data

    def evict(self):
        if self.cache:
            self.cache.pop(next(iter(self.cache)))

    def cache_stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "cache_size": len(self.cache),
        }
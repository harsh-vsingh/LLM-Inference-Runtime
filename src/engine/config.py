from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, fields
from typing import Any


_LEGACY_KEY_MAP = {
    "ENABLE_PREFIX_CACHE": "enable_prefix_cache",
    "ENABLE_CONTINUOUS_BATCHING": "enable_continuous_batching",
    "ENABLE_CHUNKED_PREFILL": "enable_chunked_prefill",
}


@dataclass
class EngineConfig(MutableMapping):
    enable_continuous_batching: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_cache: bool = True
    max_queue_depth: int = 512
    max_queue_wait_seconds: float = 120.0
    max_preemption_retries: int = 3
    max_chunk_size: int = 512

    def __getitem__(self, key: str) -> Any:
        attr = _LEGACY_KEY_MAP.get(key, key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        attr = _LEGACY_KEY_MAP.get(key, key)
        if not hasattr(self, attr):
            raise KeyError(f"Unknown engine config key: {key!r}")
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("EngineConfig keys cannot be deleted, only reassigned")

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in fields(self))

    def __len__(self) -> int:
        return len(fields(self))
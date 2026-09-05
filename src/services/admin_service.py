from api.schemas import EngineConfigUpdate
from core.logging import get_logger
from engine.async_engine import AsyncInferenceEngine

logger = get_logger(__name__)


class CacheClearBlockedError(Exception):
    """Raised when CLEAR_CACHE is requested while requests are in flight."""


def update_config(engine: AsyncInferenceEngine, config: EngineConfigUpdate) -> None:
    if config.ENABLE_PREFIX_CACHE is not None:
        engine.config.enable_prefix_cache = config.ENABLE_PREFIX_CACHE
    if config.ENABLE_CHUNKED_PREFILL is not None:
        engine.config.enable_chunked_prefill = config.ENABLE_CHUNKED_PREFILL
    if config.CLEAR_CACHE:
        clear_cache(engine)


def clear_cache(engine: AsyncInferenceEngine) -> None:
    """
    Clears the radix cache tree and frees all KV blocks.

    A block's ref_count can legitimately be 1 forever with zero requests
    running - that's what it means for the radix cache to hold a cached
    prefix (see radix_cache.py). So "is anything still running" must be
    checked directly against the scheduler, not by asking the allocator
    whether any ref_count is nonzero - the latter is true essentially
    always once anything has ever been cached, which made this endpoint
    unusable after the very first request in an earlier version of this
    fix.

    engine.radix_cache.reset() clears the tree structure but deliberately
    does not decref (those references are considered still owned by
    whatever holds them - see reset()'s docstring). Once we've confirmed
    directly that nothing is running, those references are guaranteed to
    be cache-only, so it's safe to force-clear the allocator's ref-counts
    too via free_all(force=True).
    """
    if engine.scheduler.has_unfinished_sequences():
        raise CacheClearBlockedError(
            "Cannot clear cache: requests are currently in flight. "
            "Wait for active requests to finish, or stop sending new "
            "requests and retry."
        )

    engine.radix_cache.reset()
    engine.allocator.free_all(force=True)

    logger.info("Radix cache cleared & VRAM freed for benchmark run.")
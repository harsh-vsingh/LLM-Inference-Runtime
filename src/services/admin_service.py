from api.schemas import EngineConfigUpdate
from engine.async_engine import AsyncInferenceEngine
from core.logging import get_logger

logger = get_logger(__name__)


def update_config(engine: AsyncInferenceEngine, config: EngineConfigUpdate) -> None:
    if config.ENABLE_PREFIX_CACHE is not None:
        engine.config["ENABLE_PREFIX_CACHE"] = config.ENABLE_PREFIX_CACHE
    if config.ENABLE_CONTINUOUS_BATCHING is not None:
        engine.config["ENABLE_CONTINUOUS_BATCHING"] = config.ENABLE_CONTINUOUS_BATCHING
    if config.ENABLE_CHUNKED_PREFILL is not None:
        engine.config["ENABLE_CHUNKED_PREFILL"] = config.ENABLE_CHUNKED_PREFILL
    if config.CLEAR_CACHE:
        clear_cache(engine)


def clear_cache(engine: AsyncInferenceEngine) -> None:
    engine.radix_cache.reset()
    engine.allocator.free_all()

    logger.info("Radix cache cleared & VRAM freed for benchmark run.")
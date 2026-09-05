import asyncio
import logging

from core.config import settings
from engine.async_engine import AsyncInferenceEngine
from models.loader import get_model_loader

logger = logging.getLogger(__name__)


class EngineRegistry:
    def __init__(self, model_name: str, use_quantization: bool = settings.use_quantization):
        self._model_name = model_name
        self._use_quantization = use_quantization
        self._engine: AsyncInferenceEngine | None = None
        self._shutdown_signal_installed = False
        self._shutdown_event: asyncio.Event | None = None

    def load(self) -> None:
        """Load the model and start the engine. Call once, at startup."""
        loader = get_model_loader(self._model_name, use_quantization=self._use_quantization)
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()

        engine = AsyncInferenceEngine(
            model,
            tokenizer,
            max_num_batched_tokens=settings.max_num_batched_tokens,
        )
        engine.start()
        self._engine = engine

    def install_signal_handlers(self, shutdown_timeout: float = 30.0) -> None:
        """
        Deprecated: do not use alongside uvicorn/FastAPI. Uvicorn installs
        its own SIGTERM/SIGINT handlers to drive its graceful shutdown
        sequence (stop accepting new connections, then run the lifespan's
        post-yield teardown). loop.add_signal_handler() REPLACES any
        existing handler for that signal rather than chaining - so
        registering a second one here means whichever call happens last
        silently wins, and the other is dropped. In practice this produced
        a hang on Ctrl+C: either this handler won and uvicorn's own
        shutdown sequence never ran (so uvicorn never exited), or uvicorn's
        handler won and this one never fired (so engine.shutdown() was
        never called at all).

        The correct integration point is the FastAPI lifespan's teardown
        (the code after `yield`) - see server.py. That code runs as part
        of uvicorn's own shutdown sequence, so there's no competition for
        the signal handler slot. Left here only as a no-op for backward
        compatibility with any code still calling it.
        """
        logger.warning(
            "install_signal_handlers() is a no-op - call engine_registry.shutdown() "
            "from the FastAPI lifespan teardown (after `yield`) instead."
        )

    async def shutdown(self, timeout: float = 30.0) -> None:
        if self._engine is not None:
            await self._engine.shutdown(timeout=timeout)

    def get_engine(self) -> AsyncInferenceEngine:
        if self._engine is None:
            raise RuntimeError(
                "EngineRegistry.get_engine() called before load(). "
                "Engine should be loaded during app startup (lifespan)."
            )
        return self._engine

    def get_model_name(self) -> str:
        return self._model_name


def create_engine_registry(
    model_name: str, use_quantization: bool = settings.use_quantization
) -> EngineRegistry:
    registry = EngineRegistry(model_name, use_quantization=use_quantization)
    registry.load()
    return registry
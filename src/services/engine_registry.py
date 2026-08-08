from models.loader import get_model_loader
from engine.async_engine import AsyncInferenceEngine
from core.config import settings


class EngineRegistry:
    
    def __init__(self, model_name: str, use_quantization: bool = True):
        self._model_name = model_name
        self._use_quantization = use_quantization
        self._engine: AsyncInferenceEngine | None = None

    def load(self) -> None:
        """Load the model and start the engine. Call once, at startup."""
        loader = get_model_loader(self._model_name, use_quantization=self._use_quantization)
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()

        engine = AsyncInferenceEngine(
            model,
            tokenizer,
            max_num_seqs=settings.max_num_seqs,
            max_num_batched_tokens=settings.max_num_batched_tokens,
        )
        engine.start()
        self._engine = engine

    def get_engine(self) -> AsyncInferenceEngine:
        if self._engine is None:
            raise RuntimeError(
                "EngineRegistry.get_engine() called before load(). "
                "Engine should be loaded during app startup (lifespan)."
            )
        return self._engine

    def get_model_name(self) -> str:
        return self._model_name


def create_engine_registry(model_name: str, use_quantization: bool = True) -> EngineRegistry:
    registry = EngineRegistry(model_name, use_quantization=use_quantization)
    registry.load()
    return registry
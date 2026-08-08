from fastapi import Request, Depends
from services.engine_registry import EngineRegistry
from engine.async_engine import AsyncInferenceEngine


def get_engine_registry(request: Request) -> EngineRegistry:
    return request.app.state.engine_registry


def get_engine(registry: EngineRegistry = Depends(get_engine_registry)) -> AsyncInferenceEngine:
    return registry.get_engine()
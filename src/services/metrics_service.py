from typing import Any

from engine.async_engine import AsyncInferenceEngine


def get_metrics(engine: AsyncInferenceEngine) -> dict[str, Any]:
    return engine.get_metrics()
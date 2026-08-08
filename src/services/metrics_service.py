from typing import Dict, Any
from engine.async_engine import AsyncInferenceEngine


def get_metrics(engine: AsyncInferenceEngine) -> Dict[str, Any]:
    return engine.get_metrics()
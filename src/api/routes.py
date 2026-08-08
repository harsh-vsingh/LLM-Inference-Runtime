import time
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from api.schemas import ChatCompletionRequest, EngineConfigUpdate
from api.dependencies import get_engine_registry, get_engine
from services.engine_registry import EngineRegistry
from engine.async_engine import AsyncInferenceEngine
from services import inference_service
from services import admin_service
from services import metrics_service
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    registry: EngineRegistry = Depends(get_engine_registry),
    engine: AsyncInferenceEngine = Depends(get_engine),
):
    model_name = registry.get_model_name()

    if req.model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is not served by this runtime."
        )

    try:
        request_id, inference_req = await inference_service.submit_chat_completion(engine, req)
    except Exception:
        logger.exception("Failed to submit chat completion request")
        raise HTTPException(status_code=500, detail="Failed to process request.")

    if req.stream:
        return StreamingResponse(
            inference_service.stream_chat_completion(request, request_id, inference_req, model_name),
            media_type="text/event-stream"
        )

    try:
        return await inference_service.aggregate_chat_completion(request_id, inference_req, model_name)
    except Exception:
        logger.exception(f"[{request_id}] Failed to aggregate chat completion response")
        raise HTTPException(status_code=500, detail="Failed to generate response.")


@router.post("/v1/completions")
async def completions(request: Dict[str, Any]):
    raise HTTPException(
        status_code=501,
        detail="The /v1/completions endpoint is not implemented.",
    )


@router.get("/v1/models")
async def list_models(registry: EngineRegistry = Depends(get_engine_registry)) -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{
            "id": registry.get_model_name(),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "optiServe"
        }]
    }


@router.post("/v1/admin/config")
async def update_engine_config(
    config: EngineConfigUpdate,
    engine: AsyncInferenceEngine = Depends(get_engine),
):
    try:
        admin_service.update_config(engine, config)
    except Exception:
        logger.exception("Failed to update engine config")
        raise HTTPException(status_code=500, detail="Failed to update engine config.")


@router.get("/metrics")
async def metrics(engine: AsyncInferenceEngine = Depends(get_engine)):
    try:
        return metrics_service.get_metrics(engine)
    except Exception:
        logger.exception("Failed to fetch metrics")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics.")
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.dependencies import get_engine, get_engine_registry
from api.schemas import ChatCompletionRequest, EngineConfigUpdate
from core.logging import get_logger
from engine.async_engine import AsyncInferenceEngine
from engine.errors import AdmissionError
from services import admin_service, inference_service, metrics_service
from services.admin_service import CacheClearBlockedError
from services.engine_registry import EngineRegistry

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
    except AdmissionError as e:
        logger.info(f"Request rejected at admission: {e.reason.value}")
        status_code = 503 if e.is_retryable else 400
        raise HTTPException(
            status_code=status_code,
            detail={"message": e.message, "reason": e.reason.value, "retryable": e.is_retryable},
        )
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
async def completions(request: dict[str, Any]):
    raise HTTPException(
        status_code=501,
        detail="The /v1/completions endpoint is not implemented.",
    )


@router.get("/v1/models")
async def list_models(registry: EngineRegistry = Depends(get_engine_registry)) -> dict[str, Any]:
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
    except CacheClearBlockedError as e:
        raise HTTPException(status_code=409, detail=str(e))
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
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from api.schemas import ChatCompletionRequest
from core.logging import get_logger
from engine.async_engine import AsyncInferenceEngine
from engine.request import InferenceRequest

logger = get_logger(__name__)


def _build_prompt(engine: AsyncInferenceEngine, req: ChatCompletionRequest) -> str:
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    return engine.tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )


def _build_inference_request(request_id: str, prompt: str, req: ChatCompletionRequest) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=req.max_tokens or 50,
        temperature=req.temperature if req.temperature is not None else 0.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
    )


async def submit_chat_completion(
    engine: AsyncInferenceEngine, req: ChatCompletionRequest
) -> tuple[str, InferenceRequest]:
    """
    Build the prompt, construct the InferenceRequest, and submit it to the
    engine. Shared by both the streaming and non-streaming paths.

    Raises whatever the tokenizer / engine raise - the route layer is
    responsible for catching and turning this into a clean HTTP error.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        prompt = _build_prompt(engine, req)
    except Exception:
        logger.exception(f"[{request_id}] Failed to build prompt from messages")
        raise

    inference_req = _build_inference_request(request_id, prompt, req)

    try:
        await engine.add_request(inference_req)
    except Exception:
        logger.exception(f"[{request_id}] engine.add_request failed")
        raise

    logger.info(f"[{request_id}] submitted (stream={req.stream}, max_tokens={req.max_tokens})")
    return request_id, inference_req


async def stream_chat_completion(
    http_request: Request,
    request_id: str,
    inference_req: InferenceRequest,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """SSE generator yielding chat.completion.chunk events, then [DONE]."""
    try:
        created_time = int(time.time())
        while True:
            if await http_request.is_disconnected():
                inference_req.abort()
                logger.info(f"[{request_id}] client disconnected, aborting")
                break

            token = await inference_req.output_queue.get()
            if token is None:
                break

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        if not inference_req.is_aborted:
            m = inference_req.metrics

            # NOTE: was `len(inference_req.prompt)`, which counts characters
            # of the rendered chat-template string, not tokens. The engine
            # is the only thing that knows the real tokenized length (it
            # owns the tokenizer), so it populates metrics.prompt_tokens at
            # admission time - see engine/runtime call sites.
            prompt_tokens = m.prompt_tokens

            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": m.tokens_generated,
                    "total_tokens": prompt_tokens + m.tokens_generated,
                    "optiserve_metrics": {
                        "queue_latency": m.queue_latency,
                        "prefill_latency": m.prefill_latency,
                        "decode_time": m.generation_time,
                        "prefix_depth": m.prefix_depth_tokens,
                        "cache_blocks_reused": m.cache_blocks_reused,
                        "avg_decode_batch_size": round(m.avg_decode_batch_size, 2),
                        "ttft": m.ttft,
                    }
                }
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            logger.info(f"[{request_id}] stream completed ({m.tokens_generated} tokens)")

    except asyncio.CancelledError:
        inference_req.abort()
        logger.info(f"[{request_id}] stream cancelled")
        raise
    except Exception:
        inference_req.abort()
        logger.exception(f"[{request_id}] error during streaming")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": "Internal error during generation."}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def aggregate_chat_completion(
    request_id: str, inference_req: InferenceRequest, model_name: str
) -> dict[str, Any]:
    """Non-streaming path: drain the output queue into one full response."""
    chunks = []
    try:
        while True:
            token = await inference_req.output_queue.get()
            if token is None:
                break
            chunks.append(token)
        full_text = "".join(chunks)
    except Exception:
        inference_req.abort()
        logger.exception(f"[{request_id}] error during aggregation")
        raise

    logger.info(f"[{request_id}] aggregation completed ({len(full_text)} chars)")

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop"
        }]
    }
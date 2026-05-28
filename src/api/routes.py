import uuid
import json
import time
import asyncio
import torch
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from api.schemas import ChatCompletionRequest
from engine.request import InferenceRequest
from engine.async_engine import AsyncInferenceEngine
from models.loader import ModelLoader

router = APIRouter()
model_load_lock = asyncio.Lock()

def load_engine_synchronously(model_name: str) -> AsyncInferenceEngine:
    loader = ModelLoader(model_name, use_quantization=False)
    loader.load_model()
    
    engine = AsyncInferenceEngine(
        model=loader.get_model(), 
        tokenizer=loader.get_tokenizer(), 
        max_num_seqs=256, 
        max_num_batched_tokens=4096
    )
    engine.start()
    return engine

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, http_request: Request):
    app_state = http_request.app.state
    
    if not hasattr(app_state, "engines"):
        if hasattr(app_state, "engine"):
            app_state.engines = {app_state.model_name: app_state.engine}
        else:
            app_state.engines = {}

    requested_model = req.model

    if requested_model not in app_state.engines:
        try:
            config = await asyncio.to_thread(AutoConfig.from_pretrained, requested_model)
            if config.model_type != "llama":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Architecture '{config.model_type}' is not supported. Only 'llama' models are supported."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to fetch model config from Hub: {str(e)}")

        async with model_load_lock:
            if requested_model not in app_state.engines:
                try:
                    new_engine = await asyncio.to_thread(load_engine_synchronously, requested_model)
                    app_state.engines[requested_model] = new_engine
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load weights: {str(e)}")
                    
    engine: AsyncInferenceEngine = app_state.engines[requested_model]
    
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = engine.tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    inference_req = InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=req.max_tokens or 50,
        temperature=req.temperature if req.temperature is not None else 0.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
    )
    
    await engine.add_request(inference_req)

    if req.stream:
        async def stream_generator():
            try:
                created_time = int(time.time())
                while True:
                    if await http_request.is_disconnected():
                        inference_req.abort()
                        break
                    
                    token = await inference_req.output_queue.get()
                    if token is None:
                        break
                    
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                if not inference_req.is_aborted:
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    
            except asyncio.CancelledError:
                inference_req.abort()
                raise
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
        
    else:
        full_text = ""
        while True:
            token = await inference_req.output_queue.get()
            if token is None:
                break
            full_text += token

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop"
            }]
        }

@router.post("/v1/completions")
async def completions(request: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": "Completion response not yet fully implemented"}

@router.get("/v1/models")
async def list_models(http_request: Request) -> Dict[str, Any]:
    app_state = http_request.app.state
    
    if hasattr(app_state, "engines"):
        loaded_models = list(app_state.engines.keys())
    elif hasattr(app_state, "model_name"):
        loaded_models = [app_state.model_name]
    else:
        loaded_models = []

    data = []
    for model_name in loaded_models:
        data.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "optiServe"
        })
        
    return {
        "object": "list",
        "data": data
    }
import uuid
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from api.schemas import ChatCompletionRequest
from engine.request import InferenceRequest

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, http_request: Request):
    engine = http_request.app.state.engine
    
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = engine.tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    inference_req = InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=req.max_tokens or 50
    )
    
    await engine.add_request(inference_req)

    async def stream_generator():
        created_time = int(time.time())
        while True:
            token = await inference_req.output_queue.get()
            
            if token is None:
                break
                
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": req.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        return {"error": "Non-streaming not implemented yet. Set stream=True"}

@router.post("/v1/completions")
async def completions(request: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": "Completion response"}

@router.get("/v1/models")
async def list_models() -> List[str]:
    return ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
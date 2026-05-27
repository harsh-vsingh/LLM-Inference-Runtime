from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": "Chat completion response"}

@router.post("/v1/completions")
async def completions(request: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": "Completion response"}

@router.get("/v1/models")
async def list_models() -> List[str]:
    return ["model1", "model2", "model3"]
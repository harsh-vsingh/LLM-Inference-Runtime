from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import time

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=50, gt=0, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    stream: Optional[bool] = False

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]

class ChatCompletionStreamResponseDelta(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]


class EngineConfigUpdate(BaseModel):
    ENABLE_PREFIX_CACHE: bool | None = None
    ENABLE_CHUNKED_PREFILL: bool | None = None
    CLEAR_CACHE: bool = False
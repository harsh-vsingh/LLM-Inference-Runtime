from enum import Enum
from typing import List, Optional, Any
import time
from engine.request import InferenceRequest

class SequenceStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2

class Sequence:
    def __init__(self, request: InferenceRequest, prompt_token_ids: List[int], eos_token_id: int):
        self.request = request
        self.prompt_token_ids = prompt_token_ids
        self.eos_token_id = eos_token_id
        
        self.generated_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING
        self.past_key_values: Optional[Any] = None
        self.prev_text = ""

        self.arrival_time = time.time()
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.finish_time = 0.0

    @property
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def is_finished(self) -> bool:
        if self.status == SequenceStatus.FINISHED:
            return True
        if self.request.is_cancelled:
            return True
        if len(self.generated_token_ids) >= self.request.max_new_tokens:
            return True
        if self.generated_token_ids and self.generated_token_ids[-1] == self.eos_token_id:
            return True
        return False
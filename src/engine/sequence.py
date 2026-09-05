"""
A single generation sequence (one request's token 
stream through the scheduler/engine).
"""

from enum import Enum

from engine.request import InferenceRequest


class SequenceStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(
        self,
        request: InferenceRequest,
        prompt_token_ids: list[int],
        eos_token_id: int,
    ):
        self.request = request
        self.prompt_token_ids = prompt_token_ids
        self.original_prompt_len = len(prompt_token_ids)
        self.generated_token_ids: list[int] = []
        self.eos_token_id = eos_token_id

        self.status = SequenceStatus.WAITING
        self.block_table: list[int] = []

        self.cached_prefix_len = 0
        self.computed_len = 0
        self.prev_text = ""

    @property
    def arrival_time(self) -> float:
        return self.request.metrics.created_at

    @property
    def uncached_token_ids(self) -> list[int]:
        return self.prompt_token_ids[self.cached_prefix_len:]

    @property
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def get_num_uncomputed_tokens(self) -> int:
        return self.get_len - self.computed_len

    def is_finished(self) -> bool:
        if self.status == SequenceStatus.FINISHED:
            return True

        if self.request.is_aborted:
            return True

        if len(self.generated_token_ids) >= self.request.max_new_tokens:
            return True

        if (
            self.generated_token_ids
            and self.generated_token_ids[-1] == self.eos_token_id
        ):
            return True

        return False
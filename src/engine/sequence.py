from enum import Enum
from typing import List, Optional, Any
import time
from engine.request import InferenceRequest


class SequenceStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2



class Sequence:
    def __init__(
        self,
        request: InferenceRequest,
        prompt_token_ids: List[int],
        eos_token_id: int,
    ):
        self.request = request
        self.prompt_token_ids = prompt_token_ids
        self.generated_token_ids: List[int] = []
        self.eos_token_id = eos_token_id

        self.status = SequenceStatus.WAITING

        self.block_table: List[int] = []

        self.arrival_time = time.time()
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.finish_time = 0.0

        self.cached_prefix_len = 0
        self.computed_len = 0  
        self.prev_text = ""

    @property
    def uncached_token_ids(self):
        return self.prompt_token_ids[
            self.cached_prefix_len:
        ]

    @property
    def get_len(self):
        return (
            len(self.prompt_token_ids) +
            len(self.generated_token_ids)
        )
        
    def get_num_uncomputed_tokens(self) -> int:
        return self.get_len - self.computed_len

    def is_finished(self):
        if self.status == SequenceStatus.FINISHED:
            return True

        if self.request.is_aborted:
            return True

        if (
            len(self.generated_token_ids) >=
            self.request.max_new_tokens
        ):
            return True

        if (
            self.generated_token_ids and
            self.generated_token_ids[-1] == self.eos_token_id
        ):
            return True

        return False


    @property
    def queue_latency(self):
        if self.start_time == 0.0:
            return 0.0

        return self.start_time - self.arrival_time


    @property
    def ttft(self):
        if self.first_token_time == 0.0:
            return 0.0

        return self.first_token_time - self.arrival_time


    @property
    def generation_time(self):
        if self.finish_time == 0.0:
            return 0.0

        return self.finish_time - self.first_token_time


    @property
    def tpot(self):
        generated = len(self.generated_token_ids)

        if generated <= 1:
            return 0.0

        return self.generation_time / (generated - 1)


    @property
    def throughput(self):
        total_time = self.finish_time - self.arrival_time

        if total_time <= 0:
            return 0.0

        return len(self.generated_token_ids) / total_time
import time
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    created_at: float = field(default_factory=time.time)
    admitted_at: float = 0.0
    prefill_start: float = 0.0
    first_token_time: float = 0.0
    finished_at: float = 0.0

    tokens_generated: int = 0
    prompt_tokens: int = 0

    prefix_depth_tokens: int = 0
    cache_blocks_reused: int = 0

    decode_steps: int = 0
    sum_decode_batch_size: int = 0

    preemption_count: int = 0

    def record_admitted(self) -> None:
        if self.admitted_at == 0.0:
            self.admitted_at = time.time()

    def record_prefill_start(self) -> None:
        if self.prefill_start == 0.0:
            self.prefill_start = time.time()

    def record_token_generated(self) -> None:
        if self.first_token_time == 0.0:
            self.first_token_time = time.time()
        self.tokens_generated += 1

    def record_decode_step(self, batch_size: int) -> None:
        self.decode_steps += 1
        self.sum_decode_batch_size += batch_size

    def record_finished(self) -> None:
        if self.finished_at == 0.0:
            self.finished_at = time.time()

    def record_preemption(self) -> None:
        self.preemption_count += 1

    @property
    def queue_latency(self) -> float:
        if self.admitted_at == 0.0:
            return 0.0
        return max(0.0, self.admitted_at - self.created_at)

    @property
    def ttft(self) -> float:
        if self.first_token_time == 0.0:
            return 0.0
        return max(0.0, self.first_token_time - self.created_at)

    @property
    def prefill_latency(self) -> float:
        if self.first_token_time == 0.0 or self.prefill_start == 0.0:
            return 0.0
        return max(0.0, self.first_token_time - self.prefill_start)

    @property
    def generation_time(self) -> float:
        end = self.finished_at if self.finished_at > 0.0 else time.time()
        if self.first_token_time == 0.0:
            return 0.0
        return max(0.0, end - self.first_token_time)

    @property
    def tpot(self) -> float:
        if self.tokens_generated <= 1:
            return 0.0
        return self.generation_time / (self.tokens_generated - 1)

    @property
    def total_time(self) -> float:
        end = self.finished_at if self.finished_at > 0.0 else time.time()
        return max(0.0, end - self.created_at)

    @property
    def throughput(self) -> float:
        t = self.total_time
        if t <= 0:
            return 0.0
        return self.tokens_generated / t

    @property
    def avg_decode_batch_size(self) -> float:
        if self.decode_steps == 0:
            return 0.0
        return self.sum_decode_batch_size / self.decode_steps

    def as_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "admitted_at": self.admitted_at,
            "prefill_start": self.prefill_start,
            "first_token_time": self.first_token_time,
            "finished_at": self.finished_at,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "prefix_depth_tokens": self.prefix_depth_tokens,
            "cache_blocks_reused": self.cache_blocks_reused,
            "decode_steps": self.decode_steps,
            "sum_decode_batch_size": self.sum_decode_batch_size,
            "preemption_count": self.preemption_count,
        }
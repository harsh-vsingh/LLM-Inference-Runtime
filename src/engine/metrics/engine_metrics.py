import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class FinishedRequestSample:
    ttft_ms: float
    tpot_ms: float
    throughput_tps: float


class EngineMetrics:
    def __init__(self, recent_window: int = 128):
        self.start_time = time.time()
        self.total_prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.total_generated_tokens = 0
        self._recent: Deque[FinishedRequestSample] = deque(maxlen=recent_window)
        self._interval_prefill_time = 0.0
        self._interval_decode_time = 0.0
        self._interval_steps = 0
        self._last_tokens_snapshot = 0
        self._last_report_time = time.time()
        self.current_tokens_per_second = 0.0
        self.total_requests_admitted = 0
        self.total_requests_rejected = 0
        self.total_preemptions = 0

    def record_prompt(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.cached_prompt_tokens += cached_tokens

    def record_step(self, duration: float, is_prefill: bool, tokens_out: int) -> None:
        if is_prefill:
            self._interval_prefill_time += duration
        else:
            self._interval_decode_time += duration
        self._interval_steps += 1
        self.total_generated_tokens += tokens_out

    def record_admitted(self) -> None:
        self.total_requests_admitted += 1

    def record_rejected(self) -> None:
        self.total_requests_rejected += 1

    def record_preemption(self) -> None:
        self.total_preemptions += 1

    def record_finished_request(self, ttft: float, tpot: float, throughput: float) -> None:
        self._recent.append(
            FinishedRequestSample(ttft_ms=ttft * 1000, tpot_ms=tpot * 1000, throughput_tps=throughput)
        )

    def flush_interval(self) -> Dict[str, float]:
        
        avg_prefill_ms = (
            (self._interval_prefill_time / self._interval_steps) * 1000
            if self._interval_steps > 0 else 0.0
        )
        avg_decode_ms = (
            (self._interval_decode_time / self._interval_steps) * 1000
            if self._interval_steps > 0 else 0.0
        )

        self._interval_prefill_time = 0.0
        self._interval_decode_time = 0.0
        self._interval_steps = 0

        now = time.time()
        dt = max(now - self._last_report_time, 1e-6)
        generated_since_last = self.total_generated_tokens - self._last_tokens_snapshot
        self.current_tokens_per_second = generated_since_last / dt

        self._last_tokens_snapshot = self.total_generated_tokens
        self._last_report_time = now

        return {"avg_prefill_ms": avg_prefill_ms, "avg_decode_ms": avg_decode_ms}

    @property
    def cache_hit_rate_pct(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return (self.cached_prompt_tokens / self.total_prompt_tokens) * 100

    @property
    def avg_ttft_ms(self) -> float:
        if not self._recent:
            return 0.0
        return sum(s.ttft_ms for s in self._recent) / len(self._recent)

    @property
    def avg_tpot_ms(self) -> float:
        if not self._recent:
            return 0.0
        return sum(s.tpot_ms for s in self._recent) / len(self._recent)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
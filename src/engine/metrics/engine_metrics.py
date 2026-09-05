import time
from dataclasses import dataclass

from engine.metrics.windowed import RateCounter, TimeWindowedStats


@dataclass
class FinishedRequestSample:
    ttft_ms: float
    tpot_ms: float
    throughput_tps: float


class EngineMetrics:

    def __init__(self, latency_window_s: float = 60.0):
        self.start_time = time.time()

        self.total_prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.total_generated_tokens = 0

        self.total_requests_admitted = 0
        self.total_requests_rejected = 0
        self.total_preemptions = 0

        self._throughput = RateCounter(min_interval_s=0.2)

        self._ttft = TimeWindowedStats(window_s=latency_window_s)
        self._tpot = TimeWindowedStats(window_s=latency_window_s)
        self._prefill_step_ms = TimeWindowedStats(window_s=latency_window_s)
        self._decode_step_ms = TimeWindowedStats(window_s=latency_window_s)

        self.current_decode_batch_size: int = 0

        self._last_log_time = time.time()

    def record_prompt(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.cached_prompt_tokens += cached_tokens

    def record_step(
        self,
        duration: float,
        is_prefill: bool,
        tokens_out: int,
        decode_batch_size: int = 0,
    ) -> None:
        if is_prefill:
            self._prefill_step_ms.add(duration * 1000)
        else:
            self._decode_step_ms.add(duration * 1000)
            self.current_decode_batch_size = decode_batch_size

        self.total_generated_tokens += tokens_out
        self._throughput.add(tokens_out)

    def record_admitted(self) -> None:
        self.total_requests_admitted += 1

    def record_rejected(self) -> None:
        self.total_requests_rejected += 1

    def record_preemption(self) -> None:
        self.total_preemptions += 1

    def record_finished_request(self, ttft: float, tpot: float, throughput: float) -> None:
        self._ttft.add(ttft * 1000)
        self._tpot.add(tpot * 1000)


    def flush_interval(self) -> dict[str, float]:
        """
        Called periodically by the background logger.
        """
        return {
            "avg_prefill_ms": self._prefill_step_ms.mean(),
            "avg_decode_ms": self._decode_step_ms.mean(),
        }

    @property
    def current_tokens_per_second(self) -> float:
        return self._throughput.rate()

    @property
    def cache_hit_rate_pct(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return (self.cached_prompt_tokens / self.total_prompt_tokens) * 100

    @property
    def avg_ttft_ms(self) -> float:
        return self._ttft.mean()

    @property
    def avg_tpot_ms(self) -> float:
        return self._tpot.mean()

    def latency_summary(self) -> dict[str, dict]:
        return {
            "ttft_ms": self._ttft.summary(),
            "tpot_ms": self._tpot.summary(),
        }

    def step_time_summary(self) -> dict[str, dict]:
        return {
            "prefill_ms": self._prefill_step_ms.summary(),
            "decode_ms": self._decode_step_ms.summary(),
        }

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
import time
from collections import deque
from dataclasses import dataclass


class RateCounter:
    """
    Tracks a cumulative counter and reports a rate computed fresh on every
    read, using the counter's value since the last read or a min window
    """

    def __init__(self, window_s: float = 10.0, max_samples: int = 20_000):
        self._window_s = window_s
        self._max_samples = max_samples
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, amount)

    def add(self, n: int) -> None:
        self._events.append((time.time(), n))
        if len(self._events) > self._max_samples:
            self._events.popleft()

    def _trim(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def rate(self) -> float:
        now = time.time()
        self._trim(now)
        if not self._events:
            return 0.0

        total = sum(amount for _, amount in self._events)
        span = min(now - self._events[0][0], self._window_s)
        if span <= 0:
            span = self._window_s
        return total / span

    @property
    def total(self) -> int:
        return sum(amount for _, amount in self._events)


@dataclass
class _Sample:
    t: float
    value: float


class TimeWindowedStats:
    """
    Keeps samples with timestamps and reports mean/percentiles over the
    last `window_s` seconds, trimming older samples on read.
    """

    def __init__(self, window_s: float = 60.0, max_samples: int = 20_000):
        self._window_s = window_s
        self._max_samples = max_samples
        self._samples: deque[_Sample] = deque()

    def add(self, value: float) -> None:
        self._samples.append(_Sample(t=time.time(), value=value))
        if len(self._samples) > self._max_samples:
            self._samples.popleft()

    def _trim(self) -> None:
        cutoff = time.time() - self._window_s
        while self._samples and self._samples[0].t < cutoff:
            self._samples.popleft()

    def count(self) -> int:
        self._trim()
        return len(self._samples)

    def mean(self) -> float:
        self._trim()
        if not self._samples:
            return 0.0
        return sum(s.value for s in self._samples) / len(self._samples)

    def percentile(self, p: float) -> float:
        """p in [0, 100]."""
        self._trim()
        if not self._samples:
            return 0.0
        values = sorted(s.value for s in self._samples)
        idx = min(int(len(values) * (p / 100.0)), len(values) - 1)
        return values[idx]

    def p50(self) -> float:
        return self.percentile(50)

    def p99(self) -> float:
        return self.percentile(99)

    def summary(self) -> dict:
        self._trim()
        if not self._samples:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0}
        values = sorted(s.value for s in self._samples)
        n = len(values)
        return {
            "count": n,
            "mean": sum(values) / n,
            "p50": values[min(int(n * 0.50), n - 1)],
            "p99": values[min(int(n * 0.99), n - 1)],
        }
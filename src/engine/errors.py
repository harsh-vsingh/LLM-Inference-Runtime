"""
Structured exceptions for the engine.

AdmissionError in particular is designed to be consumed by a future API
gateway doing load- and cache-aware routing: the `reason` code tells the
caller whether retrying this instance later (or a sibling instance right
now) could succeed, or whether the request can never succeed anywhere.
"""

from enum import Enum


class AdmissionRejectReason(str, Enum):
    # Transient - retrying later, or on a sibling instance, may succeed.
    QUEUE_FULL = "queue_full"
    KV_BUDGET_EXHAUSTED = "kv_budget_exhausted"

    # Permanent - this request cannot succeed on this engine config, ever.
    # A gateway should not retry these anywhere running the same model/config.
    PROMPT_EXCEEDS_CAPACITY = "prompt_exceeds_capacity"

    # Lifecycle - engine is not currently accepting work.
    ENGINE_DRAINING = "engine_draining"
    ENGINE_NOT_STARTED = "engine_not_started"

    # Retry-budget exhaustion after repeated preemption.
    PREEMPTION_RETRIES_EXHAUSTED = "preemption_retries_exhausted"

    # Backstop - request sat in queue past the configured timeout.
    QUEUE_TIMEOUT = "queue_timeout"


class AdmissionError(Exception):
    """
    Raised synchronously from add_request() when a request cannot be
    admitted. Carries a structured `reason` so callers (including a future
    gateway) can decide whether to retry, retry elsewhere, or give up.
    """

    def __init__(self, reason: AdmissionRejectReason, message: str | None = None):
        self.reason = reason
        self.message = message or reason.value
        super().__init__(self.message)

    @property
    def is_retryable(self) -> bool:
        return self.reason in (
            AdmissionRejectReason.QUEUE_FULL,
            AdmissionRejectReason.KV_BUDGET_EXHAUSTED,
            AdmissionRejectReason.ENGINE_DRAINING,
        )


class EngineNotRunningError(Exception):
    """Raised when an operation requires the engine's process loop to be running."""


class BlockAllocationError(Exception):
    """
    Raised by BlockAllocator.allocate() when there are not enough free
    blocks. This is a per-sequence, recoverable condition (the caller
    should preempt/retry that one sequence) - it is intentionally NOT an
    AdmissionError, since it happens after admission, to a sequence that's
    already running. Callers must not let this propagate to a handler that
    would tear down unrelated sequences in the same batch.
    """


class RequestAbortedError(Exception):
    """Raised internally when a sequence is aborted mid-flight (e.g. during shutdown)."""
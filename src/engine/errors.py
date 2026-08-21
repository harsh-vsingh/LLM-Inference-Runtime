"""
Structured exceptions for the engine.
"""

from enum import Enum


class AdmissionRejectReason(str, Enum):
    # Retrying later may succeed.
    QUEUE_FULL = "queue_full"
    KV_BUDGET_EXHAUSTED = "kv_budget_exhausted"

    # Request cannot succeed on this engine config.
    PROMPT_EXCEEDS_CAPACITY = "prompt_exceeds_capacity"

    # Engine is not currently accepting work.
    ENGINE_DRAINING = "engine_draining"


class AdmissionError(Exception):
    """
    Raised synchronously from add_request() when a request cannot be
    admitted. Carries a structured `reason` so callers can decide whether to retry, retry elsewhere, or give up.
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
    should preempt/retry that one sequence)
    """


class RequestAbortedError(Exception):
    """Raised internally when a sequence is aborted mid-flight (e.g. during shutdown)."""
"""
SatQuery AI - B2 Reliability Exceptions
=======================================
Structured exceptions for circuit breakers, retries, and service communication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ReliabilityError(Exception):
    """Base exception for all B2 reliability layer failures."""
    pass


class MLServiceCircuitOpenError(ReliabilityError):
    """
    Raised immediately when an ML service circuit is in OPEN state
    to prevent hammering unhealthy endpoints.
    """
    def __init__(
        self,
        service_name: str,
        circuit_state: str,
        cooldown_remaining_seconds: float,
        next_probe_time: Optional[float] = None,
    ):
        self.error_code = "ML_SERVICE_CIRCUIT_OPEN"
        self.service_name = service_name
        self.circuit_state = circuit_state
        self.cooldown_remaining_seconds = round(max(0.0, cooldown_remaining_seconds), 2)
        self.next_probe_time = next_probe_time
        super().__init__(
            f"[{self.error_code}] Circuit for ML service '{service_name}' is {circuit_state}. "
            f"Call rejected. Cooldown remaining: {self.cooldown_remaining_seconds:.1f}s."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "service_name": self.service_name,
            "circuit_state": self.circuit_state,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "next_probe_time": self.next_probe_time,
        }


class MaxRetriesExceededError(ReliabilityError):
    """Raised when all configured retry attempts have been exhausted."""
    def __init__(
        self,
        service_name: str,
        total_attempts: int,
        last_exception: Optional[Exception],
        attempt_history: List[Dict[str, Any]],
    ):
        self.error_code = "MAX_RETRIES_EXCEEDED"
        self.service_name = service_name
        self.total_attempts = total_attempts
        self.last_exception = last_exception
        self.attempt_history = attempt_history
        cause_msg = str(last_exception) if last_exception else "Unknown root cause"
        super().__init__(
            f"[{self.error_code}] Service '{service_name}' failed after {total_attempts} attempts. "
            f"Last error: {cause_msg}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "service_name": self.service_name,
            "total_attempts": self.total_attempts,
            "last_error": str(self.last_exception) if self.last_exception else None,
            "attempt_history": self.attempt_history,
        }


class NonRetryableServiceError(ReliabilityError):
    """Raised when a permanent/deterministic error occurs that must NOT be retried."""
    def __init__(self, service_name: str, status_code: Optional[int], message: str):
        self.error_code = "NON_RETRYABLE_SERVICE_ERROR"
        self.service_name = service_name
        self.status_code = status_code
        super().__init__(f"[{self.error_code}] Non-retryable error from '{service_name}' (status {status_code}): {message}")

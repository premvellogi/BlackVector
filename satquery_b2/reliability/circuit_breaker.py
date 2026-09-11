"""
SatQuery AI - B2 Circuit Breaker for Laptop-Served ML Endpoints
===============================================================
Thread-safe state machine protecting against repeated calls to failing ML services.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from satquery_b2.reliability.exceptions import MLServiceCircuitOpenError

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitState(str, Enum):
    """Standard Circuit Breaker States."""
    CLOSED = "CLOSED"        # Normal operation: requests allowed, failures tracked
    OPEN = "OPEN"            # Unhealthy service: fast failure, requests rejected immediately
    HALF_OPEN = "HALF_OPEN"  # Recovery trial: allows limited probe requests to test health


@dataclass
class CircuitBreakerConfig:
    """Configuration for a CircuitBreaker instance."""
    failure_threshold: int = 3                # Consecutive failures to trip circuit
    recovery_timeout_seconds: float = 10.0    # Cooldown period before attempting probe
    half_open_probe_limit: int = 1            # Max concurrent probe requests in HALF_OPEN
    consecutive_success_threshold: int = 1    # Successes in HALF_OPEN to close circuit


@dataclass
class CircuitMetrics:
    """Telemetry and execution metrics for a CircuitBreaker."""
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejected_calls: int = 0
    transition_history: List[Dict[str, Any]] = field(default_factory=list)


class CircuitBreaker:
    """
    Thread-safe Circuit Breaker implementing CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine.
    """
    def __init__(self, service_name: str, config: Optional[CircuitBreakerConfig] = None):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # State tracking counters
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._active_probes = 0
        self._last_state_change_time = time.monotonic()
        self._last_failure_time: Optional[float] = None
        self._metrics = CircuitMetrics()

    @property
    def state(self) -> CircuitState:
        """Returns the current evaluated state (evaluating cooldown timeout if OPEN)."""
        with self._lock:
            self._evaluate_state_transition()
            return self._state

    def _evaluate_state_transition(self) -> None:
        """Internal helper to check if cooldown has elapsed in OPEN state."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_state_change_time
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN, reason="Recovery cooldown elapsed")

    def _transition_to(self, new_state: CircuitState, reason: str = "") -> None:
        """Transitions state and records transition event."""
        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.monotonic()
        self._active_probes = 0

        if new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._consecutive_successes = 0
        elif new_state == CircuitState.OPEN:
            self._consecutive_successes = 0

        event = {
            "timestamp": time.time(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason,
        }
        self._metrics.transition_history.append(event)
        logger.info("[%s] Circuit state changed: %s -> %s (%s)", self.service_name, old_state.value, new_state.value, reason)

    def can_execute(self) -> bool:
        """
        Determines whether a new call is permitted.
        Raises MLServiceCircuitOpenError if the call should be rejected.
        """
        with self._lock:
            self._evaluate_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_state_change_time
                remaining = self.config.recovery_timeout_seconds - elapsed
                self._metrics.total_rejected_calls += 1
                raise MLServiceCircuitOpenError(
                    service_name=self.service_name,
                    circuit_state=self._state.value,
                    cooldown_remaining_seconds=max(0.0, remaining),
                    next_probe_time=self._last_state_change_time + self.config.recovery_timeout_seconds,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._active_probes < self.config.half_open_probe_limit:
                    self._active_probes += 1
                    return True
                else:
                    self._metrics.total_rejected_calls += 1
                    raise MLServiceCircuitOpenError(
                        service_name=self.service_name,
                        circuit_state="HALF_OPEN (Probe in flight)",
                        cooldown_remaining_seconds=1.0,
                    )

        return False

    def record_success(self) -> None:
        """Records a successful operation against the ML service."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.config.consecutive_success_threshold:
                    self._transition_to(CircuitState.CLOSED, reason="Successful probe in HALF_OPEN")
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Records a failed operation against the ML service."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Failed probe in HALF_OPEN immediately returns to OPEN
                self._transition_to(CircuitState.OPEN, reason=f"Probe failed in HALF_OPEN: {str(error)}")
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        reason=f"Reached failure threshold ({self._consecutive_failures}/{self.config.failure_threshold})",
                    )

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Wraps a target function call within the circuit breaker lifecycle.
        """
        self.can_execute()
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    def reset(self) -> None:
        """Manually resets the circuit breaker to clean CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED, reason="Manual reset")
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._active_probes = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Returns snapshot of circuit breaker performance and health."""
        with self._lock:
            return {
                "service_name": self.service_name,
                "current_state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "total_calls": self._metrics.total_calls,
                "total_successes": self._metrics.total_successes,
                "total_failures": self._metrics.total_failures,
                "total_rejected_calls": self._metrics.total_rejected_calls,
                "transitions_count": len(self._metrics.transition_history),
            }

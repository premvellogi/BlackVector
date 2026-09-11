"""
SatQuery AI - B2 Retry Wrapper for Laptop-Served ML Endpoints
=============================================================
Provides intelligent, classified retry execution with exponential backoff and jitter.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from satquery_b2.reliability.circuit_breaker import CircuitBreaker
from satquery_b2.reliability.exceptions import (
    MaxRetriesExceededError,
    MLServiceCircuitOpenError,
    NonRetryableServiceError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


# Status codes considered transient/retryable
RETRYABLE_STATUS_CODES: Set[int] = {429, 502, 503, 504}

# Status codes considered deterministic/non-retryable
NON_RETRYABLE_STATUS_CODES: Set[int] = {400, 401, 403, 404, 405, 422}


@dataclass
class RetryConfig:
    """Configuration options for exponential retry logic."""
    max_attempts: int = 3
    base_backoff_seconds: float = 0.2
    max_backoff_seconds: float = 5.0
    backoff_multiplier: float = 2.0
    enable_jitter: bool = True
    per_attempt_timeout_seconds: Optional[float] = 30.0
    total_deadline_seconds: Optional[float] = 120.0
    is_idempotent: bool = True


def is_exception_retryable(exc: Exception) -> Tuple[bool, str]:
    """
    Evaluates whether an exception represents a transient failure eligible for retry.
    Returns (is_retryable, failure_category).
    """
    # Check circuit breaker errors (never retry circuit breaker rejections)
    if isinstance(exc, MLServiceCircuitOpenError):
        return False, "CIRCUIT_OPEN"

    # Standard network and timeout errors
    if isinstance(exc, (TimeoutError, ConnectionError, ConnectionResetError, BrokenPipeError, OSError)):
        return True, "NETWORK_TRANSIENT"

    # HTTP library exceptions (httpx / requests)
    exc_type_name = type(exc).__name__
    if "Timeout" in exc_type_name or "ConnectError" in exc_type_name or "NetworkError" in exc_type_name:
        return True, "HTTP_TRANSPORT"

    # Check status code if attached to exception
    status_code = getattr(exc, "status_code", getattr(exc, "code", None))
    if status_code is not None:
        try:
            code = int(status_code)
            if code in RETRYABLE_STATUS_CODES:
                return True, f"HTTP_{code}"
            if code in NON_RETRYABLE_STATUS_CODES:
                return False, f"HTTP_{code}_CLIENT_ERROR"
        except (ValueError, TypeError):
            pass

    # Deterministic programming errors
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, IndexError)):
        return False, "DETERMINISTIC_CLIENT_ERROR"

    # Default: non-retryable unless classified
    return False, "UNCLASSIFIED_ERROR"


def compute_backoff_delay(attempt: int, config: RetryConfig, retry_after_header: Optional[float] = None) -> float:
    """Calculates backoff delay with exponential scaling and full jitter."""
    if retry_after_header is not None and retry_after_header > 0:
        return min(retry_after_header, config.max_backoff_seconds)

    # Exponential backoff formula: base * (multiplier ^ (attempt - 1))
    raw_backoff = config.base_backoff_seconds * (config.backoff_multiplier ** (attempt - 1))
    capped_backoff = min(raw_backoff, config.max_backoff_seconds)

    if config.enable_jitter:
        # Full jitter: uniform random in [0, capped_backoff]
        return random.uniform(0.0, capped_backoff)
    return capped_backoff


def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    service_name: str = "ml_service",
    config: Optional[RetryConfig] = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> T:
    """
    Executes a callable with classified retries, backoff, and observability.
    """
    cfg = config or RetryConfig()
    attempt = 1
    start_time = time.monotonic()
    attempt_history: List[Dict[str, Any]] = []

    while attempt <= cfg.max_attempts:
        # Check global deadline
        if cfg.total_deadline_seconds:
            elapsed = time.monotonic() - start_time
            if elapsed >= cfg.total_deadline_seconds:
                raise MaxRetriesExceededError(
                    service_name=service_name,
                    total_attempts=attempt - 1,
                    last_exception=TimeoutError(f"Total deadline exceeded ({cfg.total_deadline_seconds}s)"),
                    attempt_history=attempt_history,
                )

        attempt_start = time.monotonic()
        try:
            logger.debug("[%s] Invoking attempt %d/%d (correlation_id=%s)", service_name, attempt, cfg.max_attempts, correlation_id)
            result = func(*args, **kwargs)
            attempt_duration = time.monotonic() - attempt_start
            attempt_history.append({
                "attempt": attempt,
                "status": "SUCCESS",
                "duration_seconds": round(attempt_duration, 4),
            })
            return result

        except Exception as exc:
            attempt_duration = time.monotonic() - attempt_start
            retryable, category = is_exception_retryable(exc)

            attempt_record = {
                "attempt": attempt,
                "status": "FAILED",
                "duration_seconds": round(attempt_duration, 4),
                "error_category": category,
                "error_message": str(exc),
            }
            attempt_history.append(attempt_record)

            # Check if retry is allowed
            if not retryable or attempt >= cfg.max_attempts:
                logger.warning(
                    "[%s] Attempt %d/%d failed with non-retryable error (%s): %s",
                    service_name, attempt, cfg.max_attempts, category, str(exc),
                )
                raise MaxRetriesExceededError(
                    service_name=service_name,
                    total_attempts=attempt,
                    last_exception=exc,
                    attempt_history=attempt_history,
                )

            # Compute backoff
            retry_after = getattr(exc, "retry_after", None)
            delay = compute_backoff_delay(attempt, cfg, retry_after)

            logger.info(
                "[%s] Attempt %d/%d failed (%s). Retrying in %.2fs (correlation_id=%s)",
                service_name, attempt, cfg.max_attempts, category, delay, correlation_id,
            )
            time.sleep(delay)
            attempt += 1

    raise MaxRetriesExceededError(
        service_name=service_name,
        total_attempts=cfg.max_attempts,
        last_exception=None,
        attempt_history=attempt_history,
    )


def execute_with_reliability(
    circuit_breaker: CircuitBreaker,
    func: Callable[..., T],
    *args: Any,
    retry_config: Optional[RetryConfig] = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> T:
    """
    Coordinated reliability executor uniting CircuitBreaker and RetryWrapper.

    Explicit Semantic Rule:
    1. Circuit breaker gates entry (rejects if OPEN).
    2. Retries run internally.
    3. If retries succeed, circuit breaker records ONE success.
    4. If all retries are exhausted or non-retryable error occurs, circuit breaker records ONE failure.
    This prevents internal retries from distorting the circuit failure threshold.
    """
    # 1. Gate call through circuit breaker
    circuit_breaker.can_execute()

    try:
        # 2. Execute with retry policy
        result = execute_with_retry(
            func,
            *args,
            service_name=circuit_breaker.service_name,
            config=retry_config,
            correlation_id=correlation_id,
            **kwargs,
        )
        # 3. On overall success
        circuit_breaker.record_success()
        return result
    except Exception as e:
        # 4. On overall failure
        circuit_breaker.record_failure(e)
        raise

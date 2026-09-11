from __future__ import annotations

from typing import Any, Dict

import httpx

from .config import settings
from .models import (
    ConfidenceRequest,
    ConfidenceResponse,
    MLRequest,
    MLResponse,
    ValidationRequest,
    ValidationResponse,
)


class HTTPClientError(RuntimeError):
    pass


class ServiceClient:
    def __init__(self) -> None:
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 3:
                    continue
        raise HTTPClientError(f"POST failed after retries: {url}: {last_error}")

    async def validate(self, req: ValidationRequest) -> ValidationResponse:
        data = await self._post(f"{settings.b2_base_url}/validate", req.model_dump(mode="json"))
        return ValidationResponse.model_validate(data)

    async def confidence(self, req: ConfidenceRequest) -> ConfidenceResponse:
        data = await self._post(f"{settings.confidence_base_url}/confidence", req.model_dump(mode="json"))
        return ConfidenceResponse.model_validate(data)

    async def ml(self, endpoint: str, req: MLRequest) -> MLResponse:
        data = await self._post(f"{settings.ml_base_url}{endpoint}", req.model_dump(mode="json"))
        return MLResponse.model_validate(data)

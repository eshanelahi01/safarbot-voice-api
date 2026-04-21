import time
from typing import Any

import requests

from app.config import settings


class BackendServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
        operation: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.operation = operation

    @property
    def client_status_code(self) -> int:
        if self.status_code in {420, 429}:
            return 503
        if self.status_code is not None and 500 <= self.status_code <= 599:
            return 502
        return self.status_code or 502

    def to_response_data(self) -> dict:
        return {
            "message": str(self),
            "operation": self.operation,
            "upstream_status": self.status_code,
            "upstream_body": self.response_body,
        }


def _build_url(path: str) -> str:
    if not settings.BACKEND_URL:
        raise BackendServiceError("BACKEND_URL is not configured")
    return f"{settings.BACKEND_URL}{path}"


def _read_json(response: requests.Response):
    try:
        return response.json()
    except ValueError as exc:
        raise BackendServiceError("Backend returned a non-JSON response") from exc


def _make_headers(authorization: str | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    return headers


def _safe_response_body(response: requests.Response):
    try:
        return response.json()
    except ValueError:
        body = response.text.strip()
        return body[:500] if body else None


def _request_json(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    payload: dict | None = None,
    authorization: str | None = None,
    operation: str,
):
    attempts = max(1, settings.BACKEND_RETRY_ATTEMPTS)
    last_exception: requests.RequestException | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(
                method,
                _build_url(path),
                params=params,
                json=payload,
                headers=_make_headers(authorization),
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < attempts:
                time.sleep(settings.BACKEND_RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise BackendServiceError(f"{operation} failed: {exc}", operation=operation) from exc

        if response.ok:
            return _read_json(response)

        if response.status_code in {420, 429, 500, 502, 503, 504} and attempt < attempts:
            time.sleep(settings.BACKEND_RETRY_BACKOFF_SECONDS * attempt)
            continue

        raise BackendServiceError(
            f"{operation} failed with status {response.status_code}",
            status_code=response.status_code,
            response_body=_safe_response_body(response),
            operation=operation,
        )

    raise BackendServiceError(
        f"{operation} failed: {last_exception}",
        operation=operation,
    )


def get_routes(slots: dict, authorization: str | None = None):
    try:
        return _request_json(
            "GET",
            "/api/routes",
            params={
                "from": slots.get("from"),
                "to": slots.get("to"),
                "date": slots.get("date"),
            },
            authorization=authorization,
            operation="route_lookup",
        )
    except BackendServiceError:
        raise


def book_ticket(slots: dict, authorization: str | None = None):
    try:
        return _request_json(
            "POST",
            "/api/bookings",
            payload=slots,
            authorization=authorization,
            operation="booking",
        )
    except BackendServiceError:
        raise

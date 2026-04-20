import requests

from app.config import settings


class BackendServiceError(RuntimeError):
    pass


def _build_url(path: str) -> str:
    if not settings.BACKEND_URL:
        raise BackendServiceError("BACKEND_URL is not configured")
    return f"{settings.BACKEND_URL}{path}"


def _read_json(response: requests.Response):
    try:
        return response.json()
    except ValueError as exc:
        raise BackendServiceError("Backend returned a non-JSON response") from exc


def get_routes(slots: dict):
    try:
        response = requests.get(
            _build_url("/api/routes"),
            params={
                "from": slots.get("from"),
                "to": slots.get("to"),
                "date": slots.get("date"),
            },
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _read_json(response)
    except requests.RequestException as exc:
        raise BackendServiceError(f"Route lookup failed: {exc}") from exc


def book_ticket(slots: dict):
    try:
        response = requests.post(
            _build_url("/api/bookings"),
            json=slots,
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _read_json(response)
    except requests.RequestException as exc:
        raise BackendServiceError(f"Booking request failed: {exc}") from exc

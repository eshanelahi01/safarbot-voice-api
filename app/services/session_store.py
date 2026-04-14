from collections import OrderedDict
from copy import deepcopy
from threading import Lock
import time

from app.config import settings


def new_session_state() -> dict:
    return {
        "reply_lang": None,
        "slots": {},
        "route_choice": None,
        "seat_count": None,
        "last_action": None,
    }


class MemorySessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl_seconds = max(1, ttl_seconds)
        self.max_sessions = max(1, max_sessions)
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = Lock()

    def _is_expired(self, record: dict, now: float) -> bool:
        return now - record["updated_at"] > self.ttl_seconds

    def _prune(self, now: float):
        expired_keys = [
            session_id
            for session_id, record in self._sessions.items()
            if self._is_expired(record, now)
        ]
        for session_id in expired_keys:
            self._sessions.pop(session_id, None)

        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def get(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            self._prune(now)

            record = self._sessions.get(session_id)
            if record is not None and self._is_expired(record, now):
                self._sessions.pop(session_id, None)
                record = None

            session = deepcopy(record["data"]) if record is not None else new_session_state()
            self._sessions[session_id] = {
                "data": deepcopy(session),
                "updated_at": now,
            }
            self._sessions.move_to_end(session_id)
            return session

    def peek(self, session_id: str) -> dict | None:
        now = time.time()
        with self._lock:
            self._prune(now)

            record = self._sessions.get(session_id)
            if record is None:
                return None

            if self._is_expired(record, now):
                self._sessions.pop(session_id, None)
                return None

            return deepcopy(record["data"])

    def save(self, session_id: str, session: dict):
        now = time.time()
        with self._lock:
            self._sessions[session_id] = {
                "data": deepcopy(session),
                "updated_at": now,
            }
            self._sessions.move_to_end(session_id)
            self._prune(now)

    def clear(self):
        with self._lock:
            self._sessions.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)


session_store = MemorySessionStore(
    ttl_seconds=settings.SESSION_TTL_SECONDS,
    max_sessions=settings.SESSION_MAX_SESSIONS,
)

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "SafarBot Render API"
    APP_VERSION: str = "1.0.0"

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    INTENT_MODEL_REPO: str = os.getenv("INTENT_MODEL_REPO", "")
    SLOT_MODEL_REPO: str = os.getenv("SLOT_MODEL_REPO", "")

    INTENT_MIN_CONF: float = float(os.getenv("INTENT_MIN_CONF", "0.50"))
    SLOT_MIN_CONF: float = float(os.getenv("SLOT_MIN_CONF", "0.65"))
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    SESSION_MAX_SESSIONS: int = int(os.getenv("SESSION_MAX_SESSIONS", "10000"))


settings = Settings()

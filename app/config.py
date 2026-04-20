import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _resolve_device() -> str:
    try:
        import torch

        if os.getenv("USE_GPU", "0") == "1" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "SafarBot AI Service")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    INTENT_MODEL: str = os.getenv("INTENT_MODEL", os.getenv("INTENT_MODEL_REPO", ""))
    SLOT_MODEL: str = os.getenv("SLOT_MODEL", os.getenv("SLOT_MODEL_REPO", ""))
    DEVICE: str = _resolve_device()

    BACKEND_URL: str = os.getenv("BACKEND_URL", "").rstrip("/")
    REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
    MODEL_LOAD_ON_STARTUP: bool = os.getenv("MODEL_LOAD_ON_STARTUP", "0") == "1"

    INTENT_MIN_CONF: float = float(os.getenv("INTENT_MIN_CONF", "0.50"))
    SLOT_MIN_CONF: float = float(os.getenv("SLOT_MIN_CONF", "0.65"))
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    SESSION_MAX_SESSIONS: int = int(os.getenv("SESSION_MAX_SESSIONS", "10000"))

    NORMALIZER_ASSET_PATH: Path = ASSETS_DIR / "normalizer_assets_v5.json"
    BUSINESS_RULES_PATH: Path = ASSETS_DIR / "business_rules_runtime.json"

    @property
    def INTENT_MODEL_REPO(self) -> str:
        return self.INTENT_MODEL

    @property
    def SLOT_MODEL_REPO(self) -> str:
        return self.SLOT_MODEL


settings = Settings()

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
    ENABLE_RULE_BASED_NLU_FALLBACK: bool = os.getenv("ENABLE_RULE_BASED_NLU_FALLBACK", "1") == "1"
    STT_MODEL_SIZE: str = os.getenv("STT_MODEL_SIZE", "small")
    STT_COMPUTE_TYPE: str = os.getenv("STT_COMPUTE_TYPE", "int8")

    BACKEND_URL: str = os.getenv("BACKEND_URL", "").rstrip("/")
    REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
    BACKEND_RETRY_ATTEMPTS: int = int(os.getenv("BACKEND_RETRY_ATTEMPTS", "2"))
    BACKEND_RETRY_BACKOFF_SECONDS: float = float(os.getenv("BACKEND_RETRY_BACKOFF_SECONDS", "0.75"))
    MODEL_LOAD_ON_STARTUP: bool = os.getenv("MODEL_LOAD_ON_STARTUP", "0") == "1"

    INTENT_MIN_CONF: float = float(os.getenv("INTENT_MIN_CONF", "0.50"))
    SLOT_MIN_CONF: float = float(os.getenv("SLOT_MIN_CONF", "0.65"))
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    SESSION_MAX_SESSIONS: int = int(os.getenv("SESSION_MAX_SESSIONS", "10000"))
    TTS_AZURE_ENDPOINT: str = os.getenv("TTS_AZURE_ENDPOINT", "").rstrip("/")
    TTS_AZURE_KEY: str = os.getenv("TTS_AZURE_KEY", "")
    TTS_AZURE_VOICE_EN: str = os.getenv("TTS_AZURE_VOICE_EN", "en-US-JennyNeural")
    TTS_AZURE_VOICE_UR: str = os.getenv("TTS_AZURE_VOICE_UR", "ur-PK-UzmaNeural")
    TTS_GTTS_ENABLED: bool = os.getenv("TTS_GTTS_ENABLED", "1") == "1"
    TTS_GTTS_TLD: str = os.getenv("TTS_GTTS_TLD", "com")
    TTS_OUTPUT_FORMAT: str = os.getenv(
        "TTS_OUTPUT_FORMAT",
        "audio-16khz-32kbitrate-mono-mp3",
    )
    TTS_REQUEST_TIMEOUT_SECONDS: float = float(
        os.getenv(
            "TTS_REQUEST_TIMEOUT_SECONDS",
            os.getenv("REQUEST_TIMEOUT_SECONDS", "12"),
        )
    )

    NORMALIZER_ASSET_PATH: Path = ASSETS_DIR / "normalizer_assets_v5.json"
    BUSINESS_RULES_PATH: Path = ASSETS_DIR / "business_rules_runtime.json"

    @property
    def INTENT_MODEL_REPO(self) -> str:
        return self.INTENT_MODEL

    @property
    def SLOT_MODEL_REPO(self) -> str:
        return self.SLOT_MODEL


settings = Settings()

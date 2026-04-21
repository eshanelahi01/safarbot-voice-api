from pathlib import Path
from importlib.util import find_spec
from tempfile import NamedTemporaryFile

from app.config import settings
from app.core.normalizer import normalize_text
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _whisper_installed() -> bool:
    return find_spec("faster_whisper") is not None


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        return None
    return WhisperModel


def _audio_suffix(audio_format: str | None) -> str:
    normalized = str(audio_format or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    if ";" in normalized:
        normalized = normalized.split(";", 1)[0]
    normalized = normalized.lstrip(".")

    aliases = {
        "mpeg": "mp3",
        "mpga": "mp3",
        "x-wav": "wav",
        "oga": "ogg",
        "x-m4a": "m4a",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"wav", "mp3", "ogg", "webm", "m4a", "mp4", "flac", "aac"}:
        normalized = "wav"
    return f".{normalized}"


class WhisperSTTService:
    def __init__(self):
        self.model = None
        self.ready = False
        self.error = None

    def load(self):
        self.ready = False
        self.error = None
        self.model = None

        WhisperModel = _import_whisper_model()
        if WhisperModel is None:
            self.error = "faster-whisper is not installed"
            return

        try:
            self.model = WhisperModel(
                settings.STT_MODEL_SIZE,
                device=settings.DEVICE,
                compute_type=settings.STT_COMPUTE_TYPE,
            )
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to load STT model")

    def transcribe(self, audio: bytes, audio_format: str | None = None) -> str:
        if not audio:
            raise RuntimeError("audio payload is empty")

        if not self.ready or self.model is None:
            self.load()

        if not self.ready or self.model is None:
            raise RuntimeError(self.error or "STT model is not ready")

        temp_path: str | None = None
        try:
            with NamedTemporaryFile(suffix=_audio_suffix(audio_format), delete=False) as handle:
                handle.write(audio)
                temp_path = handle.name

            segments, _ = self.model.transcribe(
                temp_path,
                beam_size=1,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return normalize_text(transcript)
        except Exception as exc:
            self.error = str(exc)
            raise RuntimeError(self.error) from exc
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def status(self) -> dict:
        configured = _whisper_installed()
        return {
            "ready": self.ready,
            "configured": configured,
            "error": self.error if not self.ready else None,
            "model": settings.STT_MODEL_SIZE,
            "device": settings.DEVICE,
        }


stt_service = WhisperSTTService()


def transcribe_audio(audio: bytes, audio_format: str | None = None) -> str:
    return stt_service.transcribe(audio, audio_format=audio_format)


def stt_status() -> dict:
    return stt_service.status()

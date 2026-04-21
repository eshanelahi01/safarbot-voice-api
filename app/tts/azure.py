import base64
from io import BytesIO
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None

try:
    from gtts import gTTS
except ModuleNotFoundError:
    gTTS = None


class AzureTTSService:
    def __init__(self):
        self.error = None

    def _configured(self) -> bool:
        return bool(settings.TTS_AZURE_ENDPOINT and settings.TTS_AZURE_KEY)

    def _gtts_available(self) -> bool:
        return settings.TTS_GTTS_ENABLED and gTTS is not None

    def _local_available(self) -> bool:
        return pyttsx3 is not None

    def status(self) -> dict:
        configured = self._configured()
        gtts_available = self._gtts_available()
        local_available = self._local_available()
        available_engines = []
        if configured:
            available_engines.append("azure")
        if gtts_available:
            available_engines.append("gtts")
        if local_available:
            available_engines.append("pyttsx3")
        return {
            "ready": bool(available_engines),
            "configured": configured,
            "engine": available_engines[0] if available_engines else "text_only",
            "available_engines": available_engines,
            "error": None if available_engines else "No TTS provider is configured",
        }

    def _voice_for_lang(self, lang: str) -> str:
        return settings.TTS_AZURE_VOICE_UR if lang in {"ur", "mixed"} else settings.TTS_AZURE_VOICE_EN

    def _gtts_lang(self, lang: str) -> str:
        return "ur" if lang in {"ur", "mixed"} else "en"

    def _synthesize_gtts(self, text: str, lang: str) -> dict:
        if not self._gtts_available():
            return {
                "audio_base64": "",
                "audio_mime_type": "",
                "engine": "text_only",
                "error": "gTTS is not installed",
            }

        buffer = BytesIO()
        try:
            tts = gTTS(
                text=text,
                lang=self._gtts_lang(lang),
                tld=settings.TTS_GTTS_TLD,
                slow=False,
            )
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()
            if not audio_bytes:
                raise RuntimeError("gTTS generated an empty audio payload")

            return {
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "audio_mime_type": "audio/mpeg",
                "engine": "gtts",
                "error": None,
            }
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to synthesize TTS with gTTS")
            return {
                "audio_base64": "",
                "audio_mime_type": "",
                "engine": "text_only",
                "error": self.error,
            }

    def _pick_local_voice(self, engine, lang: str) -> str | None:
        voices = engine.getProperty("voices") or []
        if not voices:
            return None

        preferred_tokens = ("urdu", "pakistan", "ur") if lang in {"ur", "mixed"} else (
            "english",
            "zira",
            "david",
            "hazel",
        )

        for voice in voices:
            sample = " ".join(
                str(value)
                for value in (
                    getattr(voice, "id", ""),
                    getattr(voice, "name", ""),
                    getattr(voice, "languages", ""),
                )
            ).lower()
            if any(token in sample for token in preferred_tokens):
                return voice.id

        return voices[0].id

    def _synthesize_local(self, text: str, lang: str) -> dict:
        if pyttsx3 is None:
            return {
                "audio_base64": "",
                "audio_mime_type": "",
                "engine": "text_only",
                "error": "pyttsx3 is not installed",
            }

        temp_path: str | None = None
        engine = None
        try:
            engine = pyttsx3.init()
            voice_id = self._pick_local_voice(engine, lang)
            if voice_id:
                engine.setProperty("voice", voice_id)

            with NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name

            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            engine.stop()

            audio_bytes = Path(temp_path).read_bytes()
            if not audio_bytes:
                raise RuntimeError("Local TTS generated an empty audio file")

            return {
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "audio_mime_type": "audio/wav",
                "engine": "pyttsx3",
                "error": None,
            }
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to synthesize TTS with pyttsx3")
            return {
                "audio_base64": "",
                "audio_mime_type": "",
                "engine": "text_only",
                "error": self.error,
            }
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def synthesize(self, text: str, lang: str = "en") -> dict:
        if not text.strip():
            return {
                "audio_base64": "",
                "audio_mime_type": "",
                "engine": "text_only",
                "error": "reply text is empty",
            }

        errors: list[str] = []

        if self._configured():
            endpoint = f"{settings.TTS_AZURE_ENDPOINT}/cognitiveservices/v1"
            voice = self._voice_for_lang(lang)
            ssml = (
                "<speak version='1.0' xml:lang='en-US'>"
                f"<voice name='{voice}'>{escape(text)}</voice>"
                "</speak>"
            )
            headers = {
                "Ocp-Apim-Subscription-Key": settings.TTS_AZURE_KEY,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": settings.TTS_OUTPUT_FORMAT,
                "User-Agent": settings.APP_NAME,
            }

            try:
                response = requests.post(
                    endpoint,
                    data=ssml.encode("utf-8"),
                    headers=headers,
                    timeout=settings.TTS_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return {
                    "audio_base64": base64.b64encode(response.content).decode("ascii"),
                    "audio_mime_type": "audio/mpeg",
                    "engine": "azure",
                    "error": None,
                }
            except requests.RequestException as exc:
                self.error = str(exc)
                logger.exception("Failed to synthesize TTS with Azure")
                errors.append(f"azure: {self.error}")

        if self._gtts_available():
            gtts_result = self._synthesize_gtts(text, lang)
            if not gtts_result.get("error"):
                return gtts_result
            errors.append(f"gtts: {gtts_result['error']}")

        if self._local_available():
            local_result = self._synthesize_local(text, lang)
            if not local_result.get("error"):
                return local_result
            errors.append(f"pyttsx3: {local_result['error']}")

        error_message = "; ".join(errors) if errors else "No TTS provider is configured"
        self.error = error_message
        return {
            "audio_base64": "",
            "audio_mime_type": "",
            "engine": "text_only",
            "error": error_message,
        }


tts_service = AzureTTSService()


def synthesize_text(text: str, lang: str = "en") -> dict:
    return tts_service.synthesize(text, lang=lang)


def tts_status() -> dict:
    return tts_service.status()

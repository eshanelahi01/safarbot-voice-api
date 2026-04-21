from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException

from app.config import settings
from app.core.dialogue import decide, dialogue_status, get_session, load_dialogue_rules
from app.core.normalizer import load_normalizer_assets, normalizer_status
from app.nlu import load_models, nlu_status
from app.schemas import ChatResponse, Query, TextRequest, VoiceChatRequest, VoiceResponse
from app.services.voice_pipeline import build_chat_response, build_voice_response
from app.stt import stt_status
from app.tts import tts_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_normalizer_assets()
    load_dialogue_rules()
    if settings.MODEL_LOAD_ON_STARTUP:
        load_models()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


def dependency_status() -> dict:
    backend_ready = bool(settings.BACKEND_URL)
    return {
        "nlu": nlu_status(),
        "normalizer_assets": normalizer_status(),
        "dialogue_rules": dialogue_status(),
        "backend": {
            "ready": backend_ready,
            "error": None if backend_ready else "BACKEND_URL is not configured",
        },
        "stt": stt_status(),
        "tts": tts_status(),
    }


def ensure_dependencies_ready() -> dict:
    if not normalizer_status()["ready"]:
        load_normalizer_assets()
    if not dialogue_status()["ready"]:
        load_dialogue_rules()

    status = dependency_status()
    required_services = ("nlu", "normalizer_assets", "dialogue_rules", "tts")
    if not all(status[name]["ready"] for name in required_services):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service not ready",
                "dependencies": status,
            },
        )
    return status


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "chat": "/chat",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    status = dependency_status()
    required_services = ("nlu", "normalizer_assets", "dialogue_rules", "tts")
    if not all(status[name]["ready"] for name in required_services):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service not ready",
                "dependencies": status,
            },
        )
    return {"ready": True, "dependencies": status}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: Query, authorization: str | None = Header(default=None, alias="Authorization")):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    ensure_dependencies_ready()
    return build_chat_response(payload, authorization=authorization)


@app.post("/voice/text", response_model=VoiceResponse)
def voice_text(payload: TextRequest, authorization: str | None = Header(default=None, alias="Authorization")):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    ensure_dependencies_ready()
    return build_voice_response(
        VoiceChatRequest(
            text=payload.text,
            session_id=payload.session_id,
            context=payload.context,
            response_mode="voice",
        ),
        authorization=authorization,
    )


@app.post("/voice/chat", response_model=VoiceResponse)
def voice_chat(
    payload: VoiceChatRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    if not payload.text.strip() and not payload.audio_base64.strip():
        raise HTTPException(status_code=400, detail="Provide either text or audio_base64")

    ensure_dependencies_ready()
    return build_voice_response(payload, authorization=authorization)


@app.post("/api/voice/chat", response_model=VoiceResponse)
def api_voice_chat(
    payload: VoiceChatRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    if not payload.text.strip() and not payload.audio_base64.strip():
        raise HTTPException(status_code=400, detail="Provide either text or audio_base64")

    ensure_dependencies_ready()
    return build_voice_response(payload, authorization=authorization)

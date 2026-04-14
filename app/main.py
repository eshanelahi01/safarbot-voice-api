from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.schemas import TextRequest, VoiceResponse
from app.services.dialogue_service import decide, get_session
from app.services.fuzzy_service import fuzzy_service
from app.services.model_loader import registry
from app.services.nlu_service import predict_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    fuzzy_service.load()
    registry.load()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


def dependency_status() -> dict:
    return {
        "model_registry": registry.status(),
        "fuzzy_catalog": fuzzy_service.status(),
    }


def ensure_dependencies_ready() -> dict:
    status = dependency_status()
    if not all(service["ready"] for service in status.values()):
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
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    status = ensure_dependencies_ready()
    return {"ready": True, "dependencies": status}


@app.post("/voice/text", response_model=VoiceResponse)
def voice_text(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    ensure_dependencies_ready()

    session = get_session(payload.session_id, payload.context)
    nlu = predict_text(payload.text, expected_action=session.get("last_action"))
    decision = decide(payload.session_id, nlu, payload.context, session=session)

    return VoiceResponse(
        session_id=payload.session_id,
        user_text=nlu["user_text"],
        detected_lang=nlu["detected_lang"],
        intent=nlu["intent"],
        intent_confidence=nlu["intent_confidence"],
        slots_raw=nlu["slots_raw"],
        slots_normalized=nlu["slots_normalized"],
        correction_meta=nlu["correction_meta"],
        next_action=decision["next_action"],
        reply_text=decision["reply_text"],
        audio_base64="",
        routes_preview=[],
        conversation_state=decision["conversation_state"],
    )

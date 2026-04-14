from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.schemas import TextRequest, VoiceResponse
from app.services.dialogue_service import decide
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
    ready = registry.ready and fuzzy_service.ready
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"ready": True}


@app.post("/voice/text", response_model=VoiceResponse)
def voice_text(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    nlu = predict_text(payload.text)
    decision = decide(payload.session_id, nlu, payload.context)

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
    )
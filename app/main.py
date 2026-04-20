from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.core.dialogue import decide, dialogue_status, get_session, load_dialogue_rules
from app.core.normalizer import load_normalizer_assets, normalizer_status
from app.core.tools import BackendServiceError, book_ticket, get_routes
from app.nlu import load_models, model_status, predict_text
from app.schemas import ChatResponse, Query, TextRequest, VoiceResponse


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
        **model_status(),
        "normalizer_assets": normalizer_status(),
        "dialogue_rules": dialogue_status(),
        "backend": {
            "ready": backend_ready,
            "error": None if backend_ready else "BACKEND_URL is not configured",
        },
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
        "chat": "/chat",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    status = dependency_status()
    if not all(service["ready"] for service in status.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service not ready",
                "dependencies": status,
            },
        )
    return {"ready": True, "dependencies": status}


def _build_response(payload: Query) -> ChatResponse:
    ensure_dependencies_ready()

    session = get_session(payload.session_id, payload.context)
    try:
        nlu = predict_text(payload.text, expected_action=session.get("last_action"))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "NLU models are not ready",
                "error": str(exc),
                "dependencies": dependency_status(),
            },
        ) from exc
    decision = decide(payload.session_id, nlu, payload.context, session=session)

    response_type = "message"
    payload_data = None

    tool_payload = dict(decision["conversation_state"]["slots"])
    if decision["conversation_state"].get("route_choice") is not None:
        tool_payload["route_choice"] = decision["conversation_state"]["route_choice"]
    if decision["conversation_state"].get("seat_count") is not None:
        tool_payload["seat_count"] = decision["conversation_state"]["seat_count"]

    try:
        if decision["next_action"] == "CALL_GET_ROUTES":
            payload_data = get_routes(tool_payload)
            response_type = "routes"
        elif decision["next_action"] == "CALL_BOOK":
            payload_data = book_ticket(tool_payload)
            response_type = "booking"
    except BackendServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_slots = dict(decision["conversation_state"]["slots"])
    if decision["conversation_state"].get("route_choice") is not None:
        response_slots["route_choice"] = decision["conversation_state"]["route_choice"]
    if decision["conversation_state"].get("seat_count") is not None:
        response_slots["seat_count"] = decision["conversation_state"]["seat_count"]

    return ChatResponse(
        session_id=payload.session_id,
        user_text=nlu["user_text"],
        detected_lang=nlu["detected_lang"],
        intent=nlu["intent"],
        confidence=nlu["intent_confidence"],
        slots=response_slots,
        slots_raw=nlu["slots_raw"],
        correction_meta=nlu["correction_meta"],
        action=decision["next_action"],
        response=decision["reply_text"],
        type=response_type,
        data=payload_data,
        conversation_state=decision["conversation_state"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: Query):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    return _build_response(payload)


@app.post("/voice/text", response_model=VoiceResponse)
def voice_text(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    response = _build_response(payload)

    return VoiceResponse(
        session_id=response.session_id,
        user_text=response.user_text,
        detected_lang=response.detected_lang,
        intent=response.intent,
        intent_confidence=response.confidence,
        slots_raw=response.slots_raw,
        slots_normalized=response.slots,
        correction_meta=response.correction_meta,
        next_action=response.action,
        reply_text=response.response,
        audio_base64="",
        routes_preview=[],
        conversation_state=response.conversation_state,
    )

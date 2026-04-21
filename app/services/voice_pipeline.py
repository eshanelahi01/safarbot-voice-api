import base64
import binascii
from typing import Any

from app.core.dialogue import decide, get_session
from app.core.tools import BackendServiceError, book_ticket, get_routes
from app.nlu import predict_text
from app.schemas import ChatResponse, Query, RoutePreview, VoiceChatRequest, VoiceResponse
from app.stt import transcribe_audio
from app.tts import synthesize_text


PIPELINE_STAGES = [
    "stt",
    "normalize_text",
    "nlu",
    "normalize_slots",
    "dialogue",
    "tool_caller",
    "tts",
]


def _build_tool_payload(conversation_state: dict) -> dict:
    tool_payload = dict(conversation_state.get("slots", {}))
    if conversation_state.get("route_choice") is not None:
        tool_payload["route_choice"] = conversation_state["route_choice"]
    if conversation_state.get("seat_count") is not None:
        tool_payload["seat_count"] = conversation_state["seat_count"]
    return tool_payload


def _build_response_slots(conversation_state: dict) -> dict:
    response_slots = dict(conversation_state.get("slots", {}))
    if conversation_state.get("route_choice") is not None:
        response_slots["route_choice"] = conversation_state["route_choice"]
    if conversation_state.get("seat_count") is not None:
        response_slots["seat_count"] = conversation_state["seat_count"]
    return response_slots


def _backend_error_reply(reply_lang: str, action: str) -> str:
    booking_message = {
        "en": "I could not reach the booking service right now. Please try again in a moment.",
        "ur": "اس وقت بکنگ سروس سے رابطہ نہیں ہو سکا۔ براہ کرم تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "mixed": "Abhi booking service tak pohanch nahi ho saki. Please thori dair baad dobara try karein.",
    }
    routes_message = {
        "en": "I could not fetch live routes right now. Please try again in a moment.",
        "ur": "اس وقت لائیو روٹس حاصل نہیں ہو سکے۔ براہ کرم تھوڑی دیر بعد دوبارہ کوشش کریں۔",
        "mixed": "Abhi live routes fetch nahi ho sake. Please thori dair baad dobara try karein.",
    }
    mapping = booking_message if action == "CALL_BOOK" else routes_message
    return mapping.get(reply_lang, mapping["en"])


def _coerce_price(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_routes_preview(payload_data: Any) -> list[RoutePreview]:
    if not isinstance(payload_data, list):
        return []

    previews: list[RoutePreview] = []
    for index, item in enumerate(payload_data[:3], start=1):
        if not isinstance(item, dict):
            continue
        previews.append(
            RoutePreview(
                option=index,
                route_id=item.get("_id") or item.get("id") or item.get("routeId"),
                provider=item.get("provider") or item.get("operator") or item.get("company"),
                departure_time=item.get("departure_time")
                or item.get("departureTime")
                or item.get("time"),
                price=_coerce_price(item.get("price")),
            )
        )
    return previews
def _normalize_audio_format(audio_format: str | None) -> str:
    normalized = str(audio_format or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    if ";" in normalized:
        normalized = normalized.split(";", 1)[0]
    normalized = normalized.lstrip(".")
    return normalized or "wav"


def _decode_audio_payload(audio_base64: str, audio_format: str = "wav") -> tuple[bytes, str]:
    payload = (audio_base64 or "").strip()
    if not payload:
        raise ValueError("audio_base64 is empty")

    resolved_format = _normalize_audio_format(audio_format)
    if payload.startswith("data:") and "," in payload:
        header, payload = payload.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
        if mime_type:
            resolved_format = _normalize_audio_format(mime_type)

    try:
        return base64.b64decode(payload), resolved_format
    except (ValueError, binascii.Error) as exc:
        raise ValueError("audio_base64 is not valid base64 data") from exc


def _run_turn(
    payload: Query,
    *,
    authorization: str | None = None,
) -> dict:
    session = get_session(payload.session_id, payload.context)
    nlu = predict_text(payload.text, expected_action=session.get("last_action"))
    decision = decide(payload.session_id, nlu, payload.context, session=session)

    response_type = "message"
    payload_data = None
    reply_text = decision["reply_text"]
    correction_meta = dict(nlu["correction_meta"])

    try:
        if decision["next_action"] == "CALL_GET_ROUTES":
            payload_data = get_routes(
                _build_tool_payload(decision["conversation_state"]),
                authorization=authorization,
            )
            response_type = "routes"
        elif decision["next_action"] == "CALL_BOOK":
            payload_data = book_ticket(
                _build_tool_payload(decision["conversation_state"]),
                authorization=authorization,
            )
            response_type = "booking"
    except BackendServiceError as exc:
        response_type = "error"
        payload_data = exc.to_response_data()
        reply_text = _backend_error_reply(decision["reply_lang"], decision["next_action"])
        correction_meta["backend_error"] = payload_data

    correction_meta["nlu_backend"] = nlu.get("nlu_backend", "unknown")

    return {
        "nlu": nlu,
        "decision": decision,
        "response_type": response_type,
        "payload_data": payload_data,
        "reply_text": reply_text,
        "correction_meta": correction_meta,
        "response_slots": _build_response_slots(decision["conversation_state"]),
    }


def build_chat_response(
    payload: Query,
    *,
    authorization: str | None = None,
) -> ChatResponse:
    result = _run_turn(payload, authorization=authorization)
    nlu = result["nlu"]
    decision = result["decision"]
    tts_result = synthesize_text(result["reply_text"], lang=decision["reply_lang"])
    correction_meta = dict(result["correction_meta"])

    response_type = result["response_type"]
    payload_data = result["payload_data"]
    if tts_result.get("error"):
        correction_meta["tts_error"] = tts_result["error"]
        if response_type != "error":
            response_type = "error"
            payload_data = {
                "message": tts_result["error"],
                "stage": "tts",
            }

    return ChatResponse(
        session_id=payload.session_id,
        user_text=nlu["user_text"],
        detected_lang=nlu["detected_lang"],
        reply_lang=decision["reply_lang"],
        intent=nlu["intent"],
        confidence=nlu["intent_confidence"],
        slots=result["response_slots"],
        slots_raw=nlu["slots_raw"],
        correction_meta=correction_meta,
        action=decision["next_action"],
        response=result["reply_text"],
        audio_base64=tts_result["audio_base64"],
        audio_mime_type=tts_result["audio_mime_type"],
        type=response_type,
        data=payload_data,
        pipeline_meta={
            "input_source": "text",
            "stt_engine": "passthrough",
            "nlu_engine": nlu.get("nlu_backend", "unknown"),
            "tts_engine": tts_result["engine"],
            "pipeline": PIPELINE_STAGES,
        },
        conversation_state=decision["conversation_state"],
    )


def _build_stt_error_response(
    payload: VoiceChatRequest,
    *,
    error_message: str,
    response_mode: str,
) -> VoiceResponse:
    reply_text = "Voice transcription is not available right now. Please send text or configure STT."
    tts_result = synthesize_text(reply_text, lang="en") if response_mode in {"voice", "both"} else {
        "audio_base64": "",
        "audio_mime_type": "",
        "engine": "disabled",
        "error": None,
    }
    correction_meta = {
        "stt_error": error_message,
        "nlu_backend": "unavailable",
    }

    return VoiceResponse(
        session_id=payload.session_id,
        user_text="",
        detected_lang="en",
        reply_lang="en",
        intent="fallback",
        intent_confidence=0.0,
        slots_raw={},
        slots_normalized={},
        correction_meta=correction_meta,
        next_action="FALLBACK",
        reply_text=reply_text,
        audio_base64=tts_result["audio_base64"],
        audio_mime_type=tts_result["audio_mime_type"],
        routes_preview=[],
        type="error",
        data={"message": error_message, "stage": "stt"},
        pipeline_meta={
            "input_source": "voice",
            "stt_engine": "faster-whisper",
            "tts_engine": tts_result["engine"],
            "pipeline": PIPELINE_STAGES,
        },
        conversation_state={},
    )


def build_voice_response(
    payload: VoiceChatRequest,
    *,
    authorization: str | None = None,
) -> VoiceResponse:
    text = payload.text.strip()
    input_source = "text"
    stt_engine = "passthrough"

    if not text:
        try:
            audio_bytes, audio_format = _decode_audio_payload(
                payload.audio_base64,
                payload.audio_format,
            )
            text = transcribe_audio(audio_bytes, audio_format=audio_format)
            input_source = "voice"
            stt_engine = "faster-whisper"
        except (ValueError, RuntimeError) as exc:
            return _build_stt_error_response(
                payload,
                error_message=str(exc),
                response_mode=payload.response_mode,
            )

    turn_payload = Query(
        text=text,
        session_id=payload.session_id,
        context=payload.context,
    )
    result = _run_turn(turn_payload, authorization=authorization)
    nlu = result["nlu"]
    decision = result["decision"]

    tts_result = {
        "audio_base64": "",
        "audio_mime_type": "",
        "engine": "disabled",
        "error": None,
    }
    if payload.response_mode in {"voice", "both"}:
        tts_result = synthesize_text(result["reply_text"], lang=decision["reply_lang"])

    correction_meta = dict(result["correction_meta"])
    if input_source == "voice":
        correction_meta["transcribed_from_audio"] = True
    if tts_result.get("error"):
        correction_meta["tts_error"] = tts_result["error"]

    response_type = result["response_type"]
    payload_data = result["payload_data"]
    if tts_result.get("error") and response_type != "error":
        response_type = "error"
        payload_data = {
            "message": tts_result["error"],
            "stage": "tts",
        }

    return VoiceResponse(
        session_id=payload.session_id,
        user_text=nlu["user_text"],
        detected_lang=nlu["detected_lang"],
        reply_lang=decision["reply_lang"],
        intent=nlu["intent"],
        intent_confidence=nlu["intent_confidence"],
        slots_raw=nlu["slots_raw"],
        slots_normalized=result["response_slots"],
        correction_meta=correction_meta,
        next_action=decision["next_action"],
        reply_text=result["reply_text"],
        audio_base64=tts_result["audio_base64"],
        audio_mime_type=tts_result["audio_mime_type"],
        routes_preview=_build_routes_preview(payload_data),
        type=response_type,
        data=payload_data,
        pipeline_meta={
            "input_source": input_source,
            "stt_engine": stt_engine,
            "nlu_engine": nlu.get("nlu_backend", "unknown"),
            "tts_engine": tts_result["engine"],
            "pipeline": PIPELINE_STAGES,
        },
        conversation_state=decision["conversation_state"],
    )

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _default_session_id() -> str:
    return uuid4().hex


class Query(BaseModel):
    text: str
    session_id: str = Field(default_factory=_default_session_id)
    context: Dict[str, Any] = Field(default_factory=dict)


class RoutePreview(BaseModel):
    option: int
    route_id: Optional[str] = None
    provider: Optional[str] = None
    departure_time: Optional[str] = None
    price: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    user_text: str
    detected_lang: str
    intent: str
    confidence: float
    slots: Dict[str, Any]
    slots_raw: Dict[str, Any] = Field(default_factory=dict)
    correction_meta: Dict[str, Any] = Field(default_factory=dict)
    action: str
    response: str
    type: str = "message"
    data: Optional[Any] = None
    conversation_state: Dict[str, Any] = Field(default_factory=dict)


class TextRequest(Query):
    pass


class VoiceResponse(BaseModel):
    session_id: str
    user_text: str
    detected_lang: str
    intent: str
    intent_confidence: float
    slots_raw: Dict[str, Any]
    slots_normalized: Dict[str, Any]
    correction_meta: Dict[str, Any]
    next_action: str
    reply_text: str
    audio_base64: str = ""
    routes_preview: List[RoutePreview] = Field(default_factory=list)
    conversation_state: Dict[str, Any] = Field(default_factory=dict)

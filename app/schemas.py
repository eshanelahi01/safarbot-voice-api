from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    session_id: str
    text: str
    context: Dict[str, Any] = Field(default_factory=dict)


class RoutePreview(BaseModel):
    option: int
    route_id: Optional[str] = None
    provider: Optional[str] = None
    departure_time: Optional[str] = None
    price: Optional[float] = None


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

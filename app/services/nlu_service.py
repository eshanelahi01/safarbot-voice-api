from app.config import settings
from app.services.fuzzy_service import fuzzy_service
from app.services.model_loader import registry
from app.text_utils import (
    detect_reply_lang,
    extract_date_from_text,
    extract_route_choice,
    normalize_text,
    parse_seat_count,
)


def predict_text(text: str, expected_action: str | None = None) -> dict:
    if not registry.ready or registry.intent_pipe is None or registry.slot_pipe is None:
        raise RuntimeError("NLU models are not ready")

    text = normalize_text(text)
    detected_lang = detect_reply_lang(text)

    intent_scores = registry.intent_pipe(text) or []
    if intent_scores and isinstance(intent_scores[0], list):
        intent_scores = intent_scores[0]

    if not intent_scores:
        intent_scores = [{"label": "FALLBACK", "score": 0.0}]

    best_intent = intent_scores[0]
    slots_raw = {}

    for ent in registry.slot_pipe(text) or []:
        label = ent["entity_group"]
        value = ent["word"]
        score = float(ent["score"])

        if score < settings.SLOT_MIN_CONF:
            continue

        if label == "FROM":
            slots_raw["from"] = value
        elif label == "TO":
            slots_raw["to"] = value
        elif label == "DATE":
            slots_raw["date"] = value
        elif label == "TIME":
            slots_raw["time"] = value
        elif label == "SEAT_COUNT":
            slots_raw["seat_count"] = value
        elif label == "PROVIDER":
            slots_raw["provider"] = value

    date_rule = extract_date_from_text(text)
    if date_rule and "date" not in slots_raw:
        slots_raw["date"] = date_rule

    route_choice = extract_route_choice(text, expected_action=expected_action)
    if route_choice is not None:
        slots_raw["route_choice"] = route_choice

    seat_count = parse_seat_count(text, expected_action=expected_action)
    if seat_count is not None:
        slots_raw["seat_count"] = seat_count
    elif route_choice is not None and expected_action != "ASK_SEAT_COUNT":
        slots_raw.pop("seat_count", None)

    slots_normalized, correction_meta = fuzzy_service.normalize_slots(slots_raw)

    if date_rule:
        slots_normalized["date"] = date_rule

    return {
        "user_text": text,
        "detected_lang": detected_lang,
        "intent": best_intent["label"],
        "intent_confidence": float(best_intent["score"]),
        "slots_raw": slots_raw,
        "slots_normalized": slots_normalized,
        "correction_meta": correction_meta,
        "top3": intent_scores,
    }

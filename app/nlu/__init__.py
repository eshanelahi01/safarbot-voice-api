from app.core.normalizer import (
    detect_reply_lang,
    extract_entities,
    extract_raw_entities,
    normalize_text,
)
from app.nlu.intent import (
    intent_status,
    load_intent_model,
    predict_intent_with_scores,
)
from app.nlu.slot import load_slot_model, predict_slots, slot_status


def load_models():
    load_intent_model()
    load_slot_model()


def model_status() -> dict:
    return {
        "intent_model": intent_status(),
        "slot_model": slot_status(),
    }


def predict_text(text: str, expected_action: str | None = None) -> dict:
    user_text = normalize_text(text)
    detected_lang = detect_reply_lang(user_text)

    intent, confidence, top3 = predict_intent_with_scores(user_text)
    slot_pairs = predict_slots(user_text)
    slots_raw = extract_raw_entities(slot_pairs)
    slots_normalized, correction_meta = extract_entities(
        slot_pairs,
        text=user_text,
        expected_action=expected_action,
    )

    return {
        "user_text": user_text,
        "detected_lang": detected_lang,
        "intent": intent,
        "intent_confidence": confidence,
        "slots_raw": slots_raw,
        "slots_normalized": slots_normalized,
        "correction_meta": correction_meta,
        "top3": top3,
    }

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
from app.nlu.rule_based import predict_rule_based
from app.nlu.slot import load_slot_model, predict_slots, slot_status
from app.config import settings


def _merge_missing_slots(primary: dict, supplement: dict) -> tuple[dict, list[str]]:
    merged = dict(primary)
    filled_keys: list[str] = []

    for key, value in supplement.items():
        if key not in merged or merged.get(key) in {None, ""}:
            merged[key] = value
            filled_keys.append(key)

    return merged, filled_keys


def load_models():
    load_intent_model()
    load_slot_model()


def model_status() -> dict:
    return {
        "intent_model": intent_status(),
        "slot_model": slot_status(),
    }


def nlu_status() -> dict:
    models = model_status()
    model_ready = models["intent_model"]["ready"] and models["slot_model"]["ready"]
    fallback_ready = settings.ENABLE_RULE_BASED_NLU_FALLBACK
    return {
        "ready": model_ready or fallback_ready,
        "mode": "transformers" if model_ready else "rule_based",
        "fallback": {
            "ready": fallback_ready,
            "engine": "rule_based",
        },
        "models": models,
    }


def predict_text(text: str, expected_action: str | None = None) -> dict:
    user_text = normalize_text(text)
    detected_lang = detect_reply_lang(user_text)

    try:
        intent, confidence, top3 = predict_intent_with_scores(user_text)
        slot_pairs = predict_slots(user_text)
        slots_raw = extract_raw_entities(slot_pairs)
        slots_normalized, correction_meta = extract_entities(
            slot_pairs,
            text=user_text,
            expected_action=expected_action,
        )
        supplemental = predict_rule_based(user_text, expected_action=expected_action)
        slots_raw, filled_raw = _merge_missing_slots(slots_raw, supplemental["slots_raw"])
        slots_normalized, filled_normalized = _merge_missing_slots(
            slots_normalized,
            supplemental["slots_normalized"],
        )
        supplemented = sorted(set(filled_raw + filled_normalized))
        if supplemented:
            correction_meta["rule_based_slot_supplement"] = supplemented
        nlu_backend = "transformers"
    except RuntimeError as exc:
        if not settings.ENABLE_RULE_BASED_NLU_FALLBACK:
            raise

        fallback = predict_rule_based(user_text, expected_action=expected_action)
        intent = fallback["intent"]
        confidence = fallback["intent_confidence"]
        top3 = fallback["top3"]
        slots_raw = fallback["slots_raw"]
        slots_normalized = fallback["slots_normalized"]
        correction_meta = dict(fallback["correction_meta"])
        correction_meta["nlu_fallback_reason"] = str(exc)
        nlu_backend = fallback["nlu_backend"]

    return {
        "user_text": user_text,
        "detected_lang": detected_lang,
        "intent": intent,
        "intent_confidence": confidence,
        "slots_raw": slots_raw,
        "slots_normalized": slots_normalized,
        "correction_meta": correction_meta,
        "top3": top3,
        "nlu_backend": nlu_backend,
    }

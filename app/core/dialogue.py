import json
from pathlib import Path

from app.config import settings
from app.core.session_store import new_session_state, session_store


DEFAULT_RULES = {
    "supported_languages": ["en", "ur", "mixed"],
    "actions": {
        "GREETING": {
            "en": "Hello! Where do you want to go?",
            "ur": "السلام علیکم، آپ کہاں جانا چاہتے ہیں؟",
            "mixed": "Hello, aap kahan jana chahte hain?",
        },
        "ASK_FROM": {
            "en": "What city are you departing from?",
            "ur": "آپ کہاں سے روانہ ہونا چاہتے ہیں؟",
            "mixed": "Aap kis city se jana chahte hain?",
        },
        "ASK_TO": {
            "en": "What is your destination city?",
            "ur": "آپ کہاں جانا چاہتے ہیں؟",
            "mixed": "Aap kahan jana chahte hain?",
        },
        "ASK_DATE": {
            "en": "Which date do you want to travel?",
            "ur": "آپ کس تاریخ کو سفر کرنا چاہتے ہیں؟",
            "mixed": "Aap kis date ko travel karna chahte hain?",
        },
        "CALL_GET_ROUTES": {
            "en": "Here are the available routes. Please choose one.",
            "ur": "یہ دستیاب روٹس ہیں، براہ کرم ایک منتخب کریں۔",
            "mixed": "Yeh available routes hain, please aik select karein.",
        },
        "ASK_ROUTE_CHOICE": {
            "en": "Please choose one of the available routes.",
            "ur": "براہ کرم دستیاب روٹس میں سے ایک منتخب کریں۔",
            "mixed": "Please available routes mein se aik choose karein.",
        },
        "ASK_SEAT_COUNT": {
            "en": "How many seats do you want?",
            "ur": "آپ کتنی سیٹیں چاہتے ہیں؟",
            "mixed": "Aap kitni seats chahte hain?",
        },
        "ASK_PAYMENT": {
            "en": "How would you like to pay?",
            "ur": "آپ ادائیگی کیسے کرنا چاہتے ہیں؟",
            "mixed": "Aap payment kaise karna chahte hain?",
        },
        "PAYMENT_UNSUPPORTED": {
            "en": "Only cash payment is supported right now.",
            "ur": "اس وقت صرف نقد ادائیگی دستیاب ہے۔",
            "mixed": "Abhi sirf cash payment supported hai.",
        },
        "ASK_CONFIRM_BOOKING": {
            "en": "Please confirm if you want to book this ticket.",
            "ur": "براہ کرم تصدیق کریں کہ کیا آپ یہ ٹکٹ بک کرنا چاہتے ہیں۔",
            "mixed": "Please confirm karein ke aap yeh ticket book karna chahte hain.",
        },
        "CALL_BOOK": {
            "en": "Booking is being placed now.",
            "ur": "اب بکنگ کی جا رہی ہے۔",
            "mixed": "Ab booking place ki ja rahi hai.",
        },
        "FALLBACK": {
            "en": "Please say that again.",
            "ur": "براہ کرم دوبارہ کہیں۔",
            "mixed": "Please dobara batayein.",
        },
    },
}

_dialogue_rules = dict(DEFAULT_RULES)
_dialogue_ready = False
_dialogue_error = None


def load_dialogue_rules():
    global _dialogue_rules, _dialogue_ready, _dialogue_error

    _dialogue_ready = False
    _dialogue_error = None
    _dialogue_rules = dict(DEFAULT_RULES)

    try:
        rules_path = Path(settings.BUSINESS_RULES_PATH)
        if not rules_path.exists():
            raise FileNotFoundError(f"Dialogue rules not found at {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as handle:
            file_rules = json.load(handle)

        merged = dict(DEFAULT_RULES)
        merged.update(file_rules)
        merged["actions"] = {
            **DEFAULT_RULES["actions"],
            **file_rules.get("actions", {}),
        }
        _dialogue_rules = merged
        _dialogue_ready = True
    except Exception as exc:
        _dialogue_error = str(exc)


def dialogue_status() -> dict:
    return {
        "ready": _dialogue_ready,
        "error": _dialogue_error,
        "asset_path": str(settings.BUSINESS_RULES_PATH),
    }


def _supported_languages() -> set[str]:
    return set(_dialogue_rules.get("supported_languages", ["en"]))


def _sanitize_slots(slots: dict | None) -> dict:
    if not isinstance(slots, dict):
        return {}
    return {key: value for key, value in slots.items() if value is not None}


def _coerce_positive_int(value):
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def serialize_session(session: dict) -> dict:
    state = new_session_state()

    reply_lang = session.get("reply_lang")
    if reply_lang in _supported_languages():
        state["reply_lang"] = reply_lang

    state["slots"] = _sanitize_slots(session.get("slots"))
    state["route_choice"] = _coerce_positive_int(session.get("route_choice"))
    state["seat_count"] = _coerce_positive_int(session.get("seat_count"))

    last_action = session.get("last_action")
    if isinstance(last_action, str) and last_action:
        state["last_action"] = last_action

    return state


def _merge_session(base: dict, override: dict) -> dict:
    merged = serialize_session(base)
    override_state = serialize_session(override)

    if override_state["reply_lang"] is not None:
        merged["reply_lang"] = override_state["reply_lang"]

    merged["slots"].update(override_state["slots"])

    for key in ("route_choice", "seat_count", "last_action"):
        if override_state[key] is not None:
            merged[key] = override_state[key]

    return merged


def _extract_conversation_state(context: dict | None) -> dict | None:
    if not isinstance(context, dict):
        return None

    candidate = context.get("conversation_state")
    if isinstance(candidate, dict):
        return candidate

    session_keys = {"reply_lang", "slots", "route_choice", "seat_count", "last_action"}
    if session_keys.intersection(context):
        return context

    return None


def get_session(session_id: str, context: dict | None = None) -> dict:
    session = session_store.get(session_id)
    context_state = _extract_conversation_state(context)
    if context_state is not None:
        session = _merge_session(session, context_state)
    return session


def choose_reply_lang(session: dict, detected_lang: str) -> str:
    supported = _supported_languages()

    if session["reply_lang"] is None:
        session["reply_lang"] = detected_lang if detected_lang in supported else "en"
        return session["reply_lang"]

    if detected_lang in supported - {"en"}:
        session["reply_lang"] = detected_lang

    return session["reply_lang"]


def build_reply(action: str, lang: str) -> str:
    lang = lang if lang in _supported_languages() else "en"
    action_rules = _dialogue_rules.get("actions", {}).get(action)
    if not isinstance(action_rules, dict):
        action_rules = _dialogue_rules["actions"]["FALLBACK"]
    return action_rules.get(lang, action_rules.get("en", "Please say that again."))


def next_action(
    intent: str,
    slots: dict,
    route_choice: int | None = None,
    seat_count: int | None = None,
    last_action: str | None = None,
    intent_confidence: float = 1.0,
) -> str:
    if intent_confidence < settings.INTENT_MIN_CONF:
        return "FALLBACK"

    if intent == "greeting" and not any(
        [slots.get("from"), slots.get("to"), slots.get("date"), route_choice, seat_count]
    ):
        return "GREETING"

    if not slots.get("from"):
        return "ASK_FROM"
    if not slots.get("to"):
        return "ASK_TO"
    if not slots.get("date"):
        return "ASK_DATE"

    if route_choice is None:
        if last_action == "CALL_GET_ROUTES" and intent != "search_routes":
            return "ASK_ROUTE_CHOICE"
        return "CALL_GET_ROUTES"

    if seat_count is None:
        return "ASK_SEAT_COUNT"

    payment = slots.get("payment")
    if not payment:
        return "ASK_PAYMENT"
    if payment != "cash":
        return "PAYMENT_UNSUPPORTED"

    if intent == "confirm_booking":
        return "CALL_BOOK"

    return "ASK_CONFIRM_BOOKING"


def _trip_changed(existing_slots: dict, incoming_slots: dict) -> bool:
    for key in ("from", "to", "date"):
        if key in incoming_slots and incoming_slots[key] != existing_slots.get(key):
            return True
    return False


def decide(
    session_id: str,
    nlu: dict,
    context: dict | None = None,
    session: dict | None = None,
) -> dict:
    session = session or get_session(session_id, context)
    lang = choose_reply_lang(session, nlu["detected_lang"])

    incoming_slots = _sanitize_slots(nlu.get("slots_normalized"))
    if _trip_changed(session["slots"], incoming_slots):
        session["route_choice"] = None
        session["seat_count"] = None
        session["slots"].pop("payment", None)

    for key, value in incoming_slots.items():
        if key in {"route_choice", "seat_count"}:
            continue
        session["slots"][key] = value

    incoming_route_choice = _coerce_positive_int(incoming_slots.get("route_choice"))
    if incoming_route_choice is not None and incoming_route_choice != session.get("route_choice"):
        session["route_choice"] = incoming_route_choice
        session["seat_count"] = None
        session["slots"].pop("payment", None)

    incoming_seat_count = _coerce_positive_int(incoming_slots.get("seat_count"))
    if incoming_seat_count is not None:
        session["seat_count"] = incoming_seat_count

    action = next_action(
        nlu["intent"],
        session["slots"],
        route_choice=session["route_choice"],
        seat_count=session["seat_count"],
        last_action=session.get("last_action"),
        intent_confidence=nlu["intent_confidence"],
    )

    if action == "PAYMENT_UNSUPPORTED":
        session["slots"].pop("payment", None)

    session["last_action"] = action
    session_store.save(session_id, session)

    conversation_state = serialize_session(session)
    return {
        "session": conversation_state,
        "conversation_state": conversation_state,
        "next_action": action,
        "reply_text": build_reply(action, lang),
        "reply_lang": lang,
    }

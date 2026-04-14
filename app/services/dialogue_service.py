from app.config import settings
from app.services.session_store import new_session_state, session_store


SUPPORTED_LANGS = {"en", "ur", "mixed"}

FINAL_NOTICE = {
    "en": "Please purchase this ticket from the terminal at least 2 hours before departure.",
    "ur": "براہ کرم روانگی سے کم از کم 2 گھنٹے پہلے ٹرمینل سے یہ ٹکٹ خرید لیں۔",
    "mixed": "Please terminal se ticket departure se kam az kam 2 ghantay pehle purchase kar lein.",
}


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
    if reply_lang in SUPPORTED_LANGS:
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
    if session["reply_lang"] is None:
        session["reply_lang"] = detected_lang if detected_lang in SUPPORTED_LANGS else "en"
        return session["reply_lang"]

    if detected_lang in {"ur", "mixed"}:
        session["reply_lang"] = detected_lang

    return session["reply_lang"]


def build_reply(action: str, lang: str) -> str:
    lang = lang if lang in SUPPORTED_LANGS else "en"
    replies = {
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
        "SEARCH_ROUTES": {
            "en": "I have your departure city, destination, and date. Your backend can now search routes.",
            "ur": "مجھے آپ کی روانگی، منزل، اور تاریخ مل گئی ہے۔ اب بیک اینڈ روٹس تلاش کر سکتا ہے۔",
            "mixed": "Departure, destination aur date mil gayi hai. Ab backend routes search kar sakta hai.",
        },
        "ASK_SEAT_COUNT": {
            "en": "How many seats do you want?",
            "ur": "آپ کتنی سیٹیں چاہتے ہیں؟",
            "mixed": "Aap kitni seats chahte hain?",
        },
        "FINAL_NOTICE": FINAL_NOTICE,
        "FALLBACK": {
            "en": "Please say that again.",
            "ur": "براہ کرم دوبارہ کہیں۔",
            "mixed": "Please dobara batayein.",
        },
    }
    return replies.get(action, replies["FALLBACK"]).get(lang, replies["FALLBACK"]["en"])


def decide(
    session_id: str,
    nlu: dict,
    context: dict | None = None,
    session: dict | None = None,
) -> dict:
    session = session or get_session(session_id, context)

    lang = choose_reply_lang(session, nlu["detected_lang"])

    for key, value in nlu["slots_normalized"].items():
        session["slots"][key] = value

    if nlu["slots_normalized"].get("route_choice") is not None:
        session["route_choice"] = nlu["slots_normalized"]["route_choice"]

    if nlu["slots_normalized"].get("seat_count") is not None:
        session["seat_count"] = nlu["slots_normalized"]["seat_count"]

    slots = session["slots"]

    if nlu["intent_confidence"] < settings.INTENT_MIN_CONF:
        action = "FALLBACK"
    elif not slots.get("from"):
        action = "ASK_FROM"
    elif not slots.get("to"):
        action = "ASK_TO"
    elif not slots.get("date"):
        action = "ASK_DATE"
    elif session["route_choice"] is None:
        action = "SEARCH_ROUTES"
    elif session["seat_count"] is None:
        action = "ASK_SEAT_COUNT"
    else:
        action = "FINAL_NOTICE"

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

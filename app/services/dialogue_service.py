from app.config import settings


SESSION_STORE = {}

FINAL_NOTICE = {
    "en": "Please purchase this ticket from the terminal at least 2 hours before departure.",
    "ur": "براہِ کرم روانگی سے کم از کم 2 گھنٹے پہلے ٹرمینل سے یہ ٹکٹ خرید لیں۔",
    "mixed": "Please terminal se ticket departure se kam az kam 2 ghantay pehle purchase kar lein.",
}


def get_session(session_id: str) -> dict:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            "reply_lang": None,
            "slots": {},
            "route_choice": None,
            "seat_count": None,
            "last_action": None,
        }
    return SESSION_STORE[session_id]


def choose_reply_lang(session: dict, detected_lang: str) -> str:
    if session["reply_lang"] is None:
        session["reply_lang"] = detected_lang
        return detected_lang

    if detected_lang in ["ur", "mixed"]:
        session["reply_lang"] = detected_lang

    return session["reply_lang"]


def build_reply(action: str, lang: str) -> str:
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
            "ur": "براہِ کرم دوبارہ کہیں۔",
            "mixed": "Please dobara batayein.",
        },
    }
    return replies.get(action, replies["FALLBACK"])[lang]


def decide(session_id: str, nlu: dict, context: dict | None = None) -> dict:
    context = context or {}
    session = get_session(session_id)

    lang = choose_reply_lang(session, nlu["detected_lang"])

    for k, v in nlu["slots_normalized"].items():
        session["slots"][k] = v

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

    return {
        "session": session,
        "next_action": action,
        "reply_text": build_reply(action, lang),
        "reply_lang": lang,
    }
import re
from datetime import datetime, timedelta


URDU_RE = re.compile(r"[\u0600-\u06FF]")

URDU_NUMBER_MAP = {
    "ایک": 1,
    "دو": 2,
    "تین": 3,
    "چار": 4,
    "پانچ": 5,
    "چھ": 6,
    "سات": 7,
    "آٹھ": 8,
    "نو": 9,
    "دس": 10,
}


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_reply_lang(text: str) -> str:
    urdu_chars = len(URDU_RE.findall(text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if urdu_chars > 0 and latin_chars > 0:
        return "mixed"
    if urdu_chars > 0:
        return "ur"
    return "en"


def extract_date_from_text(text: str, today=None):
    today = today or datetime.utcnow().date()
    text = normalize_text(text)
    lower = text.lower()

    if "پرسوں" in text or "day after tomorrow" in lower:
        return (today + timedelta(days=2)).isoformat()
    if "کل" in text or "tomorrow" in lower:
        return (today + timedelta(days=1)).isoformat()
    if "آج" in text or "today" in lower:
        return today.isoformat()

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
        "پیر": 0,
        "منگل": 1,
        "بدھ": 2,
        "جمعرات": 3,
        "جمعہ": 4,
        "ہفتہ": 5,
        "اتوار": 6,
    }

    for k, v in weekday_map.items():
        if k in lower or k in text:
            delta = (v - today.weekday() + 7) % 7
            delta = 7 if delta == 0 else delta
            return (today + timedelta(days=delta)).isoformat()

    return None


def parse_seat_count(text: str):
    text = normalize_text(text)
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    for k, v in URDU_NUMBER_MAP.items():
        if k in text:
            return v
    return None


def extract_route_choice(text: str):
    t = normalize_text(text).lower()
    if any(x in t for x in ["first", "option 1", "number 1", "پہلی", "پہلا"]):
        return 1
    if any(x in t for x in ["second", "option 2", "number 2", "دوسری", "دوسرا"]):
        return 2
    if any(x in t for x in ["third", "option 3", "number 3", "تیسری", "تیسرا"]):
        return 3
    return None
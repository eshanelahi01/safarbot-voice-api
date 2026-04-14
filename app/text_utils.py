import re
from datetime import date, datetime, timedelta


URDU_RE = re.compile(r"[\u0600-\u06FF]")

EN_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

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

SEAT_HINTS_EN = (
    "seat",
    "seats",
    "ticket",
    "tickets",
    "passenger",
    "passengers",
    "person",
    "people",
)

SEAT_HINTS_UR = (
    "سیٹ",
    "سیٹیں",
    "ٹکٹ",
    "ٹکٹس",
    "مسافر",
    "افراد",
)

EXPLICIT_ROUTE_PATTERNS = {
    1: ("first", "1st", "option 1", "number 1", "route 1", "bus 1"),
    2: ("second", "2nd", "option 2", "number 2", "route 2", "bus 2"),
    3: ("third", "3rd", "option 3", "number 3", "route 3", "bus 3"),
}

URDU_ROUTE_PATTERNS = {
    1: ("پہلی", "پہلا", "آپشن 1", "نمبر 1"),
    2: ("دوسری", "دوسرا", "آپشن 2", "نمبر 2"),
    3: ("تیسری", "تیسرا", "آپشن 3", "نمبر 3"),
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


def _extract_number_value(text: str) -> int | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    match = re.search(r"\b(\d+)\b", normalized)
    if match:
        return int(match.group(1))

    lower = normalized.lower()
    for word, value in EN_NUMBER_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return value

    for word, value in URDU_NUMBER_MAP.items():
        if word in normalized:
            return value

    return None


def extract_date_from_text(text: str, today: date | None = None):
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

    for weekday, weekday_index in weekday_map.items():
        if weekday in lower or weekday in text:
            delta = (weekday_index - today.weekday() + 7) % 7
            delta = 7 if delta == 0 else delta
            return (today + timedelta(days=delta)).isoformat()

    return None


def parse_seat_count(text: str, expected_action: str | None = None):
    normalized = normalize_text(text)
    if not normalized:
        return None

    lower = normalized.lower()
    seat_hints = any(keyword in lower for keyword in SEAT_HINTS_EN) or any(
        keyword in normalized for keyword in SEAT_HINTS_UR
    )
    explicit_route_choice = extract_route_choice(normalized)

    if explicit_route_choice is not None and not seat_hints:
        return None

    if not seat_hints and expected_action != "ASK_SEAT_COUNT":
        return None

    return _extract_number_value(normalized)


def extract_route_choice(text: str, expected_action: str | None = None):
    normalized = normalize_text(text)
    if not normalized:
        return None

    lower = normalized.lower()

    for value, patterns in EXPLICIT_ROUTE_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return value

    for value, patterns in URDU_ROUTE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return value

    if expected_action == "SEARCH_ROUTES":
        stripped = normalized.strip()
        if stripped.isdigit():
            bare_value = int(stripped)
            return bare_value if 1 <= bare_value <= 3 else None

        lower_stripped = stripped.lower()
        for word, value in EN_NUMBER_MAP.items():
            if lower_stripped == word and 1 <= value <= 3:
                return value

        for word, value in URDU_NUMBER_MAP.items():
            if stripped == word and 1 <= value <= 3:
                return value

    return None

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from rapidfuzz import fuzz, process

from app.config import settings


URDU_RE = re.compile(r"[\u0600-\u06FF]")
SPELLED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z][\-./]){1,}[A-Za-z]\b")

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

DEFAULT_ASSETS = {
    "cities": [],
    "providers": [],
    "terminals": [],
    "city_aliases": {},
    "provider_aliases": {},
    "terminal_aliases": {},
    "payment_aliases": {
        "cash": "cash",
        "naqad": "cash",
        "naqd": "cash",
        "نقد": "cash",
        "cash on terminal": "cash",
    },
}


def _collapse_spelled_acronyms(text: str) -> str:
    return SPELLED_ACRONYM_RE.sub(
        lambda match: re.sub(r"[^A-Za-z]", "", match.group(0)),
        text,
    )


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = _collapse_spelled_acronyms(text)
    return re.sub(r"\s+", " ", text)


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


def extract_date_from_text(text: str, today: date | None = None) -> str | None:
    today = today or datetime.utcnow().date()
    normalized = normalize_text(text)
    lower = normalized.lower()

    if "پرسوں" in normalized or "parson" in lower or "day after tomorrow" in lower:
        return (today + timedelta(days=2)).isoformat()
    if "کل" in normalized or "kal" in lower or "tomorrow" in lower:
        return (today + timedelta(days=1)).isoformat()
    if "آج" in normalized or "aj" in lower or "aaj" in lower or "today" in lower:
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
        if weekday in lower or weekday in normalized:
            delta = (weekday_index - today.weekday() + 7) % 7
            delta = 7 if delta == 0 else delta
            return (today + timedelta(days=delta)).isoformat()

    return None


def extract_route_choice(text: str, expected_action: str | None = None) -> int | None:
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

    if expected_action in {"SEARCH_ROUTES", "CALL_GET_ROUTES", "ASK_ROUTE_CHOICE"}:
        if normalized.isdigit():
            bare_value = int(normalized)
            return bare_value if 1 <= bare_value <= 3 else None

        lower_stripped = normalized.lower()
        for word, value in EN_NUMBER_MAP.items():
            if lower_stripped == word and 1 <= value <= 3:
                return value

        for word, value in URDU_NUMBER_MAP.items():
            if normalized == word and 1 <= value <= 3:
                return value

    return None


def parse_seat_count(text: str, expected_action: str | None = None) -> int | None:
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


def extract_payment_from_text(text: str) -> str | None:
    normalized = normalize_text(text).casefold()
    if not normalized:
        return None

    for alias, canonical in normalizer_service.assets["payment_aliases"].items():
        if alias.casefold() in normalized:
            return canonical

    return None


class NormalizerService:
    def __init__(self):
        self.assets = dict(DEFAULT_ASSETS)
        self.ready = False
        self.error = None

    def load(self):
        self.ready = False
        self.error = None
        self.assets = dict(DEFAULT_ASSETS)

        try:
            asset_path = Path(settings.NORMALIZER_ASSET_PATH)
            if not asset_path.exists():
                raise FileNotFoundError(f"Normalizer assets not found at {asset_path}")

            with open(asset_path, "r", encoding="utf-8") as handle:
                file_assets = json.load(handle)

            merged_assets = dict(DEFAULT_ASSETS)
            merged_assets.update(file_assets)
            self.assets = merged_assets
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "asset_path": str(settings.NORMALIZER_ASSET_PATH),
        }

    def fuzzy_match(self, value: str, candidates: list[str], threshold: int) -> tuple[str | None, float]:
        if not value or not candidates:
            return None, 0

        match = process.extractOne(value, candidates, scorer=fuzz.WRatio)
        if not match:
            return None, 0

        best_value, score, _ = match
        if score >= threshold:
            return best_value, float(score)
        return None, float(score)

    def _canonicalize(
        self,
        value: str | None,
        alias_map: dict[str, str],
        candidates: list[str],
        threshold: int,
    ) -> tuple[str | None, dict]:
        if value is None:
            return None, {}

        normalized = normalize_text(value)
        alias_match = alias_map.get(normalized.casefold())
        if alias_match:
            return alias_match, {"source": "alias"}

        best, score = self.fuzzy_match(normalized, candidates, threshold)
        if best:
            return best, {"source": "fuzzy", "score": score}

        return normalized, {}

    def normalize_slots(
        self,
        slots_raw: dict,
        text: str = "",
        expected_action: str | None = None,
    ) -> tuple[dict, dict]:
        if not self.ready:
            self.load()

        normalized = {
            key: value
            for key, value in dict(slots_raw).items()
            if value is not None and value != ""
        }
        meta = {}

        for key, asset_key, meta_key, threshold in (
            ("from", "cities", "from", 83),
            ("to", "cities", "to", 83),
            ("provider", "providers", "provider", 80),
            ("terminal", "terminals", "terminal", 80),
        ):
            alias_map = self.assets.get(f"{key}_aliases", {})
            if key in {"from", "to"}:
                alias_map = self.assets.get("city_aliases", {})
            elif key == "provider":
                alias_map = self.assets.get("provider_aliases", {})
            elif key == "terminal":
                alias_map = self.assets.get("terminal_aliases", {})

            canonical, slot_meta = self._canonicalize(
                normalized.get(key),
                alias_map,
                self.assets.get(asset_key, []),
                threshold,
            )
            if canonical:
                normalized[key] = canonical
            if slot_meta:
                meta[f"{meta_key}_normalization"] = slot_meta

        if normalized.get("payment"):
            normalized["payment"] = self.normalize_payment(str(normalized["payment"]))
        else:
            payment = extract_payment_from_text(text)
            if payment:
                normalized["payment"] = payment
                meta["payment_source"] = "rule"

        if normalized.get("seat_count") is not None:
            normalized["seat_count"] = _extract_number_value(str(normalized["seat_count"]))

        normalized_date = normalized.get("date")
        if normalized_date:
            date_rule = extract_date_from_text(str(normalized_date))
            normalized["date"] = date_rule or normalize_text(str(normalized_date))
            if date_rule:
                meta["date_source"] = "slot_rule"
        else:
            date_rule = extract_date_from_text(text)
        if date_rule and not normalized.get("date"):
            normalized["date"] = date_rule
            meta["date_source"] = "rule"

        route_choice = normalized.get("route_choice")
        if route_choice is None:
            route_choice = extract_route_choice(text, expected_action=expected_action)
            if route_choice is not None:
                meta["route_choice_source"] = "rule"
        elif isinstance(route_choice, str) and route_choice.isdigit():
            route_choice = int(route_choice)

        if route_choice is not None:
            normalized["route_choice"] = route_choice

        seat_count = normalized.get("seat_count")
        if seat_count is None:
            seat_count = parse_seat_count(text, expected_action=expected_action)
            if seat_count is not None:
                meta["seat_count_source"] = "rule"
        if seat_count is not None:
            normalized["seat_count"] = seat_count

        return normalized, meta

    def normalize_city(self, word: str) -> str:
        canonical, _ = self._canonicalize(
            word,
            self.assets.get("city_aliases", {}),
            self.assets.get("cities", []),
            83,
        )
        return canonical or word

    def normalize_payment(self, word: str) -> str:
        return self.assets.get("payment_aliases", {}).get(word.casefold(), word)


def _aggregate_slot_pairs(slot_pairs: list[tuple[str, str, float]] | list[tuple[str, str]]) -> list[tuple[str, str]]:
    aggregated: list[tuple[str, str]] = []
    current_words: list[str] = []
    current_label: str | None = None

    for item in slot_pairs:
        word = str(item[0])
        label = str(item[1]).upper()
        prefix, _, entity = label.partition("-")

        if prefix == "B" and entity:
            if current_words and current_label:
                aggregated.append((" ".join(current_words), current_label))
            current_words = [word]
            current_label = entity
            continue

        if prefix == "I" and entity and current_label == entity:
            current_words.append(word)
            continue

        if current_words and current_label:
            aggregated.append((" ".join(current_words), current_label))
            current_words = []
            current_label = None

        normalized_label = entity if prefix in {"B", "I"} and entity else label
        aggregated.append((word, normalized_label))

    if current_words and current_label:
        aggregated.append((" ".join(current_words), current_label))

    return aggregated


def extract_raw_entities(slot_pairs: list[tuple[str, str, float]] | list[tuple[str, str]]) -> dict:
    raw = {
        "from": None,
        "to": None,
        "date": None,
        "time": None,
        "seat_count": None,
        "payment": None,
        "provider": None,
        "terminal": None,
        "route_choice": None,
    }

    for word, label in _aggregate_slot_pairs(slot_pairs):
        if label == "FROM":
            raw["from"] = word
        elif label == "TO":
            raw["to"] = word
        elif label == "DATE":
            raw["date"] = word
        elif label == "TIME":
            raw["time"] = word
        elif label == "SEAT_COUNT":
            raw["seat_count"] = word
        elif label == "PAYMENT":
            raw["payment"] = word
        elif label == "PROVIDER":
            raw["provider"] = word
        elif label == "TERMINAL":
            raw["terminal"] = word
        elif label in {"ROUTE_CHOICE", "ROUTE"}:
            raw["route_choice"] = word

    return {key: value for key, value in raw.items() if value is not None}


normalizer_service = NormalizerService()


def load_normalizer_assets():
    normalizer_service.load()


def normalizer_status() -> dict:
    return normalizer_service.status()


def extract_entities(
    slot_pairs: list[tuple[str, str, float]] | list[tuple[str, str]],
    text: str = "",
    expected_action: str | None = None,
) -> tuple[dict, dict]:
    raw = extract_raw_entities(slot_pairs)
    return normalizer_service.normalize_slots(raw, text=text, expected_action=expected_action)

import re

from app.core.normalizer import (
    extract_date_from_text,
    extract_payment_from_text,
    extract_route_choice,
    normalizer_service,
    normalize_text,
    parse_seat_count,
)


GREETING_KEYWORDS = (
    "hi",
    "hello",
    "salam",
    "assalam",
    "aoa",
    "hey",
    "اسلام",
    "السلام",
)
CONFIRM_KEYWORDS = (
    "yes",
    "book",
    "confirm",
    "done",
    "proceed",
    "kar do",
    "kr do",
    "haan",
    "han",
    "جی",
    "ہاں",
    "بک",
)
NEGATIVE_KEYWORDS = (
    "no",
    "not",
    "cancel",
    "stop",
    "nah",
    "nahi",
    "نہیں",
)
SEAT_KEYWORDS = (
    "seat",
    "seats",
    "ticket",
    "tickets",
    "passenger",
    "passengers",
    "person",
    "people",
    "سیٹ",
    "ٹکٹ",
    "مسافر",
)
FROM_HINTS = ("from", "depart", "departing", "se", "sy", "say", "سے")
TO_HINTS = ("to", "for", "tak", "تک")


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    text_casefold = text.casefold()
    return any(pattern.casefold() in text_casefold for pattern in patterns)


def _build_catalog_candidates(candidates: list[str], alias_map: dict[str, str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        pair = (normalize_text(candidate), candidate)
        if pair not in seen:
            seen.add(pair)
            items.append(pair)

    for alias, canonical in alias_map.items():
        pair = (normalize_text(alias), canonical)
        if pair not in seen:
            seen.add(pair)
            items.append(pair)

    return items


def _find_catalog_mentions(
    text: str,
    candidates: list[str],
    alias_map: dict[str, str],
) -> list[dict]:
    normalized_text = normalize_text(text)
    text_casefold = normalized_text.casefold()
    mentions: list[dict] = []

    for phrase, canonical in _build_catalog_candidates(candidates, alias_map):
        target = phrase.casefold()
        if not target:
            continue

        pattern = re.compile(rf"(?<!\w){re.escape(target)}(?!\w)")
        for match in pattern.finditer(text_casefold):
            mentions.append(
                {
                    "value": canonical,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    mentions.sort(key=lambda item: (item["start"], item["end"] - item["start"]))

    deduped: list[dict] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for mention in mentions:
        key = (mention["start"], mention["end"], mention["value"])
        if key in seen_spans:
            continue
        seen_spans.add(key)
        deduped.append(mention)

    return deduped


def _find_first_after(mentions: list[dict], anchor: int) -> str | None:
    for mention in mentions:
        if mention["start"] >= anchor:
            return mention["value"]
    return None


def _find_last_before(mentions: list[dict], anchor: int) -> str | None:
    for mention in reversed(mentions):
        if mention["end"] <= anchor:
            return mention["value"]
    return None


def _extract_city_slots(text: str, expected_action: str | None = None) -> dict:
    if not normalizer_service.ready:
        normalizer_service.load()

    mentions = _find_catalog_mentions(
        text,
        normalizer_service.assets.get("cities", []),
        normalizer_service.assets.get("city_aliases", {}),
    )
    if not mentions:
        return {}

    text_casefold = normalize_text(text).casefold()
    slots: dict[str, str] = {}

    for hint in FROM_HINTS:
        anchor = text_casefold.find(f"{hint.casefold()} ")
        if anchor >= 0:
            candidate = _find_first_after(mentions, anchor)
            if candidate:
                slots["from"] = candidate
                break

    for hint in TO_HINTS:
        anchor = text_casefold.find(f"{hint.casefold()} ")
        if anchor >= 0:
            candidate = _find_first_after(mentions, anchor)
            if candidate:
                slots["to"] = candidate
                break

    for connector in (" se ", " sy ", " say ", " سے ", " from "):
        anchor = text_casefold.find(connector)
        if anchor >= 0:
            slots.setdefault("from", _find_last_before(mentions, anchor))
            slots.setdefault("to", _find_first_after(mentions, anchor + len(connector)))

    if "to" in text_casefold:
        anchor = text_casefold.find("to")
        slots.setdefault("to", _find_first_after(mentions, anchor))

    distinct_mentions = []
    seen_values: set[str] = set()
    for mention in mentions:
        if mention["value"] in seen_values:
            continue
        seen_values.add(mention["value"])
        distinct_mentions.append(mention["value"])

    if len(distinct_mentions) >= 2:
        slots.setdefault("from", distinct_mentions[0])
        slots.setdefault("to", distinct_mentions[1])
    elif len(distinct_mentions) == 1:
        only_city = distinct_mentions[0]
        if expected_action == "ASK_FROM":
            slots["from"] = only_city
        elif expected_action == "ASK_TO":
            slots["to"] = only_city

    return {key: value for key, value in slots.items() if value}


def _extract_catalog_slot(text: str, catalog_key: str, alias_key: str) -> str | None:
    if not normalizer_service.ready:
        normalizer_service.load()

    mentions = _find_catalog_mentions(
        text,
        normalizer_service.assets.get(catalog_key, []),
        normalizer_service.assets.get(alias_key, {}),
    )
    if not mentions:
        return None
    return mentions[0]["value"]


def infer_slots(text: str, expected_action: str | None = None) -> tuple[dict, dict]:
    raw_slots = _extract_city_slots(text, expected_action=expected_action)

    for key, catalog_key, alias_key in (
        ("provider", "providers", "provider_aliases"),
        ("terminal", "terminals", "terminal_aliases"),
    ):
        value = _extract_catalog_slot(text, catalog_key, alias_key)
        if value:
            raw_slots[key] = value

    date_value = extract_date_from_text(text)
    if date_value:
        raw_slots["date"] = date_value

    route_choice = extract_route_choice(text, expected_action=expected_action)
    if route_choice is not None:
        raw_slots["route_choice"] = route_choice

    seat_count = parse_seat_count(text, expected_action=expected_action)
    if seat_count is not None:
        raw_slots["seat_count"] = seat_count

    payment = extract_payment_from_text(text)
    if payment:
        raw_slots["payment"] = payment

    slots_normalized, correction_meta = normalizer_service.normalize_slots(
        raw_slots,
        text=text,
        expected_action=expected_action,
    )
    return raw_slots, {
        **correction_meta,
        "nlu_backend": "rule_based",
    }


def infer_intent(text: str, slots: dict, expected_action: str | None = None) -> tuple[str, float, list[dict]]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return "fallback", 0.0, [{"label": "fallback", "score": 0.0}]

    if _contains_any(normalized_text, GREETING_KEYWORDS) and not slots:
        return "greeting", 0.98, [{"label": "greeting", "score": 0.98}]

    if _contains_any(normalized_text, CONFIRM_KEYWORDS) and not _contains_any(
        normalized_text,
        NEGATIVE_KEYWORDS,
    ):
        return "confirm_booking", 0.92, [{"label": "confirm_booking", "score": 0.92}]

    if slots.get("route_choice") is not None:
        return "select_route", 0.9, [{"label": "select_route", "score": 0.9}]

    if slots.get("seat_count") is not None and (
        expected_action == "ASK_SEAT_COUNT" or _contains_any(normalized_text, SEAT_KEYWORDS)
    ):
        return "select_seats", 0.9, [{"label": "select_seats", "score": 0.9}]

    if any(slots.get(key) for key in ("from", "to", "date", "provider", "terminal")):
        return "search_routes", 0.88, [{"label": "search_routes", "score": 0.88}]

    if slots.get("payment"):
        return "provide_payment", 0.86, [{"label": "provide_payment", "score": 0.86}]

    if _contains_any(normalized_text, GREETING_KEYWORDS):
        return "greeting", 0.8, [{"label": "greeting", "score": 0.8}]

    return "fallback", 0.35, [{"label": "fallback", "score": 0.35}]


def predict_rule_based(text: str, expected_action: str | None = None) -> dict:
    user_text = normalize_text(text)
    slots_raw, correction_meta = infer_slots(user_text, expected_action=expected_action)
    slots_normalized = dict(slots_raw)
    slots_normalized.update(
        normalizer_service.normalize_slots(
            slots_raw,
            text=user_text,
            expected_action=expected_action,
        )[0]
    )
    intent, confidence, top3 = infer_intent(
        user_text,
        slots_normalized,
        expected_action=expected_action,
    )

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "slots_raw": slots_raw,
        "slots_normalized": slots_normalized,
        "correction_meta": correction_meta,
        "top3": top3,
        "nlu_backend": "rule_based",
    }

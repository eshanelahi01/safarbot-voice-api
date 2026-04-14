import json
from pathlib import Path

from rapidfuzz import fuzz, process


CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


class FuzzyService:
    def __init__(self):
        self.catalog = {"cities": [], "providers": [], "terminals": []}
        self.ready = False

    def load(self):
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
        self.ready = True

    def fuzzy_match(self, value: str, candidates: list[str], threshold: int):
        if not value or not candidates:
            return None, 0
        match = process.extractOne(value, candidates, scorer=fuzz.WRatio)
        if not match:
            return None, 0
        best_value, score, _ = match
        if score >= threshold:
            return best_value, score
        return None, score

    def normalize_slots(self, slots_raw: dict) -> tuple[dict, dict]:
        normalized = dict(slots_raw)
        meta = {}

        if slots_raw.get("from"):
            best, score = self.fuzzy_match(str(slots_raw["from"]), self.catalog["cities"], 83)
            if best:
                normalized["from"] = best
                meta["from_score"] = score

        if slots_raw.get("to"):
            best, score = self.fuzzy_match(str(slots_raw["to"]), self.catalog["cities"], 83)
            if best:
                normalized["to"] = best
                meta["to_score"] = score

        if slots_raw.get("provider"):
            best, score = self.fuzzy_match(str(slots_raw["provider"]), self.catalog["providers"], 80)
            if best:
                normalized["provider"] = best
                meta["provider_score"] = score

        if slots_raw.get("terminal"):
            best, score = self.fuzzy_match(str(slots_raw["terminal"]), self.catalog["terminals"], 80)
            if best:
                normalized["terminal"] = best
                meta["terminal_score"] = score

        return normalized, meta


fuzzy_service = FuzzyService()
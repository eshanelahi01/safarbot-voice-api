import unittest
from unittest.mock import patch

from app.core.normalizer import extract_entities, normalizer_service


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        normalizer_service.load()

    def test_extract_entities_applies_city_alias_and_rule_slots(self):
        slots, meta = extract_entities(
            [("isb", "FROM"), ("Lahore", "TO")],
            text="isb to Lahore kal 2 seats cash",
            expected_action="ASK_SEAT_COUNT",
        )

        self.assertEqual(slots["from"], "Islamabad")
        self.assertEqual(slots["to"], "Lahore")
        self.assertEqual(slots["seat_count"], 2)
        self.assertEqual(slots["payment"], "cash")
        self.assertIn("date_source", meta)

    @patch("app.core.normalizer.extract_date_from_text", return_value="2026-04-22")
    def test_extract_entities_normalizes_spelled_acronym_and_relative_slot_date(self, *_):
        slots, meta = extract_entities(
            [("I-S-B", "FROM"), ("tomorrow.", "DATE")],
            text="I-S-B to Lahore tomorrow.",
        )

        self.assertEqual(slots["from"], "Islamabad")
        self.assertEqual(slots["date"], "2026-04-22")
        self.assertEqual(meta["date_source"], "slot_rule")


if __name__ == "__main__":
    unittest.main()

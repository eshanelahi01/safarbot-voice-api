import unittest

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


if __name__ == "__main__":
    unittest.main()

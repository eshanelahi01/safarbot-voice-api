import unittest
from datetime import date

from app.text_utils import extract_date_from_text, extract_route_choice, parse_seat_count


class TextUtilsTests(unittest.TestCase):
    def test_explicit_route_choice_does_not_turn_into_seat_count(self):
        self.assertEqual(extract_route_choice("option 2"), 2)
        self.assertIsNone(parse_seat_count("option 2"))

    def test_bare_number_respects_dialogue_state(self):
        self.assertEqual(extract_route_choice("2", expected_action="SEARCH_ROUTES"), 2)
        self.assertEqual(parse_seat_count("2", expected_action="ASK_SEAT_COUNT"), 2)
        self.assertIsNone(parse_seat_count("2"))

    def test_seat_hints_still_parse_count(self):
        self.assertEqual(parse_seat_count("I need 3 seats"), 3)
        self.assertEqual(parse_seat_count("3 passengers please"), 3)

    def test_relative_urdu_date_is_normalized(self):
        self.assertEqual(extract_date_from_text("آج", today=date(2026, 4, 14)), "2026-04-14")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from app.core.normalizer import load_normalizer_assets
from app.nlu import predict_text


class RuleBasedNLUFallbackTests(unittest.TestCase):
    def setUp(self):
        load_normalizer_assets()

    @patch("app.nlu.predict_slots", side_effect=RuntimeError("slot model unavailable"))
    @patch("app.nlu.predict_intent_with_scores", side_effect=RuntimeError("intent model unavailable"))
    def test_predict_text_falls_back_to_rule_based_nlu(self, *_):
        result = predict_text("isb to Lahore kal 2 seats cash", expected_action="ASK_SEAT_COUNT")

        self.assertEqual(result["nlu_backend"], "rule_based")
        self.assertEqual(result["intent"], "select_seats")
        self.assertEqual(result["slots_normalized"]["from"], "Islamabad")
        self.assertEqual(result["slots_normalized"]["to"], "Lahore")
        self.assertEqual(result["slots_normalized"]["seat_count"], 2)
        self.assertEqual(result["slots_normalized"]["payment"], "cash")
        self.assertIn("nlu_fallback_reason", result["correction_meta"])

    @patch(
        "app.nlu.predict_slots",
        return_value=[
            ("Lahore", "TO", 0.99),
            ("tomorrow.", "DATE", 0.99),
        ],
    )
    @patch(
        "app.nlu.predict_intent_with_scores",
        return_value=("search_routes", 0.91, [{"label": "search_routes", "score": 0.91}]),
    )
    @patch("app.core.normalizer.extract_date_from_text", return_value="2026-04-22")
    def test_predict_text_supplements_missing_transformer_slots_with_rule_based(self, *_):
        result = predict_text("I-S-B to Lahore tomorrow")

        self.assertEqual(result["nlu_backend"], "transformers")
        self.assertEqual(result["slots_normalized"]["from"], "Islamabad")
        self.assertEqual(result["slots_normalized"]["to"], "Lahore")
        self.assertEqual(result["slots_normalized"]["date"], "2026-04-22")
        self.assertIn("from", result["correction_meta"]["rule_based_slot_supplement"])


if __name__ == "__main__":
    unittest.main()

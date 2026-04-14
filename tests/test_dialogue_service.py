import unittest

from app.services.dialogue_service import decide, get_session
from app.services.session_store import session_store


class DialogueServiceTests(unittest.TestCase):
    def setUp(self):
        session_store.clear()

    def test_context_round_trip_completes_existing_conversation(self):
        context = {
            "conversation_state": {
                "reply_lang": "en",
                "slots": {
                    "from": "Lahore",
                    "to": "Islamabad",
                    "date": "2026-04-20",
                },
                "route_choice": 1,
                "last_action": "ASK_SEAT_COUNT",
            }
        }
        session = get_session("session-1", context)

        decision = decide(
            "session-1",
            {
                "detected_lang": "en",
                "slots_normalized": {"seat_count": 2},
                "intent_confidence": 0.95,
            },
            context=context,
            session=session,
        )

        self.assertEqual(decision["next_action"], "FINAL_NOTICE")
        self.assertEqual(decision["conversation_state"]["seat_count"], 2)
        self.assertEqual(decision["conversation_state"]["slots"]["from"], "Lahore")


if __name__ == "__main__":
    unittest.main()

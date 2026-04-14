import unittest

from app.services.session_store import MemorySessionStore, new_session_state


class MemorySessionStoreTests(unittest.TestCase):
    def test_store_evicts_oldest_session_when_capacity_is_reached(self):
        store = MemorySessionStore(ttl_seconds=60, max_sessions=2)

        first = new_session_state()
        first["last_action"] = "ASK_FROM"
        store.save("a", first)

        second = new_session_state()
        second["last_action"] = "ASK_TO"
        store.save("b", second)

        third = new_session_state()
        third["last_action"] = "ASK_DATE"
        store.save("c", third)

        self.assertEqual(store.size(), 2)
        self.assertIsNone(store.peek("a"))
        self.assertEqual(store.peek("b")["last_action"], "ASK_TO")
        self.assertEqual(store.peek("c")["last_action"], "ASK_DATE")


if __name__ == "__main__":
    unittest.main()

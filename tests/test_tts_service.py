import unittest
from unittest.mock import patch

from app.tts.azure import synthesize_text, tts_service, tts_status


class TTSServiceTests(unittest.TestCase):
    @patch.object(tts_service, "_configured", return_value=False)
    @patch.object(tts_service, "_gtts_available", return_value=True)
    @patch.object(tts_service, "_local_available", return_value=True)
    @patch.object(
        tts_service,
        "_synthesize_gtts",
        return_value={
            "audio_base64": "ZmFrZS1hdWRpbw==",
            "audio_mime_type": "audio/mpeg",
            "engine": "gtts",
            "error": None,
        },
    )
    @patch.object(
        tts_service,
        "_synthesize_local",
        return_value={
            "audio_base64": "bG9jYWw=",
            "audio_mime_type": "audio/wav",
            "engine": "pyttsx3",
            "error": None,
        },
    )
    def test_synthesize_prefers_gtts_before_local_fallback(self, local_mock, gtts_mock, *_):
        result = synthesize_text("Testing voice response", lang="en")

        gtts_mock.assert_called_once_with("Testing voice response", "en")
        local_mock.assert_not_called()
        self.assertEqual(result["engine"], "gtts")

    @patch.object(tts_service, "_configured", return_value=False)
    @patch.object(tts_service, "_gtts_available", return_value=True)
    @patch.object(tts_service, "_local_available", return_value=False)
    def test_status_reports_gtts_as_ready_engine(self, *_):
        status = tts_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["engine"], "gtts")
        self.assertIn("gtts", status["available_engines"])


if __name__ == "__main__":
    unittest.main()

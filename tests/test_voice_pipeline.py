import base64
import unittest
from unittest.mock import patch

from app.core.dialogue import load_dialogue_rules
from app.core.normalizer import load_normalizer_assets
from app.core.tools import BackendServiceError
from app.schemas import Query, VoiceChatRequest
from app.services.session_store import session_store
from app.services.voice_pipeline import build_chat_response, build_voice_response


class VoicePipelineTests(unittest.TestCase):
    def setUp(self):
        session_store.clear()
        load_normalizer_assets()
        load_dialogue_rules()

    @patch(
        "app.services.voice_pipeline.synthesize_text",
        return_value={
            "audio_base64": "ZmFrZS1hdWRpbw==",
            "audio_mime_type": "audio/mpeg",
            "engine": "azure",
            "error": None,
        },
    )
    @patch(
        "app.services.voice_pipeline.predict_text",
        return_value={
            "user_text": "hello",
            "detected_lang": "en",
            "intent": "greeting",
            "intent_confidence": 0.98,
            "slots_raw": {},
            "slots_normalized": {},
            "correction_meta": {"nlu_backend": "rule_based"},
            "top3": [{"label": "greeting", "score": 0.98}],
            "nlu_backend": "rule_based",
        },
    )
    def test_chat_response_always_contains_tts_and_passthrough_stt(self, *_):
        response = build_chat_response(Query(text="hello"))

        self.assertEqual(response.audio_base64, "ZmFrZS1hdWRpbw==")
        self.assertEqual(response.pipeline_meta["stt_engine"], "passthrough")
        self.assertEqual(response.pipeline_meta["tts_engine"], "azure")
        self.assertEqual(response.type, "message")

    @patch(
        "app.services.voice_pipeline.synthesize_text",
        return_value={
            "audio_base64": "ZmFrZS1hdWRpbw==",
            "audio_mime_type": "audio/mpeg",
            "engine": "azure",
            "error": None,
        },
    )
    @patch(
        "app.services.voice_pipeline.get_routes",
        side_effect=BackendServiceError(
            "route_lookup failed with status 420",
            status_code=420,
            response_body={"message": "slow down"},
            operation="route_lookup",
        ),
    )
    @patch(
        "app.services.voice_pipeline.predict_text",
        return_value={
            "user_text": "lahore to islamabad tomorrow",
            "detected_lang": "en",
            "intent": "search_routes",
            "intent_confidence": 0.95,
            "slots_raw": {"from": "Lahore", "to": "Islamabad", "date": "2026-04-22"},
            "slots_normalized": {"from": "Lahore", "to": "Islamabad", "date": "2026-04-22"},
            "correction_meta": {"nlu_backend": "rule_based"},
            "top3": [{"label": "search_routes", "score": 0.95}],
            "nlu_backend": "rule_based",
        },
    )
    def test_backend_420_becomes_structured_error_payload(self, *_):
        response = build_chat_response(Query(text="lahore to islamabad tomorrow"))

        self.assertEqual(response.type, "error")
        self.assertEqual(response.data["upstream_status"], 420)
        self.assertEqual(response.audio_base64, "ZmFrZS1hdWRpbw==")
        self.assertIn("live routes", response.response.lower())

    @patch(
        "app.services.voice_pipeline.synthesize_text",
        return_value={
            "audio_base64": "ZmFrZS1hdWRpbw==",
            "audio_mime_type": "audio/mpeg",
            "engine": "azure",
            "error": None,
        },
    )
    @patch(
        "app.services.voice_pipeline.get_routes",
        return_value=[
            {
                "_id": "route-1",
                "provider": "Daewoo",
                "departure_time": "09:00",
                "price": 2500,
            }
        ],
    )
    @patch(
        "app.services.voice_pipeline.predict_text",
        return_value={
            "user_text": "lahore to islamabad tomorrow",
            "detected_lang": "en",
            "intent": "search_routes",
            "intent_confidence": 0.95,
            "slots_raw": {"from": "Lahore", "to": "Islamabad", "date": "2026-04-22"},
            "slots_normalized": {"from": "Lahore", "to": "Islamabad", "date": "2026-04-22"},
            "correction_meta": {"nlu_backend": "rule_based"},
            "top3": [{"label": "search_routes", "score": 0.95}],
            "nlu_backend": "rule_based",
        },
    )
    @patch("app.services.voice_pipeline.transcribe_audio", return_value="lahore to islamabad tomorrow")
    def test_voice_response_runs_stt_and_tts(self, *_):
        payload = VoiceChatRequest(audio_base64=base64.b64encode(b"fake-audio").decode("ascii"))
        response = build_voice_response(payload)

        self.assertEqual(response.pipeline_meta["input_source"], "voice")
        self.assertEqual(response.pipeline_meta["stt_engine"], "faster-whisper")
        self.assertEqual(response.pipeline_meta["tts_engine"], "azure")
        self.assertEqual(len(response.routes_preview), 1)
        self.assertEqual(response.routes_preview[0].route_id, "route-1")
        self.assertEqual(response.audio_base64, "ZmFrZS1hdWRpbw==")

    @patch(
        "app.services.voice_pipeline.synthesize_text",
        return_value={
            "audio_base64": "ZmFrZS1hdWRpbw==",
            "audio_mime_type": "audio/mpeg",
            "engine": "gtts",
            "error": None,
        },
    )
    @patch(
        "app.services.voice_pipeline.predict_text",
        return_value={
            "user_text": "hello",
            "detected_lang": "en",
            "intent": "greeting",
            "intent_confidence": 0.98,
            "slots_raw": {},
            "slots_normalized": {},
            "correction_meta": {"nlu_backend": "rule_based"},
            "top3": [{"label": "greeting", "score": 0.98}],
            "nlu_backend": "rule_based",
        },
    )
    @patch("app.services.voice_pipeline.transcribe_audio", return_value="hello")
    def test_voice_response_passes_detected_audio_format_to_stt(
        self,
        transcribe_audio_mock,
        *_,
    ):
        payload = VoiceChatRequest(
            audio_base64=f"data:audio/mp3;base64,{base64.b64encode(b'fake-audio').decode('ascii')}",
            audio_format="wav",
        )

        response = build_voice_response(payload)

        transcribe_audio_mock.assert_called_once_with(b"fake-audio", audio_format="mp3")
        self.assertEqual(response.pipeline_meta["input_source"], "voice")
        self.assertEqual(response.audio_mime_type, "audio/mpeg")


if __name__ == "__main__":
    unittest.main()

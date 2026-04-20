from app.nlu import load_models, model_status
from app.nlu.intent import intent_predictor
from app.nlu.slot import slot_predictor


class ModelRegistry:
    def load(self):
        load_models()

    @property
    def ready(self) -> bool:
        status = model_status()
        return status["intent_model"]["ready"] and status["slot_model"]["ready"]

    @property
    def error(self):
        status = model_status()
        return status["intent_model"]["error"] or status["slot_model"]["error"]

    @property
    def intent_tokenizer(self):
        return intent_predictor.tokenizer

    @property
    def intent_model(self):
        return intent_predictor.model

    @property
    def slot_tokenizer(self):
        return slot_predictor.tokenizer

    @property
    def slot_model(self):
        return slot_predictor.model

    @property
    def intent_pipe(self):
        return None

    @property
    def slot_pipe(self):
        return None

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            **model_status(),
        }


registry = ModelRegistry()

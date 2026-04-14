from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from app.config import settings


class ModelRegistry:
    def __init__(self):
        self.error = None
        self._clear()

    def _clear(self):
        self.intent_tokenizer = None
        self.intent_model = None
        self.slot_tokenizer = None
        self.slot_model = None
        self.intent_pipe = None
        self.slot_pipe = None
        self.ready = False

    def load(self):
        self.error = None
        self._clear()

        try:
            self._load()
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            self._clear()

    def _load(self):
        missing_settings = []
        if not settings.HF_TOKEN:
            missing_settings.append("HF_TOKEN")
        if not settings.INTENT_MODEL_REPO:
            missing_settings.append("INTENT_MODEL_REPO")
        if not settings.SLOT_MODEL_REPO:
            missing_settings.append("SLOT_MODEL_REPO")

        if missing_settings:
            joined_settings = ", ".join(missing_settings)
            raise ValueError(f"Missing required settings: {joined_settings}")

        self.intent_tokenizer = AutoTokenizer.from_pretrained(
            settings.INTENT_MODEL_REPO,
            token=settings.HF_TOKEN,
        )
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            settings.INTENT_MODEL_REPO,
            token=settings.HF_TOKEN,
            low_cpu_mem_usage=True,
        )

        self.slot_tokenizer = AutoTokenizer.from_pretrained(
            settings.SLOT_MODEL_REPO,
            token=settings.HF_TOKEN,
        )
        self.slot_model = AutoModelForTokenClassification.from_pretrained(
            settings.SLOT_MODEL_REPO,
            token=settings.HF_TOKEN,
            low_cpu_mem_usage=True,
        )

        self.intent_pipe = pipeline(
            "text-classification",
            model=self.intent_model,
            tokenizer=self.intent_tokenizer,
            top_k=3,
            device=-1,
        )

        self.slot_pipe = pipeline(
            "token-classification",
            model=self.slot_model,
            tokenizer=self.slot_tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
        }


registry = ModelRegistry()

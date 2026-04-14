from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from app.config import settings


class ModelRegistry:
    def __init__(self):
        self.intent_tokenizer = None
        self.intent_model = None
        self.slot_tokenizer = None
        self.slot_model = None
        self.intent_pipe = None
        self.slot_pipe = None
        self.ready = False

    def load(self):
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN is not set")
        if not settings.INTENT_MODEL_REPO:
            raise ValueError("INTENT_MODEL_REPO is not set")
        if not settings.SLOT_MODEL_REPO:
            raise ValueError("SLOT_MODEL_REPO is not set")

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

        self.ready = True


registry = ModelRegistry()
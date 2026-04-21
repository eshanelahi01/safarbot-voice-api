from importlib.util import find_spec

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _torch_installed() -> bool:
    return find_spec("torch") is not None


def _transformers_installed() -> bool:
    return find_spec("transformers") is not None


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError:
        return None, None
    return AutoModelForSequenceClassification, AutoTokenizer


class IntentPredictor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.ready = False
        self.error = None

    def load(self):
        self.ready = False
        self.error = None
        self.tokenizer = None
        self.model = None

        if not settings.INTENT_MODEL:
            self.error = "INTENT_MODEL is not configured"
            return

        torch = _import_torch()
        if torch is None:
            self.error = "torch is not installed"
            return

        AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self.error = "transformers is not installed"
            return

        try:
            kwargs = {}
            if settings.HF_TOKEN:
                kwargs["token"] = settings.HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(settings.INTENT_MODEL, **kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.INTENT_MODEL,
                low_cpu_mem_usage=True,
                **kwargs,
            ).to(settings.DEVICE)
            self.model.eval()
            self.ready = True
            logger.info("Intent model loaded from %s", settings.INTENT_MODEL)
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to load intent model")

    def predict(self, text: str) -> tuple[str, float, list[dict]]:
        if self.error is not None and self.tokenizer is None and self.model is None:
            raise RuntimeError(self.error)

        if not self.ready or self.tokenizer is None or self.model is None:
            self.load()

        if not self.ready or self.tokenizer is None or self.model is None:
            raise RuntimeError(self.error or "Intent model is not ready")

        torch = _import_torch()
        if torch is None:
            raise RuntimeError("torch is not installed")
        AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers is not installed")

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(settings.DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        top_k = min(3, probabilities.shape[-1])
        top_indices = torch.topk(probabilities, k=top_k).indices.tolist()
        top_scores = [
            {
                "label": self.model.config.id2label.get(index, str(index)),
                "score": float(probabilities[index].item()),
            }
            for index in top_indices
        ]

        best = top_scores[0] if top_scores else {"label": "fallback", "score": 0.0}
        return best["label"], float(best["score"]), top_scores

    def status(self) -> dict:
        status_error = self.error
        if status_error is None:
            if not settings.INTENT_MODEL:
                status_error = "INTENT_MODEL is not configured"
            elif not _torch_installed():
                status_error = "torch is not installed"
            elif not _transformers_installed():
                status_error = "transformers is not installed"

        deferred = (
            not self.ready
            and status_error is None
            and not settings.MODEL_LOAD_ON_STARTUP
        )

        return {
            "ready": self.ready or deferred,
            "loaded": self.ready,
            "deferred": deferred,
            "error": status_error,
            "model": settings.INTENT_MODEL,
            "device": settings.DEVICE,
        }


intent_predictor = IntentPredictor()


def load_intent_model():
    intent_predictor.load()


def predict_intent(text: str) -> tuple[str, float]:
    label, confidence, _ = intent_predictor.predict(text)
    return label, confidence


def predict_intent_with_scores(text: str) -> tuple[str, float, list[dict]]:
    return intent_predictor.predict(text)


def intent_status() -> dict:
    return intent_predictor.status()

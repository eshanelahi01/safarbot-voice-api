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
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ModuleNotFoundError:
        return None, None
    return AutoModelForTokenClassification, AutoTokenizer


class SlotPredictor:
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

        if not settings.SLOT_MODEL:
            self.error = "SLOT_MODEL is not configured"
            return

        torch = _import_torch()
        if torch is None:
            self.error = "torch is not installed"
            return

        AutoModelForTokenClassification, AutoTokenizer = _import_transformers()
        if AutoTokenizer is None or AutoModelForTokenClassification is None:
            self.error = "transformers is not installed"
            return

        try:
            kwargs = {}
            if settings.HF_TOKEN:
                kwargs["token"] = settings.HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(settings.SLOT_MODEL, **kwargs)
            self.model = AutoModelForTokenClassification.from_pretrained(
                settings.SLOT_MODEL,
                low_cpu_mem_usage=True,
                **kwargs,
            ).to(settings.DEVICE)
            self.model.eval()
            self.ready = True
            logger.info("Slot model loaded from %s", settings.SLOT_MODEL)
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to load slot model")

    def predict(self, text: str) -> list[tuple[str, str, float]]:
        if self.error is not None and self.tokenizer is None and self.model is None:
            raise RuntimeError(self.error)

        if not self.ready or self.tokenizer is None or self.model is None:
            self.load()

        if not self.ready or self.tokenizer is None or self.model is None:
            raise RuntimeError(self.error or "Slot model is not ready")

        torch = _import_torch()
        if torch is None:
            raise RuntimeError("torch is not installed")
        AutoModelForTokenClassification, AutoTokenizer = _import_transformers()
        if AutoTokenizer is None or AutoModelForTokenClassification is None:
            raise RuntimeError("transformers is not installed")

        tokens = text.split()
        if not tokens:
            return []

        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            key: value.to(settings.DEVICE)
            for key, value in encoded.items()
            if hasattr(value, "to")
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu()

        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        results = []
        seen_word_ids = set()

        for index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue

            seen_word_ids.add(word_id)
            label_index = predictions[index]
            confidence = float(probabilities[index][label_index].item())
            if confidence < settings.SLOT_MIN_CONF:
                continue

            label = self.model.config.id2label.get(label_index, str(label_index))
            results.append((tokens[word_id], label, confidence))

        return results

    def status(self) -> dict:
        status_error = self.error
        if status_error is None:
            if not settings.SLOT_MODEL:
                status_error = "SLOT_MODEL is not configured"
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
            "model": settings.SLOT_MODEL,
            "device": settings.DEVICE,
        }


slot_predictor = SlotPredictor()


def load_slot_model():
    slot_predictor.load()


def predict_slots(text: str) -> list[tuple[str, str, float]]:
    return slot_predictor.predict(text)


def slot_status() -> dict:
    return slot_predictor.status()

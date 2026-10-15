"""Reusable wrapper for the published DeBERTa political-leaning model."""

from dataclasses import asdict, dataclass
import math

MODEL_ID = "matous-volf/political-leaning-deberta-large"
TOKENIZER_ID = "microsoft/deberta-v3-large"
MAX_TOKENS = 512
LABELS = {"LABEL_0": "left", "LABEL_1": "center", "LABEL_2": "right"}


@dataclass(frozen=True)
class Grade:
    label: str
    confidence: float
    probabilities: dict[str, float]
    input_tokens: int
    truncated: bool
    model: str = MODEL_ID

    def to_dict(self) -> dict:
        return asdict(self)


class PoliticalLeaningClassifier:
    """Load once on first use and reuse for multiple passages. CPU by default."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Install optional dependencies: "
                    "python -m pip install -r requirements-political-leaning.txt"
                ) from exc
            self._pipeline = pipeline(
                "text-classification",
                model=MODEL_ID,
                tokenizer=TOKENIZER_ID,
                device=self.device,
                framework="pt",
                trust_remote_code=False,
            )
        return self._pipeline

    def grade(self, text: str, *, truncate: bool = False) -> Grade:
        """Grade a passage; reject inputs over 512 tokens unless opted in."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        classifier = self._load()
        count = len(classifier.tokenizer(
            text, add_special_tokens=True, truncation=False
        )["input_ids"])
        truncated = count > MAX_TOKENS
        if truncated and not truncate:
            raise ValueError(
                f"Input has {count} tokens; maximum is {MAX_TOKENS} including "
                "special tokens. Shorten the passage or explicitly enable truncation."
            )
        scores = classifier(
            text, top_k=None, truncation=truncate, max_length=MAX_TOKENS
        )
        return self._result(scores, count, truncated)

    def grade_many(self, texts, *, truncate=False, batch_size=8):
        """Grade a batch in input order, reusing the same model."""
        texts = list(texts)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string.")
        if not texts:
            return []
        classifier = self._load()
        counts = [len(ids) for ids in classifier.tokenizer(
            texts, add_special_tokens=True, truncation=False)["input_ids"]]
        if not truncate and any(count > MAX_TOKENS for count in counts):
            raise ValueError("An input exceeds 512 tokens; shorten it or enable truncation.")
        outputs = classifier(texts, top_k=None, truncation=truncate,
                             max_length=MAX_TOKENS, batch_size=batch_size)
        if len(outputs) != len(texts):
            raise RuntimeError("Model returned the wrong number of results.")
        return [self._result(scores, count, count > MAX_TOKENS)
                for scores, count in zip(outputs, counts)]

    @staticmethod
    def _result(scores, count, truncated):
        probabilities = {}
        for item in scores:
            label = LABELS.get(item["label"])
            if label is None or label in probabilities:
                raise RuntimeError("Unexpected model label mapping.")
            score = float(item["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("Invalid model probability.")
            probabilities[label] = score
        if set(probabilities) != set(LABELS.values()) or not math.isclose(
            sum(probabilities.values()), 1.0, abs_tol=1e-5
        ):
            raise RuntimeError("Expected three normalized class probabilities.")
        label = max(probabilities, key=probabilities.get)
        return Grade(label, probabilities[label], probabilities, count, truncated)

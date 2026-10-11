"""Pluggable headline sentiment scorers: FinBERT (preferred) with VADER fallback."""

from __future__ import annotations

import logging
import os
from typing import List, Protocol

logger = logging.getLogger("trading_bot")

# Keep model downloads quiet in the bot log.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in ("httpx", "huggingface_hub", "transformers", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

FINBERT_MODEL_ID = "ProsusAI/finbert"


class SentimentScorer(Protocol):
    name: str

    def score(self, text: str) -> float: ...

    def score_many(self, texts: List[str]) -> List[float]: ...


class VaderScorer:
    """Lexicon-based scorer; fast, no downloads, weaker on financial phrasing."""

    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        compound = self._analyzer.polarity_scores(text)["compound"]
        return max(-1.0, min(1.0, float(compound)))

    def score_many(self, texts: List[str]) -> List[float]:
        return [self.score(t) for t in texts]


class FinBertScorer:
    """Transformer scorer fine-tuned on financial news. Score = P(positive) - P(negative)."""

    name = "finbert"

    def __init__(self, model_id: str = FINBERT_MODEL_ID, max_length: int = 128) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        for noisy in ("httpx", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()
        self._max_length = max_length
        labels = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._pos_idx = next(i for i, name in labels.items() if name.startswith("pos"))
        self._neg_idx = next(i for i, name in labels.items() if name.startswith("neg"))
        logger.info("[INFO] FinBERT loaded (%s)", model_id)

    def score_many(self, texts: List[str]) -> List[float]:
        clean = [(t or "").strip() for t in texts]
        if not any(clean):
            return [0.0 for _ in clean]
        with self._torch.no_grad():
            encoded = self._tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            logits = self._model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)
        scores = (probs[:, self._pos_idx] - probs[:, self._neg_idx]).tolist()
        return [0.0 if not t else max(-1.0, min(1.0, float(s))) for t, s in zip(clean, scores)]

    def score(self, text: str) -> float:
        return self.score_many([text])[0]


def build_scorer(name: str = "finbert") -> SentimentScorer:
    """Return the requested scorer, falling back to VADER if FinBERT cannot be loaded."""
    choice = (name or "finbert").strip().lower()
    if choice == "finbert":
        try:
            return FinBertScorer()
        except Exception as exc:
            logger.error("[ERROR] FinBERT unavailable (%s); falling back to VADER", exc)
    return VaderScorer()

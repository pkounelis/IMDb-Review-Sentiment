"""Run a pretrained HuggingFace sentiment model on review text."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import HF_BATCH_SIZE, MAX_LENGTH, SENTIMENT_MODEL_NAME

_tokenizer = None
_model = None
_device = None
_positive_label_id: int | None = None


def _positive_class_id(model) -> int:
    for label_id, label_name in model.config.id2label.items():
        if str(label_name).upper() == "POSITIVE":
            return int(label_id)
    return 1


def load_sentiment_model():
    global _tokenizer, _model, _device, _positive_label_id
    if _model is not None:
        return _tokenizer, _model, _device, _positive_label_id

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    _model.to(_device)
    _model.eval()
    _positive_label_id = _positive_class_id(_model)
    return _tokenizer, _model, _device, _positive_label_id


def predict_sentiment(
    texts: list[str],
    batch_size: int = HF_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer, model, device, positive_id = load_sentiment_model()

    all_preds: list[int] = []
    all_probs: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_probs.extend(probs[:, positive_id].tolist())

    return np.array(all_preds), np.array(all_probs)

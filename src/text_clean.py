"""Text cleaning for sklearn and HuggingFace sentiment models."""

from __future__ import annotations

import re

HTML_RE = re.compile(r"<[^>]+>")


def clean_for_sklearn(text: str) -> str:
    """Lowercase, strip HTML tags, normalize whitespace."""
    text = HTML_RE.sub(" ", text)
    text = text.lower()
    return " ".join(text.split())


def clean_for_sentiment(text: str) -> str:
    """Light clean for HF sentiment models — preserve casing, strip HTML only."""
    text = HTML_RE.sub(" ", text)
    return " ".join(text.split())

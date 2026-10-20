"""Minimal classifier interface used by future local model adapters."""

from __future__ import annotations

from typing import Protocol

from .schema import PredictionResult, validate_nonempty_text


class EmotionClassifier(Protocol):
    """A future classifier accepts text but returns metadata, never stored raw text."""

    def predict(
        self, text: str, *, safe_input_ref: str | None = None
    ) -> PredictionResult:
        """Return scores in the shared five-label schema."""
        ...


__all__ = ["EmotionClassifier", "validate_nonempty_text"]

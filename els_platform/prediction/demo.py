"""Fixed demonstration-only predictions; this module does not infer from text."""

from __future__ import annotations

from .interface import EmotionClassifier
from .schema import EmotionLabel, PredictionMode, PredictionResult, validate_nonempty_text


_DEMONSTRATION_SCORES = {
    EmotionLabel.BORED: 0.62,
    EmotionLabel.CONFIDENT: 0.24,
    EmotionLabel.CONFUSED: 0.84,
    EmotionLabel.CURIOUS: 0.58,
    EmotionLabel.FRUSTRATED: 0.33,
}
_DEMONSTRATION_THRESHOLDS = {label: 0.60 for label in EmotionLabel}


class DemonstrationClassifier:
    """Return the same static fixture for every non-empty input.

    The values are illustrative UI fixtures, not model outputs and not driven by text.
    """

    model_name = "Demonstration fixture"
    model_version = "0.1.0"

    def predict(
        self, text: str, *, safe_input_ref: str | None = None
    ) -> PredictionResult:
        validate_nonempty_text(text)
        return PredictionResult(
            scores=_DEMONSTRATION_SCORES,
            thresholds=_DEMONSTRATION_THRESHOLDS,
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_mode=PredictionMode.DEMONSTRATION,
            safe_input_ref=safe_input_ref,
        )

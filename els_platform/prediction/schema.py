"""Shared, privacy-conscious prediction types for emotion classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType
from typing import Mapping


class EmotionLabel(str, Enum):
    BORED = "Bored"
    CONFIDENT = "Confident"
    CONFUSED = "Confused"
    CURIOUS = "Curious"
    FRUSTRATED = "Frustrated"


EMOTION_LABELS: tuple[EmotionLabel, ...] = tuple(EmotionLabel)
_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class PredictionMode(str, Enum):
    DEMONSTRATION = "demonstration"
    MODEL_INFERENCE = "model_inference"


class PredictionOutcome(str, Enum):
    LABELS_SELECTED = "labels_selected"
    NO_CLEAR_CATEGORY = "no_clear_category"


def validate_nonempty_text(text: str) -> str:
    """Reject empty input while returning the caller's original text unchanged."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty study challenge before analysis.")
    return text


def _validated_scores(
    values: Mapping[EmotionLabel | str, Real], *, field_name: str
) -> Mapping[EmotionLabel, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping of all five labels to numbers.")

    normalized: dict[EmotionLabel, float] = {}
    for raw_label, raw_value in values.items():
        try:
            label = raw_label if isinstance(raw_label, EmotionLabel) else EmotionLabel(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown emotion label in {field_name}: {raw_label!r}.") from exc
        if label in normalized:
            raise ValueError(f"{field_name} contains a duplicate label: {label.value}.")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"{field_name}[{label.value!r}] must be a real number.")
        value = float(raw_value)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name}[{label.value!r}] must be finite and between 0 and 1.")
        normalized[label] = value

    missing = set(EMOTION_LABELS) - set(normalized)
    extra = set(normalized) - set(EMOTION_LABELS)
    if missing or extra:
        missing_names = ", ".join(label.value for label in EMOTION_LABELS if label in missing)
        raise ValueError(f"{field_name} must include every target label; missing: {missing_names}.")

    return MappingProxyType({label: normalized[label] for label in EMOTION_LABELS})


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Prediction metadata without retaining the submitted free text.

    ``safe_input_ref`` is optional and must be an opaque identifier, never the input text.
    Scores are displayable as confidence-style values only with their documented mode;
    demonstration scores are illustrative and are not validated probabilities.
    """

    scores: Mapping[EmotionLabel | str, Real]
    thresholds: Mapping[EmotionLabel | str, Real]
    model_name: str
    model_version: str
    prediction_mode: PredictionMode
    safe_input_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", _validated_scores(self.scores, field_name="scores"))
        object.__setattr__(
            self, "thresholds", _validated_scores(self.thresholds, field_name="thresholds")
        )
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string.")
        if not isinstance(self.prediction_mode, PredictionMode):
            try:
                mode = PredictionMode(self.prediction_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("prediction_mode must identify demonstration or model inference.") from exc
            object.__setattr__(self, "prediction_mode", mode)
        if self.safe_input_ref is not None and (
            not isinstance(self.safe_input_ref, str)
            or _SAFE_REFERENCE.fullmatch(self.safe_input_ref) is None
        ):
            raise ValueError("safe_input_ref must be an opaque identifier without spaces or text content.")

    @property
    def selected_labels(self) -> tuple[EmotionLabel, ...]:
        return tuple(
            label for label in EMOTION_LABELS if self.scores[label] >= self.thresholds[label]
        )

    @property
    def outcome(self) -> PredictionOutcome:
        return (
            PredictionOutcome.LABELS_SELECTED
            if self.selected_labels
            else PredictionOutcome.NO_CLEAR_CATEGORY
        )

    @property
    def is_demonstration(self) -> bool:
        return self.prediction_mode is PredictionMode.DEMONSTRATION

    def score_rows(self) -> list[dict[str, str | float | bool]]:
        """Return display-safe rows, keeping the keyword/rule layer separate."""
        selected = set(self.selected_labels)
        return [
            {
                "Emotion": label.value,
                "Illustrative score" if self.is_demonstration else "Model score": self.scores[label],
                "Selection threshold": self.thresholds[label],
                "Selected": label in selected,
            }
            for label in EMOTION_LABELS
        ]

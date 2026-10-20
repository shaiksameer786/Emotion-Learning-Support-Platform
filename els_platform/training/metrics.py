"""Privacy-safe multi-label metrics and validation-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ..prediction.schema import EMOTION_LABELS, EmotionLabel


class EvaluationError(ValueError):
    """Raised for malformed evaluation data or an invalid evaluation sequence."""


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    thresholds: tuple[float | None, ...]
    validation_f1: tuple[float | None, ...]
    status: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_split": "validation",
            "thresholds": {
                label.value: self.thresholds[index]
                for index, label in enumerate(EMOTION_LABELS)
            },
            "validation_f1": {
                label.value: self.validation_f1[index]
                for index, label in enumerate(EMOTION_LABELS)
            },
            "status": {label.value: self.status[index] for index, label in enumerate(EMOTION_LABELS)},
        }


@dataclass(frozen=True, slots=True)
class FinalizedEvaluationPlan:
    thresholds: tuple[float, ...]
    configuration_digest: str
    threshold_source: str = "validation"

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(EMOTION_LABELS):
            raise EvaluationError("Finalized thresholds must match the five-label order.")
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in self.thresholds):
            raise EvaluationError("Thresholds must be finite values in (0, 1].")
        if not self.configuration_digest:
            raise EvaluationError("A configuration digest is required to finalize evaluation.")
        if self.threshold_source != "validation":
            raise EvaluationError("Thresholds must be selected on validation data only.")

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold_source": self.threshold_source,
            "configuration_digest": self.configuration_digest,
            "thresholds": {
                label.value: self.thresholds[index]
                for index, label in enumerate(EMOTION_LABELS)
            },
        }


def select_thresholds(
    targets: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    known_mask: Sequence[Sequence[int]],
    *,
    split_name: str,
    candidates: Sequence[float] = tuple(value / 100 for value in range(5, 100, 5)),
) -> ThresholdSelection:
    """Select per-label F1 thresholds from validation predictions only.

    A label needs at least one known positive and known negative in validation. When
    that is not true its threshold remains undefined; no test-set fallback is used.
    """
    if split_name != "validation":
        raise EvaluationError("Threshold selection is allowed on the validation split only.")
    values, probabilities, mask = _validate_arrays(targets, scores, known_mask)
    if not candidates:
        raise EvaluationError("Provide at least one candidate threshold.")
    threshold_candidates = tuple(float(value) for value in candidates)
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in threshold_candidates):
        raise EvaluationError("Candidate thresholds must be finite values in (0, 1].")

    thresholds: list[float | None] = []
    best_f1_values: list[float | None] = []
    statuses: list[str] = []
    for label_index, label in enumerate(EMOTION_LABELS):
        observed = [i for i, row in enumerate(mask) if row[label_index] == 1]
        positives = sum(values[i][label_index] for i in observed)
        negatives = len(observed) - positives
        if not observed or positives == 0 or negatives == 0:
            thresholds.append(None)
            best_f1_values.append(None)
            statuses.append("insufficient_validation_class_coverage")
            continue
        results: list[tuple[float, float]] = []
        for threshold in threshold_candidates:
            tp = fp = fn = 0
            for row_index in observed:
                predicted = probabilities[row_index][label_index] >= threshold
                actual = values[row_index][label_index] == 1
                tp += int(predicted and actual)
                fp += int(predicted and not actual)
                fn += int(not predicted and actual)
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator else 0.0
            results.append((f1, threshold))
        best_f1, best_threshold = max(results, key=lambda pair: (pair[0], -abs(pair[1] - 0.5), -pair[1]))
        thresholds.append(best_threshold)
        best_f1_values.append(best_f1)
        statuses.append("selected_on_validation")
    return ThresholdSelection(tuple(thresholds), tuple(best_f1_values), tuple(statuses))


def finalize_evaluation_plan(
    selection: ThresholdSelection,
    *,
    configuration_digest: str,
) -> FinalizedEvaluationPlan:
    if not isinstance(selection, ThresholdSelection):
        raise TypeError("selection must be a ThresholdSelection from validation data.")
    if any(value is None for value in selection.thresholds):
        missing = [
            label.value for index, label in enumerate(EMOTION_LABELS)
            if selection.thresholds[index] is None
        ]
        raise EvaluationError(
            "Cannot finalize held-out evaluation; validation has no usable threshold for: "
            + ", ".join(missing)
        )
    return FinalizedEvaluationPlan(
        thresholds=tuple(float(value) for value in selection.thresholds if value is not None),
        configuration_digest=configuration_digest,
    )


def calculate_metrics(
    targets: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    known_mask: Sequence[Sequence[int]],
    thresholds: Mapping[EmotionLabel | str, float] | Sequence[float],
) -> dict[str, object]:
    """Calculate masked multi-label metrics and per-label binary confusion matrices.

    Matrix rows are actual negative/positive and columns predicted negative/positive.
    Unknown label decisions are excluded. F1 is zero for false-positive-only labels;
    it is undefined only when there are no actual or predicted positives.
    """
    values, probabilities, mask = _validate_arrays(targets, scores, known_mask)
    threshold_values = _validate_thresholds(thresholds)
    label_reports: dict[str, object] = {}
    macro_f1_values: list[float] = []
    total_tp = total_fp = total_fn = 0
    exact_rows = exact_matches = 0
    hamming_errors = known_bits = 0
    for label_index, label in enumerate(EMOTION_LABELS):
        observed = [row for row in range(len(values)) if mask[row][label_index] == 1]
        tn = tp = fp = fn = 0
        for row in observed:
            predicted = probabilities[row][label_index] >= threshold_values[label_index]
            actual = values[row][label_index] == 1
            tp += int(predicted and actual)
            fp += int(predicted and not actual)
            fn += int(not predicted and actual)
            tn += int(not predicted and not actual)
            hamming_errors += int(predicted != actual)
            known_bits += 1
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / support if support else None
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        if f1 is not None:
            macro_f1_values.append(f1)
        label_reports[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "actual_labels": ["negative", "positive"],
                "predicted_labels": ["negative", "positive"],
                "counts": [[tn, fp], [fn, tp]],
            },
            "support": support,
            "known_count": len(observed),
            "undefined": "no known positive examples" if support == 0 else None,
            "precision_undefined": "no predicted positives" if tp + fp == 0 else None,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    for row in range(len(values)):
        if all(mask[row]):
            exact_rows += 1
            if all(
                (probabilities[row][index] >= threshold_values[index]) == (values[row][index] == 1)
                for index in range(len(EMOTION_LABELS))
            ):
                exact_matches += 1
    micro_denominator = 2 * total_tp + total_fp + total_fn
    return {
        "label_order": [label.value for label in EMOTION_LABELS],
        "per_label": label_reports,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else None,
        "macro_f1_defined_labels": len(macro_f1_values),
        "micro_f1": 2 * total_tp / micro_denominator if micro_denominator else None,
        "subset_accuracy": exact_matches / exact_rows if exact_rows else None,
        "subset_accuracy_rows": exact_rows,
        "hamming_loss": hamming_errors / known_bits if known_bits else None,
        "known_label_decisions": known_bits,
        "thresholds": {label.value: threshold_values[i] for i, label in enumerate(EMOTION_LABELS)},
    }


def evaluate_heldout_test(
    targets: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    known_mask: Sequence[Sequence[int]],
    *,
    plan: FinalizedEvaluationPlan,
    configuration_digest: str,
    split_name: str,
) -> dict[str, object]:
    """Score test data only after validation thresholds and the model config are locked."""
    if split_name != "test":
        raise EvaluationError("Held-out evaluation requires the test split.")
    if configuration_digest != plan.configuration_digest:
        raise EvaluationError("Configuration changed after thresholds were finalized; retune on validation.")
    report = calculate_metrics(targets, scores, known_mask, plan.thresholds)
    report["evaluation_split"] = "test"
    report["threshold_source"] = "validation"
    report["configuration_digest"] = configuration_digest
    return report


def _validate_arrays(
    targets: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    known_mask: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[float]], list[list[int]]]:
    if not (len(targets) == len(scores) == len(known_mask)) or not targets:
        raise EvaluationError("Targets, scores, and masks must have the same nonzero row count.")
    width = len(EMOTION_LABELS)
    converted_targets: list[list[int]] = []
    converted_scores: list[list[float]] = []
    converted_mask: list[list[int]] = []
    for y_row, s_row, m_row in zip(targets, scores, known_mask):
        if not (len(y_row) == len(s_row) == len(m_row) == width):
            raise EvaluationError("Every row must have five values in canonical label order.")
        target_row: list[int] = []
        score_row: list[float] = []
        mask_row: list[int] = []
        for y, score, known in zip(y_row, s_row, m_row):
            if isinstance(y, bool) or y not in (0, 1):
                raise EvaluationError("Targets must contain only 0 or 1.")
            if isinstance(known, bool) or known not in (0, 1):
                raise EvaluationError("Known masks must contain only 0 or 1.")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise EvaluationError("Scores must be numeric probabilities.")
            score_float = float(score)
            if not math.isfinite(score_float) or not 0 <= score_float <= 1:
                raise EvaluationError("Scores must be finite values in [0, 1].")
            if y == 1 and known != 1:
                raise EvaluationError("A positive target cannot be masked as unknown.")
            target_row.append(int(y))
            score_row.append(score_float)
            mask_row.append(int(known))
        converted_targets.append(target_row)
        converted_scores.append(score_row)
        converted_mask.append(mask_row)
    return converted_targets, converted_scores, converted_mask


def _validate_thresholds(
    thresholds: Mapping[EmotionLabel | str, float] | Sequence[float],
) -> tuple[float, ...]:
    if isinstance(thresholds, Mapping):
        normalized: dict[EmotionLabel, float] = {}
        for raw_label, raw_value in thresholds.items():
            try:
                label = raw_label if isinstance(raw_label, EmotionLabel) else EmotionLabel(raw_label)
            except (TypeError, ValueError) as exc:
                raise EvaluationError("Thresholds contain an unknown label.") from exc
            if label in normalized:
                raise EvaluationError("Thresholds contain a duplicate label.")
            normalized[label] = raw_value
        if set(normalized) != set(EMOTION_LABELS):
            raise EvaluationError("Thresholds must define all five target labels.")
        values = tuple(normalized[label] for label in EMOTION_LABELS)
    else:
        values = tuple(thresholds)
    if len(values) != len(EMOTION_LABELS):
        raise EvaluationError("Thresholds must contain five values in canonical label order.")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise EvaluationError("Thresholds must be numeric values.")
        number = float(value)
        if not math.isfinite(number) or not 0.0 < number <= 1.0:
            raise EvaluationError("Thresholds must be finite values in (0, 1].")
        converted.append(number)
    return tuple(converted)

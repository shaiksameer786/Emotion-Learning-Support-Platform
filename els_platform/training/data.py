"""Training authorization and safe tensor-target preparation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ..prediction.schema import EMOTION_LABELS, EmotionLabel
from ..preprocessing import PreparedExample, normalize_text
from .splitting import DatasetSplits


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SAFE_MANIFEST_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
ALLOWED_SOURCE_KINDS = frozenset({"reviewed_public", "consent_approved_supplement"})


class TrainingDataError(ValueError):
    """Raised when splits or their source are not safe to use for training."""


@dataclass(frozen=True, slots=True)
class TrainingAuthorization:
    """Explicit human-review assertions required before a caller may train.

    The reference and dataset ID must be opaque, non-identifying codes. This object
    records assertions; it does not replace institutional, legal, or ethics review.
    """

    approved_for_training: bool
    source_kind: str
    dataset_id: str
    approval_reference: str
    license_or_consent_reviewed: bool
    label_mapping_reviewed: bool
    data_protection_reviewed: bool

    def validate(self) -> None:
        if self.approved_for_training is not True:
            raise TrainingDataError("Training refused: the dataset is not approved for training.")
        if not isinstance(self.source_kind, str) or self.source_kind not in ALLOWED_SOURCE_KINDS:
            raise TrainingDataError(
                "Training refused: demonstration and test-only fixtures are not approved data sources."
            )
        for label, value in (
            ("dataset_id", self.dataset_id),
            ("approval_reference", self.approval_reference),
        ):
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise TrainingDataError(f"{label} must be an opaque, non-identifying reference.")
        if self.license_or_consent_reviewed is not True:
            raise TrainingDataError("Training refused: license or consent review is incomplete.")
        if self.label_mapping_reviewed is not True:
            raise TrainingDataError("Training refused: the label mapping has not been reviewed.")
        if self.data_protection_reviewed is not True:
            raise TrainingDataError("Training refused: data-protection review is incomplete.")


@dataclass(frozen=True, slots=True)
class EncodedTargets:
    """Five-dimensional targets plus a mask for labels known on each example."""

    values: tuple[int, ...]
    known_mask: tuple[int, ...]


def encode_multihot(example: PreparedExample) -> EncodedTargets:
    """Encode canonical positive labels and known-label mask in shared label order."""
    if not isinstance(example, PreparedExample):
        raise TypeError("example must be a PreparedExample from the dataset pipeline.")
    positives = _validate_label_sequence(example.labels, "labels")
    known = _validate_label_sequence(example.known_labels, "known_labels")
    if not positives.issubset(known):
        raise TrainingDataError("Every positive label must also be marked known.")
    if not known:
        raise TrainingDataError("At least one target label must be known for each example.")
    return EncodedTargets(
        values=tuple(int(label in positives) for label in EMOTION_LABELS),
        known_mask=tuple(int(label in known) for label in EMOTION_LABELS),
    )


def validate_training_splits(
    splits: DatasetSplits,
    authorization: TrainingAuthorization,
) -> dict[str, object]:
    """Check approvals, useful label coverage, non-empty splits, and leakage.

    Counts only are returned. All five labels need a known positive in the training
    partition to fit the requested five-output classifier. This is a structural check,
    not a universal sample-size or data-sufficiency guarantee.
    """
    if not isinstance(splits, DatasetSplits):
        raise TrainingDataError("Training requires DatasetSplits from the reviewed split pipeline.")
    if not isinstance(authorization, TrainingAuthorization):
        raise TrainingDataError("An explicit TrainingAuthorization is required.")
    authorization.validate()
    groups = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    for name, examples in groups.items():
        if not examples:
            raise TrainingDataError(f"The {name} split is empty.")
        for example in examples:
            encode_multihot(example)

    report = splits.report
    reported_rows = getattr(report, "rows_per_split", None)
    if not isinstance(reported_rows, dict) or reported_rows != {
        name: len(examples) for name, examples in groups.items()
    }:
        raise TrainingDataError("Split report row counts do not match the actual partitions.")

    manifest_metadata = {
        "dataset_reference": splits.dataset_reference,
        "dataset_id": splits.dataset_id,
        "source_kind": splits.source_kind,
        "approval_reference": splits.approval_reference,
        "split_manifest_fingerprint": splits.split_manifest_fingerprint,
    }
    if any(value is not None for value in manifest_metadata.values()):
        if any(not isinstance(value, str) for value in manifest_metadata.values()):
            raise TrainingDataError("Persisted split metadata is incomplete.")
        if not _SAFE_ID.fullmatch(splits.dataset_reference or ""):
            raise TrainingDataError("Dataset source reference must be an opaque non-identifying code.")
        if not _SAFE_MANIFEST_FINGERPRINT.fullmatch(splits.split_manifest_fingerprint or ""):
            raise TrainingDataError("Split manifest fingerprint must be a SHA-256 hex digest.")
        if (
            splits.source_kind != authorization.source_kind
            or splits.dataset_id != authorization.dataset_id
            or splits.approval_reference != authorization.approval_reference
        ):
            raise TrainingDataError("Split manifest approval reference does not match the supplied authorization.")

    positive_counts = {label: sum(label in ex.labels for ex in splits.train) for label in EMOTION_LABELS}
    known_counts = {label: sum(label in ex.known_labels for ex in splits.train) for label in EMOTION_LABELS}
    train_negative_counts = {
        label: sum(label in ex.known_labels and label not in ex.labels for ex in splits.train)
        for label in EMOTION_LABELS
    }
    validation_positive_counts = {
        label: sum(label in ex.labels and label in ex.known_labels for ex in splits.validation)
        for label in EMOTION_LABELS
    }
    validation_negative_counts = {
        label: sum(label in ex.known_labels and label not in ex.labels for ex in splits.validation)
        for label in EMOTION_LABELS
    }
    unusable_train = [
        label.value for label in EMOTION_LABELS
        if positive_counts[label] == 0 or train_negative_counts[label] == 0
    ]
    if unusable_train:
        raise TrainingDataError(
            "Training split needs at least one known positive and known negative for: "
            + ", ".join(unusable_train)
        )
    unusable_validation = [
        label.value for label in EMOTION_LABELS
        if validation_positive_counts[label] == 0 or validation_negative_counts[label] == 0
    ]
    if unusable_validation:
        raise TrainingDataError(
            "Validation split needs known positive and negative examples to select thresholds for: "
            + ", ".join(unusable_validation)
        )

    _validate_partition_disjointness(groups)
    return {
        "rows_per_split": {name: len(examples) for name, examples in groups.items()},
        "train_positive_counts": {label.value: positive_counts[label] for label in EMOTION_LABELS},
        "train_known_counts": {label.value: known_counts[label] for label in EMOTION_LABELS},
        "train_negative_counts": {label.value: train_negative_counts[label] for label in EMOTION_LABELS},
        "split_seed": getattr(report, "seed", None),
        "grouping_strategy": getattr(report, "grouping_strategy", "unknown"),
        "source_kind": authorization.source_kind,
        "dataset_id": authorization.dataset_id,
        "approval_reference": authorization.approval_reference,
        "dataset_reference": splits.dataset_reference,
        "split_manifest_fingerprint": splits.split_manifest_fingerprint,
        "privacy": "Counts and opaque review references only; no text or group IDs.",
    }


def _validate_label_sequence(values: Sequence[EmotionLabel], field_name: str) -> set[EmotionLabel]:
    if not isinstance(values, (tuple, list)):
        raise TrainingDataError(f"{field_name} must be a sequence of canonical EmotionLabel values.")
    result: set[EmotionLabel] = set()
    for value in values:
        if not isinstance(value, EmotionLabel):
            raise TrainingDataError(f"{field_name} contains an unknown or unsupported label.")
        if value in result:
            raise TrainingDataError(f"{field_name} contains duplicate labels.")
        result.add(value)
    return result


def _validate_partition_disjointness(groups: dict[str, Sequence[PreparedExample]]) -> None:
    seen_text: dict[str, str] = {}
    seen_group: dict[str, str] = {}
    for split_name, examples in groups.items():
        for example in examples:
            text_key = normalize_text(example.text)
            prior_text_split = seen_text.get(text_key)
            if prior_text_split is not None and prior_text_split != split_name:
                raise TrainingDataError("Exact normalized text occurs across dataset partitions.")
            seen_text[text_key] = split_name
            if example.group_id is not None:
                prior_group_split = seen_group.get(example.group_id)
                if prior_group_split is not None and prior_group_split != split_name:
                    raise TrainingDataError("A supplied group_id occurs across dataset partitions.")
                seen_group[example.group_id] = split_name

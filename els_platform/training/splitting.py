"""Deterministic group-aware train/validation/test splitting without model dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Mapping

from ..prediction.schema import EMOTION_LABELS, EmotionLabel
from ..preprocessing import PreparedDataset, PreparedExample


DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
SPLIT_NAMES = ("train", "validation", "test")


class DatasetSplitError(ValueError):
    """Raised when a validated dataset cannot be split safely."""


@dataclass(frozen=True, slots=True)
class SplitReport:
    seed: int
    requested_ratios: Mapping[str, float]
    rows_per_split: Mapping[str, int]
    groups_per_split: Mapping[str, int]
    positive_counts: Mapping[str, Mapping[EmotionLabel, int]]
    known_counts: Mapping[str, Mapping[EmotionLabel, int]]
    grouping_strategy: str
    group_count: int

    def to_dict(self) -> dict[str, object]:
        """Return split diagnostics without text or group identifiers."""
        return {
            "report_version": 1,
            "seed": self.seed,
            "requested_ratios": dict(self.requested_ratios),
            "rows_per_split": dict(self.rows_per_split),
            "groups_per_split": dict(self.groups_per_split),
            "positive_label_counts": {
                split: {label.value: counts.get(label, 0) for label in EMOTION_LABELS}
                for split, counts in self.positive_counts.items()
            },
            "known_label_counts": {
                split: {label.value: counts.get(label, 0) for label in EMOTION_LABELS}
                for split, counts in self.known_counts.items()
            },
            "grouping_strategy": self.grouping_strategy,
            "group_count": self.group_count,
            "privacy": "Report excludes input text and group identifiers.",
        }


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: tuple[PreparedExample, ...]
    validation: tuple[PreparedExample, ...]
    test: tuple[PreparedExample, ...]
    report: SplitReport
    # Bound only in memory to the exact partitions used during fitting. This digest
    # is deliberately excluded from repr, equality, reports, and checkpoints.
    split_fingerprint: str = field(init=False, repr=False, compare=False)
    dataset_reference: str | None = field(default=None, repr=False, compare=False)
    dataset_id: str | None = field(default=None, repr=False, compare=False)
    source_kind: str | None = field(default=None, repr=False, compare=False)
    approval_reference: str | None = field(default=None, repr=False, compare=False)
    split_manifest_fingerprint: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_fingerprint", _fingerprint_splits(self))


def split_dataset(
    dataset: PreparedDataset,
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
) -> DatasetSplits:
    """Split by supplied group IDs, or by normalized-text hash when no IDs exist.

    Without the optional ``group_id`` input column this only prevents exact normalized
    text leakage. It cannot provide participant-level separation.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetSplitError("seed must be an integer.")
    if not dataset.report.is_valid:
        raise DatasetSplitError("Resolve dataset validation issues before creating splits.")
    split_ratios = _validate_ratios(ratios)
    has_group_ids = dataset.report.group_id_column_present

    grouped: dict[str, list[tuple[int, PreparedExample]]] = defaultdict(list)
    for index, example in enumerate(dataset.examples):
        group_key = (
            "provided:" + example.group_id
            if has_group_ids and example.group_id is not None
            else "text:" + hashlib.sha256(example.text.encode("utf-8")).hexdigest()
        )
        grouped[group_key].append((index, example))
    if len(grouped) < len(SPLIT_NAMES):
        raise DatasetSplitError("At least three independent split groups are required.")

    total_rows = len(dataset.examples)
    global_positive = Counter(
        label for example in dataset.examples for label in example.labels
    )
    global_known = Counter(
        label for example in dataset.examples for label in example.known_labels
    )
    target_rows = {name: total_rows * split_ratios[name] for name in SPLIT_NAMES}
    target_positive = {
        name: {label: global_positive[label] * split_ratios[name] for label in EMOTION_LABELS}
        for name in SPLIT_NAMES
    }
    target_known = {
        name: {label: global_known[label] * split_ratios[name] for label in EMOTION_LABELS}
        for name in SPLIT_NAMES
    }

    # Put rare positive labels and larger groups earlier; hash ties for stable results
    # that do not depend on the order in which source rows happen to be read.
    def group_priority(item: tuple[str, list[tuple[int, PreparedExample]]]) -> tuple[float, int, str]:
        key, members = item
        positives = {
            label for _, example in members for label in example.labels
        }
        rarity = sum(1.0 / max(global_positive[label], 1) for label in positives)
        stable = _stable_key(seed, key)
        return (-rarity, -len(members), stable)

    ordered_groups = sorted(grouped.items(), key=group_priority)
    assignment: dict[str, str] = {}
    row_counts = {name: 0 for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    positive_counts = {name: Counter() for name in SPLIT_NAMES}
    known_counts = {name: Counter() for name in SPLIT_NAMES}

    # Seed one independent group into each partition so none is accidentally empty.
    first_split_order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-split_ratios[name], _stable_key(seed, "initial:" + name)),
    )
    for (group_key, members), split_name in zip(ordered_groups, first_split_order):
        assignment[group_key] = split_name
        _add_group(members, split_name, row_counts, group_counts, positive_counts, known_counts)

    for group_key, members in ordered_groups[len(first_split_order) :]:
        best_split = min(
            SPLIT_NAMES,
            key=lambda name: (
                _objective(
                    name,
                    members,
                    row_counts,
                    positive_counts,
                    known_counts,
                    target_rows,
                    target_positive,
                    target_known,
                ),
                _stable_key(seed, group_key + ":" + name),
            ),
        )
        assignment[group_key] = best_split
        _add_group(members, best_split, row_counts, group_counts, positive_counts, known_counts)

    split_members: dict[str, list[tuple[int, PreparedExample]]] = {
        name: [] for name in SPLIT_NAMES
    }
    for group_key, members in grouped.items():
        split_members[assignment[group_key]].extend(members)
    for members in split_members.values():
        members.sort(key=lambda member: member[0])

    strategy = (
        "provided group_id values (group disjoint; participant/source meaning is not verified)"
        if has_group_ids
        else "normalized text hash only (participant-level grouping unavailable)"
    )
    report = SplitReport(
        seed=seed,
        requested_ratios=split_ratios,
        rows_per_split=dict(row_counts),
        groups_per_split=dict(group_counts),
        positive_counts={name: dict(counts) for name, counts in positive_counts.items()},
        known_counts={name: dict(counts) for name, counts in known_counts.items()},
        grouping_strategy=strategy,
        group_count=len(grouped),
    )
    return DatasetSplits(
        train=tuple(example for _, example in split_members["train"]),
        validation=tuple(example for _, example in split_members["validation"]),
        test=tuple(example for _, example in split_members["test"]),
        report=report,
    )


def _validate_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    if set(values) != set(SPLIT_NAMES):
        raise DatasetSplitError("ratios must define train, validation, and test exactly once.")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DatasetSplitError("split ratios must be numeric values.")
        if not math.isfinite(float(value)) or value <= 0:
            raise DatasetSplitError("split ratios must be finite and greater than zero.")
        values[name] = float(value)
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetSplitError("split ratios must sum to 1.0.")
    return values


def _objective(
    candidate: str,
    members: list[tuple[int, PreparedExample]],
    row_counts: Mapping[str, int],
    positive_counts: Mapping[str, Counter[EmotionLabel]],
    known_counts: Mapping[str, Counter[EmotionLabel]],
    target_rows: Mapping[str, float],
    target_positive: Mapping[str, Mapping[EmotionLabel, float]],
    target_known: Mapping[str, Mapping[EmotionLabel, float]],
) -> float:
    rows_after = dict(row_counts)
    rows_after[candidate] += len(members)
    positives_after = {name: Counter(positive_counts[name]) for name in SPLIT_NAMES}
    known_after = {name: Counter(known_counts[name]) for name in SPLIT_NAMES}
    for _, example in members:
        positives_after[candidate].update(example.labels)
        known_after[candidate].update(example.known_labels)

    cost = 0.0
    for name in SPLIT_NAMES:
        row_target = max(target_rows[name], 1.0)
        cost += 0.6 * (rows_after[name] - target_rows[name]) ** 2 / row_target
        for label in EMOTION_LABELS:
            pos_target = max(target_positive[name][label], 1.0)
            known_target = max(target_known[name][label], 1.0)
            cost += (positives_after[name][label] - target_positive[name][label]) ** 2 / pos_target
            cost += 0.25 * (
                (known_after[name][label] - target_known[name][label]) ** 2 / known_target
            )
    return cost


def _add_group(
    members: list[tuple[int, PreparedExample]],
    split_name: str,
    row_counts: dict[str, int],
    group_counts: dict[str, int],
    positive_counts: dict[str, Counter[EmotionLabel]],
    known_counts: dict[str, Counter[EmotionLabel]],
) -> None:
    row_counts[split_name] += len(members)
    group_counts[split_name] += 1
    for _, example in members:
        positive_counts[split_name].update(example.labels)
        known_counts[split_name].update(example.known_labels)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _fingerprint_splits(splits: DatasetSplits) -> str:
    """Return an in-memory binding for exact split membership and annotations.

    The digest is never written to a file or report. It lets the high-level test
    evaluator reject a different split object after a run has been fitted.
    """
    digest = hashlib.sha256()
    for split_name in SPLIT_NAMES:
        digest.update(split_name.encode("ascii") + b"\0")
        for example in getattr(splits, split_name):
            row = {
                "text": getattr(example, "text", None),
                "labels": [getattr(label, "value", repr(label)) for label in getattr(example, "labels", ())],
                "known_labels": [
                    getattr(label, "value", repr(label))
                    for label in getattr(example, "known_labels", ())
                ],
                "group_id": getattr(example, "group_id", None),
            }
            encoded = json.dumps(row, ensure_ascii=True, sort_keys=True, default=repr).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()

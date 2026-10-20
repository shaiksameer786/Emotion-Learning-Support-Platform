"""Privacy-conscious validation and conservative normalization for local datasets.

Dataset rows are retained in memory for a future approved training run. Reports contain
counts and validation codes only; they never contain text or group identifiers.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .prediction.schema import EMOTION_LABELS, EmotionLabel


REQUIRED_COLUMNS = frozenset({"text", "labels"})
OPTIONAL_COLUMNS = frozenset({"known_labels", "group_id"})
ALLOWED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
_GROUP_ID = re.compile(r"[^\s\x00-\x1f\x7f]{1,128}\Z")


class DatasetContractError(ValueError):
    """Raised when the file or mapping configuration violates the input contract."""


@dataclass(frozen=True, slots=True)
class PreparedExample:
    """A normalized row for downstream training; text is deliberately hidden from repr."""

    text: str = field(repr=False)
    labels: tuple[EmotionLabel, ...]
    known_labels: tuple[EmotionLabel, ...]
    group_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A row-level issue with no text or group identifier."""

    row_number: int
    code: str


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Counts and diagnostics safe to write as a local JSON audit report."""

    input_rows: int
    accepted_records: int
    invalid_rows: int
    missing_text_rows: int
    invalid_text_rows: int
    missing_label_rows: int
    missing_known_label_rows: int
    duplicate_rows_dropped: int
    conflicting_duplicate_groups: int
    unsupported_label_occurrences: int
    distinct_unsupported_label_values: int
    positive_label_counts: Mapping[EmotionLabel, int]
    known_label_counts: Mapping[EmotionLabel, int]
    group_id_column_present: bool
    provided_group_count: int | None
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return self.input_rows > 0 and self.invalid_rows == 0 and self.accepted_records > 0

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit data without text, identifiers, or source row values."""
        return {
            "report_version": 1,
            "input_rows": self.input_rows,
            "accepted_records": self.accepted_records,
            "invalid_rows": self.invalid_rows,
            "missing_text_rows": self.missing_text_rows,
            "invalid_text_rows": self.invalid_text_rows,
            "missing_label_rows": self.missing_label_rows,
            "missing_known_label_rows": self.missing_known_label_rows,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "conflicting_duplicate_groups": self.conflicting_duplicate_groups,
            "unsupported_label_occurrences": self.unsupported_label_occurrences,
            "distinct_unsupported_label_values": self.distinct_unsupported_label_values,
            "label_distribution": {
                label.value: {
                    "positive_count": self.positive_label_counts.get(label, 0),
                    "known_count": self.known_label_counts.get(label, 0),
                }
                for label in EMOTION_LABELS
            },
            "group_id_column_present": self.group_id_column_present,
            "provided_group_count": self.provided_group_count,
            "issues": [
                {"row_number": issue.row_number, "code": issue.code}
                for issue in self.issues
            ],
            "privacy": "Report excludes input text and group identifiers.",
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Accepted examples plus an audit report; splits must reject invalid reports."""

    examples: tuple[PreparedExample, ...]
    report: DatasetValidationReport


def normalize_text(text: str) -> str:
    """Normalize Unicode to NFC and collapse whitespace without rewriting language.

    Case, punctuation, accents, emoji, and negation are preserved. Control characters
    other than ordinary whitespace are rejected instead of silently stripped.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized = unicodedata.normalize("NFC", text)
    for character in normalized:
        if unicodedata.category(character) == "Cc" and not character.isspace():
            raise ValueError("text contains a disallowed control character")
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("text is empty after normalization")
    return normalized


def prepare_records(
    records: Iterable[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
    label_mapping: Mapping[str, str | EmotionLabel] | None = None,
    assume_all_labels_known: bool = False,
) -> PreparedDataset:
    """Validate and normalize rows while retaining no text in the report.

    ``labels`` and ``known_labels`` are JSON arrays (or arrays in JSONL/API records).
    A positive label is not inferred for any absent category. If ``known_labels`` is
    omitted, only the listed positive labels are known unless the caller explicitly
    opts into a complete five-label annotation with ``assume_all_labels_known=True``.
    """
    source_records = [dict(record) for record in records]
    if columns is None:
        if not source_records:
            raise DatasetContractError("Cannot validate an empty input without column names.")
        declared_columns = tuple(source_records[0].keys())
    else:
        declared_columns = tuple(columns)

    if len(set(declared_columns)) != len(declared_columns):
        raise DatasetContractError("Input column names must be unique.")
    if any(not isinstance(column, str) or not column for column in declared_columns):
        raise DatasetContractError("Input column names must be non-empty strings.")
    missing_columns = REQUIRED_COLUMNS - set(declared_columns)
    extra_columns = set(declared_columns) - ALLOWED_COLUMNS
    if missing_columns or extra_columns:
        parts: list[str] = []
        if missing_columns:
            parts.append("missing required columns: " + ", ".join(sorted(missing_columns)))
        if extra_columns:
            parts.append("unsupported columns are present")
        raise DatasetContractError("Invalid dataset schema (" + "; ".join(parts) + ").")
    if assume_all_labels_known and "known_labels" in declared_columns:
        raise DatasetContractError(
            "Use either a known_labels column or assume_all_labels_known, not both."
        )

    mapping = _validated_label_mapping(label_mapping)
    issues: list[ValidationIssue] = []
    candidates: list[tuple[int, PreparedExample]] = []
    unsupported_values: set[str] = set()
    unsupported_occurrences = 0
    missing_text_rows = 0
    invalid_text_rows = 0
    missing_label_rows = 0
    missing_known_label_rows = 0
    invalid_row_numbers: set[int] = set()
    has_known_column = "known_labels" in declared_columns
    has_group_column = "group_id" in declared_columns

    for row_number, row in enumerate(source_records, start=1):
        row_invalid = False
        if set(row) - set(declared_columns):
            issues.append(ValidationIssue(row_number, "unexpected_row_columns"))
            invalid_row_numbers.add(row_number)
            continue

        if _is_missing(row.get("text")):
            missing_text_rows += 1
            issues.append(ValidationIssue(row_number, "missing_text"))
            invalid_row_numbers.add(row_number)
            row_invalid = True
            text = ""
        else:
            try:
                text = normalize_text(row["text"])  # type: ignore[arg-type]
            except (ValueError, TypeError, KeyError):
                invalid_text_rows += 1
                issues.append(ValidationIssue(row_number, "invalid_text"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
                text = ""

        raw_labels = row.get("labels")
        if _is_missing(raw_labels):
            missing_label_rows += 1
            issues.append(ValidationIssue(row_number, "missing_labels"))
            invalid_row_numbers.add(row_number)
            row_invalid = True
            label_values: list[object] = []
        else:
            try:
                label_values = _parse_label_array(raw_labels)
            except ValueError:
                issues.append(ValidationIssue(row_number, "invalid_labels_format"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
                label_values = []

        mapped_labels, bad_labels, duplicate_labels = _map_label_values(label_values, mapping)
        if bad_labels:
            unsupported_values.update(bad_labels)
            unsupported_occurrences += len(bad_labels)
            issues.append(ValidationIssue(row_number, "unsupported_labels"))
            invalid_row_numbers.add(row_number)
            row_invalid = True
        if duplicate_labels:
            issues.append(ValidationIssue(row_number, "duplicate_labels"))
            invalid_row_numbers.add(row_number)
            row_invalid = True

        if has_known_column:
            raw_known = row.get("known_labels")
            if _is_missing(raw_known):
                missing_known_label_rows += 1
                issues.append(ValidationIssue(row_number, "missing_known_labels"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
                known_values: list[object] = []
            else:
                try:
                    known_values = _parse_label_array(raw_known)
                except ValueError:
                    issues.append(ValidationIssue(row_number, "invalid_known_labels_format"))
                    invalid_row_numbers.add(row_number)
                    row_invalid = True
                    known_values = []
            mapped_known, bad_known, duplicate_known = _map_label_values(known_values, mapping)
            if bad_known:
                unsupported_values.update(bad_known)
                unsupported_occurrences += len(bad_known)
                issues.append(ValidationIssue(row_number, "unsupported_known_labels"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
            if duplicate_known:
                issues.append(ValidationIssue(row_number, "duplicate_known_labels"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
            known_labels = set(mapped_known)
        elif assume_all_labels_known:
            known_labels = set(EMOTION_LABELS)
        else:
            known_labels = set(mapped_labels)

        if not set(mapped_labels).issubset(known_labels):
            issues.append(ValidationIssue(row_number, "positive_label_not_marked_known"))
            invalid_row_numbers.add(row_number)
            row_invalid = True
        if not known_labels:
            issues.append(ValidationIssue(row_number, "no_known_labels"))
            invalid_row_numbers.add(row_number)
            row_invalid = True

        group_id: str | None = None
        if has_group_column:
            raw_group_id = row.get("group_id")
            if (
                not isinstance(raw_group_id, str)
                or _GROUP_ID.fullmatch(raw_group_id.strip()) is None
            ):
                issues.append(ValidationIssue(row_number, "missing_or_invalid_group_id"))
                invalid_row_numbers.add(row_number)
                row_invalid = True
            else:
                group_id = raw_group_id.strip()

        if not row_invalid:
            candidates.append(
                (
                    row_number,
                    PreparedExample(
                        text=text,
                        labels=tuple(label for label in EMOTION_LABELS if label in mapped_labels),
                        known_labels=tuple(label for label in EMOTION_LABELS if label in known_labels),
                        group_id=group_id,
                    ),
                )
            )

    deduplicated, duplicate_rows_dropped, conflicting_duplicate_groups, duplicate_issues = (
        _deduplicate_candidates(candidates)
    )
    issues.extend(duplicate_issues)
    invalid_row_numbers.update(issue.row_number for issue in duplicate_issues)

    positive_counts = Counter(
        label for _, example in deduplicated for label in example.labels
    )
    known_counts = Counter(
        label for _, example in deduplicated for label in example.known_labels
    )
    group_count = (
        len({example.group_id for _, example in deduplicated}) if has_group_column else None
    )
    report = DatasetValidationReport(
        input_rows=len(source_records),
        accepted_records=len(deduplicated),
        invalid_rows=len(invalid_row_numbers),
        missing_text_rows=missing_text_rows,
        invalid_text_rows=invalid_text_rows,
        missing_label_rows=missing_label_rows,
        missing_known_label_rows=missing_known_label_rows,
        duplicate_rows_dropped=duplicate_rows_dropped,
        conflicting_duplicate_groups=conflicting_duplicate_groups,
        unsupported_label_occurrences=unsupported_occurrences,
        distinct_unsupported_label_values=len(unsupported_values),
        positive_label_counts=dict(positive_counts),
        known_label_counts=dict(known_counts),
        group_id_column_present=has_group_column,
        provided_group_count=group_count,
        issues=tuple(sorted(issues, key=lambda issue: (issue.row_number, issue.code))),
    )
    return PreparedDataset(
        examples=tuple(example for _, example in deduplicated),
        report=report,
    )


def _validated_label_mapping(
    label_mapping: Mapping[str, str | EmotionLabel] | None,
) -> dict[str, EmotionLabel]:
    normalized: dict[str, EmotionLabel] = {}
    if label_mapping is None:
        return normalized
    if not isinstance(label_mapping, Mapping):
        raise DatasetContractError("label_mapping must map source label names to target labels.")
    canonical = {label.value.casefold(): label for label in EMOTION_LABELS}
    for source, target in label_mapping.items():
        if not isinstance(source, str) or not source.strip():
            raise DatasetContractError("Mapping source labels must be non-empty strings.")
        source_key = unicodedata.normalize("NFC", source.strip()).casefold()
        if source_key in normalized:
            raise DatasetContractError("Mapping contains duplicate normalized source labels.")
        target_value = target.value if isinstance(target, EmotionLabel) else target
        if not isinstance(target_value, str) or target_value.casefold() not in canonical:
            raise DatasetContractError("Every mapping destination must be one of the five target labels.")
        target_label = canonical[target_value.casefold()]
        if source_key in canonical and canonical[source_key] is not target_label:
            raise DatasetContractError("A canonical target label cannot be remapped to another label.")
        normalized[source_key] = target_label
    return normalized


def _parse_label_array(value: object) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("labels must be a JSON array") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("labels must be a JSON array")
    return list(value)


def _map_label_values(
    values: Sequence[object], mapping: Mapping[str, EmotionLabel]
) -> tuple[list[EmotionLabel], list[str], bool]:
    canonical = {label.value.casefold(): label for label in EMOTION_LABELS}
    mapped: list[EmotionLabel] = []
    unsupported: list[str] = []
    seen: set[EmotionLabel] = set()
    duplicate = False
    for value in values:
        if not isinstance(value, str) or not value.strip():
            unsupported.append("<invalid-label-format>")
            continue
        key = unicodedata.normalize("NFC", value.strip()).casefold()
        label = canonical.get(key, mapping.get(key))
        if label is None:
            unsupported.append(value)
            continue
        if label in seen:
            duplicate = True
            continue
        seen.add(label)
        mapped.append(label)
    return mapped, unsupported, duplicate


def _deduplicate_candidates(
    candidates: Sequence[tuple[int, PreparedExample]],
) -> tuple[list[tuple[int, PreparedExample]], int, int, list[ValidationIssue]]:
    by_text: dict[str, list[tuple[int, PreparedExample]]] = defaultdict(list)
    for row_number, example in candidates:
        by_text[example.text].append((row_number, example))

    kept: list[tuple[int, PreparedExample]] = []
    duplicate_rows = 0
    conflicts = 0
    issues: list[ValidationIssue] = []
    for duplicate_set in by_text.values():
        if len(duplicate_set) == 1:
            kept.append(duplicate_set[0])
            continue
        signatures = {
            (example.labels, example.known_labels, example.group_id)
            for _, example in duplicate_set
        }
        if len(signatures) == 1:
            duplicate_rows += len(duplicate_set) - 1
            kept.append(duplicate_set[0])
        else:
            conflicts += 1
            issues.extend(
                ValidationIssue(row_number, "conflicting_duplicate_text")
                for row_number, _ in duplicate_set
            )
    kept.sort(key=lambda row: row[0])
    return kept, duplicate_rows, conflicts, issues


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)

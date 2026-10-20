"""Local CSV/JSONL loading, audit serialization, and preparation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..prediction.schema import EMOTION_LABELS, EmotionLabel
from ..preprocessing import (
    DatasetContractError,
    DatasetValidationReport,
    PreparedDataset,
    PreparedExample,
    prepare_records,
)
from .data import TrainingAuthorization, TrainingDataError, validate_training_splits
from .splitting import DatasetSplitError, DatasetSplits, SPLIT_NAMES, SplitReport, split_dataset


_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SPLIT_FILES = {name: f"{name}.jsonl" for name in SPLIT_NAMES}
_MANIFEST_FILES = {**_SPLIT_FILES, "report": "split_report.json"}
_GROUP_ID_STRATEGY = "provided group_id values (group disjoint; participant/source meaning is not verified)"
_TEXT_HASH_STRATEGY = "normalized text hash only (participant-level grouping unavailable)"
_GROUPING_LIMITATION = (
    "Persisted group IDs are checked for cross-split disjointness, but their participant/source "
    "meaning is not verified. Without group IDs, only exact normalized-text separation is available."
)
_APPROVAL_CHECKS = (
    "approved_for_training",
    "license_or_consent_reviewed",
    "label_mapping_reviewed",
    "data_protection_reviewed",
)


class DatasetManifestError(ValueError):
    """Raised when persisted split files or their training manifest are invalid."""


@dataclass(frozen=True, slots=True)
class PreparedSplitBundle:
    """Parsed file bundle without training authorization or DatasetSplits capability."""

    train: tuple[PreparedExample, ...]
    validation: tuple[PreparedExample, ...]
    test: tuple[PreparedExample, ...]
    report: SplitReport
    source: Mapping[str, object]
    manifest_fingerprint: str

    @property
    def dataset_reference(self) -> str:
        return str(self.source["dataset_reference"])


def load_dataset_file(
    path: str | Path,
    *,
    label_mapping: Mapping[str, str] | None = None,
    assume_all_labels_known: bool = False,
) -> PreparedDataset:
    """Load a UTF-8 CSV or JSONL file and validate it against the project contract."""
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        columns, records = _read_csv(source)
    elif suffix in {".jsonl", ".ndjson"}:
        columns, records = _read_jsonl(source)
    else:
        raise DatasetContractError("Supported input extensions are .csv, .jsonl, and .ndjson.")
    return prepare_records(
        records,
        columns=columns,
        label_mapping=label_mapping,
        assume_all_labels_known=assume_all_labels_known,
    )


def write_audit_report(report: DatasetValidationReport, path: str | Path) -> Path:
    """Write only the report's privacy-safe counts and codes as UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def write_split_outputs(splits: DatasetSplits, output_dir: str | Path) -> Path:
    """Write local JSONL splits plus a text-free split summary.

    The split JSONL files necessarily contain the supplied text for later training.
    Keep them under the ignored ``data/processed/`` directory and control access.
    When source group IDs were used, the opaque IDs are retained in these restricted
    files for reload-time leakage checks; they are not model inputs or report fields.
    """
    has_group_ids = splits.report.grouping_strategy == _GROUP_ID_STRATEGY
    groups = {name: getattr(splits, name) for name in SPLIT_NAMES}
    _validate_persisted_group_assignment(groups, splits.report, require_group_ids=has_group_ids)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, examples in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        rows = [
            {
                "text": example.text,
                "labels": [label.value for label in example.labels],
                "known_labels": [label.value for label in example.known_labels],
            }
            for example in examples
        ]
        if has_group_ids:
            for row, example in zip(rows, examples):
                row["group_id"] = example.group_id
        file_path = destination / f"{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = destination / "split_report.json"
    summary_path.write_text(
        json.dumps(splits.report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def write_training_manifest(
    splits: DatasetSplits,
    output_dir: str | Path,
    *,
    dataset_reference: str,
    authorization: TrainingAuthorization,
) -> Path:
    """Write a privacy-minimized, approval-bound manifest beside prepared JSONL.

    This must be called after `write_split_outputs` and after human source review.
    Fingerprints cover ordered text/annotation content, but are integrity digests,
    not anonymization or proof that the caller's approval assertions are true.
    """
    if not isinstance(splits, DatasetSplits):
        raise DatasetManifestError("A DatasetSplits object from the preparation pipeline is required.")
    try:
        authorization.validate()
        validate_training_splits(splits, authorization)
    except (TrainingDataError, TypeError) as exc:
        raise DatasetManifestError(f"Cannot manifest unapproved or unusable splits: {exc}") from exc
    if not isinstance(dataset_reference, str) or not _SAFE_REFERENCE.fullmatch(dataset_reference):
        raise DatasetManifestError("dataset_reference must be an opaque, non-identifying version code.")

    directory = Path(output_dir)
    persisted_report = _read_split_report(directory / "split_report.json")
    if persisted_report.to_dict() != splits.report.to_dict():
        raise DatasetManifestError("Persisted split report does not match the supplied split object.")
    persisted = {
        split_name: _read_split_examples(
            directory / _SPLIT_FILES[split_name],
            include_group_id=splits.report.grouping_strategy == _GROUP_ID_STRATEGY,
        )
        for split_name in SPLIT_NAMES
    }
    fingerprints = {
        split_name: _examples_content_fingerprint(examples, include_group_id=True)
        for split_name, examples in persisted.items()
    }
    for split_name in SPLIT_NAMES:
        if fingerprints[split_name] != _examples_content_fingerprint(
            getattr(splits, split_name), include_group_id=True
        ):
            raise DatasetManifestError(f"Persisted {split_name} data does not match the supplied split object.")

    source = {
        "dataset_reference": dataset_reference,
        "dataset_id": authorization.dataset_id,
        "source_kind": authorization.source_kind,
        "approval_reference": authorization.approval_reference,
        "approval_checks": {key: getattr(authorization, key) for key in _APPROVAL_CHECKS},
    }
    manifest = _build_training_manifest(splits.report, source, fingerprints)
    manifest_path = directory / "training_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def read_prepared_split_files(split_dir: str | Path) -> PreparedSplitBundle:
    """Validate persisted files and fingerprints without granting training approval.

    The returned bundle is not a `DatasetSplits` instance and cannot be passed to a
    model fit function. Use `load_training_splits` with a reviewed authorization.
    """
    directory = Path(split_dir)
    try:
        manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetManifestError("A readable training_manifest.json is required.") from exc
    if not isinstance(manifest, dict):
        raise DatasetManifestError("Training manifest must be a JSON object.")
    _validate_manifest(manifest)
    report = _read_split_report(directory / _MANIFEST_FILES["report"])
    _compare_manifest_to_report(manifest, report)
    groups = {
        split_name: _read_split_examples(
            directory / _SPLIT_FILES[split_name],
            include_group_id=(
                manifest["manifest_version"] == 2
                and manifest["split"]["grouping_strategy"] == _GROUP_ID_STRATEGY
            ),
        )
        for split_name in SPLIT_NAMES
    }
    split_manifest = manifest["split"]
    include_group_id_in_fingerprint = manifest["manifest_version"] == 2
    for split_name, examples in groups.items():
        if len(examples) != report.rows_per_split[split_name]:
            raise DatasetManifestError(f"{split_name} file row count differs from split_report.json.")
        if _examples_content_fingerprint(
            examples, include_group_id=include_group_id_in_fingerprint
        ) != split_manifest["content_fingerprints"][split_name]:
            raise DatasetManifestError(f"{split_name} content fingerprint does not match the manifest.")
    _validate_loaded_label_counts(groups, report)
    _validate_loaded_text_separation(groups)
    _validate_persisted_group_assignment(
        groups,
        report,
        require_group_ids=manifest["split"]["grouping_strategy"] == _GROUP_ID_STRATEGY,
    )
    return PreparedSplitBundle(
        train=groups["train"],
        validation=groups["validation"],
        test=groups["test"],
        report=report,
        source=manifest["source"],
        manifest_fingerprint=manifest["manifest_fingerprint"],
    )


def load_training_splits(
    split_dir: str | Path,
    *,
    authorization: TrainingAuthorization,
) -> DatasetSplits:
    """Load approved persisted splits after manifest, schema, and leakage checks."""
    if not isinstance(authorization, TrainingAuthorization):
        raise DatasetManifestError("An explicit TrainingAuthorization is required.")
    try:
        authorization.validate()
    except TrainingDataError as exc:
        raise DatasetManifestError(f"Persisted data is not approved for training: {exc}") from exc
    bundle = read_prepared_split_files(split_dir)
    source = bundle.source
    _validate_manifest_authorization(source, authorization)

    # Construct the fit-capable object only after file, manifest, and approval checks.
    splits = DatasetSplits(
        train=bundle.train,
        validation=bundle.validation,
        test=bundle.test,
        report=bundle.report,
        dataset_reference=bundle.dataset_reference,
        dataset_id=authorization.dataset_id,
        source_kind=authorization.source_kind,
        approval_reference=authorization.approval_reference,
        split_manifest_fingerprint=bundle.manifest_fingerprint,
    )
    try:
        validate_training_splits(splits, authorization)
    except TrainingDataError as exc:
        raise DatasetManifestError(f"Loaded splits are not usable for training: {exc}") from exc
    return splits


def _validate_manifest_authorization(
    source: Mapping[str, object],
    authorization: TrainingAuthorization,
) -> None:
    if source.get("source_kind") not in {"reviewed_public", "consent_approved_supplement"}:
        raise DatasetManifestError("Demonstration and test-only fixtures cannot be loaded for training.")
    if (
        source.get("source_kind") != authorization.source_kind
        or source.get("dataset_id") != authorization.dataset_id
        or source.get("approval_reference") != authorization.approval_reference
    ):
        raise DatasetManifestError("Manifest dataset/approval reference does not match authorization.")
    checks = source.get("approval_checks")
    if not isinstance(checks, Mapping) or any(checks.get(key) is not True for key in _APPROVAL_CHECKS):
        raise DatasetManifestError("Manifest does not record all required human review assertions.")


def _build_training_manifest(
    report: SplitReport,
    source: Mapping[str, object],
    content_fingerprints: Mapping[str, str],
) -> dict[str, object]:
    report_dict = report.to_dict()
    manifest: dict[str, object] = {
        "manifest_version": 2,
        "label_order": [label.value for label in EMOTION_LABELS],
        "files": dict(_MANIFEST_FILES),
        "source": dict(source),
        "split": {
            "seed": report.seed,
            "requested_ratios": dict(report.requested_ratios),
            "rows_per_split": dict(report.rows_per_split),
            "groups_per_split": dict(report.groups_per_split),
            "positive_label_counts": report_dict["positive_label_counts"],
            "known_label_counts": report_dict["known_label_counts"],
            "grouping_strategy": report.grouping_strategy,
            "group_count": report.group_count,
            "content_fingerprints": dict(content_fingerprints),
        },
        "leakage_safeguards": {
            "normalized_text_disjointness_rechecked_on_load": True,
            "grouping_strategy": report.grouping_strategy,
            "group_id_values_stored_in_split_files": report.grouping_strategy == _GROUP_ID_STRATEGY,
            "group_id_semantics_verified": False,
            "participant_grouping_limitation": _GROUPING_LIMITATION,
        },
    }
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    expected_fields = {
        "manifest_version", "label_order", "files", "source", "split",
        "leakage_safeguards", "manifest_fingerprint",
    }
    manifest_version = manifest.get("manifest_version")
    if (
        set(manifest) != expected_fields
        or type(manifest_version) is not int
        or manifest_version not in {1, 2}
    ):
        raise DatasetManifestError("Training manifest schema/version is unsupported.")
    if manifest.get("label_order") != [label.value for label in EMOTION_LABELS]:
        raise DatasetManifestError("Manifest label order differs from the canonical five-label order.")
    if manifest.get("files") != _MANIFEST_FILES:
        raise DatasetManifestError("Manifest split filenames do not match the supported pipeline outputs.")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "dataset_reference", "dataset_id", "source_kind", "approval_reference", "approval_checks"
    }:
        raise DatasetManifestError("Manifest source metadata is missing or contains unsupported fields.")
    for key in ("dataset_reference", "dataset_id", "approval_reference"):
        value = source.get(key)
        if not isinstance(value, str) or not _SAFE_REFERENCE.fullmatch(value):
            raise DatasetManifestError(f"Manifest {key} must be an opaque non-identifying reference.")
    source_kind = source.get("source_kind")
    checks = source.get("approval_checks")
    if not isinstance(source_kind, str) or source_kind not in {
        "reviewed_public", "consent_approved_supplement", "test_fixture"
    }:
        raise DatasetManifestError("Manifest source kind is unsupported.")
    if not isinstance(checks, Mapping) or set(checks) != set(_APPROVAL_CHECKS):
        raise DatasetManifestError("Manifest human-review fields do not match the supported schema.")
    if any(type(checks[key]) is not bool for key in _APPROVAL_CHECKS):
        raise DatasetManifestError("Manifest human-review values must be booleans.")
    if source_kind == "test_fixture":
        if any(checks.values()):
            raise DatasetManifestError("Test-only fixtures cannot carry training approval flags.")
    elif any(checks[key] is not True for key in _APPROVAL_CHECKS):
        raise DatasetManifestError("Reviewed sources require every human-approval assertion.")

    split = manifest.get("split")
    expected_split = {
        "seed", "requested_ratios", "rows_per_split", "groups_per_split",
        "positive_label_counts", "known_label_counts", "grouping_strategy",
        "group_count", "content_fingerprints",
    }
    if not isinstance(split, Mapping) or set(split) != expected_split:
        raise DatasetManifestError("Manifest split settings do not match the supported schema.")
    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetManifestError("Manifest split seed must be a non-negative integer.")
    ratios = split.get("requested_ratios")
    if not isinstance(ratios, Mapping) or set(ratios) != set(SPLIT_NAMES):
        raise DatasetManifestError("Manifest split ratios must define all three partitions.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0
        for value in ratios.values()
    ) or not math.isclose(sum(float(value) for value in ratios.values()), 1.0, abs_tol=1e-9):
        raise DatasetManifestError("Manifest split ratios must be positive and sum to one.")
    _validate_split_count_map(split.get("rows_per_split"), allow_zero=False, name="rows_per_split")
    _validate_split_count_map(split.get("groups_per_split"), allow_zero=False, name="groups_per_split")
    group_count = split.get("group_count")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 3:
        raise DatasetManifestError("Manifest must record at least three source split groups.")
    if sum(split["groups_per_split"].values()) != group_count:
        raise DatasetManifestError("Manifest group counts do not sum to group_count.")
    if any(
        split["groups_per_split"][name] > split["rows_per_split"][name]
        for name in SPLIT_NAMES
    ):
        raise DatasetManifestError("Manifest group counts cannot exceed row counts.")
    for count_key in ("positive_label_counts", "known_label_counts"):
        counts = split.get(count_key)
        if not isinstance(counts, Mapping) or set(counts) != set(SPLIT_NAMES):
            raise DatasetManifestError(f"Manifest {count_key} must cover all partitions.")
        for name in SPLIT_NAMES:
            _validate_label_count_map(counts[name], count_key)
    strategy = split.get("grouping_strategy")
    if not isinstance(strategy, str) or strategy not in {
        _GROUP_ID_STRATEGY,
        _TEXT_HASH_STRATEGY,
    }:
        raise DatasetManifestError("Manifest grouping strategy is not recognized.")
    if strategy == _TEXT_HASH_STRATEGY:
        if split["group_count"] != sum(split["rows_per_split"].values()):
            raise DatasetManifestError("Text-hash split group count must match deduplicated rows.")
        if split["groups_per_split"] != split["rows_per_split"]:
            raise DatasetManifestError("Text-hash split groups must match deduplicated rows per partition.")
    elif manifest_version == 1:
        raise DatasetManifestError(
            "Legacy group-aware split manifest has no persisted group IDs; regenerate the split bundle."
        )
    fingerprints = split.get("content_fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != set(SPLIT_NAMES):
        raise DatasetManifestError("Manifest requires one content fingerprint for each split.")
    if any(not isinstance(value, str) or not _SHA256_HEX.fullmatch(value) for value in fingerprints.values()):
        raise DatasetManifestError("Split content fingerprints must be SHA-256 hex values.")

    safeguards = manifest.get("leakage_safeguards")
    expected_safeguards = {
        "normalized_text_disjointness_rechecked_on_load": True,
        "grouping_strategy": strategy,
        "group_id_values_stored_in_split_files": (
            strategy == _GROUP_ID_STRATEGY if manifest_version == 2 else False
        ),
        "group_id_semantics_verified": False,
        "participant_grouping_limitation": (
            _GROUPING_LIMITATION if manifest_version == 2 else (
                "Participant-level separation depends on suitable source group IDs; "
                "when unavailable, only exact normalized-text grouping is provided."
            )
        ),
    }
    if safeguards != expected_safeguards:
        raise DatasetManifestError("Manifest leakage safeguards are missing, changed, or unsupported.")
    fingerprint = manifest.get("manifest_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_HEX.fullmatch(fingerprint):
        raise DatasetManifestError("Manifest fingerprint must be SHA-256 hex.")
    if fingerprint != _manifest_fingerprint(manifest):
        raise DatasetManifestError("Training manifest fingerprint does not match its content.")


def _read_split_report(path: Path) -> SplitReport:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetManifestError("A readable split_report.json is required.") from exc
    expected = {
        "report_version", "seed", "requested_ratios", "rows_per_split", "groups_per_split",
        "positive_label_counts", "known_label_counts", "grouping_strategy", "group_count", "privacy",
    }
    report_version = raw.get("report_version") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected
        or type(report_version) is not int
        or report_version != 1
    ):
        raise DatasetManifestError("split_report.json schema/version is unsupported.")
    if raw.get("privacy") != "Report excludes input text and group identifiers.":
        raise DatasetManifestError("Split report privacy declaration is missing or changed.")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetManifestError("Split report seed must be a non-negative integer.")
    ratios = raw.get("requested_ratios")
    if not isinstance(ratios, Mapping) or set(ratios) != set(SPLIT_NAMES):
        raise DatasetManifestError("Split report ratios must define all partitions.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0
        for value in ratios.values()
    ) or not math.isclose(sum(float(value) for value in ratios.values()), 1.0, abs_tol=1e-9):
        raise DatasetManifestError("Split report ratios must be positive and sum to one.")
    rows = raw.get("rows_per_split")
    groups = raw.get("groups_per_split")
    _validate_split_count_map(rows, allow_zero=False, name="rows_per_split")
    _validate_split_count_map(groups, allow_zero=False, name="groups_per_split")
    group_count = raw.get("group_count")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 3:
        raise DatasetManifestError("Split report must record at least three groups.")
    if sum(groups.values()) != group_count or any(groups[name] > rows[name] for name in SPLIT_NAMES):
        raise DatasetManifestError("Split report group counts are inconsistent.")
    strategy = raw.get("grouping_strategy")
    if not isinstance(strategy, str) or strategy not in {
        "provided group_id values (group disjoint; participant/source meaning is not verified)",
        "normalized text hash only (participant-level grouping unavailable)",
    }:
        raise DatasetManifestError("Split report grouping strategy is not recognized.")
    positive_counts = raw.get("positive_label_counts")
    known_counts = raw.get("known_label_counts")
    for name, count_map in (("positive_label_counts", positive_counts), ("known_label_counts", known_counts)):
        if not isinstance(count_map, Mapping) or set(count_map) != set(SPLIT_NAMES):
            raise DatasetManifestError(f"Split report {name} must cover all partitions.")
        for split_name in SPLIT_NAMES:
            _validate_label_count_map(count_map[split_name], name)
            if any(count_map[split_name][label.value] > rows[split_name] for label in EMOTION_LABELS):
                raise DatasetManifestError(f"Split report {name} exceeds its row count.")
    if any(
        positive_counts[name][label.value] > known_counts[name][label.value]
        for name in SPLIT_NAMES for label in EMOTION_LABELS
    ):
        raise DatasetManifestError("Split report positive labels exceed known-label counts.")
    if strategy == "normalized text hash only (participant-level grouping unavailable)":
        if group_count != sum(rows.values()) or groups != rows:
            raise DatasetManifestError("Text-hash split groups must match deduplicated rows.")
    return SplitReport(
        seed=seed,
        requested_ratios={name: float(value) for name, value in ratios.items()},
        rows_per_split=dict(rows),
        groups_per_split=dict(groups),
        positive_counts={
            name: {EmotionLabel(label): count for label, count in counts.items()}
            for name, counts in positive_counts.items()
        },
        known_counts={
            name: {EmotionLabel(label): count for label, count in counts.items()}
            for name, counts in known_counts.items()
        },
        grouping_strategy=strategy,
        group_count=group_count,
    )


def _read_split_examples(
    path: Path, *, include_group_id: bool = False
) -> tuple[PreparedExample, ...]:
    records: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetManifestError(f"Invalid JSON in {path.name} line {line_number}.") from exc
                if not isinstance(row, dict):
                    raise DatasetManifestError(f"{path.name} line {line_number} must be a JSON object.")
                records.append(row)
    except (OSError, UnicodeError) as exc:
        raise DatasetManifestError(f"Cannot read prepared split file {path.name}.") from exc
    if not records:
        raise DatasetManifestError(f"Prepared split file {path.name} is empty.")
    try:
        columns = ("text", "labels", "known_labels", "group_id") if include_group_id else (
            "text", "labels", "known_labels"
        )
        prepared = prepare_records(records, columns=columns)
    except DatasetContractError as exc:
        raise DatasetManifestError(f"Prepared split file {path.name} violates the dataset contract.") from exc
    if not prepared.report.is_valid or len(prepared.examples) != len(records):
        raise DatasetManifestError(
            f"Prepared split file {path.name} violates the dataset contract (invalid or duplicate rows)."
        )
    return prepared.examples


def _validate_split_count_map(value: object, *, allow_zero: bool, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(SPLIT_NAMES):
        raise DatasetManifestError(f"Manifest {name} must define train, validation, and test.")
    lower_bound = 0 if allow_zero else 1
    if any(isinstance(count, bool) or not isinstance(count, int) or count < lower_bound for count in value.values()):
        raise DatasetManifestError(f"Manifest {name} contains invalid counts.")


def _validate_label_count_map(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {label.value for label in EMOTION_LABELS}:
        raise DatasetManifestError(f"Manifest {name} must contain all five canonical labels.")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in value.values()):
        raise DatasetManifestError(f"Manifest {name} contains invalid label counts.")


def _compare_manifest_to_report(manifest: Mapping[str, object], report: SplitReport) -> None:
    split = manifest["split"]
    report_dict = report.to_dict()
    for key in (
        "seed", "requested_ratios", "rows_per_split", "groups_per_split",
        "positive_label_counts", "known_label_counts", "grouping_strategy", "group_count",
    ):
        if split.get(key) != report_dict.get(key):
            raise DatasetManifestError(f"Manifest {key} does not match split_report.json.")


def _validate_loaded_label_counts(
    groups: Mapping[str, tuple[PreparedExample, ...]],
    report: SplitReport,
) -> None:
    for split_name, examples in groups.items():
        positive = {label: 0 for label in EMOTION_LABELS}
        known = {label: 0 for label in EMOTION_LABELS}
        for example in examples:
            for label in example.labels:
                positive[label] += 1
            for label in example.known_labels:
                known[label] += 1
        if positive != {label: report.positive_counts[split_name].get(label, 0) for label in EMOTION_LABELS}:
            raise DatasetManifestError(f"{split_name} positive-label counts differ from split_report.json.")
        if known != {label: report.known_counts[split_name].get(label, 0) for label in EMOTION_LABELS}:
            raise DatasetManifestError(f"{split_name} known-label counts differ from split_report.json.")


def _validate_loaded_text_separation(groups: Mapping[str, tuple[PreparedExample, ...]]) -> None:
    from ..preprocessing import normalize_text

    assigned: dict[str, str] = {}
    for split_name, examples in groups.items():
        for example in examples:
            normalized = normalize_text(example.text)
            previous = assigned.get(normalized)
            if previous is not None and previous != split_name:
                raise DatasetManifestError("Exact normalized text appears in more than one persisted split.")
            assigned[normalized] = split_name


def _validate_persisted_group_assignment(
    groups: Mapping[str, object],
    report: SplitReport,
    *,
    require_group_ids: bool,
) -> None:
    """Check persisted grouping tokens without claiming what those tokens represent."""
    group_owners: dict[str, str] = {}
    group_counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        examples = groups[split_name]
        if not isinstance(examples, (tuple, list)):
            raise DatasetManifestError("Split groups must be sequences of prepared examples.")
        seen_in_split: set[str] = set()
        for example in examples:
            group_id = getattr(example, "group_id", None)
            if require_group_ids:
                if not isinstance(group_id, str) or not group_id:
                    raise DatasetManifestError(
                        "Group-aware split files must retain a valid group_id on every row."
                    )
                previous_split = group_owners.get(group_id)
                if previous_split is not None and previous_split != split_name:
                    raise DatasetManifestError("A group_id appears in more than one persisted split.")
                group_owners[group_id] = split_name
                seen_in_split.add(group_id)
            elif group_id is not None:
                raise DatasetManifestError(
                    "Text-hash split files cannot contain group IDs absent from the split manifest."
                )
        group_counts[split_name] = len(seen_in_split) if require_group_ids else len(examples)

    if group_counts != dict(report.groups_per_split):
        raise DatasetManifestError("Persisted group counts differ from split_report.json.")
    if not require_group_ids and report.grouping_strategy != _TEXT_HASH_STRATEGY:
        raise DatasetManifestError("Group-aware split cannot be verified without persisted group IDs.")


def _examples_content_fingerprint(
    examples: tuple[PreparedExample, ...], *, include_group_id: bool = True
) -> str:
    digest = hashlib.sha256()
    for example in examples:
        row = {
            "text": example.text,
            "labels": [label.value for label in example.labels],
            "known_labels": [label.value for label in example.known_labels],
        }
        if include_group_id:
            row["group_id"] = example.group_id
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _manifest_fingerprint(manifest: Mapping[str, object]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            raise DatasetContractError("CSV input is empty or has no header row.")
        if len(set(header)) != len(header):
            raise DatasetContractError("CSV header names must be unique.")
        records: list[dict[str, object]] = []
        for row_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(header):
                raise DatasetContractError(
                    f"CSV row {row_number} does not match the header column count."
                )
            records.append(dict(zip(header, values)))
    return tuple(header), records


def _read_jsonl(path: Path) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetContractError(f"Invalid JSON on line {line_number}.") from exc
            if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
                raise DatasetContractError(
                    f"JSONL line {line_number} must contain one object with string keys."
                )
            for key in row:
                if key not in columns:
                    columns.append(key)
            records.append(row)
    if not columns:
        raise DatasetContractError("JSONL input is empty or contains only blank lines.")
    return tuple(columns), records


def _read_mapping_file(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetContractError("Label mapping file must be a JSON object.") from exc
    if not isinstance(mapping, dict) or any(
        not isinstance(source, str) or not isinstance(target, str)
        for source, target in mapping.items()
    ):
        raise DatasetContractError("Label mapping file must map source-label strings to target-label strings.")
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a reviewed local dataset, write a safe audit, and create fixed-seed splits."
    )
    parser.add_argument("input", type=Path, help="UTF-8 .csv, .jsonl, or .ndjson input")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/processed/dataset_audit.json"),
        help="Text-free audit JSON destination (default: data/processed/dataset_audit.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Local directory for prepared train/validation/test JSONL files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed (default: 42)")
    parser.add_argument(
        "--assume-all-labels-known",
        action="store_true",
        help="Treat every omitted target label as an explicit negative; use only for reviewed complete labels.",
    )
    parser.add_argument(
        "--label-mapping-file",
        type=Path,
        help="Optional reviewed JSON mapping from exact source labels to the five target labels.",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset_file(
            args.input,
            label_mapping=_read_mapping_file(args.label_mapping_file),
            assume_all_labels_known=args.assume_all_labels_known,
        )
        write_audit_report(dataset.report, args.audit_output)
    except (DatasetContractError, OSError, UnicodeError) as exc:
        print(f"Dataset preparation stopped: {exc}", file=sys.stderr)
        return 2

    if not dataset.report.is_valid:
        print(
            f"Dataset validation failed: {dataset.report.invalid_rows} invalid row(s), "
            f"{dataset.report.accepted_records} accepted record(s); "
            f"privacy-safe report written to {args.audit_output}.",
            file=sys.stderr,
        )
        return 2

    try:
        splits = split_dataset(dataset, seed=args.seed)
        write_split_outputs(splits, args.output_dir)
    except (DatasetSplitError, OSError) as exc:
        print(f"Dataset splitting stopped: {exc}", file=sys.stderr)
        return 2

    print(
        f"Validated {dataset.report.accepted_records} record(s); "
        f"split report and local data files written under {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

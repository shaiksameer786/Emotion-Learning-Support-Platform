"""Privacy-conscious checkpoint metadata and atomic local weight persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

from ..prediction.schema import EMOTION_LABELS
from .config import BertModelConfig, BiLSTMModelConfig, TrainingConfig
from .data import ALLOWED_SOURCE_KINDS
from .engine import TrainingRunResult
from .metrics import FinalizedEvaluationPlan
from .optional import require_torch


_SAFE_MODEL_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")
_VALID_METRIC_KEYS = frozenset(
    {
        "macro_f1", "macro_f1_defined_labels", "micro_f1", "subset_accuracy",
        "subset_accuracy_rows", "hamming_loss", "known_label_decisions",
    }
)
_VALID_PER_LABEL_KEYS = frozenset(
    {
        "precision", "recall", "f1", "support", "known_count", "undefined",
        "precision_undefined", "confusion_matrix",
    }
)
_CHECKPOINT_METRIC_KEYS = frozenset(_VALID_METRIC_KEYS | {"per_label"})
_RAW_METRIC_REQUIRED_KEYS = frozenset(_CHECKPOINT_METRIC_KEYS | {"label_order", "thresholds"})
_RAW_METRIC_CONTEXT_KEYS = frozenset(
    {"evaluation_split", "threshold_source", "configuration_digest"}
)
_SAFE_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SAFE_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?\Z")
_GROUPING_STRATEGIES = frozenset(
    {
        "provided group_id values (group disjoint; participant/source meaning is not verified)",
        "normalized text hash only (participant-level grouping unavailable)",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "format_version", "model_type", "model_version", "label_order", "training_config",
        "model_config", "configuration_digest", "best_epoch", "validation_history",
        "threshold_selection", "evaluation_plan", "validation_metrics", "training_data_summary",
        "vocabulary", "tokenizer", "privacy", "test_metrics", "weights_sha256",
    }
)


class CheckpointMetadataError(ValueError):
    """Raised when checkpoint metadata has unsafe fields or inconsistent labels."""


def build_checkpoint_metadata(
    run: TrainingRunResult,
    *,
    model_type: str,
    model_version: str,
    training_config: TrainingConfig,
    model_config: BiLSTMModelConfig | BertModelConfig,
    evaluation_plan: FinalizedEvaluationPlan,
    test_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create reproducibility metadata without examples, identifiers, or secret values."""
    if model_type not in {"bilstm", "bert"}:
        raise CheckpointMetadataError("model_type must be 'bilstm' or 'bert'.")
    if not isinstance(model_version, str) or _SAFE_MODEL_VERSION.fullmatch(model_version) is None:
        raise CheckpointMetadataError("model_version must be a short non-identifying version string.")
    if tuple(training_config.label_order) != tuple(label.value for label in EMOTION_LABELS):
        raise CheckpointMetadataError("Training configuration label order differs from the shared schema.")
    if tuple(run.threshold_selection.thresholds) != tuple(evaluation_plan.thresholds):
        raise CheckpointMetadataError("Finalized thresholds differ from validation-selected thresholds.")
    if evaluation_plan.configuration_digest != run.configuration_digest:
        raise CheckpointMetadataError("Evaluation plan was finalized for a different configuration.")
    if evaluation_plan.threshold_source != "validation":
        raise CheckpointMetadataError("Checkpoint thresholds must come from validation data.")
    if model_type == "bilstm" and run.vocabulary is None:
        raise CheckpointMetadataError("BiLSTM checkpoint metadata requires a training-fitted vocabulary.")
    if model_type == "bert" and run.vocabulary is not None:
        raise CheckpointMetadataError("BERT checkpoint must not contain a BiLSTM vocabulary.")

    split_summary = {
        key: run.training_summary.get(key)
        for key in (
            "rows_per_split", "train_positive_counts", "train_known_counts",
            "train_negative_counts", "split_seed", "grouping_strategy", "source_kind",
            "dataset_id", "approval_reference", "dataset_reference", "split_manifest_fingerprint",
        )
    }
    if not all(isinstance(split_summary.get(key), str) for key in ("dataset_id", "approval_reference")):
        raise CheckpointMetadataError("Approved dataset and review references are required in training metadata.")
    if (
        run.dataset_reference != split_summary.get("dataset_reference")
        or run.split_manifest_fingerprint != split_summary.get("split_manifest_fingerprint")
    ):
        raise CheckpointMetadataError("Checkpoint dataset reference and manifest fingerprint do not match the fitted run.")
    if not isinstance(run.dataset_reference, str) or not _SAFE_OPAQUE_ID.fullmatch(run.dataset_reference):
        raise CheckpointMetadataError("A safe dataset source/version reference is required for checkpointing.")
    if not isinstance(run.split_manifest_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", run.split_manifest_fingerprint
    ):
        raise CheckpointMetadataError("A validated split manifest fingerprint is required for checkpointing.")
    _validate_training_data_summary(split_summary)
    metadata: dict[str, object] = {
        "format_version": 2,
        "model_type": model_type,
        "model_version": model_version,
        "label_order": [label.value for label in EMOTION_LABELS],
        "training_config": training_config.to_dict(),
        "model_config": model_config.to_dict(),
        "configuration_digest": run.configuration_digest,
        "best_epoch": run.best_epoch,
        "validation_history": run.history_dicts(),
        "threshold_selection": run.threshold_selection.to_dict(),
        "evaluation_plan": evaluation_plan.to_dict(),
        "validation_metrics": _sanitize_metrics(run.validation_metrics),
        "training_data_summary": split_summary,
        "vocabulary": run.vocabulary.to_dict() if run.vocabulary is not None else None,
        "tokenizer": _bert_tokenizer_summary(run.bert_tokenizer, model_config)
        if model_type == "bert"
        else None,
        "privacy": (
            "Metadata excludes raw text, participant/group identifiers, API keys, and per-example outputs."
        ),
    }
    if test_metrics is not None:
        if test_metrics.get("evaluation_split") != "test":
            raise CheckpointMetadataError("Test metrics must be an explicitly held-out test report.")
        if test_metrics.get("threshold_source") != "validation":
            raise CheckpointMetadataError("Test metrics must use validation-selected thresholds.")
        if test_metrics.get("configuration_digest") != run.configuration_digest:
            raise CheckpointMetadataError("Test metrics configuration does not match the checkpoint.")
        metadata["test_metrics"] = _sanitize_metrics(test_metrics)
    _validate_checkpoint_metadata(metadata)
    _assert_json_safe(metadata)
    return metadata


def save_checkpoint(
    model: object,
    weights_path: str | Path,
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    """Save state_dict and JSON metadata using temporary files and atomic replacement."""
    payload = dict(metadata)
    unknown_fields = set(payload) - _CHECKPOINT_FIELDS
    if unknown_fields:
        raise CheckpointMetadataError("Checkpoint metadata contains unsupported fields.")
    if payload.get("label_order") != [label.value for label in EMOTION_LABELS]:
        raise CheckpointMetadataError("Checkpoint label order must match the shared schema exactly.")
    if payload.get("privacy") != (
        "Metadata excludes raw text, participant/group identifiers, API keys, and per-example outputs."
    ):
        raise CheckpointMetadataError("Checkpoint metadata privacy declaration is missing or changed.")
    _validate_checkpoint_metadata(payload)
    _assert_json_safe(payload)
    torch = require_torch()
    destination = Path(weights_path)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    weight_temp = _temporary_path(destination.parent, destination.name)
    metadata_temp = _temporary_path(metadata_path.parent, metadata_path.name)
    try:
        torch.save(model.state_dict(), weight_temp)
        payload["weights_sha256"] = _sha256_file(weight_temp)
        _validate_checkpoint_metadata(payload)
        _assert_json_safe(payload)
        metadata_temp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(weight_temp, destination)
        os.replace(metadata_temp, metadata_path)
    finally:
        weight_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return destination, metadata_path


def load_checkpoint(
    model: object,
    weights_path: str | Path,
    *,
    expected_configuration_digest: str,
    expected_model_type: str,
    expected_dataset_reference: str,
    expected_split_manifest_fingerprint: str,
    map_location: str = "cpu",
) -> tuple[object, dict[str, object]]:
    """Verify local checkpoint files and load weights into a caller-built model.

    The caller must build the architecture from the reviewed configuration first.
    PyTorch's restricted weights-only deserializer is required; this function never
    loads arbitrary pickled Python objects. BERT tokenizer assets remain separate and
    must be available locally at their pinned revision.
    """
    if not isinstance(expected_configuration_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_configuration_digest
    ) is None:
        raise CheckpointMetadataError("Expected configuration digest must be SHA-256 hex.")
    if expected_model_type not in {"bilstm", "bert"}:
        raise CheckpointMetadataError("expected_model_type must be 'bilstm' or 'bert'.")
    if not isinstance(expected_dataset_reference, str) or not _SAFE_OPAQUE_ID.fullmatch(expected_dataset_reference):
        raise CheckpointMetadataError("Expected dataset reference must be a safe opaque code.")
    if not isinstance(expected_split_manifest_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_split_manifest_fingerprint
    ):
        raise CheckpointMetadataError("Expected split manifest fingerprint must be SHA-256 hex.")

    weights = Path(weights_path)
    metadata_path = weights.with_suffix(weights.suffix + ".json")
    if not weights.is_file() or not metadata_path.is_file():
        raise CheckpointMetadataError("Checkpoint weights and matching JSON metadata must both exist.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointMetadataError("Checkpoint metadata is missing or invalid JSON.") from exc
    if not isinstance(metadata, dict):
        raise CheckpointMetadataError("Checkpoint metadata must contain a JSON object.")
    if metadata.get("label_order") != [label.value for label in EMOTION_LABELS]:
        raise CheckpointMetadataError("Checkpoint label order must match the shared schema exactly.")
    _validate_checkpoint_metadata(metadata)
    if metadata.get("configuration_digest") != expected_configuration_digest:
        raise CheckpointMetadataError("Checkpoint configuration does not match the requested model configuration.")
    if metadata.get("model_type") != expected_model_type:
        raise CheckpointMetadataError("Checkpoint model type does not match the requested architecture.")
    training_summary = metadata.get("training_data_summary")
    if (
        training_summary.get("dataset_reference") != expected_dataset_reference
        or training_summary.get("split_manifest_fingerprint") != expected_split_manifest_fingerprint
    ):
        raise CheckpointMetadataError("Checkpoint dataset reference or split manifest does not match the requested record.")
    expected_weights_hash = metadata.get("weights_sha256")
    if not isinstance(expected_weights_hash, str):
        raise CheckpointMetadataError("Checkpoint metadata has no weights checksum.")
    if _sha256_file(weights) != expected_weights_hash:
        raise CheckpointMetadataError("Checkpoint weights checksum does not match metadata.")

    torch = require_torch()
    try:
        state = torch.load(weights, map_location=map_location, weights_only=True)
        if not isinstance(state, Mapping):
            raise CheckpointMetadataError("Checkpoint weights must contain a state-dict mapping.")
        model.load_state_dict(state, strict=True)
    except CheckpointMetadataError:
        raise
    except Exception as exc:
        raise CheckpointMetadataError("Checkpoint weights could not be loaded into this model.") from exc
    return model, metadata


def _bert_tokenizer_summary(tokenizer: object | None, model_config: object) -> dict[str, object]:
    if tokenizer is None or not isinstance(model_config, BertModelConfig):
        raise CheckpointMetadataError("BERT checkpoint requires its tokenizer and reviewed checkpoint config.")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise CheckpointMetadataError("Tokenizer must expose a numeric vocab_size.")
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "checkpoint_id": model_config.checkpoint_id,
        "revision": model_config.revision,
        "max_sequence_length": model_config.max_sequence_length,
        "local_files_only": True,
    }


def _validate_training_data_summary(summary: Mapping[str, object]) -> None:
    expected = {
        "rows_per_split", "train_positive_counts", "train_known_counts", "train_negative_counts",
        "split_seed", "grouping_strategy", "source_kind", "dataset_id", "approval_reference",
        "dataset_reference", "split_manifest_fingerprint",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected:
        raise CheckpointMetadataError("Training data summary has unsupported or missing fields.")
    source_kind = summary.get("source_kind")
    if not isinstance(source_kind, str) or source_kind not in ALLOWED_SOURCE_KINDS:
        raise CheckpointMetadataError("Checkpoint data source is not an approved source kind.")
    for field in ("dataset_id", "approval_reference", "dataset_reference"):
        value = summary.get(field)
        if not isinstance(value, str) or _SAFE_OPAQUE_ID.fullmatch(value) is None:
            raise CheckpointMetadataError(f"Checkpoint {field} must be an opaque non-identifying code.")
    manifest_fingerprint = summary.get("split_manifest_fingerprint")
    if not isinstance(manifest_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_fingerprint) is None:
        raise CheckpointMetadataError("Checkpoint split manifest fingerprint must be SHA-256 hex.")
    grouping_strategy = summary.get("grouping_strategy")
    if not isinstance(grouping_strategy, str) or grouping_strategy not in _GROUPING_STRATEGIES:
        raise CheckpointMetadataError("Checkpoint grouping strategy is not a supported split strategy.")
    seed = summary.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CheckpointMetadataError("Checkpoint split seed must be a non-negative integer.")
    row_counts = summary.get("rows_per_split")
    if not isinstance(row_counts, Mapping) or set(row_counts) != {"train", "validation", "test"}:
        raise CheckpointMetadataError("Checkpoint must record counts for each split.")
    for count in row_counts.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CheckpointMetadataError("Checkpoint split counts must be positive integers.")
    canonical = {label.value for label in EMOTION_LABELS}
    for key in ("train_positive_counts", "train_known_counts", "train_negative_counts"):
        counts = summary.get(key)
        if not isinstance(counts, Mapping) or set(counts) != canonical:
            raise CheckpointMetadataError("Checkpoint training label counts must match all five labels.")
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise CheckpointMetadataError("Checkpoint label counts must be non-negative integers.")


def _validate_checkpoint_metadata(payload: Mapping[str, object]) -> None:
    required = {
        "format_version", "model_type", "model_version", "label_order", "training_config",
        "model_config", "configuration_digest", "best_epoch", "validation_history",
        "threshold_selection", "evaluation_plan", "validation_metrics", "training_data_summary",
        "vocabulary", "tokenizer", "privacy",
    }
    optional = {"test_metrics", "weights_sha256"}
    if not required.issubset(payload) or set(payload) - required - optional:
        raise CheckpointMetadataError("Checkpoint metadata fields do not match the supported format.")
    if payload.get("format_version") != 2:
        raise CheckpointMetadataError("Unsupported checkpoint metadata version.")
    if payload.get("label_order") != [label.value for label in EMOTION_LABELS]:
        raise CheckpointMetadataError("Checkpoint label order must match the shared schema exactly.")
    if payload.get("model_type") not in {"bilstm", "bert"}:
        raise CheckpointMetadataError("Checkpoint model_type is invalid.")
    if not isinstance(payload.get("model_version"), str) or _SAFE_MODEL_VERSION.fullmatch(payload["model_version"]) is None:
        raise CheckpointMetadataError("Checkpoint model_version must be a safe version code.")
    digest = payload.get("configuration_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise CheckpointMetadataError("Checkpoint configuration digest must be SHA-256 hex.")
    epoch = payload.get("best_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CheckpointMetadataError("Checkpoint best_epoch must be positive.")

    training_config = payload.get("training_config")
    expected_training = {
        "seed", "batch_size", "epochs", "learning_rate", "weight_decay",
        "max_sequence_length", "device", "task_type", "label_order",
    }
    if not isinstance(training_config, Mapping) or set(training_config) != expected_training:
        raise CheckpointMetadataError("Training configuration fields do not match the supported schema.")
    if training_config.get("label_order") != [label.value for label in EMOTION_LABELS]:
        raise CheckpointMetadataError("Training configuration label order is inconsistent.")
    if training_config.get("device") not in {"auto", "cpu", "cuda"} or training_config.get("task_type") != "multi_label":
        raise CheckpointMetadataError("Training configuration has unsupported device or task settings.")
    for key in ("seed", "batch_size", "epochs", "max_sequence_length"):
        value = training_config.get(key)
        minimum = 0 if key == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CheckpointMetadataError("Training configuration contains an invalid integer setting.")
    for key in ("learning_rate", "weight_decay"):
        value = training_config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CheckpointMetadataError("Training configuration contains an invalid numeric setting.")
    if training_config["learning_rate"] <= 0 or training_config["weight_decay"] < 0:
        raise CheckpointMetadataError("Training configuration learning settings are outside valid ranges.")

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise CheckpointMetadataError("model_config must be a mapping.")
    if payload["model_type"] == "bilstm":
        expected_model = {
            "embedding_dim", "hidden_size", "num_layers", "dropout",
            "max_vocabulary_size", "min_token_frequency", "hash_space_size",
        }
        if set(model_config) != expected_model or payload.get("tokenizer") is not None:
            raise CheckpointMetadataError("BiLSTM configuration or tokenizer metadata is inconsistent.")
        for key in (
            "embedding_dim", "hidden_size", "num_layers", "max_vocabulary_size",
            "min_token_frequency", "hash_space_size",
        ):
            value = model_config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CheckpointMetadataError("BiLSTM configuration contains an invalid integer setting.")
        dropout = model_config.get("dropout")
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
            raise CheckpointMetadataError("BiLSTM dropout is invalid.")
        vocabulary = payload.get("vocabulary")
        _validate_vocabulary(vocabulary)
    else:
        expected_model = {
            "checkpoint_id", "revision", "max_sequence_length", "license_reviewed",
            "language_fit_reviewed", "resource_requirements_reviewed", "local_files_only",
        }
        if set(model_config) != expected_model or payload.get("vocabulary") is not None:
            raise CheckpointMetadataError("BERT configuration or vocabulary metadata is inconsistent.")
        checkpoint_id = model_config.get("checkpoint_id")
        revision = model_config.get("revision")
        if not isinstance(checkpoint_id, str) or _SAFE_CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
            raise CheckpointMetadataError("BERT checkpoint identifier is not a safe repository ID.")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None:
            raise CheckpointMetadataError("BERT revision must be a pinned commit hash.")
        for key in ("license_reviewed", "language_fit_reviewed", "resource_requirements_reviewed"):
            if model_config.get(key) is not True:
                raise CheckpointMetadataError("BERT checkpoint reviews are incomplete.")
        if model_config.get("local_files_only") is not True:
            raise CheckpointMetadataError("BERT checkpoint loading must remain local-only.")
        sequence_length = model_config.get("max_sequence_length")
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 2:
            raise CheckpointMetadataError("BERT max_sequence_length is invalid.")
        _validate_tokenizer_summary(payload.get("tokenizer"), model_config)

    if _configuration_digest_from_metadata(training_config, model_config) != digest:
        raise CheckpointMetadataError("Checkpoint configuration digest does not match its stored configurations.")

    thresholds = payload.get("threshold_selection")
    plan = payload.get("evaluation_plan")
    canonical = {label.value for label in EMOTION_LABELS}
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "selection_split", "thresholds", "validation_f1", "status"
    } or thresholds.get("selection_split") != "validation":
        raise CheckpointMetadataError("Threshold selection must identify validation as its source.")
    if any(not isinstance(thresholds.get(key), Mapping) or set(thresholds[key]) != canonical for key in ("thresholds", "validation_f1", "status")):
        raise CheckpointMetadataError("Selected thresholds must cover the canonical labels.")
    if any(
        isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < float(value) <= 1
        for value in thresholds["thresholds"].values()
    ):
        raise CheckpointMetadataError("Checkpoint thresholds must be numeric and finalized.")
    if any(value is not None and (
        isinstance(value, bool) or not isinstance(value, (float, int))
        or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
    ) for value in thresholds["validation_f1"].values()):
        raise CheckpointMetadataError("Validation F1 values must be finite numbers or null.")
    if any(value not in {"selected_on_validation", "insufficient_validation_class_coverage"} for value in thresholds["status"].values()):
        raise CheckpointMetadataError("Threshold selection status is invalid.")
    if any(value != "selected_on_validation" for value in thresholds["status"].values()):
        raise CheckpointMetadataError("All five thresholds must be selected before checkpoint finalization.")
    if not isinstance(plan, Mapping) or set(plan) != {
        "threshold_source", "configuration_digest", "thresholds"
    } or plan.get("threshold_source") != "validation":
        raise CheckpointMetadataError("Finalized evaluation plan must use validation thresholds.")
    if plan.get("configuration_digest") != digest or plan.get("thresholds") != thresholds["thresholds"]:
        raise CheckpointMetadataError("Finalized evaluation plan does not match checkpoint config/thresholds.")

    history = payload.get("validation_history")
    if not isinstance(history, list) or not history:
        raise CheckpointMetadataError("Validation history is required.")
    for row in history:
        if not isinstance(row, Mapping) or set(row) != {"epoch", "train_loss", "validation_loss"}:
            raise CheckpointMetadataError("Validation history contains unsupported fields.")
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                raise CheckpointMetadataError("Validation history must contain finite numeric values only.")
        if isinstance(row["epoch"], bool) or not isinstance(row["epoch"], int) or row["epoch"] < 1:
            raise CheckpointMetadataError("Validation history epoch values must be positive integers.")
    if payload.get("validation_metrics") != _sanitize_metrics(payload.get("validation_metrics")):
        raise CheckpointMetadataError("Validation metrics contain unsupported fields or values.")
    if "test_metrics" in payload:
        if payload["test_metrics"] != _sanitize_metrics(payload["test_metrics"]):
            raise CheckpointMetadataError("Test metrics contain unsupported fields or values.")
    _validate_training_data_summary(payload.get("training_data_summary"))
    if "weights_sha256" in payload and (
        not isinstance(payload["weights_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["weights_sha256"]) is None
    ):
        raise CheckpointMetadataError("weights_sha256 must be a SHA-256 hex digest.")


def _validate_vocabulary(value: object) -> None:
    expected = {"format", "hash_space_size", "bucket_ids", "pad_id", "unknown_id", "vocabulary_size"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("format") != "sha256_hashed_training_bucket_v1":
        raise CheckpointMetadataError("BiLSTM checkpoint requires hashed training vocabulary metadata.")
    buckets = value.get("bucket_ids")
    size = value.get("hash_space_size")
    if not isinstance(buckets, list) or any(
        isinstance(bucket, bool) or not isinstance(bucket, int) for bucket in buckets
    ):
        raise CheckpointMetadataError("Vocabulary bucket metadata must be integers only.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise CheckpointMetadataError("Vocabulary hash space must be a positive integer.")
    if value.get("pad_id") != 0 or value.get("unknown_id") != 1 or value.get("vocabulary_size") != len(buckets) + 2:
        raise CheckpointMetadataError("Vocabulary size or special token IDs are inconsistent.")
    if len(set(buckets)) != len(buckets) or any(bucket < 0 or bucket >= size for bucket in buckets):
        raise CheckpointMetadataError("Vocabulary bucket IDs are invalid.")


def _validate_tokenizer_summary(value: object, model_config: Mapping[str, object]) -> None:
    expected = {
        "class", "vocab_size", "checkpoint_id", "revision", "max_sequence_length", "local_files_only",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CheckpointMetadataError("BERT tokenizer summary does not match the supported schema.")
    tokenizer_class = value.get("class")
    vocab_size = value.get("vocab_size")
    if not isinstance(tokenizer_class, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", tokenizer_class) is None:
        raise CheckpointMetadataError("Tokenizer class name is invalid.")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise CheckpointMetadataError("Tokenizer vocabulary size must be positive.")
    for key in ("checkpoint_id", "revision", "max_sequence_length", "local_files_only"):
        if value.get(key) != model_config.get(key):
            raise CheckpointMetadataError("Tokenizer summary does not match its reviewed model config.")


def _sanitize_metrics(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointMetadataError("Metrics must be a mapping.")
    fields = set(value)
    is_serialized = fields == _CHECKPOINT_METRIC_KEYS
    is_report = (
        _RAW_METRIC_REQUIRED_KEYS.issubset(fields)
        and fields <= (_RAW_METRIC_REQUIRED_KEYS | _RAW_METRIC_CONTEXT_KEYS)
    )
    if not (is_serialized or is_report):
        raise CheckpointMetadataError("Metrics do not match a complete supported report schema.")

    if is_report:
        if value.get("label_order") != [label.value for label in EMOTION_LABELS]:
            raise CheckpointMetadataError("Metric report label order differs from the shared schema.")
        _validate_metric_thresholds(value.get("thresholds"))
        context_fields = fields & _RAW_METRIC_CONTEXT_KEYS
        if context_fields and context_fields != _RAW_METRIC_CONTEXT_KEYS:
            raise CheckpointMetadataError("Metric report evaluation context is incomplete.")
        if context_fields:
            if (
                value.get("evaluation_split") != "test"
                or value.get("threshold_source") != "validation"
                or not isinstance(value.get("configuration_digest"), str)
                or re.fullmatch(r"[0-9a-f]{64}", value["configuration_digest"]) is None
            ):
                raise CheckpointMetadataError("Metric report evaluation context is invalid.")

    sanitized: dict[str, object] = {}
    for key in _VALID_METRIC_KEYS:
        sanitized[key] = _metric_ratio_or_none(value[key], key) if key in {
            "macro_f1", "micro_f1", "subset_accuracy", "hamming_loss"
        } else value[key]

    per_label = value["per_label"]
    canonical_labels = {label.value for label in EMOTION_LABELS}
    if not isinstance(per_label, Mapping) or set(per_label) != canonical_labels:
        raise CheckpointMetadataError("Per-label metrics must contain exactly the five canonical labels.")

    clean_labels: dict[str, object] = {}
    total_tp = total_fp = total_fn = total_known = 0
    defined_f1: list[float] = []
    for label in EMOTION_LABELS:
        row = per_label[label.value]
        if not isinstance(row, Mapping) or set(row) != _VALID_PER_LABEL_KEYS:
            raise CheckpointMetadataError("Per-label metrics do not match the complete supported schema.")

        support = _nonnegative_integer(row["support"], "support")
        known_count = _nonnegative_integer(row["known_count"], "known_count")
        matrix = _sanitize_confusion_matrix(row["confusion_matrix"])
        tn, fp = matrix["counts"][0]
        fn, tp = matrix["counts"][1]
        if tn + fp + fn + tp != known_count:
            raise CheckpointMetadataError("Confusion matrix total differs from known_count.")
        if fn + tp != support:
            raise CheckpointMetadataError("Confusion matrix positive support differs from support.")

        expected_precision = tp / (tp + fp) if tp + fp else None
        expected_recall = tp / support if support else None
        f1_denominator = 2 * tp + fp + fn
        expected_f1 = 2 * tp / f1_denominator if f1_denominator else None
        precision = _metric_ratio_or_none(row["precision"], "precision")
        recall = _metric_ratio_or_none(row["recall"], "recall")
        f1 = _metric_ratio_or_none(row["f1"], "f1")
        _require_metric_match(precision, expected_precision, "precision")
        _require_metric_match(recall, expected_recall, "recall")
        _require_metric_match(f1, expected_f1, "F1")

        expected_undefined = "no known positive examples" if support == 0 else None
        expected_precision_undefined = "no predicted positives" if tp + fp == 0 else None
        if row["undefined"] != expected_undefined:
            raise CheckpointMetadataError("Per-label undefined reason disagrees with its support.")
        if row["precision_undefined"] != expected_precision_undefined:
            raise CheckpointMetadataError("Per-label precision undefined reason disagrees with predictions.")

        clean_labels[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "known_count": known_count,
            "undefined": expected_undefined,
            "precision_undefined": expected_precision_undefined,
            "confusion_matrix": matrix,
        }
        if f1 is not None:
            defined_f1.append(float(f1))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_known += known_count

    macro_f1_defined_labels = _nonnegative_integer(
        value["macro_f1_defined_labels"], "macro_f1_defined_labels"
    )
    if macro_f1_defined_labels != len(defined_f1) or macro_f1_defined_labels > len(EMOTION_LABELS):
        raise CheckpointMetadataError("macro_f1_defined_labels disagrees with per-label metrics.")
    known_label_decisions = _nonnegative_integer(
        value["known_label_decisions"], "known_label_decisions"
    )
    if known_label_decisions != total_known:
        raise CheckpointMetadataError("known_label_decisions disagrees with per-label known counts.")
    subset_accuracy_rows = _nonnegative_integer(value["subset_accuracy_rows"], "subset_accuracy_rows")
    if subset_accuracy_rows == 0:
        if value["subset_accuracy"] is not None:
            raise CheckpointMetadataError("subset_accuracy must be null when there are no fully known rows.")
    else:
        subset_accuracy = _metric_ratio_or_none(value["subset_accuracy"], "subset_accuracy")
        if subset_accuracy is None:
            raise CheckpointMetadataError("subset_accuracy must be defined when fully known rows exist.")
        if subset_accuracy_rows * len(EMOTION_LABELS) > known_label_decisions:
            raise CheckpointMetadataError("subset_accuracy_rows exceeds the fully known decision count.")

    micro_denominator = 2 * total_tp + total_fp + total_fn
    expected_micro = 2 * total_tp / micro_denominator if micro_denominator else None
    expected_macro = sum(defined_f1) / len(defined_f1) if defined_f1 else None
    expected_hamming = (total_fp + total_fn) / total_known if total_known else None
    _require_metric_match(_metric_ratio_or_none(value["macro_f1"], "macro_f1"), expected_macro, "macro_f1")
    _require_metric_match(_metric_ratio_or_none(value["micro_f1"], "micro_f1"), expected_micro, "micro_f1")
    _require_metric_match(_metric_ratio_or_none(value["hamming_loss"], "hamming_loss"), expected_hamming, "hamming_loss")

    sanitized["macro_f1_defined_labels"] = macro_f1_defined_labels
    sanitized["subset_accuracy_rows"] = subset_accuracy_rows
    sanitized["known_label_decisions"] = known_label_decisions
    sanitized["per_label"] = clean_labels
    return sanitized


def _sanitize_confusion_matrix(value: object) -> dict[str, object]:
    """Validate the binary matrix encoding: actual rows, predicted columns."""
    if not isinstance(value, Mapping) or set(value) != {
        "actual_labels", "predicted_labels", "counts"
    }:
        raise CheckpointMetadataError("Per-label confusion matrix has an invalid schema.")
    if value.get("actual_labels") != ["negative", "positive"] or value.get(
        "predicted_labels"
    ) != ["negative", "positive"]:
        raise CheckpointMetadataError("Confusion matrix label order is invalid.")
    counts = value.get("counts")
    if not isinstance(counts, list) or len(counts) != 2:
        raise CheckpointMetadataError("Confusion matrix counts must be a 2-by-2 list.")
    clean_counts: list[list[int]] = []
    for row in counts:
        if not isinstance(row, list) or len(row) != 2:
            raise CheckpointMetadataError("Confusion matrix counts must be a 2-by-2 list.")
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in row):
            raise CheckpointMetadataError("Confusion matrix counts must be non-negative integers.")
        clean_counts.append(list(row))
    return {
        "actual_labels": ["negative", "positive"],
        "predicted_labels": ["negative", "positive"],
        "counts": clean_counts,
    }


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointMetadataError(f"{field} must be a non-negative integer.")
    return value


def _metric_ratio_or_none(value: object, field: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise CheckpointMetadataError(f"{field} must be a number in [0, 1] or null.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CheckpointMetadataError(f"{field} must be a finite number in [0, 1] or null.")
    return int(value) if isinstance(value, int) else number


def _require_metric_match(
    actual: float | int | None,
    expected: float | None,
    field: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise CheckpointMetadataError(f"{field} must be null under the documented undefined-metric convention.")
    elif actual is None or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise CheckpointMetadataError(f"{field} disagrees with confusion-matrix counts.")


def _validate_metric_thresholds(value: object) -> None:
    canonical_labels = {label.value for label in EMOTION_LABELS}
    if not isinstance(value, Mapping) or set(value) != canonical_labels:
        raise CheckpointMetadataError("Metric report thresholds must cover the canonical labels.")
    for threshold in value.values():
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (float, int))
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) <= 1.0
        ):
            raise CheckpointMetadataError("Metric report thresholds must be finite numbers in (0, 1].")


def _assert_json_safe(value: object) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointMetadataError("Checkpoint metadata must contain finite JSON-safe values only.") from exc


def _temporary_path(directory: Path, filename: str) -> Path:
    handle, name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    os.close(handle)
    return Path(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_digest_from_metadata(
    training_config: Mapping[str, object],
    model_config: Mapping[str, object],
) -> str:
    encoded = json.dumps(
        {"training": training_config, "model": model_config},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

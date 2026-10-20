"""Immutable, resource-conscious configuration for model training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re

from ..prediction.schema import EMOTION_LABELS


CANONICAL_LABEL_ORDER: tuple[str, ...] = tuple(label.value for label in EMOTION_LABELS)
SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda"})
_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?\Z")
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}\Z")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Shared settings; training sees train/validation, and test is a final-only stage."""

    seed: int = 42
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_sequence_length: int = 128
    device: str = "cpu"
    checkpoint_dir: Path = Path("artifacts/models")
    task_type: str = "multi_label"
    label_order: tuple[str, ...] = field(default_factory=lambda: CANONICAL_LABEL_ORDER)

    def __post_init__(self) -> None:
        for name in ("seed", "batch_size", "epochs", "max_sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            invalid = value < 0 if name == "seed" else value < 1
            if invalid:
                raise ValueError(f"{name} must be {'non-negative' if name == 'seed' else 'positive'}.")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or (value <= 0 if name == "learning_rate" else value < 0):
                raise ValueError(f"{name} has an invalid value.")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"device must be one of {sorted(SUPPORTED_DEVICES)}.")
        if self.task_type != "multi_label":
            raise ValueError("Only the five-label multi_label task is supported.")
        if tuple(self.label_order) != CANONICAL_LABEL_ORDER:
            raise ValueError("label_order must exactly match the shared canonical label order.")
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    @classmethod
    def for_bert(cls, **overrides: object) -> "TrainingConfig":
        """Return a conservative BERT fine-tuning starting point, not an optimum."""
        values: dict[str, object] = {"learning_rate": 2e-5, "batch_size": 4, "epochs": 3}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        # Output destinations are operational paths and can reveal usernames; they do
        # not affect model behavior and are omitted from digests/checkpoint metadata.
        result.pop("checkpoint_dir", None)
        result["label_order"] = list(self.label_order)
        return result


@dataclass(frozen=True, slots=True)
class BiLSTMModelConfig:
    """Small trainable embedding and bidirectional LSTM starting defaults."""

    embedding_dim: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    max_vocabulary_size: int = 20_000
    min_token_frequency: int = 1
    hash_space_size: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim", "hidden_size", "num_layers", "max_vocabulary_size",
            "min_token_frequency", "hash_space_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_vocabulary_size < 3:
            raise ValueError("max_vocabulary_size must leave room for padding, unknown, and one token.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be numeric.")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1).")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BertModelConfig:
    """BERT-family configuration with explicit checkpoint review gates.

    No default checkpoint is selected. Loading is local-only so training setup cannot
    silently fetch model files; acquisition requires a separately approved step.
    """

    checkpoint_id: str | None = None
    revision: str | None = None
    max_sequence_length: int = 128
    license_reviewed: bool = False
    language_fit_reviewed: bool = False
    resource_requirements_reviewed: bool = False
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_sequence_length, bool) or not isinstance(self.max_sequence_length, int):
            raise TypeError("max_sequence_length must be an integer.")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2.")
        if self.local_files_only is not True:
            raise ValueError("BERT checkpoint loading is local-only in this infrastructure.")

    def validate_checkpoint_review(self) -> None:
        if not isinstance(self.checkpoint_id, str) or _CHECKPOINT_ID.fullmatch(self.checkpoint_id) is None:
            raise ValueError("Set a reviewed BERT-family repository checkpoint_id before loading a model.")
        if not isinstance(self.revision, str) or _PINNED_REVISION.fullmatch(self.revision) is None:
            raise ValueError("Pin the reviewed checkpoint to a 40-64 character commit hash.")
        if self.license_reviewed is not True:
            raise ValueError("Review the checkpoint license before loading a model.")
        if self.language_fit_reviewed is not True:
            raise ValueError("Review checkpoint language fit before loading a model.")
        if self.resource_requirements_reviewed is not True:
            raise ValueError("Review checkpoint resource requirements before loading a model.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configuration_digest(training: TrainingConfig, model: object) -> str:
    """Stable digest used to bind validation thresholds to finalized config."""
    model_data = model.to_dict() if hasattr(model, "to_dict") else model
    payload = {"training": training.to_dict(), "model": model_data}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

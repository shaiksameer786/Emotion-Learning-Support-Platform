"""Dataset preparation and optional, approval-gated model training infrastructure."""

from .config import BertModelConfig, BiLSTMModelConfig, TrainingConfig
from .checkpoints import CheckpointMetadataError, load_checkpoint, save_checkpoint
from .data import TrainingAuthorization, TrainingDataError, encode_multihot, validate_training_splits
from .engine import TrainingRunResult, evaluate_model_on_test, fit_bert, fit_bilstm
from .dataset import (
    DatasetManifestError,
    PreparedSplitBundle,
    load_training_splits,
    read_prepared_split_files,
    write_training_manifest,
)
from .metrics import (
    EvaluationError,
    FinalizedEvaluationPlan,
    calculate_metrics,
    evaluate_heldout_test,
    finalize_evaluation_plan,
    select_thresholds,
)

__all__ = [
    "BertModelConfig",
    "BiLSTMModelConfig",
    "CheckpointMetadataError",
    "DatasetManifestError",
    "EvaluationError",
    "FinalizedEvaluationPlan",
    "PreparedSplitBundle",
    "TrainingAuthorization",
    "TrainingConfig",
    "TrainingDataError",
    "TrainingRunResult",
    "calculate_metrics",
    "encode_multihot",
    "evaluate_model_on_test",
    "evaluate_heldout_test",
    "finalize_evaluation_plan",
    "fit_bert",
    "fit_bilstm",
    "load_checkpoint",
    "load_training_splits",
    "read_prepared_split_files",
    "save_checkpoint",
    "select_thresholds",
    "validate_training_splits",
    "write_training_manifest",
]

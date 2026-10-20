"""Reproducible fit loops that use training and validation data only."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Callable, Mapping, Sequence

from ..preprocessing import PreparedExample
from ..prediction.schema import EMOTION_LABELS
from .config import (
    BertModelConfig,
    BiLSTMModelConfig,
    TrainingConfig,
    configuration_digest,
)
from .data import TrainingAuthorization, encode_multihot, validate_training_splits
from .metrics import ThresholdSelection, calculate_metrics, select_thresholds
from .metrics import FinalizedEvaluationPlan, evaluate_heldout_test
from .models import build_bilstm_model, load_bert_components
from .optional import require_torch
from .splitting import DatasetSplits
from .vocabulary import HashedVocabulary


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """In-memory model and privacy-safe validation outputs; contains no learner text."""

    model: object
    history: tuple[EpochRecord, ...]
    best_epoch: int
    threshold_selection: ThresholdSelection
    validation_metrics: Mapping[str, object]
    configuration_digest: str
    training_summary: Mapping[str, object]
    split_fingerprint: str | None = None
    dataset_reference: str | None = None
    split_manifest_fingerprint: str | None = None
    vocabulary: HashedVocabulary | None = None
    bert_tokenizer: object | None = None

    def history_dicts(self) -> list[dict[str, float | int]]:
        return [
            {
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "validation_loss": record.validation_loss,
            }
            for record in self.history
        ]


BatchEncoder = Callable[[Sequence[PreparedExample]], Mapping[str, object]]


def fit_bilstm(
    splits: DatasetSplits,
    authorization: TrainingAuthorization,
    *,
    training_config: TrainingConfig | None = None,
    model_config: BiLSTMModelConfig | None = None,
) -> TrainingRunResult:
    """Fit a small BiLSTM after explicit source and split validation."""
    training = training_config or TrainingConfig()
    model_settings = model_config or BiLSTMModelConfig()
    summary = validate_training_splits(splits, authorization)
    vocabulary = HashedVocabulary.fit_training_split(splits, model_settings)
    model = build_bilstm_model(vocabulary.size, model_settings)

    def encode_batch(examples: Sequence[PreparedExample]) -> Mapping[str, object]:
        sequences = [
            vocabulary.encode(example.text, max_length=training.max_sequence_length)
            for example in examples
        ]
        return {"input_ids": sequences}

    return _fit(
        model,
        splits,
        training,
        model_settings,
        encode_batch,
        summary,
        vocabulary=vocabulary,
    )


def fit_bert(
    splits: DatasetSplits,
    authorization: TrainingAuthorization,
    *,
    bert_config: BertModelConfig,
    training_config: TrainingConfig | None = None,
) -> TrainingRunResult:
    """Fine-tune a reviewed, pinned BERT-family checkpoint from local files only."""
    training = training_config or TrainingConfig.for_bert()
    if training.max_sequence_length != bert_config.max_sequence_length:
        raise ValueError("TrainingConfig and BertModelConfig must use the same maximum sequence length.")
    summary = validate_training_splits(splits, authorization)
    components = load_bert_components(bert_config)
    tokenizer = components.tokenizer

    def encode_batch(examples: Sequence[PreparedExample]) -> Mapping[str, object]:
        encoded = tokenizer(
            [example.text for example in examples],
            truncation=True,
            max_length=bert_config.max_sequence_length,
            padding="max_length",
            return_tensors="pt",
        )
        return dict(encoded)

    return _fit(
        components.model,
        splits,
        training,
        bert_config,
        encode_batch,
        summary,
        bert_tokenizer=tokenizer,
    )


def evaluate_model_on_test(
    run: TrainingRunResult,
    splits: DatasetSplits,
    authorization: TrainingAuthorization,
    *,
    training_config: TrainingConfig,
    model_config: BiLSTMModelConfig | BertModelConfig,
    evaluation_plan: FinalizedEvaluationPlan,
) -> dict[str, object]:
    """Run the untouched test partition after validation thresholds/config are finalized."""
    if not isinstance(evaluation_plan, FinalizedEvaluationPlan):
        raise TypeError("A finalized validation-based evaluation plan is required before test scoring.")
    current_digest = configuration_digest(training_config, model_config)
    if current_digest != run.configuration_digest:
        raise ValueError("Model configuration changed after fitting; retrain and tune on validation first.")
    if current_digest != evaluation_plan.configuration_digest:
        raise ValueError("Test scoring requires the same finalized configuration digest.")
    if tuple(run.threshold_selection.thresholds) != tuple(evaluation_plan.thresholds):
        raise ValueError("Test scoring thresholds differ from validation-selected thresholds.")
    validate_training_splits(splits, authorization)
    if run.split_fingerprint is None or run.split_fingerprint != splits.split_fingerprint:
        raise ValueError("Test scoring requires the exact train/validation/test partitions used for fitting.")
    if (
        run.dataset_reference != splits.dataset_reference
        or run.split_manifest_fingerprint != splits.split_manifest_fingerprint
    ):
        raise ValueError("Test scoring split manifest differs from the one used for fitting.")
    for field_name in ("source_kind", "dataset_id", "approval_reference"):
        if run.training_summary.get(field_name) != getattr(authorization, field_name):
            raise ValueError("Test scoring authorization does not match the fitted dataset review.")

    if run.vocabulary is not None:
        if not isinstance(model_config, BiLSTMModelConfig) or run.bert_tokenizer is not None:
            raise ValueError("Training result and BiLSTM configuration do not match.")

        def encode_batch(examples: Sequence[PreparedExample]) -> Mapping[str, object]:
            return {
                "input_ids": [
                    run.vocabulary.encode(example.text, max_length=training_config.max_sequence_length)
                    for example in examples
                ]
            }

    elif run.bert_tokenizer is not None:
        if not isinstance(model_config, BertModelConfig):
            raise ValueError("Training result and BERT configuration do not match.")

        def encode_batch(examples: Sequence[PreparedExample]) -> Mapping[str, object]:
            encoded = run.bert_tokenizer(
                [example.text for example in examples],
                truncation=True,
                max_length=model_config.max_sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            return dict(encoded)

    else:
        raise ValueError("Training result does not include model tokenizer metadata.")

    torch = require_torch()
    device = _resolve_device(training_config.device, torch)
    model = run.model.to(device)
    model.eval()
    targets: list[list[int]] = []
    probabilities: list[list[float]] = []
    masks: list[list[int]] = []
    with torch.no_grad():
        for start in range(0, len(splits.test), training_config.batch_size):
            examples = splits.test[start : start + training_config.batch_size]
            features = _encode_tensors(encode_batch(examples), torch, device)
            batch_targets, batch_mask = _targets_tensors(examples, torch, device)
            logits = model(**features)
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().to(torch.int64).tolist())
            masks.extend(batch_mask.detach().cpu().to(torch.int64).tolist())
    return evaluate_heldout_test(
        targets,
        probabilities,
        masks,
        plan=evaluation_plan,
        configuration_digest=current_digest,
        split_name="test",
    )


def _fit(
    model,
    splits: DatasetSplits,
    config: TrainingConfig,
    model_config: object,
    encode_batch: BatchEncoder,
    training_summary: Mapping[str, object],
    *,
    vocabulary: HashedVocabulary | None = None,
    bert_tokenizer: object | None = None,
) -> TrainingRunResult:
    torch = require_torch()
    _seed_everything(config.seed, torch)
    device = _resolve_device(config.device, torch)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[EpochRecord] = []
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    best_validation: tuple[list[list[int]], list[list[float]], list[list[int]]] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = list(range(len(splits.train)))
        random.Random(config.seed + epoch).shuffle(order)
        train_sum = 0.0
        train_known = 0
        for indices in _batches(order, config.batch_size):
            examples = [splits.train[index] for index in indices]
            features = _encode_tensors(encode_batch(examples), torch, device)
            targets, mask = _targets_tensors(examples, torch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**features)
            loss, loss_sum, known_count = _masked_loss(logits, targets, mask, torch)
            loss.backward()
            optimizer.step()
            train_sum += loss_sum
            train_known += known_count

        validation_loss, validation_rows = _run_validation(
            model, splits.validation, encode_batch, config.batch_size, torch, device
        )
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_sum / train_known if train_known else float("nan"),
                validation_loss=validation_loss,
            )
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_validation = validation_rows

    if best_state is None or best_validation is None:
        raise RuntimeError("Training produced no usable validation checkpoint.")
    model.load_state_dict(best_state)
    targets, scores, known_mask = best_validation
    threshold_selection = select_thresholds(
        targets, scores, known_mask, split_name="validation"
    )
    thresholds = {
        label.value: threshold
        for label, threshold in zip(EMOTION_LABELS, threshold_selection.thresholds)
        if threshold is not None
    }
    validation_metrics = (
        calculate_metrics(targets, scores, known_mask, thresholds)
        if len(thresholds) == len(threshold_selection.thresholds)
        else {
            "message": "Per-label validation thresholds are incomplete; metrics requiring a full threshold set were not calculated.",
            "threshold_selection": threshold_selection.to_dict(),
        }
    )
    return TrainingRunResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        threshold_selection=threshold_selection,
        validation_metrics=validation_metrics,
        configuration_digest=configuration_digest(config, model_config),
        training_summary=dict(training_summary),
        split_fingerprint=splits.split_fingerprint,
        dataset_reference=splits.dataset_reference,
        split_manifest_fingerprint=splits.split_manifest_fingerprint,
        vocabulary=vocabulary,
        bert_tokenizer=bert_tokenizer,
    )


def _run_validation(model, examples, encode_batch, batch_size, torch, device):
    model.eval()
    rows: list[tuple[list[int], list[float], list[int]]] = []
    weighted_loss = 0.0
    known_total = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            features = _encode_tensors(encode_batch(batch), torch, device)
            targets, mask = _targets_tensors(batch, torch, device)
            logits = model(**features)
            _, loss_sum, known_count = _masked_loss(logits, targets, mask, torch)
            weighted_loss += loss_sum
            known_total += known_count
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            target_rows = targets.detach().cpu().to(torch.int64).tolist()
            mask_rows = mask.detach().cpu().to(torch.int64).tolist()
            rows.extend(zip(target_rows, probabilities, mask_rows))
    if not known_total:
        raise ValueError("Validation split contains no known label decisions.")
    return weighted_loss / known_total, (
        [row[0] for row in rows],
        [row[1] for row in rows],
        [row[2] for row in rows],
    )


def _targets_tensors(examples, torch, device):
    encoded = [encode_multihot(example) for example in examples]
    targets = torch.tensor([item.values for item in encoded], dtype=torch.float32, device=device)
    mask = torch.tensor([item.known_mask for item in encoded], dtype=torch.float32, device=device)
    return targets, mask


def _encode_tensors(features, torch, device):
    return {
        key: value.to(device) if hasattr(value, "to") else torch.tensor(value, dtype=torch.long, device=device)
        for key, value in features.items()
    }


def _masked_loss(logits, targets, mask, torch):
    if logits.ndim != 2 or tuple(logits.shape) != tuple(targets.shape):
        raise ValueError("Model output must have one logit for each of the five labels.")
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    known_count = int(mask.sum().item())
    if known_count == 0:
        raise ValueError("Batch contains no known labels.")
    loss_sum_tensor = (elementwise * mask).sum()
    return loss_sum_tensor / mask.sum(), float(loss_sum_tensor.detach().item()), known_count


def _batches(indices: Sequence[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str, torch):
    if requested == "cpu" or requested == "auto":
        return torch.device("cpu") if requested == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(requested)

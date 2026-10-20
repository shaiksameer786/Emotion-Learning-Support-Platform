"""Unit tests use explicitly synthetic, TEST-ONLY examples; never train a model here."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from els_platform.prediction.schema import EMOTION_LABELS, EmotionLabel
from els_platform.preprocessing import PreparedExample
from els_platform.training.checkpoints import (
    CheckpointMetadataError,
    _sanitize_metrics,
    build_checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
)
from els_platform.training.config import (
    BiLSTMModelConfig,
    BertModelConfig,
    TrainingConfig,
    configuration_digest,
)
from els_platform.training.dataset import (
    DatasetManifestError,
    _build_training_manifest,
    _examples_content_fingerprint,
    _manifest_fingerprint,
    _read_split_examples,
    _validate_manifest,
    _validate_manifest_authorization,
    load_training_splits,
    read_prepared_split_files,
    write_split_outputs,
)
from els_platform.training.data import (
    TrainingAuthorization,
    TrainingDataError,
    encode_multihot,
    validate_training_splits,
)
from els_platform.training.engine import (
    EpochRecord,
    TrainingRunResult,
    evaluate_model_on_test,
    fit_bert,
    fit_bilstm,
)
from els_platform.training.metrics import (
    EvaluationError,
    FinalizedEvaluationPlan,
    ThresholdSelection,
    calculate_metrics,
    evaluate_heldout_test,
    finalize_evaluation_plan,
    select_thresholds,
)
from els_platform.training.models import load_bert_components
from els_platform.training.optional import OptionalDependencyError, require_torch
from els_platform.training.splitting import DatasetSplits, SplitReport, split_dataset
from els_platform.training.vocabulary import HashedVocabulary
from els_platform.preprocessing import prepare_records


def _example(text: str, positives=(), *, known=EMOTION_LABELS, group_id=None) -> PreparedExample:
    positive_set = set(positives)
    known_set = set(known)
    return PreparedExample(
        text=text,
        labels=tuple(label for label in EMOTION_LABELS if label in positive_set),
        known_labels=tuple(label for label in EMOTION_LABELS if label in known_set),
        group_id=group_id,
    )


def _splits() -> DatasetSplits:
    train = tuple(
        _example(f"train token {i}", (label,), group_id=f"train-{i}")
        for i, label in enumerate(EMOTION_LABELS)
    )
    validation = tuple(
        _example(
            f"validation token {row}",
            tuple(label for index, label in enumerate(EMOTION_LABELS) if (row >> index) & 1),
            group_id=f"validation-{row}",
        )
        for row in range(32)
    )
    test = tuple(
        _example(
            f"heldout token {row}",
            tuple(label for index, label in enumerate(EMOTION_LABELS) if (row >> index) & 1),
            group_id=f"heldout-{row}",
        )
        for row in range(8)
    )
    report = SplitReport(
        seed=42,
        requested_ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        rows_per_split={"train": len(train), "validation": len(validation), "test": len(test)},
        groups_per_split={"train": len(train), "validation": len(validation), "test": len(test)},
        positive_counts={},
        known_counts={},
        grouping_strategy="test-only synthetic fixture groups",
        group_count=len(train) + len(validation) + len(test),
    )
    return DatasetSplits(train=train, validation=validation, test=test, report=report)


def _approved(source_kind="reviewed_public", approved=True) -> TrainingAuthorization:
    return TrainingAuthorization(
        approved_for_training=approved,
        source_kind=source_kind,
        dataset_id="test-dataset-001",
        approval_reference="test-approval-001",
        license_or_consent_reviewed=True,
        label_mapping_reviewed=True,
        data_protection_reviewed=True,
    )


def _synthetic_metric_report():
    """Complete metric report from tiny synthetic arrays; never real observations."""
    targets = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0],
    ]
    scores = [
        [0.9, 0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.1, 0.1],
        [0.9, 0.1, 0.9, 0.9, 0.1],
    ]
    known = [[1, 1, 0, 1, 1] for _ in targets]
    return calculate_metrics(targets, scores, known, (0.5,) * 5)


def _synthetic_prepared_splits():
    """Create TEST-ONLY synthetic partitions for file-contract tests; never fit them."""
    records = []
    for repetition in range(8):
        for bitset in range(32):
            labels = [
                label.value
                for index, label in enumerate(EMOTION_LABELS)
                if bitset & (1 << index)
            ]
            records.append({"text": f"test-only fixture {repetition} {bitset}", "labels": labels})
    prepared = prepare_records(
        records,
        columns=("text", "labels"),
        assume_all_labels_known=True,
    )
    return split_dataset(prepared, seed=42)


class TrainingInfrastructureTests(unittest.TestCase):
    def test_multi_hot_encoding_uses_canonical_order_and_known_mask(self):
        item = _example("test only", (EmotionLabel.CURIOUS, EmotionLabel.BORED), known=(EmotionLabel.BORED, EmotionLabel.CURIOUS))
        encoded = encode_multihot(item)
        self.assertEqual(encoded.values, (1, 0, 0, 1, 0))
        self.assertEqual(encoded.known_mask, (1, 0, 0, 1, 0))

    def test_unknown_and_invalid_labels_are_rejected(self):
        item = PreparedExample("test only", labels=("Joy",), known_labels=EMOTION_LABELS)  # type: ignore[arg-type]
        with self.assertRaises(TrainingDataError):
            encode_multihot(item)

    def test_hashed_vocabulary_fits_training_only(self):
        splits = _splits()
        config = BiLSTMModelConfig(hash_space_size=1_000_003)
        vocabulary = HashedVocabulary.fit_training_split(splits, config)
        unseen_id = vocabulary.encode("validation_only_unique_token_999", max_length=4)[0]
        # First token is validation_only_unique_token_999 as one Unicode word and was not in train.
        self.assertEqual(unseen_id, HashedVocabulary.UNK_ID)
        self.assertEqual(vocabulary, HashedVocabulary.fit_training_split(splits, config))
        self.assertNotIn("train token", str(vocabulary.to_dict()))

    def test_configuration_is_reproducible_and_label_order_is_fixed(self):
        first = TrainingConfig(seed=7)
        second = TrainingConfig(seed=7)
        model = BiLSTMModelConfig()
        self.assertEqual(configuration_digest(first, model), configuration_digest(second, model))
        with self.assertRaises(ValueError):
            TrainingConfig(label_order=tuple(reversed(first.label_order)))

    def test_metrics_and_undefined_label_metrics(self):
        targets = [[0, 0, 0, 0, 0], [1, 1, 0, 1, 1]]
        scores = [[0.1] * 5, [0.9] * 5]
        known = [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]
        report = calculate_metrics(targets, scores, known, (0.5,) * 5)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["micro_f1"], 1.0)
        self.assertIsNone(report["per_label"]["Confused"]["f1"])
        self.assertIsNone(report["per_label"]["Confused"]["precision"])
        self.assertEqual(report["per_label"]["Confused"]["support"], 0)
        missed = calculate_metrics(
            [[1] * 5], [[0.1] * 5], [[1] * 5], (0.5,) * 5
        )
        self.assertIsNone(missed["per_label"]["Bored"]["precision"])
        self.assertEqual(missed["per_label"]["Bored"]["precision_undefined"], "no predicted positives")
        self.assertEqual(missed["per_label"]["Bored"]["f1"], 0.0)

    def test_zero_positive_support_with_false_positives_has_zero_f1_and_matrix(self):
        targets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        scores = [[0.9, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]]
        known = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]

        report = calculate_metrics(targets, scores, known, (0.5,) * 5)

        bored = report["per_label"]["Bored"]
        self.assertEqual(bored["support"], 0)
        self.assertIsNone(bored["recall"])
        self.assertEqual(bored["precision"], 0.0)
        self.assertEqual(bored["f1"], 0.0)
        self.assertEqual(report["macro_f1"], 0.0)
        self.assertEqual(
            bored["confusion_matrix"],
            {
                "actual_labels": ["negative", "positive"],
                "predicted_labels": ["negative", "positive"],
                "counts": [[1, 1], [0, 0]],
            },
        )
        self.assertEqual(report["per_label"]["Confident"]["known_count"], 0)

    def test_confusion_matrix_counts_only_known_label_decisions(self):
        targets = [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],  # Unknown Bored decision must not enter the matrix.
        ]
        scores = [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1],
        ]
        known = [[1, 0, 0, 0, 0] for _ in targets]
        known[-1] = [0, 0, 0, 0, 0]

        report = calculate_metrics(targets, scores, known, (0.5,) * 5)

        self.assertEqual(
            report["per_label"]["Bored"]["confusion_matrix"]["counts"],
            [[1, 1], [0, 1]],
        )
        self.assertEqual(report["per_label"]["Bored"]["known_count"], 3)

    def test_threshold_selection_is_validation_only_and_test_is_gated(self):
        targets = [[0] * 5, [1] * 5]
        scores = [[0.1] * 5, [0.9] * 5]
        known = [[1] * 5, [1] * 5]
        with self.assertRaises(EvaluationError):
            select_thresholds(targets, scores, known, split_name="test")
        selected = select_thresholds(targets, scores, known, split_name="validation", candidates=(0.5,))
        plan = finalize_evaluation_plan(selected, configuration_digest="fixed-config")
        test_report = evaluate_heldout_test(
            targets, scores, known,
            plan=plan,
            configuration_digest="fixed-config",
            split_name="test",
        )
        self.assertEqual(test_report["evaluation_split"], "test")
        with self.assertRaises(EvaluationError):
            evaluate_heldout_test(
                targets, scores, known,
                plan=plan,
                configuration_digest="changed-config",
                split_name="test",
            )

    def test_no_positive_validation_class_leaves_threshold_undefined(self):
        targets = [[0] * 5, [1, 1, 0, 1, 1]]
        scores = [[0.2] * 5, [0.8] * 5]
        known = [[1] * 5, [1] * 5]
        selected = select_thresholds(targets, scores, known, split_name="validation")
        self.assertIsNone(selected.thresholds[2])
        with self.assertRaises(EvaluationError):
            finalize_evaluation_plan(selected, configuration_digest="digest")

    def test_checkpoint_metadata_contains_label_order_and_no_example_text(self):
        config = TrainingConfig()
        model_config = BiLSTMModelConfig()
        digest = configuration_digest(config, model_config)
        selection = ThresholdSelection((0.5,) * 5, (0.8,) * 5, ("selected_on_validation",) * 5)
        plan = finalize_evaluation_plan(selection, configuration_digest=digest)
        splits = _splits()
        run = TrainingRunResult(
            model=object(),
            history=(EpochRecord(1, 0.8, 0.7),),
            best_epoch=1,
            threshold_selection=selection,
            validation_metrics=_synthetic_metric_report(),
            configuration_digest=digest,
            training_summary={
                "rows_per_split": {"train": 5, "validation": 32, "test": 8},
                "train_positive_counts": {label.value: 1 for label in EMOTION_LABELS},
                "train_known_counts": {label.value: 5 for label in EMOTION_LABELS},
                "train_negative_counts": {label.value: 4 for label in EMOTION_LABELS},
                "split_seed": 42,
                "grouping_strategy": "normalized text hash only (participant-level grouping unavailable)",
                "source_kind": "reviewed_public",
                "dataset_id": "test-dataset-001",
                "approval_reference": "test-approval-001",
                "dataset_reference": "reviewed-source-v1",
                "split_manifest_fingerprint": "b" * 64,
            },
            split_fingerprint=splits.split_fingerprint,
            dataset_reference="reviewed-source-v1",
            split_manifest_fingerprint="b" * 64,
            vocabulary=HashedVocabulary.fit_training_split(splits, model_config),
        )
        metadata = build_checkpoint_metadata(
            run,
            model_type="bilstm",
            model_version="infra-v1",
            training_config=config,
            model_config=model_config,
            evaluation_plan=plan,
        )
        self.assertEqual(metadata["label_order"], [label.value for label in EMOTION_LABELS])
        self.assertEqual(
            metadata["validation_metrics"]["per_label"]["Bored"]["confusion_matrix"]["counts"],
            [[1, 1], [0, 1]],
        )
        self.assertNotIn("test only learner words", str(metadata))
        mismatched_plan = FinalizedEvaluationPlan((0.6,) * 5, digest)
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            evaluate_model_on_test(
                run,
            splits,
                _approved(),
                training_config=config,
                model_config=model_config,
                evaluation_plan=mismatched_plan,
            )
        broken = dict(metadata)
        broken["label_order"] = list(reversed(broken["label_order"]))
        with self.assertRaises(CheckpointMetadataError):
            save_checkpoint(object(), "unused.pt", broken)

        class MockModel:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {"weights": [1, 2, 3]}

            def load_state_dict(self, state, *, strict):
                self.loaded = (state, strict)

        class MockTorch:
            @staticmethod
            def save(state, path):
                Path(path).write_bytes(b"mocked torch serialization")

            @staticmethod
            def load(path, *, map_location, weights_only):
                self.assertEqual(map_location, "cpu")
                self.assertIs(weights_only, True)
                return {"weights": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as temporary_directory:
            weights_path = Path(temporary_directory) / "model.pt"
            model = MockModel()
            # This verifies application-level checksum/schema wiring only. It does
            # not validate real PyTorch serialization or a trained model artifact.
            with patch("els_platform.training.checkpoints.require_torch", return_value=MockTorch):
                save_checkpoint(model, weights_path, metadata)
                loaded_model, loaded_metadata = load_checkpoint(
                    MockModel(),
                    weights_path,
                    expected_configuration_digest=digest,
                    expected_model_type="bilstm",
                    expected_dataset_reference="reviewed-source-v1",
                    expected_split_manifest_fingerprint="b" * 64,
                )
                self.assertEqual(loaded_model.loaded, ({"weights": [1, 2, 3]}, True))
                self.assertEqual(len(loaded_metadata["weights_sha256"]), 64)
                with self.assertRaisesRegex(CheckpointMetadataError, "configuration"):
                    load_checkpoint(
                        MockModel(),
                        weights_path,
                        expected_configuration_digest="0" * 64,
                        expected_model_type="bilstm",
                        expected_dataset_reference="reviewed-source-v1",
                        expected_split_manifest_fingerprint="b" * 64,
                    )
                metadata_path = weights_path.with_suffix(".pt.json")
                original_metadata = metadata_path.read_text(encoding="utf-8")
                changed_metadata = json.loads(original_metadata)
                changed_metadata["model_config"]["hidden_size"] += 1
                metadata_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
                with self.assertRaisesRegex(CheckpointMetadataError, "digest does not match"):
                    load_checkpoint(
                        MockModel(),
                        weights_path,
                        expected_configuration_digest=digest,
                        expected_model_type="bilstm",
                        expected_dataset_reference="reviewed-source-v1",
                        expected_split_manifest_fingerprint="b" * 64,
                    )
                metadata_path.write_text(original_metadata, encoding="utf-8")
                changed_metadata = json.loads(original_metadata)
                changed_metadata["training_data_summary"]["split_manifest_fingerprint"] = "c" * 64
                metadata_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
                with self.assertRaisesRegex(CheckpointMetadataError, "manifest does not match"):
                    load_checkpoint(
                        MockModel(),
                        weights_path,
                        expected_configuration_digest=digest,
                        expected_model_type="bilstm",
                        expected_dataset_reference="reviewed-source-v1",
                        expected_split_manifest_fingerprint="b" * 64,
                    )
                metadata_path.write_text(original_metadata, encoding="utf-8")
                weights_path.write_bytes(b"tampered mock weights")
                with self.assertRaisesRegex(CheckpointMetadataError, "checksum"):
                    load_checkpoint(
                        MockModel(),
                        weights_path,
                        expected_configuration_digest=digest,
                        expected_model_type="bilstm",
                        expected_dataset_reference="reviewed-source-v1",
                        expected_split_manifest_fingerprint="b" * 64,
                    )

    def test_training_refuses_unapproved_and_test_fixture_sources(self):
        splits = _splits()
        with self.assertRaises(TrainingDataError):
            validate_training_splits(splits, _approved(approved=False))
        with self.assertRaises(TrainingDataError):
            validate_training_splits(splits, _approved(source_kind="test_fixture"))
        with self.assertRaises(TrainingDataError):
            fit_bilstm(splits, _approved(source_kind="test_fixture"))

    def test_training_refuses_cross_partition_leakage(self):
        splits = _splits()
        leaked_validation = (_example(splits.train[0].text, splits.train[0].labels),) + splits.validation[1:]
        leaked = DatasetSplits(splits.train, leaked_validation, splits.test, splits.report)
        with self.assertRaises(TrainingDataError):
            validate_training_splits(leaked, _approved())

    def test_high_level_test_evaluation_is_bound_to_the_fitted_splits(self):
        splits = _splits()
        config = TrainingConfig()
        model_config = BiLSTMModelConfig()
        digest = configuration_digest(config, model_config)
        selection = ThresholdSelection((0.5,) * 5, (0.8,) * 5, ("selected_on_validation",) * 5)
        run = TrainingRunResult(
            model=object(),
            history=(EpochRecord(1, 0.8, 0.7),),
            best_epoch=1,
            threshold_selection=selection,
            validation_metrics={},
            configuration_digest=digest,
            training_summary={
                "source_kind": "reviewed_public",
                "dataset_id": "test-dataset-001",
                "approval_reference": "test-approval-001",
            },
            split_fingerprint=splits.split_fingerprint,
        )
        plan = finalize_evaluation_plan(selection, configuration_digest=digest)
        changed_test = (_example("changed heldout record", (EMOTION_LABELS[0],)),) + splits.test[1:]
        changed_splits = DatasetSplits(splits.train, splits.validation, changed_test, splits.report)
        with self.assertRaisesRegex(ValueError, "exact train/validation/test partitions"):
            evaluate_model_on_test(
                run,
                changed_splits,
                _approved(),
                training_config=config,
                model_config=model_config,
                evaluation_plan=plan,
            )

    def test_preparation_split_and_training_preflight_connect_without_ml_dependencies(self):
        # Synthetic TEST-ONLY records exercise data plumbing; this is not a dataset.
        records = []
        for repetition in range(8):
            for bitset in range(32):
                labels = [
                    label.value
                    for index, label in enumerate(EMOTION_LABELS)
                    if bitset & (1 << index)
                ]
                records.append({"text": f"test fixture record {repetition} {bitset}", "labels": labels})
        prepared = prepare_records(
            records,
            columns=("text", "labels"),
            assume_all_labels_known=True,
        )
        self.assertTrue(prepared.report.is_valid)
        splits = split_dataset(prepared, seed=42)
        summary = validate_training_splits(splits, _approved())
        vocabulary = HashedVocabulary.fit_training_split(splits, BiLSTMModelConfig())
        self.assertEqual(sum(summary["rows_per_split"].values()), len(records))
        self.assertGreater(vocabulary.size, 2)
        self.assertEqual(splits.split_fingerprint, split_dataset(prepared, seed=42).split_fingerprint)

    def test_mocked_bilstm_fit_wiring_uses_train_fitted_vocabulary(self):
        from els_platform.training import engine

        splits = _splits()
        captured = {}

        def fake_fit(model, passed_splits, config, model_config, encode_batch, summary, **kwargs):
            captured["model"] = model
            captured["splits"] = passed_splits
            captured["validation_features"] = encode_batch((
                _example("validationonlytoken", (), group_id="validation-probe"),
            ))
            captured["vocabulary"] = kwargs["vocabulary"]
            return "mocked fit result"

        with patch.object(engine, "build_bilstm_model", return_value="mocked bilstm"), patch.object(
            engine, "_fit", side_effect=fake_fit
        ):
            result = fit_bilstm(splits, _approved())
        self.assertEqual(result, "mocked fit result")
        self.assertEqual(captured["model"], "mocked bilstm")
        self.assertEqual(captured["splits"], splits)
        self.assertEqual(captured["validation_features"]["input_ids"][0][0], HashedVocabulary.UNK_ID)

    def test_mocked_bert_fit_wiring_uses_matching_multilabel_sequence_config(self):
        from els_platform.training import engine

        splits = _splits()
        calls = []

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                calls.append((texts, kwargs))
                length = kwargs["max_length"]
                return {
                    "input_ids": [[1] * length for _ in texts],
                    "attention_mask": [[1] * length for _ in texts],
                }

        tokenizer = MockTokenizer()
        training = TrainingConfig.for_bert(max_sequence_length=16)
        bert = BertModelConfig(
            checkpoint_id="reviewed-model-id",
            revision="a" * 40,
            max_sequence_length=16,
            license_reviewed=True,
            language_fit_reviewed=True,
            resource_requirements_reviewed=True,
        )
        captured = {}

        def fake_fit(model, passed_splits, config, model_config, encode_batch, summary, **kwargs):
            captured["model"] = model
            captured["features"] = encode_batch(passed_splits.validation[:2])
            captured["tokenizer"] = kwargs["bert_tokenizer"]
            captured["model_config"] = model_config
            return "mocked fit result"

        with patch.object(
            engine, "load_bert_components", return_value=SimpleNamespace(tokenizer=tokenizer, model="mocked bert")
        ), patch.object(engine, "_fit", side_effect=fake_fit):
            result = fit_bert(splits, _approved(), bert_config=bert, training_config=training)
        self.assertEqual(result, "mocked fit result")
        self.assertEqual(captured["model"], "mocked bert")
        self.assertEqual(captured["model_config"], bert)
        self.assertEqual(captured["features"]["input_ids"], [[1] * 16, [1] * 16])
        self.assertEqual(captured["tokenizer"], tokenizer)
        self.assertEqual(calls[0][1], {
            "truncation": True,
            "max_length": 16,
            "padding": "max_length",
            "return_tensors": "pt",
        })

    def test_split_report_counts_must_match_partitions(self):
        splits = _splits()
        shortened_test = DatasetSplits(
            splits.train,
            splits.validation,
            splits.test[:-1],
            splits.report,
        )
        with self.assertRaisesRegex(TrainingDataError, "report row counts"):
            validate_training_splits(shortened_test, _approved())

    def test_training_refuses_empty_partition(self):
        splits = _splits()
        empty_validation = DatasetSplits(splits.train, (), splits.test, splits.report)
        with self.assertRaises(TrainingDataError):
            validate_training_splits(empty_validation, _approved())

    def test_training_refuses_validation_without_two_classes_for_thresholds(self):
        splits = _splits()
        all_negative = tuple(
            _example(f"validation only negatives {row}", (), group_id=f"validation-negative-{row}")
            for row in range(8)
        )
        updated_report = replace(
            splits.report,
            rows_per_split={
                "train": len(splits.train),
                "validation": len(all_negative),
                "test": len(splits.test),
            },
        )
        unusable = DatasetSplits(splits.train, all_negative, splits.test, updated_report)
        with self.assertRaisesRegex(TrainingDataError, "select thresholds"):
            validate_training_splits(unusable, _approved())

    def test_missing_optional_dependencies_and_unreviewed_checkpoint_are_clear(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(OptionalDependencyError):
                require_torch()
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(OptionalDependencyError):
                fit_bilstm(_splits(), _approved())
        with self.assertRaisesRegex(ValueError, "checkpoint_id"):
            load_bert_components(BertModelConfig())

    def test_unavailable_checkpoint_is_not_fetched(self):
        from els_platform.training import models

        class MissingModelHub:
            class AutoTokenizer:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    raise OSError("not cached")

            class AutoModel:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    raise AssertionError("model loading should stop at missing tokenizer")

        reviewed = BertModelConfig(
            checkpoint_id="reviewed-model-id",
            revision="a" * 40,
            license_reviewed=True,
            language_fit_reviewed=True,
            resource_requirements_reviewed=True,
        )
        with patch.object(models, "require_torch", return_value=object()), patch.object(
            models, "require_transformers", return_value=MissingModelHub
        ):
            with self.assertRaisesRegex(models.CheckpointUnavailableError, "no download was attempted"):
                load_bert_components(reviewed)


class CheckpointMetricValidationTests(unittest.TestCase):
    @staticmethod
    def _copy_report():
        return json.loads(json.dumps(_synthetic_metric_report()))

    def test_complete_report_and_serialized_metric_schema_are_accepted(self):
        report = self._copy_report()
        sanitized = _sanitize_metrics(report)
        self.assertEqual(_sanitize_metrics(sanitized), sanitized)
        self.assertEqual(set(sanitized["per_label"]), {label.value for label in EMOTION_LABELS})

    def test_matrix_total_must_match_known_count(self):
        report = self._copy_report()
        report["per_label"]["Bored"]["confusion_matrix"]["counts"][0][0] += 1
        with self.assertRaisesRegex(CheckpointMetadataError, "total differs from known_count"):
            _sanitize_metrics(report)

    def test_matrix_positive_support_must_match_support(self):
        report = self._copy_report()
        report["per_label"]["Bored"]["support"] += 1
        with self.assertRaisesRegex(CheckpointMetadataError, "positive support differs from support"):
            _sanitize_metrics(report)

    def test_precision_recall_and_f1_must_match_matrix(self):
        for metric, value in (("precision", 0.25), ("recall", 0.5), ("f1", 0.5)):
            with self.subTest(metric=metric):
                report = self._copy_report()
                report["per_label"]["Bored"][metric] = value
                with self.assertRaisesRegex(CheckpointMetadataError, "disagrees with confusion-matrix counts"):
                    _sanitize_metrics(report)

    def test_undefined_metric_conventions_are_preserved(self):
        false_positive_report = calculate_metrics(
            [[0] * 5, [0] * 5],
            [[0.9, 0.1, 0.1, 0.1, 0.1], [0.1] * 5],
            [[1] * 5, [1] * 5],
            (0.5,) * 5,
        )
        sanitized = _sanitize_metrics(false_positive_report)
        bored = sanitized["per_label"]["Bored"]
        self.assertEqual(
            (bored["support"], bored["precision"], bored["recall"], bored["f1"]),
            (0, 0.0, None, 0.0),
        )
        self.assertEqual(bored["undefined"], "no known positive examples")
        self.assertIsNone(bored["precision_undefined"])

        no_predictions = sanitized["per_label"]["Confident"]
        self.assertIsNone(no_predictions["precision"])
        self.assertIsNone(no_predictions["recall"])
        self.assertIsNone(no_predictions["f1"])
        self.assertEqual(no_predictions["precision_undefined"], "no predicted positives")

        no_known_report = calculate_metrics(
            [[0] * 5], [[0.1] * 5], [[0] * 5], (0.5,) * 5
        )
        no_known = _sanitize_metrics(no_known_report)["per_label"]["Bored"]
        self.assertEqual(no_known["known_count"], 0)
        self.assertIsNone(no_known["precision"])
        self.assertIsNone(no_known["recall"])
        self.assertIsNone(no_known["f1"])

    def test_invalid_counters_types_ranges_and_unexpected_fields_are_rejected(self):
        mutations = (
            ("negative support", lambda r: r["per_label"]["Bored"].update(support=-1)),
            ("boolean support", lambda r: r["per_label"]["Bored"].update(support=True)),
            ("float known count", lambda r: r["per_label"]["Bored"].update(known_count=2.0)),
            ("negative matrix count", lambda r: r["per_label"]["Bored"]["confusion_matrix"]["counts"][0].__setitem__(0, -1)),
            ("boolean matrix count", lambda r: r["per_label"]["Bored"]["confusion_matrix"]["counts"][0].__setitem__(0, True)),
            ("float matrix count", lambda r: r["per_label"]["Bored"]["confusion_matrix"]["counts"][0].__setitem__(0, 1.0)),
            ("boolean aggregate count", lambda r: r.update(macro_f1_defined_labels=True)),
            ("negative aggregate count", lambda r: r.update(known_label_decisions=-1)),
            ("ratio above range", lambda r: r["per_label"]["Bored"].update(precision=1.1)),
            ("ratio wrong type", lambda r: r["per_label"]["Bored"].update(recall="0.5")),
            ("boolean ratio", lambda r: r["per_label"]["Bored"].update(f1=False)),
            ("aggregate ratio below range", lambda r: r.update(macro_f1=-0.1)),
            ("undefined subset accuracy with rows", lambda r: r.update(subset_accuracy=None, subset_accuracy_rows=1)),
            ("unexpected top-level field", lambda r: r.update(accuracy=1.0)),
            ("unexpected label field", lambda r: r["per_label"]["Bored"].update(extra=0)),
            ("unexpected matrix field", lambda r: r["per_label"]["Bored"]["confusion_matrix"].update(axis="wrong")),
            ("missing aggregate field", lambda r: r.pop("micro_f1")),
            ("missing per-label field", lambda r: r["per_label"]["Bored"].pop("known_count")),
        )
        for name, mutate in mutations:
            with self.subTest(case=name):
                report = self._copy_report()
                mutate(report)
                with self.assertRaises(CheckpointMetadataError):
                    _sanitize_metrics(report)


class PersistedSplitManifestTests(unittest.TestCase):
    def _write_test_only_bundle(self, directory: Path):
        splits = _synthetic_prepared_splits()
        write_split_outputs(splits, directory)
        source = {
            "dataset_reference": "test-source-v1",
            "dataset_id": "test-dataset-001",
            "source_kind": "test_fixture",
            "approval_reference": "test-approval-001",
            "approval_checks": {key: False for key in (
                "approved_for_training",
                "license_or_consent_reviewed",
                "label_mapping_reviewed",
                "data_protection_reviewed",
            )},
        }
        fingerprints = {
            name: _examples_content_fingerprint(getattr(splits, name))
            for name in ("train", "validation", "test")
        }
        manifest = _build_training_manifest(splits.report, source, fingerprints)
        (directory / "training_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        return splits, manifest

    def _write_grouped_test_only_bundle(self, directory: Path):
        """Persist synthetic test-only group tokens; these are not participant IDs."""
        records = [
            {
                "text": f"SYNTHETIC TEST ONLY grouped row {index}",
                "labels": ["Confused"] if index % 2 == 0 else ["Curious"],
                "group_id": f"synthetic_test_only_group_{index // 2:02d}",
            }
            for index in range(12)
        ]
        prepared = prepare_records(records, assume_all_labels_known=True)
        splits = split_dataset(prepared, seed=17)
        write_split_outputs(splits, directory)
        source = {
            "dataset_reference": "test-source-v1",
            "dataset_id": "test-dataset-001",
            "source_kind": "test_fixture",
            "approval_reference": "test-approval-001",
            "approval_checks": {key: False for key in (
                "approved_for_training",
                "license_or_consent_reviewed",
                "label_mapping_reviewed",
                "data_protection_reviewed",
            )},
        }
        fingerprints = {
            name: _examples_content_fingerprint(getattr(splits, name))
            for name in ("train", "validation", "test")
        }
        manifest = _build_training_manifest(splits.report, source, fingerprints)
        (directory / "training_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        return splits, manifest

    @staticmethod
    def _set_legacy_manifest_format(manifest, splits):
        manifest["manifest_version"] = 1
        manifest["split"]["content_fingerprints"] = {
            name: _examples_content_fingerprint(
                getattr(splits, name), include_group_id=False
            )
            for name in ("train", "validation", "test")
        }
        manifest["leakage_safeguards"] = {
            "normalized_text_disjointness_rechecked_on_load": True,
            "grouping_strategy": manifest["split"]["grouping_strategy"],
            "group_id_values_stored_in_split_files": False,
            "group_id_semantics_verified": False,
            "participant_grouping_limitation": (
                "Participant-level separation depends on suitable source group IDs; "
                "when unavailable, only exact normalized-text grouping is provided."
            ),
        }
        manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)

    def test_manifest_is_stable_and_test_fixture_bundle_is_never_training_approved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            splits, first = self._write_test_only_bundle(directory)
            source = first["source"]
            fingerprints = first["split"]["content_fingerprints"]
            second = _build_training_manifest(splits.report, source, fingerprints)
            self.assertEqual(first["manifest_fingerprint"], second["manifest_fingerprint"])

            bundle = read_prepared_split_files(directory)
            self.assertNotIsInstance(bundle, DatasetSplits)
            self.assertEqual(bundle.manifest_fingerprint, first["manifest_fingerprint"])
            with self.assertRaisesRegex(DatasetManifestError, "test-only fixtures"):
                load_training_splits(directory, authorization=_approved())
            with self.assertRaisesRegex(DatasetManifestError, "not approved"):
                load_training_splits(
                    directory,
                    authorization=_approved(source_kind="test_fixture", approved=False),
                )

    def test_tampered_manifest_and_split_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _, manifest = self._write_test_only_bundle(directory)
            changed = json.loads(json.dumps(manifest))
            changed["split"]["seed"] += 1
            (directory / "training_manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(DatasetManifestError, "fingerprint"):
                read_prepared_split_files(directory)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_only_bundle(directory)
            train_path = directory / "train.jsonl"
            lines = train_path.read_text(encoding="utf-8").splitlines()
            first_row = json.loads(lines[0])
            first_row["text"] += " changed"
            lines[0] = json.dumps(first_row)
            train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetManifestError, "content fingerprint"):
                read_prepared_split_files(directory)

    def test_group_ids_are_persisted_and_disjointness_is_rechecked_on_reload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            splits, manifest = self._write_grouped_test_only_bundle(directory)

            self.assertEqual(manifest["manifest_version"], 2)
            self.assertIs(manifest["leakage_safeguards"]["group_id_values_stored_in_split_files"], True)
            bundle = read_prepared_split_files(directory)
            self.assertTrue(all(example.group_id is not None for example in bundle.train + bundle.validation + bundle.test))
            group_owner = {}
            for split_name in ("train", "validation", "test"):
                for example in getattr(bundle, split_name):
                    group_owner.setdefault(example.group_id, set()).add(split_name)
            self.assertTrue(all(len(owners) == 1 for owners in group_owner.values()))
            self.assertEqual(
                {name: len({example.group_id for example in getattr(bundle, name)}) for name in ("train", "validation", "test")},
                dict(splits.report.groups_per_split),
            )

    def test_loader_rejects_cross_split_group_even_with_updated_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _, manifest = self._write_grouped_test_only_bundle(directory)
            train_path = directory / "train.jsonl"
            validation_path = directory / "validation.jsonl"
            train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            validation_rows = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines()]
            train_rows[0]["group_id"] = validation_rows[0]["group_id"]
            train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")

            changed_train = _read_split_examples(train_path, include_group_id=True)
            manifest["split"]["content_fingerprints"]["train"] = _examples_content_fingerprint(changed_train)
            manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
            (directory / "training_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )

            with self.assertRaisesRegex(DatasetManifestError, "group_id appears in more than one persisted split"):
                read_prepared_split_files(directory)

    def test_loader_rejects_missing_or_blank_persisted_group_ids(self):
        for mutation in ("missing", "blank"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary_directory:
                directory = Path(temporary_directory)
                self._write_grouped_test_only_bundle(directory)
                train_path = directory / "train.jsonl"
                rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
                if mutation == "missing":
                    rows[0].pop("group_id")
                else:
                    rows[0]["group_id"] = "   "
                train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

                with self.assertRaisesRegex(DatasetManifestError, "violates the dataset contract"):
                    read_prepared_split_files(directory)

    def test_loader_rejects_inconsistent_reported_group_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _, manifest = self._write_grouped_test_only_bundle(directory)
            report_path = directory / "split_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            groups = dict(report["groups_per_split"])
            rows = report["rows_per_split"]
            donor = next(name for name in ("train", "validation", "test") if groups[name] > 1)
            recipient = next(
                name for name in ("train", "validation", "test")
                if name != donor and groups[name] < rows[name]
            )
            groups[donor] -= 1
            groups[recipient] += 1
            report["groups_per_split"] = groups
            report_path.write_text(json.dumps(report), encoding="utf-8")

            manifest["split"]["groups_per_split"] = groups
            manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
            (directory / "training_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )

            with self.assertRaisesRegex(DatasetManifestError, "Persisted group counts differ"):
                read_prepared_split_files(directory)

    def test_legacy_text_hash_bundles_remain_readable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            splits, manifest = self._write_test_only_bundle(directory)
            self._set_legacy_manifest_format(manifest, splits)
            (directory / "training_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )

            bundle = read_prepared_split_files(directory)
            self.assertTrue(all(example.group_id is None for example in bundle.train + bundle.validation + bundle.test))

    def test_legacy_group_aware_bundle_is_rejected_as_unverifiable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            splits, manifest = self._write_grouped_test_only_bundle(directory)
            self._set_legacy_manifest_format(manifest, splits)
            (directory / "training_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )

            with self.assertRaisesRegex(DatasetManifestError, "Legacy group-aware split manifest"):
                read_prepared_split_files(directory)

    def test_manifest_rejects_malformed_discriminators_without_type_errors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, manifest = self._write_test_only_bundle(Path(temporary_directory))
            malformed_cases = []
            changed = json.loads(json.dumps(manifest))
            changed["source"]["source_kind"] = []
            malformed_cases.append(changed)
            changed = json.loads(json.dumps(manifest))
            changed["split"]["grouping_strategy"] = []
            malformed_cases.append(changed)
            changed = json.loads(json.dumps(manifest))
            changed["manifest_version"] = True
            malformed_cases.append(changed)
            changed = json.loads(json.dumps(manifest))
            changed["manifest_version"] = 1.0
            malformed_cases.append(changed)
            for changed in malformed_cases:
                with self.subTest(changed=changed):
                    with self.assertRaises(DatasetManifestError):
                        _validate_manifest(changed)

    def test_manifest_split_report_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_only_bundle(directory)
            report_path = directory / "split_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["seed"] += 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(DatasetManifestError):
                read_prepared_split_files(directory)

    def test_manifest_approval_reference_must_match_explicit_authorization(self):
        source = {
            "dataset_reference": "test-source-v1",
            "dataset_id": "test-dataset-001",
            "source_kind": "reviewed_public",
            "approval_reference": "different-test-approval",
            "approval_checks": {
                "approved_for_training": True,
                "license_or_consent_reviewed": True,
                "label_mapping_reviewed": True,
                "data_protection_reviewed": True,
            },
        }
        with self.assertRaisesRegex(DatasetManifestError, "does not match authorization"):
            _validate_manifest_authorization(source, _approved())

    def test_manifest_writer_rejects_test_fixture_authorization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            splits = _synthetic_prepared_splits()
            write_split_outputs(splits, directory)
            with self.assertRaisesRegex(DatasetManifestError, "unapproved"):
                from els_platform.training.dataset import write_training_manifest

                write_training_manifest(
                    splits,
                    directory,
                    dataset_reference="test-source-v1",
                    authorization=_approved(source_kind="test_fixture", approved=False),
                )

    def test_loader_rejects_invalid_prepared_split_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_only_bundle(directory)
            train_path = directory / "train.jsonl"
            rows = train_path.read_text(encoding="utf-8").splitlines()
            first_row = json.loads(rows[0])
            first_row["unsupported_extra"] = "not permitted"
            rows[0] = json.dumps(first_row)
            train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetManifestError, "violates the dataset contract"):
                read_prepared_split_files(directory)

    def test_cpu_smoke_command_reports_missing_framework_as_skipped(self):
        from els_platform.training import smoke_test

        output = io.StringIO()
        with patch.object(
            smoke_test,
            "require_torch",
            side_effect=OptionalDependencyError("PyTorch is not installed for this test."),
        ), redirect_stdout(output):
            exit_code = smoke_test.main(["--model", "bilstm"])
        self.assertEqual(exit_code, 0)
        self.assertIn("SKIPPED", output.getvalue())
        self.assertIn("PyTorch is not installed", output.getvalue())

    def test_real_cpu_framework_smoke_or_explicit_unittest_skip(self):
        from els_platform.training.smoke_test import run_bilstm_cpu_smoke

        try:
            require_torch()
        except OptionalDependencyError as exc:
            self.skipTest(f"PyTorch unavailable; real CPU framework smoke not run: {exc}")
        result = run_bilstm_cpu_smoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["output_shape"], [2, 5])
        self.assertIs(result["optimizer_step"], False)
        self.assertIs(result["performance_metrics"], False)

    def test_bert_sequence_length_cannot_exceed_local_encoder_positions(self):
        from els_platform.training import models

        class MockTokenizerLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                self.assertIs(kwargs["local_files_only"], True)
                return object()

        class MockModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return SimpleNamespace(
                    config=SimpleNamespace(hidden_size=8, max_position_embeddings=16)
                )

        reviewed = BertModelConfig(
            checkpoint_id="reviewed-model-id",
            revision="a" * 40,
            max_sequence_length=32,
            license_reviewed=True,
            language_fit_reviewed=True,
            resource_requirements_reviewed=True,
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=MockTokenizerLoader,
            AutoModel=MockModelLoader,
        )
        with patch.object(models, "require_torch", return_value=object()), patch.object(
            models, "require_transformers", return_value=fake_transformers
        ):
            with self.assertRaisesRegex(models.CheckpointUnavailableError, "max_position_embeddings"):
                load_bert_components(reviewed)


if __name__ == "__main__":
    unittest.main()

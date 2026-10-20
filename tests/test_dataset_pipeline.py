"""Synthetic test-only fixtures for the local dataset preparation pipeline.

Every row created here is invented test input. These fixtures are not observations,
approved training data, or evaluation data and are never written outside temp folders.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from els_platform.preprocessing import (
    DatasetContractError,
    EmotionLabel,
    normalize_text,
    prepare_records,
)
from els_platform.training.dataset import (
    load_dataset_file,
    main as prepare_dataset_main,
    write_audit_report,
    write_split_outputs,
)
from els_platform.training.splitting import DatasetSplitError, split_dataset


class SyntheticDatasetValidationTests(unittest.TestCase):
    def test_schema_and_multilabel_annotations_are_validated(self) -> None:
        dataset = prepare_records(
            [
                {
                    "text": "I am not stuck; I want to explore another example.",
                    "labels": ["Curious", "Confident"],
                    "known_labels": ["Curious", "Confident", "Confused"],
                }
            ]
        )
        self.assertTrue(dataset.report.is_valid)
        self.assertEqual(
            dataset.examples[0].labels,
            (EmotionLabel.CONFIDENT, EmotionLabel.CURIOUS),
        )
        self.assertEqual(
            dataset.examples[0].known_labels,
            (EmotionLabel.CONFIDENT, EmotionLabel.CONFUSED, EmotionLabel.CURIOUS),
        )
        self.assertEqual(dataset.report.known_label_counts[EmotionLabel.CONFUSED], 1)

    def test_normalization_preserves_language_signals(self) -> None:
        value = "  I\tDON'T\nunderstand—why?\v  Cafe\u0301 🙂 "
        normalized = normalize_text(value)
        self.assertEqual(normalized, "I DON'T understand—why? Café 🙂")
        self.assertIn("DON'T", normalized)
        self.assertIn("?", normalized)

    def test_only_expected_schema_columns_are_allowed(self) -> None:
        with self.assertRaisesRegex(DatasetContractError, "unsupported columns"):
            prepare_records(
                [{"text": "synthetic", "labels": ["Confused"], "student_name": "Test only"}]
            )

    def test_source_label_mapping_must_target_a_supported_category(self) -> None:
        with self.assertRaisesRegex(DatasetContractError, "one of the five target labels"):
            prepare_records(
                [{"text": "synthetic", "labels": ["joy"]}],
                label_mapping={"joy": "Happy"},
            )

    def test_explicit_mapping_is_applied_without_inventing_other_labels(self) -> None:
        dataset = prepare_records(
            [{"text": "Synthetic study challenge", "labels": ["confusion", "curiosity"]}],
            label_mapping={"confusion": "Confused", "curiosity": "Curious"},
        )
        self.assertTrue(dataset.report.is_valid)
        self.assertEqual(
            dataset.examples[0].labels,
            (EmotionLabel.CONFUSED, EmotionLabel.CURIOUS),
        )
        self.assertNotIn(EmotionLabel.BORED, dataset.examples[0].known_labels)

    def test_unsupported_and_malformed_labels_are_reported_without_text(self) -> None:
        private_looking_fixture = "DEMONSTRATION TEST ONLY: learner text must never leak."
        dataset = prepare_records(
            [
                {
                    "text": private_looking_fixture,
                    "labels": [
                        "this is not a category; redact the whole value",
                        "SyntheticPossibleIdentifier",
                    ],
                },
                {"text": "another synthetic row", "labels": ["anger"]},
            ]
        )
        report_json = json.dumps(dataset.report.to_dict())
        self.assertFalse(dataset.report.is_valid)
        self.assertNotIn(private_looking_fixture, report_json)
        self.assertNotIn("this is not a category", report_json)
        self.assertNotIn("SyntheticPossibleIdentifier", report_json)
        self.assertNotIn('"anger"', report_json)
        self.assertEqual(dataset.report.unsupported_label_occurrences, 3)
        self.assertEqual(dataset.report.distinct_unsupported_label_values, 3)

    def test_missing_values_are_counted_and_rows_rejected(self) -> None:
        dataset = prepare_records(
            [
                {"text": " ", "labels": ["Confused"]},
                {"text": "Synthetic only: missing labels", "labels": None},
            ]
        )
        self.assertEqual(dataset.report.missing_text_rows, 1)
        self.assertEqual(dataset.report.missing_label_rows, 1)
        self.assertEqual(dataset.report.invalid_rows, 2)
        self.assertEqual(dataset.examples, ())

    def test_exact_duplicates_are_dropped_but_conflicts_are_rejected(self) -> None:
        same_rows = [
            {"text": " Synthetic repeated text ", "labels": ["Curious"]},
            {"text": "Synthetic   repeated text", "labels": ["Curious"]},
        ]
        deduplicated = prepare_records(same_rows)
        self.assertTrue(deduplicated.report.is_valid)
        self.assertEqual(deduplicated.report.duplicate_rows_dropped, 1)
        self.assertEqual(deduplicated.report.accepted_records, 1)

        conflicting = prepare_records(
            [
                {"text": "Synthetic same text", "labels": ["Curious"]},
                {"text": "Synthetic same text", "labels": ["Confused"]},
            ]
        )
        self.assertFalse(conflicting.report.is_valid)
        self.assertEqual(conflicting.report.conflicting_duplicate_groups, 1)
        self.assertEqual(conflicting.examples, ())

    def test_complete_label_assumption_is_explicit(self) -> None:
        row = {"text": "Synthetic complete annotation", "labels": ["Confident"]}
        partial = prepare_records([row])
        complete = prepare_records([row], assume_all_labels_known=True)
        self.assertNotIn(EmotionLabel.BORED, partial.examples[0].known_labels)
        self.assertIn(EmotionLabel.BORED, complete.examples[0].known_labels)

    def test_audit_report_excludes_text_and_group_identifiers(self) -> None:
        raw_fixture = "SYNTHETIC ONLY - privacy audit must not contain this phrase"
        group_fixture = "opaque_test_group_9A8B7C6D"
        dataset = prepare_records(
            [
                {
                    "text": raw_fixture,
                    "labels": ["Bored"],
                    "group_id": group_fixture,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_path = write_audit_report(
                dataset.report,
                Path(temporary_directory) / "audit.json",
            )
            audit_text = audit_path.read_text(encoding="utf-8")
        self.assertNotIn(raw_fixture, audit_text)
        self.assertNotIn(group_fixture, audit_text)
        self.assertIn('"Bored"', audit_text)


class SyntheticDatasetFileTests(unittest.TestCase):
    def test_csv_and_jsonl_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            csv_path = directory / "test_only.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["text", "labels", "known_labels"])
                writer.writeheader()
                writer.writerow(
                    {
                        "text": "SYNTHETIC TEST ONLY: CSV row",
                        "labels": '["Confused", "Curious"]',
                        "known_labels": '["Confused", "Curious", "Bored"]',
                    }
                )
            csv_dataset = load_dataset_file(csv_path)
            self.assertTrue(csv_dataset.report.is_valid)
            self.assertEqual(len(csv_dataset.examples[0].labels), 2)

            jsonl_path = directory / "test_only.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "text": "SYNTHETIC TEST ONLY: JSONL row",
                        "labels": ["Confident"],
                        "known_labels": ["Confident", "Frustrated"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            jsonl_dataset = load_dataset_file(jsonl_path)
            self.assertTrue(jsonl_dataset.report.is_valid)
            self.assertIn(EmotionLabel.FRUSTRATED, jsonl_dataset.examples[0].known_labels)

    def test_cli_writes_audit_and_reproducible_prepared_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "synthetic_test_only.csv"
            with source.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["text", "labels"])
                writer.writeheader()
                for index in range(6):
                    writer.writerow(
                        {
                            "text": f"SYNTHETIC TEST ONLY unique row {index}",
                            "labels": json.dumps(["Curious"]),
                        }
                    )
            audit = directory / "audit.json"
            output_dir = directory / "prepared"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exit_code = prepare_dataset_main(
                    [
                        str(source),
                        "--audit-output",
                        str(audit),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "42",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(audit.exists())
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "validation.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())
            audit_text = audit.read_text(encoding="utf-8")
            self.assertNotIn("SYNTHETIC TEST ONLY", audit_text)
            self.assertIn('"seed": 42', (output_dir / "split_report.json").read_text(encoding="utf-8"))


class SyntheticDatasetSplitTests(unittest.TestCase):
    @staticmethod
    def _grouped_fixture():
        # Synthetic test-only groups; these do not identify real people.
        rows = []
        for index in range(12):
            rows.append(
                {
                    "text": f"SYNTHETIC TEST ONLY challenge sentence {index}",
                    "labels": ["Confused"] if index % 2 else ["Curious"],
                    "group_id": f"opaque_test_group_{index // 2:02d}",
                }
            )
        return prepare_records(rows, assume_all_labels_known=True)

    def test_split_is_reproducible_and_group_disjoint(self) -> None:
        dataset = self._grouped_fixture()
        first = split_dataset(dataset, seed=17)
        second = split_dataset(dataset, seed=17)
        first_by_split = {
            name: tuple(example.text for example in getattr(first, name))
            for name in ("train", "validation", "test")
        }
        second_by_split = {
            name: tuple(example.text for example in getattr(second, name))
            for name in ("train", "validation", "test")
        }
        self.assertEqual(first_by_split, second_by_split)

        group_split = {}
        for name in ("train", "validation", "test"):
            for example in getattr(first, name):
                group_split.setdefault(example.group_id, set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in group_split.values()))
        self.assertIn("participant/source meaning is not verified", first.report.grouping_strategy)

    def test_without_group_ids_does_not_claim_participant_grouping(self) -> None:
        dataset = prepare_records(
            [
                {"text": f"Unique synthetic row {index}", "labels": ["Curious"]}
                for index in range(6)
            ]
        )
        splits = split_dataset(dataset, seed=42)
        self.assertIn("participant-level grouping unavailable", splits.report.grouping_strategy)
        self.assertEqual(
            sum(splits.report.rows_per_split.values()),
            dataset.report.accepted_records,
        )

    def test_split_refuses_invalid_data_and_invalid_ratios(self) -> None:
        invalid = prepare_records([{"text": "bad fixture", "labels": ["anger"]}])
        with self.assertRaises(DatasetSplitError):
            split_dataset(invalid)
        valid = self._grouped_fixture()
        with self.assertRaises(DatasetSplitError):
            split_dataset(valid, ratios={"train": 0.6, "validation": 0.2, "test": 0.1})

    def test_grouped_split_outputs_retain_group_ids_but_reports_do_not(self) -> None:
        dataset = self._grouped_fixture()
        splits = split_dataset(dataset, seed=5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = write_split_outputs(splits, Path(temporary_directory) / "splits")
            rows = [
                json.loads(line)
                for path in output_dir.glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            split_text = "\n".join(json.dumps(row) for row in rows)
            report_text = (output_dir / "split_report.json").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC TEST ONLY", split_text)
        self.assertTrue(all("group_id" in row for row in rows))
        self.assertTrue(all(str(row["group_id"]).startswith("opaque_test_group_") for row in rows))
        self.assertNotIn("SYNTHETIC TEST ONLY", report_text)
        self.assertNotIn("opaque_test_group_", report_text)


if __name__ == "__main__":
    unittest.main()

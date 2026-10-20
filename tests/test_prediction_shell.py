from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from els_platform.interaction_log import append_prediction_record
from els_platform.prediction.demo import DemonstrationClassifier
from els_platform.prediction.schema import (
    EMOTION_LABELS,
    EmotionLabel,
    PredictionMode,
    PredictionOutcome,
    PredictionResult,
)


def make_result(
    scores: dict[EmotionLabel, float] | None = None,
    thresholds: dict[EmotionLabel, float] | None = None,
    *,
    mode: PredictionMode = PredictionMode.MODEL_INFERENCE,
) -> PredictionResult:
    if scores is None:
        scores = {label: 0.2 for label in EMOTION_LABELS}
    if thresholds is None:
        thresholds = {label: 0.5 for label in EMOTION_LABELS}
    return PredictionResult(
        scores=scores,
        thresholds=thresholds,
        model_name="Test model",
        model_version="test",
        prediction_mode=mode,
    )


class PredictionSchemaTests(unittest.TestCase):
    def test_empty_input_is_rejected(self) -> None:
        classifier = DemonstrationClassifier()
        for text in ("", "  \n\t"):
            with self.subTest(text=repr(text)):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    classifier.predict(text)

    def test_valid_prediction_schema(self) -> None:
        result = make_result()
        self.assertEqual(set(result.scores), set(EMOTION_LABELS))
        self.assertEqual(set(result.thresholds), set(EMOTION_LABELS))
        self.assertEqual(result.outcome, PredictionOutcome.NO_CLEAR_CATEGORY)

    def test_invalid_score_values_are_rejected(self) -> None:
        for invalid in (-0.01, 1.01, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                scores = {label: 0.2 for label in EMOTION_LABELS}
                scores[EmotionLabel.BORED] = invalid  # type: ignore[assignment]
                with self.assertRaises((TypeError, ValueError)):
                    make_result(scores=scores)

    def test_multiple_labels_are_selected_independently(self) -> None:
        scores = {label: 0.1 for label in EMOTION_LABELS}
        scores[EmotionLabel.BORED] = 0.61
        scores[EmotionLabel.CONFUSED] = 0.9
        result = make_result(scores=scores)
        self.assertEqual(
            result.selected_labels,
            (EmotionLabel.BORED, EmotionLabel.CONFUSED),
        )
        self.assertEqual(result.outcome, PredictionOutcome.LABELS_SELECTED)

    def test_no_clear_category_is_explicit(self) -> None:
        result = make_result()
        self.assertEqual(result.selected_labels, ())
        self.assertEqual(result.outcome, PredictionOutcome.NO_CLEAR_CATEGORY)

    def test_demo_prediction_identifies_its_mode(self) -> None:
        result = DemonstrationClassifier().predict("A non-empty demo input")
        self.assertTrue(result.is_demonstration)
        self.assertEqual(result.prediction_mode, PredictionMode.DEMONSTRATION)
        self.assertEqual(result.model_name, "Demonstration fixture")

    def test_safe_input_reference_rejects_raw_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "opaque identifier"):
            make_result().__class__(
                scores=make_result().scores,
                thresholds=make_result().thresholds,
                model_name="Test model",
                model_version="test",
                prediction_mode=PredictionMode.MODEL_INFERENCE,
                safe_input_ref="my raw learner text",
            )


class LoggingPrivacyTests(unittest.TestCase):
    def test_raw_text_is_not_written_to_log_and_logging_defaults_off(self) -> None:
        # Synthetic test fixture only; this is not a real learner observation.
        raw_text = "I am worried about failing this exam and my name is Sam."
        result = DemonstrationClassifier().predict(raw_text)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "interactions.csv"
            self.assertFalse(append_prediction_record(result, log_path))
            self.assertFalse(log_path.exists())

            self.assertTrue(append_prediction_record(result, log_path, enabled=True))
            contents = log_path.read_text(encoding="utf-8")
            self.assertNotIn(raw_text, contents)
            self.assertNotIn("Sam", contents)
            self.assertIn("Demonstration fixture", contents)


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_without_streamlit_exceptions(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("DEMONSTRATION MODE" in item.value for item in app.warning))

    def test_demo_button_handles_empty_and_nonempty_input(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=10)

        app.button[0].click().run(timeout=10)
        self.assertTrue(any("non-empty study challenge" in item.value for item in app.error))

        app.text_area[0].set_value("DEMONSTRATION TEST FIXTURE: I am stuck on one problem.")
        app.button[0].click().run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Illustrative demonstration scores" in item.value for item in app.markdown)
        )

    def test_analytics_page_is_explicitly_demonstration_only(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=10)
        app.segmented_control[0].set_value("Demonstration analytics").run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("illustrative fixtures" in item.value for item in app.warning)
        )


if __name__ == "__main__":
    unittest.main()

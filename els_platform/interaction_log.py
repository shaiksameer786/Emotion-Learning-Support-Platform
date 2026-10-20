"""Opt-in metadata-only CSV logging. Raw learner text is not accepted by this API."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .prediction.schema import PredictionResult


LOG_COLUMNS = (
    "timestamp_utc",
    "model_name",
    "model_version",
    "prediction_mode",
    "selected_labels",
    "scores",
)


def append_prediction_record(
    result: PredictionResult,
    path: str | Path,
    *,
    enabled: bool = False,
) -> bool:
    """Append aggregate prediction metadata only when explicitly enabled.

    The input text and ``safe_input_ref`` are intentionally excluded from the CSV.
    The caller must never pass raw text as ``path`` or as a prediction field.
    """
    if not enabled:
        return False

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": result.model_name,
        "model_version": result.model_version,
        "prediction_mode": result.prediction_mode.value,
        "selected_labels": json.dumps([label.value for label in result.selected_labels]),
        "scores": json.dumps(
            {label.value: result.scores[label] for label in result.scores},
            sort_keys=True,
        ),
    }
    exists = destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(record)
    return True

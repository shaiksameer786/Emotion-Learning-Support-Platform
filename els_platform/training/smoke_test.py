"""Offline CPU model-shape smoke check using synthetic test-only token IDs."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import BiLSTMModelConfig
from .models import build_bilstm_model
from .optional import OptionalDependencyError, require_torch


def run_bilstm_cpu_smoke() -> dict[str, object]:
    """Check one initialized BiLSTM forward pass and multi-label loss on fake tokens.

    No optimizer step, dataset access, checkpoint, or performance metric is produced.
    """
    torch = require_torch()
    torch.manual_seed(13)
    model_config = BiLSTMModelConfig(
        embedding_dim=8,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        max_vocabulary_size=16,
        min_token_frequency=1,
        hash_space_size=64,
    )
    model = build_bilstm_model(vocabulary_size=5, config=model_config)
    model.to(torch.device("cpu"))
    model.eval()
    # Explicitly synthetic TEST-ONLY token sequences; these are not learner text.
    input_ids = torch.tensor([[2, 3, 0, 0], [3, 4, 2, 0]], dtype=torch.long, device="cpu")
    targets = torch.zeros((2, 5), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        logits = model(input_ids=input_ids)
        if tuple(logits.shape) != (2, 5):
            raise RuntimeError("BiLSTM smoke output must have shape (2, 5).")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Synthetic multi-label smoke loss was not finite.")
    return {
        "status": "passed",
        "model": "bilstm",
        "device": "cpu",
        "output_shape": [2, 5],
        "fixture": "synthetic_test_only_token_ids",
        "optimizer_step": False,
        "performance_metrics": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an offline BiLSTM CPU forward/loss smoke check on synthetic test-only tokens."
    )
    parser.add_argument("--model", choices=("bilstm",), default="bilstm")
    parser.parse_args(argv)
    try:
        result = run_bilstm_cpu_smoke()
    except OptionalDependencyError as exc:
        print(f"SKIPPED: CPU framework smoke check not run. {exc}")
        return 0
    except Exception as exc:
        print(f"FAILED: CPU framework smoke check failed: {exc}", file=sys.stderr)
        return 1
    print(
        "PASSED: offline CPU forward/loss shape check; "
        "synthetic TEST-ONLY inputs; no optimizer step or performance metrics. "
        f"Output shape: {result['output_shape']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

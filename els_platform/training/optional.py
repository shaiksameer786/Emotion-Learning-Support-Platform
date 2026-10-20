"""Lazy imports for optional model-training packages."""

from __future__ import annotations


class OptionalDependencyError(RuntimeError):
    """Raised when a model operation needs packages not present in the base app."""


def require_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            "PyTorch is required for model training. Install the reviewed optional model "
            "requirements in the project environment; select the official Windows CPU build "
            "if no supported GPU is available."
        ) from exc
    return torch


def require_transformers():
    try:
        import transformers
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            "Transformers is required for BERT. Install the reviewed optional model requirements."
        ) from exc
    return transformers

"""Training-only, hashed-token vocabulary for the BiLSTM baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

from .config import BiLSTMModelConfig
from .splitting import DatasetSplits


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Simple Unicode tokenization; case-folding is for model features only."""
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _bucket(token: str, size: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


@dataclass(frozen=True, slots=True)
class HashedVocabulary:
    """Maps only buckets observed in training to IDs; raw vocabulary tokens are not stored."""

    hash_space_size: int
    bucket_ids: tuple[int, ...]

    PAD_ID = 0
    UNK_ID = 1

    def __post_init__(self) -> None:
        if self.hash_space_size < 1:
            raise ValueError("hash_space_size must be positive.")
        if len(set(self.bucket_ids)) != len(self.bucket_ids):
            raise ValueError("bucket_ids must be unique.")
        if any(not 0 <= bucket < self.hash_space_size for bucket in self.bucket_ids):
            raise ValueError("bucket_ids must fit inside hash_space_size.")

    @classmethod
    def fit(cls, training_texts: Iterable[str], config: BiLSTMModelConfig) -> "HashedVocabulary":
        counts: Counter[int] = Counter()
        for text in training_texts:
            counts.update(_bucket(token, config.hash_space_size) for token in tokenize(text))
        chosen = [
            bucket
            for bucket, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= config.min_token_frequency
        ][: config.max_vocabulary_size - 2]
        return cls(hash_space_size=config.hash_space_size, bucket_ids=tuple(chosen))

    @classmethod
    def fit_training_split(
        cls,
        splits: DatasetSplits,
        config: BiLSTMModelConfig,
    ) -> "HashedVocabulary":
        """Fit using train examples only; validation and test are never inspected."""
        return cls.fit((example.text for example in splits.train), config)

    @property
    def size(self) -> int:
        return len(self.bucket_ids) + 2

    def encode(self, text: str, *, max_length: int) -> tuple[int, ...]:
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        index = {bucket: offset + 2 for offset, bucket in enumerate(self.bucket_ids)}
        tokens = tokenize(text)[:max_length]
        encoded = [index.get(_bucket(token, self.hash_space_size), self.UNK_ID) for token in tokens]
        encoded.extend([self.PAD_ID] * (max_length - len(encoded)))
        return tuple(encoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "sha256_hashed_training_bucket_v1",
            "hash_space_size": self.hash_space_size,
            "bucket_ids": list(self.bucket_ids),
            "pad_id": self.PAD_ID,
            "unknown_id": self.UNK_ID,
            "vocabulary_size": self.size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HashedVocabulary":
        if payload.get("format") != "sha256_hashed_training_bucket_v1":
            raise ValueError("Unsupported hashed vocabulary format.")
        raw_buckets = payload.get("bucket_ids")
        if not isinstance(raw_buckets, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw_buckets
        ):
            raise ValueError("bucket_ids must be a list of integers.")
        raw_size = payload.get("hash_space_size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError("hash_space_size must be an integer.")
        return cls(raw_size, tuple(raw_buckets))

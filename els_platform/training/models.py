"""Lazy model builders for BiLSTM and locally available BERT-family encoders."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BertModelConfig, BiLSTMModelConfig
from .optional import require_torch, require_transformers


class CheckpointUnavailableError(RuntimeError):
    """Raised when a reviewed BERT checkpoint is absent or unusable locally."""


@dataclass(frozen=True, slots=True)
class BertComponents:
    tokenizer: object
    model: object


def build_bilstm_model(vocabulary_size: int, config: BiLSTMModelConfig):
    """Create a bidirectional LSTM with five raw logits (sigmoid is applied at evaluation)."""
    if isinstance(vocabulary_size, bool) or vocabulary_size < 3:
        raise ValueError("vocabulary_size must include padding, unknown, and at least one token.")
    torch = require_torch()
    nn = torch.nn

    class BiLSTMClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(config.hidden_size * 2, 5)

        def forward(self, input_ids):
            lengths = input_ids.ne(0).sum(dim=1).clamp(min=1).to("cpu")
            embedded = self.embedding(input_ids)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            representation = torch.cat((forward_hidden, backward_hidden), dim=1)
            return self.output(self.dropout(representation))

    return BiLSTMClassifier()


def load_bert_components(config: BertModelConfig) -> BertComponents:
    """Load only a reviewed, pinned checkpoint already present in the local cache/path."""
    config.validate_checkpoint_review()
    require_torch()
    transformers = require_transformers()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.checkpoint_id,
            revision=config.revision,
            local_files_only=True,
        )
        encoder = transformers.AutoModel.from_pretrained(
            config.checkpoint_id,
            revision=config.revision,
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        raise CheckpointUnavailableError(
            "Reviewed BERT checkpoint is unavailable locally; no download was attempted."
        ) from exc
    hidden_size = getattr(encoder.config, "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise CheckpointUnavailableError("The selected encoder does not expose a valid hidden_size.")
    position_limit = getattr(encoder.config, "max_position_embeddings", None)
    if (
        isinstance(position_limit, int)
        and not isinstance(position_limit, bool)
        and config.max_sequence_length > position_limit
    ):
        raise CheckpointUnavailableError(
            "Configured max_sequence_length exceeds this encoder's max_position_embeddings."
        )
    nn = require_torch().nn
    dropout_probability = float(getattr(encoder.config, "hidden_dropout_prob", 0.1))

    class BertMultiLabelClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout_probability)
            self.output = nn.Linear(hidden_size, 5)

        def forward(self, **model_inputs):
            outputs = self.encoder(**model_inputs)
            return self.output(self.dropout(outputs.last_hidden_state[:, 0, :]))

    return BertComponents(tokenizer=tokenizer, model=BertMultiLabelClassifier())

"""
EHR transformer backbone used both for MLM pretraining and downstream
binary-classification finetuning.

Each encounter is encoded with three additive embeddings:
  - event embedding over the unified clinical-concept vocabulary
  - type embedding over the categorical token-type stream
  - time embedding over day-offset bins
Embeddings are summed, layer-normed via the embedding dropout, and
forwarded through `num_hidden_layers` pre-LN transformer blocks. The
[CLS] hidden state pools either to an MLM head or to a binary
downstream classifier.

`TransformerBlock` must expose `self_attn.W_Q`, `self_attn.W_K`,
`self_attn.W_V`, `self_attn.W_output`, and `pos_ffn.fc1`, `pos_ffn.fc2`
as separate `nn.Linear` submodules. LoRA injection targets these
suffix paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    """Sum of event + optional type + optional time embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.type_embeddings = (
            nn.Embedding(int(config.type_vocab_size), config.hidden_size, padding_idx=0)
            if getattr(config, "type_vocab_size", None) is not None
            else None
        )
        self.time_embeddings = (
            nn.Embedding(int(config.time_vocab_size), config.hidden_size, padding_idx=0)
            if getattr(config, "time_vocab_size", None) is not None
            else None
        )
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_types=None, time_ids=None):
        x = self.word_embeddings(input_ids)
        if self.type_embeddings is not None and token_types is not None:
            t = token_types.long().clamp(0, self.type_embeddings.num_embeddings - 1)
            x = x + self.type_embeddings(t)
        if self.time_embeddings is not None and time_ids is not None:
            t = time_ids.long().clamp(0, self.time_embeddings.num_embeddings - 1)
            x = x + self.time_embeddings(t)
        return self.emb_dropout(x)


class TransformerStack(nn.Module):
    """Stack of identical pre-LN transformer blocks."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [block_cls(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x, attn_mask):
        for block in self.transformer_blocks:
            x = block(x, attn_mask)
        return x


class MaskedPredictionHead(nn.Module):
    """Two-layer MLM head over the unified vocabulary."""

    def __init__(self, config, voc_size):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, voc_size),
        )

    def forward(self, x):
        return self.cls(x)


class BinaryPredictionHead(nn.Module):
    """Single logit linear head used for downstream binary tasks."""

    def __init__(self, config):
        super().__init__()
        self.cls = nn.Linear(int(config.hidden_size), 1)

    def forward(self, x):
        return self.cls(x)


class PretrainModel(nn.Module):
    """Backbone + MLM head; loss returned by forward."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = TransformerStack(config, block_cls)
        self.mask_token_id = int(config.mask_token_id)
        self.event_cls = MaskedPredictionHead(config, int(config.label_vocab_size))
        self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, input_ids, token_types, visit_positions, masked_labels):
        pad_mask = input_ids > 0
        attn_mask = ~pad_mask.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        x = self.embeddings(input_ids, token_types, visit_positions)
        x = self.transformer(x, attn_mask)
        # Select hidden states at masked positions and predict the gold token id.
        masked_x = x[input_ids == self.mask_token_id]
        if masked_x.numel() == 0:
            return 0.0 * x.sum()
        logits = self.event_cls(masked_x)
        labels = masked_labels.to(logits.dtype).to(logits.device)
        return self.loss_fn(logits, labels)


class FinetuneModel(nn.Module):
    """Backbone + binary classifier head over the pooled [CLS] state."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = TransformerStack(config, block_cls)
        self.downstream_cls = BinaryPredictionHead(config)

    def load_weight(self, state_dict):
        """Partial load: copy by-name where both sides have matching shapes."""
        own = dict(self.named_parameters())
        for key, value in state_dict.items():
            if key in own and own[key].shape == value.shape:
                own[key].data.copy_(value)

    def forward(self, input_ids, token_types, edge_index, visit_positions, labeled_ids):
        pad_mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1)
        x = self.embeddings(input_ids, token_types, visit_positions)
        x = self.transformer(x, ~pad_mask)
        # labeled_ids = (batch_idx, pos_idx); pos_idx points at [CLS] (=0).
        cls_hidden = x[labeled_ids][:, 0]
        return self.downstream_cls(cls_hidden)

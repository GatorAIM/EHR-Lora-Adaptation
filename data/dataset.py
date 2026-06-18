"""
Torch-side datasets and the collate factory shared by MLM pretraining
and downstream finetuning.

Each row of the preprocessed DataFrame carries three aligned lists for
one encounter:
  Events : clinical-concept tokens (strings)
  Type   : integer type ids, same length as Events
  Time   : integer day-offset ids, same length as Events

[CLS] is prepended and [SEP] appended at tokenisation time so the model
always pools the [CLS] hidden state for downstream tasks.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _as_list(x):
    """Coerce DataFrame cells (None / scalar / iterable) into a list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except Exception:
        return [x]


def _row_streams(row) -> Tuple[List[str], List[int], List[int]]:
    """Pull the (Events, Type, Time) triple and assert aligned lengths."""
    events = [str(t) for t in _as_list(row["Events"])]
    types = [int(t) for t in _as_list(row["Type"])]
    times = [int(t) for t in _as_list(row["Time"])]
    if not (len(types) == len(times) == len(events)):
        raise ValueError("Events / Type / Time must share the same length.")
    return events, types, times


def _wrap_with_special_tokens(events, types, times):
    """Prepend [CLS] and append [SEP] with default type=0 and edge times."""
    cls_type = sep_type = 0
    baseline_time = times[0] if times else 0
    sep_time = times[-1] if times else baseline_time
    return (
        ["[CLS]"] + events + ["[SEP]"],
        [cls_type] + types + [sep_type],
        [baseline_time] + times + [sep_time],
    )


class FinetuneEHRDataset(Dataset):
    """Returns one encounter ready for binary classification."""

    def __init__(self, df, tokenizer, *, task: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.task = str(task)

    def __len__(self) -> int:
        return len(self.df)

    def _label(self, row) -> float:
        if self.task not in row:
            raise KeyError(f"Missing label column {self.task!r}")
        return float(row[self.task])

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        events, types, times = _row_streams(row)
        tokens, types, times = _wrap_with_special_tokens(events, types, times)
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long
        )
        token_types = torch.tensor(types, dtype=torch.long)
        visit_positions = torch.tensor(times, dtype=torch.long)
        label = torch.tensor([self._label(row)], dtype=torch.float32)
        edge_index = None  # placeholder kept for backward signature compatibility
        return input_ids, token_types, edge_index, visit_positions, label


class PretrainEHRDataset(Dataset):
    """
    Returns one encounter for MLM. Mask sampling happens here so the
    indices are deterministic given the dataset rng state, and the
    masked positions are recorded as one-hot label rows.
    """

    def __init__(self, df, tokenizer, *, mask_rate: float = 0.15):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mask_rate = float(mask_rate)
        self._mask_id = tokenizer.vocab.word2idx["[MASK]"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        events, types, times = _row_streams(row)
        tokens, types, times = _wrap_with_special_tokens(events, types, times)
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long
        )
        token_types = torch.tensor(types, dtype=torch.long)
        visit_positions = torch.tensor(times, dtype=torch.long)

        # Sample masked positions among non-special tokens only.
        candidates = list(range(1, max(1, input_ids.numel() - 1)))
        n_to_mask = max(1, int(len(candidates) * self.mask_rate)) if candidates else 0
        n_to_mask = min(n_to_mask, len(candidates))
        masked_positions = sorted(random.sample(candidates, n_to_mask)) if n_to_mask else []

        vocab_size = len(self.tokenizer.vocab.idx2word)
        masked_labels = torch.zeros((len(masked_positions), vocab_size), dtype=torch.float32)
        for j, pos in enumerate(masked_positions):
            gold = int(input_ids[pos].item())
            if 0 <= gold < vocab_size:
                masked_labels[j, gold] = 1.0
            input_ids[pos] = self._mask_id
        return input_ids, token_types, visit_positions, masked_labels


def batcher(tokenizer, *, is_train: bool = False):
    """
    Build a collate function for either finetune or pretrain batches.

    Each branch pads the variable-length 1D tensors to the batch's max
    length with the tokenizer's [PAD] id, stacks them, and assembles the
    extra structures each model expects (labeled_ids for finetune; the
    concatenated masked_labels for pretrain).
    """
    pad_id = tokenizer.vocab.word2idx.get("[PAD]", 0)

    def _pad_1d(xs: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(int(x.numel()) for x in xs) if xs else 0
        return torch.stack(
            [F.pad(x, (0, max_len - int(x.numel())), value=pad_id) for x in xs],
            dim=0,
        )

    def _collate_finetune(batch):
        input_ids = _pad_1d([b[0] for b in batch])
        token_types = _pad_1d([b[1] for b in batch])
        visit_positions = _pad_1d([b[3] for b in batch])
        labels = torch.cat([b[4] for b in batch], dim=0)
        B = int(input_ids.size(0))
        labeled_ids = (
            torch.arange(B, dtype=torch.long).view(B, 1),
            torch.zeros((B, 1), dtype=torch.long),
        )
        return input_ids, token_types, None, visit_positions, labeled_ids, labels

    def _collate_pretrain(batch):
        input_ids = _pad_1d([b[0] for b in batch])
        token_types = _pad_1d([b[1] for b in batch])
        visit_positions = _pad_1d([b[2] for b in batch])
        masked_labels = torch.cat([b[3] for b in batch], dim=0)
        return input_ids, token_types, visit_positions, masked_labels

    def _collate(batch):
        if not batch:
            raise ValueError("Empty batch")
        first = batch[0]
        # Finetune items are length-5 with edge_index=None at position 2;
        # pretrain items are length-4.
        if len(first) == 5 and first[2] is None:
            return _collate_finetune(batch)
        if len(first) == 4:
            return _collate_pretrain(batch)
        raise ValueError(f"Unexpected sample shape: len={len(first)}")

    return _collate

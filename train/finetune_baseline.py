"""
Non-LoRA finetuning baselines: freeze-all-but-head, tune-last-N,
full finetune. They share the optimizer / loop / selection logic with
`finetune_lora` and differ only in which backbone parameters carry
`requires_grad = True`.
"""

from __future__ import annotations

from typing import Optional

import torch

from model.ehr_backbone import FinetuneModel
from train.finetune_lora import finetune_lora as _finetune_loop


def build_finetune_model(config, block_cls, base_state, *, device) -> FinetuneModel:
    """Allocate the model and partial-load the pretrained backbone."""
    model = FinetuneModel(config, block_cls)
    model.load_weight(base_state)
    return model.to(device)


def configure_freeze_all(model: FinetuneModel) -> None:
    """Train only the downstream classifier head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.downstream_cls.parameters():
        p.requires_grad = True


def configure_tune_last_n(model: FinetuneModel, n: int) -> None:
    """Train the head plus the last `n` transformer blocks."""
    configure_freeze_all(model)
    blocks = list(model.transformer.transformer_blocks)
    for block in blocks[max(0, len(blocks) - int(n)):]:
        for p in block.parameters():
            p.requires_grad = True


def configure_full_finetune(model: FinetuneModel) -> None:
    """Train every parameter in the backbone plus the head."""
    for p in model.parameters():
        p.requires_grad = True


def finetune_baseline(
    model: FinetuneModel,
    strategy: str,
    train_loader,
    val_loader,
    test_loader,
    *,
    device,
    lr: float,
    weight_decay: float,
    epochs: int,
    eval_every: int,
    patience: int,
    min_delta: float,
    pos_weight: Optional[float],
    ece_bins: int,
    n_last_layers: int = 2,
):
    """
    Apply the requested trainable-parameter configuration, then call
    into the shared training loop. Strategy must be one of
    {"freeze_all", "tune_last_n", "full_finetune"}.
    """
    if strategy == "freeze_all":
        configure_freeze_all(model)
    elif strategy == "tune_last_n":
        configure_tune_last_n(model, n_last_layers)
    elif strategy == "full_finetune":
        configure_full_finetune(model)
    else:
        raise ValueError(f"Unknown adaptation strategy: {strategy!r}")

    return _finetune_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        eval_every=eval_every,
        patience=patience,
        min_delta=min_delta,
        pos_weight=pos_weight,
        ece_bins=ece_bins,
    )

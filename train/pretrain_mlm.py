"""
Masked-language-model pretraining for the EHR transformer backbone.

A random subset of input tokens is replaced with [MASK]; the model
predicts the original token id at those positions via the MLM head.
After pretraining, the backbone is reused (with a downstream head) for
every adaptation strategy in `train/`.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from model.ehr_backbone import PretrainModel


def make_pretrain_optimizer(model, *, lr: float, weight_decay: float):
    """AdamW over all parameters; pretraining unfreezes everything."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )


def pretrain_step(model: PretrainModel, batch, optimizer, device) -> float:
    """One MLM step: forward -> loss -> backward -> optimizer.step."""
    batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
    loss = model(*batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


@torch.no_grad()
def evaluate_mlm(model: PretrainModel, loader: DataLoader, device) -> float:
    """Average MLM loss over a held-out dataloader."""
    model.eval()
    total, steps = 0.0, 0
    for batch in loader:
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        total += float(model(*batch).detach().cpu().item())
        steps += 1
    return total / max(1, steps)


def run_mlm_pretraining(
    model: PretrainModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device,
    lr: float,
    weight_decay: float,
    epochs: int,
    eval_every: int = 1,
):
    """
    Outer pretraining loop. Tracks the best-by-val-loss snapshot so the
    downstream pipeline always seeds itself from the best generalising
    backbone instead of the last-epoch one.
    """
    optimizer = make_pretrain_optimizer(model, lr=lr, weight_decay=weight_decay)
    best = {"loss": float("inf"), "state": None, "epoch": None}

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            pretrain_step(model, batch, optimizer, device)
        if int(eval_every) > 0 and epoch % int(eval_every) == 0:
            val_loss = evaluate_mlm(model, val_loader, device)
            if val_loss < best["loss"]:
                best.update(
                    loss=float(val_loss),
                    state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    epoch=int(epoch),
                )

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    return best

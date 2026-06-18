"""
LoRA finetuning of an MLM-pretrained EHR backbone for a single binary
downstream task.

Pipeline:
  1. Build `FinetuneModel`, partial-load the pretrained backbone.
  2. Inject LoRA adapters into the last `n` transformer blocks at the
     requested target submodules.
  3. Freeze the rest of the backbone; mark LoRA + classifier head as
     trainable.
  4. Train with BCE-with-logits, optionally with a positive-class
     weight for imbalanced tasks.
  5. Select the best checkpoint by validation accuracy and reload it
     for the final test-set scoring.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.ehr_backbone import FinetuneModel
from model.lora import (
    freeze_base_params,
    inject_lora,
    mark_lora_trainable,
)
from eval.metrics import evaluate_all


def build_model_with_lora(config, block_cls, base_state, *,
                          targets: Sequence[str],
                          last_n_layers: int,
                          rank: int,
                          alpha: int,
                          dropout: float,
                          device) -> FinetuneModel:
    """Allocate the model, load the pretrained backbone, inject LoRA."""
    model = FinetuneModel(config, block_cls)
    model.load_weight(base_state)
    model = model.to(device)

    n_blocks = len(model.transformer.transformer_blocks)
    layer_idxs = list(range(n_blocks - int(last_n_layers), n_blocks))
    replaced = inject_lora(
        model,
        target_suffixes=list(targets),
        layer_idxs=layer_idxs,
        r=int(rank),
        alpha=int(alpha),
        dropout=float(dropout),
    )
    if not replaced:
        raise RuntimeError("LoRA injection matched no modules.")

    freeze_base_params(model)
    mark_lora_trainable(model)
    for p in model.downstream_cls.parameters():
        p.requires_grad = True
    return model


def make_optimizer(model, *, lr: float, weight_decay: float):
    """AdamW over exactly the parameters whose requires_grad is True."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=float(lr), weight_decay=float(weight_decay))


def train_one_epoch(model, loader: DataLoader, optimizer, device, *,
                    pos_weight: Optional[float] = None) -> float:
    """Standard BCE-with-logits training step over one epoch."""
    model.train()
    pw = (torch.tensor(float(pos_weight), device=device, dtype=torch.float32)
          if pos_weight is not None else None)
    total, steps = 0.0, 0
    for batch in loader:
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        labels = batch[-1].float().view(-1)
        logits = model(*batch[:-1]).view(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
                if pw is not None else
                F.binary_cross_entropy_with_logits(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu().item())
        steps += 1
    return total / max(1, steps)


def finetune_lora(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
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
):
    """
    Train loop with best-by-val-acc selection and early stopping.

    Returns a dict with:
      best_state              - state_dict of the best snapshot
      best_epoch              - epoch index where the snapshot was taken
      best_val_metric         - validation metric block at selection
      best_test_metric        - test metric block at selection
      final_test_metric       - test metric block after reloading best
    """
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    best = {"acc": -1.0, "state": None, "epoch": None, "val": None, "test": None}
    bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)
        if int(eval_every) > 0 and epoch % int(eval_every) != 0:
            continue
        val_metric = evaluate_all(model, val_loader, device, ece_bins=ece_bins)
        test_metric = evaluate_all(model, test_loader, device, ece_bins=ece_bins)
        if val_metric["acc"] > best["acc"] + float(min_delta):
            best.update(
                acc=val_metric["acc"],
                state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                epoch=int(epoch),
                val=dict(val_metric),
                test=dict(test_metric),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    final_test = evaluate_all(model, test_loader, device, ece_bins=ece_bins)
    return {
        "best_state": best["state"],
        "best_epoch": best["epoch"],
        "best_val_metric": best["val"],
        "best_test_metric_at_selection": best["test"],
        "final_test_metric_after_reload": final_test,
    }

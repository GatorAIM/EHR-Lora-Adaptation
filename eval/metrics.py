"""
Discrimination (Accuracy / Precision / Recall / F1 / AUROC / AUPRC) and
calibration (Brier / ECE) metrics. A single inference pass collects
logits + labels once and derives every metric from them, so all numbers
in the per-run sidecar come from the exact same set of predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# calibration primitives
# ---------------------------------------------------------------------------

def brier_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score = mean((y_prob - y_true)^2). No hyperparameters."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int
) -> float:
    """
    Equal-width-bin ECE on the positive-class probability. The final
    bin is inclusive on both endpoints so the boundary value 1.0 is
    counted.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(y_true.size)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = ((y_prob >= lo) & (y_prob <= hi)) if i == n_bins - 1 \
               else ((y_prob >= lo) & (y_prob < hi))
        m = int(mask.sum())
        if m == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += abs(acc - conf) * (m / n)
    return float(ece)


# ---------------------------------------------------------------------------
# discrimination block
# ---------------------------------------------------------------------------

def discrimination_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute Accuracy / Precision / Recall / F1 at `threshold`, plus AUROC / AUPRC."""
    pred = (y_prob >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# end-to-end inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all(model, dataloader, device, *, ece_bins: int) -> Dict[str, float]:
    """
    Inference pass under model.eval() + no_grad. Aggregate every batch's
    logits and labels into two arrays, sigmoid the logits, and return a
    dict containing both discrimination and calibration numbers from the
    same logits in one go.
    """
    model.eval()
    logits, labels = [], []
    for batch in dataloader:
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        y = batch[-1].view(-1)
        logits.append(model(*batch[:-1]).view(-1))
        labels.append(y)
    logits = torch.cat(logits, dim=0).detach().cpu().numpy().astype(np.float64)
    labels = torch.cat(labels, dim=0).detach().cpu().numpy().astype(np.float64)
    y_prob = 1.0 / (1.0 + np.exp(-logits))

    out = discrimination_metrics(labels, y_prob)
    out["brier"] = brier_binary(labels, y_prob)
    out["ece"] = expected_calibration_error(labels, y_prob, n_bins=int(ece_bins))
    return out


def evaluate_all_with_bins(
    model, dataloader, device, *, ece_bin_list
) -> Dict[str, float]:
    """
    Like `evaluate_all` but reports ECE at multiple bin counts in the
    same pass. Useful for showing that the calibration story is robust
    to the bin-count choice.
    """
    out = evaluate_all(model, dataloader, device, ece_bins=int(ece_bin_list[0]))
    # Re-derive y_prob / y_true from a second pass-free recomputation.
    model.eval()
    with torch.no_grad():
        logits, labels = [], []
        for batch in dataloader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            logits.append(model(*batch[:-1]).view(-1))
            labels.append(batch[-1].view(-1))
        logits = torch.cat(logits).cpu().numpy().astype(np.float64)
        labels = torch.cat(labels).cpu().numpy().astype(np.float64)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    for k in ece_bin_list:
        out[f"ece_b{int(k)}"] = expected_calibration_error(labels, y_prob, n_bins=int(k))
    return out


# ---------------------------------------------------------------------------
# sidecar persistence
# ---------------------------------------------------------------------------

def write_metrics_sidecar(
    ckpt_path: str,
    metrics: Dict[str, float],
    *,
    adapter_config: Optional[Dict] = None,
    extras: Optional[Dict] = None,
) -> str:
    """
    Persist `<ckpt_path>.metrics.json` alongside the checkpoint. The
    sidecar records the metric block, optional adapter configuration
    so an external evaluator can rebuild the same architecture, and
    any extra bookkeeping such as best_epoch and seed.
    """
    payload = {"metrics": dict(metrics)}
    if adapter_config is not None:
        payload["adapter"] = dict(adapter_config)
    if extras:
        payload.update(extras)
    out = Path(str(ckpt_path) + ".metrics.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)

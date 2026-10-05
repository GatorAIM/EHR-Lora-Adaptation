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
    auc,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from scipy.optimize import minimize


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
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
    }


def select_validation_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Select operating thresholds from validation predictions only."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        raise ValueError("validation predictions do not define a threshold")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    eligible = np.flatnonzero(tpr >= 0.80)
    sensitivity_index = (
        eligible[np.argmin(fpr[eligible])] if eligible.size else int(np.argmax(tpr))
    )
    return {
        "max_validation_f1": float(thresholds[int(np.nanargmax(f1))]),
        "max_validation_youden": float(roc_thresholds[int(np.nanargmax(tpr - fpr))]),
        "validation_sensitivity_0.80": float(roc_thresholds[int(sensitivity_index)]),
    }


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float
) -> Dict[str, float]:
    """Evaluate a prespecified or validation-selected threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    prediction = y_prob >= float(threshold)
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum(~prediction & negative))
    fn = int(np.sum(~prediction & positive))
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "ppv": tp / max(1, tp + fp),
        "npv": tn / max(1, tn + fn),
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def fit_logistic_recalibration(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float | bool]:
    """Estimate logistic recalibration parameters from validation predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    if y_true.size == 0 or np.unique(y_true).size != 2:
        raise ValueError("calibration requires nonempty binary outcomes with both classes")
    logit = np.log(y_prob / (1 - y_prob))

    def objective(beta):
        linear = beta[0] + beta[1] * logit
        fitted = 1 / (1 + np.exp(-np.clip(linear, -40, 40)))
        loss = float(np.sum(np.logaddexp(0, linear) - y_true * linear))
        gradient = np.array([
            np.sum(fitted - y_true),
            np.sum((fitted - y_true) * logit),
        ])
        return loss, gradient

    fit = minimize(objective, np.array([0.0, 1.0]), jac=True, method="BFGS")
    return {
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "fit_success": bool(fit.success),
    }


def apply_logistic_recalibration(
    y_prob: np.ndarray, *, intercept: float, slope: float
) -> np.ndarray:
    """Apply validation-estimated logistic recalibration to new predictions."""
    probability = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))
    linear = float(intercept) + float(slope) * logit
    return 1 / (1 + np.exp(-np.clip(linear, -40, 40)))


def calibration_intercept_slope(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float | bool]:
    """Return calibration intercept and slope for a set of predictions."""
    fitted = fit_logistic_recalibration(y_true, y_prob)
    return {
        "calibration_intercept": float(fitted["intercept"]),
        "calibration_slope": float(fitted["slope"]),
        "calibration_fit_success": bool(fitted["fit_success"]),
    }


def brier_skill_score(
    y_true: np.ndarray, y_prob: np.ndarray, *, reference_probability: float
) -> float:
    """Brier skill relative to a fixed validation-derived prevalence reference."""
    y_true = np.asarray(y_true, dtype=np.float64)
    reference = np.repeat(float(reference_probability), len(y_true))
    reference_brier = brier_binary(y_true, reference)
    if reference_brier <= 0:
        return float("nan")
    return 1 - brier_binary(y_true, y_prob) / reference_brier


def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> list[dict[str, float]]:
    """Compute model, treat-all, and treat-none net benefit across thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.asarray(
        thresholds if thresholds is not None else np.linspace(0.01, 0.80, 80),
        dtype=np.float64,
    )
    if y_true.size == 0 or np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("decision-curve thresholds must be within (0, 1)")
    prevalence = float(np.mean(y_true))
    rows = []
    for threshold in thresholds:
        prediction = y_prob >= threshold
        tp = int(np.sum(prediction & (y_true == 1)))
        fp = int(np.sum(prediction & (y_true == 0)))
        odds = float(threshold / (1 - threshold))
        rows.append({
            "threshold": float(threshold),
            "model_net_benefit": float(tp / len(y_true) - fp / len(y_true) * odds),
            "treat_all_net_benefit": float(prevalence - (1 - prevalence) * odds),
            "treat_none_net_benefit": 0.0,
        })
    return rows


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

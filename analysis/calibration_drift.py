"""
Paired calibration-drift contrasts between in-site and cross-site runs.

For each (method, task) pair the analysis layer holds two streams of
runs that share the same pretrained backbone:
  - main : trained on a source / combined cohort, scored on the *target*
           site's test set, with no target-site training.
  - trans: trained directly on the *target* site and scored on the same
           target test set.

Both streams use the same target test set so the paired difference
`delta = main - trans` quantifies the calibration loss that comes from
skipping target-site adaptation. Brier and ECE are "lower is better",
so a positive delta means the cross-site model is worse calibrated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------

def pair_runs_by_method_task(
    main_records: List[dict],
    trans_records: List[dict],
) -> Dict[Tuple[str, str], Dict[str, Dict[int, dict]]]:
    """
    Index runs as `paired[(method, task)] = {"main": {seed: row}, "trans": {seed: row}}`.
    Seeds that exist on only one side are kept; downstream paired
    operations skip them.
    """
    bucket: Dict[Tuple[str, str], Dict[str, Dict[int, dict]]] = defaultdict(
        lambda: {"main": {}, "trans": {}}
    )
    for rec in main_records:
        if rec.get("seed") is None:
            continue
        bucket[(rec["method"], rec["task"])]["main"][int(rec["seed"])] = rec
    for rec in trans_records:
        if rec.get("seed") is None:
            continue
        bucket[(rec["method"], rec["task"])]["trans"][int(rec["seed"])] = rec
    return bucket


# ---------------------------------------------------------------------------
# paired delta
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[float, float, int]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(float(v))])
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def paired_delta(
    paired_bucket: Dict[Tuple[str, str], Dict[str, Dict[int, dict]]],
    *,
    metric: str,
) -> pd.DataFrame:
    """
    For each (method, task) compute the per-seed delta `main - trans`
    on `metric`, then aggregate to mean / std / n. Also exposes the two
    absolute endpoints (`main_mean`, `trans_mean`) so callers can draw
    the slope chart from a single table.
    """
    rows = []
    for (method, task), sides in sorted(paired_bucket.items()):
        seeds = sorted(set(sides["main"]) & set(sides["trans"]))
        deltas, mains, transes = [], [], []
        for seed in seeds:
            m_val = sides["main"][seed].get(metric)
            t_val = sides["trans"][seed].get(metric)
            if m_val is None or t_val is None:
                continue
            mains.append(float(m_val))
            transes.append(float(t_val))
            deltas.append(float(m_val) - float(t_val))
        d_mean, d_std, n = _mean_std(deltas)
        m_mean, _, _ = _mean_std(mains)
        t_mean, _, _ = _mean_std(transes)
        rows.append({
            "method": method,
            "task": task,
            "n_seeds": n,
            f"{metric}_main_mean": m_mean,
            f"{metric}_trans_mean": t_mean,
            f"{metric}_delta_mean": d_mean,
            f"{metric}_delta_std": d_std,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# drift verdict
# ---------------------------------------------------------------------------

def classify_drift_status(delta_mean: float, delta_std: float) -> str:
    """
    Lightweight verdict used to colour figures and to partition tasks
    into "drift-affected" vs "no-drift exception" groups:
      - "drift-affected"     : delta_mean > 0 and |delta_mean| > delta_std.
      - "no-drift exception" : otherwise.

    Lower-is-better metrics (Brier, ECE) require a positive paired delta
    for drift to exist, which is why the verdict uses `> 0` directly.
    """
    if np.isnan(delta_mean) or np.isnan(delta_std):
        return "no-drift exception"
    if delta_mean > 0 and abs(delta_mean) > delta_std:
        return "drift-affected"
    return "no-drift exception"


def build_delta_table(
    paired_bucket: Dict[Tuple[str, str], Dict[str, Dict[int, dict]]],
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """
    Concatenate per-metric paired-delta tables into one wide frame and
    attach a `verdict` column derived from the primary metric (the
    first entry of `metrics`).
    """
    if not metrics:
        raise ValueError("metrics must be non-empty")
    parts = [paired_delta(paired_bucket, metric=m) for m in metrics]
    out = parts[0]
    for other in parts[1:]:
        out = out.merge(other, on=["method", "task", "n_seeds"], how="outer")
    primary = metrics[0]
    out["verdict"] = [
        classify_drift_status(row[f"{primary}_delta_mean"],
                              row[f"{primary}_delta_std"])
        for _, row in out.iterrows()
    ]
    return out


def summarise_across_drift_tasks(
    delta_table: pd.DataFrame,
    *,
    metric: str,
    drop_exceptions: bool = True,
) -> pd.DataFrame:
    """
    Average the paired delta across tasks for each method. By default
    the "no-drift exception" tasks are dropped so the headline number
    is computed only on tasks where drift exists.
    """
    df = delta_table.copy()
    if drop_exceptions and "verdict" in df.columns:
        df = df[df["verdict"] == "drift-affected"]
    grouped = (
        df.groupby("method", as_index=False)[f"{metric}_delta_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": f"{metric}_delta_grand_mean",
            "std": f"{metric}_delta_grand_std",
            "count": "n_tasks",
        })
    )
    return grouped

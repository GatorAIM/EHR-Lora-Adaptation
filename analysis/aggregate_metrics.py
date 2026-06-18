"""
Per-run sidecars to figure-ready tables.

Every downstream / external run emits a `<ckpt>.metrics.json` sidecar
alongside its checkpoint. This module discovers sidecars on disk,
loads them, groups by (task, method, seed), reduces over seeds, and
writes wide CSV tables in the schema consumed by the paper figures.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_SEED_RE = re.compile(r"_seed(\d+)")


# ---------------------------------------------------------------------------
# discovery / loading
# ---------------------------------------------------------------------------

def discover_metric_sidecars(root: str, *, pattern: str = "*.pt.metrics.json"
                             ) -> List[Path]:
    """Recursively glob all metric sidecars under `root`."""
    return sorted(Path(root).rglob(pattern))


def load_sidecar(path: Path) -> dict:
    """Parse one sidecar JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_seed_from_name(name: str) -> Optional[int]:
    """Extract seed integer from a filename containing `_seed<int>`."""
    m = _SEED_RE.search(name)
    return int(m.group(1)) if m else None


def extract_record(sidecar_path: Path, *, method: str) -> dict:
    """
    Flatten one sidecar into a row with stable keys for downstream
    aggregation. `task` is taken from the sidecar payload when present
    and falls back to the parent directory name.
    """
    payload = load_sidecar(sidecar_path)
    metrics = payload.get("metrics") or payload.get("final_test_metric_after_reload") or {}
    return {
        "method": method,
        "task": payload.get("task") or sidecar_path.parent.name,
        "seed": payload.get("seed") or parse_seed_from_name(sidecar_path.name),
        "sidecar_path": str(sidecar_path),
        "checkpoint_path": payload.get("checkpoint_path") or payload.get("best_model_path"),
        "adapter": payload.get("adapter") or payload.get("lora") or {},
        **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }


def load_method_family(method_root: str, *, method: str) -> List[dict]:
    """All sidecars under one method directory, flattened."""
    return [extract_record(p, method=method)
            for p in discover_metric_sidecars(method_root)]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _mean_std(xs: Iterable[float]) -> Tuple[float, float, int]:
    vals = np.asarray([float(v) for v in xs if v is not None and not np.isnan(float(v))])
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    if vals.size == 1:
        return float(vals[0]), 0.0, 1
    return float(vals.mean()), float(vals.std(ddof=1)), int(vals.size)


def summarise(records: List[dict], *, metrics: Sequence[str]) -> pd.DataFrame:
    """
    Reduce per-seed records into one row per (method, task) with
    `mean +/- std` and seed count for each requested metric.
    """
    bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {m: [] for m in metrics}
    )
    for r in records:
        key = (r["method"], r["task"])
        for m in metrics:
            if m in r:
                bucket[key][m].append(float(r[m]))

    rows = []
    for (method, task), perm in sorted(bucket.items()):
        row = {"method": method, "task": task}
        for m in metrics:
            mean, std, n = _mean_std(perm[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_lora_per_task(
    lora_records: List[dict],
    *,
    selection_metric: str = "val_acc",
) -> List[dict]:
    """
    Reduce a multi-scheme LoRA sweep to one row per (task, seed) by
    picking the scheme with the highest `selection_metric`. This gives
    the per-task "LoRA (best)" column used throughout the paper.
    """
    best: Dict[Tuple[str, int], dict] = {}
    for rec in lora_records:
        seed = rec.get("seed")
        if seed is None or selection_metric not in rec:
            continue
        key = (rec["task"], int(seed))
        if key not in best or rec[selection_metric] > best[key][selection_metric]:
            best[key] = rec
    return list(best.values())


# ---------------------------------------------------------------------------
# wide-format export
# ---------------------------------------------------------------------------

def build_wide_table(
    summary_df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    task_order: Sequence[str],
    method_order: Sequence[str],
    method_display: Optional[Dict[str, str]] = None,
    digits: int = 4,
) -> pd.DataFrame:
    """
    Pivot the long summary into the canonical paper layout: one row per
    (task, metric), one column per method, cells formatted as
    `"mean +/- std"` at `digits` decimal places.
    """
    method_display = method_display or {m: m for m in method_order}
    fmt = f"{{:.{int(digits)}f}}+/-{{:.{int(digits)}f}}"
    rows = []
    for task in task_order:
        for metric in metrics:
            row = {"Task": task, "Metric": metric}
            for method in method_order:
                sub = summary_df[
                    (summary_df.method == method) & (summary_df.task == task)
                ]
                if sub.empty:
                    row[method_display[method]] = ""
                    continue
                mean = float(sub[f"{metric}_mean"].iloc[0])
                std = float(sub[f"{metric}_std"].iloc[0])
                row[method_display[method]] = fmt.format(mean, std)
            rows.append(row)
    return pd.DataFrame(rows)


def emit_figure_csv(table: pd.DataFrame, out_path: str) -> None:
    """Write the wide-format table preserving the formatted cell strings as-is."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

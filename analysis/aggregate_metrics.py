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
from scipy.stats import wilcoxon


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
    metrics = (
        payload.get("metrics")
        or payload.get("final_test_metric_after_reload")
        or payload.get("test_metrics")
        or {}
    )
    validation = (
        payload.get("final_val_metric_after_reload")
        or payload.get("best_val_metric")
        or {}
    )
    seed = payload.get("seed")
    if seed is None:
        seed = parse_seed_from_name(sidecar_path.name)
    adapter = payload.get("adapter") or payload.get("lora") or {}
    config = payload.get("config_name") or json.dumps(adapter, sort_keys=True)
    return {
        "method": method,
        "task": payload.get("task") or sidecar_path.parent.name,
        "site": payload.get("site", ""),
        "seed": seed,
        "config": config,
        "sidecar_path": str(sidecar_path),
        "checkpoint_path": payload.get("checkpoint_path") or payload.get("best_model_path"),
        "adapter": adapter,
        **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        **{f"val_{k}": float(v) for k, v in validation.items() if isinstance(v, (int, float))},
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
    selection_metric: str = "val_pr_auc",
    selection_site: Optional[str] = None,
) -> List[dict]:
    """
    Select one LoRA configuration per task using the mean validation metric.

    When `selection_site` is supplied, configurations are selected only from
    that site's validation records and then applied to records from every site.
    This supports selecting configurations internally before fixing them for
    external-site adaptation. Without `selection_site`, selection is performed
    independently within each site. Test metrics never enter selection.
    """
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for rec in lora_records:
        if rec.get("seed") is None or selection_metric not in rec:
            continue
        if selection_site is not None and str(rec.get("site", "")) != str(selection_site):
            continue
        grouped[(str(rec.get("site", "")), rec["task"], rec["config"])].append(
            float(rec[selection_metric])
        )
    selected: Dict[Tuple[str, str], str] = {}
    for (site, task, config), values in grouped.items():
        key = (site, task)
        candidate = (float(np.mean(values)), config)
        current = selected.get(key)
        if current is None:
            selected[key] = config
            continue
        current_score = float(np.mean(grouped[(site, task, current)]))
        if candidate[0] > current_score or (candidate[0] == current_score and config < current):
            selected[key] = config
    if selection_site is not None:
        fixed = {
            task: config
            for (site, task), config in selected.items()
            if site == str(selection_site)
        }
        return [rec for rec in lora_records if fixed.get(rec["task"]) == rec.get("config")]
    return [rec for rec in lora_records
            if selected.get((str(rec.get("site", "")), rec["task"])) == rec.get("config")]


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Apply Holm's step-down correction to a sequence of P values."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_wilcoxon_holm(
    records: List[dict],
    *,
    metrics: Sequence[str],
    reference_method: str,
    run_key: str = "seed",
) -> pd.DataFrame:
    """Compare each method with a reference using matched repeated runs.

    Holm correction is applied separately within each metric across all
    task-method comparisons, matching the manuscript's statistical analysis.
    """
    indexed: Dict[Tuple[str, object], Dict[str, dict]] = defaultdict(dict)
    for record in records:
        if record.get(run_key) is None:
            continue
        indexed[(record["task"], record[run_key])][record["method"]] = record

    comparisons = []
    methods = sorted({record["method"] for record in records} - {reference_method})
    tasks = sorted({record["task"] for record in records})
    for metric in metrics:
        for task in tasks:
            for method in methods:
                differences = []
                for (record_task, _), method_records in indexed.items():
                    if record_task != task:
                        continue
                    reference = method_records.get(reference_method)
                    candidate = method_records.get(method)
                    if reference is None or candidate is None:
                        continue
                    if metric not in reference or metric not in candidate:
                        continue
                    differences.append(float(candidate[metric]) - float(reference[metric]))
                if not differences:
                    continue
                array = np.asarray(differences, dtype=float)
                raw_p = 1.0 if np.allclose(array, 0) else float(wilcoxon(array).pvalue)
                comparisons.append({
                    "task": task,
                    "method": method,
                    "reference_method": reference_method,
                    "metric": metric,
                    "mean_difference": float(array.mean()),
                    "n_matched_runs": int(array.size),
                    "p_value": raw_p,
                })

    for metric in metrics:
        indices = [i for i, row in enumerate(comparisons) if row["metric"] == metric]
        adjusted = _holm_adjust([comparisons[i]["p_value"] for i in indices])
        for index, value in zip(indices, adjusted):
            comparisons[index]["p_value_holm"] = value
    return pd.DataFrame(comparisons)


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

"""External-site adaptation gains relative to Freeze-all.

All strategies must be evaluated on the same external test cohort. Gains are
computed within matched repeated runs before aggregation, matching the analysis
reported in Figure 4.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _mean_std(values: Iterable[float]) -> Tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), 0
    if array.size == 1:
        return float(array[0]), 0.0, 1
    return float(array.mean()), float(array.std(ddof=1)), int(array.size)


def external_adaptation_gains(
    records: List[dict],
    *,
    metrics: Sequence[str],
    baseline_method: str = "freeze_all",
    run_key: str = "seed",
    relative: bool = True,
) -> pd.DataFrame:
    """Calculate active-strategy gains over a matched external baseline.

    Records must contain `task`, `method`, the requested metrics, and a stable
    repeated-run identifier. Relative gains are returned as percentages:
    100 * (active - baseline) / baseline.
    """
    indexed: Dict[Tuple[str, object], Dict[str, dict]] = defaultdict(dict)
    for record in records:
        if record.get(run_key) is None:
            continue
        indexed[(record["task"], record[run_key])][record["method"]] = record

    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for (task, _), methods in indexed.items():
        baseline = methods.get(baseline_method)
        if baseline is None:
            continue
        for method, record in methods.items():
            if method == baseline_method:
                continue
            for metric in metrics:
                if metric not in baseline or metric not in record:
                    continue
                reference = float(baseline[metric])
                gain = float(record[metric]) - reference
                if relative:
                    if reference == 0:
                        continue
                    gain = 100.0 * gain / reference
                grouped[(task, method, metric)].append(gain)

    rows = []
    for (task, method, metric), values in sorted(grouped.items()):
        mean, std, count = _mean_std(values)
        rows.append({
            "task": task,
            "method": method,
            "metric": metric,
            "gain_mean": mean,
            "gain_std": std,
            "n_matched_runs": count,
            "relative_percent": bool(relative),
        })
    return pd.DataFrame(rows)

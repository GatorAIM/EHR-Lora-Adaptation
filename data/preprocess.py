"""
Raw EHR ingestion and preprocessing.

The pipeline harmonises per-site PCORnet-style tables into a single
per-encounter token sequence, applies patient-level top-K vocabulary
pruning, attaches binary cohort labels, and splits the latest available
admission year into a held-out test set.

All input / output paths are passed in by the caller as strings; this
module does not read any specific filesystem location on its own.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# table loading
# ---------------------------------------------------------------------------

_REQUIRED_TABLES = ("cohort", "demo", "dx", "px", "labnum", "labcat", "amed",
                    "vital", "cohort_with_onset")


def load_site_tables(site_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Read the canonical site tables. The caller supplies `site_dir`;
    file names follow the convention `<table>T_all.csv` (or `*_all.csv`
    for cohort / cohort-with-onset). Missing tables raise.
    """
    layout = {
        "cohort": "cohort_all.csv",
        "demo": "demoT_all.csv",
        "dx": "dxT_all.csv",
        "labnum": "labnumT_all.csv",
        "labcat": "labcatT_all.csv",
        "amed": "amedT_all.csv",
        "px": "pxT_all.csv",
        "vital": "vitalT_all.csv",
        "cohort_with_onset": "cohort_with_onset.csv",
    }
    out: Dict[str, pd.DataFrame] = {}
    for name, fname in layout.items():
        out[name] = pd.read_csv(f"{site_dir}/{fname}", low_memory=False)
    missing = [t for t in _REQUIRED_TABLES if t not in out]
    if missing:
        raise KeyError(f"site_dir is missing tables: {missing}")
    return out


# ---------------------------------------------------------------------------
# token cleaning
# ---------------------------------------------------------------------------

def harmonise_med_tokens(amed_df: pd.DataFrame, *,
                         ndc_to_rxnorm: pd.DataFrame,
                         rxnorm_to_atc: pd.DataFrame) -> pd.DataFrame:
    """
    Map medication-administration codes through NDC -> RxNorm -> ATC.
    The returned DataFrame carries a new `MED_TOKEN` column of the form
    `MED:ATC:<code>`. Rows that cannot be mapped are dropped.
    """
    df = amed_df.copy()
    df["MED_TOKEN"] = (
        df["MEDADMIN_CODE"].astype(str)
        .map(ndc_to_rxnorm.set_index("NDC")["RXNORM"])
        .map(rxnorm_to_atc.set_index("RXNORM")["ATC"])
        .map(lambda x: f"MED:ATC:{x}" if pd.notna(x) else pd.NA)
    )
    return df.dropna(subset=["MED_TOKEN"]).reset_index(drop=True)


def top_k_filter_patient(
    df: pd.DataFrame,
    *,
    token_col: str,
    patid_col: str,
    encounter_col: str,
    k: int,
) -> Tuple[Set[str], dict]:
    """
    Patient-level top-K vocabulary pruning.

    1. Inside each encounter dedupe the token stream so each token is
       counted once per encounter.
    2. Score each token by the number of distinct PATIDs that ever
       carry it.
    3. Keep the highest-scoring k tokens and drop the rest from `df`
       upstream.

    Returns (kept_tokens, statistics).
    """
    dedup = df[[patid_col, encounter_col, token_col]].drop_duplicates()
    score = dedup.groupby(token_col)[patid_col].nunique().sort_values(ascending=False)
    kept = set(score.head(int(k)).index)
    stats = {
        "n_tokens_total": int(score.size),
        "n_tokens_kept": len(kept),
        "min_patient_count_kept": int(score.head(int(k)).min()) if len(score) else 0,
    }
    return kept, stats


# ---------------------------------------------------------------------------
# token-sequence assembly
# ---------------------------------------------------------------------------

def build_token_sequences(
    tables: Dict[str, pd.DataFrame],
    *,
    max_seq_len: int,
    type_id_of: Dict[str, int],
) -> pd.DataFrame:
    """
    Merge cleaned DX / PX / LAB / AMED / vital streams into one
    chronological sequence per (SUBJECT_ID, HADM_ID). Produces aligned
    `Events`, `Type`, `Time` lists; truncated to `max_seq_len`.
    """
    pieces: List[pd.DataFrame] = []
    for name, type_token in (("dx", "DX"), ("px", "PX"),
                             ("labnum", "LAB_LOINC"), ("labcat", "LAB_LOINC"),
                             ("amed", "MED_TOKEN"), ("vital", "VITAL_TOKEN")):
        sub = tables.get(name)
        if sub is None or type_token not in sub.columns:
            continue
        cur = sub[["SUBJECT_ID", "HADM_ID", type_token, "REL_DAY"]].rename(
            columns={type_token: "Token", "REL_DAY": "Time"}
        )
        cur["Type"] = type_id_of.get(name, 0)
        pieces.append(cur)
    merged = pd.concat(pieces, ignore_index=True)
    merged = merged.sort_values(["SUBJECT_ID", "HADM_ID", "Time"])

    rows = (
        merged.groupby(["SUBJECT_ID", "HADM_ID"], as_index=False)
        .agg({
            "Token": lambda s: list(s)[:max_seq_len],
            "Type": lambda s: list(s)[:max_seq_len],
            "Time": lambda s: list(s)[:max_seq_len],
        })
        .rename(columns={"Token": "Events"})
    )
    return rows


# ---------------------------------------------------------------------------
# train / test split by admission year
# ---------------------------------------------------------------------------

def split_pretrain_test_by_year(
    encounter_df: pd.DataFrame,
    *,
    admit_date_col: str = "ADMIT_DATE",
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Use the latest admission year as the test partition; everything
    earlier becomes the pretrain / finetune pool. The test partition is
    further reduced to the last admission per subject so each test
    patient contributes exactly one row.

    Returns (pretrain_df, test_df, test_year).
    """
    df = encounter_df.copy()
    df["ADMIT_YEAR"] = pd.to_datetime(df[admit_date_col], errors="coerce").dt.year
    test_year = int(df["ADMIT_YEAR"].max())
    is_test = df["ADMIT_YEAR"] == test_year
    pretrain_df = df.loc[~is_test].drop(columns=["ADMIT_YEAR"]).reset_index(drop=True)
    test_df = (
        df.loc[is_test]
        .sort_values(["SUBJECT_ID", admit_date_col])
        .groupby("SUBJECT_ID", as_index=False).tail(1)
        .drop(columns=["ADMIT_YEAR"])
        .reset_index(drop=True)
    )
    return pretrain_df, test_df, test_year


def nested_stratified_subject_subsets(
    df: pd.DataFrame,
    *,
    label_col: str,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
    subject_col: str = "SUBJECT_ID",
    seed: int = 0,
) -> Dict[float, pd.DataFrame]:
    """Create nested subject-level samples stratified by binary outcome status."""
    if subject_col not in df or label_col not in df:
        raise KeyError(f"Expected columns {subject_col!r} and {label_col!r}")
    fractions = tuple(sorted(float(value) for value in fractions))
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("fractions must be nonempty and contained in (0, 1]")

    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).ne(0)
    subject_labels = (
        df.assign(_outcome=labels)
        .groupby(subject_col, sort=True)["_outcome"]
        .max()
    )
    positive = subject_labels[subject_labels].index.to_numpy()
    negative = subject_labels[~subject_labels].index.to_numpy()
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("At least two positive and two negative subjects are required")

    rng = np.random.default_rng(int(seed))
    positive = positive[rng.permutation(len(positive))]
    negative = negative[rng.permutation(len(negative))]
    positive_fraction = len(positive) / len(subject_labels)
    subsets = {}
    for fraction in fractions:
        n_subjects = max(10, int(round(len(subject_labels) * fraction)))
        n_positive = min(len(positive), max(2, int(round(n_subjects * positive_fraction))))
        n_negative = n_subjects - n_positive
        if n_negative < 2 or n_negative > len(negative):
            raise ValueError(
                f"Cannot allocate fraction={fraction}: positive={n_positive}, negative={n_negative}"
            )
        selected = set(positive[:n_positive]) | set(negative[:n_negative])
        subsets[fraction] = df[df[subject_col].isin(selected)].reset_index(drop=True)
    return subsets


def stratified_subject_train_validation_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    validation_fraction: float = 0.1,
    subject_col: str = "SUBJECT_ID",
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split subjects while retaining positive and negative examples in both folds."""
    if subject_col not in df or label_col not in df:
        raise KeyError(f"Expected columns {subject_col!r} and {label_col!r}")
    if not 0 < float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in (0, 1)")

    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).ne(0)
    subject_labels = (
        df.assign(_outcome=labels)
        .groupby(subject_col, sort=False)["_outcome"]
        .max()
    )
    positive = subject_labels[subject_labels].index.to_numpy()
    negative = subject_labels[~subject_labels].index.to_numpy()
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("At least two positive and two negative subjects are required")

    rng = np.random.default_rng(int(seed))
    positive = positive[rng.permutation(len(positive))]
    negative = negative[rng.permutation(len(negative))]
    n_positive = min(
        len(positive) - 1,
        max(1, int(round(len(positive) * float(validation_fraction)))),
    )
    n_negative = min(
        len(negative) - 1,
        max(1, int(round(len(negative) * float(validation_fraction)))),
    )
    validation_subjects = set(positive[:n_positive]) | set(negative[:n_negative])
    validation = df[df[subject_col].isin(validation_subjects)].reset_index(drop=True)
    training = df[~df[subject_col].isin(validation_subjects)].reset_index(drop=True)
    return training, validation


def attach_task_labels(
    df: pd.DataFrame,
    cohort_label_df: pd.DataFrame,
    *,
    label_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Left-join binary task labels keyed by (PATID, ENCOUNTERID); coerce
    them to {0, 1}. Rows that do not match remain in `df` with NaN
    labels so the caller can decide how to handle them.
    """
    cols = ["PATID", "ENCOUNTERID"] + list(label_cols)
    cohort_label_df = cohort_label_df[cols].drop_duplicates(["PATID", "ENCOUNTERID"])
    merged = df.merge(
        cohort_label_df,
        how="left",
        left_on=["SUBJECT_ID", "HADM_ID"],
        right_on=["PATID", "ENCOUNTERID"],
    ).drop(columns=["PATID", "ENCOUNTERID"])
    for c in label_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[c] = (merged[c] > 0).astype("Int64")
    return merged


def merge_sites_into_combined(per_site_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate cleaned per-site DataFrames into a single combined
    "ALL" cohort while retaining `SOURCE_SITE`. Encounter identifiers
    are required to be unique within site, not globally across sites.
    """
    parts = []
    for site, df in per_site_dfs.items():
        cur = df.copy()
        cur["SOURCE_SITE"] = site
        parts.append(cur)
    combined = pd.concat(parts, axis=0, ignore_index=True)
    dup = combined.duplicated(subset=["SOURCE_SITE", "SUBJECT_ID", "HADM_ID"], keep=False)
    if dup.any():
        raise ValueError("Duplicate encounter key detected within a source site.")
    return combined


def truncate_sequences_to_landmark(
    encounter_df: pd.DataFrame,
    *,
    landmark_day: Optional[int] = None,
    landmark_col: Optional[str] = None,
    first_input_day: int = 0,
) -> pd.DataFrame:
    """
    Apply a common prediction landmark without using future outcome status.

    Supply either a fixed `landmark_day` (for example, day 1 for admission-
    anchored tasks) or a prespecified per-row `landmark_col` (for example,
    AKI onset day for progression tasks). Events, Type, and Time remain aligned,
    and only events between `first_input_day` and the landmark are retained.
    """
    if (landmark_day is None) == (landmark_col is None):
        raise ValueError("Supply exactly one of landmark_day or landmark_col")
    output = encounter_df.copy()

    def truncate(row):
        events = list(row["Events"])
        types = list(row["Type"])
        times = list(row["Time"])
        if not (len(events) == len(types) == len(times)):
            raise ValueError("Events, Type, and Time must have aligned lengths")
        end_day = int(landmark_day if landmark_col is None else row[landmark_col])
        keep = [index for index, day in enumerate(times) if first_input_day <= int(day) <= end_day]
        row["Events"] = [events[index] for index in keep]
        row["Type"] = [types[index] for index in keep]
        row["Time"] = [times[index] for index in keep]
        return row

    return output.apply(truncate, axis=1)

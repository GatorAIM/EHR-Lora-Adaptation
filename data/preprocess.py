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
from typing import Dict, Iterable, List, Sequence, Set, Tuple

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
    "ALL" cohort. Refuses to proceed if any (PATID, ENCOUNTERID) key
    collides across sites.
    """
    parts = []
    for site, df in per_site_dfs.items():
        cur = df.copy()
        cur["SOURCE_SITE"] = site
        parts.append(cur)
    combined = pd.concat(parts, axis=0, ignore_index=True)
    dup = combined.duplicated(subset=["SUBJECT_ID", "HADM_ID"], keep=False)
    if dup.any():
        raise ValueError("Cross-site key collision detected; merge is unsafe.")
    return combined.drop(columns=["SOURCE_SITE"])

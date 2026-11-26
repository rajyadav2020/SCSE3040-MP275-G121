"""Cleaning rules, derived from the seven data checks in data_checks.py."""
import re

import numpy as np
import pandas as pd


def normalise_day_type(value: str, true_levels: list[str]) -> str | None:
    """Map 'Weekend', ' weekend', 'WEEKEND', 'fes.' ... to a real level.

    Truncated spellings ('fes.', 'wee.') keep only the first letters. A prefix that fits
    exactly one real level is mapped; an ambiguous one ('wee.' = weekday or weekend) is
    returned as None, because the true value cannot be recovered.
    """
    if not isinstance(value, str):
        return None
    letters = re.sub(r"[^a-z]", "", value.strip().lower())
    if len(letters) < 3:
        return None
    hits = [level for level in true_levels if level.startswith(letters)]
    return hits[0] if len(hits) == 1 else None


def find_stuck_run(df: pd.DataFrame, column: str, date_col: str, min_days: int = 5):
    """Longest run of consecutive dates on which *every* row has one single repeated value."""
    daily = df.groupby(date_col)[column].agg(["nunique", "first", "size"])
    daily = daily.sort_index()
    same = (daily["nunique"] == 1)
    best, cur = None, None
    prev_val = None
    for date, row in daily.iterrows():
        if same[date] and (cur is not None) and row["first"] == prev_val:
            cur[1] = date
        elif same[date]:
            if cur is not None and (best is None or _len(cur) > _len(best)):
                best = cur
            cur = [date, date, row["first"]]
        else:
            if cur is not None and (best is None or _len(cur) > _len(best)):
                best = cur
            cur = None
        prev_val = row["first"] if same[date] else None
    if cur is not None and (best is None or _len(cur) > _len(best)):
        best = cur
    if best is None or _len(best) < min_days:
        return None
    return {"start_date": str(best[0]), "end_date": str(best[1]), "value": float(best[2])}


def _len(run) -> int:
    return (pd.Timestamp(run[1]) - pd.Timestamp(run[0])).days + 1


def clean_train(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return a cleaned copy of train.csv (no leak column, normal spellings, no stuck values)."""
    d = params["data"]
    out = df.copy()
    out = out.drop(columns=[d["leak_column"]], errors="ignore")
    out[d["text_column"]] = out[d["text_column"]].map(
        lambda v: normalise_day_type(v, d["true_day_types"])
    )
    out = out.dropna(subset=[d["text_column"]])
    out = out.drop_duplicates()
    # stuck sensor: the readings in the run are not real -> treat as missing, then drop those rows
    run = find_stuck_run(out, d["stuck_column"], d["date_column"], d["stuck_min_run_days"])
    if run is not None:
        mask = (out[d["date_column"]] >= run["start_date"]) & (out[d["date_column"]] <= run["end_date"])
        out.loc[mask, d["stuck_column"]] = np.nan
        out = out[~mask]
    return out.reset_index(drop=True)


def clean_later(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """later.csv is clean; only the leak column is dropped so it matches the feature set."""
    return df.drop(columns=[params["data"]["leak_column"]], errors="ignore").reset_index(drop=True)
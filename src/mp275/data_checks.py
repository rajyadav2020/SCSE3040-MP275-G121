"""The seven data checks. Every number in findings.yaml comes from this code.

Run:  python -m mp275.data_checks
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import find_stuck_run, normalise_day_type
from .config import load_params, project_path

SENTINELS = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]
UNIT_FACTORS = [10, 12, 60, 100, 1000]


def check_leaking_column(train, holdout, target):
    """A column missing from holdout and (almost) identical to the target."""
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    best = None
    for col in missing:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = float(np.corrcoef(train[col], train[target])[0, 1])
            mae = float((train[col] - train[target]).abs().mean())
            if corr > 0.99 and (best is None or corr > best["corr"]):
                best = {"column": col, "corr": round(corr, 5), "mae_vs_target": round(mae, 3)}
    return {"present": best is not None, "column": best["column"] if best else None, "evidence": best}


def check_unit_mixup(train, group_col, numeric_cols):
    """One group's median is ~10/12/60/100/1000 times the others for some column."""
    for col in numeric_cols:
        med = train.groupby(group_col)[col].median()
        for g, v in med.items():
            others = med.drop(g).median()
            if others == 0 or v == 0:
                continue
            ratio = v / others
            for f in UNIT_FACTORS:
                if abs(ratio - f) / f < 0.25 or abs(1 / ratio - f) / f < 0.25:
                    return {"present": True, "column": col, "group_value": g, "factor": f,
                            "evidence": {"median_ratio": round(float(ratio), 3)}}
    return {"present": False, "column": None, "group_value": None, "factor": None,
            "evidence": {"max_group_median_ratio": float(max(
                (train.groupby(group_col)[c].median().max() / train.groupby(group_col)[c].median().min())
                for c in numeric_cols))}}


def check_sentinel(train, numeric_cols):
    """A special impossible number repeated many times (-1, -999, 999, 9999 ...)."""
    for col in numeric_cols:
        counts = train[col].value_counts()
        for s in SENTINELS:
            if s in counts.index and counts[s] >= 3:
                others = train.loc[train[col] != s, col]
                if s < others.min() or s > others.max():
                    return {"present": True, "column": col, "value": s, "count": int(counts[s])}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicates(train):
    """Extra copies: rows that repeat another row (also with the id ignored)."""
    full = int(train.duplicated().sum())
    no_id = int(train.drop(columns=["record_id"]).duplicated().sum())
    count = max(full, no_id)
    return {"present": count > 0, "count": count if count > 0 else None,
            "evidence": {"full_row": full, "ignoring_record_id": no_id}}


def check_stuck_sensor(train, numeric_cols, date_col, min_days):
    for col in numeric_cols:
        run = find_stuck_run(train, col, date_col, min_days)
        if run:
            return {"present": True, "column": col, "start_date": run["start_date"],
                    "end_date": run["end_date"], "evidence": {"stuck_value": run["value"]}}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train, target):
    """A small group of targets sits far above the rest, with an empty gap below them."""
    s = np.sort(train[target].values)
    top = s[int(len(s) * 0.9):]
    ratios = top[1:] / np.maximum(top[:-1], 1)
    i = int(np.argmax(ratios)) if len(ratios) else 0
    big_gap = len(ratios) > 0 and ratios[i] > 3
    if big_gap:
        count = int((s >= top[i + 1]).sum())
        return {"present": True, "count": count, "evidence": {"gap_ratio": round(float(ratios[i]), 2)}}
    return {"present": False, "count": None,
            "evidence": {"max_target": float(s[-1]), "median_target": float(np.median(s)),
                         "largest_gap_ratio_in_top_10pct": round(float(ratios.max()), 3) if len(ratios) else None}}


def check_spelling_variants(train, text_cols, true_levels):
    for col in text_cols:
        raw = train[col].nunique()
        fixed = train[col].map(lambda v: normalise_day_type(v, true_levels))
        if raw > len(true_levels) and fixed.notna().sum() > 0 and col == "day_type":
            return {"present": True, "column": col, "true_levels": int(fixed.nunique()),
                    "evidence": {"raw_levels": int(raw), "ambiguous_rows": int(fixed.isna().sum()),
                                 "spellings": sorted(train[col].unique().tolist())}}
    return {"present": False, "column": None, "true_levels": None}


def run_all(params: dict | None = None) -> dict:
    params = params or load_params()
    d = params["data"]
    train = pd.read_csv(project_path(params["paths"]["train"]))
    holdout = pd.read_csv(project_path(params["paths"]["holdout"]))
    numeric = d["numeric_features"]
    report = {
        "leaking_column": check_leaking_column(train, holdout, d["target"]),
        "unit_mixup": check_unit_mixup(train, d["group_column"], numeric),
        "sentinel_value": check_sentinel(train, numeric),
        "duplicate_rows": check_duplicates(train),
        "stuck_sensor": check_stuck_sensor(train, numeric, d["date_column"], d["stuck_min_run_days"]),
        "target_typos": check_target_typos(train, d["target"]),
        "spelling_variants": check_spelling_variants(train, [d["text_column"], d["group_column"]], d["true_day_types"]),
    }
    return report


def main():
    params = load_params()
    report = run_all(params)
    out = project_path(params["paths"]["data_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
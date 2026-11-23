"""Model pipelines and small helpers shared by training, retraining and serving."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def sha256_of(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def feature_columns(params: dict) -> list[str]:
    d = params["data"]
    return list(d["numeric_features"]) + list(d["categorical_features"])


def build_pipeline(cfg: dict, params: dict, random_state: int = 42) -> Pipeline:
    d = params["data"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), d["numeric_features"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), d["categorical_features"]),
    ])
    kind = cfg["model"]
    if kind == "ridge":
        est = Ridge(alpha=cfg.get("alpha", 1.0))
    elif kind == "random_forest":
        est = RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 200), max_depth=cfg.get("max_depth"),
            min_samples_leaf=cfg.get("min_samples_leaf", 1), random_state=random_state, n_jobs=-1)
    elif kind == "gradient_boosting":
        est = GradientBoostingRegressor(
            n_estimators=cfg.get("n_estimators", 300), learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 3), random_state=random_state)
    else:
        raise ValueError(f"unknown model {kind}")
    return Pipeline([("prep", pre), ("model", est)])


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def split_xy(df: pd.DataFrame, params: dict):
    return df[feature_columns(params)], df[params["data"]["target"]]


def sort_by_date(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df.sort_values([params["data"]["date_column"], "record_id"]).reset_index(drop=True)


def fit_weighted(pipe: Pipeline, X, y, weights) -> Pipeline:
    """Fit a pipeline with per-row sample weights (the final step is named 'model')."""
    return pipe.fit(X, y, model__sample_weight=np.asarray(weights, dtype=float))


def choose_recent_weight(cfg, params, old: pd.DataFrame, recent: pd.DataFrame, grid, random_state=42):
    """Pick how much to up-weight recent rows, using only data that v2 is allowed to see.

    `recent` is the first 70% of later.csv (sorted by date). We fit on `old` plus the first 80%
    of `recent`, validate on the last 20% of `recent`, and return the weight with the lowest MAE.
    The final 30% of later.csv (the promotion check set) is never touched here.
    """
    k = int(len(recent) * 0.8)
    fit_recent, val_recent = recent.iloc[:k], recent.iloc[k:]
    Xo, yo = split_xy(old, params)
    Xf, yf = split_xy(fit_recent, params)
    Xv, yv = split_xy(val_recent, params)
    X = pd.concat([Xo, Xf], ignore_index=True)
    y = pd.concat([yo, yf], ignore_index=True)
    scores = {}
    for w in grid:
        sw = np.r_[np.ones(len(Xo)), np.full(len(Xf), float(w))]
        pipe = fit_weighted(build_pipeline(cfg, params, random_state), X, y, sw)
        scores[w] = mae(yv, pipe.predict(Xv))
    return min(scores, key=scores.get), scores
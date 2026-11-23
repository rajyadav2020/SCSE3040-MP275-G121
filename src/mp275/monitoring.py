"""Drift (PSI) and per-group error."""
import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with `bins` equal-frequency bins taken from `expected`."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, columns: list[str], bins: int = 10) -> dict:
    return {c: psi(train[c], later[c], bins) for c in columns}


def mae_by_group(df: pd.DataFrame, y_pred, target: str, group: str) -> dict:
    err = (df[target].values - np.asarray(y_pred)).__abs__()
    return pd.Series(err, index=df.index).groupby(df[group]).mean().to_dict()
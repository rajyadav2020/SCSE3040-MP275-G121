"""Load settings from params.yaml (the single place for paths and model settings)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str) -> Path:
    return ROOT / relative
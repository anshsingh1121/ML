"""Utility functions and helpers."""

import pandas as pd
import json
from pathlib import Path

def _resolve_path(filepath: str) -> str:
    p = Path(filepath)
    if not p.is_absolute():
        # Resolve relative to project root (2 levels up from src/utils)
        project_root = Path(__file__).resolve().parent.parent.parent
        p = project_root / p
    return str(p)

def robust_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    filepath = _resolve_path(filepath)
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # If all fail, let it raise normally
    return pd.read_csv(filepath, encoding="utf-8", **kwargs)

def robust_open(filepath: str, mode: str = "r", **kwargs):
    filepath = _resolve_path(filepath)
    if "w" in mode or "a" in mode:
        if "encoding" not in kwargs and "b" not in mode:
            kwargs["encoding"] = "utf-8"
        return open(filepath, mode, **kwargs)
    
    if "b" in mode:
        return open(filepath, mode, **kwargs)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            f = open(filepath, mode, encoding=enc, **kwargs)
            f.read()
            f.seek(0)
            return f
        except UnicodeDecodeError:
            f.close()
            continue
    return open(filepath, mode, encoding="utf-8", **kwargs)

def robust_json_load(filepath: str):
    filepath = _resolve_path(filepath)
    with robust_open(filepath, "r") as f:
        return json.load(f)

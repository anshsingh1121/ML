"""Utility functions and helpers."""

import pandas as pd
import json

def robust_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # If all fail, let it raise normally
    return pd.read_csv(filepath, encoding="utf-8", **kwargs)

def robust_open(filepath: str, mode: str = "r", **kwargs):
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
    with robust_open(filepath, "r") as f:
        return json.load(f)

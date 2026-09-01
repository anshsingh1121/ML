import pandas as pd
import streamlit as st
import os
from pathlib import Path

@st.cache_data
def load_csv_data(filepath, max_rows=None):
    """Safely loads a CSV file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=max_rows)
        return df
    except Exception:
        return None

@st.cache_data
def load_processed_incidents():
    """Loads the master engineered incidents dataset."""
    return load_csv_data("data/processed/master_engineered_incidents.csv")

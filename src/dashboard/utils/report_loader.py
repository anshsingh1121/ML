import json
import streamlit as st
from pathlib import Path
from PIL import Image

@st.cache_data
def load_json_report(filepath):
    """Safely loads a JSON report."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_image(filepath):
    """Safely loads an image."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        return Image.open(path)
    except Exception:
        return None

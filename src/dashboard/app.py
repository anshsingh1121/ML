import streamlit as st
import sys
import os
from pathlib import Path

# Ensure the root path is accessible so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.hybrid.recommendation_engine import HybridRecommendationEngine
from src.dashboard.components.layout import apply_enterprise_theme
from src.dashboard.components import (
    executive_overview,
    incident_intelligence,
    semantic_evidence,
    explainability,
    model_performance,
    historical_analytics,
    system_health
)

st.set_page_config(
    page_title="AI Incident Intelligence Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global enterprise CSS
apply_enterprise_theme()

@st.cache_resource
def get_engine():
    """Cache the engine initialization to prevent reloading on UI interactions."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        return HybridRecommendationEngine()
    except Exception as e:
        st.error(f"Failed to load Hybrid Engine: {e}")
        return None

engine = get_engine()

# Sidebar Navigation
st.sidebar.title("🧠 AI Command Center")
st.sidebar.markdown("Enterprise Incident Intelligence")

pages = {
    "Executive Overview": executive_overview.render,
    "Incident Intelligence": lambda: incident_intelligence.render(engine),
    "Semantic Evidence": semantic_evidence.render,
    "Explainability": explainability.render,
    "Model Performance": model_performance.render,
    "Historical Analytics": historical_analytics.render,
    "System Health": system_health.render
}

selection = st.sidebar.radio("Navigation", list(pages.keys()))

st.sidebar.markdown("---")
if st.sidebar.button("Reset Session State"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
    
# Render selected page
try:
    pages[selection]()
except Exception as e:
    st.error(f"An error occurred while rendering the page: {e}")
    st.exception(e)

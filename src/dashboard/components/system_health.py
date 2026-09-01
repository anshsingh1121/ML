import streamlit as st
import os
from pathlib import Path
from src.dashboard.components.layout import render_section_header

def check_file_exists(filepath):
    return Path(filepath).exists()

def render():
    render_section_header("System & Platform Health", "Real-time diagnostic checks of AI artifacts")
    
    health_checks = [
        {"Category": "Models", "Name": "CatBoost Classifier (Assignment Group)", "Path": "models/catboost_assignment_group.pkl"},
        {"Category": "Models", "Name": "CatBoost Regressor (MTTR)", "Path": "models/catboost_resolution_time_hours.pkl"},
        {"Category": "Models", "Name": "Feature Preprocessing Pipeline", "Path": "models/preprocessing_pipeline.pkl"},
        {"Category": "Semantic Engine", "Name": "FAISS Index", "Path": "indexes/historical_incidents.index"},
        {"Category": "Semantic Engine", "Name": "Vectorized Context (Pickle)", "Path": "indexes/vectorized_context.pkl"},
        {"Category": "Data", "Name": "Processed Master Dataset", "Path": "data/processed/master_engineered_incidents.csv"},
        {"Category": "Reports", "Name": "Classification Evaluation", "Path": "reports/classification_report.json"},
        {"Category": "Reports", "Name": "EDA Report", "Path": "reports/eda_report.json"},
    ]
    
    st.markdown("### Artifact Readiness")
    
    for check in health_checks:
        exists = check_file_exists(check["Path"])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.markdown(f"**{check['Category']}**")
        with col2:
            st.markdown(check['Name'])
        with col3:
            if exists:
                st.markdown("✅ `<span style='color:green;font-weight:bold'>READY</span>`", unsafe_allow_html=True)
            else:
                st.markdown("❌ `<span style='color:red;font-weight:bold'>MISSING</span>`", unsafe_allow_html=True)
                
        st.markdown("---")
        
    st.info("If critical models or indexes are missing, please run `python main.py full-pipeline` in the repository root to regenerate all artifacts.")

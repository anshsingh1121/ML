import streamlit as st
import pandas as pd
import sys
import os

# Ensure the root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.hybrid.recommendation_engine import HybridRecommendationEngine
from src.utils.config_manager import ConfigManager

st.set_page_config(
    page_title="AI Incident Intelligence",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_engine():
    """Cache the engine initialization so it doesn't reload models on every UI interaction."""
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    return HybridRecommendationEngine()

st.title("🧠 Enterprise AI Incident Intelligence")
st.markdown("Predict Assignment Groups and Resolution Times instantly using CatBoost & Semantic Neural Search.")

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to load AI models. Ensure you have run 'python main.py full-pipeline' first. Error: {e}")
    st.stop()

# Build the layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Core Incident Details")
    short_desc = st.text_input("Short Description (Title)", value="Database connection timeout on payment gateway")
    desc = st.text_area("Full Description", value="Users are reporting timeout errors when trying to process payments. Stack trace indicates connection pool exhausted on db-node-04.", height=150)
    
    cat = st.selectbox("Category", ["Software", "Hardware", "Network", "Database", "Security", "Inquiry / Help", "UNKNOWN"])
    subcat = st.text_input("Subcategory", value="Oracle DB")
    ci = st.text_input("Configuration Item (CMDB CI)", value="db-node-04")
    
with col2:
    st.header("2. Severity & Corporate Schema")
    prio = st.selectbox("Priority", [1, 2, 3, 4, 5], index=2)
    impact = st.selectbox("Business Impact", [1, 2, 3], index=1)
    sev = st.selectbox("Severity", [1, 2, 3], index=1)
    
    st.markdown("### Corporate Custom Fields")
    u_caused_by = st.text_input("Caused By (u_caused_by)", value="")
    u_dev_id = st.text_input("Dev Release ID (u_development_release_id)", value="")
    u_vendor = st.text_input("Vendor Ticket Ref (u_vendor_ticket_ref)", value="")
    u_impact = st.text_area("Customer Impact (u_describe_customer_impact)", value="", height=68)

if st.button("Predict Routing & MTTR 🚀", use_container_width=True, type="primary"):
    with st.spinner("Analyzing incident via Hybrid AI Engine..."):
        # Construct the payload
        ticket_dict = {
            "number": "INC_DEMO_001",
            "short_description": short_desc,
            "description": desc,
            "category": cat,
            "subcategory": subcat,
            "cmdb_ci": ci,
            "priority": prio,
            "business_impact": impact,
            "severity": sev,
            "u_caused_by": u_caused_by,
            "u_development_release_id": u_dev_id,
            "u_vendor_ticket_ref": u_vendor,
            "u_describe_customer_impact": u_impact,
            "caused_by": u_caused_by, # Mapping fallback
            "incident_state": "New"
        }
        
        try:
            # Execute recommendation
            result = engine.recommend(ticket_dict, top_k=5, export_reports=False)
            
            st.success("Analysis Complete!")
            
            # Display Top-Level Results
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Assignment Group", result["recommended_assignment_group"])
            with res_col2:
                st.metric("Confidence Score", f"{result['confidence_score']:.1%}", result["confidence_tier"])
            with res_col3:
                st.metric("Estimated MTTR", f"{result['estimated_resolution_time_hours']:.1f} Hours")
                
            # Display Reasoning
            st.markdown("---")
            st.subheader("💡 AI Reasoning Justification")
            st.info(result["reasoning"])
            
            # Display Historical Evidence
            st.markdown("---")
            st.subheader("📚 Top 5 Historical Precedents (Semantic Search)")
            evidence_df = pd.DataFrame(result["historical_evidence"])
            if not evidence_df.empty:
                st.dataframe(
                    evidence_df[["rank", "number", "similarity_score", "historical_assignment_group", "historical_resolution_time"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No historical precedents found in the FAISS index.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

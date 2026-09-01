import streamlit as st
import pandas as pd
from src.dashboard.components.layout import render_section_header

def render(engine):
    render_section_header("Incident Intelligence & Prediction", "Submit an incident for hybrid AI analysis")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Core Incident Details")
        short_desc = st.text_input("Short Description (Title)", value="Database connection timeout on payment gateway")
        desc = st.text_area("Full Description", value="Users are reporting timeout errors when trying to process payments. Stack trace indicates connection pool exhausted on db-node-04.", height=150)
        
        c1, c2 = st.columns(2)
        with c1:
            cat = st.selectbox("Category", ["Software", "Hardware", "Network", "Database", "Security", "Inquiry / Help", "UNKNOWN"])
        with c2:
            subcat = st.text_input("Subcategory", value="Oracle DB")
            
        ci = st.text_input("Configuration Item (CMDB CI)", value="db-node-04")
        
    with col2:
        st.subheader("2. Severity & Corporate Context")
        c3, c4, c5 = st.columns(3)
        with c3:
            prio = st.selectbox("Priority", [1, 2, 3, 4, 5], index=2)
        with c4:
            impact = st.selectbox("Business Impact", [1, 2, 3], index=1)
        with c5:
            sev = st.selectbox("Severity", [1, 2, 3], index=1)
        
        with st.expander("Corporate Custom Fields (Advanced)", expanded=False):
            u_caused_by = st.text_input("Caused By (u_caused_by)", value="")
            u_dev_id = st.text_input("Dev Release ID", value="")
            u_vendor = st.text_input("Vendor Ticket Ref", value="")
            u_impact = st.text_area("Customer Impact", value="", height=68)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Analyze Incident 🚀", use_container_width=True, type="primary"):
        with st.spinner("Executing Hybrid AI Pipeline (RF Classification + FAISS Semantic Search)..."):
            ticket_dict = {
                "number": "INC_NEW_001",
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
                "caused_by": u_caused_by,
                "incident_state": "New"
            }
            
            try:
                result = engine.recommend(ticket_dict, top_k=5, export_reports=False)
                st.session_state.prediction_result = result
                st.session_state.ticket_dict = ticket_dict
                st.success("Analysis Complete! View results below and explore Evidence & Explainability pages.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                
    # If prediction exists, show results
    if "prediction_result" in st.session_state:
        st.markdown("---")
        res = st.session_state.prediction_result
        
        st.subheader("Prediction Command Center")
        
        # Top level metrics
        c_res1, c_res2, c_res3 = st.columns(3)
        with c_res1:
            st.metric("Predicted Assignment Group", res.get("recommended_assignment_group", "UNKNOWN"))
        with c_res2:
            conf = res.get("confidence_score", 0.0)
            tier = res.get("confidence_tier", "UNKNOWN")
            st.metric("Confidence Score", f"{conf:.1%}", tier, delta_color="normal" if conf > 0.7 else "off")
        with c_res3:
            mttr = res.get("estimated_resolution_time_hours", 0.0)
            st.metric("Estimated MTTR", f"{mttr:.1f} Hours")
            
        st.markdown("### 💡 AI Reasoning Justification")
        
        status_color = "success" if tier == "HIGH" else ("warning" if tier == "MEDIUM" else "error")
        st.info(res.get("reasoning", "No reasoning available."))
        
        # Provide actionable routing status
        if tier == "HIGH":
            st.success("RECOMMENDATION: AUTO-ROUTE ELIGIBLE")
        elif tier == "MEDIUM":
            st.warning("RECOMMENDATION: ASSISTED ROUTING (Review Suggested)")
        else:
            st.error("RECOMMENDATION: MANUAL REVIEW REQUIRED")

        st.markdown("*To dive deeper into how this prediction was made, visit the **Semantic Evidence** and **Explainability** pages.*")

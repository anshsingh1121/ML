import streamlit as st
import pandas as pd
from src.dashboard.components.layout import render_section_header
from src.dashboard.utils.report_loader import load_json_report, load_image

def render():
    render_section_header("Executive Overview", "High-level platform metrics and operational impact")
    
    # Load data
    class_report = load_json_report("reports/classification_report.json")
    eda_report = load_json_report("reports/eda_report.json")
    business_impact_img = load_image("visual_business_impact.png")
    
    # Build Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if eda_report:
            total_incidents = eda_report.get("basic_statistics", {}).get("total_records", "N/A")
            st.metric("Total Analyzed", total_incidents, help="Total historical incidents processed.")
        else:
            st.metric("Total Analyzed", "N/A", help="EDA report not found.")
            
    with col2:
        if class_report:
            acc = class_report.get("accuracy", 0)
            if acc > 0:
                st.metric("Routing Accuracy", f"{acc:.1%}", help="Overall assignment group classification accuracy.")
            else:
                st.metric("Routing Accuracy", "N/A")
        else:
            st.metric("Routing Accuracy", "N/A")
            
    with col3:
        if class_report:
            macro_avg = class_report.get("macro avg", {})
            f1 = macro_avg.get("f1-score", 0)
            if f1 > 0:
                st.metric("F1 Score (Macro)", f"{f1:.3f}")
            else:
                st.metric("F1 Score", "N/A")
        else:
            st.metric("F1 Score", "N/A")
            
    with col4:
        if eda_report:
            avg_mttr = eda_report.get("numerical_features", {}).get("resolution_time_hours", {}).get("mean", 0)
            if avg_mttr > 0:
                st.metric("Avg Historical MTTR", f"{avg_mttr:.1f} hrs", help="Mean resolution time from historical data.")
            else:
                st.metric("Avg Historical MTTR", "N/A")
        else:
            st.metric("Avg Historical MTTR", "N/A")
            
    st.markdown("---")
    
    # Display Business Impact Image
    st.subheader("Platform Business Impact")
    if business_impact_img:
        st.image(business_impact_img, use_container_width=True, caption="Manual vs AI Triage Comparison")
    else:
        st.info("Business impact visualization unavailable (`visual_business_impact.png`).")
        
    # Add a distribution summary if EDA report is available
    if eda_report:
        st.markdown("---")
        st.subheader("Data Profile Summary")
        cat_dist = eda_report.get("categorical_features", {})
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Assignment Groups**")
            groups = cat_dist.get("assignment_group", {}).get("top_values", {})
            if groups:
                df_groups = pd.DataFrame(list(groups.items()), columns=["Group", "Count"])
                st.dataframe(df_groups, hide_index=True, use_container_width=True)
            else:
                st.write("No assignment group data.")
                
        with c2:
            st.markdown("**Top Categories**")
            cats = cat_dist.get("category", {}).get("top_values", {})
            if cats:
                df_cats = pd.DataFrame(list(cats.items()), columns=["Category", "Count"])
                st.dataframe(df_cats, hide_index=True, use_container_width=True)
            else:
                st.write("No category data.")

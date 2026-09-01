import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.components.layout import render_section_header
from src.dashboard.utils.report_loader import load_image, load_json_report

def render():
    render_section_header("Explainability Engine", "SHAP Feature Attribution & Decision Transparency")
    
    tab1, tab2 = st.tabs(["Global Model Explanations", "Local Prediction Explanations"])
    
    with tab1:
        st.markdown("### What drives the AI across all incidents?")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**Global Feature Importance**")
            bar_img = load_image("reports/shap_bar.png")
            if bar_img:
                st.image(bar_img, use_container_width=True)
            else:
                fallback_bar = load_image("visual_feature_importance.png")
                if fallback_bar:
                    st.image(fallback_bar, use_container_width=True)
                else:
                    st.info("SHAP Bar chart unavailable.")
                    
        with col_img2:
            st.markdown("**SHAP Beeswarm Summary**")
            summary_img = load_image("reports/shap_summary.png")
            if summary_img:
                st.image(summary_img, use_container_width=True)
            else:
                st.info("SHAP Summary chart unavailable.")
                
    with tab2:
        st.markdown("### Current Incident Feature Attribution")
        
        if "prediction_result" not in st.session_state:
            st.warning("No active prediction found. Please submit an incident to view local explanations.")
        else:
            res = st.session_state.prediction_result
            
            # Hybrid Recommendation Engine doesn't output SHAP directly into the recommendation payload 
            # by default (it's in the background or requires SHAP explainer call). 
            # If the user has run full pipeline, we might have prediction metadata.
            
            meta = load_json_report("reports/prediction_metadata.json")
            if meta and "predictions" in meta and len(meta["predictions"]) > 0:
                # We show the latest prediction metadata from the batch as a demonstration
                # Since real-time SHAP per-inference requires `shap_explainer.py` to be invoked online.
                # We will just show the structure of the last evaluated incident in the background.
                st.info("Showing local SHAP explanation from the most recent batch inference.")
                latest = meta["predictions"][-1]
                
                features = latest.get("top_contributing_features", [])
                if features:
                    df_local = pd.DataFrame(features)
                    
                    # Interactive waterfall-like bar chart
                    fig = px.bar(
                        df_local, 
                        y="feature", 
                        x="shap_contribution", 
                        orientation="h",
                        title=f"Top Drivers for Prediction: {latest.get('predicted_class', latest.get('predicted_value'))}",
                        color="shap_contribution",
                        color_continuous_scale=px.colors.diverging.Tealrose,
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No feature contribution data found in the metadata.")
            else:
                st.info("Real-time local SHAP extraction requires invoking the Explainability Engine on this specific payload. (Currently running in batch mode).")
                
                # Show sample images if batch ran
                st.markdown("#### Sample Local Explainability (Batch)")
                c1, c2 = st.columns(2)
                with c1:
                    w_img = load_image("reports/shap_waterfall_sample.png")
                    if w_img:
                        st.image(w_img, use_container_width=True)
                with c2:
                    d_img = load_image("reports/shap_decision_sample.png")
                    if d_img:
                        st.image(d_img, use_container_width=True)

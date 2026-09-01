import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.components.layout import render_section_header
from src.dashboard.utils.data_loader import load_processed_incidents

def render():
    render_section_header("Historical Analytics", "Interactive exploration of incident data")
    
    df = load_processed_incidents()
    
    if df is None or df.empty:
        st.warning("Historical dataset not found or empty (expected at `data/processed/master_engineered_incidents.csv`).")
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    # Safely get unique values and handle potential missing columns
    def get_unique_safe(df, col):
        if col in df.columns:
            return sorted([str(x) for x in df[col].dropna().unique()])
        return []
        
    priorities = get_unique_safe(df, "priority")
    if priorities:
        selected_priorities = st.sidebar.multiselect("Priority", options=priorities, default=priorities)
        if selected_priorities:
            df = df[df["priority"].astype(str).isin(selected_priorities)]
            
    categories = get_unique_safe(df, "category")
    if categories:
        selected_categories = st.sidebar.multiselect("Category", options=categories, default=[])
        if selected_categories:
            df = df[df["category"].astype(str).isin(selected_categories)]
            
    st.markdown(f"**Showing {len(df):,} incidents based on filters.**")
    
    tab1, tab2 = st.tabs(["Volume & Distribution", "MTTR Analytics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if "category" in df.columns:
                cat_counts = df["category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig1 = px.bar(cat_counts.head(10), x="Count", y="Category", orientation="h", 
                              title="Top 10 Categories", color="Count", color_continuous_scale="Blues")
                fig1.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig1, use_container_width=True)
                
        with col2:
            if "assignment_group" in df.columns:
                ag_counts = df["assignment_group"].value_counts().reset_index()
                ag_counts.columns = ["Assignment Group", "Count"]
                fig2 = px.bar(ag_counts.head(10), x="Count", y="Assignment Group", orientation="h", 
                              title="Top 10 Assignment Groups", color="Count", color_continuous_scale="Teal")
                fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig2, use_container_width=True)
                
        if "opened_at_hour" in df.columns:
            hour_counts = df["opened_at_hour"].value_counts().reset_index().sort_values("opened_at_hour")
            hour_counts.columns = ["Hour of Day", "Incident Count"]
            fig3 = px.line(hour_counts, x="Hour of Day", y="Incident Count", markers=True, 
                           title="Incident Volume by Hour of Day")
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        if "resolution_time_hours" in df.columns:
            st.markdown("### Resolution Time Analysis")
            
            c1, c2 = st.columns(2)
            
            with c1:
                if "priority" in df.columns:
                    fig4 = px.box(df, x="priority", y="resolution_time_hours", 
                                  title="MTTR Distribution by Priority",
                                  points="outliers")
                    fig4.update_yaxes(type="log", title="Resolution Time (Hours) [Log Scale]")
                    st.plotly_chart(fig4, use_container_width=True)
                    
            with c2:
                if "category" in df.columns:
                    cat_mttr = df.groupby("category")["resolution_time_hours"].median().reset_index().sort_values("resolution_time_hours", ascending=False).head(10)
                    fig5 = px.bar(cat_mttr, x="resolution_time_hours", y="category", orientation="h",
                                  title="Median MTTR by Top 10 Categories (Hours)")
                    fig5.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Resolution time data not available for MTTR analytics.")

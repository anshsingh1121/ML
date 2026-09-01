import streamlit as st
import pandas as pd
from src.dashboard.components.layout import render_section_header
from src.dashboard.utils.report_loader import load_image
import plotly.express as px
import plotly.graph_objects as go

def render():
    render_section_header("Semantic Evidence", "Historical consensus driven by NLP Vectorization")
    
    if "prediction_result" not in st.session_state:
        st.warning("No active prediction found. Please submit an incident on the 'Incident Intelligence' page first.")
        return
        
    res = st.session_state.prediction_result
    evidence = res.get("historical_evidence", [])
    
    if not evidence:
        st.info("No historical precedents were retrieved for this incident.")
        return
        
    st.subheader("Top Historical Precedents")
    df_evidence = pd.DataFrame(evidence)
    
    # Create visual similarity bars
    fig = go.Figure()
    
    # Sort for chart (top to bottom)
    df_chart = df_evidence.sort_values("rank", ascending=False)
    
    fig.add_trace(go.Bar(
        x=df_chart["similarity_score"],
        y=df_chart["number"],
        orientation='h',
        text=[f"{s:.1%}" for s in df_chart["similarity_score"]],
        textposition='auto',
        marker=dict(
            color=df_chart["similarity_score"],
            colorscale="Viridis",
            showscale=False
        )
    ))
    
    fig.update_layout(
        title="Semantic Similarity to Current Incident",
        xaxis_title="Similarity Score",
        yaxis_title="Historical Incident Number",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Table
    st.markdown("### Precedent Details")
    styled_df = df_evidence[["rank", "number", "similarity_score", "historical_assignment_group", "historical_resolution_time"]].copy()
    styled_df["similarity_score"] = styled_df["similarity_score"].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": "Rank",
            "number": "Incident ID",
            "similarity_score": "Similarity",
            "historical_assignment_group": "Historical Group",
            "historical_resolution_time": "Historical MTTR (hrs)"
        }
    )
    
    st.markdown("---")
    st.subheader("How does Semantic Search work?")
    
    nlp_img = load_image("visual_nlp_vectorization.png")
    if nlp_img:
        st.image(nlp_img, use_container_width=True)
    else:
        st.info("NLP architecture visualization unavailable.")

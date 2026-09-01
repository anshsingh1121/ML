import streamlit as st

def apply_enterprise_theme():
    """Injects custom CSS for an enterprise SaaS look."""
    st.markdown("""
        <style>
        /* Global Styling */
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 1.25rem 1rem 1rem 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        
        @media (prefers-color-scheme: dark) {
            div[data-testid="metric-container"] {
                background-color: #1e1e1e;
                border-color: #333;
                box-shadow: none;
            }
        }
        
        div[data-testid="metric-container"]:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            padding-top: 1rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        h1 {
            color: #1E3A8A; /* Dark blue */
        }
        @media (prefers-color-scheme: dark) {
            h1 {
                color: #60A5FA;
            }
        }
        
        /* Section dividers */
        hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
            opacity: 0.3;
        }
        
        /* Button primary styling */
        .stButton>button[data-baseweb="button"] {
            border-radius: 6px;
            font-weight: 600;
        }
        
        /* Alert/Success boxes */
        .stAlert {
            border-radius: 6px;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

def render_section_header(title, subtitle=None):
    """Renders a consistent section header."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
        
def render_metric_card(label, value, delta=None, help_text=None):
    """Helper to render a metric."""
    st.metric(label=label, value=value, delta=delta, help=help_text)

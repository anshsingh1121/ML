import streamlit as st
import pandas as pd
from src.dashboard.components.layout import render_section_header
from src.dashboard.utils.report_loader import load_json_report, load_image

def render():
    render_section_header("Model Performance", "Classifier and Regressor Analytics")
    
    class_report = load_json_report("reports/classification_report.json")
    
    if not class_report:
        st.warning("Classification report not found. Ensure 'python main.py evaluate' was run.")
        return
        
    st.subheader("Assignment Group Classifier")
    
    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    acc = class_report.get("accuracy", 0)
    macro = class_report.get("macro avg", {})
    weighted = class_report.get("weighted avg", {})
    
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Macro F1", f"{macro.get('f1-score', 0):.3f}")
    c3.metric("Weighted F1", f"{weighted.get('f1-score', 0):.3f}")
    c4.metric("Support", int(macro.get("support", 0)))
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Per-Class Metrics", "Confusion Matrix", "ROC Curve"])
    
    with tab1:
        st.markdown("### Class-Level Performance")
        # Extract class metrics, ignore "accuracy", "macro avg", "weighted avg"
        classes_data = []
        for k, v in class_report.items():
            if isinstance(v, dict) and k not in ["macro avg", "weighted avg"]:
                row = v.copy()
                row["Class"] = k
                classes_data.append(row)
                
        if classes_data:
            df_classes = pd.DataFrame(classes_data)
            df_classes = df_classes[["Class", "precision", "recall", "f1-score", "support"]]
            
            # Format
            st.dataframe(
                df_classes.style.format({
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:.0f}"
                }).background_gradient(subset=["f1-score"], cmap="Blues"),
                use_container_width=True,
                height=400
            )
            
    with tab2:
        cm_img = load_image("reports/confusion_matrix.png")
        if cm_img:
            st.image(cm_img, use_container_width=True)
        else:
            st.info("Confusion Matrix image unavailable.")
            
    with tab3:
        roc_img = load_image("reports/roc_curve.png")
        if roc_img:
            st.image(roc_img, use_container_width=True)
        else:
            st.info("ROC Curve image unavailable.")

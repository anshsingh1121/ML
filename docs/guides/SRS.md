# Software Requirements Specification (SRS)
**Project:** AI-Powered Incident Intelligence Platform (IIP)  
**Organization:** First Citizens Bank  
**Version:** 1.0.0 (Phase 1 Approved)  
**Date:** 2026-07-10  

---

## 1. Introduction & Purpose
This document defines the functional, non-functional, data, and architectural requirements for the AI-Powered Incident Intelligence Platform at First Citizens Bank. The platform addresses chronic operational inefficiencies in handling thousands of monthly ServiceNow incidents, specifically targeting incorrect assignment routing, prolonged MTTR, SLA breaches, and redundant root cause investigations.

---

## 2. Functional Requirements by Module

### Module 1: Exploratory Data Analysis (EDA)
- **REQ-EDA-1:** System shall compute statistical summaries across all numerical and categorical incident fields.
- **REQ-EDA-2:** System shall identify class imbalance ratios across Assignment Groups, Categories, and Priorities.

### Module 2: Data Cleaning
- **REQ-CLEAN-1:** System shall impute missing values using domain-standard fallback defaults without dropping records prematurely.
- **REQ-CLEAN-2:** System shall normalize and sanitize all unstructured text fields (`short_description`, `description`, `close_notes`).

### Module 3: Feature Engineering
- **REQ-FE-1:** System shall extract temporal indicators (business hours vs. weekend/after-hours, elapsed duration).
- **REQ-FE-2:** System shall apply proper categorical encoding (`LabelEncoding` / `FrequencyEncoding`) suitable for Random Forest processing.

### Module 4: Assignment Group Prediction
- **REQ-ML-AG-1:** System shall train a `RandomForestClassifier` to predict target Assignment Groups with confidence scores.
- **REQ-ML-AG-2:** System shall output the top-3 most likely Assignment Groups along with their respective probabilities.

### Module 5: Resolution Recommendation
- **REQ-REC-1:** System shall extract and rank historical `close_notes` and `resolution_code` from top semantically similar incidents.
- **REQ-REC-2:** System shall synthesize actionable resolution steps to guide support engineers.

### Module 6: Resolution Time Prediction
- **REQ-ML-RT-1:** System shall train a `RandomForestRegressor` to estimate expected resolution time (`resolution_time_hours`).
- **REQ-ML-RT-2:** System shall flag tickets at elevated risk of SLA breach (`resolution_time_hours > sla_target`).

### Module 7: Hyperparameter Optimization
- **REQ-HPO-1:** System shall execute `RandomizedSearchCV` across Random Forest hyperparameter grids (`n_estimators`, `max_depth`, `class_weight`).

### Module 8: Explainable AI (XAI)
- **REQ-XAI-1:** System shall compute permutation feature importance across all trained Random Forest models.
- **REQ-XAI-2:** System shall calculate local SHAP values to explain individual ticket predictions to support technicians.

### Module 9: Optional Local RAG
- **REQ-RAG-1:** System shall support integration with local `Ollama` LLM (`llama3.1:8b`) to summarize similar incident close notes without sending data to external APIs.

### Module 10: Enterprise Dashboard
- **REQ-DASH-1:** System shall provide an interactive `Streamlit` dashboard allowing engineers to input or select tickets and view AI predictions, similar incidents, and SHAP explanations in real time.

### Module 11: Professional Reporting
- **REQ-REP-1:** System shall generate styled `OpenPyXL` Excel workbooks containing daily, weekly, and monthly SLA and assignment trend analytics.

### Module 12: Deployment & Automation
- **REQ-DEP-1:** System shall provide automated cross-platform batch (`.bat` / `.sh`) execution scripts covering conda environment creation, dataset generation, and service launching.

---

## 3. Non-Functional Requirements & Constraints

- **Security & Privacy (Bank Zero-Egress Mandate):** All processing, model training, text embedding generation (`all-MiniLM-L6-v2`), and LLM inference (`Ollama`) must execute **exclusively on-premises within the local environment**. Absolutely zero network egress to third-party or cloud AI endpoints is permitted.
- **Hardware Optimization:** Architecture must operate efficiently on a Windows development environment equipped with an Intel Core i5, NVIDIA GTX GPU, and 16GB RAM by utilizing memory-bounded chunking and generator patterns.
- **Scalability:** The data pipeline and dataset generator must scale seamlessly from 10,000 records up to 1,000,000+ records without encountering memory exhaustion (`Out-Of-Memory` errors).

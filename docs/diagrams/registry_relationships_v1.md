# Enterprise Registry Relationships Architecture (`v1.5.0`)

**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  
**Governance Level:** Core Platform Contract & Single Source of Truth  
**Target Release:** `v1.5.0-alpha`

---

## 1. Executive Summary

To prevent schema drift, hardcoded column arrays, and target leakage across our multi-module enterprise platform, First Citizens Bank enforces the **Central Enterprise Registry Layer**. This architectural enhancement introduces four interconnected registries (`Feature Registry`, `Feature Lineage Tracker`, `Model Registry`, and `Embedding Registry`) that act as the binding contract across `Random Forest`, `FAISS`, `TF-IDF Dashboard`, `TF-IDF REST Endpoints`, and `Future RAG`.

---

## 2. Registry Relationships & Governance Graph (Mermaid)

```mermaid
graph TD
    classDef regStyle fill:#0f172a,stroke:#6366f1,stroke-width:3px,color:#ffffff;
    classDef linStyle fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef conStyle fill:#334155,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef modStyle fill:#1d4ed8,stroke:#34d399,stroke-width:2px,color:#ffffff;

    subgraph Central_Registry_Layer ["Central Enterprise Registry Layer (Single Source of Truth)"]
        FR["1. Feature Registry<br/>(`src/data/feature_registry.py`)<br/>• 38 Raw + 11 Derived Attributes<br/>• 22 Governance Dimensions<br/>• Leakage Tiers: Safe / Warning / Blocked"] ::: regStyle
        FL["2. Feature Lineage Tracker<br/>(`src/data/feature_lineage.py`)<br/>• Parent-Child Ancestry<br/>• Mathematical Formulas<br/>• Transformation Stages"] ::: linStyle
        MR["3. Model Registry<br/>(`src/ml/model_registry.py`)<br/>• SHA256 Checksum Verification<br/>• Feature Registry Version Match<br/>• Hyperparameters & Metrics"] ::: regStyle
        ER["4. Embedding & FAISS Registry<br/>(`src/ml/embedding_registry.py`)<br/>• tfidf-svd-384 (384-D)<br/>• 256 Token Truncation Rule<br/>• Distance Metrics (L2/IP)"] ::: regStyle
    end

    FR <-->|Derivation & Ancestry Rules| FL
    MR -->|Validates Authorized Features & Version| FR
    ER -->|Validates Text Input Schema| FR

    Con["5. Pipeline Contract Validator<br/>(`src/data/pipeline_contracts.py`)<br/>• Dynamic Consumer Adapters<br/>• Schema Compliance Engine<br/>• Target Leakage Interceptor"] ::: conStyle
    FR & MR & ER --> Con

    subgraph Downstream_Consumers ["Downstream Enterprise Consumers"]
        RF["Random Forest Engine<br/>(`get_catboost_features()`)"] ::: modStyle
        ST["SentenceTransformer & FAISS<br/>(`get_embedding_text_features()`)"] ::: modStyle
        UI["TF-IDF EDA Dashboard<br/>(`get_dashboard_kpi_features()`)"] ::: modStyle
        API["TF-IDF REST Ingestion<br/>(`get_api_request_schema()`)"] ::: modStyle
        RAG["Future RAG Knowledge Base<br/>(`get_rag_knowledge_features()`)"] ::: modStyle
    end

    Con --> RF & ST & UI & API & RAG
```

---

## 3. Detailed Contract Specifications

### 3.1 Feature Registry (`FeatureRegistry`)
- **Responsibility:** Manages 49 total features (`38` raw ServiceNow schema columns + `11` derived engineering flags).
- **22-Dimension Matrix:** Every feature definition requires `business_name`, `technical_name`, `data_type`, `nullable`, `cardinality`, `missing_percentage`, `business_meaning`, `ml_importance`, `target_leakage_classification`, `encoding_strategy`, `imputation_strategy`, `scaling_strategy`, `feature_engineering_rules`, `catboost_usage`, `embedding_usage`, `faiss_metadata_usage`, `dashboard_usage`, `api_exposure`, `future_rag_usage`, `explainability_usage`, `required_or_optional`, `deprecated_status`.
- **Target Leakage Enforcement:**
  - `Safe`: Available immediately when ticket opens (`short_description`, `priority`, `category`).
  - `Warning`: Caution required (`change_request`, `problem_record`).
  - `Blocked`: Post-resolution outcomes explicitly rejected from triage models (`close_notes`, `resolved_at`, `resolution_code`, `made_sla`, `u_caused_by`).

### 3.2 Feature Lineage (`FeatureLineageTracker`)
- **Responsibility:** Records exact transformations producing derived columns (`opened_at_hour_sin/cos`, `is_business_hours`, `resolution_time_hours`).
- **Auditability:** Enables instant back-traceability from any downstream ML model coefficient back to the underlying raw IT operational field.

### 3.3 Model & Embedding Registries (`ModelRegistry` & `EmbeddingRegistry`)
- **Responsibility:** Enforces cryptographic integrity (`SHA256`) and version matching (`v1.5.0`) on all `.joblib` / `.pkl` and `.faiss` vector indexes.
- **Safety Interlock:** If a model file has been modified outside the training pipeline or attempts to use `Blocked` leakage features, `ModelRegistry.verify_and_load_model_path()` raises a fatal exception.

---

## 4. Verification & Readiness for Phase 2 EDA
All downstream modules have been tested via `PipelineContractValidator` (`tests/unit/test_pipeline_contracts.py`), verifying 100% compliance across `Random Forest`, `Embeddings`, `FAISS`, `Dashboard`, `API`, and `RAG`.

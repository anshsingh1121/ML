# Feature Engineering Pipeline Architecture (`v1.5.0`)

**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  
**Document Version:** `v1.0.0` (Phase 1.5 Specification)  
**Execution Environment:** On-Premises Local Processing (Zero Cloud Egress)  

---

## High-Level Feature Engineering Flow

The following Mermaid architecture diagram illustrates the end-to-end data processing, validation, cleaning, encoding, feature storage, and multi-model consumption pipeline:

```mermaid
graph TD
    %% Define Node Styles
    classDef rawStyle fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef valStyle fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef cleanStyle fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef encStyle fill:#334155,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef storeStyle fill:#1d4ed8,stroke:#93c5fd,stroke-width:3px,color:#ffffff;
    classDef modelStyle fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#ffffff;
    classDef simStyle fill:#047857,stroke:#34d399,stroke-width:2px,color:#ffffff;

    %% Data Ingestion & Validation
    Raw["Raw ServiceNow Incidents<br/>(CSV / Parquet / REST API)"] ::: rawStyle
    Val["Dataset Validator<br/>(12 Quality Rules & Leakage Checks)"] ::: valStyle
    Report["Validation & Readiness Reports<br/>(Markdown & JSON Audit)"] ::: valStyle

    Raw --> Val
    Val -->|Log Anomalies & Certify| Report
    Val -->|Validated Dataset| Clean["Data Cleaning Layer<br/>(Null Imputation & Whitespace Normalization)"] ::: cleanStyle

    %% Feature Encoding Split
    Clean --> EncCat["Categorical & Numerical Encoding<br/>(Ordinal / Label / Target / Sine-Cosine)"] ::: encStyle
    Clean --> EncText["Text Normalization & Tokenization<br/>(short_description & description)"] ::: encStyle

    %% Feature Store
    EncCat --> Store["On-Premises Feature Store<br/>(Structured Arrays & Metadata Manifests)"] ::: storeStyle
    EncText --> Store

    %% Model Consumption
    Store --> RF["Random Forest Classifier / Regressor<br/>(Assignment Group & MTTR Prediction)"] ::: modelStyle
    Store --> ST["SentenceTransformer<br/>(all-MiniLM-L6-v2 384-D Dense Vectors)"] ::: modelStyle

    %% Vector Store & Similarity
    ST --> FAISS["FAISS Vector Index<br/>(IVFFlat / IndexFlatIP Euclidean & Cosine)"] ::: modelStyle
    Store --> Struc["Structural Exact Matching<br/>(CMDB CI / Business Service / Category)"] ::: modelStyle

    FAISS --> Hybrid["Hybrid Similarity Engine<br/>(70% Semantic Dense + 30% Structural Boost)"] ::: simStyle
    Struc --> Hybrid
```

---

## Pipeline Stage Descriptions

1. **Raw ServiceNow Incidents (`Raw`):**
   Involves ingesting historical incidents from `.csv`, `.parquet`, or real-time REST API streams into Pandas dataframes (`datasets/synthetic/vX/`).

2. **Dataset Validation & Readiness (`Val` $\rightarrow$ `Report`):**
   Every batch passes through `DatasetValidator` (checking all 12 enterprise data rules) and `MLReadinessEvaluator` (verifying zero target leakage, cardinality bounds, and token length distribution). If anomalies exceed governance limits, the pipeline halts with a failed quality certification.

3. **Data Cleaning Layer (`Clean`):**
   Standardizes null/empty strings to `'UNKNOWN'`, trims whitespace, resolves parent-child cluster links, and handles outlier values without mutating immutable raw history.

4. **Categorical & Numerical Encoding (`EncCat`):**
   - **Ordinal/Label Encoding:** Applied to `priority`, `impact`, `urgency`, `severity`.
   - **Target / Frequency Encoding:** Applied to high-cardinality fields (`subcategory`, `assignment_group`, `business_service`).
   - **Cyclic Sine/Cosine Encoding:** Applied to temporal timestamps (`opened_at_hour`, `opened_at_dayofweek`) to maintain daily continuous shift continuity.

5. **Text Normalization & Tokenization (`EncText`):**
   Prepares `short_description` and `description` for dense neural embeddings by truncating to 256 tokens (`all-MiniLM-L6-v2` sequence maximum) and removing HTML/escape artifacts.

6. **On-Premises Feature Store (`Store`):**
   Centralized, memory-mapped storage array holding preprocessed structural matrices and serialized text corpora ready for training and inference without re-running data prep.

7. **Multi-Model Processing (`RF`, `ST` $\rightarrow$ `FAISS` $\rightarrow$ `Hybrid`):**
   - **Random Forest (`RF`):** Consumes tabular numerical and encoded features to classify `assignment_group` (with `class_weight='balanced'`) and predict `resolution_time_hours`.
   - **SentenceTransformer (`ST`):** Generates 384-D L2-normalized dense embeddings, indexed inside local `FAISS` vector databases.
   - **Hybrid Similarity Engine (`Hybrid`):** Blends semantic vector distance ($0.70$ weight) with categorical structural exact matching ($0.30$ weight on `cmdb_ci`, `category`, `business_service`) to recommend historical resolutions with maximum precision.

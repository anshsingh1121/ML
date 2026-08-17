# Enterprise AI Incident Intelligence Architecture

This document outlines the complete end-to-end data flow, machine learning training pipelines, and live inference mechanisms powering the AI Incident Intelligence Platform.

## System Architecture Diagram

```mermaid
graph TD
    %% Styling Definitions
    classDef dataLayer fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#ecf0f1;
    classDef prepLayer fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;
    classDef mlLayer fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
    classDef nlpLayer fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;
    classDef hybridLayer fill:#f39c12,stroke:#f1c40f,stroke-width:2px,color:#fff;
    classDef uiLayer fill:#c0392b,stroke:#e74c3c,stroke-width:2px,color:#fff;

    %% 1. Data Ingestion & Quality Gate
    subgraph Phase 1: Data Ingestion & Quality Gate
        A[(Raw Corporate Data<br/>incidents.csv)]:::dataLayer --> B{Data Quality Gate}:::prepLayer
        B -->|Pass| C[Data Preprocessing Engine]:::prepLayer
        B -->|Fail| Z[Quarantine / Error Log]:::dataLayer
    end

    %% 2. Preprocessing & Feature Engineering
    subgraph Phase 2: Feature Engineering
        C --> D[Text Processing<br/>Regex, Stopwords]:::prepLayer
        C --> E[Categorical Encoding<br/>OneHot, Ordinal]:::prepLayer
        C --> F[Temporal Engineering<br/>Time-to-Resolution]:::prepLayer
        D & E & F --> G{Train / Validation Split<br/>80% / 20%}:::prepLayer
    end

    %% 3. Dual AI Training Pipeline
    subgraph Phase 3: Dual AI Engine Training
        %% ML Path
        G -->|Structured Features| H(RandomizedSearchCV<br/>Hyperparameter Optimization):::mlLayer
        H --> I[CatBoost Classifier<br/>Assignment Group Target]:::mlLayer
        H --> J[CatBoost Regressor<br/>Resolution Time MTTR Target]:::mlLayer
        
        %% NLP Path
        G -->|Text Features| K(NLP Embeddings<br/>TF-IDF + Truncated SVD):::nlpLayer
        K --> L[FAISS Vector Database<br/>Exact Similarity Index]:::nlpLayer
    end

    %% 4. Live Production Inference
    subgraph Phase 4: Live Hybrid Inference (Streamlit)
        M[Streamlit UI<br/>User Inputs Ticket]:::uiLayer --> N[Feature Sync<br/>Zero-Crash Fallback]:::prepLayer
        
        %% Routing the live ticket
        N --> O[CatBoost Prediction]:::mlLayer
        N --> P[FAISS Semantic Search]:::nlpLayer
        
        %% The Decision Engine
        O -.-> Q{Hybrid Decision Engine}:::hybridLayer
        P -.-> Q
        
        %% Final Output Synthesis
        Q --> R1[Predicted Team]:::hybridLayer
        Q --> R2[Estimated MTTR]:::hybridLayer
        Q --> R3[Confidence Tier<br/>Auto-Route vs Review]:::hybridLayer
        Q --> R4[Top 5 Precedents]:::hybridLayer
        
        R1 & R2 & R3 & R4 --> S[Display to User]:::uiLayer
    end
```

---

## Architectural Deep Dive

### 1. Data Ingestion & Quality Gate
The system ingest the corporate 25-column `.csv` schema. The `QualityGate` enforces strict rules (e.g., dropping records with missing target variables) to prevent garbage-in, garbage-out (GIGO) scenarios.

### 2. Feature Engineering & Zero-Leakage Split
The data is strictly split 80/20 *before* any statistical transformations occur. This mathematical separation prevents "Target Leakage"—ensuring the AI doesn't memorize the validation set. Missing fields are injected with `UNKNOWN` safe defaults.

### 3. Dual AI Engine Training
The platform trains two completely separate AI systems simultaneously:
* **The Mathematical Engine (CatBoost):** A gradient boosting algorithm wrapped in a Scikit-Learn `RandomizedSearchCV` shell. It tests 30 different tree architectures (depth, learning rate, L2 regularization) to find the absolute mathematically optimal model for the 6,000-row dataset, generating both a Classifier (Routing) and Regressor (Time).
* **The Memory Engine (FAISS):** A Natural Language Processing pipeline that vectorizes the text descriptions (TF-IDF + SVD) into multi-dimensional space, and saves them into an Exact-Match FAISS Vector Index.

### 4. Live Hybrid Inference
When a new ticket is submitted via the **Streamlit Dashboard**, the `HybridRecommendationEngine` activates:
1. It queries **CatBoost** to get a baseline ML prediction.
2. It queries **FAISS** to pull the Top 5 most semantically similar historical tickets.
3. The **Decision Engine** fuses these results. If FAISS finds overwhelming historical consensus that contradicts CatBoost, it will override the prediction and adjust the Confidence Tier accordingly.

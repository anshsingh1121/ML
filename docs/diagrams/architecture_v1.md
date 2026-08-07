# AI-Powered Incident Intelligence Platform — Architecture Document

| Field                | Value                                        |
|----------------------|----------------------------------------------|
| **Version**          | 1.0                                          |
| **Date**             | 2026-07-10                                   |
| **Status**           | Under Review                                 |
| **Author**           | Principal Software Architect                 |
| **Organization**     | First Citizens Bank — IT Operations           |
| **Classification**   | Internal / Confidential                      |

---

## Table of Contents

1. [High-Level System Architecture](#1-high-level-system-architecture)
2. [Component Diagram](#2-component-diagram)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Deployment Diagram](#4-deployment-diagram)

---

## 1. High-Level System Architecture

This diagram presents a top-down view of the platform's layered architecture. Data flows downward from ingestion through processing and machine learning into the resolution engine, and finally surfaces through the presentation layer. The infrastructure layer runs orthogonally, providing cross-cutting concerns (configuration, logging, model persistence) to every other layer.

```mermaid
graph TB
    %% ── Styling ──────────────────────────────────────────────
    classDef ingestion   fill:#1e3a5f,stroke:#4a90d9,stroke-width:2px,color:#e8f4fd,font-weight:bold
    classDef processing  fill:#2d4a22,stroke:#6abf4b,stroke-width:2px,color:#e8fde8,font-weight:bold
    classDef ml          fill:#5c2d82,stroke:#a855f7,stroke-width:2px,color:#f3e8ff,font-weight:bold
    classDef resolution  fill:#7c3a1a,stroke:#f97316,stroke-width:2px,color:#fff7ed,font-weight:bold
    classDef presentation fill:#1a3c5c,stroke:#06b6d4,stroke-width:2px,color:#ecfeff,font-weight:bold
    classDef infra       fill:#3b3b3b,stroke:#9ca3af,stroke-width:2px,color:#f3f4f6,font-weight:bold

    %% ── Layer 1 · Data Ingestion ─────────────────────────────
    subgraph L1["① DATA INGESTION LAYER"]
        direction LR
        SN["ServiceNow REST API"]
        CSV["CSV / Excel Import"]
        MI["Manual Input"]
    end

    %% ── Layer 2 · Processing ─────────────────────────────────
    subgraph L2["② PROCESSING LAYER"]
        direction LR
        EDA["Exploratory Data Analysis"]
        DC["Data Cleaning & Validation"]
        FE["Feature Engineering"]
    end

    %% ── Layer 3 · Machine Learning ───────────────────────────
    subgraph L3["③ MACHINE LEARNING LAYER"]
        direction LR
        RF["Random Forest Classifiers"]
        ST["Sentence Transformer (all-MiniLM-L6-v2)"]
        FI["FAISS Vector Index"]
    end

    %% ── Layer 4 · Resolution ─────────────────────────────────
    subgraph L4["④ RESOLUTION LAYER"]
        direction LR
        SE["Similarity Engine (α=0.7 / β=0.3)"]
        RR["Resolution Recommender"]
        RAG["RAG Engine (Ollama llama3.1:8b)"]
    end

    %% ── Layer 5 · Presentation ───────────────────────────────
    subgraph L5["⑤ PRESENTATION LAYER"]
        direction LR
        SD["Streamlit Dashboard"]
        XR["Excel Reports"]
    end

    %% ── Layer 6 · Infrastructure (cross-cutting) ─────────────
    subgraph L6["⑥ INFRASTRUCTURE LAYER (Cross-Cutting)"]
        direction LR
        CFG["Config (YAML)"]
        LOG["Logging"]
        MR["Model Registry"]
    end

    %% ── Inter-layer flows ────────────────────────────────────
    L1 -->|"Raw incident records"| L2
    L2 -->|"Clean, engineered features"| L3
    L3 -->|"Predictions & embeddings"| L4
    L4 -->|"Recommendations & explanations"| L5

    L6 -.->|"Supports"| L1
    L6 -.->|"Supports"| L2
    L6 -.->|"Supports"| L3
    L6 -.->|"Supports"| L4
    L6 -.->|"Supports"| L5

    %% ── Apply styles ─────────────────────────────────────────
    class SN,CSV,MI ingestion
    class EDA,DC,FE processing
    class RF,ST,FI ml
    class SE,RR,RAG resolution
    class SD,XR presentation
    class CFG,LOG,MR infra
```

---

## 2. Component Diagram

This diagram maps every source module (`src/` subdirectory and file) and illustrates the dependency graph between them. Arrows indicate "depends on" or "calls into" relationships. Understanding these dependencies is critical for build ordering, test isolation, and change-impact analysis.

```mermaid
graph LR
    %% ── Styling ──────────────────────────────────────────────
    classDef dataStyle       fill:#1e3a5f,stroke:#4a90d9,stroke-width:2px,color:#e8f4fd,font-weight:bold
    classDef modelStyle      fill:#5c2d82,stroke:#a855f7,stroke-width:2px,color:#f3e8ff,font-weight:bold
    classDef resolStyle      fill:#7c3a1a,stroke:#f97316,stroke-width:2px,color:#fff7ed,font-weight:bold
    classDef explainStyle    fill:#065f46,stroke:#10b981,stroke-width:2px,color:#ecfdf5,font-weight:bold
    classDef dashStyle       fill:#1a3c5c,stroke:#06b6d4,stroke-width:2px,color:#ecfeff,font-weight:bold
    classDef reportStyle     fill:#713f12,stroke:#eab308,stroke-width:2px,color:#fefce8,font-weight:bold
    classDef utilStyle       fill:#3b3b3b,stroke:#9ca3af,stroke-width:2px,color:#f3f4f6,font-weight:bold

    %% ── src/data/ ────────────────────────────────────────────
    subgraph DATA["src/data/"]
        D1["ingestion.py"]
        D2["cleaning.py"]
        D3["feature_engineering.py"]
    end

    %% ── src/models/ ──────────────────────────────────────────
    subgraph MODELS["src/models/"]
        M1["assignment_predictor.py"]
        M2["resolution_time_predictor.py"]
        M3["hyperparameter_optimizer.py"]
        M4["model_registry.py"]
    end

    %% ── src/resolution/ ──────────────────────────────────────
    subgraph RESOLUTION["src/resolution/"]
        R1["similarity_engine.py"]
        R2["resolution_recommender.py"]
        R3["rag_engine.py"]
    end

    %% ── src/explainability/ ──────────────────────────────────
    subgraph EXPLAIN["src/explainability/"]
        E1["shap_explainer.py"]
    end

    %% ── src/dashboard/ ───────────────────────────────────────
    subgraph DASHBOARD["src/dashboard/"]
        A1["app.py"]
        A2["pages/"]
    end

    %% ── src/reporting/ ───────────────────────────────────────
    subgraph REPORTING["src/reporting/"]
        RP1["excel_reporter.py"]
    end

    %% ── src/utils/ ───────────────────────────────────────────
    subgraph UTILS["src/utils/"]
        U1["logger.py"]
        U2["config_loader.py"]
    end

    %% ── Intra-package dependencies ───────────────────────────
    D1 -->|"feeds"| D2
    D2 -->|"feeds"| D3
    M3 -->|"tunes"| M1
    M3 -->|"tunes"| M2
    M1 -->|"registers"| M4
    M2 -->|"registers"| M4
    R1 -->|"drives"| R2
    R3 -->|"augments"| R2

    %% ── Cross-package dependencies ───────────────────────────
    D3 -->|"features"| M1
    D3 -->|"features"| M2
    D3 -->|"embeddings"| R1
    M1 -->|"predictions"| R2
    M2 -->|"predictions"| R2
    M4 -->|"loads models"| R2
    M1 -->|"model objects"| E1
    M2 -->|"model objects"| E1

    %% ── Dashboard consumes everything ────────────────────────
    R2 -->|"recommendations"| A1
    E1 -->|"explanations"| A1
    A1 --> A2
    R2 -->|"data"| RP1

    %% ── Utils are consumed by all ────────────────────────────
    U1 -.->|"logging"| D1
    U1 -.->|"logging"| M1
    U1 -.->|"logging"| R1
    U1 -.->|"logging"| A1
    U2 -.->|"config"| D1
    U2 -.->|"config"| M1
    U2 -.->|"config"| R1
    U2 -.->|"config"| A1

    %% ── Apply styles ─────────────────────────────────────────
    class D1,D2,D3 dataStyle
    class M1,M2,M3,M4 modelStyle
    class R1,R2,R3 resolStyle
    class E1 explainStyle
    class A1,A2 dashStyle
    class RP1 reportStyle
    class U1,U2 utilStyle
```

---

## 3. Data Flow Diagram

This diagram traces the journey of a single incident record from its origin in ServiceNow through every transformation stage until it reaches the end-user on the Streamlit dashboard. Each node describes the data shape at that point in the pipeline.

```mermaid
graph TD
    %% ── Styling ──────────────────────────────────────────────
    classDef source      fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#dbeafe,font-weight:bold
    classDef ingest      fill:#1e3a5f,stroke:#4a90d9,stroke-width:2px,color:#e8f4fd,font-weight:bold
    classDef process     fill:#2d4a22,stroke:#6abf4b,stroke-width:2px,color:#e8fde8,font-weight:bold
    classDef train       fill:#5c2d82,stroke:#a855f7,stroke-width:2px,color:#f3e8ff,font-weight:bold
    classDef index       fill:#4a1d6a,stroke:#c084fc,stroke-width:2px,color:#f5f3ff,font-weight:bold
    classDef infer       fill:#7c3a1a,stroke:#f97316,stroke-width:2px,color:#fff7ed,font-weight:bold
    classDef output      fill:#1a3c5c,stroke:#06b6d4,stroke-width:2px,color:#ecfeff,font-weight:bold
    classDef storage     fill:#3b3b3b,stroke:#9ca3af,stroke-width:2px,color:#f3f4f6,font-weight:bold

    %% ── Source ───────────────────────────────────────────────
    SRC["🏢 ServiceNow Instance\n(Incident Table)"]

    %% ── Ingestion ────────────────────────────────────────────
    ING["📥 ingestion.py\nREST API pull / CSV import\n→ Raw DataFrame (all fields)"]

    %% ── Cleaning ─────────────────────────────────────────────
    CLN["🧹 cleaning.py\nNull handling, deduplication,\ntype casting, outlier removal\n→ Validated DataFrame"]

    %% ── Feature Engineering ──────────────────────────────────
    FE["⚙️ feature_engineering.py\nText preprocessing, label encoding,\nTF-IDF, temporal features\n→ Feature Matrix (X) + Labels (y)"]

    %% ── Training Branch ──────────────────────────────────────
    TR_RF["🌲 Random Forest Training\nassignment_predictor.py\nresolution_time_predictor.py\n→ Trained .joblib models"]
    TR_ST["🔤 Sentence Transformer\nall-MiniLM-L6-v2\n→ 384-dim embedding vectors"]

    %% ── Indexing ─────────────────────────────────────────────
    FAISS["📊 FAISS Index Build\nIVFFlat / FlatL2\n→ .faiss index file"]

    %% ── Model Persistence ────────────────────────────────────
    REG["💾 Model Registry\nVersioned .joblib + .faiss\n+ metadata JSON"]

    %% ── Inference ────────────────────────────────────────────
    INF_NEW["🆕 New Incident Arrives\n(Real-time or batch)"]
    INF_RF["🎯 RF Inference\nAssignment Group, Priority,\nCategory, Resolution Time"]
    INF_SIM["🔍 Similarity Search\nα=0.7 semantic + β=0.3 structural\n→ Top-K similar incidents"]

    %% ── Resolution ───────────────────────────────────────────
    RES["💡 Resolution Recommender\nMerge RF predictions +\nSimilar incident resolutions"]
    RAG["🤖 RAG Engine (Optional)\nOllama llama3.1:8b\nContext-augmented explanation"]

    %% ── Explainability ───────────────────────────────────────
    SHAP["📈 SHAP Explainer\nFeature importance per prediction"]

    %% ── Output ───────────────────────────────────────────────
    DASH["📊 Streamlit Dashboard\nInteractive predictions,\nexplanations, analytics"]
    EXCEL["📄 Excel Reports\nBatch results export"]

    %% ── Flows ────────────────────────────────────────────────
    SRC -->|"JSON / CSV"| ING
    ING -->|"Raw DataFrame"| CLN
    CLN -->|"Clean DataFrame"| FE

    FE -->|"Feature matrix"| TR_RF
    FE -->|"Text corpus"| TR_ST

    TR_RF -->|".joblib"| REG
    TR_ST -->|"Embeddings"| FAISS
    FAISS -->|".faiss"| REG

    INF_NEW -->|"Incident payload"| INF_RF
    INF_NEW -->|"Short description"| INF_SIM
    REG -->|"Load models"| INF_RF
    REG -->|"Load index"| INF_SIM

    INF_RF -->|"Predictions"| RES
    INF_SIM -->|"Similar incidents"| RES
    RES -->|"Context"| RAG
    INF_RF -->|"Model + data"| SHAP

    RES -->|"Recommendations"| DASH
    RAG -->|"Augmented response"| DASH
    SHAP -->|"Explanations"| DASH
    RES -->|"Batch data"| EXCEL

    %% ── Apply styles ─────────────────────────────────────────
    class SRC source
    class ING ingest
    class CLN,FE process
    class TR_RF,TR_ST train
    class FAISS index
    class REG storage
    class INF_NEW,INF_RF,INF_SIM infer
    class RES,RAG infer
    class SHAP train
    class DASH,EXCEL output
```

---

## 4. Deployment Diagram

This diagram illustrates the on-premises deployment topology. The entire platform executes within the bank's secure internal network on a developer workstation. No data leaves the perimeter. A Conda virtual environment isolates all Python dependencies, and local storage houses models, data, logs, and configuration.

```mermaid
graph TB
    %% ── Styling ──────────────────────────────────────────────
    classDef network     fill:#0f172a,stroke:#3b82f6,stroke-width:3px,color:#dbeafe,font-weight:bold
    classDef workstation fill:#1e293b,stroke:#64748b,stroke-width:2px,color:#e2e8f0,font-weight:bold
    classDef conda       fill:#2d4a22,stroke:#6abf4b,stroke-width:2px,color:#e8fde8,font-weight:bold
    classDef appcomp     fill:#5c2d82,stroke:#a855f7,stroke-width:2px,color:#f3e8ff,font-weight:bold
    classDef storage     fill:#7c3a1a,stroke:#f97316,stroke-width:2px,color:#fff7ed,font-weight:bold
    classDef external    fill:#1a3c5c,stroke:#06b6d4,stroke-width:2px,color:#ecfeff,font-weight:bold
    classDef ollama      fill:#713f12,stroke:#eab308,stroke-width:2px,color:#fefce8,font-weight:bold

    %% ── Bank Network Boundary ────────────────────────────────
    subgraph BANK["🏦 First Citizens Bank — Secure Internal Network"]
        direction TB

        %% ── ServiceNow (internal) ────────────────────────────
        SNOW["🌐 ServiceNow Instance\n(On-Prem / Internal URL)"]

        %% ── Developer Workstation ────────────────────────────
        subgraph WS["💻 Developer Workstation"]
            direction TB

            %% ── Conda Environment ────────────────────────────
            subgraph CONDA["🐍 Conda Environment (incident_env)"]
                direction LR
                PY["Python 3.11"]
                DEPS["Dependencies:\nscikit-learn, sentence-transformers,\nfaiss-cpu, streamlit, shap,\npandas, openpyxl, requests,\nlangchain, PyYAML"]
            end

            %% ── Application Components ───────────────────────
            subgraph APP["📦 Application Components"]
                direction TB
                SRC_CODE["src/\n├─ data/\n├─ models/\n├─ resolution/\n├─ explainability/\n├─ dashboard/\n├─ reporting/\n└─ utils/"]
                CFG_FILE["config/\n└─ config.yaml"]
                SCRIPTS["scripts/\n├─ setup.bat\n├─ train.bat\n└─ run.bat"]
            end

            %% ── Ollama (optional, local) ─────────────────────
            OLL["🤖 Ollama Server (Local)\nModel: llama3.1:8b\nPort: 11434"]

            %% ── Streamlit Server ─────────────────────────────
            STR["🖥️ Streamlit Dev Server\nPort: 8501\n(localhost only)"]

            %% ── Local Storage ────────────────────────────────
            subgraph STORE["💾 Local Storage"]
                direction LR
                S_MODELS["models/\n├─ assignment_rf.joblib\n├─ resolution_time_rf.joblib\n├─ faiss_index.faiss\n└─ metadata.json"]
                S_DATA["data/\n├─ raw/\n├─ processed/\n└─ exports/"]
                S_LOGS["logs/\n└─ app.log"]
            end
        end

        %% ── User Access ──────────────────────────────────────
        USER_BR["👤 Analyst / Engineer\n(Web Browser → localhost:8501)"]
    end

    %% ── Flows ────────────────────────────────────────────────
    SNOW -->|"REST API\n(Internal Network)"| SRC_CODE
    CONDA -->|"Runtime"| APP
    SRC_CODE -->|"Read/Write"| STORE
    SRC_CODE -->|"Serves"| STR
    SRC_CODE -->|"API call\n(localhost:11434)"| OLL
    STR -->|"HTTP"| USER_BR
    CFG_FILE -.->|"Loaded by"| SRC_CODE
    SCRIPTS -.->|"Automates"| CONDA

    %% ── Apply styles ─────────────────────────────────────────
    class SNOW external
    class WS workstation
    class PY,DEPS conda
    class SRC_CODE,CFG_FILE,SCRIPTS appcomp
    class S_MODELS,S_DATA,S_LOGS storage
    class OLL ollama
    class STR external
    class USER_BR network
```

---

## Revision History

| Version | Date       | Author                        | Changes                          |
|---------|------------|-------------------------------|----------------------------------|
| 1.0     | 2026-07-10 | Principal Software Architect  | Initial architecture diagrams    |

---

> **Note:** This document is versioned and subject to formal review. All modifications must be approved by the Architecture Review Board before implementation begins.

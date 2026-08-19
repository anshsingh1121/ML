# ServiceNow Incident Intelligence Platform
## Presentation Script — 11 Slides

---

# SLIDE 1: PROBLEM STATEMENT

## The Pain of Manual IT Triage

Organizations receive **thousands of IT incidents daily** through ServiceNow. Every ticket must be manually read, understood, and routed by an L1 Helpdesk agent to the correct team.

**This manual triaging creates 4 critical business problems:**

| Problem | Business Impact |
|---|---|
| **Incorrect Assignment Groups** | Tickets bounce between 3-4 teams. Each bounce adds 2-8 hours of dead time. |
| **Delayed Resolutions** | Misrouted tickets sit in wrong queues for hours/days. |
| **SLA Breaches** | Priority 1 incidents (4-hour SLA) can't afford a single misroute. Manual routing averages 15+ minutes per ticket. |
| **Knowledge Loss** | Resolved issues aren't recalled when the same problem appears again. Technicians start from scratch every time. |

> **Goal:** Build an AI system that instantly reads a ticket, predicts the correct team, estimates resolution time, and surfaces historical precedents — all in under 1 second.

---

# SLIDE 2: PROPOSED SOLUTION — ALL 9 OBJECTIVES MAPPED

## A Dual-Engine Hybrid AI Platform

| # | Objective | What We Built | Key Technology |
|---|---|---|---|
| 1 | Exploratory Data Analysis | `EnterpriseEDAEngine` — Shannon Entropy, Gini Impurity, 5 diagnostic charts, correlation matrices | Pandas, Matplotlib, Seaborn |
| 2 | Feature Engineering | `FeatureEngineeringEngine` — 7 categories of derived features (temporal, cyclic sin/cos, interaction, text-stats, frequency encodings) with 22-dimension governance registry | NumPy, FeatureRegistry |
| 3 | Assignment Group Prediction | `CatBoostClassifier` in scikit-learn pipeline with zero-leakage enforcement | CatBoost, scikit-learn |
| 4 | Resolution Recommendation Engine | `HybridRecommendationEngine` — Fuses CatBoost predictions + FAISS semantic search + confidence scoring + natural language reasoning | CatBoost + FAISS + Custom Engines |
| 5 | Resolution Time Prediction | `CatBoostRegressor` with log1p/expm1 target transform | CatBoost |
| 6 | Hyperparameter Tuning | `RandomizedSearchCV` — 5-fold CV, 30 iterations across 4 hyperparameters | scikit-learn |
| 7 | Explainable AI | `SHAPIntelligenceExplainer` — Global beeswarm + local waterfall plots via TreeExplainer | SHAP |
| 8 | Gen AI Resolution Assistant | FAISS Semantic Search surfaces Top-5 historically resolved precedents with resolution notes | FAISS, TF-IDF+SVD |
| 9 | Application/Dashboard | Streamlit real-time web UI with instant predictions, AI reasoning, and historical evidence | Streamlit |

---

# SLIDE 3: TECHNOLOGY STACK

| Layer | Technologies | Role |
|---|---|---|
| **ML & AI** | CatBoost, scikit-learn, FAISS (Meta), SHAP, NumPy, SciPy | Gradient boosting, pipelines, vector search, explainability |
| **NLP** | TfidfVectorizer, TruncatedSVD, L2 Normalization, Custom IT Lemmatizer | Text → 384-D dense vectors via Latent Semantic Analysis |
| **Data** | Pandas, OpenPyXL, XlsxWriter | DataFrame ops, CSV/Excel I/O |
| **Visualization** | Matplotlib, Seaborn, Plotly | Confusion matrices, ROC curves, SHAP plots, EDA charts |
| **Web UI** | Streamlit | Real-time dashboard with cached model inference |
| **Config** | PyYAML, Pydantic, python-dotenv | 4 YAML config files + .env secrets |
| **Serialization** | Joblib | Save/load scikit-learn pipelines as `.pkl` |
| **Testing** | pytest (23 test files), Ruff, mypy | 80% coverage threshold, type checking |
| **Language** | Python 3.11+ | Type hints, f-strings, dataclasses throughout |

---

# SLIDE 4: PIPELINE PART 1 — DATA INTELLIGENCE (Stages 1-8)

## From Raw CSV → ML-Ready Dataset

```
incidents.csv → Validate → Clean → Enrich → Engineer → NLP Preprocess → EDA → Split
```

### Stage 1-2: Data Validation & ML Readiness
- **12 automated quality rules** (nulls, duplicates, timestamp integrity, SLA consistency, valid priorities/categories)
- **ML Readiness audit:** Target leakage detection (blocks `close_notes`, `resolved_at`), Shannon Entropy for class imbalance, token capacity check

### Stage 3: Data Cleaning — 8-Step Remediation
| Step | Operation |
|---|---|
| 1 | Duplicate removal (keeps latest per ticket number) |
| 2 | Business rule standardization (`"4 - Low"` → integer `4`, clips priority to [1,5]) |
| 3 | Schema & type enforcement (datetime/int/float/boolean coercion) |
| 4 | Missing value imputation (text→`"Not Provided"`, numeric→median, categorical→mode) |
| 5 | Category validation (null/blank → `"Unknown"`) |
| 6 | Timestamp correction (`resolved_at < opened_at` → corrected to `+4 hours`) |
| 7 | Outlier winsorization (caps `reassignment_count` at 99th%ile or 15) |
| 8 | String normalization (trim whitespace) |

### Stage 4-5: Feature Engineering — 7 Categories
| Category | Key Features | Formula |
|---|---|---|
| Temporal | `opened_at_hour`, `is_weekend`, `is_business_hours`, `is_holiday` | Datetime extraction |
| Cyclic | `hour_sin`, `hour_cos`, `dayofweek_sin/cos` | $\sin(2\pi x / T)$, $\cos(2\pi x / T)$ |
| Interaction | `priority_x_business_impact` | $\text{priority} \times \text{impact}$ |
| Text Stats | `short_description_word_count`, `description_char_count` | `len(text.split())` |
| Frequency | `assignment_group_freq`, `caller_freq` | `value_counts(normalize=True)` |

### Stage 6: NLP Text Preprocessing
6-step pipeline: Unicode normalize → Strip HTML → Replace error codes → Remove punctuation → Filter stopwords (preserving 40+ IT keywords: `server`, `timeout`, `firewall`, `vpn`, `dns`) → IT lemmatization (`failures→failure`, `servers→server`)

### Stage 7: EDA — Automated Analysis
Shannon Entropy, Gini Impurity, correlation matrices, 5 diagnostic PNG charts

### Stage 8: Stratified Splitting
70/15/15 Train/Val/Test with **zero-leakage verification** (set intersection check: `Train ∩ Test = ∅`)

---

# SLIDE 5: PIPELINE PART 2 — MODEL TRAINING (Stages 9-10)

## CatBoost Classifier + Regressor with HPO

### Why CatBoost?
| Advantage | Why It Matters |
|---|---|
| Native categorical handling | No manual one-hot encoding for `category`, `assignment_group` |
| Ordered boosting | Prevents target leakage during tree construction |
| Symmetric trees | Sub-second inference for dashboard predictions |
| Built-in L2 regularization | Prevents overfitting on 6,000-row dataset |

### Complete Scikit-Learn Pipeline Architecture
```
Pipeline:
  Step 1: EnterpriseFeatureExtractor
          ├── Generates combined_text (short_description + description)
          ├── Computes priority_x_business_impact interaction
          └── Generates cyclic sin/cos temporal features
  Step 2: ColumnTransformer
          ├── FrequencyEncoder → high-cardinality categoricals
          ├── OneHotEncoder → low-cardinality categoricals
          ├── SimpleImputer(median) → numerical features
          └── TfidfVectorizer → combined text
  Step 3: CatBoostClassifier / CatBoostRegressor
```

### Hyperparameter Tuning (Objective 6)
| Parameter | Search Space | Method |
|---|---|---|
| `iterations` | [100, 300, 500, 800] | RandomizedSearchCV |
| `depth` | [4, 5, 6, 7] | 5-fold cross-validation |
| `learning_rate` | [0.03, 0.05, 0.1] | 30 iterations |
| `l2_leaf_reg` | [3, 5, 7, 9] | Scoring: `f1_weighted` |

### Regression Target Transform
Resolution times are right-skewed. We apply log-transform:
- **Train:** $y' = \log(1 + y)$
- **Predict:** $\hat{y} = e^{\hat{y'}} - 1$

### Zero-Leakage Enforcement
Every predictor is cross-referenced against FeatureRegistry. Any `blocked` column (`close_notes`, `resolved_at`, `close_code`) → **ValueError, training halts.**

---

# SLIDE 6: PIPELINE PART 3 — SEMANTIC SEARCH ENGINE (Stage 11)

## Text → Vector → FAISS Index → Top-K Retrieval

### Step 1: Text → 384-D Dense Vectors
```
"Database connection timeout on payment server DB-01"
    ↓
Construct semantic string:
"[Category: Database] [Service: Payment] [Priority: 2] Database connection timeout..."
    ↓
TF-IDF Vectorization (sparse matrix)
    ↓
TruncatedSVD → 384 dimensions (dense vector)
    ↓
L2 Normalization (unit length)
    ↓
[0.041, -0.088, 0.012, 0.094, ..., 0.031]  ← 384 numbers
```

### Step 2: FAISS Index
All 6,000 historical ticket vectors are loaded into **FAISS IndexFlatIP** (Exact Inner Product Search).

For L2-normalized vectors: $\text{cosine\_similarity}(a, b) = a \cdot b$

**100% recall guaranteed** — mathematically impossible to miss a match.

### Step 3: Query
When a new ticket arrives:
1. Convert to vector (same pipeline)
2. Search Top-K nearest neighbors in FAISS
3. Return historical precedents with similarity scores, resolution notes, and routing consensus

**This directly addresses Objective 4 (Resolution Recommendation) and Objective 8 (Knowledge Retrieval).**

---

# SLIDE 7: PIPELINE PART 4 — HYBRID INTELLIGENCE + EXPLAINABLE AI (Stage 12)

## The Crown Jewel: Fusing ML + Historical Evidence

### 6-Step Hybrid Workflow
```
Step 1: Parse Input (JSON/text → normalized ticket dict)
            ↓
Step 2: CatBoost Prediction → Predicted Group + Probabilities + MTTR
            ↓
Step 3: FAISS Search → Top-5 similar historical tickets + consensus group
            ↓
Step 4: Decision Fusion
        ├─ If AGREED:          Use CatBoost, +0.1 confidence bonus
        ├─ If CatBoost strong: Use CatBoost, -0.05 penalty
        └─ If FAISS dominant:  Override with historical consensus
            ↓
Step 5: Confidence Score = 0.6×CatBoost + 0.4×FAISS ± bonus
        Blended MTTR = 0.5×CatBoost_MTTR + 0.5×FAISS_median_MTTR
            ↓
Step 6: Reasoning → Natural language explanation (zero LLMs)
```

### Confidence Tiers
| Score | Tier | Action |
|---|---|---|
| ≥ 0.88 | Very High | Auto-route |
| ≥ 0.75 | High | Auto-route with notification |
| ≥ 0.60 | Moderate | Route with manual review |
| ≥ 0.45 | Low | Require human confirmation |
| < 0.45 | Review Required | Escalate to senior agent |

### Explainable AI (Objective 7) — SHAP
SHAP computes each feature's contribution to every prediction using game theory:
- **Global:** Which features matter most across ALL predictions (beeswarm + bar chart)
- **Local:** Why THIS ticket was routed to THIS team (waterfall + decision plot)

### Dashboard (Objective 9)
Streamlit web UI → Enter ticket details → Click "Predict" → See team, MTTR, confidence, reasoning, and Top-5 precedents in < 1 second.

---

# SLIDE 8: FUTURE IMPROVEMENTS — HNSW SCALABILITY

## Scaling from 6,000 to 10,000,000+ Tickets

### Current: Exact Search — O(N)
`IndexFlatIP` checks **every vector** per query. Fast at 6K, bottleneck at 10M.

### Future: HNSW — O(log N)
**Hierarchical Navigable Small World** builds a multi-layered graph:

| Layer | Density | Role |
|---|---|---|
| Layer 2 (Top) | Very sparse | Global "highway" — fast entry point |
| Layer 1 (Mid) | Medium | Regional navigation |
| Layer 0 (Base) | 100% vectors | Dense local search for exact match |

**Search:** Enter sparse top → hop down through layers → find nearest neighbor in ~30 hops instead of 10,000,000 comparisons.

### Config Change
```yaml
# Current:
faiss:
  index_type: Flat        # Exact, 100% recall

# Future:
faiss:
  index_type: HNSW        # Approximate, ~99% recall
  M: 32                   # Graph connections per node
  efSearch: 128           # Query-time accuracy control
```

**Result:** Sub-millisecond search at scale with negligible accuracy trade-off.

---

# SLIDE 9: MARKET STUDY

## The AI-Powered ITSM Market

### Market Size
| Metric | Value |
|---|---|
| Global ITSM Market (2024) | $12.2 Billion |
| Projected (2030) | $28.1 Billion |
| CAGR | 14.9% |

### Key Trends Aligned with Our Platform
| Industry Trend | Our Solution |
|---|---|
| **AIOps Adoption** | Dual-engine architecture (ML + Semantic) is a core AIOps pattern |
| **Shift-Left Strategy** | Auto-routing eliminates L1→L2 escalation delays |
| **Explainable AI Mandates** | SHAP provides auditable AI decisions (banking regulation compliant) |
| **On-Premise Security** | 100% local — no data egress to cloud |
| **Knowledge Management** | FAISS semantic search replaces static knowledge bases |

### Competitive Edge
| Competitor | Their Approach | Our Advantage |
|---|---|---|
| ServiceNow Predictive Intelligence | Cloud-based | We run on-premise |
| BMC Helix AIOps | Proprietary | We use open-source, fully customizable |
| IBM Watson AIOps | Heavy enterprise suite | Lightweight, single-command deployment |

---

# SLIDE 10: CONCLUSION

## All 9 Objectives — Delivered

| # | Objective | Status | Key Metric |
|---|---|---|---|
| 1 | EDA | ✅ | 5 diagnostic charts, entropy + Gini computed |
| 2 | Feature Engineering | ✅ | 49 features × 22 governance dimensions |
| 3 | Assignment Group Prediction | ✅ | CatBoost classifier with zero-leakage pipeline |
| 4 | Resolution Recommendation | ✅ | Hybrid engine: ML + FAISS + confidence + reasoning |
| 5 | Resolution Time Prediction | ✅ | CatBoost regressor with log1p transform |
| 6 | Hyperparameter Tuning | ✅ | RandomizedSearchCV (5-fold, 30 iterations) |
| 7 | Explainable AI | ✅ | SHAP global + local attribution |
| 8 | Knowledge Retrieval | ✅ | FAISS Top-5 historical precedent search |
| 9 | Dashboard | ✅ | Streamlit real-time web UI |

### Platform Stats
| Metric | Value |
|---|---|
| Production Code | ~7,500 lines across 33 Python files |
| Test Code | ~4,000 lines across 23 test files |
| Pipeline Stages | 12 sequential stages |
| Inference Latency | < 1 second |
| Deployment | Single command: `python main.py full-pipeline` |

---

# SLIDE 11: LIVE DEMO + Q&A

## Let the AI Prove Itself

1. Launch dashboard: `python run_dashboard.py`
2. Enter a sample ticket: *"Database connection timeout on payment gateway server DB-01"*
3. Click **"Predict Routing & MTTR 🚀"**
4. Show: Predicted Team → Confidence Tier → MTTR → AI Reasoning → Top-5 Precedents

**Invite audience to suggest their own ticket descriptions!**

---

### Questions?

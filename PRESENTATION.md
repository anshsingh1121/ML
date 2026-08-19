# ServiceNow Incident Intelligence Platform
## Complete Presentation Flow — Slide-by-Slide Script

---

# SLIDE 1: TITLE SLIDE

**Title:** ServiceNow Incident Intelligence Platform
**Subtitle:** AI-Powered Incident Triage, Resolution Prediction & Semantic Knowledge Retrieval
**Presenter:** [Your Name]

---

# SLIDE 2: PROBLEM STATEMENT

## The Pain of Manual IT Triage

Organizations receive **thousands of IT incidents daily** through ServiceNow. Every single ticket must be manually read, understood, and routed by an L1 Helpdesk agent to the correct specialized technical team.

**This manual triaging creates 4 critical business problems:**

| # | Problem | Business Impact |
|---|---|---|
| 1 | **Incorrect Assignment Groups** | Tickets bounce between 3-4 teams before reaching the right one. Each bounce adds 2-8 hours of dead time. |
| 2 | **Delayed Resolutions** | Misrouted tickets sit in the wrong queue for hours/days, violating response commitments. |
| 3 | **SLA Breaches** | Priority 1 incidents (4-hour SLA) cannot afford even a single misroute. Manual routing averages 15+ minutes per ticket. |
| 4 | **Knowledge Loss** | When a technician solves a complex issue, that resolution knowledge lives only in their head. The next time an identical issue appears, another technician starts from scratch. |

> **The Core Question:** Can we build an AI system that instantly reads a ticket description, predicts the correct team, estimates resolution time, and surfaces historical precedents — all in under 1 second?

---

# SLIDE 3: PROPOSED SOLUTION

## An AI-Powered Incident Intelligence Platform

We built a **Dual-Engine Hybrid AI System** that addresses all 9 project objectives:

| # | Objective | Our Implementation | Status |
|---|---|---|---|
| 1 | Exploratory Data Analysis | `EnterpriseEDAEngine` — Shannon Entropy, Gini Impurity, 5 diagnostic charts, correlation matrices | ✅ Complete |
| 2 | Feature Engineering | `FeatureEngineeringEngine` — 7 categories of derived features (temporal, cyclic, interaction, text-statistical, frequency encodings) with 22-dimension governance registry | ✅ Complete |
| 3 | Assignment Group Prediction | `CatBoostClassifier` — Gradient boosting with native categorical handling, wrapped in scikit-learn pipeline | ✅ Complete |
| 4 | Resolution Recommendation Engine | `HybridRecommendationEngine` — Fuses CatBoost ML predictions with FAISS semantic search, confidence scoring, and natural language reasoning | ✅ Complete |
| 5 | Resolution Time Prediction | `CatBoostRegressor` — Log-transformed MTTR prediction with inverse `expm1` restoration | ✅ Complete |
| 6 | Hyperparameter Tuning | `RandomizedSearchCV` — 5-fold CV, 30 iterations across depth/learning_rate/iterations/L2 regularization | ✅ Complete |
| 7 | Explainable AI | `SHAPIntelligenceExplainer` — Global beeswarm + bar charts, local waterfall + decision plots via TreeExplainer | ✅ Complete |
| 8 | Generative AI Resolution Assistant | FAISS Semantic Similarity Engine automatically surfaces Top-5 historically resolved precedents with resolution notes | ✅ Complete (Non-LLM approach) |
| 9 | Application/Dashboard | `Streamlit` real-time web dashboard with instant predictions, AI reasoning, and historical evidence table | ✅ Complete |

---

# SLIDE 4: TECHNOLOGY STACK

## Libraries & Frameworks

### Machine Learning & AI
| Technology | Role in Our Platform |
|---|---|
| **CatBoost** | Core prediction engine — Gradient Boosting optimized for categorical corporate data (assignment groups, priorities, categories) |
| **scikit-learn** | Pipeline construction (`Pipeline`, `ColumnTransformer`), preprocessing (`OneHotEncoder`, `SimpleImputer`, `TfidfVectorizer`), HPO (`RandomizedSearchCV`), evaluation metrics |
| **FAISS (Meta)** | Vector similarity search engine — Stores all historical tickets as 384-D mathematical vectors and finds the Top-K most similar ones in milliseconds |
| **SHAP** | Explainable AI — Game-theoretic feature attribution proving *why* the AI made each decision |

### NLP (Natural Language Processing)
| Technology | Role |
|---|---|
| **TF-IDF** (`TfidfVectorizer`) | Converts raw ticket text into weighted term-frequency vectors |
| **Truncated SVD** | Reduces sparse TF-IDF matrix into dense 384-dimensional vectors (Latent Semantic Analysis) |
| **L2 Normalization** | Unit-length vectors enabling cosine similarity via inner product |
| **Custom IT Lemmatizer** | Rule-based lemmatization preserving 40+ protected IT keywords (`server`, `timeout`, `firewall`, `vpn`, `dns`) |

### Data & Visualization
| Technology | Role |
|---|---|
| **Pandas** | DataFrame operations, CSV I/O, datetime handling |
| **NumPy** | Numerical arrays, trigonometry for cyclic features, log transforms |
| **Matplotlib + Seaborn** | Confusion matrices, ROC curves, feature importance charts, EDA diagnostic plots |

### Infrastructure
| Technology | Role |
|---|---|
| **Streamlit** | Live web dashboard for real-time predictions |
| **PyYAML** | Configuration management (4 YAML config files) |
| **Joblib** | Serialization of trained scikit-learn pipelines to `.pkl` files |
| **Python 3.11+** | Core language with type hints throughout |

---

# SLIDE 5: PIPELINE OVERVIEW — THE BIG PICTURE

## 12-Stage End-to-End Pipeline

Our platform executes a **12-stage sequential pipeline** divided into 4 phases:

```
╔══════════════════════════════════════════════════════════════════╗
║                 PHASE 1: DATA INTELLIGENCE                      ║
║  Stage 1: Data Validation (12 quality rules)                    ║
║  Stage 2: ML Readiness Assessment                               ║
║  Stage 3: Data Cleaning (8-step remediation)                    ║
║  Stage 4: External Data Enrichment                              ║
║  Stage 5: Feature Engineering (7 feature categories)            ║
║  Stage 6: NLP Text Preprocessing (6-step normalization)         ║
║  Stage 7: Exploratory Data Analysis (EDA)                       ║
║  Stage 8: Train/Validation/Test Splitting                       ║
╠══════════════════════════════════════════════════════════════════╣
║                 PHASE 2: MODEL TRAINING                         ║
║  Stage 9:  CatBoost Classifier (Assignment Group)               ║
║  Stage 10: CatBoost Regressor (Resolution Time / MTTR)          ║
╠══════════════════════════════════════════════════════════════════╣
║                 PHASE 3: SEMANTIC INDEX                         ║
║  Stage 11: TF-IDF + SVD Embeddings → FAISS Vector Index         ║
╠══════════════════════════════════════════════════════════════════╣
║                 PHASE 4: HYBRID INFERENCE                       ║
║  Stage 12: Hybrid Recommendation (CatBoost + FAISS Fusion)      ║
╚══════════════════════════════════════════════════════════════════╝
```

**Single Command Execution:**
```bash
python main.py full-pipeline
```

---

# SLIDE 6: PIPELINE DEEP-DIVE — STAGE 1: DATA VALIDATION

## Objective 1 (Partial): Understanding Data Quality

**Script:** `src/data/validation.py` → `DatasetValidator` (452 lines)
**Trigger:** `python main.py validate`

Before any AI can learn, the raw data must be audited. Our system runs **12 automated quality rules:**

| Rule ID | What It Checks | Why It Matters |
|---|---|---|
| CHK-01 | Missing values in `number`, `priority`, `category`, `assignment_group` | Missing target = unusable training record |
| CHK-02 | Duplicate ticket numbers | Duplicates inflate model confidence artificially |
| CHK-03 | Timestamp integrity: `opened_at ≤ resolved_at ≤ closed_at` | Inverted timestamps create negative resolution times |
| CHK-04 | Non-null, non-blank categories | Category is a primary predictor |
| CHK-05 | Non-null assignment groups | Assignment group is the classification target |
| CHK-06 | Priority in valid range [1-5] | Out-of-range values corrupt business rule logic |
| CHK-07 | SLA consistency: `made_sla` flag vs actual resolution hours | Contradictory SLA data misleads the model |
| CHK-08 | Resolution time ≥ 0 and matches `(resolved_at - opened_at)` | Negative times are physically impossible |
| CHK-09 | Valid CMDB Configuration Items | CMDB linkage is a valuable predictor |
| CHK-10 | Valid Business Service references | Service context improves routing accuracy |
| CHK-11 | Non-empty description text | Description is the #1 input to the NLP engine |
| CHK-12 | Non-empty short description | Short description is the primary text feature |

**Output:** `reports/validation_report.json` + `reports/validation_report.md`

---

# SLIDE 7: PIPELINE DEEP-DIVE — STAGE 2: ML READINESS ASSESSMENT

## Pre-Training Diagnostic

**Script:** `src/data/readiness.py` → `MLReadinessEvaluator` (375 lines)
**Trigger:** `python main.py readiness`

Before training, the system performs a comprehensive ML feasibility audit:

| Check | What It Computes | Formula |
|---|---|---|
| **Target Leakage Detection** | Identifies post-resolution columns (`close_notes`, `resolved_at`, `made_sla`) that would "leak" the answer to the model | Blocklist matching against `POST_RESOLUTION_FIELDS` |
| **Class Imbalance** | Shannon Entropy of target distribution | $H = -\sum p_i \log_2 p_i$ |
| **Gini Impurity** | Measures class distribution purity | $G = 1 - \sum p_i^2$ |
| **Text Token Capacity** | Checks if descriptions exceed 256-token budget | $\text{tokens} \approx \lceil \text{words} \times 1.3 \rceil$ |
| **High Cardinality** | Identifies categorical columns with too many unique values | `nunique / len(df)` ratio |
| **Correlation Matrix** | Pearson correlation across numerical fields | Flags highly correlated feature pairs |

**Output:** Actionable preprocessing recommendations (e.g., "Drop `close_notes` — blocked leakage", "Apply class reweighting for imbalanced `assignment_group`")

---

# SLIDE 8: PIPELINE DEEP-DIVE — STAGE 3: DATA CLEANING

## 8-Step Automated Remediation

**Script:** `src/preprocessing/cleaner.py` → `EnterpriseDataCleaner` (454 lines)
**Trigger:** `python main.py clean`

| Step | Operation | Technical Detail |
|---|---|---|
| 1 | **Duplicate Removal** | Sorts by `opened_at`, keeps latest record per ticket `number` |
| 2 | **Business Rule Standardization** | Parses mixed text/numeric priorities (`"4 - Low"`, `"Critical"`, `"Sev 1"`) into integer tiers. Clips `priority` to [1,5], `urgency`/`impact` to [1,3] |
| 3 | **Schema & Type Enforcement** | Coerces columns to FeatureRegistry-defined types: `datetime`, `integer`, `float`, `boolean` via `pd.to_datetime`, `pd.to_numeric` |
| 4 | **Missing Value Imputation** | Text → `"Not Provided"`, Numeric → median/zero, Categorical → mode/`"Unknown"` |
| 5 | **Category Validation** | Maps null/whitespace-only categories to `"Unknown"` |
| 6 | **Timestamp Correction** | If `resolved_at < opened_at`: corrects to `opened_at + 4 hours`. If `closed_at < resolved_at`: corrects to `resolved_at + 24 hours` |
| 7 | **Outlier Winsorization** | Caps `reassignment_count` at 99th percentile or max 15. Caps `reopen_count` at 99th percentile or max 8 |
| 8 | **String Normalization** | Trims leading/trailing whitespace across all string columns |

**Governance:** Every transformation is logged into a JSON audit trail (`reports/cleaning_report.json`) with exact counts of modified records.

---

# SLIDE 9: PIPELINE DEEP-DIVE — STAGE 5: FEATURE ENGINEERING

## Objective 2: Feature Engineering — 7 Categories of Derived Features

**Script:** `src/preprocessing/engineer.py` → `FeatureEngineeringEngine` (470 lines)
**Trigger:** `python main.py engineer`

### Temporal Features
| Feature | Formula | Why |
|---|---|---|
| `opened_at_hour` | Extracted from `opened_at` datetime (0-23) | Tickets filed at 2 AM behave differently than 2 PM |
| `opened_at_dayofweek` | 0=Monday, 6=Sunday | Weekend incidents have different routing patterns |
| `is_weekend` | 1 if Saturday/Sunday, else 0 | Binary weekend indicator |
| `is_business_hours` | 1 if Mon-Fri 08:00-18:00, else 0 | After-hours incidents may need on-call teams |
| `is_holiday` | 1 if US Federal/Banking holiday, else 0 | Holiday staffing affects resolution times |

### Cyclic Temporal Encoding
Raw hours (0-23) have a discontinuity problem: hour 23 appears far from hour 0, but they are actually adjacent. We solve this with trigonometric encoding:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$

This maps the circular hour onto a continuous circle where hour 23 and hour 0 are mathematically adjacent.

### Interaction Features
| Feature | Formula | Why |
|---|---|---|
| `priority_x_business_impact` | `priority × business_impact` | Non-linear severity signal (Priority 1 + High Impact = Critical) |
| `category_assignment_interaction` | `category + "_" + assignment_group` | Captures routing patterns specific to category-group pairs |

### Text Statistics
| Feature | Formula |
|---|---|
| `short_description_word_count` | `len(text.split())` |
| `short_description_char_count` | `len(text)` |
| `description_word_count` | `len(text.split())` |
| `description_char_count` | `len(text)` |

### Historical Frequency Encodings
| Feature | Formula | Why |
|---|---|---|
| `assignment_group_freq` | `value_counts(normalize=True)` | Encodes how common each group is (rare groups behave differently) |
| `business_service_freq` | `value_counts(normalize=True)` | Service popularity as a signal |
| `caller_freq` | `value_counts(normalize=True)` | Frequent callers may have patterns |

### Governance
Every engineered feature is automatically registered in the **Feature Registry** (22 governance dimensions per feature) and its derivation formula is logged in the **Feature Lineage Tracker** (parent→child graph).

---

# SLIDE 10: PIPELINE DEEP-DIVE — STAGE 6: NLP TEXT PREPROCESSING

## Preparing Text for the AI Brain

**Script:** `src/preprocessing/text_preprocessor.py` → `TextPreprocessor` (273 lines)

Incident descriptions contain messy corporate text: email headers, HTML tags, error boilerplate, and conversational filler words. We clean this with a **6-step NLP pipeline:**

| Step | Operation | Example |
|---|---|---|
| 1 | **Unicode NFKC Normalization** + lowercasing | `"SERVER—DOWN"` → `"server-down"` |
| 2 | **Strip HTML/XML tags** and email headers | `"<b>Error</b> From: admin@..."` → `"Error"` |
| 3 | **Replace error boilerplate** | `"[system error code: 0x80004005]"` → `"system_error"` |
| 4 | **Remove punctuation** (preserving IT symbols `-`, `_`, `/`, `.`) | `"Error!!! Check now!!!"` → `"Error Check now"` |
| 5 | **Filter stopwords** while **protecting 40+ IT keywords** | Removes: `"the"`, `"is"`, `"and"`. Preserves: `"server"`, `"down"`, `"error"`, `"timeout"`, `"firewall"`, `"vpn"`, `"dns"` |
| 6 | **IT Domain Lemmatization** | `"failures"` → `"failure"`, `"servers"` → `"server"`, `"crashes"` → `"crash"` |

**Token Budget:** Estimates BPE tokens as $\lceil \text{words} \times 1.3 \rceil$ and flags sequences exceeding 256 tokens.

---

# SLIDE 11: PIPELINE DEEP-DIVE — STAGE 7: EXPLORATORY DATA ANALYSIS

## Objective 1: EDA — Automated Statistical Intelligence

**Script:** `src/preprocessing/eda.py` → `EnterpriseEDAEngine` (597 lines)
**Trigger:** `python main.py eda`

| Analysis | Metrics Computed |
|---|---|
| **Numerical Profiling** | Mean, std, skewness, kurtosis, IQR, outlier counts & percentages |
| **Categorical Intelligence** | Cardinality, Top-10 frequencies, Shannon Entropy ($H$), Gini Impurity ($G$) |
| **Text Analysis** | Character/word/token length distributions, 256-token budget violations |
| **Temporal Patterns** | 24-hour hourly arrival distribution, 7-day weekday distribution |
| **Correlation Analysis** | Pearson ($r$) linear + Spearman ($\rho$) rank correlation matrices |
| **Target Leakage Audit** | Classifies every column as `safe`, `warning`, or `blocked` |

### 5 Automated Diagnostic Charts Generated:
1. **Category Distribution** — Top 10 incident categories barplot
2. **Priority vs SLA** — Stacked bar of SLA compliance by priority tier
3. **Hourly Arrival** — 24-hour incident frequency line plot
4. **Correlation Heatmap** — Numerical feature correlation matrix
5. **Text Word Counts** — Distribution histogram with KDE

**Output:** `reports/eda_report.json` + `.md` + `.html` + 5 PNG charts in `reports/figures/`

---

# SLIDE 12: PIPELINE DEEP-DIVE — STAGE 8: ZERO-LEAKAGE SPLITTING

## Mathematically Safe Data Partitioning

**Script:** `src/preprocessing/splitter.py` → `DatasetSplitter` (287 lines)
**Trigger:** `python main.py split`

### Why Zero-Leakage Matters
If even a single test record accidentally appears in the training set, the model memorizes the answer instead of learning patterns. This creates artificially inflated accuracy that collapses in production.

### 3 Splitting Strategies
| Strategy | Method | When to Use |
|---|---|---|
| **Stratified** (Default) | `sklearn.train_test_split(stratify=y)` preserving class ratios | When class imbalance exists |
| **Time-Based** | Chronological sort by `opened_at`, sequential slice | When temporal drift matters |
| **Random** | Standard two-stage random split | Baseline comparison |

### Automatic Rare Class Handling
Classes with fewer than 3 samples are automatically grouped into `"Rare / Other"` to prevent stratification failures.

### Zero-Leakage Verification
After splitting, the system performs **set intersection checks:**
- `Train ∩ Validation = ∅`
- `Train ∩ Test = ∅`
- `Validation ∩ Test = ∅`

If ANY overlap is detected, the system raises a `ValueError` and refuses to proceed.

---

# SLIDE 13: PIPELINE DEEP-DIVE — STAGE 9: ASSIGNMENT GROUP PREDICTION

## Objective 3: CatBoost Classification

**Script:** `src/ml/catboost/trainer.py` → `EnterpriseCatBoostTrainer` (536 lines)
**Trigger:** `python main.py train --target assignment_group`

### Why CatBoost?
| Advantage | Explanation |
|---|---|
| **Native Categorical Handling** | Unlike XGBoost/LightGBM, CatBoost natively encodes categorical features without manual one-hot encoding |
| **Ordered Boosting** | Prevents target leakage during tree construction |
| **Symmetric Trees** | Faster inference (important for sub-second dashboard predictions) |
| **Built-in L2 Regularization** | Prevents overfitting on small (6,000 row) datasets |

### Complete Scikit-Learn Pipeline Architecture
```
Pipeline:
  Step 1: EnterpriseFeatureExtractor
    ├── Generates combined_text (short_description + description)
    ├── Computes priority_x_business_impact interaction
    └── Generates cyclic sin/cos temporal features

  Step 2: ColumnTransformer
    ├── Branch A: FrequencyEncoder → high-cardinality categoricals
    ├── Branch B: OneHotEncoder → low-cardinality categoricals
    ├── Branch C: SimpleImputer(median) → numerical features
    └── Branch D: TfidfVectorizer → combined text passthrough

  Step 3: CatBoostClassifier
    └── Predicts assignment_group with class probabilities
```

### Zero-Leakage Enforcement
Before training, the system cross-references every predictor column against the Feature Registry. Any column classified as `blocked` (e.g., `close_notes`, `resolved_at`, `close_code`) is rejected with a `ValueError`.

**Output:** `models/catboost_assignment_group.pkl` — Serialized complete pipeline

---

# SLIDE 14: PIPELINE DEEP-DIVE — STAGE 10: RESOLUTION TIME PREDICTION

## Objective 5: CatBoost Regression (MTTR)

**Script:** `src/ml/catboost/trainer.py` → `EnterpriseCatBoostTrainer` (536 lines)
**Trigger:** `python main.py train --target resolution_time_hours`

### The Log-Transform Trick
Resolution times are heavily right-skewed (most tickets resolve in hours, but some take weeks). Training directly on raw hours produces poor predictions. We apply:

**During Training:**
$$y_{\text{train}} = \log(1 + y_{\text{raw}})$$

**During Inference:**
$$\hat{y}_{\text{predicted}} = e^{\hat{y}_{\text{model}}} - 1$$

This `log1p` / `expm1` transform normalizes the target distribution, dramatically improving regression accuracy.

### Additional Safety
The `assignment_group` column is excluded from regressor predictors to prevent forward leakage (knowing the team shouldn't help predict time).

**Output:** `models/catboost_resolution_time_hours.pkl` — Serialized regression pipeline

---

# SLIDE 15: PIPELINE DEEP-DIVE — STAGE 6 (OBJECTIVE 6): HYPERPARAMETER TUNING

## Objective 6: Automated Hyperparameter Optimization

**Script:** `src/ml/catboost/trainer.py` — Uses `sklearn.model_selection.RandomizedSearchCV`

### Configuration (`config/model_config.yaml`)
| Parameter | Search Space | Purpose |
|---|---|---|
| `iterations` | [100, 300, 500, 800] | Number of boosting rounds |
| `depth` | [4, 5, 6, 7] | Maximum tree depth (controls complexity) |
| `learning_rate` | [0.03, 0.05, 0.1] | Step size per boosting round |
| `l2_leaf_reg` | [3, 5, 7, 9] | L2 regularization (prevents overfitting) |

### HPO Settings
| Setting | Value | Reason |
|---|---|---|
| **Method** | `RandomizedSearchCV` | Explores more of the space than GridSearch in less time |
| **CV Folds** | 5 | 5-fold cross-validation for robust evaluation |
| **Iterations** | 30 | Tests 30 random hyperparameter combinations |
| **Scoring** | `f1_weighted` (classifier), `neg_MAE` (regressor) | Handles class imbalance |
| **n_jobs** | 1 | Avoids thread contention with CatBoost's internal multithreading |

### Why `n_jobs=1`?
CatBoost uses **all CPU cores internally** for each model. Setting `n_jobs > 1` would spawn multiple CatBoost instances fighting for the same cores, causing severe bottlenecking. `n_jobs=1` queues models sequentially, giving each one 100% CPU power.

---

# SLIDE 16: PIPELINE DEEP-DIVE — STAGE 11: SEMANTIC SEARCH ENGINE

## Objective 4 & 8: Resolution Recommendation + Knowledge Retrieval

### Step 1: Text → Vector Conversion
**Script:** `src/ml/semantic/embedding_generator.py` → `SemanticEmbeddingGenerator` (266 lines)

Each incident record is converted into a **384-dimensional dense mathematical vector:**

```
Input:  "Database connection timeout on payment gateway server DB-01"
          ↓
Step 1: Construct semantic string
        "[Category: Database] [Service: Payment Gateway | CI: DB-01] 
         [Priority: 2] Database connection timeout on payment gateway server DB-01"
          ↓
Step 2: TF-IDF Vectorization (sparse matrix)
          ↓
Step 3: TruncatedSVD → 384 dimensions (dense vector)
          ↓
Step 4: L2 Normalization (unit length)
          ↓
Output: [0.041, -0.088, 0.012, 0.094, ..., 0.031]  (384 numbers)
```

### Step 2: FAISS Vector Index
**Script:** `src/ml/semantic/faiss_index.py` → `FAISSVectorIndex` (269 lines)

All 6,000 historical ticket vectors are loaded into a **FAISS IndexFlatIP** (Exact Inner Product Search). For L2-normalized vectors:

$$\text{cosine\_similarity}(a, b) = a \cdot b = \text{inner\_product}(a, b)$$

When a new ticket arrives, the system computes its vector and finds the Top-K nearest neighbors by scanning every stored vector. This guarantees **100% recall** (mathematically impossible to miss a match).

### Step 3: Similarity Engine
**Script:** `src/ml/semantic/similarity_engine.py` → `SemanticSimilarityEngine` (260 lines)

Orchestrates embedding generation + FAISS search, returns structured results with similarity scores, routing consensus, and historical resolution notes.

---

# SLIDE 17: PIPELINE DEEP-DIVE — STAGE 12: HYBRID INTELLIGENCE ENGINE

## The Crown Jewel — Fusing ML + Historical Evidence

**Scripts:** `src/ml/hybrid/` — 4 specialized sub-engines (780 lines total)

### 6-Step Hybrid Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  Step 1: INPUT PARSING                                       │
│  Parse JSON/dict/free-text → normalized ticket dictionary    │
│                        ↓                                     │
│  Step 2: CATBOOST PREDICTION                                 │
│  Classifier → Predicted Group + Probabilities                │
│  Regressor  → Predicted MTTR (hours)                         │
│                        ↓                                     │
│  Step 3: FAISS SEMANTIC SEARCH                               │
│  Embed query → Search Top-5 similar historical tickets       │
│  Extract consensus group + median historical MTTR            │
│                        ↓                                     │
│  Step 4: DECISION FUSION                                     │
│  Compare CatBoost prediction vs FAISS consensus              │
│  ┌─ If AGREED: Use CatBoost, apply +0.1 bonus               │
│  ├─ If CatBoost dominant (conf > 0.7): Use CatBoost, -0.05  │
│  └─ If FAISS dominant: Override with historical consensus    │
│                        ↓                                     │
│  Step 5: CONFIDENCE SCORING                                  │
│  Fused Score = 0.6×CatBoost_conf + 0.4×FAISS_conf ± bonus   │
│  Blended MTTR = 0.5×CatBoost_MTTR + 0.5×FAISS_median_MTTR  │
│                        ↓                                     │
│  Step 6: REASONING GENERATION                                │
│  Natural language explanation (zero LLMs — deterministic)    │
│  Executive summary + bullet breakdown + evidence table       │
└──────────────────────────────────────────────────────────────┘
```

### Confidence Tiers
| Fused Score | Tier | Action |
|---|---|---|
| ≥ 0.88 | **Very High** | Auto-route without human review |
| ≥ 0.75 | **High** | Auto-route with notification |
| ≥ 0.60 | **Moderate** | Route with manual review flag |
| ≥ 0.45 | **Low** | Suggest but require human confirmation |
| < 0.45 | **Review Required** | Escalate to senior agent |

---

# SLIDE 18: PIPELINE DEEP-DIVE — EXPLAINABLE AI

## Objective 7: SHAP — Why Did the AI Decide This?

**Script:** `src/ml/explainability/shap_explainer.py` → `SHAPIntelligenceExplainer` (357 lines)

### What is SHAP?
SHAP (SHapley Additive exPlanations) uses **cooperative game theory** to compute each feature's contribution to a prediction:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \; (|N|-|S|-1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

In simple terms: For each prediction, SHAP tells us *"the Short Description contributed +15% toward the Database Team, while Priority contributed -3%."*

### Two Levels of Explanation

| Level | What It Shows | Chart Generated |
|---|---|---|
| **Global** | Which features matter most across ALL predictions | `shap_summary.png` (beeswarm), `shap_bar.png` (importance) |
| **Local** | Why THIS specific ticket was routed to THIS specific team | `shap_waterfall_sample.png`, `shap_decision_sample.png` |

### Business Name Translation
Raw feature names like `freq__assignment_group` or `onehot__category_Hardware` are automatically translated into human-readable names like `Assignment Group (Frequency)` or `Category: Hardware` using the Feature Registry's `resolve_business_name()` method.

---

# SLIDE 19: PIPELINE DEEP-DIVE — THE DASHBOARD

## Objective 9: Application/Dashboard

**Script:** `src/dashboard/app.py` (118 lines)
**Launch:** `python run_dashboard.py`

### User Interface Layout
| Column 1 (Core Fields) | Column 2 (Severity & Custom) |
|---|---|
| Short Description (text input) | Priority (selectbox: 1-4) |
| Full Description (text area) | Business Impact (selectbox: 1-3) |
| Category (selectbox: 8 categories) | Severity (selectbox: 1-3) |
| Subcategory (text input) | `u_caused_by` (custom corporate field) |
| CMDB CI (text input) | `u_vendor_ticket_ref` (custom field) |

### On "Predict" Click
1. Formats input into ticket dictionary
2. Calls `HybridRecommendationEngine.recommend(ticket_dict, top_k=5)`
3. Displays **3 metric cards:** Predicted Team, Confidence Score/Tier, Estimated MTTR
4. Shows **AI Reasoning** executive summary
5. Renders **Top-5 Historical Precedents** table with similarity scores

### Performance
- Models loaded once via `@st.cache_resource` (cached singleton)
- Sub-second inference latency
- No data leaves the corporate network

---

# SLIDE 20: FUTURE IMPROVEMENTS — HNSW SCALABILITY

## Scaling from 6,000 to 10,000,000+ Tickets

### Current Architecture: Exact Search (O(N))
Our FAISS engine currently uses `IndexFlatIP` — it checks **every single vector** in the database for every query. At 6,000 tickets, this takes milliseconds. At 10 million tickets, this becomes a bottleneck.

### Future Architecture: HNSW (O(log N))
**HNSW (Hierarchical Navigable Small World)** builds a multi-layered graph pyramid:

| Layer | Density | Purpose |
|---|---|---|
| **Layer 2** (Top) | Very sparse (few nodes) | Global "highway" — fast entry point |
| **Layer 1** (Middle) | Medium density | Regional navigation |
| **Layer 0** (Bottom) | 100% of all tickets | Dense local search for exact match |

### How HNSW Searches
1. Enter at the **sparse top layer** — quickly hop to the nearest node
2. Drop down to the **middle layer** — refine the search region
3. Drop to the **dense bottom layer** — find the exact nearest neighbor

Instead of checking 10,000,000 vectors ($O(N)$), the AI performs ~30 hops through the graph ($O(\log N)$) to find the same answer.

### FAISS Configuration Change
```yaml
# Current (6,000 tickets):
faiss:
  index_type: Flat        # Exact search, 100% recall

# Future (10,000,000+ tickets):
faiss:
  index_type: HNSW        # Approximate search, ~99% recall
  M: 32                   # Connections per node
  efConstruction: 200     # Build-time accuracy
  efSearch: 128           # Query-time accuracy
```

*(Insert `hnsw_presentation.png` diagram on this slide)*

---

# SLIDE 21: MARKET STUDY

## The IT Service Management AI Market

### Market Size & Growth
| Metric | Value |
|---|---|
| **Global ITSM Market (2024)** | $12.2 Billion |
| **Projected (2030)** | $28.1 Billion |
| **CAGR** | 14.9% |
| **AI in ITSM Segment** | Fastest growing sub-segment |

*Source: Grand View Research, MarketsandMarkets, Gartner*

### Key Industry Trends
| Trend | Our Platform's Alignment |
|---|---|
| **AIOps Adoption** | Our dual-engine architecture (ML + Semantic) is a core AIOps pattern |
| **Shift-Left Strategy** | Auto-routing tickets to the right team eliminates L1 → L2 escalation delays |
| **Explainable AI Mandates** | Financial institutions require AI decisions to be auditable — our SHAP engine provides this |
| **On-Premise Security** | Banking regulations prohibit cloud AI for sensitive data — our platform runs 100% locally |
| **Knowledge Management** | Industry moving from static knowledge bases to AI-powered semantic search — exactly what FAISS provides |

### Competitive Landscape
| Solution | Approach | Our Advantage |
|---|---|---|
| **ServiceNow Predictive Intelligence** | Cloud-based ML | We run on-premise, no data egress |
| **BMC Helix AIOps** | Proprietary platform | We use open-source (CatBoost + FAISS), fully customizable |
| **Freshservice Freddy AI** | SaaS-only | We deploy inside corporate firewalls |
| **IBM Watson AIOps** | Heavy enterprise suite | We are lightweight, single-command deployment |

### Our Unique Differentiators
1. **Hybrid Intelligence:** No competitor fuses gradient boosting + vector similarity + confidence scoring + natural language reasoning in a single pipeline
2. **Zero-Leakage Governance:** 22-dimension feature registry with automatic target leakage blocking
3. **Explainability Built-In:** SHAP attribution is a first-class citizen, not an afterthought
4. **Single-Command Deployment:** `python main.py full-pipeline` — no complex infrastructure required

---

# SLIDE 22: DATA GOVERNANCE — THE BACKBONE

## What Makes This Enterprise-Grade

### Feature Registry — 22 Governance Dimensions Per Feature
**Script:** `src/data/feature_registry.py` (670 lines) — The largest single file in the project.

Every feature (49 total: 38 raw + 11 engineered) is tracked across:

| Dimension | Example Value |
|---|---|
| `business_name` | "Incident Priority Level" |
| `technical_name` | "priority" |
| `data_type` | "integer" |
| `target_leakage_classification` | "safe" / "warning" / "blocked" |
| `encoding_strategy` | "ordinal" |
| `imputation_strategy` | "mode" |
| `catboost_usage` | "predictor" |
| `embedding_usage` | "excluded" |
| `faiss_metadata_usage` | "filter_field" |
| `dashboard_usage` | "kpi_filter" |
| `explainability_usage` | "shap_feature" |

### Feature Lineage — Derivation Graph
**Script:** `src/data/feature_lineage.py` (273 lines)

Tracks exactly how every engineered feature was mathematically derived:
```
opened_at → opened_at_hour → opened_at_hour_sin
                              [formula: sin(2π × hour / 24)]
```

### Model Registry — Cryptographic Integrity
**Script:** `src/ml/model_registry.py` (248 lines)

Every trained model is registered with:
- SHA256 cryptographic checksum (detects corruption)
- Feature compliance verification (rejects models using blocked features)
- Version tracking (`:latest`, `:v1`, `:v2`)

---

# SLIDE 23: TESTING & QUALITY ASSURANCE

## 23 Unit Test Files Covering Every Module

| Test File | Module Tested |
|---|---|
| `test_catboost_trainer.py` | Training pipeline construction |
| `test_cleaner.py` | 8-step cleaning logic |
| `test_hybrid_engine.py` | Decision fusion & confidence scoring |
| `test_semantic_similarity.py` | FAISS search accuracy |
| `test_shap_explainer.py` | SHAP attribution |
| `test_feature_registry.py` | Governance rules |
| ... (17 more) | Full coverage across all modules |

**Coverage Threshold:** 80% minimum (enforced in `pyproject.toml`)
**Framework:** pytest with singleton reset fixtures for test isolation

---

# SLIDE 24: CONCLUSION

## What We Built

| Objective | Delivered |
|---|---|
| ✅ EDA | Automated analysis with Shannon Entropy, Gini, 5 diagnostic charts |
| ✅ Feature Engineering | 7 categories of derived features with 22-dimension governance |
| ✅ Assignment Group Prediction | CatBoost Classifier with zero-leakage pipeline |
| ✅ Resolution Recommendation | Hybrid Engine fusing ML + FAISS + Confidence + Reasoning |
| ✅ Resolution Time Prediction | CatBoost Regressor with log1p/expm1 transform |
| ✅ Hyperparameter Tuning | RandomizedSearchCV (5-fold CV, 30 iterations) |
| ✅ Explainable AI | SHAP global + local attribution with business name translation |
| ✅ Knowledge Retrieval | FAISS semantic search surfacing Top-5 historical precedents |
| ✅ Dashboard | Streamlit real-time web UI with sub-second inference |

## Key Metrics
| Metric | Value |
|---|---|
| **Total Production Code** | ~7,500 lines across 33 Python files |
| **Total Test Code** | ~4,000+ lines across 23 test files |
| **Pipeline Stages** | 12 sequential stages |
| **Features Governed** | 49 features × 22 dimensions |
| **Inference Latency** | < 1 second |
| **Deployment** | Single command: `python main.py full-pipeline` |

---

# SLIDE 25: LIVE DEMONSTRATION

## Let the AI Prove Itself

1. Open the Streamlit dashboard (`python run_dashboard.py`)
2. Enter a sample ticket: *"Database connection timeout on payment gateway server DB-01. Users unable to process transactions."*
3. Click **"Predict Routing & MTTR 🚀"**
4. Show:
   - ✅ Predicted Assignment Group
   - ✅ Confidence Score & Tier
   - ✅ Estimated MTTR (hours)
   - ✅ AI Reasoning Justification
   - ✅ Top-5 Historical Precedents

**Invite the audience to suggest their own ticket descriptions!**

---

# SLIDE 26: Q&A

## Questions & Technical Deep-Dive

Open the floor for questions on:
- Architecture decisions
- Algorithm choices
- Deployment considerations
- Scalability roadmap
- Integration with existing ITSM workflows

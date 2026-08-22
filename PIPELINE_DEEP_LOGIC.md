# Incident Intelligence Platform: Deep Logic Flow
## A Complete End-to-End Code & Logic Breakdown

This document explains the deep logic of the entire pipeline, connecting the 30 core mandatory files. It translates the mathematical and architectural decisions into simple words, while highlighting the exact code sections responsible for the magic.

---

## Phase 1: The Foundation & Governance

Before any data is processed, the platform establishes its rules and environment.

### 1. `src/utils/config_manager.py` (The Settings Loader)
**Logic:** Instead of hardcoding paths and settings, the platform reads `config/config.yaml` and `model_config.yaml`. The `ConfigManager` acts as a Singleton (only one instance ever exists) and automatically resolves environment variables.
**Key Code:**
```python
def _resolve_env_vars(self, config: Any) -> Any:
    # Turns "${DATA_INPUT_PATH:data/raw/incidents.csv}" into the actual value.
    # If the env var doesn't exist, it uses the fallback default.
```

### 2. `src/utils/__init__.py` (Bulletproof File Reader)
**Logic:** Corporate systems often save CSVs in weird encodings (like `cp1252` on Windows), which crash normal pandas. This utility forcefully tries multiple encodings until one works.
**Key Code:**
```python
def robust_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
```

### 3. `src/data/feature_registry.py` (The Supreme Data Law)
**Logic:** This is the most critical file for preventing "Target Leakage". Target leakage is when a model cheats by looking at data it wouldn't have in real life (e.g., looking at `close_notes` to predict the routing team). The registry defines 22 dimensions for every feature.
**Key Code:**
```python
def get_catboost_predictors(self, target_type: str = "assignment_group") -> List[str]:
    # It only allows features explicitly marked as 'safe' (no leakage) 
    # AND designated as a 'predictor'.
    safe_feats = self.get_features_by_leakage("safe")
    results = [f.technical_name for f in safe_feats if f.catboost_usage == "predictor"]
    return sorted(results)
```

### 4. `src/data/feature_lineage.py` (The Audit Trail)
**Logic:** Whenever the pipeline creates a new feature (like extracting "Hour" from a timestamp), this script records the exact mathematical formula used. This is mandatory for corporate AI governance.

---

## Phase 2: Data Quality & Readiness

### 5. `src/data/readiness.py` & `quality_gate.py` (The ML Diagnostics)
**Logic:** Before training, the platform analyzes if the data is even good enough for ML. It checks for **Class Imbalance** using Shannon Entropy.
**Key Code (Shannon Entropy):**
```python
# High Entropy = Classes are balanced (Good for ML)
# Low Entropy = One class dominates, model will be biased (Bad)
counts = df[col].value_counts()
probs = counts / len(df)
entropy = -sum(p * math.log2(p) for p in probs if p > 0)
```

---

## Phase 3: The Preprocessing Factory

This is where dirty raw data is cleaned, enriched, and transformed.

### 6. `src/preprocessing/cleaner.py` (The Janitor)
**Logic:** Fixes missing values dynamically. Instead of guessing how to fill missing data, it asks the `FeatureRegistry` for the exact strategy.
**Key Code:**
```python
# Dynamically applies the correct imputation strategy per column
feat_def = self.registry.get_feature(col)
strategy = feat_def.imputation_strategy if feat_def else "constant_unknown"

if strategy == "median":
    df[col] = df[col].fillna(df[col].median())
elif strategy == "mode":
    df[col] = df[col].fillna(df[col].mode()[0])
```

### 7. `src/preprocessing/engineer.py` (The Feature Factory)
**Logic:** ML models struggle with cyclical time (they don't know that Hour 23 is next to Hour 0). This script uses Trigonometry (Sine/Cosine) to map hours onto a circle.
**Key Code:**
```python
# Hour 23 and Hour 0 are now adjacent on a mathematical circle
df["opened_at_hour_sin"] = np.sin(2 * np.pi * df["opened_at_hour"] / 24.0)
df["opened_at_hour_cos"] = np.cos(2 * np.pi * df["opened_at_hour"] / 24.0)
```

### 8. `src/preprocessing/text_preprocessor.py` (The NLP Editor)
**Logic:** Cleans ticket descriptions. It removes standard "stopwords" (like "the", "and") but explicitly protects IT keywords (like "server", "timeout").
**Key Code:**
```python
# Removes useless words but keeps important IT context
if remove_stopwords:
    words = [w for w in words if w not in self.stopwords or w in PROTECTED_IT_KEYWORDS]
# Normalizes IT terminology (e.g., "crashes" -> "crash")
if lemmatize:
    words = [IT_LEMMATIZATION_RULES.get(w, w) for w in words]
```

### 9. `src/preprocessing/splitter.py` (The Card Dealer)
**Logic:** Divides data into Train/Validation/Test (70/15/15). Crucially, it runs a **Zero Leakage Check** to mathematically prove that no single ticket exists in multiple splits.
**Key Code:**
```python
train_ids = set(train_df["number"].dropna())
val_ids = set(val_df["number"].dropna())
test_ids = set(test_df["number"].dropna())

# Intersections must be exactly zero
total_overlap = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
assert total_overlap == 0, "FAIL_LEAKAGE_DETECTED"
```

---

## Phase 4: Machine Learning Training

### 10. `src/ml/catboost/transformers.py` & `trainer.py` (The ML Brain)
**Logic:** The platform uses Scikit-Learn `Pipeline`s. This means it packages the exact data transformations (like Frequency Encoding) *inside* the model file. When you predict a new ticket, you just pass raw data—the pipeline handles the rest.
**Key Code (Frequency Encoder):**
```python
# Converts categorical strings into probabilities based on how often they occur
counts = series.value_counts(normalize=True).to_dict()
self.mapping_[col] = counts

def transform(self, X):
    # Maps 'Database Team' to 0.08, 'Unknown' to 0.0001
    df[col] = df[col].map(self.mapping_[col]).fillna(0.0001)
```

### 11. `src/ml/model_registry.py` (The Security Guard)
**Logic:** Saves the model as `.pkl` and calculates a SHA-256 cryptographic hash. If someone tampers with the model file on disk, the system will refuse to load it.
**Key Code:**
```python
current_sha256 = self.compute_sha256(fp)
if current_sha256 != meta.sha256_checksum:
    raise ModelValidationException("SHA256 Checksum Mismatch! Model file may be corrupted or tampered!")
```

---

## Phase 5: Semantic Search (Historical Precedents)

Instead of just predicting a class, the platform finds the most similar *past* tickets using Facebook AI Similarity Search (FAISS).

### 12. `src/ml/semantic/embedding_generator.py` (Text to Vectors)
**Logic:** Converts a ticket's text into a 384-dimensional mathematical vector using TF-IDF and Singular Value Decomposition (SVD), then normalizes it.
**Key Code:**
```python
self._pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=25000, stop_words="english")),
    ("svd", TruncatedSVD(n_components=384, random_state=42))
])
embeddings = self._pipeline.fit_transform(semantic_texts)
# Normalizes vectors so dot-product equals cosine similarity
embeddings = normalize(embeddings, norm='l2', axis=1) 
```

### 13. `src/ml/semantic/faiss_index.py` (The Search Engine)
**Logic:** Loads vectors into an exact inner-product search index. This allows searching through 10,000s of tickets in milliseconds.
**Key Code:**
```python
self.index = faiss.IndexFlatIP(self.dimension) # IP = Inner Product
self.index.add(embeddings.astype(np.float32))

# Search returns top_k similarity scores and row indices
scores, indices = self.index.search(query_vector, top_k)
```

---

## Phase 6: The Hybrid Decision Engine (The Final Boss)

This is the ultimate orchestration layer where ML Predictions meet Semantic Search.

### 14. `src/ml/hybrid/decision_engine.py` (The Judge)
**Logic:** The Random Forest (ML) predicts team X. FAISS (History) shows the last 5 similar tickets went to team Y. The decision engine acts as a judge and fuses them.
**Key Code:**
```python
agreement = (rf_group == mode_group) # Do ML and History agree?

if agreement:
    recommended_group = rf_group
    reason_code = "AGREEMENT"
else:
    # If they disagree, check if ML confidence is extremely high (> 70%)
    if rf_conf >= self.rf_dominant_thresh:
        recommended_group = rf_group
        reason_code = "RF_DOMINANT"
    else:
        # Otherwise, trust historical precedents
        recommended_group = mode_group
        reason_code = "SEMANTIC_DOMINANT"

# MTTR (Resolution Time) is a blended average of ML Regression and Historical Median
fused_mttr = (rf_mttr * 0.5) + (sem_mttr * 0.5)
```

### 15. `src/ml/hybrid/confidence_engine.py` (The Scorer)
**Logic:** Calculates the final confidence score mathematically.
**Key Code:**
```python
# Base blend: 60% ML Confidence + 40% Semantic Similarity Score
base_score = (rf_confidence * 0.6) + (sem_confidence * 0.4)

# Add bonus if engines agree, penalty if they disagree
fused_score = base_score + 0.1 if agreement else base_score - 0.05
```

### 16. `src/ml/hybrid/reasoning_engine.py` (The Explainer)
**Logic:** Translates the decision matrix into an English executive summary so stakeholders understand *why* the AI made its decision.
**Key Code:**
```python
if agreement:
    exec_summary = f"Both the ML model and historical precedent analysis agree: route to '{rec_group}'. Confidence: {tier} ({fused_conf:.2%})."
else:
    if reason_code == "RF_DOMINANT":
        exec_summary = f"ML model predicted '{rf_group}' with high confidence, superseding historical precedents favoring '{mode_group}'."
```

---

## Phase 7: The Dashboard Interface

### 17. `src/dashboard/app.py` (The Streamlit UI)
**Logic:** Exposes the Hybrid Engine to end-users via a web interface. It uses `@st.cache_resource` to load the massive ML and FAISS models into memory exactly *once*, making subsequent button clicks instant.
**Key Code:**
```python
@st.cache_resource
def get_engine():
    # Only runs once! Caches the heavy models in RAM.
    return HybridRecommendationEngine()

engine = get_engine()
if st.button("Predict Routing"):
    result = engine.recommend(ticket_dict)
    st.metric("Predicted Assignment Group", result["recommended_assignment_group"])
```

## Summary of Remaining Core Files

To ensure complete coverage of the 30 core mandatory files, here is the logic for the remaining orchestration, diagnostic, and metadata scripts:

### Data Governance Additions
* **`src/data/pipeline_contracts.py`:** Acts as the strict border patrol between raw data and preprocessing. It checks `df.columns` against the `FeatureRegistry` to ensure all required fields are present before cleaning begins.

### Preprocessing Additions
* **`src/preprocessing/enricher.py`:** Automatically checks if `cmdb.csv` or `shift_schedules.csv` exist in the raw folder. If they do, it performs a `pandas` left join (`how="left"`) to append external enterprise context to the tickets.
* **`src/preprocessing/eda.py`:** Calculates statistical distributions (like `value_counts(normalize=True)`) and generates matplotlib/seaborn charts. Crucially, it **does not modify** the dataframe; it acts strictly as an observer.

### ML Diagnostic Additions
* **`src/ml/catboost/evaluator.py`:** Tests the model against the 15% holdout set. Computes strict classification metrics (F1 Macro, Precision Weighted) and pulls the `.get_feature_importances()` from the CatBoost pipeline.
* **`src/ml/explainability/shap_explainer.py`:** Uses the mathematically proven SHAP (SHapley Additive exPlanations) library (`shap.TreeExplainer`). It calculates exactly how much each feature contributed to a specific ticket's prediction (e.g., "The word 'server' added +15% to Network Team").

### Semantic Metadata & Orchestration
* **`src/ml/semantic/embedding_registry.py`:** Just like the model registry, this saves metadata about the FAISS index (dimension size, number of vectors, index type) to ensure the search engine stays synchronized with the embedded data.
* **`src/ml/semantic/similarity_engine.py`:** The wrapper that connects `embedding_generator` (creating vectors) with `faiss_index` (searching them), formatting the raw mathematical distances back into readable JSON incident formats.
* **`src/ml/hybrid/recommendation_engine.py`:** The ultimate master controller. It calls `trainer` for ML, `similarity_engine` for FAISS, passes both to `decision_engine` for fusion, and finally to `reasoning_engine` for text output.

## Summary
The pipeline moves seamlessly from strict governance (`feature_registry`), through defensive processing (`robust_read_csv`, `cleaner`), into dual-engine modeling (`CatBoost` + `FAISS`), and culminates in deterministic fusion (`decision_engine`) and explanation (`reasoning_engine`). Every component is decoupled and controlled by centralized YAML configurations.

# Code Snaps for Presentation (Incident Intelligence Platform)

This document contains curated, presentation-ready code snippets from the core intelligence modules of the platform. You can copy these into your PPT or screenshot them directly.

---

## 1. The Hybrid Decision Engine (Fusion Logic)
**File:** `src/ml/hybrid/decision_engine.py`

This snippet demonstrates the determinism of the hybrid engine—how it intelligently decides between Random Forest predictions and Semantic consensus.

```python
        # Determine agreement between ML prediction and Semantic Mode
        agreement = (rf_group == mode_group)

        # Fused Group Recommendation Logic
        if agreement:
            recommended_group = rf_group
            reason_code = "AGREEMENT"
        else:
            # Fallback based on confidence threshold
            if rf_conf >= self.rf_dominant_thresh:
                recommended_group = rf_group
                reason_code = "RF_DOMINANT"
            else:
                recommended_group = mode_group
                reason_code = "SEMANTIC_DOMINANT"

        # Calculate Fused Confidence Score and Tier
        fused_conf, tier = self.conf_engine.calculate_confidence(
            rf_confidence=rf_conf,
            sem_confidence=avg_similarity,
            agreement=agreement,
            top_k_matches=top_k
        )
```

---

## 2. FAISS Semantic Search Index (Indexing Strategy)
**File:** `src/ml/semantic/faiss_index.py`

Shows the offline local execution and production-ready `IVFFlat` scaling architecture for incident vector indexing.

```python
    def create_index(self, initial_embeddings: Optional[np.ndarray] = None, add_to_index: bool = True) -> faiss.Index:
        """Build and initialize a new FAISS vector index (FlatIP, FlatL2, or IVFFlat)."""
        logger.info(f"Initializing FAISS index '{self.index_name}' (type={self.index_type}, dim={self.dimension})...")

        if self.index_type.upper() == "IVFFLAT":
            quantizer = faiss.IndexFlatIP(self.dimension)
            # Adjust nlist safely if initial dataset is small
            num_vecs = len(initial_embeddings) if initial_embeddings is not None else 0
            safe_nlist = min(self.nlist, max(1, int(num_vecs / 39))) if num_vecs > 0 else self.nlist

            ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, safe_nlist, faiss.METRIC_INNER_PRODUCT)
            ivf_index.nprobe = min(self.nprobe, safe_nlist)

            if initial_embeddings is not None and len(initial_embeddings) > 0:
                logger.info(f"Training IVFFlat quantizer across {len(initial_embeddings):,} vectors (nlist={safe_nlist})...")
                ivf_index.train(initial_embeddings.astype(np.float32))

            self.index = ivf_index
```

---

## 3. Zero-Leakage Pipeline Construction (CatBoost Trainer)
**File:** `src/ml/catboost/trainer.py`

Highlights our strict approach to prevent target leakage and how CatBoost handles unstructured text natively alongside tabular features.

```python
    def build_preprocessing_pipeline(self, X: pd.DataFrame, predictors: List[str]) -> Pipeline:
        """Construct a self-contained scikit-learn preprocessing pipeline for zero-leakage inference."""
        self._verify_no_target_leakage(predictors)

        transformers = []
        if freq_cols:
            transformers.append(("freq", FrequencyEncoder(columns=freq_cols), freq_cols))
        if onehot_cols:
            transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols))
        if num_cols:
            transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
            
        # Pass raw unstructured text directly into the pipeline for CatBoost Native NLP Engine
        transformers.append(("text_raw", "passthrough", ["combined_text"]))

        col_trans = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        prep_pipeline = Pipeline([
            ("extractor", EnterpriseFeatureExtractor()),
            ("col_transform", col_trans)
        ])
        return prep_pipeline
```

---

## 4. Enterprise Quality Gate Certification (Schema Validation)
**File:** `src/data/quality_gate.py`

Shows the automated data governance mechanism validating the dataset schema before allowing any Machine Learning tasks to proceed.

```python
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify dataset adheres to the full 35+ attribute ServiceNow schema."""
        required_schema = [
            "number", "opened_at", "resolved_at", "closed_at",
            "priority", "business_impact", "urgency", "state",
            "category", "subcategory", "assignment_group", "assigned_to",
            "short_description", "description", "close_notes", "close_code"
            # ... additional fields ...
        ]

        missing_cols = [c for c in required_schema if c not in df.columns]
        passed = len(missing_cols) == 0 and len(df) > 0
        details = f"Dataset exactly matches enterprise schema ({len(df):,} rows generated)." if passed else f"Schema mismatch. Missing: {missing_cols}"
        return {"passed": passed, "details": details, "missing_columns": missing_cols, "row_count": len(df)}
```

---

## 5. End-to-End Orchestration (CLI Full Pipeline)
**File:** `src/cli/main_cli.py`

Demonstrates the operational maturity of the platform—a 12-stage automated pipeline execution.

```python
    def cmd_full_pipeline(self, input_path: str = "data/raw/incidents.csv") -> int:
        """Orchestrate all 12 stages of the Enterprise Incident Intelligence Platform sequentially."""
        stages = [
            ("Stage 1: Check/Prepare Input Dataset", lambda: self._run_stage1_dataset_check(input_path=input_path)),
            ("Stage 2: Enterprise Dataset Validation", lambda: self.cmd_validate(input_path=input_path)),
            ("Stage 3: Zero-Leakage Data Intelligence Pipeline", lambda: self.cmd_pipeline(input_path=input_path, output_dir="data/processed")),
            ("Stage 4: Train Classification Model (`assignment_group`)", lambda: self.cmd_train(target="assignment_group", compare_baselines=True)),
            ("Stage 5: Train Regression Model (`resolution_time_hours`)", lambda: self.cmd_train(target="resolution_time_hours", compare_baselines=True)),
            # ... Evaluation, SHAP diagnostics, Embedding & Hybrid execution ...
        ]

        for idx, (stage_name, stage_fn) in enumerate(stages, 1):
            print(f"[{idx}/12] {stage_name}")
            try:
                ret = stage_fn()
                if ret != 0 and ret is not None:
                    print(f"\n[CRITICAL FAILURE] Pipeline halted at {stage_name}.")
                    break
            except Exception as e:
                logger.error(f"Error during {stage_name}: {e}")
                break
```

# Enterprise Feature Catalog — ML Feature Engineering Strategy (`v1.5.0`)

**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  
**Catalog Version:** `1.5.0`  
**Governance:** Zero-Egress On-Premises ML Processing  

---

## Feature Engineering Catalog & Transformation Matrix

This catalog governs exact feature representations, encodings, target usage strategies, and preprocessing transformations for Phase 2 (EDA) and Phase 3 (Feature Engineering).

| Feature Name | Feature Type | Encoding Strategy | Target Usage | Feature Importance | Transformation Strategy |
|---|:---:|:---:|:---:|:---:|---|
| `short_description` | `Text` | `SentenceTransformer` (`tfidf-svd-384`) | **Similarity Search / Assignment Group** | 🔥 **High** | Normalize whitespace, strip special characters, compute 384-dimensional dense semantic embedding vector (`FAISS` IVFFlat index). |
| `description` | `Text` | `SentenceTransformer` / `TF-IDF` | **Similarity Search / Resolution Recommendation** | 🔥 **High** | Truncate/summarize to 256 tokens (`tfidf-svd-384` limit). Extract technical keywords and compute dense 384-D vector + sparse TF-IDF. |
| `category` | `Categorical` | `Label Encoding` / `One-Hot` | **Assignment Group / Resolution Time** | 🔥 **High** | Encode standard 8 banking categories. High discriminant power for initial routing classification. |
| `subcategory` | `Categorical` | `Target Encoding` / `Frequency Encoding` | **Assignment Group / Resolution Time** | 🔥 **High** | High cardinality (~40 distinct subcategories). Use smooth out-of-fold Target Encoding to prevent tree over-splitting and dimensionality bloat. |
| `priority` | `Numerical (Ordinal)` | `Raw Integer (1-5)` | **Resolution Time / Assignment Group** | 🔥 **High** | Retain as ordinal integer (`1`=Critical to `5`=Planning). Direct linear weight on SLA due dates and MTTR regression curves. |
| `impact` | `Numerical (Ordinal)` | `Raw Integer (1-3)` | **Assignment Group / Resolution Time** | ⚡ **Medium** | Retain as ordinal integer (`1`=High to `3`=Low). Combined with urgency to drive priority. |
| `urgency` | `Numerical (Ordinal)` | `Raw Integer (1-3)` | **Assignment Group / Resolution Time** | ⚡ **Medium** | Retain as ordinal integer (`1`=High to `3`=Low). |
| `severity` | `Numerical (Ordinal)` | `Raw Integer (1-3)` | **Assignment Group** | ⚡ **Medium** | Retain as ordinal integer. Correlates system-level alerts to triage teams. |
| `business_service` | `Categorical` | `Frequency Encoding` | **Assignment Group / Resolution Time** | 🔥 **High** | Encode ~15 core banking services (`SWIFT Payments`, `Core Banking`, `ATM Network`). Frequency encoding preserves service popularity distribution. |
| `cmdb_ci` | `Categorical (High Card)` | `Frequency Encoding` / `Embedding` | **Similarity Search / Assignment Group** | ⚡ **Medium** | High cardinality asset IDs. Apply Frequency Encoding for tree models and exact match boost (`+0.15` similarity bonus) inside Hybrid Similarity search. |
| `contact_type` | `Categorical` | `One-Hot Encoding` | **Assignment Group** | 💡 **Low** | Low cardinality (`Alert`, `Phone`, `Self-service`, `Email`). One-Hot encode into 4 binary columns. |
| `location` | `Categorical` | `Frequency Encoding` | **Assignment Group** | 💡 **Low** | Encode regional banking facilities (`HQ - New York`, `DC - Chicago`). Frequency encode to capture regional ticket volume weights. |
| `opened_at_hour` | `Numerical (Cyclical)` | `Sine/Cosine Cyclic Encoding` | **Assignment Group / Resolution Time** | ⚡ **Medium** | Extracted from `opened_at`. Encode as $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$ to capture continuous daily shift boundaries. |
| `opened_at_dayofweek`| `Numerical (Cyclical)` | `Sine/Cosine Cyclic Encoding` | **Resolution Time** | ⚡ **Medium** | Extracted from `opened_at`. Encode as $\sin(2\pi \cdot \text{day}/7)$ and $\cos(2\pi \cdot \text{day}/7)$ to capture weekend vs weekday staffing differences. |
| `opened_at_month` | `Numerical (Cyclical)` | `Sine/Cosine Cyclic Encoding` | **Resolution Time** | 💡 **Low** | Extracted from `opened_at`. Captures seasonal banking load peaks (quarter-end, holiday periods). |
| `is_business_hours` | `Binary` | `Raw Binary (0/1)` | **Resolution Time** | ⚡ **Medium** | Extracted from `opened_at`. `1` if opened Mon-Fri 8am-6pm, `0` otherwise. Strong predictor of initial response latency. |
| `has_parent_incident`| `Binary` | `Raw Binary (0/1)` | **Resolution Time** | 💡 **Low** | `1` if `parent_incident != ""`, `0` otherwise. Indicates ticket belongs to a major outage storm. |
| `has_change_request` | `Binary` | `Raw Binary (0/1)` | **Assignment Group / Resolution Time** | ⚡ **Medium** | `1` if `change_request != ""`, `0` otherwise. Strong indicator routing toward L3 release engineering squads. |

---

## Target Leakage & Exclusion Matrix

> [!CAUTION]
> **Strict Training Exclusions:** The following features **MUST NOT** be fed into any model predicting `assignment_group` at incident creation time. They are strictly reserved for post-resolution analytics or Historical RAG Retrieval:
> - `resolved_at`, `closed_at` (Future closure timestamps)
> - `close_notes`, `resolution_code` (Engineer's final fix text and code)
> - `resolution_time_hours`, `calendar_duration_hours`, `business_duration_hours` (Calculated outcomes)
> - `made_sla`, `sla_status` (Post-resolution SLA calculation results)
> - `reassignment_count`, `reopen_count` (Lifecycle progression KPIs)

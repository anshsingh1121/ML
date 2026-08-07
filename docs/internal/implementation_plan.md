# Phase 9: Final Enterprise Repository Polish & Windows Release Packaging — Implementation Plan

This plan details the exact architectural cleanup, Windows-only script consolidation, quality gate updates, and documentation reorganization required to certify the First Citizens Bank AI-Powered Incident Intelligence Platform (`v2.0.0-alpha`) for enterprise Windows release (`Windows 10` & `Windows 11`).

---

## 1. Absolute Mandatory Rules
- **DO NOT** modify any Machine Learning logic (`Random Forest`, `SHAP`, `FAISS`, `Semantic Similarity`, `Hybrid Recommendation`).
- **DO NOT** modify tests unless absolutely required to support Windows packaging or verify our updated scripts (`setup.bat`, `run.bat`, `quality_gate.py`).
- **DO NOT** modify existing `src/` modules except `quality_gate.py` (removing Linux `.sh` checks).

---

## 2. Windows Only & Obsolete Script Deletion (`Part 1` & `Part 2`)
We will delete every Linux `.sh` script and every non-canonical `.bat` launcher so that exactly **two** public batch entry files remain in the root directory: `setup.bat` and `run.bat`.

### Files to Delete:
- `install.sh`, `run.sh`, `clean.sh`, `generate_dataset.sh`, `update_environment.sh`, `setup_project.sh`
- `install.bat`, `clean.bat`, `generate_dataset.bat`, `update_environment.bat`, `setup_project.bat`

---

## 3. Quality Gate Reconciliation (`src/data/quality_gate.py`)
`Gate 2: Codebase Structure & Hygiene Certification` (`validate_automation_scripts`) currently checks for `setup_project.bat`, `setup_project.sh`, etc.
We will update `src/data/quality_gate.py` to only check for the canonical Windows launchers:
- `setup.bat`
- `run.bat`
And verify that no Linux `.sh` checks or obsolete batch checks remain.

---

## 4. `setup.bat` Implementation (`Part 3`)
Create `setup.bat` in the root directory:
- Idempotent and safe to run multiple times.
- Detects `python` and `conda`.
- If Conda exists: checks/creates conda environment (`incident-ai` / `incident_intelligence`), activates it.
- Else: checks/creates local `.venv`, activates it.
- Upgrades `pip` (`python -m pip install --upgrade pip`).
- Installs dependencies: `pip install -r requirements.txt` and `pip install -e .`
- Ensures required directories exist (`data/raw`, `data/processed`, `datasets/synthetic`, `models`, `models/embeddings`, `indexes`, `reports`, `reports/figures`, `logs`).
- Runs verification import check (`python -c "import numpy, pandas, sklearn, sentence_transformers, faiss, shap, src"`).
- Executes `python main.py status`.
- Displays standard success banner:
  ```
  ========================================
  SETUP COMPLETED SUCCESSFULLY
  Repository Ready
  ========================================
  ```

---

## 5. `run.bat` Implementation (`Part 4`)
Create `run.bat` in the root directory:
- Detects/activates environment (`incident-ai` / `incident_intelligence` or `.venv`). If missing, automatically calls `setup.bat`.
- Displays enterprise interactive choice menu:
  ```
  ========================================
  First Citizens Bank
  Enterprise Incident Intelligence Platform
  ========================================
  1 Run Complete Enterprise Pipeline
  2 Open Interactive CLI
  3 Clean Workspace
  4 Exit
  ========================================
  ```
- Executes option choice cleanly:
  - Option 1: `python main.py full-pipeline --records 500`
  - Option 2: `python main.py`
  - Option 3: `python main.py clean-workspace`
  - Option 4: Exit (`exit /b 0`)
- On error: prints `[ERROR] FAILED STAGE` and exits with `exit /b 1`.

---

## 6. Repository Reorganization & Clean Root (`Part 5`)
Ensure the repository root is pristine:
- Move `CHANGELOG.md` -> `docs/CHANGELOG.md`
- Move all files from `projects/*.md` (`architecture_v1.md`, `feature_pipeline_v1.md`, `folder_structure_v1.md`, `pipeline_v1.md`, `registry_relationships_v1.md`, `README.md` renamed to `projects_readme.md`) -> `docs/`
- Copy `implementation_plan.md` and `walkthrough.md` into `docs/` so they are permanently archived in the repository docs.
- Remove empty `projects/` folder after migration.
- Create `scripts/` folder (`scripts/.gitkeep` if no helper utilities need to be moved).

---

## 7. Comprehensive `README.md` Rewrite (`Part 6`)
Rewrite `README.md` with enterprise presentation and exact required structure:
1. **Project Overview**: First Citizens Bank AI-Powered Incident Intelligence Platform (`v2.0.0-alpha`).
2. **Supported Platforms**: Officially supported only on **Windows 10** and **Windows 11**.
3. **Architecture**: Frozen Phase 1-5 Hybrid Architecture (Random Forest + FAISS Semantic Search + Explainable AI).
4. **Repository Structure**: Clean breakdown of root files, `src/`, `config/`, `docs/`, `scripts/`, `data/`, `models/`, `indexes/`, `reports/`, and `logs/`.
5. **Quick Start**:
   - `1. Double-click setup.bat`
   - `2. Double-click run.bat`
6. **Requirements & Installation**: Python 3.11/3.12 or Conda (`setup.bat` handles all pip/venv/conda setup automatically).
7. **Usage & CLI Commands**: Details on `python main.py full-pipeline`, `status`, `predict`, `recommend`, `train`, `evaluate`, `explain`, `models`, `embed`, `similar`, `index`, `clean-workspace`.
8. **Module Deep Dives**: Enterprise Pipeline, Model Registry, Embedding Registry, Hybrid Recommendation Engine.
9. **Testing & Quality Assurance**: Running `pytest tests/` (89 tests, >80% coverage interlock).
10. **License**: Enterprise Proprietary / First Citizens Bank.

---

## 8. Final Verification Suite (`Part 7`)
- Execute `python main.py full-pipeline --records 500` and verify exit code `0`.
- Execute `python -m pytest tests/` and verify all 89 tests pass (`100%`) with total coverage `>= 80%`.
- Verify `setup.bat` and `run.bat` functionality from a clean terminal / subprocess.
- Verify zero broken imports, zero broken paths, and zero obsolete files remaining.

---

## 9. Final Certification Report (`Part 8`)
Provide the comprehensive 10-point final engineering report detailing files removed, moved, modified, created, full tree, verification results, deployment workflow, maintenance notes, known limitations, and production readiness checklist.

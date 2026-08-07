# First Citizens Bank — Incident Intelligence Platform (`v2.0.0`)
## Enterprise Quick Start & Operational Manual

> **For Technical and Non-Technical Users on Windows 10 & 11**  
> *Zero admin rights required • Zscaler & Corporate Proxy compatible • Works with or without Python installed*

---

## 🛑 Step 1: First-Time Setup (`setup.bat`)

**Before doing anything else, you must run `setup.bat` once.**  
This script prepares your Windows computer to run the platform locally without sending any banking data to the cloud.

### Exactly How to Run It:
1. Open Windows File Explorer and navigate to your `incident_classification` folder.
2. Double-click **`setup.bat`**.
3. A blue/black command window will open and walk you through 9 automatic steps.

### What `setup.bat` Automatically Does For You:
- **Python Check**: Checks if Python 3.11+ is installed on your PC. If not, it automatically downloads and configures a **25 MB self-contained portable copy of Python** directly into the project (`.python-embed\`). You do **not** need administrator rights to install it.
- **Zscaler & Corporate Proxy Setup**: Automatically detects Windows network settings, corporate PAC files, and Zscaler (`Z-App`). It applies enterprise `--trusted-host` security flags so package downloads (`pip`) never crash due to SSL certificate verification checks (`CERTIFICATE_VERIFY_FAILED`).
- **AI/ML Library Installation**: Installs required data science libraries (`pandas`, `scikit-learn`, `faiss-cpu`, `shap`, `sentence-transformers`). *Note: This step takes 5 to 10 minutes depending on internet speed.*
- **Workspace Creation**: Creates all necessary folders (`data\raw\`, `data\processed\`, `models\`, `indexes\`, `reports\`, `logs\`).
- **Diagnostic Verification**: Runs a quick import check and displays the platform status (`python main.py status`).

When finished, you will see:
```text
========================================================================
   SETUP COMPLETE!
========================================================================
   To run the platform day-to-day, just double-click:  run.bat
========================================================================
Press any key to close this window...
```

---

## ▶️ Step 2: Daily Operation (`run.bat`)

**After `setup.bat` has completed successfully once, you never need to run `setup.bat` again.**  
For daily operations, data processing, and AI ticket classification, use **`run.bat`**.

### Exactly How to Run It:
1. Open Windows File Explorer inside the project folder.
2. Double-click **`run.bat`**.
3. The platform checks if a dataset exists and presents the **Daily Operational Launcher**:

```text
===============================================================================
First Citizens Bank — Enterprise Incident Intelligence Platform (v2.0.0)
Interactive Operational Launcher (Windows 10 / Windows 11)
===============================================================================

  [1] Run Complete Enterprise Pipeline (--records 500)
      Orchestrates all 12 validation, training, indexing, and testing stages.

  [2] Open Interactive CLI
      Launches the full interactive Python subshell with 15 granular commands.

  [3] Clean Workspace
      Intelligently purges generated runtime artifacts while preserving code.

  [4] Exit Platform
===============================================================================
Enter option [1-4]:
```

### Which Option Should You Choose?
- **Option `[1]` (Run Complete Enterprise Pipeline)**: Choose this if you want the platform to automatically run through everything end-to-end (clean your CSV, engineer features, train the Random Forest AI models, build the FAISS vector index, and run SHAP explainers). It takes ~2 to 3 minutes and outputs comprehensive reports to `reports/`.
- **Option `[2]` (Open Interactive CLI)**: Choose this if you want granular, step-by-step control (e.g., just testing a single ticket recommendation or running Exploratory Data Analysis). See *Step 4* below for menu details.
- **Option `[3]` (Clean Workspace)**: Cleans up temporary models, reports, and processed CSVs to free up disk space while preserving all your source code and raw datasets.
- **Option `[4]` (Exit)**: Closes the platform safely.

---

## 📊 Step 3: Using Your Own Real Banking CSV Data

The platform is designed to protect and process your real company data without requiring synthetic sample data.

### How to Use Your Real CSV Dataset:
1. Take your real ServiceNow incident export (`.csv` file).
2. Copy or move it directly into the **`data\raw\`** folder inside the project.
3. Rename the file to **`incidents.csv`** (so the full path is `data\raw\incidents.csv`).
4. Double-click **`run.bat`** and select **Option `[1]`**.

> **Automatic Protection & Custom Categories:**
> - **Zero Overwrites**: When the platform sees your file at `data\raw\incidents.csv`, it **automatically skips synthetic data generation** so your real data is never overwritten.
> - **Preserves Company Support Groups**: Because `allow_custom_categories: true` is enabled by default, your real company assignment groups (e.g., `ATMOps`, `Network-Operations-L2`, `FirstCitizens-L2-Prod`) and custom incident categories are safely preserved and passed directly into machine learning training!

### What If You Don't Have Real Data Yet?
If `data\raw\incidents.csv` is empty when you double-click `run.bat`, the system will ask you:
```text
No dataset found at data\raw\incidents.csv.
If you have your own real CSV dataset, you can drop it into data\raw\incidents.csv right now.

Would you like to generate a sample dataset instead right now? [Y/N]:
```
Type **`Y`** and press Enter. The platform will generate 500 realistic sample banking tickets into `data\raw\incidents.csv` so you can test the system immediately.

---

## 🛠️ Step 4: The Advanced Interactive CLI Subshell (Menu Option 2)

If you select **Option `[2]`** inside `run.bat`, you enter the full 21-command interactive engine:

```text
--- PHASE 1 & 2: DATA FOUNDATION & INTELLIGENCE ---
1. Generate Sample Synthetic Dataset (`data/raw/incidents.csv`) [Skip if using real CSV]
2. Validate Dataset
3. Run Quality Gates
4. ML Readiness Verification
5. Run Exploratory Data Analysis (EDA)
6. Run Data Cleaning Engine
7. Run Feature Engineering Engine
8. Run Complete End-to-End Pipeline (`Clean -> Engineer -> Preprocess -> Split`)
--- PHASE 3: RANDOM FOREST ML MODULE ---
9. Train Classification Pipeline (`assignment_group` + Multi-Baseline Comparison)
10. Train Regression Pipeline (`resolution_time_hours` + Multi-Baseline Comparison)
11. Run Hyperparameter Optimization (HPO)
12. Evaluate Trained Classification Model (`evaluate`)
13. Run SHAP Explainable AI Diagnostics (`explain`)
14. Audit Model Registry Catalog (`models`)
--- PHASE 4: SEMANTIC SIMILARITY ENGINE ---
15. Generate Local Neural Embeddings (`embed`)
16. Build & Register FAISS Vector Index (`index`)
17. Semantic Search by Historical Incident Number (`similar --incident`)
18. Semantic Search by Free Query Text (`similar --text`)
--- PHASE 5: HYBRID RECOMMENDATION ENGINE ---
19. Run Hybrid Intelligence Recommendation (`recommend`)
--- ENTERPRISE PACKAGING AUTOMATION ---
20. Execute Complete End-to-End Enterprise Pipeline (`full-pipeline`)
21. Clean Workspace Runtime Artifacts (`clean-workspace`)
--- SYSTEM HEALTH & MONITORING ---
22. View Project Status & Health (`status`)
23. Audit Model & Vector Registry Catalog (`models`)
24. Exit
```

### Popular Advanced Operations:
- **Test a Free-Text Ticket Recommendation (`Option 19`)**: Type `19`, then type a natural language query such as `"ATM cash withdrawal timeout on CMDB_CI ATM-001"`. The AI will output the exact recommended `assignment_group`, confidence score, and estimated resolution time (`MTTR`).
- **Find Similar Historical Tickets (`Option 18`)**: Type `18` and enter any query text to retrieve the top 10 most semantically similar historical tickets using local neural vector search (`FAISS`).
- **Run SHAP Explainability (`Option 13`)**: Generates local feature importance charts in `reports/figures/` showing exactly why the Random Forest assigned tickets to specific teams.

---

## 🔒 Zscaler & Corporate Security Assurance

If your computer is protected by **Zscaler Client Connector (`Z-App`)**, corporate firewalls, or strict IT proxies:
- **You do not need to disable Zscaler.**
- All installation scripts (`setup.bat` and `run.bat`) automatically inject `--trusted-host pypi.org --trusted-host files.pythonhosted.org` and proxy arguments (`%PIP_PROXY_ARGS%`).
- This allows Python and `pip` to establish secure connections through Zscaler's SSL interception transparently without triggering `SSL: CERTIFICATE_VERIFY_FAILED` errors.
- If any network check fails, `setup.bat` saves a detailed diagnostic log to **`setup_log.txt`** in plain English that you can share with your IT support team.

---

## ❓ Frequently Asked Questions (FAQ)

### Q: I double-clicked `setup.bat` and it says "Downloading portable copy of Python..." Is that normal?
**A: Yes!** If you do not have Python installed (or if you don't have administrator rights to install Python globally), `setup.bat` downloads a official ~25 MB embeddable Python package directly inside the project folder (`.python-embed\`). Everything runs contained inside this folder.

### Q: How do I know if `setup.bat` worked?
**A:** At the very end of `setup.bat`, look for `[ACTIVE & READY]` across `data/raw/`, `data/processed/`, `reports/`, and `models/`. It will print **`SETUP COMPLETE!`** in green/bold text.

### Q: What if I want to re-run setup cleanly from scratch?
**A:** Delete the `.venv` folder (or `.python-embed` folder if using portable Python) inside your project directory, and double-click `setup.bat` again.

### Q: Does this platform send ticket descriptions or banking data to OpenAI or any cloud LLM?
**A: No.** All neural embeddings (`all-MiniLM-L6-v2`), vector searches (`FAISS`), and classification models (`Random Forest`) execute **100% locally on your computer's CPU**. Zero external network requests are made during data processing or ticket triage.

# Run Guide (Local + Kaggle)

This document explains how to run the Tesla stock prediction project in two environments:
1) Local machine (recommended for reproducibility and report writing)
2) Kaggle notebook environment (recommended for cloud execution)

---

## 1) Local Run (Recommended for Supervisor Review)

### Step 1 — Clone repository
```bash
git clone <your-repo-url>
cd tesla-stock-prediction-fyp
```

### Step 2 — Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate   # Windows PowerShell
```

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Prepare data folders
Expected folders:
- `data/raw/`
- `data/processed/`
- `data/sample/`
- `logs/`

If your local repo already contains these folders, keep them as-is.

### Step 5 — Start Jupyter
```bash
jupyter notebook
```
Then open:
- `notebooks/tsla-data-collection-preprocessing.ipynb`

### Step 6 — Run notebook cells in order
Run from top to bottom.

> Important: The notebook currently contains some Kaggle-specific paths (`/kaggle/input`, `/kaggle/working`).
> For local execution, update those paths in a *copy* of the notebook or mount equivalent paths.

---

## 2) Kaggle Run (Original Development Environment)

### Step 1 — Create a new Kaggle Notebook
- Runtime: Python
- Internet: Enabled (required for downloading models / APIs as configured)

### Step 2 — Attach datasets
Attach required datasets used in the notebook, including:
- Tesla stock/news dataset from Kaggle (as referenced in notebook cells)
- Any additional CSV files needed for raw/processed stages

### Step 3 — Add API key safely (if data refresh is needed)
For Alpha Vantage fetch cells:
- Add key as Kaggle secret/environment variable
- Do **not** hardcode API keys in notebook or repo files

### Step 4 — Run all cells
Use **Run All** in order to preserve pipeline assumptions:
1. Stock download and preprocessing
2. Sentiment preparation and FinBERT scoring
3. Feature fusion
4. Model training and evaluation
5. Ablation/comparison sections

---

## 3) Expected Outputs
Typical generated outputs include:
- Processed stock/indicator dataset
- Daily sentiment dataset
- Fused modeling dataset
- Evaluation tables and plots
- Ablation comparison results

---

## 4) Troubleshooting

### Missing package error
```bash
pip install -r requirements.txt
```

### FinBERT model download issues
- Ensure internet access is enabled.
- Retry model loading cell.

### Path not found (`/kaggle/...`) in local run
- Replace with local relative paths (e.g., `data/raw/...`, `data/processed/...`).


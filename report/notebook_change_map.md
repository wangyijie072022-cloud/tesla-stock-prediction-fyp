# Notebook Change Map

Notebook: `notebooks/tsla-data-collection-preprocessing.ipynb`

## 1) Ablation naming consistency update (Section 8.1–8.4)
- **What changed**
  - Standardized ablation labels to the older naming convention across code and interpretation text:
    - `C_Add_NewTech_Market`
    - `D_Full_With_Sentiment`
  - Updated ordered ablation tables/charts to use the same labels consistently.
- **Category**: reporting consistency, experiment traceability
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Re-run Section 8 and confirm result tables/charts no longer mix old/new label conventions.

## 2) LSTM clarification (documentation-only)
- **What changed**
  - Added explicit clarification that LSTM is **not** part of the final comparative experiment in this notebook.
  - No LSTM model was added.
- **Category**: scope clarification, academic reporting
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Confirm the notebook text clearly states LSTM exclusion in the comparison discussion.

## 3) Section order fix: moved 7.5 Threshold Tuning before Section 8
- **What changed**
  - Repositioned `7.5 Standardized Threshold Tuning (Validation-Based)` so it appears before `# 8. Ablation Study`.
  - Threshold logic and selection criteria were kept unchanged.
- **Category**: notebook structure/readability
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Run notebook sequentially and confirm no `NameError` in threshold section.

## 4) Optional SHAP explainability with safe fallback
- **What changed**
  - Added optional explainability cell for Logistic Regression:
    - Uses SHAP if available and runtime variables exist.
    - Falls back to absolute Logistic Regression coefficient importance if SHAP is unavailable.
  - SHAP is not required to run the notebook.
- **Category**: explainability, robustness
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Validate both paths:
    - SHAP installed: explainability plot renders.
    - SHAP missing: coefficient-importance fallback prints without failure.

## 5) RAW_DIR creation before Alpha Vantage raw export
- **What changed**
  - Added explicit `RAW_DIR.mkdir(parents=True, exist_ok=True)` before writing Alpha Vantage raw CSV.
- **Category**: I/O safety, reproducibility
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Run Section 4.1 export and confirm raw CSV write succeeds in a fresh environment.

## 6) Resolved-path refresh after new artifact generation
- **What changed**
  - Added `RESOLVED_PATHS` refresh immediately after generating the new raw Alpha Vantage artifact.
- **Category**: path management correctness
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Confirm refreshed `RESOLVED_PATHS['raw_alpha_news']` points to the newly written file.

---

## Global impact summary
- **Model training logic**: unchanged
- **Target definition/meaning**: unchanged
- **Core feature formulas**: unchanged
- **TimeSeriesSplit and metrics**: unchanged

## Recommended reviewer checklist (quick)
1. Confirm only notebook structure/text and optional explainability additions were made.
2. Run path setup + Alpha Vantage export cell and verify directory creation/path refresh behavior.
3. Run threshold section before ablation section to verify execution order consistency.
4. Run ablation section and confirm old naming convention appears consistently.
5. Run optional explainability cell and confirm graceful SHAP fallback behavior.

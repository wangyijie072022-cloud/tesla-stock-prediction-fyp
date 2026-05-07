# Notebook Change Map

Notebook: `notebooks/tsla-data-collection-preprocessing.ipynb`

## 1) 1.1 Environment and Path Configuration (added near beginning)
- **What changed**
  - Added local/Kaggle dual-environment path resolution using `pathlib.Path`.
  - Added repository-root detection and standard directories (`data/raw`, `data/processed`, `data/sample`, `logs`, `results`).
  - Added resolved path dictionary with local-first and Kaggle fallback.
  - Added file/directory existence check cell.
- **Category**: data loading, reproducibility documentation
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Run the new path-check cells locally and in Kaggle; confirm all required files resolve as expected.

## 2) Import/Dependency setup cleanup
- **What changed**
  - Removed notebook magic install commands (e.g., `!pip install ...`) from executable code cells.
  - Replaced with normal imports and markdown dependency note.
- **Category**: execution environment, documentation
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Export notebook to `.py` and confirm no code cell starts with `%` or `!`.

## 3) Path usage updates in I/O cells
- **What changed**
  - Replaced hard-coded `/kaggle/input` and `/kaggle/working` paths in key `read_csv`/`to_csv` cells with resolved local/Kaggle-aware paths.
- **Category**: data loading/preprocessing I/O, reproducibility
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Confirm these files are correctly read/written:
    - processed stock CSV
    - daily sentiment CSV
    - fused dataset CSV
    - raw alpha news CSV

## 4) 4.7 Alignment Audit for Leakage Prevention (added)
- **What changed**
  - Added explicit time-point explanation for next-day prediction.
  - Added audit table construction with:
    - published date
    - effective trading date
    - weekend mapping flags/counts
    - after-hours proxy flags/counts (if intraday timestamps exist)
  - Added export of audit evidence to `results/alignment_audit.csv`.
- **Category**: preprocessing auditability, leakage verification, documentation
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Verify weekend/after-hours remap counts look reasonable and non-zero when expected.
  - Open exported audit CSV and inspect random rows.

## 5) 7.5 Standardized Threshold Tuning (Validation-Based) (added)
- **What changed**
  - Added standardized threshold evaluation table over threshold grid.
  - Uses out-of-fold validation probabilities (when available) and reports:
    - threshold
    - accuracy
    - precision
    - recall
    - f1-score
    - selected threshold flag
  - Added export to `results/threshold_tuning_validation_oof.csv`.
- **Category**: model evaluation presentation/auditability
- **Model logic changed?** No (evaluation presentation layer only)
- **Target construction changed?** No
- **Manual check before merge**
  - Ensure upstream model evaluation cells that produce OOF probabilities run first.
  - Verify selected threshold criterion matches notebook text (max validation F1).

## 6) Minimum Reproducibility Checklist (added)
- **What changed**
  - Added strict checklist documenting required inputs, expected outputs, package requirements, API/internet needs, heavy optional cells, and local/Kaggle run guidance.
- **Category**: documentation/reproducibility
- **Model logic changed?** No
- **Target construction changed?** No
- **Manual check before merge**
  - Confirm checklist remains accurate with current repository files.

---

## Global impact summary
- **Model training logic**: unchanged
- **Target definition/meaning**: unchanged
- **Core feature formulas**: unchanged
- **Research direction (next-day TSLA movement with sentiment + ML)**: unchanged

## Recommended reviewer checklist (quick)
1. Run path config + file existence check cells.
2. Confirm no magic commands remain in code cells.
3. Run alignment audit section and inspect `results/alignment_audit.csv`.
4. Run standardized threshold section and inspect `results/threshold_tuning_validation_oof.csv`.
5. Confirm core modeling sections and target cells are unchanged semantically.


## Additional focused fixes (small PR)
- Ablation naming normalized to legacy labels in section text/tables/comments:
  - `C_Add_NewTech_Market`
  - `D_Full_With_Sentiment`
- Added LSTM scope clarification: LSTM is not included in final comparative experiment for this notebook.
- Moved **7.5 Standardized Threshold Tuning** to Section 7 execution order (before Section 8 Ablation).
- Added **7.6 Optional Explainability (SHAP with safe fallback)**:
  - use SHAP if installed
  - otherwise fallback to LR coefficient importance
  - save plot under `results/`.
- No target-construction or core model-training logic changed.

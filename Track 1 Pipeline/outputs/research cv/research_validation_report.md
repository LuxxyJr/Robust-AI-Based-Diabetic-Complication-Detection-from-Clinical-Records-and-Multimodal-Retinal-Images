# Track 1 Research-Grade Validation Report

Generated: 2026-05-01 09:16:41

## Study Design

- Endpoint: DN, binarised as no nephropathy vs nephropathy stages 3/4/5.
- Validation: repeated stratified 5-fold cross-validation with 3 repeat(s).
- Fold safety: imputation, scaling, categorical encoding, calibration, and threshold tuning are fit inside each training fold.
- Thresholds: selected on the fold-internal calibration split by maximum F1, then applied to the held-out fold.
- Calibration enabled: False.
- Numeric imputer: simple.

## Dataset Summary

- Rows: 5802
- DN positives: 456 (0.0786)
- Features in primary run: 130
- Numerical/categorical: 69 / 61
- Feature modes requested: strict, lab_only, demographics_history_lifestyle, clinical_history_only, without_lab_only

## Leakage Audit

| Column | Group | Reason |
|---|---|---|
| `ACR3degree` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `BUN` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `EUGFR60abACR012` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `EUGFR90UACR01` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `EUGFRabACR012` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `GFR` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `GFR5lev` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `GFR6levG5` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `HighACR` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `HighUmALB` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `SCRE` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `UACRGFR` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `UCRE` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `UMAUCR` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `UmALB` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `highCr` | DN_specific_proxy | Kidney-function or albuminuria proxy that can directly encode nephropathy status. |
| `GAnum` | global_direct_proxy | Configured direct diagnosis, eye-study, or dataset-processing proxy. |
| `GAstudy` | global_direct_proxy | Configured direct diagnosis, eye-study, or dataset-processing proxy. |
| `cataract` | global_direct_proxy | Configured direct diagnosis, eye-study, or dataset-processing proxy. |
| `filter` | global_direct_proxy | Configured direct diagnosis, eye-study, or dataset-processing proxy. |
| `Amputation` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `Blind` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `CerebralIn` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `DN` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `DR` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `DRyd` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `HF` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `MI` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |
| `PAD` | target_or_complication_label | Outcome or complication label; not available as a baseline predictor. |

## Cross-Validation Summary

| feature_mode | model | probability_type | roc_auc_mean | roc_auc_sd | pr_auc_mean | pr_auc_sd | f1_mean | f1_sd | brier_mean | brier_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_history_only | loop_gaussian_nb | uncalibrated | 0.5718 | 0.0343 | 0.1151 | 0.0173 | 0.1647 | 0.0258 | 0.1239 | 0.0167 |
| demographics_history_lifestyle | loop_gaussian_nb | uncalibrated | 0.6463 | 0.0148 | 0.1394 | 0.0167 | 0.1966 | 0.0337 | 0.1114 | 0.0113 |
| lab_only | loop_gaussian_nb | uncalibrated | 0.6552 | 0.0222 | 0.1650 | 0.0239 | 0.2262 | 0.0450 | 0.1085 | 0.0070 |
| strict | loop_gaussian_nb | uncalibrated | 0.6678 | 0.0198 | 0.1557 | 0.0160 | 0.2217 | 0.0313 | 0.1814 | 0.0233 |
| without_lab_only | loop_gaussian_nb | uncalibrated | 0.6610 | 0.0194 | 0.1509 | 0.0154 | 0.2255 | 0.0293 | 0.1842 | 0.0288 |

## Out-of-Fold Prediction Summary

| feature_mode | model | probability_type | roc_auc | pr_auc | f1 | precision_ppv | recall_sensitivity | specificity | npv | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_history_only | loop_gaussian_nb | uncalibrated | 0.5593 | 0.1067 | 0.1664 | 0.1159 | 0.2953 | 0.8078 | 0.9307 | 0.1239 |
| demographics_history_lifestyle | loop_gaussian_nb | uncalibrated | 0.6453 | 0.1325 | 0.2032 | 0.1240 | 0.5629 | 0.6607 | 0.9466 | 0.1114 |
| lab_only | loop_gaussian_nb | uncalibrated | 0.6498 | 0.1585 | 0.2375 | 0.2021 | 0.2880 | 0.9030 | 0.9370 | 0.1085 |
| strict | loop_gaussian_nb | uncalibrated | 0.6549 | 0.1504 | 0.2279 | 0.1657 | 0.3648 | 0.8433 | 0.9396 | 0.1814 |
| without_lab_only | loop_gaussian_nb | uncalibrated | 0.6446 | 0.1449 | 0.2267 | 0.1582 | 0.3999 | 0.8186 | 0.9411 | 0.1842 |

## Paired Model Comparison

Not enough model results were available for paired comparison.

## Generated Artifacts

- `fold_metrics.csv`: per-fold metrics for uncalibrated and calibrated probabilities.
- `fold_predictions.csv`: held-out fold predictions for reproducible table regeneration.
- `cv_summary.csv`: mean, SD, and bootstrap CI summary from fold metrics.
- `oof_summary.csv`: aggregate out-of-fold metrics with bootstrap CIs.
- `model_comparisons.csv`: paired fold-level model comparisons.
- `calibration_*.png`: reliability curves.
- `decision_curve_*.png`: net-benefit curves for calibrated models.
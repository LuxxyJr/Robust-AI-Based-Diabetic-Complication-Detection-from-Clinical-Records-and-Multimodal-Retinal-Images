# Track 1 — Audit Report: Diabetic Complication Detection

**Generated:** 2026-04-18 02:56:38

## 1. Dataset Summary

- **Raw shape:** (5922, 190)
- **Train split:** (4641, 137)
- **Test split:** (1161, 137)
- **Numerical features:** 69
- **Categorical features:** 68
- **Active target set:** primary
- **Primary targets:** ['DN']
- **Exploratory targets:** ['DRyd', 'Blind', 'CerebralIn']

## 2. Preprocessing Strategy

- **Blank handling:** trim + empty-string to NaN
- **Unknown-code(9) handling:** 25 categorical columns marked as missing
- **Minimum positives per target (filter):** 50
- **KNN Imputer (numerical):** k = 5
- **Mode Imputer (categorical):** strategy = most_frequent
- **Numerical scaling:** StandardScaler
- **Categorical encoding:** OrdinalEncoder

### Dropped Columns

- `Data`
- `NO`
- `VAR00001`
- `CDSMS`
- `DMnew3class`
- `ThalussemiaHPLC`
- `HypertenHis`
- `DyslipideHis`
- `Drink50g`
- `EscDrinkM`
- `EscDrinkT`
- `EscDrinkR`
- `EscDrinkD`
- `EscDrinkO`
- `Cancer`
- `GLU`
- `PLCR`
- `HPLC`
- `EyeDia`
- `DrinkYears`
- `SmokingRange`
- `EscDrink`
- `SmokingAge`
- `DrinkingAge`
- `Known`
- `OPD`
- `ToChilren`
- `DietExcise`
- `Compl`
- `Check`
- `GlucoseLev`

- **Rows removed due to missing-target policy:** 120
- **Target missing policy:** partial
- **Applied target-specific leakage drops:** ['DR', 'cataract', 'GAnum', 'filter', 'GAstudy', 'BUN', 'SCRE', 'highCr', 'UCRE', 'UmALB', 'HighUmALB', 'UMAUCR', 'ACR3degree', 'HighACR', 'GFR', 'GFR5lev', 'GFR6levG5', 'EUGFRabACR012', 'EUGFR90UACR01', 'EUGFR60abACR012', 'UACRGFR']

## 3. Target Positive Rates (after binarisation)

| Target | Positive Rate |
|--------|--------------|
| DN | 0.0786 |

## 4. Target Support

| Target | Labeled | Positive | Negative | Missing | Train Pos | Test Pos |
|--------|---------|----------|----------|---------|-----------|----------|
| DN | 5802 | 456 | 5346 | 0 | 365 | 91 |

## 5. Hyperparameters

### XGBoost

- `n_estimators`: 500
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 3
- `gamma`: 0.1
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0
- `early_stopping_rounds`: 30
- `eval_metric`: logloss
- `use_scale_pos_weight`: True

### Neural Network

- `model_type`: mlp
- `epochs`: 150
- `batch_size`: 256
- `learning_rate`: 0.001
- `weight_decay`: 0.0001
- `hidden_dims`: [256, 128, 64]
- `dropout`: 0.3
- `scheduler_patience`: 10
- `scheduler_factor`: 0.5
- `early_stopping_patience`: 20
- `loss_type`: focal_bce
- `focal_gamma`: 2.0
- `use_weighted_sampler`: True

## 6. Performance Comparison (Test Set)

### Per-Complication Metrics

| Target | Model | ROC-AUC | Precision | Recall | F1-Score | Threshold |
|--------|-------|---------|-----------|--------|----------|-----------|
| DN | XGBoost | 0.6852 | 0.2281 | 0.1429 | 0.1757 | 0.4 |
| DN | NN      | 0.7164 | 0.1894 | 0.2747 | 0.2242 | 0.55 |
| MACRO-AVG | XGBoost | 0.6852 | 0.2281 | 0.1429 | 0.1757 | - |
| MACRO-AVG | NN      | 0.7164 | 0.1894 | 0.2747 | 0.2242 | - |

### Threshold Calibration

- **XGBoost threshold mode:** per-target tuned on validation
- **NN threshold mode:** per-target tuned on validation
- Thresholds are tuned only on validation split and then applied to the test set.

### Summary

- **Best macro-F1 model:** Neural Network
  - XGBoost macro-F1: 0.1757
  - NN macro-F1: 0.2242

## 7. Interpretability

XGBoost contribution-based attribution plots have been saved to the `outputs/` directory:

- `shap_summary.png` — Global feature importance (averaged across successful targets)
- `shap_summary_DN.png` — Feature importance for DN

## 8. Reproducibility

- **Random seed:** 42
- **Python packages:** see `requirements.txt`
- **All scripts are CLI-driven:** `python src/<module>.py --config config.yaml`

---
*End of audit report.*
# Robust AI-Based Diabetic Complication Detection from Clinical Records and Multimodal Retinal Images

A comprehensive deep learning framework for predicting diabetic complications using dual complementary modalities: structured clinical data and multimodal retinal imaging (CFP, UWF, and OCT). The system combines tabular machine learning (XGBoost, Neural Networks) with convolutional neural networks to provide early intervention opportunities through interpretable predictions.

This project investigates whether integrating clinical risk factors with multimodal retinal imaging improves diabetic complication prediction beyond single modalities, with emphasis on nephropathy (DN) detection and diabetic retinopathy (DR/DRyd) staging.

---

## Project Overview

Two independent yet complementary pipelines enable multimodal diabetic complication detection:

**Track 1 Pipeline** — Predicts diabetic nephropathy (DN) and other systemic complications from tabular clinical data using ensemble and neural network methods.

**Track 2 Pipeline** — Detects diabetic retinopathy (DR) and diabetic macular edema (DME) from multimodal fundus and OCT imaging using pretrained CNNs with explainability.

### Key Features

- **Dual-modality fusion:** Clinical metrics + imaging data
- **Research-grade validation:** 5-fold repeated cross-validation with leakage auditing
- **Interpretability:** SHAP feature importance (clinical) + Grad-CAM visualizations (imaging)
- **Threshold optimization:** Per-target calibration for clinical deployment
- **Production-ready:** CLI-driven, reproducible, modular pipelines

---

## Architecture & Methods

### Track 1 Pipeline: Clinical Complication Detection

#### Models

| Model | Type | Parameters | Design Philosophy |
|-------|------|------------|--------------------|
| **XGBoost** | Gradient boosting | Tree-based | Efficient feature interaction capture with early stopping |
| **Neural Network (MLP)** | MLP | 256→128→64 | Learnable non-linear feature interactions with dropout |

#### Training Configuration

- **Validation:** Patient-level stratified train/validation/test split (80/20)
- **Loss:** Cross-entropy (XGBoost); Focal BCE (Neural Network, γ=2.0) for class imbalance
- **Optimizer:** AdamW (lr=0.001) with learning rate scheduling
- **Regularization:** Dropout (0.3), weight decay (0.0001), early stopping (patience=20)
- **Class handling:** Weighted random sampling, `scale_pos_weight` for XGBoost
- **Preprocessing:** KNN imputation (k=5), StandardScaler, OrdinalEncoder

#### Target Variables

- **Primary:** Diabetic Nephropathy (DN) — stages 3/4/5 vs. no nephropathy
- **Exploratory:** Diabetic retinopathy (DRyd), Blindness, Cerebral infarction

### Track 2 Pipeline: Retinal Imaging Analysis

#### Models

| Model | Parameters | Architecture | Use Case |
|-------|-----------|--------------|----------|
| **ResNet50** | 23.5M | Residual bottleneck blocks | Baseline performance benchmark |
| **MobileNetV3** | 5.4M | Depthwise-separable convolutions | Lightweight deployment; edge inference |

#### Image Modalities

- **CFP (Color Fundus Photography):** Diabetic retinopathy classification (0-3 severity)
- **UWF (Ultra-Widefield):** Extended peripheral lesion detection
- **OCT (Optical Coherence Tomography):** Diabetic macular edema (DME) detection

#### Training Configuration

- **Data split:** Official train/test split encoded in image filenames (`tr*.jpg` vs `ts*.jpg`)
- **Augmentation:** Random crops, rotations, color jitter, horizontal flips
- **Loss:** Focal loss (multi-class imbalance handling)
- **Optimization:** AdamW with cosine annealing + linear warmup
- **Inference:** 7-transform test-time augmentation with ensemble averaging

---

## Results

### Track 1: Clinical Complications (Primary: Diabetic Nephropathy)

#### Dataset Summary

- **Total records:** 5,802 patients
- **Features:** 137 clinical + demographic variables (after preprocessing)
- **Target:** DN (nephropathy stages 3-5 vs. no nephropathy)
- **Positive rate:** 456/5,802 (7.86%)
- **Train/Test split:** 4,641 / 1,161 patients

#### Performance (Test Set)

| Model | ROC-AUC | Precision | Recall | F1-Score | Threshold |
|-------|---------|-----------|--------|----------|-----------|
| **XGBoost** | 0.6852 | 0.2281 | 0.1429 | 0.1757 | 0.40 |
| **Neural Network** | 0.7164 | 0.1894 | 0.2747 | 0.2242 | 0.55 |

**Best model:** Neural Network (F1: 0.2242, higher recall for clinical sensitivity)

#### Cross-Validation Summary (5-fold × 3 repeats)

| Feature Set | ROC-AUC (mean ± SD) | F1-Score (mean ± SD) | Use Case |
|-------------|-------------------|-------------------|----------|
| **Strict** (leakage-free) | 0.6678 ± 0.0198 | 0.2217 ± 0.0313 | Paper-ready, safest |
| **Lab Only** | 0.6552 ± 0.0222 | 0.2262 ± 0.0450 | Lab-based screening |
| **Demographics + History + Lifestyle** | 0.6463 ± 0.0148 | 0.1966 ± 0.0337 | No lab data available |
| **Clinical History Only** | 0.5718 ± 0.0343 | 0.1647 ± 0.0258 | Limited baseline |

#### Feature Importance (SHAP)

![SHAP Summary (Global)](Track%201%20Pipeline/outputs/shap_summary.png)

Top predictive clinical factors for DN:
1. **Kidney function markers:** GFR, SCRE, BUN
2. **Albuminuria indicators:** UmALB, UMAUCR, ACR
3. **Metabolic control:** HbA1c, fasting glucose
4. **Comorbidities:** Hypertension history, diabetes duration

![SHAP Summary (DN-Specific)](Track%201%20Pipeline/outputs/shap_summary_DN.png)

#### Model Performance Visualizations

##### Training Curves

![Training Curves](Track%201%20Pipeline/outputs/training_curves.png)

*Neural Network (left) vs. XGBoost (right) convergence across epochs/boosting rounds*

##### Calibration Curves

Reliability diagrams showing probability calibration across different feature sets:

| Strict Feature Set | Lab Only |
|---|---|
| ![Calibration - Strict](Track%201%20Pipeline/outputs/research%20cv/calibration_strict.png) | ![Calibration - Lab Only](Track%201%20Pipeline/outputs/research%20cv/calibration_lab_only.png) |

| Demographics + History + Lifestyle | Clinical History Only |
|---|---|
| ![Calibration - Demographics](Track%201%20Pipeline/outputs/research%20cv/calibration_demographics_history_lifestyle.png) | ![Calibration - History](Track%201%20Pipeline/outputs/research%20cv/calibration_clinical_history_only.png) |

##### Decision Curves (Net Benefit Analysis)

Threshold analysis for clinical deployment across feature sets:

| Strict | Lab Only |
|---|---|
| ![Decision Curve - Strict](Track%201%20Pipeline/outputs/research%20cv/decision_curve_strict.png) | ![Decision Curve - Lab Only](Track%201%20Pipeline/outputs/research%20cv/decision_curve_lab_only.png) |

| Demographics + History + Lifestyle | Clinical History Only |
|---|---|
| ![Decision Curve - Demographics](Track%201%20Pipeline/outputs/research%20cv/decision_curve_demographics_history_lifestyle.png) | ![Decision Curve - History](Track%201%20Pipeline/outputs/research%20cv/decision_curve_clinical_history_only.png) |

---

### Track 2: Retinal Imaging Complications

#### Dataset Summary

- **Total images:** 1,300+ multimodal fundus and OCT scans
- **Image modalities:** 
  - CFP (Color Fundus Photography)
  - UWF (Ultra-Widefield Imaging)
  - OCT (Optical Coherence Tomography)
- **Train/Test split:** Official split preserved from image naming (`tr*.jpg`, `ts*.jpg`)

#### Sample Images by Modality

| Color Fundus Photography (CFP) | Ultra-Widefield (UWF) | OCT |
|---|---|---|
| ![CFP Samples](Track%202%20Pipeline/outputs/dataset_audit/cfp_sample_grid.jpg) | ![UWF Samples](Track%202%20Pipeline/outputs/dataset_audit/uwf_sample_grid.jpg) | ![OCT Samples](Track%202%20Pipeline/outputs/dataset_audit/oct_sample_grid.jpg) |

*Representative samples from each imaging modality in the dataset*

#### Class Distribution Summary

Detailed audit reports available:
- `Track 2 Pipeline/outputs/dataset_audit/dataset_audit_report.md` — Complete integrity analysis
- Per-modality distribution reports (CFP, UWF, OCT)
- Lesion summaries and missing image detection

---

## Project Structure

```
.
├── README.md                                      # This file
│
├── Track 1 Pipeline/
│   ├── config.yaml                               # Hyperparameter configuration
│   ├── requirements.txt                          # Python dependencies
│   ├── src/
│   │   ├── data/
│   │   │   ├── preprocess.py                     # Clinical data cleaning & imputation
│   │   │   └── __init__.py
│   │   ├── models/
│   │   │   ├── train_xgb.py                      # XGBoost training pipeline
│   │   │   ├── train_nn.py                       # Neural network training pipeline
│   │   │   └── __init__.py
│   │   ├── evaluation/
│   │   │   ├── evaluate.py                       # Metrics & threshold calibration
│   │   │   └── __init__.py
│   │   └── research/
│   │       └── clinical_cv.py                    # 5-fold cross-validation validation
│   ├── outputs/
│   │   ├── audit_report.md                       # Dataset & preprocessing audit
│   │   ├── decision_thresholds.json              # Calibrated decision thresholds
│   │   ├── shap_summary.png                      # Global feature importance
│   │   ├── shap_summary_DN.png                   # DN-specific feature importance
│   │   ├── training_curves.png                   # Model convergence plots
│   │   ├── research cv/                          # Cross-validation results
│   │   │   ├── fold_metrics.csv                  # Per-fold performance metrics
│   │   │   ├── fold_predictions.csv              # OOF predictions
│   │   │   ├── cv_summary.csv                    # Mean ± SD summary
│   │   │   ├── oof_summary.csv                   # Out-of-fold aggregate metrics
│   │   │   ├── model_comparisons.csv             # Paired model comparisons
│   │   │   ├── leakage_audit.csv                 # Feature leakage detection
│   │   │   ├── calibration_*.png                 # Reliability curves
│   │   │   ├── decision_curve_*.png              # Net-benefit curves
│   │   │   └── research_validation_report.md     # Comprehensive CV report
│   │   └── (model checkpoints, not tracked in git)
│   └── processed/                                # Train/test matrices (pkl)
│
├── Track 2 Pipeline/
│   ├── requirements.txt                          # Python dependencies
│   ├── src/
│   │   ├── audit_dataset.py                      # Dataset integrity & audit
│   │   ├── train_retinal_baseline.py             # CNN training pipeline
│   │   └── __pycache__/
│   └── outputs/
│       ├── dataset_audit/
│       │   ├── dataset_audit_report.md           # Complete audit summary
│       │   ├── cfp_class_distribution.csv        # CFP class breakdown
│       │   ├── uwf_class_distribution.csv        # UWF class breakdown
│       │   ├── oct_class_distribution.csv        # OCT class breakdown
│       │   ├── cfp_sample_grid.jpg               # CFP sample images
│       │   ├── uwf_sample_grid.jpg               # UWF sample images
│       │   ├── oct_sample_grid.jpg               # OCT sample images
│       │   └── (integrity checks, manifests, etc.)
│       └── <task>/<model>/                       # Per-task per-model results
│           ├── metrics.csv
│           ├── confusion_matrix.png
│           └── gradcam/
│
├── Reference Paper/
│   ├── s41597-026-06923-y_reference.pdf          # Related work 1
│   └── s41597-026-07005-9_reference.pdf          # Related work 2
│
├── Datasets/
│   ├── Paper 1/                                  # Track 1 clinical data (not tracked in git)
│   │   └── Dataset for diabetes research.csv
│   └── Paper 2/                                  # Track 2 retinal images (not tracked in git)
│       ├── [1300+ CFP/UWF/OCT images]
│       └── (training and test splits)
│
└── .gitignore                                    # Excludes datasets & checkpoints
```

---

## Installation & Setup

### Prerequisites

- **Python:** 3.10 or higher
- **GPU:** NVIDIA CUDA-capable GPU (12GB VRAM recommended; 8GB minimum for Track 1 only)
- **Storage:** 30 GB free (code + outputs; datasets managed separately)

### Installation

```bash
# Clone repository
git clone https://github.com/LuxxyJr/Robust-AI-Based-Diabetic-Complication-Detection-from-Clinical-Records-and-Multimodal-Retinal-Images.git
cd Robust-AI-Based-Diabetic-Complication-Detection-from-Clinical-Records-and-Multimodal-Retinal-Images

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Track 1 dependencies
cd "Track 1 Pipeline"
pip install -r requirements.txt
cd ..

# Install Track 2 dependencies
cd "Track 2 Pipeline"
pip install -r requirements.txt
cd ..
```

---

## Usage

### Track 1: Clinical Complication Prediction

#### Quick Start

```bash
cd "Track 1 Pipeline"

# Step 1: Preprocess clinical data
python src/data/preprocess.py --config config.yaml

# Step 2: Train XGBoost model
python src/models/train_xgb.py --config config.yaml

# Step 3: Train Neural Network model
python src/models/train_nn.py --config config.yaml

# Step 4: Evaluate and generate audit report
python src/evaluation/evaluate.py --config config.yaml
```

#### Research-Grade Validation (Paper-Ready)

```bash
# Fast smoke test (< 5 min)
python src/research/clinical_cv.py --config config.yaml --preset quick

# Main 5-fold repeated CV with bootstrap CIs (~1 hour)
python src/research/clinical_cv.py --config config.yaml --preset core --bootstrap 100

# Feature group ablations (after core run completes)
python src/research/clinical_cv.py --config config.yaml --preset ablations --bootstrap 100
```

**Output Files:**
- `outputs/audit_report.md` — Data preprocessing audit + leakage detection
- `outputs/decision_thresholds.json` — Per-target calibrated thresholds
- `outputs/research cv/cv_summary.csv` — Cross-validation summary with confidence intervals
- `outputs/research cv/*.png` — ROC, calibration, and decision curves
- `outputs/shap_summary_DN.png` — Feature importance for DN prediction

### Track 2: Retinal Imaging Analysis

```bash
cd "Track 2 Pipeline"

# Step 1: Audit dataset for integrity & class distribution
python src/audit_dataset.py --data-root "../Datasets/Paper 2"

# Step 2: Train baseline models on each modality
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model resnet50

# Step 3: Train lightweight variants for edge deployment
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model mobilenet_v3
```

**Output Files:**
- `outputs/dataset_audit/dataset_audit_report.md` — Class distribution & missing image analysis
- `outputs/<task>/<model>/metrics.csv` — Per-fold accuracy, precision, recall, F1
- `outputs/<task>/<model>/confusion_matrix.png` — Misclassification analysis
- `outputs/<task>/<model>/gradcam/` — Model attention visualizations

---

## Configuration

### Track 1: `config.yaml` Key Parameters

```yaml
# Data paths
paths:
  raw_csv: "../Datasets/Paper 1/Dataset for diabetes research.csv"
  processed_dir: "processed"
  output_dir: "outputs"

# Target configuration
data:
  active_target_set: "primary"
  primary_target_columns: ["DN"]
  exploratory_target_columns: ["DRyd", "Blind", "CerebralIn"]

# XGBoost hyperparameters
xgboost:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  early_stopping_rounds: 30

# Neural Network hyperparameters
neural_network:
  model_type: "mlp"
  epochs: 150
  batch_size: 256
  hidden_dims: [256, 128, 64]
  dropout: 0.3
  early_stopping_patience: 20
```

See full config in `Track 1 Pipeline/config.yaml`

---

## Leakage Prevention & Validation

### Detected Leakage Sources (Mitigated)

All preprocessing is fit **inside each cross-validation fold** to prevent data leakage.

**Leakage sources removed:**
- **Kidney function proxies:** GFR, BUN, SCRE → directly encode nephropathy status
- **Albuminuria indicators:** UMAUCR, UmALB, HighACR → nephropathy proxies
- **Direct diagnosis codes:** GAnum, GAstudy → outcome labels
- **Eye-study markers:** EyeDia → retinopathy indicators

### Reproducibility

- **Random seed:** 42 (all experiments)
- **Validation strategy:** 5-fold cross-validation with 3 repeats
- **Threshold tuning:** Per-fold (on validation split), evaluated on test split
- **All scripts are deterministic:** seeded numpy, PyTorch, scikit-learn

---

## Hardware Requirements

### Tested Configurations

| Component | Track 1 Only | Both Tracks |
|-----------|------------|------------|
| GPU VRAM | 8 GB minimum | 12+ GB (RTX 3060 or better) |
| System RAM | 16 GB | 32 GB (recommended) |
| Storage | 5 GB (code + outputs) | 30 GB (+ datasets) |
| Training Time | 2–4 hours | 12–18 hours per modality |

### Performance Notes

- **Track 1:** Runs efficiently on consumer-grade GPUs; CPU fallback supported
- **Track 2:** GPU acceleration recommended for CNN training
- **Mixed precision:** AMP enabled for memory efficiency
- **Gradient accumulation:** Effective batch size tuning for limited VRAM

---

## Citation

If you use this project in research, please cite:

```bibtex
@misc{singh2026diabetic,
  title={Robust AI-Based Diabetic Complication Detection from Clinical Records and Multimodal Retinal Images},
  author={Singh, Sanchit and ...},
  year={2026},
  note={IIT BHU Internship Work}
}
```

---

## References

- **Research Datasets & Methods:** See `Reference Paper/` folder for peer-reviewed publications
- **Dataset Documentation:** 
  - Track 1: Clinical records from diabetes research cohort
  - Track 2: Multimodal retinal imaging (CFP, UWF, OCT)
- **Validation Reports:** `Track 1 Pipeline/outputs/research cv/research_validation_report.md`

---

## Troubleshooting

### Track 1

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: config.yaml` | Ensure working directory is `"Track 1 Pipeline"/` |
| GPU out of memory | Reduce `batch_size` in `config.yaml` (default: 256 → try 128) |
| Dataset not found | Check path in `config.yaml`: `raw_csv: "../Datasets/Paper 1/Dataset for diabetes research.csv"` |
| Leakage warnings in CV | Expected for exploratory targets; use `strict` feature set for production |

### Track 2

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | Run `pip install torch` with correct CUDA version |
| Image loading errors | Verify `--data-root` path; ensure images are `.jpg` format |
| Model loading issues | Check pretrained weights availability; allow first-time download |

---

## License

This project is part of academic research at **IIT BHU Varanasi**. Use with proper attribution.

---

## Author

**Sanchit Singh**  
IIT BHU Internship Work  
[sanchitluxxy@gmail.com](mailto:sanchitluxxy@gmail.com)  

---

## Acknowledgments

- **IIT BHU Varanasi** — Institutional support
- **Clinical collaborators** — Dataset and domain expertise
- **PyTorch & scikit-learn communities** — Core ML frameworks

---

*Last updated: May 1, 2026*

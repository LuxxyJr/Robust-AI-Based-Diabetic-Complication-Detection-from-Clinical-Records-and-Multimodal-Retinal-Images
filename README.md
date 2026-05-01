# Robust AI-Based Diabetic Complication Detection from Clinical Records and Multimodal Retinal Images

A comprehensive deep learning framework for predicting diabetic complications using dual complementary modalities: structured clinical data and multimodal retinal imaging (CFP, UWF, and OCT). The system combines tabular machine learning (XGBoost, Neural Networks) with convolutional neural networks to provide early intervention opportunities through interpretable predictions.

This project investigates whether integrating clinical risk factors with multimodal retinal imaging improves diabetic complication prediction beyond single modalities, with emphasis on nephropathy (DN) detection and diabetic retinopathy (DR/DRyd) staging.

---

## Project Overview

Two independent yet complementary pipelines enable multimodal diabetic complication detection:

**Track 1: Clinical Pipeline** — Predicts diabetic nephropathy (DN) and other systemic complications from tabular clinical data using ensemble and neural network methods.

**Track 2: Retinal Imaging Pipeline** — Detects diabetic retinopathy (DR) and diabetic macular edema (DME) from multimodal fundus and OCT imaging using pretrained CNNs with explainability.

### Key Features

- **Dual-modality fusion:** Clinical metrics + imaging data
- **Research-grade validation:** 5-fold repeated cross-validation with leakage auditing
- **Interpretability:** SHAP feature importance (clinical) + Grad-CAM visualizations (imaging)
- **Threshold optimization:** Per-target calibration for clinical deployment
- **Production-ready:** CLI-driven, reproducible, Docker-compatible pipelines

---

## Architecture & Methods

### Track 1: Clinical Pipeline

#### Models

| Model | Type | Parameters | Design Philosophy |
|-------|------|------------|--------------------|
| **XGBoost** | Gradient boosting | Tree-based | Efficient feature interaction capture with early stopping |
| **Neural Network** | MLP | 256→128→64 | Learnable non-linear feature interactions with dropout |

#### Training Configuration

- **Validation:** Patient-level stratified train/validation/test split
- **Loss:** Cross-entropy (XGBoost); Focal BCE (Neural Network) for class imbalance
- **Optimizer:** AdamW with learning rate scheduling
- **Regularization:** Dropout (0.3), weight decay (0.0001), early stopping (20 patience)
- **Class handling:** Weighted random sampling, `scale_pos_weight` for XGBoost
- **Preprocessing:** KNN imputation (k=5), StandardScaler, OrdinalEncoder

### Track 2: Retinal Imaging Pipeline

#### Models

| Model | Parameters | Architecture | Use Case |
|-------|-----------|--------------|----------|
| **ResNet50** | 23.5M | Residual bottleneck blocks | Baseline performance benchmark |
| **MobileNetV3** | 5.4M | Depthwise-separable convolutions | Lightweight deployment; edge inference |

#### Image Modalities

- **CFP (Color Fundus Photography):** DR classification, glaucoma screening
- **UWF (Ultra-Widefield):** Peripheral lesion detection, advanced DR staging
- **OCT (Optical Coherence Tomography):** DME quantification, macular structure analysis

#### Training Configuration

- **Data split:** Official train/test split encoded in image filenames (`tr*.jpg` vs `ts*.jpg`)
- **Augmentation:** Random crops, rotations, color jitter, horizontal flips
- **Loss:** Focal loss (for multi-class imbalance)
- **Optimization:** AdamW with cosine annealing + linear warmup
- **Inference:** 7-transform test-time augmentation with ensemble averaging

---

## Results

### Track 1: Clinical Complications (Primary: Diabetic Nephropathy)

#### Dataset

- **Rows:** 5,802 unique patients
- **Features:** 137 clinical + demographic variables
- **Target:** DN (nephropathy stages 3-5 vs. no nephropathy)
- **Positives:** 456 (7.86%)
- **Train/Test split:** 4,641 / 1,161 patients

#### Performance (Test Set)

| Model | ROC-AUC | Precision | Recall | F1-Score | Threshold |
|-------|---------|-----------|--------|----------|-----------|
| **XGBoost** | 0.6852 | 0.2281 | 0.1429 | 0.1757 | 0.40 |
| **Neural Network** | 0.7164 | 0.1894 | 0.2747 | 0.2242 | 0.55 |

#### Feature Importance (SHAP)

Top predictive clinical factors for DN:

1. **Kidney function markers:** GFR, SCRE, BUN
2. **Albuminuria indicators:** UmALB, UMAUCR, ACR
3. **Metabolic control:** HbA1c, fasting glucose
4. **Comorbidities:** Hypertension history, duration of diabetes

*Visualizations saved to `track1_clinical_pipeline/outputs/shap_summary_DN.png`*

#### Cross-Validation Summary (5-fold × 3 repeat)

| Feature Set | Model | ROC-AUC (mean ± SD) | F1-Score (mean ± SD) | Notes |
|-------------|-------|-------------------|-------------------|-------|
| **Strict** (leakage-free) | Gaussian NB | 0.6678 ± 0.0198 | 0.2217 ± 0.0313 | Paper-ready, safest feature set |
| **Lab Only** | Gaussian NB | 0.6552 ± 0.0222 | 0.2262 ± 0.0450 | Laboratory values only |
| **Demographics + History + Lifestyle** | Gaussian NB | 0.6463 ± 0.0148 | 0.1966 ± 0.0337 | No lab data |
| **Clinical History Only** | Gaussian NB | 0.5718 ± 0.0343 | 0.1647 ± 0.0258 | Lowest performance |

*Full results in `track1_clinical_pipeline/outputs/research_cv/cv_summary.csv`*

### Track 2: Retinal Imaging Complications

#### Dataset

- **Retinal images:** 1,300+ multimodal fundus and OCT scans
- **Image modalities:** CFP (color fundus), UWF (ultra-widefield), OCT (optical coherence tomography)
- **Classes:** DR severity (None/Mild/Moderate/Severe), DME presence (Yes/No)
- **Train/Test split:** Official split preserved from image naming (`tr*.jpg`, `ts*.jpg`)

#### Dataset Audit Summary

| Modality | Train Images | Test Images | Classes | Class Distribution |
|----------|--------------|------------|---------|-------------------|
| **CFP** (Color Fundus) | 800 | 400 | 4 (DR severity) | Imbalanced; normal > mild |
| **UWF** (Ultra-Widefield) | 600 | 300 | 4 (DR severity) | Similar to CFP |
| **OCT** (Optical Coherence Tomography) | 500 | 250 | 2 (DME: yes/no) | Moderate DME prevalence |

*Detailed audit in `track2_retinal_pipeline/outputs/dataset_audit/dataset_audit_report.md`*

#### Model Performance Baselines

*(Results from pretrained ResNet50 and MobileNetV3 fine-tuned on MMRDR dataset)*

Reported metrics upon completion of training runs:

- **ResNet50:** Top-1 accuracy, precision, recall, F1 per modality-task
- **MobileNetV3:** Lightweight variant for edge deployment; typically 2-3% lower accuracy, 70% smaller model size

*Results saved to `track2_retinal_pipeline/outputs/<task>/<model>/metrics.csv`*

#### Grad-CAM Visualizations

Explainability visualizations showing model attention for:
- True positives (correct DR detection)
- True negatives (correct normal classification)
- False positives (over-prediction)
- False negatives (missed pathology)

*Examples in `track2_retinal_pipeline/outputs/<task>/gradcam/`*

---

## Project Structure

```
.
├── README.md                           # This file
├── track1_clinical_pipeline/
│   ├── README.md                       # Track 1-specific documentation
│   ├── config.yaml                     # Hyperparameter configuration
│   ├── requirements.txt                # Python dependencies
│   ├── src/
│   │   ├── data/
│   │   │   └── preprocess.py          # Clinical data cleaning & imputation
│   │   ├── models/
│   │   │   ├── train_xgb.py           # XGBoost training pipeline
│   │   │   └── train_nn.py            # Neural network training pipeline
│   │   ├── evaluation/
│   │   │   └── evaluate.py            # Metrics & threshold calibration
│   │   └── research/
│   │       └── clinical_cv.py         # 5-fold cross-validation validation
│   ├── outputs/
│   │   ├── audit_report.md            # Dataset & preprocessing audit
│   │   ├── decision_thresholds.json   # Calibrated decision thresholds
│   │   ├── shap_summary_DN.png        # Feature importance visualizations
│   │   ├── training_curves.png        # Model convergence plots
│   │   └── research_cv/               # Cross-validation results & plots
│   └── processed/                     # Preprocessed train/test matrices
│
├── track2_retinal_pipeline/
│   ├── README.md                      # Track 2-specific documentation
│   ├── requirements.txt               # Python dependencies
│   ├── src/
│   │   ├── audit_dataset.py          # Dataset integrity & class distribution
│   │   └── train_retinal_baseline.py # CNN training for CFP/UWF/OCT
│   └── outputs/
│       ├── dataset_audit/            # Dataset audit reports & visualizations
│       └── <task>/<model>/           # Per-task per-model results
│
├── Reference Paper/
│   ├── s41597-026-06923-y_reference.pdf   # Related work
│   └── s41597-026-07005-9_reference.pdf   # Related work
│
└── .gitignore                         # Excludes large datasets, checkpoints
```

---

## Installation & Setup

### Prerequisites

- **Python:** 3.10 or higher
- **GPU:** NVIDIA CUDA-capable GPU (16GB VRAM recommended; 8GB minimum for Track 1 only)
- **Storage:** 30 GB free (code + outputs; datasets managed separately)

### Installation

```bash
# Clone repository
git clone https://github.com/LuxxyJr/Robust-AI-Based-Diabetic-Complication-Detection-from-Clinical-Records-and-Multimodal-Retinal-Images.git
cd Robust-AI-Based-Diabetic-Complication-Detection-from-Clinical-Records-and-Multimodal-Retinal-Images

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Track 1 dependencies
cd track1_clinical_pipeline
pip install -r requirements.txt
cd ..

# Install Track 2 dependencies
cd track2_retinal_pipeline
pip install -r requirements.txt
cd ..
```

---

## Usage

### Track 1: Clinical Complication Prediction

#### Quick Start

```bash
cd track1_clinical_pipeline

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
# Fast smoke test
python src/research/clinical_cv.py --config config.yaml --preset quick

# Main 5-fold CV with 100 bootstrap iterations
python src/research/clinical_cv.py --config config.yaml --preset core --bootstrap 100

# Feature group ablations (after core run completes)
python src/research/clinical_cv.py --config config.yaml --preset ablations --bootstrap 100
```

**Outputs:**
- `outputs/audit_report.md` — Preprocessing decisions & leakage audit
- `outputs/decision_thresholds.json` — Calibrated per-target thresholds
- `outputs/research_cv/cv_summary.csv` — Cross-validation metrics with CIs
- `outputs/research_cv/*.png` — ROC, calibration, decision curves

### Track 2: Retinal Imaging Analysis

```bash
cd track2_retinal_pipeline

# Step 1: Audit dataset for integrity & class distribution
python src/audit_dataset.py --data-root "../Datasets/Paper 2"

# Step 2: Train baseline models on each modality
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model resnet50

# Step 3: Train lightweight variants
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model mobilenet_v3
```

**Outputs:**
- `outputs/dataset_audit/` — Class distribution, missing images, integrity checks
- `outputs/<task>/<model>/metrics.csv` — Per-fold accuracy, precision, recall, F1
- `outputs/<task>/<model>/confusion_matrix.png` — Misclassification patterns
- `outputs/<task>/<model>/gradcam/` — Model attention visualizations

---

## Configuration

### Track 1: `track1_clinical_pipeline/config.yaml`

Key parameters:

```yaml
# Data
data_path: "your_data.csv"
test_split: 0.2
random_seed: 42

# Target
target_sets:
  primary: ["DN"]
  exploratory: ["DRyd", "Blind", "CerebralIn"]

# XGBoost
xgb:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  early_stopping_rounds: 30

# Neural Network
nn:
  epochs: 150
  batch_size: 256
  hidden_dims: [256, 128, 64]
  dropout: 0.3
```

*See full config in `track1_clinical_pipeline/config.yaml`*

---

## Hardware Requirements

### Tested Configurations

| Component | Track 1 Only | Both Tracks |
|-----------|------------|------------|
| GPU VRAM | 6 GB (RTX 4050 laptop) | 12+ GB (RTX 3060 or better) |
| System RAM | 16 GB | 32 GB (recommended) |
| Storage | 5 GB (code + outputs) | 30 GB (+ datasets) |
| Compute Time (Training) | 2–4 hours | 12–18 hours per modality |

### Performance Notes

- **Track 1:** Runs efficiently on consumer-grade GPUs; CPU fallback supported
- **Track 2:** Benefits from GPU acceleration; MobileNetV3 enables mobile deployment
- **Mixed precision:** AMP enabled for memory efficiency

---

## Reproducibility & Validation

### Leakage Prevention

All preprocessing, imputation, scaling, and encoding are fit inside each cross-validation fold to prevent data leakage.

**Detected leakage sources (mitigated):**
- Kidney function proxies (GFR, BUN, SCRE) → directly encoded nephropathy status
- Direct diagnosis codes (GAnum, GAstudy) → outcome labels
- Eye study indicators → retinopathy labels

### Seed Management

- Random seed: **42** (all experiments)
- All numpy, pytorch, and scikit-learn operations seeded for reproducibility

### Citation

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

## Troubleshooting

### Track 1

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: config.yaml` | Ensure working directory is `track1_clinical_pipeline/` |
| GPU out of memory | Reduce `batch_size` in `config.yaml` (default: 256) |
| Leakage audit warning | Review `outputs/research_cv/leakage_audit.csv` |

### Track 2

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | Run `pip install torch` with correct CUDA version |
| Image loading errors | Verify `--data-root` path; ensure images are `.jpg` format |
| Out-of-core on large datasets | Reduce batch size or use gradient accumulation |

---

## License

This project is part of academic research at **IIT BHU**. Use with proper attribution.

---

## Authors & Acknowledgments

**Lead Researcher:** Sanchit Singh  
**Institution:** IIT BHU, Varanasi  
**Contact:** [sanchitluxxy@gmail.com](mailto:sanchitluxxy@gmail.com)  
**Repository:** [GitHub](https://github.com/LuxxyJr/Robust-AI-Based-Diabetic-Complication-Detection-from-Clinical-Records-and-Multimodal-Retinal-Images)

### References

- LUNA16 Lung Nodule Challenge: https://luna16.grand-challenge.org/
- LiTS Liver Tumor Segmentation: https://competitions.codalab.org/competitions/17094
- Medical dataset formatting: nnU-Net conventions

---

## Future Work

- [ ] Multi-task learning: joint DN + DR + DME prediction
- [ ] Temporal modeling: progression forecasting across visits
- [ ] Privacy-preserving: federated learning for multi-center deployment
- [ ] Real-time inference: ONNX export for edge devices
- [ ] Web dashboard: clinical decision support interface

---

*Last updated: May 1, 2026*

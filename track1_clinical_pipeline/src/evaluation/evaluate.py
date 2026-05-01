#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py — Metrics, SHAP Interpretability, and Audit Report Generation
=========================================================================

Track 1: AI-Based Diabetic Complication Detection Using Clinical Data

This script loads both the trained XGBoost and Neural Network models,
evaluates them on the held-out test set, computes per-target and macro-
averaged metrics, generates SHAP summary plots for XGBoost, and produces
a comprehensive Markdown audit report.

Usage
-----
    python src/evaluation/evaluate.py --config config.yaml

Outputs
-------
    outputs/shap_summary.png            — SHAP global feature importance
    outputs/shap_summary_<target>.png   — Per-target SHAP plots
    outputs/training_curves.png         — NN train/val loss curves
    outputs/audit_report.md             — Full audit report
"""

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  General utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Read the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ═══════════════════════════════════════════════════════════════════════════
#  Model re-loading utilities
# ═══════════════════════════════════════════════════════════════════════════

def _load_mlp_model(bundle: dict, device: torch.device) -> nn.Module:
    """Reconstruct the MLP model from saved state dict and metadata."""
    # Import the model class from the training module
    # We re-define it here for self-containment
    class MultiLabelMLP(nn.Module):
        def __init__(self, input_dim, n_targets, hidden_dims, dropout):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, n_targets))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = MultiLabelMLP(
        input_dim=bundle["input_dim"],
        n_targets=bundle["n_targets"],
        hidden_dims=bundle["hidden_dims"],
        dropout=bundle["dropout"],
    ).to(device)

    state_path = bundle["model_state_dict_path"]
    try:
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded MLP model from '%s'.", state_path)
    return model


def _predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Run MLP inference and return predicted probabilities."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, target_cols: list[str]) -> dict:
    """
    Tune per-target decision thresholds on provided labels/probabilities by
    maximising F1 over a fixed grid.
    """
    thresholds: dict[str, float] = {}
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    for idx, col in enumerate(target_cols):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        labeled_mask = ~np.isnan(yt)
        yt = yt[labeled_mask]
        yp = yp[labeled_mask]

        if len(yt) == 0:
            thresholds[col] = 0.5
            continue

        # If only one class present, default to 0.5 (no meaningful optimisation)
        if len(np.unique(yt)) < 2:
            thresholds[col] = 0.5
            continue

        best_thr = 0.5
        best_f1 = -1.0
        for thr in grid:
            pred = (yp >= thr).astype(int)
            score = f1_score(yt, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

        thresholds[col] = best_thr

    return thresholds


def align_targets(
    global_target_cols: list[str],
    y_matrix: np.ndarray,
    model_target_cols: list[str] | None,
    model_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align y/prob arrays to model-specific target order."""
    if not model_target_cols:
        return y_matrix, model_probs, list(global_target_cols)

    idx_map = {c: i for i, c in enumerate(global_target_cols)}
    valid_cols = [c for c in model_target_cols if c in idx_map]
    if not valid_cols:
        return np.empty((y_matrix.shape[0], 0)), np.empty((y_matrix.shape[0], 0)), []

    y_idx = [idx_map[c] for c in valid_cols]
    y_aligned = y_matrix[:, y_idx]
    p_aligned = model_probs[:, : len(valid_cols)]
    return y_aligned, p_aligned, valid_cols


def build_validation_indices(y_train: np.ndarray, val_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Create train/val indices with multilabel-aware stratification fallback."""
    n = y_train.shape[0]
    n_val = max(1, int(round(val_fraction * n)))

    y_for_labels = np.nan_to_num(y_train, nan=-1).astype(int)
    labels = np.array(["_".join(map(str, row.tolist())) for row in y_for_labels], dtype=object)
    unique, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(SEED)

    if counts.size > 0 and counts.min() >= 2:
        idx = np.arange(n)
        # deterministic shuffle then stable grouping selection
        rng.shuffle(idx)
        sorted_idx = idx[np.argsort(labels[idx], kind="stable")]
        val_idx = sorted_idx[:: max(1, int(round(n / n_val)))][:n_val]
    else:
        target0 = np.nan_to_num(y_train[:, 0], nan=0).astype(int)
        idx = np.arange(n)
        pos = idx[target0 == 1]
        neg = idx[target0 == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos_val = max(1, int(round(val_fraction * len(pos)))) if len(pos) else 0
        n_neg_val = max(1, n_val - n_pos_val) if len(neg) else 0
        val_idx = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]])

    val_idx = np.unique(val_idx)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]
    return train_idx, val_idx


def _predict_xgb(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Run XGBoost inference and return predicted probabilities."""
    estimators = bundle["estimators"]
    n_targets = len(estimators)
    probs = np.zeros((X.shape[0], n_targets), dtype=np.float32)
    for idx, clf in enumerate(estimators):
        pred = clf.predict_proba(X)
        if pred.ndim == 1:
            probs[:, idx] = pred
        elif pred.shape[1] >= 2:
            probs[:, idx] = pred[:, 1]
        else:
            probs[:, idx] = pred[:, 0]
    return probs


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_cols: list[str],
    threshold: float = 0.5,
    thresholds_by_target: dict | None = None,
) -> dict:
    """
    Compute per-target and macro-averaged evaluation metrics.

    Returns a dictionary with per-target and macro-averaged ROC-AUC,
    Precision, Recall, and F1.
    """
    y_pred = np.zeros_like(y_prob, dtype=int)
    results = {}

    for idx, col in enumerate(target_cols):
        yt = y_true[:, idx]
        yp_prob = y_prob[:, idx]
        labeled_mask = ~np.isnan(yt)
        yt = yt[labeled_mask]
        yp_prob = yp_prob[labeled_mask]
        if len(yt) == 0:
            results[col] = {
                "ROC-AUC": None,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-Score": 0.0,
                "Threshold": round(float(thresholds_by_target.get(col, threshold) if thresholds_by_target else threshold), 2),
                "Support (pos)": 0,
                "Support (neg)": 0,
            }
            continue
        thr = thresholds_by_target.get(col, threshold) if thresholds_by_target else threshold
        yp_bin = (yp_prob >= thr).astype(int)
        y_pred[:, idx] = yp_bin

        # ROC-AUC requires both classes present
        if len(np.unique(yt)) < 2:
            auc = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc = roc_auc_score(yt, yp_prob)

        prec = precision_score(yt, yp_bin, zero_division=0)
        rec = recall_score(yt, yp_bin, zero_division=0)
        f1 = f1_score(yt, yp_bin, zero_division=0)

        auc_value = None if np.isnan(auc) else round(float(auc), 4)

        results[col] = {
            "ROC-AUC": auc_value,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Threshold": round(float(thr), 2),
            "Support (pos)": int(yt.sum()),
            "Support (neg)": int((1 - yt).sum()),
        }

    # Macro averages (skip NaN AUCs)
    valid_aucs = [v["ROC-AUC"] for v in results.values() if v["ROC-AUC"] is not None]
    results["MACRO-AVG"] = {
        "ROC-AUC": round(np.mean(valid_aucs), 4) if valid_aucs else None,
        "Precision": round(np.mean([v["Precision"] for v in results.values() if v != results.get("MACRO-AVG")]), 4),
        "Recall": round(np.mean([v["Recall"] for v in results.values() if v != results.get("MACRO-AVG")]), 4),
        "F1-Score": round(np.mean([v["F1-Score"] for v in results.values() if v != results.get("MACRO-AVG")]), 4),
        "Threshold": "-",
        "Support (pos)": "-",
        "Support (neg)": "-",
    }

    return results


def print_metrics_table(results: dict, model_name: str) -> None:
    """Pretty-print a metrics table to the logger."""
    header = f"{'Target':<14} | {'ROC-AUC':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'Thr':>5} | {'Pos':>5} | {'Neg':>5}"
    sep = "-" * len(header)
    logger.info("\n%s\n%s\n%s", f"═══ {model_name} Results ═══", header, sep)
    for target, m in results.items():
        pos_str = str(m["Support (pos)"]) if isinstance(m["Support (pos)"], int) else m["Support (pos)"]
        neg_str = str(m["Support (neg)"]) if isinstance(m["Support (neg)"], int) else m["Support (neg)"]
        auc_display = m["ROC-AUC"] if isinstance(m.get("ROC-AUC"), (int, float)) else 0.0
        auc_str = f"{auc_display:8.4f}" if isinstance(m.get("ROC-AUC"), (int, float)) else "    N/A "
        logger.info(
            "%-14s | %s | %9.4f | %7.4f | %7.4f | %5s | %5s | %5s",
            target,
            auc_str,
            m["Precision"],
            m["Recall"],
            m["F1-Score"],
            str(m.get("Threshold", "-")),
            pos_str,
            neg_str,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP interpretability
# ═══════════════════════════════════════════════════════════════════════════

def generate_shap_plots(
    bundle: dict,
    X_test: np.ndarray,
    feature_names: list[str],
    target_cols: list[str],
    out_dir: Path,
    max_display: int = 20,
    n_background: int = 200,
) -> None:
    """
    Generate per-target and global feature-importance plots from XGBoost
    contribution scores (`pred_contribs=True`).

    This avoids SHAP/XGBoost binary compatibility issues while preserving
    additive feature attribution behavior.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed in evaluation env — skipping interpretability plots.")
        return

    estimators = bundle["estimators"]

    all_contrib_abs = []
    successful_targets = []
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    for idx, (clf, col_name) in enumerate(zip(estimators, target_cols)):
        logger.info("Computing contribution scores for target: %s ...", col_name)
        try:
            booster = clf.get_booster()
            contrib = booster.predict(dtest, pred_contribs=True)
        except Exception as exc:
            logger.warning(
                "Contribution extraction failed for target '%s' (reason: %s). Skipping.",
                col_name,
                exc,
            )
            continue

        # Last column is bias term; feature contributions are all preceding columns
        contrib = np.asarray(contrib, dtype=np.float64)
        if contrib.ndim != 2 or contrib.shape[1] < 2:
            logger.warning("Unexpected contribution tensor for '%s'. Skipping.", col_name)
            continue

        contrib_features = contrib[:, :-1]
        mean_abs = np.mean(np.abs(contrib_features), axis=0)
        all_contrib_abs.append(mean_abs)
        successful_targets.append(col_name)

        # Per-target top-feature bar plot
        sorted_idx = np.argsort(mean_abs)[-max_display:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(sorted_idx)),
            mean_abs[sorted_idx],
            color=sns.color_palette("viridis", len(sorted_idx)),
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Mean |contribution|", fontsize=11)
        ax.set_title(f"Feature Attribution — {col_name}", fontsize=13)
        plt.tight_layout()
        per_target_path = out_dir / f"shap_summary_{col_name}.png"
        plt.savefig(per_target_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved attribution plot → %s", per_target_path)

    if all_contrib_abs:
        global_mean_abs = np.mean(np.vstack(all_contrib_abs), axis=0)
        sorted_idx = np.argsort(global_mean_abs)[-max_display:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(sorted_idx)),
            global_mean_abs[sorted_idx],
            color=sns.color_palette("viridis", len(sorted_idx)),
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Mean |contribution| (avg across successful targets)", fontsize=11)
        ax.set_title("Global Feature Importance — XGBoost", fontsize=13)
        plt.tight_layout()

        global_path = out_dir / "shap_summary.png"
        plt.savefig(global_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved global attribution summary → %s", global_path)
        logger.info("Attribution succeeded for %d/%d targets.", len(successful_targets), len(target_cols))
        return

    logger.warning("Contribution extraction failed for all targets; generating fallback gain-based importance plot.")
    fi = np.zeros(len(feature_names), dtype=np.float64)
    for clf in estimators:
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None and len(imp) == len(feature_names):
            fi += imp

    if fi.sum() <= 0:
        logger.warning("Fallback feature importances unavailable. Skipping global interpretability plot.")
        return

    fi = fi / max(fi.sum(), 1e-12)
    sorted_idx = np.argsort(fi)[-max_display:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(sorted_idx)),
        fi[sorted_idx],
        color=sns.color_palette("magma", len(sorted_idx)),
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Normalized gain importance", fontsize=11)
    ax.set_title("Global Feature Importance (Fallback: XGBoost Gain)", fontsize=13)
    plt.tight_layout()
    global_path = out_dir / "shap_summary.png"
    plt.savefig(global_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved fallback global importance plot → %s", global_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Training curves (NN)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(out_dir: Path) -> None:
    """Plot NN training and validation loss curves from saved history."""
    hist_path = out_dir / "nn_train_history.pkl"
    if not hist_path.exists():
        logger.info("No NN training history found — skipping curve plot.")
        return

    with open(hist_path, "rb") as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCEWithLogitsLoss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, history["lr"], color="tab:orange", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = out_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", curves_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Audit report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_audit_report(
    xgb_results: dict,
    nn_results: dict,
    cfg: dict,
    out_dir: Path,
    metadata: dict,
    thresholds_payload: dict | None = None,
) -> None:
    """
    Generate a comprehensive Markdown audit report summarising the full
    pipeline: data, preprocessing, hyperparameters, and final model
    comparison.
    """
    target_cols = metadata.get("target_columns", [])
    report_lines = []

    def _add(line: str = ""):
        report_lines.append(line)

    _add("# Track 1 — Audit Report: Diabetic Complication Detection")
    _add()
    _add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _add()

    # ── Dataset summary ─────────────────────────────────────────────────
    _add("## 1. Dataset Summary")
    _add()
    _add(f"- **Raw shape:** {metadata.get('raw_shape', 'N/A')}")
    _add(f"- **Train split:** {metadata.get('train_shape', 'N/A')}")
    _add(f"- **Test split:** {metadata.get('test_shape', 'N/A')}")
    _add(f"- **Numerical features:** {metadata.get('n_numerical', 'N/A')}")
    _add(f"- **Categorical features:** {metadata.get('n_categorical', 'N/A')}")
    _add(f"- **Active target set:** {metadata.get('active_target_set', 'N/A')}")
    _add(f"- **Primary targets:** {metadata.get('primary_target_columns', [])}")
    _add(f"- **Exploratory targets:** {metadata.get('exploratory_target_columns', [])}")
    _add()

    # ── Preprocessing ───────────────────────────────────────────────────
    _add("## 2. Preprocessing Strategy")
    _add()
    _add(f"- **Blank handling:** {metadata.get('blank_handling', 'trim + empty-string to NaN')}")
    _add(f"- **Unknown-code(9) handling:** {len(metadata.get('unknown_9_columns', []))} categorical columns marked as missing")
    _add(f"- **Minimum positives per target (filter):** {metadata.get('min_target_positives', 'N/A')}")
    _add(f"- **KNN Imputer (numerical):** k = {metadata.get('knn_k', 5)}")
    _add("- **Mode Imputer (categorical):** strategy = most_frequent")
    _add("- **Numerical scaling:** StandardScaler")
    _add("- **Categorical encoding:** OrdinalEncoder")
    _add()

    dropped = metadata.get("dropped_columns", [])
    if dropped:
        _add("### Dropped Columns")
        _add()
        for c in dropped:
            _add(f"- `{c}`")
        _add()

    dropped_rows = metadata.get("dropped_rows_incomplete_targets", 0)
    _add(f"- **Rows removed due to missing-target policy:** {dropped_rows}")
    _add(f"- **Target missing policy:** {metadata.get('target_missing_policy', 'complete_case')}")
    dropped_low = metadata.get("dropped_low_support_targets", [])
    dropped_split = metadata.get("dropped_zero_split_targets", [])
    if dropped_low:
        _add(f"- **Dropped low-support targets:** {dropped_low}")
    if dropped_split:
        _add(f"- **Dropped zero-support split targets:** {dropped_split}")
    applied_drops = metadata.get("applied_feature_drops", [])
    if applied_drops:
        _add(f"- **Applied target-specific leakage drops:** {applied_drops}")
    _add()

    # ── Target distribution ─────────────────────────────────────────────
    _add("## 3. Target Positive Rates (after binarisation)")
    _add()
    _add("| Target | Positive Rate |")
    _add("|--------|--------------|")
    for col, rate in metadata.get("target_positive_rates", {}).items():
        _add(f"| {col} | {rate:.4f} |")
    _add()

    _add("## 4. Target Support")
    _add()
    _add("| Target | Labeled | Positive | Negative | Missing | Train Pos | Test Pos |")
    _add("|--------|---------|----------|----------|---------|-----------|----------|")
    for col, support in metadata.get("target_support", {}).items():
        _add(
            f"| {col} | {support.get('labeled_rows', '-')} | {support.get('positive_rows', '-')} | "
            f"{support.get('negative_rows', '-')} | {support.get('missing_rows', '-')} | "
            f"{support.get('train_positive_rows', '-')} | {support.get('test_positive_rows', '-')} |"
        )
    _add()

    # ── Hyperparameters ─────────────────────────────────────────────────
    _add("## 5. Hyperparameters")
    _add()
    _add("### XGBoost")
    _add()
    xgb_cfg = cfg.get("xgboost", {})
    for k, v in xgb_cfg.items():
        _add(f"- `{k}`: {v}")
    _add()

    _add("### Neural Network")
    _add()
    nn_cfg = cfg.get("neural_network", {})
    for k, v in nn_cfg.items():
        if k != "tabnet":
            _add(f"- `{k}`: {v}")
    _add()

    # ── Performance comparison ──────────────────────────────────────────
    _add("## 6. Performance Comparison (Test Set)")
    _add()
    _add("### Per-Complication Metrics")
    _add()
    _add("| Target | Model | ROC-AUC | Precision | Recall | F1-Score | Threshold |")
    _add("|--------|-------|---------|-----------|--------|----------|-----------|")

    all_targets = [t for t in target_cols if t in xgb_results] + ["MACRO-AVG"]
    for target in all_targets:
        xm = xgb_results.get(target, {})
        nm = nn_results.get(target, {})
        _add(f"| {target} | XGBoost | {xm.get('ROC-AUC', '-')} | {xm.get('Precision', '-')} | {xm.get('Recall', '-')} | {xm.get('F1-Score', '-')} | {xm.get('Threshold', '-')} |")
        _add(f"| {target} | NN      | {nm.get('ROC-AUC', '-')} | {nm.get('Precision', '-')} | {nm.get('Recall', '-')} | {nm.get('F1-Score', '-')} | {nm.get('Threshold', '-')} |")
    _add()

    if thresholds_payload:
        _add("### Threshold Calibration")
        _add()
        x_mode = thresholds_payload.get("xgb", {}).get("mode", "fixed")
        n_mode = thresholds_payload.get("nn", {}).get("mode", "fixed")
        _add(f"- **XGBoost threshold mode:** {x_mode}")
        _add(f"- **NN threshold mode:** {n_mode}")
        _add("- Thresholds are tuned only on validation split and then applied to the test set.")
        _add()

    # ── Winner summary ──────────────────────────────────────────────────
    _add("### Summary")
    _add()
    xgb_macro = xgb_results.get("MACRO-AVG", {})
    nn_macro = nn_results.get("MACRO-AVG", {})
    xgb_f1 = xgb_macro.get("F1-Score", 0)
    nn_f1 = nn_macro.get("F1-Score", 0)
    winner = "XGBoost" if xgb_f1 >= nn_f1 else "Neural Network"
    _add(f"- **Best macro-F1 model:** {winner}")
    _add(f"  - XGBoost macro-F1: {xgb_f1}")
    _add(f"  - NN macro-F1: {nn_f1}")
    _add()

    # ── SHAP ────────────────────────────────────────────────────────────
    _add("## 7. Interpretability")
    _add()
    _add("XGBoost contribution-based attribution plots have been saved to the `outputs/` directory:")
    _add()
    _add("- `shap_summary.png` — Global feature importance (averaged across successful targets)")
    for col in target_cols:
        _add(f"- `shap_summary_{col}.png` — Feature importance for {col}")
    _add()

    # ── Reproducibility ─────────────────────────────────────────────────
    _add("## 8. Reproducibility")
    _add()
    _add(f"- **Random seed:** {SEED}")
    _add("- **Python packages:** see `requirements.txt`")
    _add("- **All scripts are CLI-driven:** `python src/<module>.py --config config.yaml`")
    _add()
    _add("---")
    _add("*End of audit report.*")

    # Write the report
    report_path = out_dir / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("Saved audit report → %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all(cfg: dict) -> None:
    """Execute the full evaluation pipeline."""

    paths_cfg = cfg["paths"]
    eval_cfg = cfg.get("evaluation", {})
    tune_thresholds = bool(eval_cfg.get("tune_thresholds", True))
    proc_dir = Path(paths_cfg["processed_dir"])
    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load test data ──────────────────────────────────────────────────
    with open(proc_dir / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(proc_dir / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    with open(proc_dir / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(proc_dir / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(proc_dir / "target_columns.pkl", "rb") as f:
        target_cols = pickle.load(f)
    with open(proc_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open(proc_dir / "preprocessing_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    X_test_np = np.asarray(X_test, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)
    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)

    # Build validation slice from train split for threshold tuning
    _, val_idx = build_validation_indices(y_train_np, val_fraction=0.15)
    X_val_np = X_train_np[val_idx]
    y_val_np = y_train_np[val_idx]

    logger.info("Test set loaded — X: %s, y: %s", X_test_np.shape, y_test_np.shape)

    # ── Evaluate XGBoost ────────────────────────────────────────────────
    xgb_model_path = out_dir / "xgb_model.pkl"
    xgb_results = {}
    xgb_thresholds = {c: 0.5 for c in target_cols}
    if xgb_model_path.exists():
        with open(xgb_model_path, "rb") as f:
            xgb_bundle = pickle.load(f)
        xgb_target_cols = xgb_bundle.get("target_columns", target_cols)
        if tune_thresholds:
            xgb_val_probs = _predict_xgb(xgb_bundle, X_val_np)
            y_val_aligned, xgb_val_probs_aligned, xgb_cols_aligned = align_targets(
                target_cols, y_val_np, xgb_target_cols, xgb_val_probs
            )
            xgb_thresholds = find_best_thresholds(y_val_aligned, xgb_val_probs_aligned, xgb_cols_aligned)
        xgb_probs = _predict_xgb(xgb_bundle, X_test_np)
        y_test_aligned, xgb_probs_aligned, xgb_cols_aligned = align_targets(
            target_cols, y_test_np, xgb_target_cols, xgb_probs
        )
        xgb_results = compute_metrics(
            y_test_aligned,
            xgb_probs_aligned,
            xgb_cols_aligned,
            thresholds_by_target=xgb_thresholds,
        )
        print_metrics_table(xgb_results, "XGBoost")
    else:
        logger.warning("XGBoost model not found at '%s' — skipping.", xgb_model_path)

    # ── Evaluate Neural Network ─────────────────────────────────────────
    nn_bundle_path = out_dir / "nn_model_bundle.pkl"
    nn_results = {}
    nn_thresholds = {c: 0.5 for c in target_cols}
    if nn_bundle_path.exists():
        with open(nn_bundle_path, "rb") as f:
            nn_bundle = pickle.load(f)

        nn_target_cols = nn_bundle.get("target_columns", target_cols)

        if nn_bundle["model_type"] == "mlp":
            model = _load_mlp_model(nn_bundle, device)
            if tune_thresholds:
                nn_val_probs = _predict_mlp(model, X_val_np, device)
                y_val_aligned, nn_val_probs_aligned, nn_cols_aligned = align_targets(
                    target_cols, y_val_np, nn_target_cols, nn_val_probs
                )
                nn_thresholds = find_best_thresholds(y_val_aligned, nn_val_probs_aligned, nn_cols_aligned)
            nn_probs = _predict_mlp(model, X_test_np, device)
        elif nn_bundle["model_type"] == "tabnet":
            # TabNet: predict per-target and stack
            estimators = nn_bundle["estimators"]
            nn_val_probs = np.zeros((X_val_np.shape[0], len(estimators)), dtype=np.float32)
            nn_probs = np.zeros((X_test_np.shape[0], len(estimators)), dtype=np.float32)
            for idx, clf in enumerate(estimators):
                val_preds = clf.predict_proba(X_val_np)
                if val_preds.ndim == 1:
                    nn_val_probs[:, idx] = val_preds
                elif val_preds.shape[1] >= 2:
                    nn_val_probs[:, idx] = val_preds[:, 1]
                else:
                    nn_val_probs[:, idx] = val_preds[:, 0]
                preds = clf.predict_proba(X_test_np)
                if preds.ndim == 1:
                    nn_probs[:, idx] = preds
                elif preds.shape[1] >= 2:
                    nn_probs[:, idx] = preds[:, 1]
                else:
                    nn_probs[:, idx] = preds[:, 0]
            if tune_thresholds:
                y_val_aligned, nn_val_probs_aligned, nn_cols_aligned = align_targets(
                    target_cols, y_val_np, nn_target_cols, nn_val_probs
                )
                nn_thresholds = find_best_thresholds(y_val_aligned, nn_val_probs_aligned, nn_cols_aligned)
        else:
            logger.warning("Unknown NN model type: '%s'.", nn_bundle["model_type"])
            nn_probs = np.zeros_like(y_test_np)

        y_test_aligned, nn_probs_aligned, nn_cols_aligned = align_targets(
            target_cols, y_test_np, nn_target_cols, nn_probs
        )
        nn_results = compute_metrics(
            y_test_aligned,
            nn_probs_aligned,
            nn_cols_aligned,
            thresholds_by_target=nn_thresholds,
        )
        print_metrics_table(nn_results, "Neural Network")
    else:
        logger.warning("NN model bundle not found at '%s' — skipping.", nn_bundle_path)

    # ── SHAP interpretability (XGBoost only) ────────────────────────────
    if xgb_model_path.exists():
        shap_target_cols = xgb_bundle.get("target_columns", target_cols)
        generate_shap_plots(
            bundle=xgb_bundle,
            X_test=X_test_np,
            feature_names=feature_names,
            target_cols=shap_target_cols,
            out_dir=out_dir,
            max_display=eval_cfg.get("shap_max_display", 20),
            n_background=eval_cfg.get("shap_background_samples", 200),
        )

    # ── Training curves (NN) ────────────────────────────────────────────
    plot_training_curves(out_dir)

    # ── Audit report ────────────────────────────────────────────────────
    generate_audit_report(
        xgb_results=xgb_results,
        nn_results=nn_results,
        cfg=cfg,
        out_dir=out_dir,
        metadata=metadata,
        thresholds_payload={
            "xgb": {
                "mode": "per-target tuned on validation" if tune_thresholds else "fixed @0.5",
                "thresholds": xgb_thresholds,
            },
            "nn": {
                "mode": "per-target tuned on validation" if tune_thresholds else "fixed @0.5",
                "thresholds": nn_thresholds,
            },
        },
    )

    thresholds_path = out_dir / "decision_thresholds.json"
    thresholds_payload = {
        "mode": "per-target tuned on validation" if tune_thresholds else "fixed @0.5",
        "xgb": xgb_thresholds,
        "nn": nn_thresholds,
    }
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)
    logger.info("Saved calibrated thresholds → %s", thresholds_path)

    logger.info("✅  Evaluation complete. Check the 'outputs/' directory.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models and generate audit report for Track 1."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_all(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_xgb.py — Tree-Based Multi-Label Model Training (XGBoost)
==============================================================

Track 1: AI-Based Diabetic Complication Detection Using Clinical Data

This script loads the preprocessed train/test splits and trains a
``MultiOutputClassifier`` wrapping ``xgb.XGBClassifier`` — one binary
classifier per target complication.  Class imbalance is handled via
``scale_pos_weight``.  Early stopping is applied using a held-out
validation fraction.

Usage
-----
    python src/models/train_xgb.py --config config.yaml

Outputs
-------
    outputs/xgb_model.pkl          — serialised MultiOutputClassifier
    outputs/xgb_train_log.pkl      — per-target training metadata
"""

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import logging
import pickle
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import yaml
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Read the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_processed_data(processed_dir: str) -> tuple:
    """Load the four pickle artefacts produced by ``preprocess.py``."""
    proc = Path(processed_dir)
    artefacts = {}
    for name in ("X_train", "X_test", "y_train", "y_test", "target_columns", "feature_names"):
        fp = proc / f"{name}.pkl"
        with open(fp, "rb") as f:
            artefacts[name] = pickle.load(f)
    return (
        artefacts["X_train"],
        artefacts["X_test"],
        artefacts["y_train"],
        artefacts["y_test"],
        artefacts["target_columns"],
        artefacts["feature_names"],
    )


def compute_scale_pos_weight(y_col: np.ndarray) -> float:
    """
    Compute ``scale_pos_weight = n_negative / n_positive`` to counteract
    class imbalance in a binary target column.

    Returns 1.0 if there are no positive samples (safety guard).
    """
    n_pos = int(np.sum(y_col == 1))
    n_neg = int(np.sum(y_col == 0))
    if n_pos == 0:
        logger.warning("No positive samples — scale_pos_weight set to 1.0")
        return 1.0
    return n_neg / n_pos


def _build_binary_arrays_for_target(y_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels_binary, labeled_mask) for partially-labeled target column."""
    labeled_mask = ~np.isnan(y_col)
    labels = y_col[labeled_mask].astype(np.int32)
    return labels, labeled_mask


def _build_stratify_labels_np(y: np.ndarray) -> np.ndarray | None:
    """Build stable stratification labels for multilabel validation split."""
    y_int = np.nan_to_num(y, nan=-1).astype(np.int32)
    labels = np.array(["_".join(map(str, row.tolist())) for row in y_int], dtype=object)
    uniq, counts = np.unique(labels, return_counts=True)
    if counts.size > 0 and counts.min() >= 2:
        return labels
    logger.warning(
        "Rare multilabel combinations in train split; falling back to first target stratification."
    )
    fallback = y_int[:, 0]
    _, fallback_counts = np.unique(fallback, return_counts=True)
    if fallback_counts.size > 0 and fallback_counts.min() >= 2:
        return fallback
    logger.warning("Could not stratify validation split safely; using random split.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost(cfg: dict) -> None:
    """
    Train one XGBClassifier per target complication, wrapped inside
    ``MultiOutputClassifier``.

    Strategy
    --------
    1. Split a 15 % validation set from the training data for early stopping.
    2. Compute ``scale_pos_weight`` per target.
    3. Train with ``early_stopping_rounds`` on the validation loss.
    4. Save the combined model and a training log.
    """
    xgb_cfg = cfg["xgboost"]
    paths_cfg = cfg["paths"]

    # ── Load data ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, target_cols, feature_names = load_processed_data(
        paths_cfg["processed_dir"]
    )
    logger.info("Loaded processed data — X_train: %s, y_train: %s", X_train.shape, y_train.shape)

    # Convert DataFrames → numpy for XGBoost
    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)

    # ── Validation split for early stopping ─────────────────────────────
    strat_labels = _build_stratify_labels_np(y_train_np)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np,
        test_size=0.15,
        random_state=SEED,
        stratify=strat_labels,
    )
    logger.info("Train/Val split → train: %d, val: %d", X_tr.shape[0], X_val.shape[0])

    # ── Train one model per target ──────────────────────────────────────
    estimators = []
    trained_target_cols = []
    train_log = {}

    for idx, col_name in enumerate(target_cols):
        y_tr_col_all = y_tr[:, idx]
        y_val_col_all = y_val[:, idx]

        y_tr_col, train_mask = _build_binary_arrays_for_target(y_tr_col_all)
        y_val_col, val_mask = _build_binary_arrays_for_target(y_val_col_all)
        X_tr_col = X_tr[train_mask]
        X_val_col = X_val[val_mask]

        if len(np.unique(y_tr_col)) < 2:
            logger.warning("[%s] Only one class in labeled train rows; skipping target.", col_name)
            continue

        if len(y_val_col) == 0 or len(np.unique(y_val_col)) < 2:
            logger.warning("[%s] Validation has <2 classes; training without early stopping.", col_name)
            use_eval = False
        else:
            use_eval = True

        # Scale pos weight
        spw = compute_scale_pos_weight(y_tr_col) if xgb_cfg["use_scale_pos_weight"] else 1.0
        logger.info(
            "[%s] Positive rate: %.4f | scale_pos_weight: %.2f",
            col_name,
            float(y_tr_col.mean()) if len(y_tr_col) else 0.0,
            spw,
        )

        clf = xgb.XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            min_child_weight=xgb_cfg["min_child_weight"],
            gamma=xgb_cfg["gamma"],
            reg_alpha=xgb_cfg["reg_alpha"],
            reg_lambda=xgb_cfg["reg_lambda"],
            scale_pos_weight=spw,
            eval_metric=xgb_cfg["eval_metric"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            random_state=SEED,
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

        if use_eval:
            clf.fit(
                X_tr_col, y_tr_col,
                eval_set=[(X_val_col, y_val_col)],
                verbose=False,
            )
            best_iter = clf.best_iteration
            best_score = clf.best_score
        else:
            clf.set_params(early_stopping_rounds=None)
            clf.fit(X_tr_col, y_tr_col, verbose=False)
            best_iter = clf.get_booster().best_iteration if hasattr(clf, "get_booster") else -1
            best_score = float("nan")
        logger.info(
            "[%s] Training complete — best_iteration: %d, best_val_%s: %.5f",
            col_name, best_iter, xgb_cfg["eval_metric"], best_score,
        )

        estimators.append(clf)
        trained_target_cols.append(col_name)
        train_log[col_name] = {
            "best_iteration": best_iter,
            "best_score": best_score,
            "scale_pos_weight": spw,
            "positive_rate": float(y_tr_col.mean()) if len(y_tr_col) else 0.0,
            "n_train_labeled": int(train_mask.sum()),
            "n_val_labeled": int(val_mask.sum()),
        }

    if not estimators:
        raise ValueError("No XGBoost targets were trainable after label-support checks.")

    # ── Wrap into MultiOutputClassifier-like container ──────────────────
    # We store them as a simple list (sklearn's MOC stores them similarly)
    model_bundle = {
        "estimators": estimators,
        "target_columns": trained_target_cols,
        "feature_names": feature_names,
        "model_type": "xgboost_multi_label",
    }

    # ── Save artefacts ──────────────────────────────────────────────────
    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved XGBoost model bundle → %s", model_path)

    log_path = out_dir / "xgb_train_log.pkl"
    with open(log_path, "wb") as f:
        pickle.dump(train_log, f)
    logger.info("Saved training log → %s", log_path)

    logger.info("✅  XGBoost training complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multi-label XGBoost model for Track 1."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_xgboost(cfg)


if __name__ == "__main__":
    main()

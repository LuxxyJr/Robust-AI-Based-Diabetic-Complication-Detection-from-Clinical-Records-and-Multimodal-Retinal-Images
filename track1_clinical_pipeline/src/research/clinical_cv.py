#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fold-safe repeated cross-validation for Track 1 DN prediction.

This runner starts from the raw clinical CSV and fits preprocessing inside
each fold. It is intentionally separate from the existing single-split
pipeline so the older outputs remain comparable while the research-grade
validation artifacts are generated in ``outputs/research_cv``.

Examples
--------
    python src/research/clinical_cv.py --config config.yaml
    python src/research/clinical_cv.py --config config.yaml --run-ablations
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

for env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at import time
    plt = None

LOGGER = logging.getLogger("clinical_cv")


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    numerical_cols: list[str]
    categorical_cols: list[str]
    leakage_audit: pd.DataFrame
    target_support: dict[str, Any]


class ConstantProbabilityClassifier(BaseEstimator, ClassifierMixin):
    """Fallback estimator for rare cases where a fold has one class only."""

    def __init__(self, positive_probability: float = 0.0):
        self.positive_probability = float(positive_probability)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        if len(y):
            self.positive_probability = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        pos = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class NumpyLogisticClassifier(BaseEstimator, ClassifierMixin):
    """Small weighted logistic-regression baseline implemented in NumPy."""

    def __init__(self, learning_rate: float = 0.05, max_iter: int = 800, l2: float = 1e-3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        LOGGER.info("numpy_logistic: converting matrix")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        LOGGER.info("numpy_logistic: matrix shape=%s x %s", n, p)
        Xb = np.column_stack([np.ones(n), X])
        self.coef_ = np.zeros(p + 1, dtype=np.float64)
        n_pos = max(float(np.sum(y == 1)), 1.0)
        n_neg = max(float(np.sum(y == 0)), 1.0)
        weights = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
        weights = weights / np.mean(weights)

        LOGGER.info("numpy_logistic: starting gradient loop")
        for _ in range(int(self.max_iter)):
            logits = np.clip(Xb @ self.coef_, -40, 40)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = (probs - y) * weights
            grad = (Xb.T @ error) / n
            grad[1:] += self.l2 * self.coef_[1:]
            self.coef_ -= self.learning_rate * grad
        LOGGER.info("numpy_logistic: finished gradient loop")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        logits = np.clip(Xb @ self.coef_, -40, 40)
        pos = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LoopGaussianNBClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes implemented with conservative Python loops."""

    def __init__(self, var_smoothing: float = 1e-6):
        self.var_smoothing = var_smoothing
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)
        self.n_features_in_ = int(X_arr.shape[1])
        self.theta_ = np.zeros((2, self.n_features_in_), dtype=np.float64)
        self.var_ = np.ones((2, self.n_features_in_), dtype=np.float64)
        self.class_prior_ = np.zeros(2, dtype=np.float64)
        for cls in (0, 1):
            rows = X_arr[y_arr == cls]
            self.class_prior_[cls] = max(float(len(rows)) / max(len(X_arr), 1), 1e-6)
            if len(rows) == 0:
                continue
            for j in range(self.n_features_in_):
                col = rows[:, j]
                mean = float(np.mean(col))
                var = float(np.var(col)) + self.var_smoothing
                self.theta_[cls, j] = mean
                self.var_[cls, j] = max(var, self.var_smoothing)
        self.class_prior_ = self.class_prior_ / self.class_prior_.sum()
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        probs = []
        for row in X_arr:
            logps = []
            for cls in (0, 1):
                logp = math.log(max(float(self.class_prior_[cls]), 1e-12))
                for j, value in enumerate(row):
                    var = float(self.var_[cls, j])
                    diff = float(value) - float(self.theta_[cls, j])
                    logp += -0.5 * (math.log(2.0 * math.pi * var) + (diff * diff / var))
                logps.append(logp)
            m = max(logps)
            exp0 = math.exp(logps[0] - m)
            exp1 = math.exp(logps[1] - m)
            denom = exp0 + exp1
            probs.append([exp0 / denom, exp1 / denom])
        return np.asarray(probs, dtype=np.float64)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def trim_and_replace_blanks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(
            lambda v: np.nan
            if pd.isna(v) or str(v).replace("\xa0", " ").strip().lower() in {"", "na", "nan", "none", "null"}
            else str(v).replace("\xa0", " ").strip()
        )
    return out


def resolve_active_targets(data_cfg: dict[str, Any]) -> list[str]:
    mode = str(data_cfg.get("active_target_set", "primary")).lower()
    mapping = {
        "primary": data_cfg.get("primary_target_columns", []),
        "exploratory": data_cfg.get("exploratory_target_columns", []),
        "all": data_cfg.get("target_columns", []),
        "custom": data_cfg.get("custom_target_columns", []),
    }
    targets = list(mapping.get(mode, []))
    if not targets:
        raise ValueError(f"No targets configured for active_target_set={mode!r}.")
    return list(dict.fromkeys(targets))


def prepare_dn_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target != "DN":
        series = pd.to_numeric(df[target], errors="coerce").replace(9, np.nan)
        return series.astype(float)
    series = pd.to_numeric(df[target], errors="coerce")
    return series.map(lambda v: 0.0 if v == 0 else (1.0 if v in (3, 4, 5) else np.nan))


def build_leakage_audit(data_cfg: dict[str, Any], target: str, available_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    always = list(data_cfg.get("always_drop_feature_columns", []))
    target_specific = list((data_cfg.get("target_specific_feature_drops", {}) or {}).get(target, []))
    target_cols = list(data_cfg.get("target_columns", []))
    dr_original = data_cfg.get("dr_original_column", "DR")

    for col in target_cols + [dr_original]:
        if col in available_columns:
            rows.append(
                {
                    "column": col,
                    "group": "target_or_complication_label",
                    "reason": "Outcome or complication label; not available as a baseline predictor.",
                }
            )
    for col in always:
        if col in available_columns:
            rows.append(
                {
                    "column": col,
                    "group": "global_direct_proxy",
                    "reason": "Configured direct diagnosis, eye-study, or dataset-processing proxy.",
                }
            )
    for col in target_specific:
        if col in available_columns:
            rows.append(
                {
                    "column": col,
                    "group": f"{target}_specific_proxy",
                    "reason": "Kidney-function or albuminuria proxy that can directly encode nephropathy status.",
                }
            )
    audit = pd.DataFrame(rows).drop_duplicates("column").sort_values(["group", "column"])
    return audit.reset_index(drop=True)


def infer_feature_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    cast = df.copy()
    numerical: list[str] = []
    categorical: list[str] = []
    for col in cast.columns:
        raw = cast[col]
        non_missing = int(raw.notna().sum())
        numeric = pd.to_numeric(raw, errors="coerce")
        rate = float(numeric.notna().sum() / non_missing) if non_missing else 0.0
        if rate >= 0.85:
            cast[col] = numeric
            if int(numeric.nunique(dropna=True)) > 15:
                numerical.append(col)
            else:
                categorical.append(col)
        else:
            cast[col] = raw.astype("string")
            categorical.append(col)
    return cast, numerical, categorical


def mark_unknown_code_9(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        numeric = pd.to_numeric(out[col], errors="coerce")
        non_na = numeric.dropna()
        if non_na.empty or not (numeric == 9).any():
            continue
        unique = np.unique(non_na.values)
        if unique.min() >= 0 and unique.max() <= 9 and len(unique) <= 10:
            out[col] = out[col].mask(numeric == 9, np.nan)
    return out


def load_research_dataset(cfg: dict[str, Any], target: str = "DN", feature_mode: str = "strict") -> DatasetBundle:
    data_cfg = cfg["data"]
    raw_path = Path(cfg["paths"]["raw_csv"])
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path

    df = pd.read_csv(raw_path, encoding=data_cfg.get("csv_encoding", "latin-1"), dtype="string", low_memory=False)
    raw_columns = list(df.columns)
    df = trim_and_replace_blanks(df)
    drop_cols = [c for c in data_cfg.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = prepare_dn_target(df, target)
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    leakage_audit = build_leakage_audit(data_cfg, target, raw_columns)
    feature_drop = [c for c in leakage_audit["column"].tolist() if c in df.columns]
    X = df.drop(columns=feature_drop, errors="ignore")

    X, numerical_cols, categorical_cols = infer_feature_types(X)
    X = mark_unknown_code_9(X, categorical_cols)

    max_missing = float(data_cfg.get("max_missing_fraction", 0.50))
    keep_cols = X.columns[X.isna().mean() <= max_missing].tolist()
    X = X[keep_cols]
    numerical_cols = [c for c in numerical_cols if c in keep_cols]
    categorical_cols = [c for c in categorical_cols if c in keep_cols]

    selected_cols = select_feature_group(feature_mode, numerical_cols, categorical_cols)
    if selected_cols is not None:
        X = X[selected_cols]
        numerical_cols = [c for c in numerical_cols if c in selected_cols]
        categorical_cols = [c for c in categorical_cols if c in selected_cols]

    support = {
        "target": target,
        "rows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "feature_mode": feature_mode,
        "n_features": int(X.shape[1]),
        "n_numerical": int(len(numerical_cols)),
        "n_categorical": int(len(categorical_cols)),
    }
    return DatasetBundle(X, y, numerical_cols, categorical_cols, leakage_audit, support)


def select_feature_group(mode: str, numerical_cols: list[str], categorical_cols: list[str]) -> list[str] | None:
    mode = mode.lower()
    if mode in {"strict", "full", "full_strict"}:
        return None
    lab_keywords = {
        "glu",
        "hba",
        "hb",
        "chol",
        "hdl",
        "ldl",
        "tg",
        "alt",
        "ast",
        "alp",
        "ggt",
        "wbc",
        "rbc",
        "plt",
        "ua",
        "bun",
        "cre",
        "cr",
        "alb",
        "tp",
        "ca",
        "k",
        "na",
        "cl",
    }
    demo_history_keywords = {
        "age",
        "sex",
        "gender",
        "bmi",
        "weight",
        "height",
        "smok",
        "drink",
        "diet",
        "exercise",
        "history",
        "his",
        "family",
        "duration",
        "year",
    }
    clinical_history_keywords = {
        "his",
        "history",
        "hypertension",
        "hyperten",
        "pad",
        "mi",
        "hf",
        "blind",
        "cerebral",
        "amputation",
        "disease",
        "therapy",
        "treat",
        "drug",
        "insulin",
    }

    def has_keyword(col: str, keywords: set[str]) -> bool:
        name = col.lower()
        return any(k in name for k in keywords)

    all_cols = numerical_cols + categorical_cols
    if mode == "lab_only":
        return [c for c in numerical_cols if has_keyword(c, lab_keywords)]
    if mode == "demographics_history_lifestyle":
        return [c for c in all_cols if has_keyword(c, demo_history_keywords)]
    if mode == "clinical_history_only":
        return [c for c in all_cols if has_keyword(c, clinical_history_keywords)]
    if mode.startswith("without_"):
        group = mode.replace("without_", "", 1)
        excluded = set(select_feature_group(group, numerical_cols, categorical_cols) or [])
        return [c for c in all_cols if c not in excluded]
    raise ValueError(f"Unknown feature mode: {mode}")


def build_preprocessor(numerical_cols: list[str], categorical_cols: list[str], knn_k: int, imputer: str = "simple") -> ColumnTransformer:
    if imputer == "knn":
        numeric_imputer = KNNImputer(n_neighbors=knn_k)
    else:
        numeric_imputer = SimpleImputer(strategy="median")
    numeric = Pipeline(
        steps=[
            ("imputer", numeric_imputer),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numerical_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def build_models(
    cfg: dict[str, Any],
    seed: int,
    model_names: list[str] | None = None,
    max_estimators: int | None = None,
) -> dict[str, ClassifierMixin]:
    available: dict[str, ClassifierMixin] = {
        "loop_gaussian_nb": LoopGaussianNBClassifier(var_smoothing=1e-4),
        "numpy_logistic": NumpyLogisticClassifier(
            learning_rate=0.05,
            max_iter=50,
            l2=1e-3,
        ),
        "logistic_regression": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100 if max_estimators is None else max_estimators,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100 if max_estimators is None else max_estimators,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=tuple(cfg.get("neural_network", {}).get("hidden_dims", [128, 64])),
            alpha=float(cfg.get("neural_network", {}).get("weight_decay", 1e-4)),
            learning_rate_init=float(cfg.get("neural_network", {}).get("learning_rate", 1e-3)),
            batch_size=int(cfg.get("neural_network", {}).get("batch_size", 256)),
            max_iter=min(int(cfg.get("neural_network", {}).get("epochs", 150)), 250),
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        ),
    }
    xgb_cfg = cfg.get("xgboost", {})
    requested_set = set(model_names or [])
    requested_xgb = model_names is None or "xgboost" in requested_set
    requested_lgb = model_names is None or "lightgbm" in requested_set

    if requested_xgb:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover
            XGBClassifier = None
            LOGGER.warning("xgboost is unavailable or failed to import: %s", exc)
    else:
        XGBClassifier = None

    if XGBClassifier is not None:
        xgb_estimators = int(xgb_cfg.get("n_estimators", 500))
        if max_estimators is not None:
            xgb_estimators = min(xgb_estimators, max_estimators)
        available["xgboost"] = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            min_child_weight=float(xgb_cfg.get("min_child_weight", 3)),
            gamma=float(xgb_cfg.get("gamma", 0.1)),
            reg_alpha=float(xgb_cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if requested_lgb:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover
            LGBMClassifier = None
            LOGGER.warning("lightgbm is unavailable or failed to import: %s", exc)
    else:
        LGBMClassifier = None

    if LGBMClassifier is not None:
        lgb_estimators = 600 if max_estimators is None else min(600, max_estimators)
        available["lightgbm"] = LGBMClassifier(
            n_estimators=lgb_estimators,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_names is None:
        return available
    requested = []
    for name in model_names:
        name = name.strip()
        if name:
            requested.append(name)
    aliases = {
        "logistic_elasticnet": "sgd_logistic",
        "logistic_regression": "sgd_logistic",
        "logistic": "sgd_logistic",
    }
    requested = [aliases.get(name, name) for name in requested]
    missing = [name for name in requested if name not in available]
    if missing:
        LOGGER.warning("Skipping unavailable/requested models: %s", ", ".join(missing))
    selected = {name: available[name] for name in requested if name in available}
    if not selected:
        raise ValueError("No requested models are available in the current environment.")
    return selected


def positive_probability(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if isinstance(proba, list):
        proba = proba[0]
    if np.ndim(proba) == 1:
        return np.asarray(proba, dtype=float)
    if proba.shape[1] == 1:
        return np.zeros(proba.shape[0], dtype=float)
    return np.asarray(proba[:, 1], dtype=float)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y_true, y_prob >= thr, zero_division=0) for thr in grid]
    return float(grid[int(np.argmax(scores))])


def specificity_npv(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return float(specificity), float(npv)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Approximate calibration slope/intercept without fitting another classifier.

    The canonical estimate uses a logistic regression of the outcome on the
    prediction logit. In this Windows/conda environment that native solver has
    shown hard-crash behavior, so we use an ordinary least-squares approximation
    of y on logit(p) for reporting stability. Brier score and calibration curves
    remain the primary calibration evidence.
    """
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    try:
        slope, intercept = np.polyfit(logits, y_true.astype(float), deg=1)
        return float(slope), float(intercept)
    except Exception:
        return float("nan"), float("nan")


def compute_metric_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model: str,
    fold: int,
    repeat: int,
    probability_type: str,
    feature_mode: str,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    spec, npv = specificity_npv(y_true, y_pred)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "feature_mode": feature_mode,
        "model": model,
        "probability_type": probability_type,
        "repeat": repeat,
        "fold": fold,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "npv": npv,
        "brier": brier_score_loss(y_true, np.clip(y_prob, 0, 1)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "n_test": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int = 200) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_fold_metrics(metrics: pd.DataFrame, seed: int, n_bootstrap: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision_ppv",
        "recall_sensitivity",
        "specificity",
        "npv",
        "brier",
        "balanced_accuracy",
        "calibration_slope",
        "calibration_intercept",
    ]
    rows = []
    group_cols = ["feature_mode", "model", "probability_type"]
    for keys, group in metrics.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values, rng, n_bootstrap=n_bootstrap)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if np.sum(~np.isnan(values)) > 1 else 0.0
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        row["n_folds"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_mode", "probability_type", "roc_auc_mean"], ascending=[True, True, False])


def summarize_oof_predictions(predictions: pd.DataFrame, seed: int, n_bootstrap: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in predictions.groupby(["feature_mode", "model", "probability_type"]):
        LOGGER.info(
            "Summarising OOF predictions for feature_mode=%s model=%s probability_type=%s",
            keys[0],
            keys[1],
            keys[2],
        )
        y = group["y_true"].to_numpy(dtype=int)
        p = group["y_prob"].to_numpy(dtype=float)
        thr = tune_threshold(y, p)
        base = compute_metric_row(y, p, thr, keys[1], -1, -1, keys[2], keys[0])
        base.pop("repeat")
        base.pop("fold")
        for metric in ["roc_auc", "pr_auc", "f1", "precision_ppv", "recall_sensitivity", "specificity", "npv", "brier"]:
            boot = []
            idx = np.arange(len(group))
            for _ in range(n_bootstrap):
                sample = rng.choice(idx, size=len(idx), replace=True)
                if len(np.unique(y[sample])) < 2:
                    continue
                boot_thr = tune_threshold(y[sample], p[sample])
                boot.append(compute_metric_row(y[sample], p[sample], boot_thr, keys[1], -1, -1, keys[2], keys[0])[metric])
            if boot:
                base[f"{metric}_ci_low"] = float(np.percentile(boot, 2.5))
                base[f"{metric}_ci_high"] = float(np.percentile(boot, 97.5))
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["feature_mode", "probability_type", "roc_auc"], ascending=[True, True, False])


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    rows = []
    for pt in thresholds:
        pred = y_prob >= pt
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
        treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"threshold": float(pt), "net_benefit": float(net_benefit), "treat_all": float(treat_all), "treat_none": 0.0})
    return pd.DataFrame(rows)


def plot_calibration(predictions: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    for feature_mode, group in predictions.groupby("feature_mode"):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        for (model, prob_type), sub in group.groupby(["model", "probability_type"]):
            y = sub["y_true"].to_numpy(dtype=int)
            p = sub["y_prob"].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                continue
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=f"{model} ({prob_type})")
        ax.set_title(f"Calibration Curve - {feature_mode}")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"calibration_{feature_mode}.png", dpi=180)
        plt.close(fig)


def plot_decision_curves(predictions: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    thresholds = np.arange(0.05, 0.51, 0.05)
    best_rows = []
    for feature_mode, group in predictions.groupby("feature_mode"):
        fig, ax = plt.subplots(figsize=(8, 7))
        for (model, prob_type), sub in group.groupby(["model", "probability_type"]):
            if prob_type != "calibrated":
                continue
            y = sub["y_true"].to_numpy(dtype=int)
            p = sub["y_prob"].to_numpy(dtype=float)
            dca = decision_curve(y, p, thresholds)
            dca["feature_mode"] = feature_mode
            dca["model"] = model
            dca["probability_type"] = prob_type
            best_rows.append(dca)
            ax.plot(dca["threshold"], dca["net_benefit"], marker="o", label=model)
        if best_rows:
            ref = best_rows[-1]
            ax.plot(ref["threshold"], ref["treat_all"], "k--", label="Treat all")
            ax.plot(ref["threshold"], ref["treat_none"], "k:", label="Treat none")
        ax.set_title(f"Decision Curve - {feature_mode}")
        ax.set_xlabel("Risk threshold")
        ax.set_ylabel("Net benefit")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"decision_curve_{feature_mode}.png", dpi=180)
        plt.close(fig)
    if best_rows:
        pd.concat(best_rows, ignore_index=True).to_csv(out_dir / "decision_curve_values.csv", index=False)


def make_prefit_calibrator(estimator: ClassifierMixin, method: str) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")


def fit_one_fold(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    repeat: int,
    fold: int,
    seed: int,
    feature_mode: str,
    model_names: list[str] | None,
    imputer: str,
    max_estimators: int | None,
    calibrate: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    X_train = bundle.X.iloc[train_idx]
    y_train = bundle.y.iloc[train_idx].to_numpy(dtype=int)
    X_test = bundle.X.iloc[test_idx]
    y_test = bundle.y.iloc[test_idx].to_numpy(dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fit_idx, cal_idx = next(splitter.split(X_train, y_train))
    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train[fit_idx]
    X_cal = X_train.iloc[cal_idx]
    y_cal = y_train[cal_idx]

    preprocessor = build_preprocessor(
        bundle.numerical_cols,
        bundle.categorical_cols,
        int(cfg.get("data", {}).get("knn_imputer_k", 5)),
        imputer=imputer,
    )
    X_fit_tx = preprocessor.fit_transform(X_fit, y_fit)
    X_cal_tx = preprocessor.transform(X_cal)
    X_test_tx = preprocessor.transform(X_test)

    models = build_models(cfg, seed, model_names=model_names, max_estimators=max_estimators)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in models.items():
        LOGGER.info(
            "Feature mode %s | repeat %d fold %d: fitting %s",
            feature_mode,
            repeat,
            fold,
            model_name,
        )
        if len(np.unique(y_fit)) < 2:
            fitted = ConstantProbabilityClassifier(float(np.mean(y_fit))).fit(X_fit_tx, y_fit)
        else:
            fitted = clone(estimator)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                fitted.fit(X_fit_tx, y_fit)

        raw_cal = positive_probability(fitted, X_cal_tx)
        raw_test = positive_probability(fitted, X_test_tx)
        raw_thr = tune_threshold(y_cal, raw_cal)
        metric_rows.append(compute_metric_row(y_test, raw_test, raw_thr, model_name, fold, repeat, "uncalibrated", feature_mode))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "probability_type": "uncalibrated",
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": test_idx,
                    "y_true": y_test,
                    "y_prob": raw_test,
                    "threshold": raw_thr,
                }
            )
        )

        if not calibrate:
            continue

        calibrated_test = None
        calibrated_cal = None
        if len(np.unique(y_cal)) == 2:
            method = "isotonic" if len(y_cal) >= 1000 else "sigmoid"
            try:
                calibrator = make_prefit_calibrator(fitted, method=method)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    calibrator.fit(X_cal_tx, y_cal)
                calibrated_cal = positive_probability(calibrator, X_cal_tx)
                calibrated_test = positive_probability(calibrator, X_test_tx)
            except Exception as exc:
                LOGGER.warning("%s calibration failed on repeat %d fold %d: %s", model_name, repeat, fold, exc)
        if calibrated_test is None:
            calibrated_cal = raw_cal
            calibrated_test = raw_test

        cal_thr = tune_threshold(y_cal, calibrated_cal)
        metric_rows.append(compute_metric_row(y_test, calibrated_test, cal_thr, model_name, fold, repeat, "calibrated", feature_mode))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "probability_type": "calibrated",
                    "repeat": repeat,
                    "fold": fold,
                    "sample_index": test_idx,
                    "y_true": y_test,
                    "y_prob": calibrated_test,
                    "threshold": cal_thr,
                }
            )
        )

    return metric_rows, prediction_frames


def paired_model_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_mode, prob_type), group in fold_metrics.groupby(["feature_mode", "probability_type"]):
        pivot = group.pivot_table(index=["repeat", "fold"], columns="model", values="roc_auc")
        if pivot.shape[1] < 2:
            continue
        ranked = pivot.mean().sort_values(ascending=False).index.tolist()
        best = ranked[0]
        for other in ranked[1:]:
            diff = pivot[best] - pivot[other]
            rows.append(
                {
                    "feature_mode": feature_mode,
                    "probability_type": prob_type,
                    "metric": "roc_auc",
                    "best_model": best,
                    "comparison_model": other,
                    "mean_difference": float(diff.mean()),
                    "sd_difference": float(diff.std(ddof=1)),
                    "ci_low": float(diff.quantile(0.025)),
                    "ci_high": float(diff.quantile(0.975)),
                    "n_paired_folds": int(diff.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    fold_summary: pd.DataFrame,
    oof_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    feature_modes: list[str],
    run_settings: dict[str, Any],
) -> None:
    def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
        if df.empty:
            return ""
        formatted = df.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else format(float(v), floatfmt))
        headers = list(formatted.columns)
        rows = formatted.astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Track 1 Research-Grade Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Study Design",
        "",
        "- Endpoint: DN, binarised as no nephropathy vs nephropathy stages 3/4/5.",
        f"- Validation: repeated stratified {run_settings['n_splits']}-fold cross-validation with {run_settings['n_repeats']} repeat(s).",
        "- Fold safety: imputation, scaling, categorical encoding, calibration, and threshold tuning are fit inside each training fold.",
        "- Thresholds: selected on the fold-internal calibration split by maximum F1, then applied to the held-out fold.",
        f"- Calibration enabled: {run_settings['calibrate']}.",
        f"- Numeric imputer: {run_settings['imputer']}.",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {bundle.target_support['rows']}",
        f"- DN positives: {bundle.target_support['positive']} ({bundle.target_support['positive_rate']:.4f})",
        f"- Features in primary run: {bundle.target_support['n_features']}",
        f"- Numerical/categorical: {bundle.target_support['n_numerical']} / {bundle.target_support['n_categorical']}",
        f"- Feature modes requested: {', '.join(feature_modes)}",
        "",
        "## Leakage Audit",
        "",
        "| Column | Group | Reason |",
        "|---|---|---|",
    ]
    for _, row in bundle.leakage_audit.iterrows():
        lines.append(f"| `{row['column']}` | {row['group']} | {row['reason']} |")
    lines.extend(["", "## Cross-Validation Summary", ""])
    if not fold_summary.empty:
        compact_cols = [
            "feature_mode",
            "model",
            "probability_type",
            "roc_auc_mean",
            "roc_auc_sd",
            "pr_auc_mean",
            "pr_auc_sd",
            "f1_mean",
            "f1_sd",
            "brier_mean",
            "brier_sd",
        ]
        lines.append(markdown_table(fold_summary[compact_cols]))
    lines.extend(["", "## Out-of-Fold Prediction Summary", ""])
    if not oof_summary.empty:
        compact_cols = ["feature_mode", "model", "probability_type", "roc_auc", "pr_auc", "f1", "precision_ppv", "recall_sensitivity", "specificity", "npv", "brier"]
        lines.append(markdown_table(oof_summary[compact_cols]))
    lines.extend(["", "## Paired Model Comparison", ""])
    if comparisons.empty:
        lines.append("Not enough model results were available for paired comparison.")
    else:
        lines.append(markdown_table(comparisons))
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `fold_metrics.csv`: per-fold metrics for uncalibrated and calibrated probabilities.",
            "- `fold_predictions.csv`: held-out fold predictions for reproducible table regeneration.",
            "- `cv_summary.csv`: mean, SD, and bootstrap CI summary from fold metrics.",
            "- `oof_summary.csv`: aggregate out-of-fold metrics with bootstrap CIs.",
            "- `model_comparisons.csv`: paired fold-level model comparisons.",
            "- `calibration_*.png`: reliability curves.",
            "- `decision_curve_*.png`: net-benefit curves for calibrated models.",
        ]
    )
    (out_dir / "research_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_cv(
    cfg: dict[str, Any],
    out_dir_override: str | None = None,
    n_bootstrap: int = 200,
    feature_modes: list[str] | None = None,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    imputer: str = "simple",
    skip_plots: bool = False,
    max_estimators: int | None = None,
    calibrate: bool = True,
) -> None:
    seed = int(cfg.get("data", {}).get("random_seed", 42))
    out_dir = Path(out_dir_override or Path(cfg["paths"]["output_dir"]) / "research_cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    if feature_modes is None:
        feature_modes = ["strict"]

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    primary_bundle: DatasetBundle | None = None

    for feature_mode in feature_modes:
        bundle = load_research_dataset(cfg, target="DN", feature_mode=feature_mode)
        LOGGER.info(
            "Starting feature mode %s with %d rows and %d features (%d numerical, %d categorical).",
            feature_mode,
            len(bundle.y),
            bundle.X.shape[1],
            len(bundle.numerical_cols),
            len(bundle.categorical_cols),
        )
        if primary_bundle is None:
            primary_bundle = bundle
        if bundle.X.shape[1] == 0:
            LOGGER.warning("Skipping feature_mode=%s because no features were selected.", feature_mode)
            continue
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(bundle.X, bundle.y), start=1):
            repeat = int(math.ceil(split_id / n_splits))
            fold = int(((split_id - 1) % n_splits) + 1)
            rows, frames = fit_one_fold(
                cfg,
                bundle,
                train_idx,
                test_idx,
                repeat,
                fold,
                seed + split_id,
                feature_mode,
                model_names=model_names,
                imputer=imputer,
                max_estimators=max_estimators,
                calibrate=calibrate,
            )
            all_metrics.extend(rows)
            all_predictions.extend(frames)
            pd.DataFrame(all_metrics).to_csv(out_dir / "fold_metrics.partial.csv", index=False)
            pd.concat(all_predictions, ignore_index=True).to_csv(out_dir / "fold_predictions.partial.csv", index=False)

        LOGGER.info("Completed feature mode %s; partial outputs updated.", feature_mode)

    if primary_bundle is None:
        raise RuntimeError("No CV run was completed.")

    LOGGER.info("Writing raw fold metrics and predictions before post-processing.")
    fold_metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "fold_predictions.csv", index=False)

    LOGGER.info("Computing fold summary with %d bootstrap resamples.", n_bootstrap)
    fold_summary = summarize_fold_metrics(fold_metrics, seed, n_bootstrap=n_bootstrap)
    fold_summary.to_csv(out_dir / "cv_summary.csv", index=False)

    LOGGER.info("Computing out-of-fold summary with %d bootstrap resamples.", n_bootstrap)
    oof_summary = summarize_oof_predictions(predictions, seed, n_bootstrap=n_bootstrap)
    oof_summary.to_csv(out_dir / "oof_summary.csv", index=False)

    LOGGER.info("Computing paired model comparisons.")
    comparisons = paired_model_comparison(fold_metrics)
    comparisons.to_csv(out_dir / "model_comparisons.csv", index=False)
    primary_bundle.leakage_audit.to_csv(out_dir / "leakage_audit.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "feature_modes": feature_modes,
                "models": model_names or "all_available",
                "seed": seed,
                "n_bootstrap": n_bootstrap,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "imputer": imputer,
                "skip_plots": skip_plots,
                "max_estimators": max_estimators,
                "calibrate": calibrate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if skip_plots:
        LOGGER.info("Skipping plots by request.")
    else:
        LOGGER.info("Generating calibration and decision-curve plots.")
        plot_calibration(predictions, out_dir)
        plot_decision_curves(predictions, out_dir)
    LOGGER.info("Writing Markdown report.")
    write_report(
        out_dir,
        cfg,
        primary_bundle,
        fold_summary,
        oof_summary,
        comparisons,
        feature_modes,
        {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "calibrate": calibrate,
            "imputer": imputer,
        },
    )
    LOGGER.info("Research CV complete. Outputs saved to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-safe repeated CV for Track 1 DN prediction.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--preset",
        choices=["quick", "core", "ablations", "full"],
        default="core",
        help="Execution preset. Use core for paper-ready main table; ablations/full for slower expanded runs.",
    )
    parser.add_argument("--run-ablations", action="store_true", help="Deprecated alias for --preset ablations.")
    parser.add_argument("--feature-modes", default=None, help="Comma-separated feature modes. Overrides preset feature modes.")
    parser.add_argument("--models", default=None, help="Comma-separated models. Defaults depend on preset.")
    parser.add_argument("--splits", type=int, default=None, help="Number of CV folds. Defaults depend on preset.")
    parser.add_argument("--repeats", type=int, default=None, help="Number of CV repeats. Defaults depend on preset.")
    parser.add_argument("--imputer", choices=["simple", "knn"], default="simple", help="Numeric imputer. simple is faster and fold-safe.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip calibration and decision-curve PNG generation.")
    parser.add_argument("--max-estimators", type=int, default=None, help="Cap tree boosting estimators for faster runs.")
    parser.add_argument("--calibrate", action="store_true", help="Force probability calibration.")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable probability calibration.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for confidence intervals.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    cfg = load_config(args.config)
    preset = "ablations" if args.run_ablations else args.preset
    preset_defaults = {
        "quick": {
            "feature_modes": ["strict"],
            "models": ["xgboost"],
            "splits": 3,
            "repeats": 1,
            "bootstrap": min(args.bootstrap, 25),
            "skip_plots": True,
            "max_estimators": 80,
            "calibrate": False,
        },
        "core": {
            "feature_modes": ["strict"],
            "models": ["loop_gaussian_nb", "xgboost"],
            "splits": 5,
            "repeats": 3,
            "bootstrap": args.bootstrap,
            "skip_plots": args.skip_plots,
            "max_estimators": args.max_estimators or 250,
            "calibrate": True,
        },
        "ablations": {
            "feature_modes": ["strict", "lab_only", "demographics_history_lifestyle", "clinical_history_only", "without_lab_only"],
            "models": ["loop_gaussian_nb"],
            "splits": 5,
            "repeats": 3,
            "bootstrap": args.bootstrap,
            "skip_plots": args.skip_plots,
            "max_estimators": args.max_estimators or 200,
            "calibrate": False,
        },
        "full": {
            "feature_modes": ["strict", "lab_only", "demographics_history_lifestyle", "clinical_history_only", "without_lab_only"],
            "models": ["loop_gaussian_nb", "sgd_logistic", "xgboost", "lightgbm", "mlp"],
            "splits": 5,
            "repeats": 3,
            "bootstrap": args.bootstrap,
            "skip_plots": args.skip_plots,
            "max_estimators": args.max_estimators,
            "calibrate": True,
        },
    }[preset]
    feature_modes = (
        [item.strip() for item in args.feature_modes.split(",") if item.strip()]
        if args.feature_modes
        else preset_defaults["feature_modes"]
    )
    model_names = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else preset_defaults["models"]
    )
    run_cv(
        cfg,
        out_dir_override=args.out_dir,
        n_bootstrap=int(preset_defaults["bootstrap"]),
        feature_modes=feature_modes,
        model_names=model_names,
        n_splits=args.splits or int(preset_defaults["splits"]),
        n_repeats=args.repeats or int(preset_defaults["repeats"]),
        imputer=args.imputer,
        skip_plots=bool(preset_defaults["skip_plots"]),
        max_estimators=args.max_estimators if args.max_estimators is not None else preset_defaults["max_estimators"],
        calibrate=(True if args.calibrate else False if args.no_calibrate else bool(preset_defaults["calibrate"])),
    )


if __name__ == "__main__":
    main()

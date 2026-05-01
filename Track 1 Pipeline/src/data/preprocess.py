#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess.py — Data Cleaning, Imputation, and Feature Engineering
==================================================================

Track 1: AI-Based Diabetic Complication Detection Using Clinical Data

This script performs end-to-end data preprocessing for the bimodal diabetes
dataset (Li et al., 2026).  It reads the raw CSV, cleans missing/unknown
sentinel values, separates numerical and categorical features, imputes,
encodes, scales, and finally saves train/test splits as pickle files.

Usage
-----
    python src/data/preprocess.py --config config.yaml

Outputs
-------
    processed/X_train.pkl, processed/X_test.pkl
    processed/y_train.pkl, processed/y_test.pkl
    processed/feature_names.pkl
    processed/preprocessing_metadata.pkl
"""

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import logging
import pickle
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Read the YAML configuration file and return as a dictionary."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from '%s'.", config_path)
    return cfg


def _trim_and_replace_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text tokens and replace blank-like cells with NaN."""
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    str_cols = df.select_dtypes(include=["string"]).columns.tolist()
    text_cols = sorted(set(obj_cols + str_cols))

    def _clean_text_token(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            cleaned = v.replace("\xa0", " ").strip()
            if cleaned == "" or cleaned.lower() in {"na", "nan", "none", "null"}:
                return np.nan
            return cleaned
        return v

    if text_cols:
        for col in text_cols:
            df[col] = df[col].map(_clean_text_token)
    logger.info("Blank-string cleanup complete. Total NaNs now: %d", int(df.isna().sum().sum()))
    return df


def _infer_and_cast_feature_types(
    df: pd.DataFrame,
    max_unique_for_cat: int = 15,
    numeric_conversion_threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Infer numeric-vs-categorical columns robustly without destroying text data.

    A column is treated as numeric only when most non-missing values can be
    converted to numbers. Low-cardinality numeric-coded columns are kept as
    categorical features.
    """
    cast_df = df.copy()
    numerical_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in cast_df.columns:
        raw = cast_df[col]
        non_missing = int(raw.notna().sum())
        numeric_series = pd.to_numeric(raw, errors="coerce")
        conversion_rate = float(numeric_series.notna().sum() / non_missing) if non_missing else 0.0

        if conversion_rate >= numeric_conversion_threshold:
            cast_df[col] = numeric_series
            n_unique = int(numeric_series.nunique(dropna=True))
            if n_unique > max_unique_for_cat:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            cast_df[col] = raw.astype("string")
            categorical_cols.append(col)

    logger.info(
        "Feature inference complete → %d numerical, %d categorical.",
        len(numerical_cols),
        len(categorical_cols),
    )
    return cast_df, numerical_cols, categorical_cols


def _mark_unknown_code_9(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace categorical code `9` with NaN only for likely survey-coded columns.

    Criteria: numeric-like categorical columns with small integer codebooks
    between 0 and 9.
    """
    updated = df.copy()
    affected_cols: list[str] = []

    for col in categorical_cols:
        s = pd.to_numeric(updated[col], errors="coerce")
        non_na = s.dropna()
        if non_na.empty or not (s == 9).any():
            continue
        unique_vals = np.unique(non_na.values)
        if unique_vals.min() >= 0 and unique_vals.max() <= 9 and len(unique_vals) <= 10:
            updated[col] = updated[col].mask(s == 9, np.nan)
            affected_cols.append(col)

    return updated, affected_cols


def _build_stratify_labels(y: pd.DataFrame, fallback_col_idx: int = 0) -> pd.Series:
    """Create a stable stratification label for multi-label train/test split."""
    y_for_strat = y.fillna(-1).astype(int)
    combo = y_for_strat.astype(str).agg("_".join, axis=1)
    counts = combo.value_counts()
    if not counts.empty and counts.min() >= 2:
        return combo
    logger.warning(
        "Rare multilabel combinations detected for stratification; "
        "falling back to target column index %d.",
        fallback_col_idx,
    )
    fallback = y.iloc[:, fallback_col_idx]
    return fallback.fillna(0).astype(int)


def _resolve_active_targets(data_cfg: dict) -> list[str]:
    """Resolve the active target list from the configured study scope."""
    mode = str(data_cfg.get("active_target_set", "primary")).lower()
    primary = list(data_cfg.get("primary_target_columns", []))
    exploratory = list(data_cfg.get("exploratory_target_columns", []))
    custom = list(data_cfg.get("custom_target_columns", []))
    full = list(data_cfg.get("target_columns", []))

    if mode == "primary":
        targets = primary
    elif mode == "exploratory":
        targets = exploratory
    elif mode == "all":
        targets = full
    elif mode == "custom":
        targets = custom
    else:
        raise ValueError(f"Unsupported data.active_target_set: {mode}")

    if not targets:
        raise ValueError(f"No targets configured for active_target_set='{mode}'.")

    # Preserve order while removing duplicates.
    seen = set()
    ordered_targets = []
    for col in targets:
        if col not in seen:
            ordered_targets.append(col)
            seen.add(col)
    return ordered_targets


def _build_feature_drop_list(data_cfg: dict, active_targets: list[str], available_columns: list[str]) -> list[str]:
    """Build the exact feature-drop list for the selected targets."""
    drops = list(data_cfg.get("always_drop_feature_columns", []))
    target_specific = data_cfg.get("target_specific_feature_drops", {}) or {}
    for target in active_targets:
        drops.extend(target_specific.get(target, []))

    seen = set()
    filtered = []
    for col in drops:
        if col in available_columns and col not in seen:
            filtered.append(col)
            seen.add(col)
    return filtered


def _filter_targets_with_min_positives(y: pd.DataFrame, min_positives: int) -> tuple[pd.DataFrame, list[str]]:
    """Keep only targets with enough positive samples for reliable learning."""
    retained: list[str] = []
    dropped: list[str] = []

    for col in y.columns:
        positives = int(y[col].sum())
        if positives >= min_positives:
            retained.append(col)
        else:
            dropped.append(col)

    if dropped:
        logger.warning(
            "Dropping low-support targets (<%d positives): %s",
            min_positives,
            dropped,
        )

    if not retained:
        raise ValueError(
            "No viable target columns remain after low-support filtering. "
            f"Reduce data.min_target_positives (current: {min_positives})."
        )

    return y[retained].copy(), dropped


def _drop_targets_with_zero_split_support(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop targets that have zero positives in either train or test split."""
    keep: list[str] = []
    dropped: list[str] = []

    for col in y_train.columns:
        train_pos = int(y_train[col].sum())
        test_pos = int(y_test[col].sum())
        if train_pos > 0 and test_pos > 0:
            keep.append(col)
        else:
            dropped.append(col)

    if dropped:
        logger.warning(
            "Dropping targets with zero positives in train or test split: %s",
            dropped,
        )

    if not keep:
        raise ValueError("No targets remain after split-support filtering.")

    return y_train[keep].copy(), y_test[keep].copy(), dropped


# ═══════════════════════════════════════════════════════════════════════════
#  Target preparation
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """
    Clean and binarise the multi-label target columns.

    Rules applied per column
    ------------------------
    * **DRyd** — already binary (0/1), keep as-is.
    * **DN** — stages 0 / 3 / 4 / 5.  Binarise: 0 → 0, {3, 4, 5} → 1.
    * **PAD, MI, HF, CerebralIn, Amputation, Blind** — values 0 (no),
      1 (yes), 9 (unknown/not answered).  Map: 0 → 0, 1 → 1, 9 → NaN.
    * Blanks / empty strings have already been normalised to NaN in
      the initial cleanup stage.
    """
    targets = pd.DataFrame(index=df.index)

    for col in target_cols:
        if col not in df.columns:
            logger.warning("Target column '%s' not found in data — skipping.", col)
            continue

        series = df[col].copy()

        if col == "DN":
            # Binarise nephropathy staging
            series = pd.to_numeric(series, errors="coerce")
            series = series.map(lambda v: 0 if v == 0 else (1 if v in (3, 4, 5) else np.nan))
            logger.info("DN binarised → 0: %d, 1: %d, NaN: %d",
                        (series == 0).sum(), (series == 1).sum(), series.isna().sum())
        else:
            series = pd.to_numeric(series, errors="coerce")
            # 9 should already be NaN from sentinel replacement, but be safe
            series = series.replace(9, np.nan)

        targets[col] = series.astype("float32")

    logger.info("Target matrix shape: %s", targets.shape)
    return targets


# ═══════════════════════════════════════════════════════════════════════════
#  Main preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(cfg: dict) -> None:
    """Execute the full preprocessing pipeline and persist artefacts."""

    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    # ── 1. Load raw CSV ─────────────────────────────────────────────────
    raw_path = paths_cfg["raw_csv"]
    logger.info("Loading raw CSV from '%s' ...", raw_path)
    df = pd.read_csv(
        raw_path,
        encoding=data_cfg["csv_encoding"],
        low_memory=False,
        dtype="string",
    )
    raw_shape = tuple(df.shape)
    logger.info("Raw shape: %s", df.shape)

    # ── 2. Drop ID / date / near-empty columns ─────────────────────────
    drop_cols = [c for c in data_cfg["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Dropped %d columns (%s). Shape now: %s",
                len(drop_cols), drop_cols, df.shape)

    # ── 3. Normalize blanks ──────────────────────────────────────────────
    df = _trim_and_replace_blanks(df)

    # ── 4. Separate targets from features ───────────────────────────────
    target_cols = _resolve_active_targets(data_cfg)
    # Also remove the original unusable DR column if still present
    dr_orig = data_cfg.get("dr_original_column", "DR")
    extra_drop = [dr_orig] if dr_orig in df.columns else []

    y = _prepare_targets(df, target_cols)
    feature_drop = target_cols + extra_drop
    # Also drop configured direct-leakage columns for the selected study scope.
    leakage_cols = _build_feature_drop_list(data_cfg, target_cols, list(df.columns))
    feature_drop += [c for c in leakage_cols if c in df.columns and c not in feature_drop]
    df.drop(columns=[c for c in feature_drop if c in df.columns], inplace=True, errors="ignore")
    logger.info("Feature matrix shape after target/leakage removal: %s", df.shape)

    target_missing_policy = str(data_cfg.get("target_missing_policy", "partial")).lower()

    # ── 5. Target-label row policy ───────────────────────────────────────
    if target_missing_policy == "complete_case":
        valid_mask = y.notna().all(axis=1)
    else:
        # Keep rows that have at least one known target label
        valid_mask = y.notna().any(axis=1)
    dropped_for_missing_targets = int((~valid_mask).sum())
    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    logger.info(
        "Rows retained under target policy '%s': %d / %d (dropped: %d)",
        target_missing_policy,
        int(valid_mask.sum()),
        int(len(valid_mask)),
        dropped_for_missing_targets,
    )
    y = y.astype("float32")

    min_target_positives = int(data_cfg.get("min_target_positives", 10))
    y, dropped_low_support_targets = _filter_targets_with_min_positives(y, min_target_positives)

    # ── 6. Infer feature types and cast safely ──────────────────────────
    df, numerical_cols, categorical_cols = _infer_and_cast_feature_types(df)

    # Replace categorical unknown code 9 where appropriate
    df, unknown_9_cols = _mark_unknown_code_9(df, categorical_cols)
    if unknown_9_cols:
        logger.info("Converted categorical code 9 -> NaN in %d columns.", len(unknown_9_cols))

    # ── 7. Drop columns with excessive missingness ──────────────────────
    max_frac = data_cfg["max_missing_fraction"]
    missing_frac = df.isna().mean()
    cols_to_drop_missing = missing_frac[missing_frac > max_frac].index.tolist()
    if cols_to_drop_missing:
        df.drop(columns=cols_to_drop_missing, inplace=True)
        logger.info("Dropped %d columns with >%.0f%% missing: %s",
                    len(cols_to_drop_missing), max_frac * 100, cols_to_drop_missing)

    # Keep feature-type lists in sync after column drops
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # ── 8. Feature type summary ──────────────────────────────────────────
    logger.info("Numerical columns (%d): %s", len(numerical_cols), numerical_cols[:10])
    logger.info("Categorical columns (%d): %s", len(categorical_cols), categorical_cols[:10])

    # ── 9. Imputation ───────────────────────────────────────────────────
    # 9a — Numerical: KNN imputer (k=5)
    if numerical_cols:
        knn_k = data_cfg.get("knn_imputer_k", 5)
        logger.info("Imputing %d numerical columns with KNNImputer (k=%d) ...", len(numerical_cols), knn_k)
        knn_imp = KNNImputer(n_neighbors=knn_k)
        df[numerical_cols] = knn_imp.fit_transform(df[numerical_cols])

    # 9b — Categorical: most-frequent imputer
    if categorical_cols:
        logger.info("Imputing %d categorical columns with SimpleImputer (most_frequent) ...", len(categorical_cols))
        # Ensure text columns are true object dtype for sklearn compatibility
        for col in categorical_cols:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                df[col] = df[col].astype(object)

        cat_imp = SimpleImputer(strategy="most_frequent")
        try:
            df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
        except ValueError as exc:
            logger.warning(
                "SimpleImputer failed on categorical block (%s). Falling back to per-column mode fill.",
                exc,
            )
            for col in categorical_cols:
                series = df[col]
                mode = series.mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "MISSING"
                df[col] = series.fillna(fill_value)

    logger.info("Post-imputation NaN count: %d", int(df.isna().sum().sum()))

    # ── 10. Encoding & Scaling ──────────────────────────────────────────
    # 10a — Ordinal-encode categorical features (tree models need numbers,
    #        and for the NN we feed everything as floats anyway)
    if categorical_cols:
        logger.info("Ordinal-encoding %d categorical columns ...", len(categorical_cols))
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[categorical_cols] = ord_enc.fit_transform(df[categorical_cols])

    # 10b — StandardScaler on numerical features
    if numerical_cols:
        logger.info("Standard-scaling %d numerical columns ...", len(numerical_cols))
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # ── 11. Train / test split (80/20, multilabel-aware stratification) ─
    strat_col = _build_stratify_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_seed"],
        stratify=strat_col,
    )

    dropped_zero_split_targets: list[str] = []
    if target_missing_policy == "complete_case":
        y_train, y_test, dropped_zero_split_targets = _drop_targets_with_zero_split_support(y_train, y_test)

    logger.info("Train set: X=%s, y=%s", X_train.shape, y_train.shape)
    logger.info("Test  set: X=%s, y=%s", X_test.shape, y_test.shape)

    # ── 12. Persist processed artefacts ─────────────────────────────────
    proc_dir = Path(paths_cfg["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(df.columns),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "target_columns": list(y_train.columns),
    }

    for name, obj in artefacts.items():
        fp = proc_dir / f"{name}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved → %s", fp)

    # Also save a metadata summary for the audit report
    target_support = {}
    for col in y_train.columns:
        full_col = pd.concat([y_train[col], y_test[col]], axis=0)
        target_support[col] = {
            "labeled_rows": int(full_col.notna().sum()),
            "positive_rows": int((full_col == 1).sum()),
            "negative_rows": int((full_col == 0).sum()),
            "missing_rows": int(full_col.isna().sum()),
            "train_positive_rows": int((y_train[col] == 1).sum()),
            "test_positive_rows": int((y_test[col] == 1).sum()),
        }

    metadata = {
        "raw_shape": raw_shape,
        "post_target_filter_shape": tuple(df.shape),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "n_numerical": len(numerical_cols),
        "n_categorical": len(categorical_cols),
        "target_columns": list(y_train.columns),
        "dropped_columns": drop_cols + cols_to_drop_missing,
        "active_target_set": data_cfg.get("active_target_set", "primary"),
        "exploratory_target_columns": list(data_cfg.get("exploratory_target_columns", [])),
        "primary_target_columns": list(data_cfg.get("primary_target_columns", [])),
        "applied_feature_drops": leakage_cols,
        "knn_k": data_cfg.get("knn_imputer_k", 5),
        "blank_handling": "trim + empty-string to NaN",
        "unknown_9_columns": unknown_9_cols,
        "dropped_rows_incomplete_targets": dropped_for_missing_targets,
        "target_missing_policy": target_missing_policy,
        "min_target_positives": min_target_positives,
        "dropped_low_support_targets": dropped_low_support_targets,
        "dropped_zero_split_targets": dropped_zero_split_targets,
        "target_support": target_support,
        "target_positive_rates": {
            col: float(pd.concat([y_train[col], y_test[col]], axis=0).mean()) for col in y_train.columns
        },
    }
    with open(proc_dir / "preprocessing_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    logger.info("Saved → %s", proc_dir / "preprocessing_metadata.pkl")

    logger.info("✅  Preprocessing complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the bimodal diabetes dataset for Track 1."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess(cfg)


if __name__ == "__main__":
    main()

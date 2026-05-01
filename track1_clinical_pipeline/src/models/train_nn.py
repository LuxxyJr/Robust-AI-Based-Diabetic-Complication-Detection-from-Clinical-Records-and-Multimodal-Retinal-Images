#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_nn.py — PyTorch Neural Network for Multi-Label Complication Prediction
============================================================================

Track 1: AI-Based Diabetic Complication Detection Using Clinical Data

This script implements a deep MLP with Batch Normalisation and Dropout,
trained with ``BCEWithLogitsLoss`` for multi-label classification.  It
includes a full training loop with:

* Validation-loss tracking
* ReduceLROnPlateau scheduler
* Early stopping
* Model checkpointing (best ``.pth`` weights)

An alternative TabNet path is also provided (toggled via ``config.yaml``).

Usage
-----
    python src/models/train_nn.py --config config.yaml

Outputs
-------
    outputs/nn_best_model.pth       — best model weights
    outputs/nn_train_history.pkl    — epoch-level loss/metric history
"""

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import logging
import pickle
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

class DiabetesTabularDataset(Dataset):
    """
    A simple PyTorch Dataset that wraps NumPy feature and label arrays.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (already scaled / encoded).
    y : np.ndarray, shape (n_samples, n_targets)
        Multi-label binary target matrix.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  Model Architecture: Deep MLP
# ═══════════════════════════════════════════════════════════════════════════

class MultiLabelMLP(nn.Module):
    """
    A multi-layer perceptron with Batch Normalisation and Dropout designed
    for multi-label tabular classification.

    Architecture
    ------------
    For each hidden dim in ``hidden_dims``:
        Linear → BatchNorm1d → ReLU → Dropout

    Final layer: Linear(last_hidden, n_targets) — raw logits (no sigmoid),
    because ``BCEWithLogitsLoss`` applies sigmoid internally for numerical
    stability.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_targets : int
        Number of binary target labels.
    hidden_dims : list[int]
        Sizes of hidden layers, e.g. [256, 128, 64].
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        n_targets: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        # Output head — raw logits
        layers.append(nn.Linear(prev_dim, n_targets))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (batch, n_targets)."""
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Read the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_processed_data(processed_dir: str) -> tuple:
    """Load the preprocessed pickle artefacts."""
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


def compute_pos_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute per-target positive-class weight for ``BCEWithLogitsLoss``
    to handle class imbalance.

    pos_weight_j = n_negative_j / n_positive_j
    """
    n_pos = y.sum(axis=0)
    n_neg = (1 - y).sum(axis=0)
    # Guard against division by zero
    weights = np.where(n_pos > 0, n_neg / n_pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weights_ignore_nan(y: np.ndarray) -> torch.Tensor:
    """Compute per-target positive weights using only labeled entries."""
    weights = []
    for j in range(y.shape[1]):
        col = y[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            weights.append(1.0)
            continue
        labeled = col[mask]
        n_pos = float((labeled == 1).sum())
        n_neg = float((labeled == 0).sum())
        weights.append((n_neg / n_pos) if n_pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal BCE loss for imbalanced multi-label classification."""

    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt).pow(self.gamma)
        return (focal_factor * bce).mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss that ignores NaN labels (partial multi-label)."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        per_entry = nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        masked = per_entry * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        return masked.sum() / denom


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    """Focal BCE that ignores NaN labels (partial multi-label)."""

    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(safe_targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt).pow(self.gamma)
        loss = focal_factor * bce
        masked = loss * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        return masked.sum() / denom


def compute_sample_weights(y: np.ndarray, pos_weights: np.ndarray) -> np.ndarray:
    """Compute per-sample weights for weighted random sampling in multilabel data."""
    y_safe = np.nan_to_num(y, nan=0.0)
    positive_signal = y_safe * pos_weights.reshape(1, -1)
    sample_weights = positive_signal.max(axis=1)
    sample_weights = np.where(sample_weights > 0, sample_weights, 1.0)
    return sample_weights.astype(np.float64)


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
    fallback = np.where(y_int[:, 0] < 0, 0, y_int[:, 0])
    _, fallback_counts = np.unique(fallback, return_counts=True)
    if fallback_counts.size > 0 and fallback_counts.min() >= 2:
        return fallback
    logger.warning("Could not stratify validation split safely; using random split.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on a data loader and return the mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main training driver
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_network(cfg: dict) -> None:
    """Full MLP training pipeline with checkpointing and scheduling."""

    nn_cfg = cfg["neural_network"]
    paths_cfg = cfg["paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load data ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, target_cols, feature_names = load_processed_data(
        paths_cfg["processed_dir"]
    )

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)

    # ── Validation split (15 % of training set) ────────────────────────
    strat_labels = _build_stratify_labels_np(y_train_np)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np,
        test_size=0.15,
        random_state=SEED,
        stratify=strat_labels,
    )
    logger.info("Train: %d | Val: %d | Test: %d", X_tr.shape[0], X_val.shape[0], X_test_np.shape[0])

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_ds = DiabetesTabularDataset(X_tr, y_tr)
    val_ds = DiabetesTabularDataset(X_val, y_val)

    batch_size = nn_cfg["batch_size"]
    use_weighted_sampler = bool(nn_cfg.get("use_weighted_sampler", False))
    if use_weighted_sampler:
        pos_weights_np = compute_pos_weights_ignore_nan(y_tr).numpy()
        sample_weights = compute_sample_weights(y_tr, pos_weights_np)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
        logger.info("Using WeightedRandomSampler for class-imbalance handling.")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────────
    input_dim = X_tr.shape[1]
    n_targets = y_tr.shape[1]

    if nn_cfg["model_type"] == "tabnet":
        # ── TabNet path ─────────────────────────────────────────────────
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier

            logger.info("Using TabNet model.")
            _train_tabnet(cfg, X_tr, y_tr, X_val, y_val, X_test_np, y_test_np,
                          target_cols, feature_names, paths_cfg)
            return
        except ImportError:
            logger.warning("pytorch-tabnet not installed — falling back to MLP.")

    # ── MLP path ────────────────────────────────────────────────────────
    model = MultiLabelMLP(
        input_dim=input_dim,
        n_targets=n_targets,
        hidden_dims=nn_cfg["hidden_dims"],
        dropout=nn_cfg["dropout"],
    ).to(device)

    logger.info("Model architecture:\n%s", model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total trainable parameters: %s", f"{total_params:,}")

    # ── Loss with pos_weight for class imbalance ────────────────────────
    pos_weights = compute_pos_weights_ignore_nan(y_tr).to(device)
    logger.info("pos_weights per target: %s", pos_weights.cpu().numpy().round(2))
    loss_type = str(nn_cfg.get("loss_type", "bce")).lower()
    if loss_type == "focal_bce":
        gamma = float(nn_cfg.get("focal_gamma", 2.0))
        criterion = MaskedFocalBCEWithLogitsLoss(pos_weight=pos_weights, gamma=gamma)
        logger.info("Using FocalBCEWithLogitsLoss (gamma=%.2f).", gamma)
    else:
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights)
        logger.info("Using masked BCEWithLogitsLoss.")

    # ── Optimiser & Scheduler ───────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=nn_cfg["learning_rate"],
        weight_decay=nn_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        patience=nn_cfg["scheduler_patience"],
        factor=nn_cfg["scheduler_factor"],
    )

    # ── Training loop ───────────────────────────────────────────────────
    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "nn_best_model.pth"

    epochs = nn_cfg["epochs"]
    patience = nn_cfg["early_stopping_patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        current_lr = optimiser.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # ── Checkpointing ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            marker = " ★ saved"
        else:
            epochs_no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or marker:
            logger.info(
                "Epoch %3d/%d | train_loss: %.5f | val_loss: %.5f | lr: %.2e%s",
                epoch, epochs, train_loss, val_loss, current_lr, marker,
            )

        # ── Early stopping ──────────────────────────────────────────────
        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    # ── Save the full model info for evaluation ─────────────────────────
    model_bundle = {
        "model_state_dict_path": str(best_model_path),
        "input_dim": input_dim,
        "n_targets": n_targets,
        "hidden_dims": nn_cfg["hidden_dims"],
        "dropout": nn_cfg["dropout"],
        "target_columns": target_cols,
        "feature_names": feature_names,
        "model_type": "mlp",
        "best_val_loss": best_val_loss,
    }
    bundle_path = out_dir / "nn_model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info("Saved model bundle → %s", bundle_path)

    # ── Save training history ───────────────────────────────────────────
    hist_path = out_dir / "nn_train_history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    logger.info("Saved training history → %s", hist_path)

    logger.info("✅  Neural network training complete. Best val_loss: %.5f", best_val_loss)


# ═══════════════════════════════════════════════════════════════════════════
#  TabNet fallback
# ═══════════════════════════════════════════════════════════════════════════

def _train_tabnet(
    cfg: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_cols: list[str],
    feature_names: list[str],
    paths_cfg: dict,
) -> None:
    """
    Train a TabNet model for multi-label classification using
    ``pytorch-tabnet``.  This is called as an alternative to the custom MLP
    when ``config.yaml`` sets ``model_type: tabnet``.

    TabNet operates in a per-target manner for multi-label tasks because
    the ``pytorch-tabnet`` library expects single-output classification.
    We train one TabNetClassifier per target and bundle them.
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    nn_cfg = cfg["neural_network"]
    tab_cfg = nn_cfg["tabnet"]
    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    estimators = []
    for idx, col_name in enumerate(target_cols):
        logger.info("[TabNet] Training for target: %s", col_name)

        y_tr_col = y_tr[:, idx].astype(np.int64)
        y_val_col = y_val[:, idx].astype(np.int64)

        clf = TabNetClassifier(
            n_d=tab_cfg["n_d"],
            n_a=tab_cfg["n_a"],
            n_steps=tab_cfg["n_steps"],
            gamma=tab_cfg["gamma_tabnet"],
            n_independent=tab_cfg["n_independent"],
            n_shared=tab_cfg["n_shared"],
            mask_type=tab_cfg["mask_type"],
            seed=SEED,
            verbose=0,
        )

        clf.fit(
            X_train=X_tr,
            y_train=y_tr_col,
            eval_set=[(X_val, y_val_col)],
            eval_metric=["logloss"],
            max_epochs=nn_cfg["epochs"],
            patience=nn_cfg["early_stopping_patience"],
            batch_size=nn_cfg["batch_size"],
        )

        estimators.append(clf)
        logger.info("[TabNet] %s — best val logloss: %.5f", col_name, clf.best_cost)

    # Save TabNet bundle
    model_bundle = {
        "estimators": estimators,
        "target_columns": target_cols,
        "feature_names": feature_names,
        "model_type": "tabnet",
    }
    bundle_path = out_dir / "nn_model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info("Saved TabNet model bundle → %s", bundle_path)
    logger.info("✅  TabNet training complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a neural network (MLP/TabNet) for Track 1."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_neural_network(cfg)


if __name__ == "__main__":
    main()

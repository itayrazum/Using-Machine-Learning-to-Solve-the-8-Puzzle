# -*- coding: utf-8 -*-
"""
End-to-end pipeline to build learned A* heuristics as pure model predictions.

What this script does
---------------------
1) Loads a CSV of all solvable 8-puzzle states with exact optimal costs.
   Required columns:
       pos0,pos1,pos2,pos3,pos4,pos5,pos6,pos7,pos8,cost

2) For each requested model and each training percent p:
       - Repeats N runs.
       - Each run:
           a) Draws a stratified sample of size p from the train pool
              (stratified by true optimal cost).
           b) Trains a regressor to predict cost from the board.
           c) Predicts cost for every state in the dataset.
           d) Saves a dictionary:
                    "1,2,3,4,5,6,7,8,0" -> predicted_cost
              to:
                    {out_root}/{model}/percent_{p}/run_{iii}/predict_dict.json
           e) Saves run metadata to meta.json in the same folder.

3) Directory layout
       out_root/
         <model>/
           percent_<p>/
             run_001/
               predict_dict.json
               meta.json
             run_002/
               ...
           percent_<p2>/
             ...

Models supported
----------------
- "xgboost"   (optional, requires package xgboost)
- "random_forest"  sklearn RandomForestRegressor
- "gbrt"           sklearn GradientBoostingRegressor
- "ridge"          sklearn Ridge regressor (fast linear baseline)
- "mlp"            PyTorch multilayer perceptron regressor

Feature encoding
----------------
- By default we use one-hot per position for tile values 0..8 (81 dims).
- You can optionally add simple engineered features:
      Manhattan distance and number of misplaced tiles
  These are computed from the board and help small models learn faster.
  They do not mix any external heuristic into the predictions at inference time
  because the predictions you save are purely what the model outputs.

Reproducibility
---------------
- A fixed test split is made once and never used for training.
- Each run uses a different sampling seed for the train subset.
- Model random_state or torch seeds are set per run.

Usage examples
--------------
python build_predict_dicts.py \
  --data eight_puzzle_ground_truth.csv \
  --out-root heuristics_runs \
  --models mlp random_forest \
  --percents 10 20 30 \
  --runs 100

python build_predict_dicts.py \
  --data eight_puzzle_ground_truth.csv \
  --out-root outputs_xgb \
  --models xgboost \
  --percents 10 \
  --runs 50 \
  --no-engineered
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional xgboost
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Sklearn models and tools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Torch for the MLP option
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Helpers for 8-puzzle features
# =============================================================================

POS_TO_RC = [(i // 3, i % 3) for i in range(9)]
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_POS = {v: POS_TO_RC[i] for i, v in enumerate(GOAL)}

def manhattan_of_row(row: pd.Series) -> int:
    """
    Compute Manhattan distance from this board to GOAL.
    This is used only as an optional feature for the model.
    Predictions saved to predict_dict.json are the pure model outputs.
    """
    total = 0
    for i in range(9):
        v = int(row[f"pos{i}"])
        if v == 0:
            continue
        r, c = POS_TO_RC[i]
        gr, gc = GOAL_POS[v]
        total += abs(r - gr) + abs(c - gc)
    return total

def misplaced_of_row(row: pd.Series) -> int:
    """
    Count non-blank tiles not in their goal position.
    Used only as an optional feature for the model.
    """
    mis = 0
    for i in range(9):
        v = int(row[f"pos{i}"])
        if v != 0 and v != GOAL[i]:
            mis += 1
    return mis

def state_key_from_row(row: pd.Series) -> str:
    """Create a compact key 'a,b,c,d,e,f,g,h,i' for this board."""
    return ",".join(str(int(row[f"pos{i}"])) for i in range(9))

def state_key_from_tuple(state: Tuple[int, ...]) -> str:
    return ",".join(str(x) for x in state)


# =============================================================================
# Data splitting and sampling
# =============================================================================

def split_fixed_test(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 17) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one fixed test split stratified by cost. This test set is never used for training.
    Return train_pool, test_df.
    """
    rng = np.random.default_rng(seed)
    test_idx_parts = []
    for _, grp in df.groupby("cost"):
        n = len(grp)
        k = max(1, int(round(test_frac * n)))
        idx = rng.choice(grp.index.values, size=k, replace=False)
        test_idx_parts.append(idx)
    test_idx = np.concatenate(test_idx_parts)
    test_mask = df.index.isin(test_idx)
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)

def stratified_sample_by_cost(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """
    Draw a stratified sample by cost with approximately frac proportion.
    Keeps class balance stable across runs.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby("cost"):
        n = len(grp)
        k = max(1, int(round(frac * n)))
        k = min(k, n)
        if k == n:
            parts.append(grp)
        else:
            idx = rng.choice(grp.index.values, size=k, replace=False)
            parts.append(grp.loc[idx])
    sampled = pd.concat(parts, axis=0)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =============================================================================
# Sklearn models
# =============================================================================

@dataclass
class ModelSpec:
    name: str
    params: Dict

def make_sklearn_pipeline(spec: ModelSpec, add_engineered: bool) -> Pipeline:
    """
    Build a sklearn Pipeline that:
      - one-hot encodes the 9 tile positions (values 0..8)
      - optionally passes engineered numeric features
      - feeds a regressor
    """
    tile_cols = [f"pos{i}" for i in range(9)]
    numeric_cols = ["manhattan", "misplaced"] if add_engineered else []

    enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
    transformers = [("tiles", enc, tile_cols)]
    if numeric_cols:
        transformers.append(("eng", "passthrough", numeric_cols))
    pre = ColumnTransformer(transformers)

    if spec.name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost requested but the package is not available. Install xgboost or choose another model.")
        reg = xgb.XGBRegressor(
            n_estimators=spec.params.get("n_estimators", 600),
            max_depth=spec.params.get("max_depth", 6),
            learning_rate=spec.params.get("learning_rate", 0.05),
            subsample=spec.params.get("subsample", 0.8),
            colsample_bytree=spec.params.get("colsample_bytree", 0.8),
            reg_lambda=spec.params.get("reg_lambda", 1.0),
            random_state=spec.params.get("random_state", 0),
            tree_method=spec.params.get("tree_method", "hist"),
            n_jobs=spec.params.get("n_jobs", -1),
        )
    elif spec.name == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=spec.params.get("n_estimators", 500),
            max_depth=spec.params.get("max_depth", None),
            min_samples_leaf=spec.params.get("min_samples_leaf", 1),
            n_jobs=spec.params.get("n_jobs", -1),
            random_state=spec.params.get("random_state", 0),
        )
    elif spec.name == "gbrt":
        reg = GradientBoostingRegressor(
            n_estimators=spec.params.get("n_estimators", 800),
            learning_rate=spec.params.get("learning_rate", 0.05),
            max_depth=spec.params.get("max_depth", 3),
            random_state=spec.params.get("random_state", 0),
        )
    elif spec.name == "ridge":
        reg = Ridge(alpha=spec.params.get("alpha", 2.0), random_state=spec.params.get("random_state", 0))
    else:
        raise ValueError(f"Unknown sklearn model name: {spec.name}")

    return Pipeline([("pre", pre), ("reg", reg)])


# =============================================================================
# Torch MLP
# =============================================================================

class BoardDataset(Dataset):
    """
    Dataset that turns boards into dense tensors.
    Features:
        - One-hot per position for tile value 0..8 gives 81 dims.
        - Optional engineered features appended as extra dims.
    """
    def __init__(self, df: pd.DataFrame, add_engineered: bool):
        tile_cols = [f"pos{i}" for i in range(9)]
        tiles = df[tile_cols].values.astype(np.int64)
        N = tiles.shape[0]

        one_hot = np.zeros((N, 9, 9), dtype=np.float32)
        for p in range(9):
            one_hot[np.arange(N), p, tiles[:, p]] = 1.0
        X = one_hot.reshape(N, 81)

        if add_engineered:
            feats = []
            if "manhattan" in df.columns:
                feats.append(df["manhattan"].values.reshape(-1, 1).astype(np.float32))
            if "misplaced" in df.columns:
                feats.append(df["misplaced"].values.reshape(-1, 1).astype(np.float32))
            if feats:
                X = np.concatenate([X] + feats, axis=1)

        y = df["cost"].values.astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    """
    Simple MLP for cost regression.
    Input dims:
        81 if only one-hot
        83 if with engineered features
    """
    def __init__(self, input_dim: int, hidden=(256, 256), dropout=0.10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)

def train_mlp(train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              seed: int,
              add_engineered: bool,
              max_epochs: int = 60,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              batch_size: int = 2048,
              patience: int = 10) -> Tuple[MLPRegressor, float, torch.device]:
    """
    Train the MLP with early stopping on validation MAE.
    Returns the best model, best MAE, and the device used.
    """
    if add_engineered:
        if "manhattan" not in train_df.columns:
            train_df["manhattan"] = train_df.apply(manhattan_of_row, axis=1).astype(np.int16)
        if "misplaced" not in train_df.columns:
            train_df["misplaced"] = train_df.apply(misplaced_of_row, axis=1).astype(np.int8)
        if "manhattan" not in val_df.columns:
            val_df["manhattan"] = val_df.apply(manhattan_of_row, axis=1).astype(np.int16)
        if "misplaced" not in val_df.columns:
            val_df["misplaced"] = val_df.apply(misplaced_of_row, axis=1).astype(np.int8)

    train_ds = BoardDataset(train_df, add_engineered=add_engineered)
    val_ds = BoardDataset(val_df, add_engineered=add_engineered)

    input_dim = train_ds.X.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.SmoothL1Loss(beta=1.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        # validate
        model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                preds.append(pred.detach().cpu().numpy())
                trues.append(yb.detach().cpu().numpy())
            y_pred = np.concatenate(preds, axis=0)
            y_true = np.concatenate(trues, axis=0)
            mae = float(np.mean(np.abs(y_true - y_pred)))

        print(f"epoch {epoch:03d} train_loss={(total/len(train_ds)):.4f} val_mae={mae:.4f} best={best_mae:.4f}")

        if mae + 1e-6 < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_mae, device

def predict_all_with_mlp(model: MLPRegressor,
                         df_all: pd.DataFrame,
                         add_engineered: bool,
                         device: torch.device,
                         batch_size: int = 4096) -> np.ndarray:
    """
    Predict for all states in df_all with the trained MLP.
    """
    if add_engineered:
        if "manhattan" not in df_all.columns:
            df_all["manhattan"] = df_all.apply(manhattan_of_row, axis=1).astype(np.int16)
        if "misplaced" not in df_all.columns:
            df_all["misplaced"] = df_all.apply(misplaced_of_row, axis=1).astype(np.int8)

    ds = BoardDataset(df_all, add_engineered=add_engineered)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


# =============================================================================
# Core per-run routine shared by all models
# =============================================================================

def run_once_and_save(df_all: pd.DataFrame,
                      model_name: str,
                      model_spec: Optional[ModelSpec],
                      train_frac: float,
                      run_seed: int,
                      fixed_test_seed: int,
                      out_root: Path,
                      percent: int,
                      run_idx: int,
                      add_engineered: bool,
                      save_mlp_weights: bool) -> Path:
    """
    Train on a stratified subset and write predict_dict.json for all states.

    Returns the path to the run directory that was created.
    """
    # Create run directory up front
    run_dir = out_root / model_name / f"percent_{percent}" / f"run_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prepare engineered features if requested
    if add_engineered:
        if "manhattan" not in df_all.columns:
            df_all["manhattan"] = df_all.apply(manhattan_of_row, axis=1).astype(np.int16)
        if "misplaced" not in df_all.columns:
            df_all["misplaced"] = df_all.apply(misplaced_of_row, axis=1).astype(np.int8)

    # Make fixed test once
    train_pool, test_df = split_fixed_test(df_all, test_frac=0.2, seed=fixed_test_seed)
    # Draw stratified train sample
    train_df = stratified_sample_by_cost(train_pool, frac=train_frac, seed=run_seed)

    # Quick validation split for MLP early stopping
    rng = np.random.default_rng(run_seed)
    mask = rng.random(len(train_df)) < 0.9
    sub_train_df = train_df[mask].reset_index(drop=True)
    val_df = train_df[~mask].reset_index(drop=True)

    # Predict all
    if model_name == "mlp":
        model, best_mae, device = train_mlp(
            sub_train_df, val_df, seed=run_seed, add_engineered=add_engineered
        )
        y_pred_test = predict_all_with_mlp(
            model, test_df, add_engineered=add_engineered, device=device
        )
        mae_test = float(mean_absolute_error(test_df["cost"].values.astype(np.float32), y_pred_test))
        y_pred_all = predict_all_with_mlp(
            model, df_all, add_engineered=add_engineered, device=device
        ).astype(np.float32)
        if save_mlp_weights:
            torch.save({"state_dict": model.state_dict(),
                        "input_dim": next(model.parameters()).shape[1]},
                       run_dir / "model.pt")
    else:
        # Build sklearn pipeline
        spec = model_spec if model_spec is not None else ModelSpec(model_name, {})
        spec.params["random_state"] = run_seed
        pipe = make_sklearn_pipeline(spec, add_engineered=add_engineered)

        feat_cols = [f"pos{i}" for i in range(9)]
        if add_engineered:
            feat_cols += ["manhattan", "misplaced"]

        X_train = sub_train_df[feat_cols].copy()
        y_train = sub_train_df["cost"].astype(np.int16).values
        X_val = val_df[feat_cols].copy()
        y_val = val_df["cost"].astype(np.int16).values
        X_test = test_df[feat_cols].copy()

        pipe.fit(pd.concat([X_train, X_val], axis=0),
                 np.concatenate([y_train, y_val], axis=0))

        y_pred_test = pipe.predict(X_test)
        mae_test = float(mean_absolute_error(test_df["cost"].values.astype(np.float32), y_pred_test))

        X_all = df_all[feat_cols].copy()
        y_pred_all = pipe.predict(X_all).astype(np.float32)

    # Build the prediction dictionary mapping state key to float
    keys = df_all.apply(state_key_from_row, axis=1).values
    predict_dict = {k: float(v) for k, v in zip(keys, y_pred_all)}

    # Save predict_dict.json
    pred_path = run_dir / "predict_dict.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predict_dict, f)

    # Save meta.json
    meta = {
        "model_name": model_name,
        "percent": percent,
        "train_frac": train_frac,
        "run_seed": run_seed,
        "fixed_test_seed": fixed_test_seed,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "mae_on_fixed_test": mae_test,
        "engineered_features": bool(add_engineered),
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {pred_path}")
    return run_dir


# =============================================================================
# Main orchestrator
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with pos0..pos8 and cost for all solvable states")
    ap.add_argument("--out-root", required=True, help="Output directory root for all runs")
    ap.add_argument("--models", nargs="+", default=["mlp", "random_forest"],
                    help="Any of: mlp xgboost random_forest gbrt ridge")
    ap.add_argument("--percents", nargs="+", type=int, default=[10, 20, 30],
                    help="Training percentages, for example 10 20 30")
    ap.add_argument("--runs", type=int, default=100, help="Number of repeated runs per percent")
    ap.add_argument("--seed", type=int, default=7, help="Base seed for sampling")
    ap.add_argument("--no-engineered", action="store_true",
                    help="If set, disable engineered features (manhattan and misplaced) as inputs")
    ap.add_argument("--save-mlp-weights", action="store_true",
                    help="If set and model is mlp, save model.pt per run")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    needed = [f"pos{i}" for i in range(9)] + ["cost"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    add_engineered = not args.no_engineered

    # Define model specs
    specs: Dict[str, ModelSpec] = {
        "random_forest": ModelSpec("random_forest", {"n_estimators": 500, "n_jobs": -1}),
        "gbrt":          ModelSpec("gbrt",          {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 3}),
        "ridge":         ModelSpec("ridge",         {"alpha": 2.0}),
    }
    if HAS_XGB:
        specs["xgboost"] = ModelSpec("xgboost", {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05})

    fixed_test_seed = args.seed * 13 + 11

    for model_name in args.models:
        if model_name not in specs and model_name != "mlp":
            raise ValueError(f"Unknown model '{model_name}'. Allowed: {list(specs.keys()) + ['mlp']}")

        if model_name == "xgboost" and not HAS_XGB:
            print("Warning: xgboost requested but package not available. Skipping xgboost.")

        for p in args.percents:
            if not (1 <= p <= 100):
                raise ValueError(f"Bad percent value {p}")
            train_frac = p / 100.0

            for run_idx in range(1, args.runs + 1):
                run_seed = args.seed + run_idx * 1009 + p * 17
                spec = specs.get(model_name, None)

                run_once_and_save(
                    df_all=df,
                    model_name=model_name,
                    model_spec=spec,
                    train_frac=train_frac,
                    run_seed=run_seed,
                    fixed_test_seed=fixed_test_seed,
                    out_root=out_root,
                    percent=p,
                    run_idx=run_idx,
                    add_engineered=add_engineered,
                    save_mlp_weights=args.save_mlp_weights,
                )

if __name__ == "__main__":
    main()

"""
Optional 4th model: a small PyTorch feed-forward neural network, trained on
the exact same features and train/test split (random_state=42) as the other
three models, for a genuinely apples-to-apples comparison.

WHY THIS IS SEPARATE FROM train_xg_model.py, AND MUST BE RUN LOCALLY:
This sandbox does not have PyTorch installed, and installing it here isn't
practical (a multi-hundred-MB download well beyond the environment's
per-command time budget). This script is meant to be run on your own
machine: `pip install torch` (CPU-only is plenty - this dataset and network
are both small), then `python src/train_xg_model_deep.py`.

HONEST EXPECTATIONS - PLEASE READ BEFORE RUNNING:
Tabular data (rows of distance/angle/pressure/etc., not images or text) is
the one place deep learning does NOT reliably beat simpler methods. This is
a well-documented empirical finding, not a guess: gradient-boosted trees
(what XGBoost and HistGradientBoostingClassifier already do in this repo)
tend to match or beat plain feed-forward networks on structured/tabular
data of this size and shape, and only pull ahead with either much more data
(millions of rows) or specialized architectures (TabNet, FT-Transformer,
etc.) that this simple MLP does not use. See Shwartz-Ziv & Armon, "Tabular
Data: Deep Learning is Not All You Need" (2021) for the research behind
this. With ~14,000 training rows and 36 engineered/one-hot features, expect
this network to land in the same AUC neighborhood as the other three models
(roughly 0.79-0.81) - not meaningfully better, and possibly a bit worse
before extensive tuning. That's the honest, expected outcome, and the
point of including it: a real comparison, not a guaranteed "win" for the
fanciest-sounding technique. If it DOES land clearly ahead after you run
it, that's a genuinely interesting result worth digging into further
(check for overfitting first).

Run from the repository root after `pip install torch`:
    python src/train_xg_model_deep.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit(
        "PyTorch is not installed. Run:  pip install torch\n"
        "(CPU-only is fine for this dataset size - no GPU required.)"
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "shots_raw.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

df = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
df["assist_type"] = df["assist_type"].replace("", "None").fillna("None")
df["shot_type"] = df["shot_type"].where(df["shot_type"].isin(["Open Play", "Free Kick", "Penalty"]), "Open Play")
model_df = df[df.shot_type != "Penalty"].copy().reset_index(drop=True)
model_df["distance_sq"] = model_df["distance"] ** 2
model_df["angle_x_distance"] = model_df["angle_deg"] * model_df["distance"]
model_df["log_distance"] = np.log1p(model_df["distance"])
model_df["header"] = (model_df["body_part"] == "Head").astype(int)
model_df["is_free_kick"] = (model_df["shot_type"] == "Free Kick").astype(int)

numeric_features = [
    "distance", "distance_sq", "angle_deg", "angle_x_distance", "log_distance",
    "under_pressure", "first_time", "one_on_one", "aerial_won",
    "n_opponents_close", "gk_positioned", "header", "is_free_kick",
]
categorical_features = ["technique", "play_pattern", "assist_type"]
X = model_df[numeric_features + categorical_features]
y = model_df["is_goal"].values

# Identical split (same seed, same test_size) to logreg/gboost/xgboost, so the
# AUC/Brier/LogLoss numbers below are directly comparable to model_comparison.csv
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])
preprocess.fit(X_train)
Xtr = preprocess.transform(X_train).astype(np.float32)
Xva = preprocess.transform(X_val).astype(np.float32)
Xte = preprocess.transform(X_test).astype(np.float32)
n_features = Xtr.shape[1]
print(f"Train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}, {n_features} input features")


class XgMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = XgMLP(n_features).to(device)

Xtr_t = torch.from_numpy(Xtr).to(device)
ytr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
Xva_t = torch.from_numpy(Xva).to(device)
yva_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
Xte_t = torch.from_numpy(Xte).to(device)

# Class imbalance (goal rate ~11%): weight the positive class in the loss
pos_weight = torch.tensor([(len(ytr_t) - ytr_t.sum()) / ytr_t.sum()], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 20
best_val_auc = -1.0
best_state = None
epochs_no_improve = 0
n_train = Xtr_t.shape[0]

t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(n_train, device=device)
    total_loss = 0.0
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = Xtr_t[idx], ytr_t[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        val_logits = model(Xva_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
    val_auc = roc_auc_score(y_val, val_probs)
    scheduler.step(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  train_loss={total_loss / n_train:.4f}  val_auc={val_auc:.4f}  best={best_val_auc:.4f}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
        break

print(f"Training done in {time.time() - t0:.1f}s, best val AUC {best_val_auc:.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(Xte_t)).cpu().numpy()

auc = roc_auc_score(y_test, test_probs)
brier = brier_score_loss(y_test, test_probs)
ll = log_loss(y_test, test_probs)
ap = average_precision_score(y_test, test_probs)
print(f"\nPyTorch MLP (held-out test set, n={len(y_test)}):")
print(f"  AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}  AvgPrec={ap:.4f}")

try:
    existing = pd.read_csv(os.path.join(BASE, "data", "model_comparison.csv"))
    print("\nFor comparison, the other models on the same test set:")
    print(existing.to_string(index=False))
except FileNotFoundError:
    pass

torch.save(model.state_dict(), os.path.join(MODELS, "deep_xg_model.pt"))
result_row = {"name": "PyTorch MLP (local)", "auc": auc, "brier": brier, "log_loss": ll, "avg_precision": ap}
out_path = os.path.join(BASE, "data", "deep_model_result.json")
json.dump(result_row, open(out_path, "w"), indent=2)
print(f"\nSaved models/deep_xg_model.pt and {os.path.relpath(out_path, BASE)}")
print("(This file is local-only and gitignored by default - see the note in README.md.)")

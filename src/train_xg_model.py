"""
Expected Goals (xG) model — v2
Dataset: StatsBomb open data — FIFA World Cup 2018 + Women's World Cup 2019 + UEFA Euro 2020
(167 matches, 4,309 shots, 2 genders, 3 confederJosé/international contexts)

Methodology upgrades over v1:
  - Multi-competition dataset (3x larger, cross-gender)
  - Extra features: one_on_one, aerial_won, assist_type (cross/through-ball/cut-back/corner)
  - Held-out test set + separate 5-fold CV hyperparameter search (no leakage into final metrics)
  - RandomizedSearchCV for both models
  - Bootstrap 95% confidence intervals on all metrics, incl. paired bootstrap for
    model-vs-model AUC difference significance
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              log_loss, average_precision_score)
from sklearn.calibration import calibration_curve
from scipy import stats as sp_stats
import joblib
import os

np.random.seed(42)

BASE = "/sessions/peaceful-exciting-albattani/mnt/outputs/xg_model_v2"
DATA = f"{BASE}/data/shots_raw.csv"
PLOTS = f"{BASE}/plots"
MODELS = f"{BASE}/models"
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

df = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
print("Raw shots:", df.shape)
print(df.groupby("competition_name").size())

# "None" is a real category for assist_type (no key pass), not a missing value
df["assist_type"] = df["assist_type"].replace("", "None").fillna("None")
# fold the 2 mislabeled/rare shot_type rows into Open Play (Corner/Kick Off — data quirks, n=2)
df["shot_type"] = df["shot_type"].where(df["shot_type"].isin(["Open Play", "Free Kick", "Penalty"]), "Open Play")

# ---- Penalties modeled separately: ~constant, situation-independent conversion rate ----
penalties = df[df.shot_type == "Penalty"].copy()
PENALTY_XG = penalties.is_goal.mean()
print(f"Penalty conversion rate (n={len(penalties)}): {PENALTY_XG:.3f}")

model_df = df[df.shot_type != "Penalty"].copy().reset_index(drop=True)
print("Shots used for modeling (non-penalty):", model_df.shape, "goal rate:", model_df.is_goal.mean())

# ---- Feature engineering ----
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

# ---- Held-out test set (untouched until final evaluation) ----
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, model_df.index, test_size=0.2, random_state=42, stratify=y
)
print("Train (for CV + tuning):", X_train.shape, "goal rate", y_train.mean())
print("Held-out test:", X_test.shape, "goal rate", y_test.mean())

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============ Logistic Regression: small grid over C ============
logreg_pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=3000))])
logreg_search = RandomizedSearchCV(
    logreg_pipe,
    param_distributions={"clf__C": np.logspace(-3, 2, 30)},
    n_iter=15, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
)
logreg_search.fit(X_train, y_train)
print("Best logreg C:", logreg_search.best_params_, "CV AUC:", logreg_search.best_score_)
logreg = logreg_search.best_estimator_

# ============ Gradient Boosting: randomized search ============
gboost_pipe = Pipeline([("prep", preprocess), ("clf", HistGradientBoostingClassifier(random_state=42))])
gboost_search = RandomizedSearchCV(
    gboost_pipe,
    param_distributions={
        "clf__max_depth": [3, 4, 5, 6, None],
        "clf__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
        "clf__max_iter": [100, 150, 200, 250, 300],
        "clf__l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0],
        "clf__min_samples_leaf": [10, 20, 30, 50],
    },
    n_iter=40, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
)
gboost_search.fit(X_train, y_train)
print("Best gboost params:", gboost_search.best_params_, "CV AUC:", gboost_search.best_score_)
gboost = gboost_search.best_estimator_

# ============ Cross-validated performance on the training set (model-selection honesty check) ============
def cv_report(name, pipe):
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=["roc_auc", "neg_brier_score", "neg_log_loss", "average_precision"])
    print(f"{name:20s} 5-fold CV  AUC={scores['test_roc_auc'].mean():.4f}±{scores['test_roc_auc'].std():.4f}  "
          f"Brier={-scores['test_neg_brier_score'].mean():.4f}±{scores['test_neg_brier_score'].std():.4f}  "
          f"LogLoss={-scores['test_neg_log_loss'].mean():.4f}±{scores['test_neg_log_loss'].std():.4f}")
    return {
        "name": name,
        "cv_auc_mean": scores["test_roc_auc"].mean(), "cv_auc_std": scores["test_roc_auc"].std(),
        "cv_brier_mean": -scores["test_neg_brier_score"].mean(), "cv_brier_std": scores["test_neg_brier_score"].std(),
        "cv_logloss_mean": -scores["test_neg_log_loss"].mean(), "cv_logloss_std": scores["test_neg_log_loss"].std(),
    }

cv_results = [cv_report("Logistic Regression", logreg_pipe.set_params(**logreg_search.best_params_)),
              cv_report("Gradient Boosting", gboost_pipe.set_params(**gboost_search.best_params_))]
pd.DataFrame(cv_results).to_csv(f"{BASE}/data/cv_results.csv", index=False)

# ---- Refit best models on the FULL training set ----
logreg.fit(X_train, y_train)
gboost.fit(X_train, y_train)

p_logreg = logreg.predict_proba(X_test)[:, 1]
p_gboost = gboost.predict_proba(X_test)[:, 1]
sb_xg_test = model_df.loc[idx_test, "statsbomb_xg"].fillna(model_df["statsbomb_xg"].median()).values

# ============ Final held-out test metrics ============
def evaluate(name, y_true, p):
    auc = roc_auc_score(y_true, p)
    brier = brier_score_loss(y_true, p)
    ll = log_loss(y_true, p)
    ap = average_precision_score(y_true, p)
    print(f"{name:25s} AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}  AvgPrec={ap:.4f}")
    return dict(name=name, auc=auc, brier=brier, log_loss=ll, avg_precision=ap)

results = [
    evaluate("Logistic Regression", y_test, p_logreg),
    evaluate("Gradient Boosting", y_test, p_gboost),
    evaluate("StatsBomb xG (reference)", y_test, sb_xg_test),
]
pd.DataFrame(results).to_csv(f"{BASE}/data/model_comparison.csv", index=False)

# ============ Bootstrap 95% CIs + significance test on AUC difference ============
N_BOOT = 3000
rng = np.random.RandomState(42)
n = len(y_test)
boot_auc = {"Logistic Regression": [], "Gradient Boosting": [], "StatsBomb xG": []}
boot_diff_gb_lr = []
boot_diff_gb_sb = []

preds = {"Logistic Regression": p_logreg, "Gradient Boosting": p_gboost, "StatsBomb xG": sb_xg_test}
for _ in range(N_BOOT):
    idx = rng.randint(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    aucs = {}
    for name, p in preds.items():
        aucs[name] = roc_auc_score(yb, p[idx])
        boot_auc[name].append(aucs[name])
    boot_diff_gb_lr.append(aucs["Gradient Boosting"] - aucs["Logistic Regression"])
    boot_diff_gb_sb.append(aucs["Gradient Boosting"] - aucs["StatsBomb xG"])

ci_rows = []
for name, vals in boot_auc.items():
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    ci_rows.append({"model": name, "auc_mean": vals.mean(), "ci_lo": lo, "ci_hi": hi})
    print(f"{name:20s} AUC 95% CI: [{lo:.4f}, {hi:.4f}]  (mean {vals.mean():.4f})")

diff_gb_lr = np.array(boot_diff_gb_lr)
diff_gb_sb = np.array(boot_diff_gb_sb)
p_gb_beats_lr = (diff_gb_lr > 0).mean()
p_gb_beats_sb = (diff_gb_sb > 0).mean()
lo_d, hi_d = np.percentile(diff_gb_lr, [2.5, 97.5])
lo_d2, hi_d2 = np.percentile(diff_gb_sb, [2.5, 97.5])
print(f"\nGradient Boosting - Logistic Regression AUC diff: {diff_gb_lr.mean():.4f}  "
      f"95% CI [{lo_d:.4f}, {hi_d:.4f}]  P(GB>LR)={p_gb_beats_lr:.3f}")
print(f"Gradient Boosting - StatsBomb xG AUC diff:        {diff_gb_sb.mean():.4f}  "
      f"95% CI [{lo_d2:.4f}, {hi_d2:.4f}]  P(GB>StatsBomb)={p_gb_beats_sb:.3f}")

pd.DataFrame(ci_rows).to_csv(f"{BASE}/data/bootstrap_ci.csv", index=False)
with open(f"{BASE}/data/significance_test.json", "w") as f:
    json.dump({
        "gb_minus_lr_mean": float(diff_gb_lr.mean()), "gb_minus_lr_ci": [float(lo_d), float(hi_d)],
        "p_gb_beats_lr": float(p_gb_beats_lr),
        "gb_minus_statsbomb_mean": float(diff_gb_sb.mean()), "gb_minus_statsbomb_ci": [float(lo_d2), float(hi_d2)],
        "p_gb_beats_statsbomb": float(p_gb_beats_sb),
        "n_bootstrap": N_BOOT,
    }, f, indent=2)

# ---- Save models ----
joblib.dump(logreg, f"{MODELS}/logreg_xg_model.joblib")
joblib.dump(gboost, f"{MODELS}/gboost_xg_model.joblib")
with open(f"{MODELS}/penalty_xg.json", "w") as f:
    json.dump({"penalty_xg": PENALTY_XG, "n_penalties": len(penalties)}, f)
with open(f"{MODELS}/best_hyperparameters.json", "w") as f:
    json.dump({"logreg": logreg_search.best_params_, "gboost": gboost_search.best_params_}, f, indent=2, default=str)

print("\nSaved models, CV results, bootstrap CIs, significance test.")

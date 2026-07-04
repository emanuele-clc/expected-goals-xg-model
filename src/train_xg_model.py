"""
Expected Goals (xG) model - v3
Dataset: StatsBomb open data across 8 competitions - FIFA World Cup 2018 & 2022,
Women's World Cup 2019, UEFA Euro 2020, UEFA Women's Euro 2022, Africa Cup of
Nations 2023, Copa America 2024, and La Liga 2015/16 (a full club season)
(726 matches, 17,886 shots, 2 genders, club + international contexts)

Run from anywhere; all paths are resolved relative to this repository
(the parent of this file's directory), so `python src/train_xg_model.py`
works out of the box using the shots_raw.csv already committed to the repo
- no need to re-download the StatsBomb data.

Methodology upgrades over v2:
  - Added a full club season (La Liga 2015/16, 380 matches) alongside the
    7 international tournaments, roughly doubling the dataset and adding a
    genuinely different shot-quality context (club football, ~38 games per
    team per season vs. a handful of tournament games)
  - Added XGBoost as a third model, tuned and compared head-to-head with
    logistic regression and gradient boosting
  - Full pairwise bootstrap significance testing across all 6 model pairs
    (previously only 2 comparisons were run)

Note on RandomizedSearchCV n_iter (15/20/30 for logreg/gboost/xgboost): these
were tuned to comfortably fit within a 2-CPU-core development sandbox in
well under a minute total. If you're running this on a machine with more
cores, feel free to raise them for a slightly more thorough search - expect
similar results, since each search had already started converging.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
import xgboost as xgb
import joblib

np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "shots_raw.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

df = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
print("Raw shots:", df.shape)
print(df.groupby("competition_name").size())

# "None" is a real category for assist_type (no key pass), not a missing value
df["assist_type"] = df["assist_type"].replace("", "None").fillna("None")
# fold the 2 mislabeled/rare shot_type rows into Open Play (Corner/Kick Off - data quirks, n=2)
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
    n_iter=20, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
)
gboost_search.fit(X_train, y_train)
print("Best gboost params:", gboost_search.best_params_, "CV AUC:", gboost_search.best_score_)
gboost = gboost_search.best_estimator_

# ============ XGBoost: randomized search ============
xgb_pipe = Pipeline([("prep", preprocess), ("clf", xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="auc"))])
xgb_search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions={
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
        "clf__n_estimators": [100, 150, 200, 250, 300],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "clf__min_child_weight": [1, 3, 5, 10],
    },
    n_iter=30, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
)
xgb_search.fit(X_train, y_train)
print("Best xgb params:", xgb_search.best_params_, "CV AUC:", xgb_search.best_score_)
xgb_model = xgb_search.best_estimator_

# ============ Cross-validated performance on the training set (model-selection honesty check) ============
def cv_report(name, pipe):
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=["roc_auc", "neg_brier_score", "neg_log_loss", "average_precision"])
    print(f"{name:20s} 5-fold CV  AUC={scores['test_roc_auc'].mean():.4f}+-{scores['test_roc_auc'].std():.4f}  "
          f"Brier={-scores['test_neg_brier_score'].mean():.4f}+-{scores['test_neg_brier_score'].std():.4f}  "
          f"LogLoss={-scores['test_neg_log_loss'].mean():.4f}+-{scores['test_neg_log_loss'].std():.4f}")
    return {
        "name": name,
        "cv_auc_mean": scores["test_roc_auc"].mean(), "cv_auc_std": scores["test_roc_auc"].std(),
        "cv_brier_mean": -scores["test_neg_brier_score"].mean(), "cv_brier_std": scores["test_neg_brier_score"].std(),
        "cv_logloss_mean": -scores["test_neg_log_loss"].mean(), "cv_logloss_std": scores["test_neg_log_loss"].std(),
    }

cv_results = [
    cv_report("Logistic Regression", logreg_pipe.set_params(**logreg_search.best_params_)),
    cv_report("Gradient Boosting", gboost_pipe.set_params(**gboost_search.best_params_)),
    cv_report("XGBoost", xgb_pipe.set_params(**xgb_search.best_params_)),
]
pd.DataFrame(cv_results).to_csv(os.path.join(BASE, "data", "cv_results.csv"), index=False)

# ---- Refit best models on the FULL training set ----
logreg.fit(X_train, y_train)
gboost.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

p_logreg = logreg.predict_proba(X_test)[:, 1]
p_gboost = gboost.predict_proba(X_test)[:, 1]
p_xgb = xgb_model.predict_proba(X_test)[:, 1]
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
    evaluate("XGBoost", y_test, p_xgb),
    evaluate("StatsBomb xG (reference)", y_test, sb_xg_test),
]
pd.DataFrame(results).to_csv(os.path.join(BASE, "data", "model_comparison.csv"), index=False)

# ============ Bootstrap 95% CIs + full pairwise significance testing ============
N_BOOT = 3000
rng = np.random.RandomState(42)
n = len(y_test)
preds = {"Logistic Regression": p_logreg, "Gradient Boosting": p_gboost, "XGBoost": p_xgb, "StatsBomb xG": sb_xg_test}
boot_auc = {k: [] for k in preds}
pairs = [
    ("XGBoost", "Logistic Regression"), ("XGBoost", "Gradient Boosting"),
    ("Logistic Regression", "Gradient Boosting"), ("XGBoost", "StatsBomb xG"),
    ("Logistic Regression", "StatsBomb xG"), ("Gradient Boosting", "StatsBomb xG"),
]
diffs = {p: [] for p in pairs}

for _ in range(N_BOOT):
    idx = rng.randint(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    aucs = {}
    for name, p in preds.items():
        a = roc_auc_score(yb, p[idx])
        boot_auc[name].append(a)
        aucs[name] = a
    for (a_name, b_name) in pairs:
        diffs[(a_name, b_name)].append(aucs[a_name] - aucs[b_name])

ci = {}
for name, vals in boot_auc.items():
    vals = np.array(vals)
    ci[name] = {"mean": float(vals.mean()), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}
    print(f"{name:20s} AUC 95% CI: [{ci[name]['ci_low']:.4f}, {ci[name]['ci_high']:.4f}]  (mean {ci[name]['mean']:.4f})")
json.dump(ci, open(os.path.join(BASE, "data", "bootstrap_ci.json"), "w"), indent=1)
pd.DataFrame([{"name": k, **v} for k, v in ci.items()]).to_csv(os.path.join(BASE, "data", "bootstrap_ci.csv"), index=False)

sig = {"n_boot": N_BOOT, "pairs": {}}
for (a_name, b_name), vals in diffs.items():
    vals = np.array(vals)
    p_a_beats_b = float((vals > 0).mean())
    key = f"{a_name} vs {b_name}"
    sig["pairs"][key] = {
        "mean_diff": float(vals.mean()),
        "ci_diff": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "p_a_beats_b": p_a_beats_b,
        "p_b_beats_a": 1 - p_a_beats_b,
    }
    print(f"P({a_name} beats {b_name}) = {p_a_beats_b:.3f}")
json.dump(sig, open(os.path.join(BASE, "data", "significance_test.json"), "w"), indent=1)

# ---- Save models ----
joblib.dump(logreg, os.path.join(MODELS, "logreg_xg_model.joblib"))
joblib.dump(gboost, os.path.join(MODELS, "gboost_xg_model.joblib"))
joblib.dump(xgb_model, os.path.join(MODELS, "xgboost_xg_model.joblib"))
with open(os.path.join(MODELS, "penalty_xg.json"), "w") as f:
    json.dump({"penalty_xg": PENALTY_XG, "n_penalties": len(penalties)}, f)
with open(os.path.join(MODELS, "best_hyperparameters.json"), "w") as f:
    json.dump({
        "logreg": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in logreg_search.best_params_.items()},
        "gboost": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in gboost_search.best_params_.items()},
        "xgboost": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in xgb_search.best_params_.items()},
    }, f, indent=2)

print("\nSaved models (logreg, gboost, xgboost), CV results, bootstrap CIs, pairwise significance tests.")
print("Next: run src/make_plots.py, src/export_feature_importance.py, src/team_player_performance.py, "
      "src/scouting_radar.py, and src/generate_dashboard_data.py to regenerate everything downstream.")

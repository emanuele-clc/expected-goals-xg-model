"""
Generates all visualizations for the xG model v2, using the saved models
and a fresh identical train/test split (same random_state as training,
so the held-out test set matches exactly).

Run from anywhere; all paths are resolved relative to this repository.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "shots_raw.csv")
PLOTS = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
os.makedirs(PLOTS, exist_ok=True)

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

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, model_df.index, test_size=0.2, random_state=42, stratify=y
)

logreg = joblib.load(os.path.join(MODELS, "logreg_xg_model.joblib"))
gboost = joblib.load(os.path.join(MODELS, "gboost_xg_model.joblib"))
xgboost_model = joblib.load(os.path.join(MODELS, "xgboost_xg_model.joblib"))

p_logreg = logreg.predict_proba(X_test)[:, 1]
p_gboost = gboost.predict_proba(X_test)[:, 1]
p_xgb = xgboost_model.predict_proba(X_test)[:, 1]
sb_xg_test = model_df.loc[idx_test, "statsbomb_xg"].fillna(model_df["statsbomb_xg"].median()).values

MODEL_PREDS = [
    ("Logistic Regression", p_logreg),
    ("Gradient Boosting", p_gboost),
    ("XGBoost", p_xgb),
    ("StatsBomb xG", sb_xg_test),
]
N_TRAIN, N_TEST, N_COMPS = len(X_train), len(X_test), df["competition_name"].nunique()

# 1. ROC curves
plt.figure(figsize=(6.5, 6.5))
for name, p in MODEL_PREDS:
    fpr, tpr, _ = roc_curve(y_test, p)
    auc = roc_auc_score(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - held-out test set (n={N_TEST:,})\n8 competitions: international tournaments + La Liga 2015/16")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "roc_curve.png"), dpi=150)
plt.close()

# 2. Calibration curves
plt.figure(figsize=(6.5, 6.5))
for name, p in MODEL_PREDS:
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
    plt.plot(mean_pred, frac_pos, "o-", label=name)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
plt.xlabel("Mean predicted xG")
plt.ylabel("Observed goal rate")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "calibration_curve.png"), dpi=150)
plt.close()

# 3. Bootstrap AUC distributions
n = len(y_test)
rng = np.random.RandomState(42)
boot_auc = {name: [] for name, _ in MODEL_PREDS}
preds_dict = dict(MODEL_PREDS)
for _ in range(3000):
    idx = rng.randint(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    for name, p in preds_dict.items():
        boot_auc[name].append(roc_auc_score(yb, p[idx]))

plt.figure(figsize=(7.5, 5.5))
colors = {"Logistic Regression": "tab:blue", "Gradient Boosting": "tab:orange",
          "XGBoost": "tab:purple", "StatsBomb xG": "tab:green"}
for name, vals in boot_auc.items():
    plt.hist(vals, bins=40, alpha=0.4, label=name, color=colors[name], density=True)
plt.xlabel("Bootstrap ROC AUC (n=3000 resamples of held-out test set)")
plt.ylabel("Density")
plt.title("Bootstrap distribution of test-set AUC by model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "bootstrap_auc_distribution.png"), dpi=150)
plt.close()

# 4. Feature importance - all three trained models
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, (name, pipe) in zip(axes, [("Logistic Regression", logreg), ("Gradient Boosting", gboost),
                                     ("XGBoost", xgboost_model)]):
    prep_fitted = pipe.named_steps["prep"]
    X_test_trans = prep_fitted.transform(X_test)
    feat_names = (numeric_features +
                  list(prep_fitted.named_transformers_["cat"].get_feature_names_out(categorical_features)))
    perm = permutation_importance(pipe.named_steps["clf"], X_test_trans, y_test,
                                   n_repeats=15, random_state=42, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[-12:]
    ax.barh(np.array(feat_names)[order], perm.importances_mean[order], xerr=perm.importances_std[order])
    ax.set_xlabel("Permutation importance (ROC AUC drop)")
    ax.set_title(name)
plt.suptitle("Top Feature Importances, All Three Models")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "feature_importance.png"), dpi=150)
plt.close()

# 5. Shot map (test set, best model = XGBoost)
def draw_pitch(ax):
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color="black")
    ax.plot([120, 102, 102, 120], [62, 62, 18, 18], color="black")
    ax.plot([120, 114, 114, 120], [50, 50, 30, 30], color="black")
    ax.plot([102, 102], [18, 62], color="black")
    circle = plt.Circle((108, 40), 10, color="black", fill=False)
    ax.add_patch(circle)
    ax.plot([120, 120], [36, 44], color="black", linewidth=3)
    ax.set_xlim(58, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")

test_plot_df = model_df.loc[idx_test].copy()
test_plot_df["pred_xg"] = p_xgb
test_plot_df["actual"] = test_plot_df["is_goal"]

fig, ax = plt.subplots(figsize=(10, 7))
draw_pitch(ax)
goals = test_plot_df[test_plot_df.actual == 1]
misses = test_plot_df[test_plot_df.actual == 0]
ax.scatter(misses.x, misses.y, s=misses.pred_xg * 800 + 15, c="tab:red",
           alpha=0.5, edgecolor="k", linewidth=0.3, label="No goal")
ax.scatter(goals.x, goals.y, s=goals.pred_xg * 800 + 15, c="tab:green",
           alpha=0.8, edgecolor="k", linewidth=0.5, label="Goal")
ax.set_title(f"Test-set shots - marker size = predicted xG (XGBoost, best model)\n"
             f"8 competitions ({N_COMPS} tournaments/leagues), n={N_TEST:,} test shots")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "shot_map.png"), dpi=150)
plt.close()

# 6. Our best model's xG vs StatsBomb's own xG
plt.figure(figsize=(6.5, 6.5))
plt.scatter(sb_xg_test, p_xgb, alpha=0.3, s=14)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("StatsBomb published xG")
plt.ylabel("Our XGBoost xG")
corr = np.corrcoef(sb_xg_test, p_xgb)[0, 1]
plt.title(f"Model validation: our xG vs. StatsBomb's own xG\nPearson r = {corr:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "xg_vs_statsbomb.png"), dpi=150)
plt.close()

# 7. xG vs distance
plt.figure(figsize=(7.5, 5.5))
plt.scatter(model_df.distance, model_df.statsbomb_xg, s=5, alpha=0.2)
plt.xlabel("Distance to goal (yards)")
plt.ylabel("StatsBomb xG")
plt.title("xG vs. Shot Distance - all non-penalty shots (n={:,})".format(len(model_df)))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "xg_vs_distance.png"), dpi=150)
plt.close()

# 8. Dataset composition
plt.figure(figsize=(8, 5.5))
comp_counts = df.groupby(["competition_name", "gender"]).size().unstack(fill_value=0)
comp_counts.plot(kind="bar", stacked=True, ax=plt.gca(), color=["#3b82f6", "#f472b6"])
plt.ylabel("Number of shots")
plt.title("Dataset composition by competition and gender")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "dataset_composition.png"), dpi=150)
plt.close()

print("All plots saved to", PLOTS)

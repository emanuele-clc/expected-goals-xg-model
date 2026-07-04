"""
Exports the top-8 permutation feature importances for the XGBoost model
(the best of the three trained models) as JSON, for use in the dashboard's
Feature Importance chart. make_plots.py produces the equivalent PNG for all
three models; this script produces a small machine-readable version of just
the best model's numbers, since embedding a chart image isn't as flexible
as an interactive Chart.js bar chart in docs/index.html.

Run from anywhere; all paths are resolved relative to this repository.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "shots_raw.csv")
MODELS = os.path.join(BASE, "models")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model = joblib.load(os.path.join(MODELS, "xgboost_xg_model.joblib"))
prep = xgb_model.named_steps["prep"]
X_test_trans = prep.transform(X_test)
feat_names = numeric_features + list(prep.named_transformers_["cat"].get_feature_names_out(categorical_features))

perm = permutation_importance(xgb_model.named_steps["clf"], X_test_trans, y_test,
                               n_repeats=10, random_state=42, scoring="roc_auc")
order = np.argsort(perm.importances_mean)[::-1][:8]
top = [[feat_names[i], round(float(perm.importances_mean[i]), 4)] for i in order]

json.dump(top, open(os.path.join(BASE, "data", "xgb_feature_importance_top8.json"), "w"))
print("Top 8 XGBoost feature importances:", top)
print("Saved data/xgb_feature_importance_top8.json")

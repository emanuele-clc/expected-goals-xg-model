"""
Generates all data files consumed by docs/index.html:
  - data/leaderboards_by_comp.json   {comp_code: {team: [...], player: [...]}}
  - data/shot_sample_with_comp.json  [[x, y, is_goal, pred_xg, body_part, comp_code], ...]
  - data/player_shots_nested.json    {comp_code: {team_name: {player_name: [[x,y,is_goal,xg,body_part],...]}}}
  - data/logreg_coeffs.json          scaler/coef/categories for the client-side "Try It Yourself" calculator

Applies the trained XGBoost xG model (best of the three trained models) to
every shot for the retrospective leaderboards / shot maps, and reuses the
exact held-out test split (random_state=42) for the shot-map sample so the
sample's predicted xG values are genuine out-of-sample predictions, not
overfit ones. The logistic regression model's coefficients are separately
exported for the client-side "Try It Yourself" calculator, since only a
linear model's coefficients can be faithfully replicated in plain JS
(tree ensembles like XGBoost cannot).

Run from anywhere; all paths are resolved relative to this repository.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SHOTS_PLAYER = 8
SAMPLE_SIZE = 300
RNG_SEED = 42

# (competition_name, season_name) -> (comp_code, display label)
COMP_MAP = {
    ("FIFA World Cup", "2018"): ("WC18", "World Cup 2018"),
    ("Women's World Cup", "2019"): ("WWC19", "Women's World Cup 2019"),
    ("UEFA Euro", "2020"): ("EURO20", "Euro 2020"),
    ("FIFA World Cup", "2022"): ("WC22", "World Cup 2022"),
    ("UEFA Women's Euro", "2022"): ("WEURO22", "Women's Euro 2022"),
    ("Copa America", "2024"): ("COPA24", "Copa America 2024"),
    ("African Cup of Nations", "2023"): ("AFCON23", "Africa Cup of Nations 2023"),
    ("La Liga", "2015/2016"): ("LALIGA1516", "La Liga 2015/16"),
}

df = pd.read_csv(os.path.join(BASE, "data", "shots_raw.csv"), keep_default_na=False, na_values=[""])
df["assist_type"] = df["assist_type"].replace("", "None").fillna("None")
df["shot_type"] = df["shot_type"].where(df["shot_type"].isin(["Open Play", "Free Kick", "Penalty"]), "Open Play")
df["season_name"] = df["season_name"].astype(str)

def comp_code_of(row):
    key = (row["competition_name"], row["season_name"])
    return COMP_MAP.get(key, ("UNK", "Unknown"))[0]

df["comp_code"] = df.apply(comp_code_of, axis=1)
unknown = (df["comp_code"] == "UNK").sum()
if unknown:
    print(f"WARNING: {unknown} shots did not match any known (competition_name, season_name) pair")

penalty_info = json.load(open(os.path.join(BASE, "models", "penalty_xg.json")))
PENALTY_XG = penalty_info["penalty_xg"]

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

logreg = joblib.load(os.path.join(BASE, "models", "logreg_xg_model.joblib"))
xgboost_model = joblib.load(os.path.join(BASE, "models", "xgboost_xg_model.joblib"))
model_df["xg"] = xgboost_model.predict_proba(model_df[numeric_features + categorical_features])[:, 1]

penalties = df[df.shot_type == "Penalty"].copy()
penalties["xg"] = PENALTY_XG

full = pd.concat([
    model_df[["match_id", "comp_code", "player_name", "team_name", "is_goal", "xg", "x", "y", "body_part"]],
    penalties[["match_id", "comp_code", "player_name", "team_name", "is_goal", "xg", "x", "y", "body_part"]],
], ignore_index=True)

print(f"Full shot set for aggregation: {len(full)} shots ({full.is_goal.sum()} goals, sum xG = {full.xg.sum():.1f})")
print(full.groupby("comp_code").size())

# ---------------- Leaderboards (All + per competition) ----------------
def build_leaderboards(subset):
    team = subset.groupby("team_name").agg(
        shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"),
    ).reset_index()
    team["xg_diff"] = team["goals"] - team["xg"]
    team = team.sort_values("xg_diff", ascending=False).reset_index(drop=True)
    team_rows = [
        [r.team_name, int(r.shots), int(r.goals), round(float(r.xg), 2), round(float(r.xg_diff), 2)]
        for r in team.itertuples()
    ]

    player = subset.groupby(["player_name", "team_name"]).agg(
        shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"),
    ).reset_index()
    player = player[player.shots >= MIN_SHOTS_PLAYER].copy()
    player["xg_diff"] = player["goals"] - player["xg"]
    player = player.sort_values("xg_diff", ascending=False).reset_index(drop=True)
    player_rows = [
        [r.player_name, r.team_name, int(r.shots), int(r.goals), round(float(r.xg), 2), round(float(r.xg_diff), 2)]
        for r in player.itertuples()
    ]
    return {"team": team_rows, "player": player_rows}

leaderboards = {"All": build_leaderboards(full)}
for code, _label in sorted(set(COMP_MAP.values())):
    leaderboards[code] = build_leaderboards(full[full.comp_code == code])

json.dump(leaderboards, open(os.path.join(BASE, "data", "leaderboards_by_comp.json"), "w"))
print("Saved data/leaderboards_by_comp.json with keys:", list(leaderboards.keys()))

# ---------------- Shot-map sample (genuine held-out test predictions) ----------------
X = model_df[numeric_features + categorical_features]
y = model_df["is_goal"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, model_df.index, test_size=0.2, random_state=RNG_SEED, stratify=y)

test_df = model_df.loc[idx_test].copy()
test_df["pred_xg"] = xgboost_model.predict_proba(X_test)[:, 1]

rng = np.random.RandomState(RNG_SEED)
sample_idx = rng.choice(test_df.index, size=min(SAMPLE_SIZE, len(test_df)), replace=False)
sample = test_df.loc[sample_idx]

shot_sample = [
    [round(float(r.x), 1), round(float(r.y), 1), int(r.is_goal), round(float(r.pred_xg), 3), r.body_part, r.comp_code]
    for r in sample.itertuples()
]
json.dump(shot_sample, open(os.path.join(BASE, "data", "shot_sample_with_comp.json"), "w"))
print(f"Saved data/shot_sample_with_comp.json with {len(shot_sample)} shots")

# ---------------- Per-player shot maps (full history, all shots incl. penalties) ----------------
nested = {}
for code, _label in COMP_MAP.values():
    sub = full[full.comp_code == code]
    by_team = {}
    for team_name, tgrp in sub.groupby("team_name"):
        by_player = {}
        for player_name, pgrp in tgrp.groupby("player_name"):
            by_player[player_name] = [
                [round(float(r.x), 1), round(float(r.y), 1), int(r.is_goal), round(float(r.xg), 3), r.body_part]
                for r in pgrp.itertuples()
            ]
        by_team[team_name] = by_player
    nested[code] = by_team

json.dump(nested, open(os.path.join(BASE, "data", "player_shots_nested.json"), "w"))
size_kb = os.path.getsize(os.path.join(BASE, "data", "player_shots_nested.json")) / 1024
n_combos = sum(len(players) for teams in nested.values() for players in teams.values())
print(f"Saved data/player_shots_nested.json ({size_kb:.1f} KB, {n_combos} team/player combos)")

# ---------------- Client-side calculator coefficients ----------------
prep = logreg.named_steps["prep"]
clf = logreg.named_steps["clf"]
scaler = prep.named_transformers_["num"]
ohe = prep.named_transformers_["cat"]

cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
categories = {feat: list(cats) for feat, cats in zip(categorical_features, ohe.categories_)}
all_feat_names = numeric_features + cat_feature_names

coeffs = {
    "numeric_features": numeric_features,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "cat_feature_names": cat_feature_names,
    "categories": categories,
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
    "all_feat_names": all_feat_names,
}
json.dump(coeffs, open(os.path.join(BASE, "data", "logreg_coeffs.json"), "w"))
print("Saved data/logreg_coeffs.json")

print("\nAll dashboard data files regenerated.")

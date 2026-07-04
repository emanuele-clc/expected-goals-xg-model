"""
Team and player xG performance report.

Applies the trained XGBoost xG model (the best-performing of the three
trained models on this larger, more diverse dataset) to every shot in the
dataset (not just the held-out test set - this is a retrospective descriptive
analysis, not a predictive evaluation) to compute, for every team and player
across all 8 competitions: actual goals scored vs. total expected goals (xG),
and the gap between them (over/underperformance relative to underlying
chance quality).

Run from anywhere; all paths are resolved relative to this repository.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SHOTS_PLAYER = 8

df = pd.read_csv(os.path.join(BASE, "data", "shots_raw.csv"), keep_default_na=False, na_values=[""])
df["assist_type"] = df["assist_type"].replace("", "None").fillna("None")
df["shot_type"] = df["shot_type"].where(df["shot_type"].isin(["Open Play", "Free Kick", "Penalty"]), "Open Play")

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

xgboost_model = joblib.load(os.path.join(BASE, "models", "xgboost_xg_model.joblib"))
model_df["xg"] = xgboost_model.predict_proba(model_df[numeric_features + categorical_features])[:, 1]

# Penalties get the fixed empirical conversion rate as their xG
penalties = df[df.shot_type == "Penalty"].copy()
penalties["xg"] = PENALTY_XG

full = pd.concat([
    model_df[["match_id", "competition_name", "player_name", "team_name", "is_goal", "xg"]],
    penalties[["match_id", "competition_name", "player_name", "team_name", "is_goal", "xg"]],
], ignore_index=True)

print(f"Full shot set for aggregation: {len(full)} shots ({full.is_goal.sum()} goals, "
      f"sum xG = {full.xg.sum():.1f})")

# ---------------- Team-level aggregation ----------------
team = full.groupby("team_name").agg(
    shots=("xg", "size"),
    goals=("is_goal", "sum"),
    xg=("xg", "sum"),
).reset_index()
team["xg_diff"] = team["goals"] - team["xg"]
team["xg_per_shot"] = (team["xg"] / team["shots"]).round(3)
team["conversion_rate"] = (team["goals"] / team["shots"]).round(3)
team = team.sort_values("xg_diff", ascending=False).reset_index(drop=True)
team = team.round({"xg": 2, "xg_diff": 2})
team.to_csv(os.path.join(BASE, "data", "team_performance.csv"), index=False)

# ---------------- Player-level aggregation ----------------
player = full.groupby(["player_name", "team_name"]).agg(
    shots=("xg", "size"),
    goals=("is_goal", "sum"),
    xg=("xg", "sum"),
).reset_index()
player = player[player.shots >= MIN_SHOTS_PLAYER].copy()
player["xg_diff"] = player["goals"] - player["xg"]
player["xg_per_shot"] = (player["xg"] / player["shots"]).round(3)
player = player.sort_values("xg_diff", ascending=False).reset_index(drop=True)
player = player.round({"xg": 2, "xg_diff": 2})
player.to_csv(os.path.join(BASE, "data", "player_performance.csv"), index=False)

print(f"\n=== TEAM xG OVER-PERFORMANCE (top 5, min {team.shots.min()} shots) ===")
print(team.head(5)[["team_name", "shots", "goals", "xg", "xg_diff"]].to_string(index=False))
print(f"\n=== TEAM xG UNDER-PERFORMANCE (bottom 5) ===")
print(team.tail(5)[["team_name", "shots", "goals", "xg", "xg_diff"]].to_string(index=False))

print(f"\n=== PLAYER xG OVER-PERFORMANCE (top 10, min {MIN_SHOTS_PLAYER} shots) ===")
print(player.head(10)[["player_name", "team_name", "shots", "goals", "xg", "xg_diff"]].to_string(index=False))
print(f"\n=== PLAYER xG UNDER-PERFORMANCE (bottom 10) ===")
print(player.tail(10)[["player_name", "team_name", "shots", "goals", "xg", "xg_diff"]].to_string(index=False))

print("\nSaved: data/team_performance.csv, data/player_performance.csv")

"""
Scouting Radar: a statistically-robust "clinical finishing" index.

Standard xG leaderboards rank players by raw over-performance (goals - xG).
The problem: raw over-performance is noisy for players with few shots, so a
player who scored 3 goals from 1.2 xG on just 10 shots ranks above a player
who scored 25 goals from 20 xG on 200 shots, even though the second player's
edge is far more likely to be a repeatable skill rather than a hot streak.

This script fixes that by bootstrap-resampling each player's own shots
(2000 resamples, sampling shot-level (goal - xG) residuals with replacement)
and taking the 2.5th percentile of the resulting distribution of mean
per-shot over-performance as a conservative "skill floor": a value that
stays positive only if the player over-performs their underlying shot
quality even under a pessimistic reading of their sample.

Ranking by this floor (instead of the raw mean) is exactly the kind of
volume-aware, conservative estimate a recruitment/scouting analytics team
would want: it surfaces players whose finishing edge is unlikely to be
noise, and it does NOT depend on transfer fees or reputation, so it will
naturally surface less famous players (from smaller teams / less-hyped
competitions) whose underlying shot conversion skill is genuine (a useful
starting point for identifying undervalued attacking talent).

Run from anywhere; all paths are resolved relative to this repository.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SHOTS = 15
N_BOOT = 2000
RNG_SEED = 42

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

penalties = df[df.shot_type == "Penalty"].copy()
penalties["xg"] = PENALTY_XG

full = pd.concat([
    model_df[["match_id", "competition_name", "player_name", "team_name", "is_goal", "xg"]],
    penalties[["match_id", "competition_name", "player_name", "team_name", "is_goal", "xg"]],
], ignore_index=True)
full["residual"] = full["is_goal"] - full["xg"]

rng = np.random.RandomState(RNG_SEED)
rows = []
for (player_name, team_name), grp in full.groupby(["player_name", "team_name"]):
    n = len(grp)
    if n < MIN_SHOTS:
        continue
    resid = grp["residual"].values
    goals = int(grp["is_goal"].sum())
    xg_sum = float(grp["xg"].sum())
    n_matches = grp["match_id"].nunique()
    comps = sorted(grp["competition_name"].unique().tolist())

    idx = rng.randint(0, n, size=(N_BOOT, n))
    boot_means = resid[idx].mean(axis=1)
    floor = float(np.percentile(boot_means, 2.5))
    ceiling = float(np.percentile(boot_means, 97.5))
    mean_per_shot = float(resid.mean())

    rows.append({
        "player_name": player_name,
        "team_name": team_name,
        "competitions": comps,
        "shots": n,
        "matches": int(n_matches),
        "goals": goals,
        "xg": round(xg_sum, 2),
        "xg_diff": round(goals - xg_sum, 2),
        "per_shot_overperf": round(mean_per_shot, 4),
        "floor_95ci": round(floor, 4),
        "ceiling_95ci": round(ceiling, 4),
        "reliably_positive": bool(floor > 0),
    })

radar = pd.DataFrame(rows).sort_values("floor_95ci", ascending=False).reset_index(drop=True)
radar.to_csv(os.path.join(BASE, "data", "scouting_radar.csv"), index=False)

n_reliable = int(radar["reliably_positive"].sum())
print(f"Scouting Radar: {len(radar)} players with >= {MIN_SHOTS} shots evaluated "
      f"(bootstrap N={N_BOOT} per player)")
print(f"{n_reliable} players ({100*n_reliable/len(radar):.1f}%) have a 95% CI finishing floor "
      f"strictly above zero (reliable over-performers, not just hot streaks)")

print("\n=== TOP 15 SCOUTING RADAR HITS (ranked by conservative 95% CI floor) ===")
print(radar.head(15)[["player_name", "team_name", "shots", "goals", "xg", "xg_diff",
                        "per_shot_overperf", "floor_95ci"]].to_string(index=False))

# JSON export for the dashboard (top 40 + full reliable set)
top_json = radar.head(40).to_dict(orient="records")
json.dump({
    "min_shots": MIN_SHOTS,
    "n_boot": N_BOOT,
    "n_evaluated": len(radar),
    "n_reliable": n_reliable,
    "top": top_json,
}, open(os.path.join(BASE, "data", "scouting_radar.json"), "w"))
print("\nSaved data/scouting_radar.csv and data/scouting_radar.json")

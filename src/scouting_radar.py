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
import zlib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SHOTS = 15
N_BOOT = 2000
RNG_SEED = 42

def player_rng(player_name, team_name):
    """A fresh, deterministically-seeded RandomState per player, so a player's
    bootstrap draw (and therefore their floor/ceiling) is identical whether
    they're evaluated in the pooled 'All' view or a single-competition
    subset - reproducibility must not depend on which other players happen
    to be grouped alongside them or on iteration order."""
    seed = (RNG_SEED + zlib.crc32((player_name + "||" + team_name).encode())) % (2**31)
    return np.random.RandomState(seed)

# (competition_name, season_name) -> (comp_code, display label) - same mapping used
# throughout the project (generate_dashboard_data.py, team_player_performance.py)
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
    model_df[["match_id", "competition_name", "comp_code", "player_name", "team_name", "is_goal", "xg"]],
    penalties[["match_id", "competition_name", "comp_code", "player_name", "team_name", "is_goal", "xg"]],
], ignore_index=True)
full["residual"] = full["is_goal"] - full["xg"]

def compute_radar(subset):
    """Bootstrap-based conservative skill-floor ranking for a given shot subset."""
    rows = []
    for (player_name, team_name), grp in subset.groupby(["player_name", "team_name"]):
        n = len(grp)
        if n < MIN_SHOTS:
            continue
        resid = grp["residual"].values
        goals = int(grp["is_goal"].sum())
        xg_sum = float(grp["xg"].sum())
        n_matches = grp["match_id"].nunique()
        comps = sorted(grp["competition_name"].unique().tolist())

        rng = player_rng(player_name, team_name)
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
    if not rows:
        return pd.DataFrame(columns=["player_name", "team_name", "competitions", "shots", "matches",
                                      "goals", "xg", "xg_diff", "per_shot_overperf", "floor_95ci",
                                      "ceiling_95ci", "reliably_positive"])
    return pd.DataFrame(rows).sort_values("floor_95ci", ascending=False).reset_index(drop=True)

# Pooled ("All") radar - unchanged behavior, a player's shots across every
# competition they appear in are combined into one estimate.
radar = compute_radar(full)
radar.to_csv(os.path.join(BASE, "data", "scouting_radar.csv"), index=False)

n_reliable = int(radar["reliably_positive"].sum())
print(f"Scouting Radar (All): {len(radar)} players with >= {MIN_SHOTS} shots evaluated "
      f"(bootstrap N={N_BOOT} per player)")
print(f"{n_reliable} players ({100*n_reliable/len(radar):.1f}%) have a 95% CI finishing floor "
      f"strictly above zero (reliable over-performers, not just hot streaks)")

print("\n=== TOP 15 SCOUTING RADAR HITS, ALL COMPETITIONS POOLED (ranked by 95% CI floor) ===")
print(radar.head(15)[["player_name", "team_name", "shots", "goals", "xg", "xg_diff",
                        "per_shot_overperf", "floor_95ci"]].to_string(index=False))

# Per-competition radar - lets the dashboard filter/compare the Scouting Radar
# by competition exactly like the Team/Player Leaderboards already do. Most
# tournaments will have few or zero qualifying players (MIN_SHOTS=15 is a high
# bar for a single tournament) - that's an honest reflection of statistical
# power, not a bug: only a full club season currently has enough shots/player.
by_comp = {"All": radar}
for code, _label in sorted(set(COMP_MAP.values())):
    by_comp[code] = compute_radar(full[full.comp_code == code])
    print(f"Scouting Radar ({code}): {len(by_comp[code])} evaluated, "
          f"{int(by_comp[code]['reliably_positive'].sum()) if len(by_comp[code]) else 0} reliable")

# JSON export for the dashboard (top 40 per competition, keyed like leaderboards_by_comp.json)
radar_by_comp_json = {}
for code, rdf in by_comp.items():
    n_rel = int(rdf["reliably_positive"].sum()) if len(rdf) else 0
    radar_by_comp_json[code] = {
        "min_shots": MIN_SHOTS,
        "n_boot": N_BOOT,
        "n_evaluated": len(rdf),
        "n_reliable": n_rel,
        "top": rdf.head(40).to_dict(orient="records"),
    }
json.dump(radar_by_comp_json, open(os.path.join(BASE, "data", "scouting_radar.json"), "w"))
print("\nSaved data/scouting_radar.csv (All, pooled) and data/scouting_radar.json (per-competition + All)")

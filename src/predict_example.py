"""
Quick example: load all three trained xG models (logistic regression,
gradient boosting, XGBoost) and score a handful of hand-made shots side by
side, without needing to re-download any StatsBomb data.

Run with:  python src/predict_example.py
"""
import math
import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = {
    "LogReg": os.path.join(BASE, "models", "logreg_xg_model.joblib"),
    "GBoost": os.path.join(BASE, "models", "gboost_xg_model.joblib"),
    "XGBoost": os.path.join(BASE, "models", "xgboost_xg_model.joblib"),
}

GOAL_X, GOAL_Y = 120.0, 40.0
POST1, POST2 = (120.0, 36.0), (120.0, 44.0)


def angle_to_goal(x, y):
    a = math.dist((x, y), POST1)
    b = math.dist((x, y), POST2)
    c = 8.0
    cos_angle = max(-1, min(1, (a**2 + b**2 - c**2) / (2 * a * b)))
    return math.degrees(math.acos(cos_angle))


def make_shot(x, y, body_part="Right Foot", technique="Normal", play_pattern="Regular Play",
              assist_type="OtherPass", under_pressure=0, first_time=0, one_on_one=0,
              aerial_won=0, n_opponents_close=0, gk_positioned=1, shot_type="Open Play"):
    distance = math.dist((x, y), (GOAL_X, GOAL_Y))
    angle = angle_to_goal(x, y)
    return pd.DataFrame([{
        "distance": distance,
        "distance_sq": distance ** 2,
        "angle_deg": angle,
        "angle_x_distance": angle * distance,
        "log_distance": math.log1p(distance),
        "under_pressure": under_pressure,
        "first_time": first_time,
        "one_on_one": one_on_one,
        "aerial_won": aerial_won,
        "n_opponents_close": n_opponents_close,
        "gk_positioned": gk_positioned,
        "header": int(body_part == "Head"),
        "is_free_kick": int(shot_type == "Free Kick"),
        "technique": technique,
        "play_pattern": play_pattern,
        "assist_type": assist_type,
    }])


if __name__ == "__main__":
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

    examples = {
        "Penalty spot, calm (no pressure, 1v1 not counted)": make_shot(108, 40, n_opponents_close=0),
        "Edge of the box, under pressure, 2 defenders close": make_shot(102, 40, under_pressure=1, n_opponents_close=2),
        "Tight angle, near the byline": make_shot(115, 10),
        "Six-yard box header from a cross": make_shot(116, 36, body_part="Head", assist_type="Cross", first_time=1),
        "Long-range effort, 25 yards out, central": make_shot(95, 40),
    }

    print(f"{'Scenario':55s} {'LogReg':>8s} {'GBoost':>8s} {'XGBoost':>8s}")
    print("-" * 82)
    for name, X in examples.items():
        preds = [models[m].predict_proba(X)[0, 1] for m in ("LogReg", "GBoost", "XGBoost")]
        print(f"{name:55s} {preds[0]:8.3f} {preds[1]:8.3f} {preds[2]:8.3f}")

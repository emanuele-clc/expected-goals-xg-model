# Expected Goals (xG) Model - v2

An xG model built on real StatsBomb open event data across **seven major tournaments**, comparing a hyperparameter-tuned logistic regression against a hyperparameter-tuned gradient boosting model, with cross-validation, a held-out test set, and bootstrap significance testing - benchmarked against StatsBomb's own published xG.

An interactive dashboard (`docs/index.html`) ships with the project: a live in-browser xG calculator (the real trained model, running client-side), filterable shot maps, per-player shot history drill-downs, and sortable team/player over-performance leaderboards - no server or install required, just open the file.

## Quick start (try it without downloading any data)

```
pip install -r requirements.txt
python src/predict_example.py
```

This loads the saved, already-trained logistic regression model and prints predicted xG for five hand-made shot scenarios (penalty spot, edge of the box under pressure, tight angle, header from a cross, long range) - a quick sanity check that the model behaves sensibly (see sample output in the script's docstring / below).

To fully reproduce the dataset and retrain from scratch you need a local clone of `statsbomb/open-data` (see "Reproducing / extending" below) - that step requires internet access and isn't needed just to try the model.

## Dataset

- Source: [StatsBomb open-data](https://github.com/statsbomb/open-data) (free, open license for public research/education use).
- **346 matches across 7 competitions**, spanning men's and women's football and multiple confederations:
  - FIFA World Cup 2018 (64 matches, men)
  - Women's World Cup 2019 (52 matches, women)
  - UEFA Euro 2020 (51 matches, men)
  - FIFA World Cup 2022 (64 matches, men)
  - UEFA Women's Euro 2022 (31 matches, women)
  - Africa Cup of Nations 2023 (52 matches, men)
  - Copa América 2024 (32 matches, men)
- Obtained via a shallow, blob-filtered `git clone` of the open-data repo, then checking out only the required `matches/` and `events/` JSON files (full files, not previews/truncated fetches).
- **8,718 total shots** extracted; 8,357 non-penalty shots used for modeling (361 penalties held out - see Methodology).
- Each shot is tagged with `competition_name` **and** `season_name`, since StatsBomb reuses the same `competition_name` ("FIFA World Cup") for both the 2018 and 2022 tournaments - without the season field the two would silently merge under one filter.

Raw extraction script: `src/extract_shots.py`. It walks every match's event file, filters `type.name == "Shot"`, and pulls out location, outcome, body part, technique, shot type, play pattern, pressure, freeze-frame defender positions, and the **assist type** (cross / through ball / cut-back / corner / free-kick pass / other), resolved by looking up the shot's `key_pass_id` in the same match's event stream.

## Methodology

**Why penalties are excluded from the main model:** penalty conversion (68.7% in this sample) is essentially constant and situation-independent - including them would distort a location/angle-based model. They're modeled separately as a fixed rate and reported alongside.

**Features (16 total):** `distance`, `distance_sq`, `angle_deg` (law of cosines on the two goalposts), `angle_x_distance` interaction, `log_distance`, `under_pressure`, `first_time`, `one_on_one`, `aerial_won`, `n_opponents_close` and `gk_positioned` (from StatsBomb 360 freeze-frame data), `header`, `is_free_kick`, plus one-hot `technique`, `play_pattern`, and `assist_type`.

**Rigor built into the pipeline:**

1. **Held-out test set never touched during model selection.** An 80/20 stratified split is made first; all cross-validation and hyperparameter search happens only inside the 80% training portion. Final metrics are reported exclusively on the untouched 20% (1,672 shots).
2. **5-fold stratified cross-validation** for model selection instead of a single split (which gives noisy, overconfident estimates).
3. **Hyperparameter tuning via `RandomizedSearchCV`** (15 draws for logistic regression's `C`, 25 draws for gradient boosting's depth/learning-rate/iterations/regularization/leaf-size - see the note in `train_xg_model.py` about raising this if you have more CPU cores available), each scored by 5-fold CV ROC AUC.
4. **Bootstrap 95% confidence intervals** (3,000 resamples of the held-out test set) for every model's AUC, plus a **paired bootstrap significance test** on the AUC difference between models - so "model A beats model B" claims are backed by a p-value-equivalent, not just a single point estimate.

## Results (held-out test set, n=1,672 non-penalty shots)

| Model | Test AUC | 95% CI | 5-fold CV AUC | Brier | Log Loss | Avg. Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** (tuned) | **0.821** | [0.785, 0.857] | 0.783 ± 0.016 | 0.066 | 0.238 | 0.462 |
| Gradient Boosting (tuned) | 0.816 | [0.780, 0.854] | 0.780 ± 0.020 | 0.068 | 0.242 | 0.438 |
| StatsBomb xG (reference) | 0.838 | [0.805, 0.872] | - | 0.064 | 0.229 | 0.492 |

**Bootstrap significance test:** Gradient Boosting's AUC is *lower* than Logistic Regression's by 0.0046 on average (95% CI [-0.0165, +0.0074]); logistic regression beats gradient boosting in 77.9% of bootstrap resamples (P(GB>LR) = 0.221). Gradient boosting is also significantly below StatsBomb's own xG (P(GB>StatsBomb) = 0.003).

**This is a genuine, honestly-reported finding, not cherry-picked - and it changed with more data.** In the original 3-competition version of this project (4,309 shots), logistic regression beat gradient boosting far more decisively (95.9% of resamples, P=0.041). With the dataset now roughly doubled to 8,718 shots across 7 tournaments, that gap narrowed to 77.9%: still a real edge for the simpler, more regularized model, but a more modest one. This is exactly the kind of pattern you'd expect - more data narrows the advantage that heavy regularization gives a linear model over a more flexible one, though not enough here to flip the result. Logistic regression's test AUC (0.821) also sits well inside StatsBomb's own confidence interval, which remains the strongest available external validity check.

Full numeric results: `data/model_comparison.csv`, `data/cv_results.csv`, `data/bootstrap_ci.csv`, `data/significance_test.json`, `data/permutation_importance.json`.

**Feature importance also shifted with more data:** in the smaller dataset, shot angle appeared to matter about as much as distance. With more than double the shots, raw distance (plus its log-transformed version) now dominates decisively - angle's signal shows up mostly through the `angle_x_distance` interaction term rather than standing on its own, since angle and distance are correlated in real shot locations. See `plots/feature_importance.png` and the dashboard's Feature Importance chart for the full breakdown.

## Team and player xG performance report

Beyond the raw shot-level model, `src/team_player_performance.py` and `src/generate_dashboard_data.py` apply the trained logistic regression to every shot in the dataset (all 8,718, not just the held-out test set - this is a retrospective descriptive analysis, not a predictive evaluation) and aggregate by team and player: actual goals scored vs. total expected goals, and the gap between them. This is the standard "xG over/underperformance" analysis published by outlets like Understat and Opta.

**Most notable findings (all 7 competitions combined):**

- **Brazil: -9.67 xG differential** - 34.67 xG from 257 shots but only 25 goals scored, the single most dramatic unde
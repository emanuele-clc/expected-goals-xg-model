# Expected Goals (xG) Model — v2

An xG model built on real StatsBomb open event data across three competitions, comparing a hyperparameter-tuned logistic regression against a hyperparameter-tuned gradient boosting model, with cross-validation, a held-out test set, and bootstrap significance testing — benchmarked against StatsBomb's own published xG.

## Quick start (try it without downloading any data)

```
pip install -r requirements.txt
python src/predict_example.py
```

This loads the saved, already-trained logistic regression model and prints predicted xG for five hand-made shot scenarios (penalty spot, edge of the box under pressure, tight angle, header from a cross, long range) — a quick sanity check that the model behaves sensibly (see sample output in the script's docstring / below).

To fully reproduce the dataset and retrain from scratch you need a local clone of `statsbomb/open-data` (see "Reproducing / extending" below) — that step requires internet access and isn't needed just to try the model.

## Dataset

- Source: [StatsBomb open-data](https://github.com/statsbomb/open-data) (free, open license for public research/education use).
- **167 matches across 3 competitions**: FIFA World Cup 2018 (64 matches, men), Women's World Cup 2019 (52 matches, women), UEFA Euro 2020 (51 matches, men) — chosen to add scale and cross-gender/cross-confederation variety rather than relying on a single tournament.
- Obtained via a shallow, blob-filtered `git clone` of the open-data repo, then checking out only the required `matches/` and `events/` JSON files (full files, not previews/truncated fetches).
- **4,309 total shots** extracted; 4,153 non-penalty shots used for modeling (156 penalties held out — see Methodology).

Raw extraction script: `src/extract_shots.py`. It walks every match's event file, filters `type.name == "Shot"`, and pulls out location, outcome, body part, technique, shot type, play pattern, pressure, freeze-frame defender positions, and — new in v2 — the **assist type** (cross / through ball / cut-back / corner / free-kick pass / other), resolved by looking up the shot's `key_pass_id` in the same match's event stream.

## Methodology

**Why penalties are excluded from the main model:** penalty conversion (66.7% in this larger sample) is essentially constant and situation-independent — including them would distort a location/angle-based model. They're modeled separately as a fixed rate and reported alongside.

**Features (16 total):** `distance`, `distance_sq`, `angle_deg` (law of cosines on the two goalposts), `angle_x_distance` interaction, `log_distance`, `under_pressure`, `first_time`, `one_on_one`, `aerial_won`, `n_opponents_close` and `gk_positioned` (from StatsBomb 360 freeze-frame data), `header`, `is_free_kick`, plus one-hot `technique`, `play_pattern`, and `assist_type`.

**Rigor added in v2, addressing the main weaknesses of a first pass:**

1. **Held-out test set never touched during model selection.** An 80/20 stratified split is made first; all cross-validation and hyperparameter search happens only inside the 80% training portion. Final metrics are reported exclusively on the untouched 20%.
2. **5-fold stratified cross-validation** for model selection instead of a single split (which gives noisy, overconfident estimates on a dataset this size).
3. **Hyperparameter tuning via `RandomizedSearchCV`** (15 draws for logistic regression's `C`, 40 draws for gradient boosting's depth/learning-rate/iterations/regularization/leaf-size), each scored by 5-fold CV ROC AUC.
4. **Bootstrap 95% confidence intervals** (3,000 resamples of the held-out test set) for every model's AUC, plus a **paired bootstrap significance test** on the AUC difference between models — so "model A beats model B" claims are backed by a p-value-equivalent, not just a single point estimate.

## Results (held-out test set, n=831 non-penalty shots)

| Model | Test AUC | 95% CI | 5-fold CV AUC | Brier | Log Loss | Avg. Precision |
|---|---|---|---|---|---|---|
| **Logistic Regression** (tuned) | **0.793** | [0.734, 0.848] | 0.784 ± 0.030 | 0.071 | 0.253 | 0.325 |
| Gradient Boosting (tuned) | 0.771 | [0.711, 0.833] | 0.776 ± 0.024 | 0.071 | 0.256 | 0.350 |
| StatsBomb xG (reference) | 0.811 | [0.754, 0.865] | — | 0.069 | 0.244 | 0.382 |

**Bootstrap significance test:** Gradient Boosting's AUC is *lower* than Logistic Regression's by 0.022 on average (95% CI [-0.049, +0.002]); logistic regression beats gradient boosting in 95.9% of bootstrap resamples (P(GB>LR) = 0.041). Gradient boosting is also significantly below StatsBomb's own xG (P(GB>StatsBomb) = 0.004), while logistic regression's confidence interval overlaps StatsBomb's almost entirely.

**This is a genuine, somewhat counter-intuitive finding, reported honestly rather than cherry-picked:** on this feature set and sample size, a properly regularized logistic regression (tuned `C ≈ 0.079`, i.e. fairly strong L2 regularization) generalizes *better* than gradient boosting, whose best cross-validated configuration also converged on a heavily regularized regime (`max_depth=3`, `min_samples_leaf=50`, `l2_regularization=5.0`, low learning rate). With ~4,000 shots and a handful of genuinely informative features (distance and angle dominate — see feature importance plot), there isn't enough signal/complexity for a more flexible model to out-learn a well-regularized linear one; this matches a known pattern in the sports-analytics literature for shot-level xG models with modest feature sets. Logistic regression's test AUC also lands squarely inside StatsBomb's own confidence interval, which is the strongest available external validity check.

Full numeric results: `data/model_comparison.csv`, `data/cv_results.csv`, `data/bootstrap_ci.csv`, `data/significance_test.json`.

## Team and player xG performance report

Beyond the raw shot-level model, `src/team_player_performance.py` applies the trained logistic regression to every shot in the dataset (all 4,309, not just the held-out test set — this is a retrospective descriptive analysis, not a predictive evaluation) and aggregates by team and player: actual goals scored vs. total expected goals, and the gap between them. This is the standard "xG over/underperformance" analysis published by outlets like Understat and Opta.

**Most notable findings (World Cup 2018 + Women's World Cup 2019 + Euro 2020):**

- **Germany (men, WC2018): -6.84 xG differential** — 12.84 xG from 123 shots but only 6 goals scored. This lines up with their infamous group-stage elimination: the underlying shot quality suggests they created enough to advance, but finishing (and variance) went badly against them.
- **Spain (men, WC2018): -6.56** — 30.56 xG but only 24 goals, another early-exit side (lost in the Round of 16 to Russia on penalties) whose underlying chance creation outpaced their result.
- **Neymar (Brazil): -2.72** — 4.72 xG from 27 shots, only 2 goals. A concrete number behind the tournament narrative that he underperformed in front of goal.
- **Harry Kane (England): +3.88** — 8.12 xG from 32 shots, 12 goals scored. Consistent with him winning the WC2018 Golden Boot; the model says it wasn't just shot volume, he finished well above expectation.
- **USA Women's National Team: +8.73**, the largest team-level overperformance — consistent with them winning the 2019 Women's World Cup outright.

Full tables: `data/team_performance.csv` (68 teams) and `data/player_performance.csv` (943 players, filtered to ≥8 shots, 200+ qualifying). The interactive dashboard (`docs/index.html`) includes sortab
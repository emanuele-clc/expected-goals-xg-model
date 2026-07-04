# Expected Goals (xG) Model — v2

An xG model built on real StatsBomb open event data across three competitions, comparing a hyperparameter-tuned logistic regression against a hyperparameter-tuned gradient boosting model, with cross-validation, a held-out test set, and bootstrap significance testing — benchmarked against StatsBomb's own published xG.

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

## Visualizations (`plots/`)

- `roc_curve.png` — ROC curves for all three xG estimates on the held-out test set
- `bootstrap_auc_distribution.png` — overlapping bootstrap AUC distributions, the clearest visual of the significance result above
- `calibration_curve.png` — predicted vs. observed goal rate by decile
- `feature_importance.png` — permutation importance, side by side for both models
- `shot_map.png` — test-set shot locations on a half-pitch, marker size = predicted xG (logistic regression), colored by outcome
- `xg_vs_statsbomb.png` — scatter of our model's xG against StatsBomb's own xG (validation)
- `xg_vs_distance.png` — xG decay with shot distance
- `dataset_composition.png` — shot counts by competition and gender

## Files

```
xg_model_v2/
├── README.md
├── data/
│   ├── shots_raw.csv          # 4,309 shots, 21 columns, 3 competitions
│   ├── model_comparison.csv   # final held-out test metrics
│   ├── cv_results.csv         # 5-fold CV metrics (mean ± std)
│   ├── bootstrap_ci.csv       # bootstrap 95% CI per model
│   └── significance_test.json # paired bootstrap AUC-difference test
├── models/
│   ├── logreg_xg_model.joblib
│   ├── gboost_xg_model.joblib
│   ├── penalty_xg.json         # fixed penalty conversion rate
│   └── best_hyperparameters.json
├── plots/                     # 8 PNGs described above
└── src/
    ├── extract_shots.py       # StatsBomb JSON -> shots_raw.csv (3 competitions)
    ├── train_xg_model.py      # feature engineering, CV, tuning, bootstrap, final eval
    └── make_plots.py          # all visualizations, loads saved models
```

## Reproducing / extending

To rerun end to end: shallow-clone `statsbomb/open-data` with `git clone --depth 1 --filter=blob:none`, checkout the `matches/{43,72,55}/{3,30,43}.json` and corresponding `data/events/*.json` files, point `EVENTS_DIR`/`MATCHES_DIR` in `extract_shots.py` at your clone, then run `extract_shots.py` → `train_xg_model.py` → `make_plots.py`.

Natural next steps if extending further: add more competitions/seasons (StatsBomb's open-data set now includes several full league seasons) to shrink the bootstrap CIs further; try a shallow neural network or CatBoost with native categorical handling; move from a single held-out split to nested cross-validation for an even less biased performance estimate.

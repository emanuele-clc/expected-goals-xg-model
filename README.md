# Expected Goals (xG) Model - v3

An xG model built on real StatsBomb open event data across **eight competitions** - seven international tournaments plus a full club season (La Liga 2015/16) - comparing three tuned models (logistic regression, gradient boosting, and XGBoost) with cross-validation, a held-out test set, and full pairwise bootstrap significance testing, benchmarked against StatsBomb's own published xG.

An interactive dashboard (`docs/index.html`) ships with the project: a live in-browser xG calculator (the real trained model, running client-side), filterable shot maps, per-player shot history drill-downs, sortable team/player over-performance leaderboards, and a **Scouting Radar** - a statistically-robust ranking of players whose shot over-performance is unlikely to be a hot streak - no server or install required, just open the file.

## Live demo

`docs/index.html` is a self-contained static page (all data embedded, no build step, no server) - see **Deploying to GitHub Pages** below for exact steps to publish it at a public URL.

## Quick start (try it without downloading any data)

```
pip install -r requirements.txt
python src/predict_example.py
```

This loads the three saved, already-trained models (logistic regression, gradient boosting, XGBoost) and prints predicted xG from each for five hand-made shot scenarios (penalty spot, edge of the box under pressure, tight angle, header from a cross, long range) - a quick sanity check that all three behave sensibly and broadly agree.

To fully reproduce the dataset and retrain from scratch you need a local clone of `statsbomb/open-data` (see "Reproducing / extending" below) - that step requires internet access and isn't needed just to try the model.

## Dataset

- Source: [StatsBomb open-data](https://github.com/statsbomb/open-data) (free, open license for public research/education use).
- **726 matches across 8 competitions**, spanning men's and women's football, international tournaments, and club football:

  | Competition | Season | Gender | Matches | Shots |
  |---|---|---|---|---|
  | FIFA World Cup | 2018 | Men | 64 | 1,706 |
  | Women's World Cup | 2019 | Women | 52 | 1,314 |
  | UEFA Euro | 2020 | Men | 51 | 1,289 |
  | FIFA World Cup | 2022 | Men | 64 | 1,494 |
  | UEFA Women's Euro | 2022 | Women | 31 | 881 |
  | Africa Cup of Nations | 2023 | Men | 52 | 1,244 |
  | Copa América | 2024 | Men | 32 | 790 |
  | **La Liga (club)** | **2015/16** | Men | **380** | **9,168** |

  La Liga 2015/16 is the only complete full club season available in StatsBomb's free open-data release (every other La Liga season on offer is a partial, Barcelona-only release of 33-36 matches); it alone contributes over half the dataset's shots, purely because a club plays 30-38 league matches a season versus a handful of games in a single international tournament.
- Obtained via a shallow, blob-filtered `git clone` of the open-data repo, then checking out only the required `matches/` and `events/` JSON files (full files, not previews/truncated fetches).
- **17,886 total shots** extracted; 17,428 non-penalty shots used for modeling (458 penalties held out - see Methodology).
- Each shot is tagged with `competition_name` **and** `season_name`, since StatsBomb reuses the same `competition_name` ("FIFA World Cup") for both the 2018 and 2022 tournaments - without the season field the two would silently merge under one filter.

Raw extraction script: `src/extract_shots.py`. It walks every match's event file, filters `type.name == "Shot"`, and pulls out location, outcome, body part, technique, shot type, play pattern, pressure, freeze-frame defender positions, and the **assist type** (cross / through ball / cut-back / corner / free-kick pass / other), resolved by looking up the shot's `key_pass_id` in the same match's event stream.

## Methodology

**Why penalties are excluded from the main model:** penalty conversion (69.2% in this sample, n=458) is essentially constant and situation-independent - including them would distort a location/angle-based model. They're modeled separately as a fixed rate and reported alongside.

**Features (16 total):** `distance`, `distance_sq`, `angle_deg` (law of cosines on the two goalposts), `angle_x_distance` interaction, `log_distance`, `under_pressure`, `first_time`, `one_on_one`, `aerial_won`, `n_opponents_close` and `gk_positioned` (from StatsBomb 360 freeze-frame data), `header`, `is_free_kick`, plus one-hot `technique`, `play_pattern`, and `assist_type`.

**Rigor built into the pipeline:**

1. **Held-out test set never touched during model selection.** An 80/20 stratified split is made first; all cross-validation and hyperparameter search happens only inside the 80% training portion. Final metrics are reported exclusively on the untouched 20% (3,486 shots).
2. **5-fold stratified cross-validation** for model selection instead of a single split (which gives noisy, overconfident estimates).
3. **Hyperparameter tuning via `RandomizedSearchCV`** for all three models (15 draws for logistic regression's `C`, 20 for gradient boosting, 30 for XGBoost - see the note in `train_xg_model.py` about raising these if you have more CPU cores available), each scored by 5-fold CV ROC AUC.
4. **Bootstrap 95% confidence intervals** (3,000 resamples of the held-out test set) for every model's AUC, plus a **full pairwise bootstrap significance test across all 6 model pairs** - so "model A beats model B" claims are backed by a p-value-equivalent for every comparison, not just a headline one.

## Results (held-out test set, n=3,486 non-penalty shots)

| Model | Test AUC | 95% CI | 5-fold CV AUC | Brier | Log Loss | Avg. Precision |
|---|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.799 | [0.774, 0.824] | 0.806 ± 0.016 | 0.0738 | 0.2608 | 0.396 |
| Gradient Boosting (tuned) | 0.801 | [0.775, 0.826] | 0.803 ± 0.021 | 0.0738 | 0.2606 | 0.402 |
| **XGBoost** (tuned) | **0.801** | **[0.775, 0.826]** | 0.804 ± 0.020 | 0.0737 | 0.2605 | 0.404 |
| StatsBomb xG (reference) | 0.824 | [0.799, 0.847] | - | 0.0708 | 0.2497 | 0.439 |

**Bootstrap pairwise significance (3,000 resamples, all 6 pairs):**

| Comparison | Mean AUC diff | 95% CI | P(row beats column) |
|---|---|---|---|
| XGBoost vs. Logistic Regression | +0.0020 | [-0.0058, +0.0098] | 68.9% |
| XGBoost vs. Gradient Boosting | +0.0005 | [-0.0033, +0.0045] | 58.6% |
| Logistic Regression vs. Gradient Boosting | -0.0015 | [-0.0099, +0.0073] | 37.4% |
| XGBoost vs. StatsBomb xG | -0.0224 | [-0.0344, -0.0109] | 0.0% |
| Logistic Regression vs. StatsBomb xG | -0.0244 | [-0.0363, -0.0130] | 0.0% |
| Gradient Boosting vs. StatsBomb xG | -0.0229 | [-0.0352, -0.0109] | 0.0% |

**This is a genuine, honestly-reported finding, and it changed with more data - twice now.** In the original 3-competition version of this project, logistic regression beat gradient boosting decisively. Once the dataset grew to 7 tournaments, that gap narrowed sharply. Now, with La Liga's full club season added (17,886 shots total, more than double the previous version), **XGBoost has taken over as the best of the three trained models** - though only by a thin margin (it beats logistic regression in 68.9% of bootstrap resamples and gradient boosting in just 58.6%, nowhere near a knockout). More strikingly, **every trained model now loses to StatsBomb's own xG in 100% of bootstrap resamples** - a bigger, more decisive gap than in the smaller dataset, where the confidence intervals still overlapped. The likely reason: mixing club-season shots with international-tournament shots asks 
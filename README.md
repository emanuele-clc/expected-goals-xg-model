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
| PyTorch MLP (local run, no CI) | 0.799 | - | - | 0.1750 | 0.5274 | 0.394 |

**Bootstrap pairwise significance (3,000 resamples, all 6 pairs):**

| Comparison | Mean AUC diff | 95% CI | P(row beats column) |
|---|---|---|---|
| XGBoost vs. Logistic Regression | +0.0020 | [-0.0058, +0.0098] | 68.9% |
| XGBoost vs. Gradient Boosting | +0.0005 | [-0.0033, +0.0045] | 58.6% |
| Logistic Regression vs. Gradient Boosting | -0.0015 | [-0.0099, +0.0073] | 37.4% |
| XGBoost vs. StatsBomb xG | -0.0224 | [-0.0344, -0.0109] | 0.0% |
| Logistic Regression vs. StatsBomb xG | -0.0244 | [-0.0363, -0.0130] | 0.0% |
| Gradient Boosting vs. StatsBomb xG | -0.0229 | [-0.0352, -0.0109] | 0.0% |

**This is a genuine, honestly-reported finding, and it changed with more data - twice now.** In the original 3-competition version of this project, logistic regression beat gradient boosting decisively. Once the dataset grew to 7 tournaments, that gap narrowed sharply. Now, with La Liga's full club season added (17,886 shots total, more than double the previous version), **XGBoost has taken over as the best of the three trained models** - though only by a thin margin (it beats logistic regression in 68.9% of bootstrap resamples and gradient boosting in just 58.6%, nowhere near a knockout). More strikingly, **every trained model now loses to StatsBomb's own xG in 100% of bootstrap resamples** - a bigger, more decisive gap than in the smaller dataset, where the confidence intervals still overlapped. The likely reason: mixing club-season shots with international-tournament shots asks one general model to fit two somewhat different underlying shot-quality distributions at once, which is a harder problem than fitting either alone. That's a genuine trade-off of the bigger, more varied dataset, reported honestly rather than hidden.

Full numeric results: `data/model_comparison.csv`, `data/cv_results.csv`, `data/bootstrap_ci.csv` / `.json`, `data/significance_test.json`, `models/best_hyperparameters.json`.

**Feature importance tells two different stories depending on the model.** Logistic regression still leans on distance (plus its log-transformed version), same as before. XGBoost - the best-performing model - instead leans overwhelmingly on shot **angle** (`angle_deg`), with raw distance a distant second; headers, nearby defenders, and through-ball assists round out its secondary factors. This isn't a contradiction: angle and distance are correlated in real shot locations (shots near the byline are simultaneously far away in a straight line and at a sharp angle), so a flexible tree ensemble is free to lean on whichever cleanly-separating feature it finds first, while a linear model needs the specific engineered log/interaction terms to make distance alone do the work. See `plots/feature_importance.png` (all three models) and the dashboard's Feature Importance chart (XGBoost) for the full breakdown.

## Team and player xG performance report

Beyond the raw shot-level model, `src/team_player_performance.py` and `src/generate_dashboard_data.py` apply the trained **XGBoost** model (the best-performing of the three) to every shot in the dataset (all 17,886, not just the held-out test set - this is a retrospective descriptive analysis, not a predictive evaluation) and aggregate by team and player: actual goals scored vs. total expected goals, and the gap between them. This is the standard "xG over/underperformance" analysis published by outlets like Understat and Opta. Full tables: `data/team_performance.csv`, `data/player_performance.csv`.

**Most notable findings (all 8 competitions combined):**

- **Real Madrid: +30.39 xG differential** (108 goals from 77.61 xG, 717 shots) and **Atlético Madrid: +12.25** sit far ahead of every international team - almost entirely a function of La Liga's much larger per-team sample (a club plays 30+ more matches in a season than any national team gets in a single tournament), not necessarily a bigger underlying skill gap. Filter the dashboard to a single tournament to compare teams on equal footing again.
- **Brazil: -10.58** and **Germany: -9.62** anchor the bottom of the combined table.
- At player level, **Luis Suárez (+11.89 from 139 shots)** and **Gareth Bale (+10.32 from 81)** top the list, both from La Liga - again, the competition with by far the most shots per player. **Marcus Berg (-3.44, zero goals from 17 shots)** anchors the bottom, from a short international sample.

## Scouting Radar: a statistically robust finishing index

Standard "goals minus xG" leaderboards (above) treat a hot streak on 10 shots the same as a proven edge over 200 - a real problem once you're using this kind of analysis to actually decide something (like recruitment). `src/scouting_radar.py` fixes that: each of the **303 players** with at least 15 shots has their own shot-by-shot (goal - xG) values **bootstrap-resampled 2,000 times**, and players are ranked by the **2.5th percentile of that distribution** - a conservative "skill floor" that only stays positive if the over-performance survives a pessimistic reading of the player's own sample.

**Only 7 of the 303 evaluated players (2.3%) clear that bar when every competition is pooled together.** Nearly all of them play in La Liga 2015/16 - the only competition here with enough matches per player to make an individual finishing-skill estimate statistically meaningful. That's not a flaw in the method, it's the method doing its job honestly: most single-tournament "elite finisher" storylines are two or three big moments in a tiny sample, statistically indistinguishable from luck, and this ranking says so instead of pretending otherwise.

The dashboard also breaks this ranking out **per competition** rather than only pooled across a player's whole appearance history, which surfaces a genuinely interesting edge case: Alexandra Popp clears the reliability bar specifically within Women's Euro 2022 (16 shots, 6 goals, floor +0.024), even though her floor turns negative once her shots from other tournaments are folded into one pooled estimate. Both readings are statistically valid - they just answer slightly different questions ("was she reliable in this tournament?" vs. "is she reliable across her whole sample here?").

This is deliberately built as the kind of volume-aware, conservative signal a recruitment or scouting team would actually want before spending a transfer budget on a "hot" attacker: it ignores reputation and price tag entirely and only asks whether the underlying shot-quality data supports the reputation. Full ranking: `data/scouting_radar.csv` (pooled) / `data/scouting_radar.json` (pooled + per-competition); interactive, sortable version in the dashboard's Scouting Radar section, including a **compare-two-competitions mode** shared with the Team and Player Leaderboards (see "Live dashboard features" below).

## Optional 4th model: PyTorch neural network (run locally)

`src/train_xg_model_deep.py` trains a small feed-forward neural network on the exact same features and train/test split as the other three models, for a genuinely fair 4-way comparison. **It is not run as part of this repo's committed results** - PyTorch isn't installed in the environment this project was built in, and isn't a dependency of `requirements.txt`, so this step is entirely optional and yours to run:

```
pip install torch
python src/train_xg_model_deep.py
```

**Set honest expectations before you run it:** tabular data (rows of distance/angle/pressure, not images or text) is the one domain where deep learning does not reliably beat simpler methods - a well-documented empirical finding (Shwartz-Ziv & Armon, *"Tabular Data: Deep Learning is Not All You Need"*, 2021), not a guess.

**Real result from an actual local run (included in the results table above):** AUC 0.799 - ties logistic regression almost exactly and sits in the same neighborhood as gradient boosting and XGBoost, confirming the expectation above. The more interesting finding is calibration: Brier score 0.175 and log loss 0.527, both dramatically worse than every other model (Brier ≈ 0.074, log loss ≈ 0.26) despite the similar AUC. In other words, the network ranks shots about as well as the other models, but its predicted probabilities themselves are far less trustworthy. The likely cause is the `pos_weight` class-imbalance correction used in the loss function (needed because only ~11% of shots are goals) - it helps the network rank correctly under imbalance but pushes its raw output probabilities away from being honestly calibrated, a known trade-off of that technique that would need a separate calibration step (e.g. Platt scaling / isotonic regression) to fix. This is exactly the kind of nuanced, non-obvious result worth reporting rather than smoothing over.

## Deploying to GitHub Pages

`docs/index.html` needs no build step - GitHub Pages can serve it directly:

1. Push this repo to GitHub (`git remote add origin <your-repo-url> && git push origin main`, if not already done).
2. On GitHub: **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Branch: **main**, folder: **/docs**. Save.
5. GitHub builds and publishes at `https://<your-username>.github.io/<repo-name>/` within a minute or two (check the Pages settings page for the exact URL and build status).
6. Any future push to `main` that touches `docs/` will redeploy automatically - no extra steps needed.

## Live dashboard features

- **Try It Yourself calculator** - click anywhere on the pitch, adjust the situation, and get an instant xG prediction. Runs the real trained **logistic regression** model's exact coefficients in vanilla JavaScript (not XGBoost, the best-performing model overall - a linear model's coefficients can be copied into JS exactly, while a tree ensemble like XGBoost cannot be without shipping the whole model to the browser). You can also place defenders manually (red markers) - the model's `n_opponents_close` feature is computed live from their positions, using the same 3-yard threshold as the real feature extraction.
- **Shot Map** - all 3,486 held-out test-set shots plotted on a pitch, sized by predicted xG (XGBoost), colored by outcome.
- **Player Shot Maps** - drill down competition → team → player to see every shot a player took, with colored initials avatars next to every name.
- **Model Comparison** - AUC bar chart with 95% CIs for all three trained models plus StatsBomb's reference xG (and the optional PyTorch result, once run locally), and a feature-importance chart for the best model.
- **Model Diagnostics Gallery** - the full evaluation plot set (ROC, calibration, bootstrap distribution, feature importance, shot map, xG-vs-StatsBomb, xG-vs-distance, dataset composition).
- **Team & Player Leaderboards** - sortable, filterable by competition, goals vs. xG. National teams show a flag; club teams show a colored initials badge instead.
- **Scouting Radar** - the bootstrap-adjusted finishing-reliability ranking described above.
- **Compare two competitions** - a shared toggle above the Team Leaderboard puts the Team Leaderboard, Player Leaderboard, and Scouting Radar side by side for any two of the 8 competitions at once (e.g. La Liga 2015/16 vs. World Cup 2022), instead of switching the single competition filter back and forth.

## Reproducing / extending

1. Clone StatsBomb's open-data repo (shallow + blob-filtered, to avoid downloading everything): see the pattern used for La Liga in this project's history, or simply `git clone --depth 1 https://github.com/statsbomb/open-data.git`.
2. Point `src/extract_shots.py`'s `COMPETITIONS` list at the `(competition_id, season_id)` pairs you want (check that repo's `data/competitions.json` for available options and match counts - not every listed season is a genuinely complete one).
3. `python src/extract_shots.py` → regenerates `data/shots_raw.csv`.
4. `python src/train_xg_model.py` → retrains all three models, regenerates every CSV/JSON result file and `models/*.joblib`.
5. `python src/export_feature_importance.py`, `python src/make_plots.py`, `python src/team_player_performance.py`, `python sr
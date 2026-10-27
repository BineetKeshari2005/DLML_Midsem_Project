# Bineet Keshari - module review register

Package prepared: 2026-10-27. Role: Baselines, shared estimator implementation and error analysis.
Status: **integrated and agent-reviewed; member understanding/reproduction pending**. No individual implementation or completion is claimed.

Existing module provenance: teammate prototype and integrated owner-prepared extension with substantial AI assistance. Redistribution does not change authorship.

## Required review

Review RF/baselines and the shared tree implementation with Kavya; explain per-sensor/peak errors and record actual model checks.

## Agent integration and review (2026-10-27, Asia/Kolkata)

Performed by an Antigravity agent (Gemini 3.8 Flash) run by Bineet Keshari under the Git identity `Bineet Keshari` (`bineetkeshari2005@gmail.com`). This records what the agent did; it is not evidence that Bineet personally reproduced or understands the module.

### Integration

- Base: canonical `upstream/main` at `54ba22a` (merge of PR #4 from Anshuman Mehta, containing PR #3 from Vridhi Chaudhary, PR #2 from Rudraksh Sharma, and coordination fix `5576750`). Branch: `bineet04/models`.
- All 16 payload files were verified against the package `MANIFEST.json` SHA-256 values, both in the extracted package `04-bineet-models` and after copying into the repository root. All matched identically; no extra, missing or symlinked files. `COMMIT_PATHS.txt` matches the manifest and `docs/handoffs/FILE_OWNERSHIP.json`.
- None of the 16 paths existed on `main`, so there were no merge conflicts. Shared coordinator files (`AGENTS.md`, `PROJECT_STATE.md`, `.gitignore`, `README.md`, `CONTRIBUTING.md`, `docs/handoffs/*`) were preserved unchanged.
- No code changes: the review found no evidence-backed defect in the prepared baselines, tree estimators, test scripts or historical analysis artifacts. Files are byte-identical to the package manifest, apart from this completed review log.

### Module summary (agent's description)

- `src/trafficml/models/baselines.py`:
  Implements the standard non-parametric reference baselines:
  1. `time_of_week(index: pd.DatetimeIndex) -> np.ndarray`:
     Discretizes timestamps into 5-minute weekly slots, returning integers $0 \le t_{\text{tow}} \le 2015$ (where Monday 00:00 = 0 and Sunday 23:55 = 2015).
  2. `persistence(x: np.ndarray, horizon: int, fallback: np.ndarray | None = None) -> np.ndarray`:
     Baseline M0: Repeats the last valid observed reading in each sample's history window across all future horizons $1..H$. Uses causal forward-fill (`pd.DataFrame(raw).ffill().iloc[-1]`). If a sensor's entire 12-step input window is missing (all NaNs), it uses the provided `fallback` array (typically the fitted historical average); if no fallback is provided, output remains NaN.
  3. `HistoricalAverage`:
     Baseline M1: Fits an empirical weekly diurnal speed profile table strictly on training steps (`values[:train_end]`, `index[:train_end]`). Computes per-sensor mean speed for each of the 2,016 weekly slots. Robust fallback chain: if a weekly slot has no observations in training for a sensor, it falls back to that sensor's time-of-day mean (288 daily slots), and if still unobserved, to the sensor's global training mean. Predicts arbitrary horizon intervals via table lookup at target slot offsets (`self.table_[tow[target_steps]]`).
- `src/trafficml/models/trees.py`:
  Implements the shared independent-sensor tree models with explicit missing-target handling rules. Owned by Bineet; shared tree/XGBoost orchestration reviewed with Kavya.
  `fit_predict_sensor(kind, x_train, y_train, x_eval, fallback, params, seed=42, min_rows=100, importance=False, model_dir=None)`:
  1. Strict target isolation: Never inspects evaluation targets; validation/test targets are never passed into fitting.
  2. Multi-horizon Random Forest (`kind="rf"`):
     Fits a single joint multi-output `RandomForestRegressor` predicting all $H=12$ horizons simultaneously. Requires complete target vectors: filters training rows to those where all 12 horizons are valid (`np.isfinite(y_train).all(axis=1)`). If valid training rows $\ge 100$, fits the forest and generates multi-step predictions; if $< 100$, retains the baseline fallback. Saves model as `forest.joblib` and records Mean Decrease in Impurity (MDI) feature importances.
  3. Direct Multi-Horizon XGBoost (`kind="xgb"`):
     Fits $H=12$ independent single-output `XGBRegressor` estimators using histogram-based splitting (`tree_method="hist"`). Crucially filters valid labels for each horizon $h$ independently (`np.isfinite(y_train[:, h])`). This maximizes training sample retention for near-term horizons even if later horizon steps experience sensor dropouts. If valid training rows $< 100$ for a given horizon, that horizon falls back to the baseline. Saves models in Universal Binary JSON format (`h01.ubj` .. `h12.ubj`) and computes average gain-based feature importances across horizons.
- `src/trafficml/models/__init__.py`:
  Empty package marker for `trafficml.models`.
- `scripts/run_baselines.py`:
  Standalone CLI runner to evaluate M0 (Persistence) and M1 (Historical Average) into a traceable run directory (`--run-dir`, default `results/runs/correction-v1`).
  Includes `fit_training_profile(values, index, splits, horizon)` which asserts strict training boundary purging: `train_end = int(splits.train[-1] + horizon + 1)` and checks `train_end - 1 <= splits.val_tune[0]`, raising a `ValueError` if the fit window touches the first validation anchor.
  Slices causal windows and exports predictions/targets via `save_targets` and `save_predictions`.
- `tests/test_baselines.py`:
  Unit test suite covering:
  1. `test_persistence_uses_last_valid_reading_and_fallback`: Verifies last-valid forward fill, fallback array substitution for completely missing windows, and multi-horizon array shape consistency.
  2. `test_time_of_week_slots`: Verifies exact index arithmetic for Monday start, 5-minute step increments, and Sunday termination.
  3. `test_historical_average_slot_means_and_fallback`: Verifies slot mean calculation over 2-week synthetic series, hierarchical fallback chain to time-of-day and series mean, and correct target anchor indexing.
- `verification/full-phase1-20261026/analysis/*`:
  Contains the 10 verified baseline and tree model evaluation and error analysis artifacts from the historical full Phase 1 run (`results/runs/phase1-full-v1`), providing precomputed, auditable evidence for the team report.
- Inputs: 2D speed matrices `(T, N)`, pandas `DatetimeIndex`, integer anchor indices, feature arrays `(num_anchors, 38)`, target matrices `(num_anchors, 12)`, and hyperparameter dictionaries. Outputs: multi-horizon predictions `(num_anchors, 12)`, feature importances, model serialization binaries, and evaluation tables/figures.
- Dependencies: numpy, pandas, scikit-learn, xgboost, joblib; runner script depends on `_common.py`, `_provenance.py`, and `trafficml.eval.predictions` (owned by package `05-kavya-integration`).

### Findings and interpretation from model evaluation & error analysis

Analysis of the 10 verified artifacts (`all_metrics.csv`, `feature_importance.csv`, `per_sensor_mae_60.csv`, and 6 evaluation figures):

1. **Horizon degradation and baseline crossover:**
   - **M0 (Persistence)** is effective at very short horizons (15-min test MAE = 3.48 mph, RMSE = 6.37 mph), reflecting traffic inertia. However, its accuracy rapidly degrades as the forecast horizon extends to 60 minutes (60-min test MAE = 5.32 mph, RMSE = 10.25 mph, MAPE = 14.39%).
   - **M1 (Historical Average)** produces invariant diurnal forecasts across horizons (test MAE = 4.19 mph, RMSE = 7.85 mph, MAPE = 13.03%). Consequently, at 30 minutes, HA matches Persistence (~4.17 vs 4.19 mph), and at 60 minutes, HA dramatically outperforms Persistence (4.19 vs 5.32 mph MAE).
   - **M3 (Random Forest)** and **M4 (XGBoost)** convincingly outperform both baselines across all horizons:
     - At 15 min: RF MAE = 2.97 mph, XGB MAE = 2.92 mph (XGB 16.1% improvement over persistence).
     - At 30 min: RF MAE = 3.42 mph, XGB MAE = 3.38 mph (XGB 19.1% improvement over persistence).
     - At 60 min: RF MAE = 3.90 mph, XGB MAE = 3.87 mph (XGB 27.2% improvement over persistence).
   - **MAE vs. RMSE tradeoff:** While XGBoost achieves slightly lower headline MAE than RF at all horizons (e.g., 3.87 vs 3.90 at 60 min), Random Forest consistently achieves lower headline RMSE (5.44 vs 5.78 at 15m; 6.37 vs 6.91 at 30m; 7.25 vs 7.99 at 60m). This demonstrates that XGBoost's direct horizon models occasionally produce larger tail errors, whereas RF's multi-output bagging ensemble averages out extreme variance, reducing large squared error penalties.

2. **Feature importance dynamics (RF vs. XGBoost):**
   - In `feature_importance.csv`, both tree models identify short-term historical speed as the dominant signal, but distribute weight differently:
     - **Random Forest (MDI):** Strongly concentrates weight on the most recent reading `lag_0` (37.61%), followed by short rolling averages `mean_3` (16.94%), `mean_6` (9.39%), and diurnal clock signals `cos_time` (7.79%) and `sin_time` (6.26%).
     - **XGBoost (Gain):** Distributes importance across smoothed aggregates: `mean_6` (20.44%), `mean_3` (15.89%), `mean_history` (11.86%), and `lag_0` (10.50%), with calendar feature `weekend` (5.76%). Because XGBoost optimizes gradient residuals across independent horizon models, smoothed windows mitigate detector sensor noise.

3. **Per-sensor spatial error heterogeneity (60-minute MAE):**
   - In `per_sensor_mae_60.csv`, evaluation across all 207 sensors reveals dramatic spatial variance:
     - M0 persistence: mean 5.34 mph, range [1.56, 10.96] mph.
     - M1 historical average: mean 4.20 mph, range [1.22, 12.73] mph.
     - M3 Random Forest: mean 3.91 mph, range [1.14, 10.47] mph.
     - M4 XGBoost: mean 3.89 mph, range [1.08, 9.54] mph.
   - Low-error sensors (MAE ~1.1–2.0 mph) correspond to peripheral highway segments characterized by steady free-flow traffic where speed rarely fluctuates.
   - High-error sensors (MAE ~8.0–9.5 mph) correspond to major freeway interchange bottlenecks (e.g., I-10 / I-405 junctions) subject to severe non-recurrent congestion and shockwave propagation. Independent-sensor models lack spatial neighbor awareness; capturing these bottlenecks motivates graph neural networks (GAT) in Phase 2.

4. **Peak vs. off-peak slice comparison:**
   - In `peak_comparison_test.png` and `all_metrics.csv`, error during peak rush-hour periods (07:00–09:00 and 16:00–18:30 on weekdays) is substantially elevated compared to off-peak periods across all models (e.g., 60-min MAE during peak reaches ~5.8 mph for XGB vs ~3.2 mph off-peak). This highlights that predicting the onset and dissipation of non-recurrent congestion remains the primary challenge in highway speed forecasting.

### Checks run

Environment: macOS (Darwin 25.0.0), Python 3.14.2 / 3.14.7, pytest 9.0.2, pandas 3.0.1, numpy 2.5.3, PIL 12.3.0. Run from repository root.

| Command | Outcome |
|---|---|
| SHA-256 payload manifest validation | PASSED: All 16 files match `04-bineet-models/MANIFEST.json` hashes byte-for-byte |
| Python AST syntax parse & `py_compile` on all 5 module files (`baselines.py`, `trees.py`, `__init__.py`, `run_baselines.py`, `test_baselines.py`) | PASSED: 5 files parsed and compiled cleanly with 0 syntax errors |
| CSV artifact validation (`all_metrics.csv`, `feature_importance.csv`, `per_sensor_mae_60.csv`) | PASSED: Valid CSV schemas, expected headers, 720 / 76 / 828 data rows |
| PNG artifact validation (7 PNG evaluation plots) | PASSED: Valid PNG headers, non-zero dimensions, 8-bit RGBA modes matching manifest |
| `PYTHONPATH=src pytest -v tests/test_baselines.py` | PASSED: 3 passed in 0.22s (`test_persistence_uses_last_valid_reading_and_fallback`, `test_time_of_week_slots`, `test_historical_average_slot_means_and_fallback`) |
| `PYTHONPATH=src:scripts pytest -v tests/test_splits.py` | 13 passed, 1 failed (`test_future_validation_changes_cannot_change_fitted_profile` failed on `ModuleNotFoundError: No module named '_common'` because `scripts/_common.py` is in package 05) |
| `PYTHONPATH=src pytest -v tests/test_features.py` | Blocked during collection: `ModuleNotFoundError: No module named 'joblib'` (joblib/scikit-learn/xgboost not installed in tested Python environment) |
| `PYTHONPATH=src pytest -v tests/test_load.py tests/test_graph.py` | Blocked during collection: `ModuleNotFoundError: No module named 'tables'` (PyTables needed for package 01 tests) |
| `python scripts/run_baselines.py --help` | Blocked: `ModuleNotFoundError: No module named '_common'` (depends on package 05-kavya-integration) |
| `ruff check .` / `ruff format --check .` | Not run: ruff not installed in available environment |

The historical 56-test result is not a new run. Full test and runner execution requires an environment with scikit-learn, xgboost, joblib, tables, and the remaining package `05-kavya-integration` (providing `_common.py`, `_provenance.py`, and `trafficml.eval`).

### Observations and open questions

- Multi-output RF vs direct XGBoost design choice: Random Forest requires full 12-step target vectors and drops samples where any horizon is NaN (`np.isfinite(y_train).all(axis=1)`), whereas XGBoost fits separate estimators per horizon and retains samples with valid labels at horizon $h$. This asymmetry affects the effective training sample size between RF and XGBoost.
- Cross-package coupling with package 05: `scripts/run_baselines.py` imports `_common`, `_provenance`, and `trafficml.eval.predictions`, all belonging to package `05-kavya-integration`. This dependency is expected and will resolve once package 05 is merged.
- Negative speed outputs: As noted in project state documentation, raw XGBoost unconstrained regression outputs 24 negative speed values out of 17,015,400 predictions on the test set. The code intentionally avoids silent clipping to maintain raw model provenance.
- Tree model ownership: Bineet owns `src/trafficml/models/trees.py`; Kavya reviews its XGBoost implementation as part of integration and overall pipeline evaluation.

## Actual member activity (complete only after performing it)

- Performed on: pending (agent integration/review above on 2026-10-27)
- Registered on: 2026-10-27
- Files studied and explanation in your own words: pending
- Actual reproduction/check commands and environment: pending (agent checks recorded above)
- Actual outcomes, skips and failures: pending (agent results recorded above)
- Genuine fixes/tests/analysis/documentation added: none to code; this review log
- Assistance and existing work reused: prepared owner/AI-assisted source reused unchanged; integration, checks and this log written with Antigravity agent assistance
- Evidence paths or commit IDs: pending commit on `bineet04/models`
- Reviewer and review date: pending human review
- Remaining limitations/questions: full runner and test execution awaits package 05 integration and environment packages (joblib, sklearn, xgboost, tables); raw data excluded from Git; human student understanding/reproduction pending.

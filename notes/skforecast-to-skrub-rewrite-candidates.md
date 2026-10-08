# skforecast Gallery → skrub Rewrite Candidates

Source: https://skforecast.org/latest/examples/examples_english (fetched 2026-09-22; 38 examples, hosted as cienciadedatos.net tutorials)
Same exercise as `aeon-to-skrub-rewrite-candidates.md` and `tslearn-to-skrub-rewrite-candidates.md`.

## Gallery overview (4 sections)
- **Getting started (10)**: end-to-end ML forecasting, statistical models (ARIMA/SARIMAX/ETS/ARAR), GB libraries (XGBoost/LightGBM/CatBoost), categorical-series forecasting, visualization, agentic/studio tools.
- **Global models (8)**: multi-series forecasters, single-vs-global comparisons, thousand-series scaling, Kaggle sticker-sales and M5 walkthroughs, RNN/LSTM at scale, clustering-before-forecasting.
- **Advanced (15)**: foundation models, probabilistic forecasting (intervals, CRPS, conformal), SHAP interpretability, drift detection, trend/differentiation with trees, missing values, stacking, anomaly detection, leakage in pre-trained models, GPU/intelex acceleration.
- **Case studies (5)**: energy demand (+weather exog), web traffic (+events), intermittent demand, COVID-weighting, Bitcoin.

## Why overlap is maximal here
skforecast already converts forecasting into tabular supervised learning (lags + window features + calendar features + sklearn estimator + time-based backtesting) — the exact paradigm the sessionization script and skrub's `AggJoiner`/DataOps implement. The differentiators skrub brings on top:
1. **Messy relational exogenous data**: skforecast needs a clean, aligned `exog` dataframe; skrub joins/encodes multiple dirty auxiliary tables (`Joiner` fuzzy joins, `AggJoiner`/`MultiAggJoiner`, `TableVectorizer`, `GapEncoder`, `TextEncoder`/`LLMEncoder`).
2. **Structural search** over the whole featurization graph (`choose_from`/`choose_bool`, `ParamSearch`/Optuna) vs skforecast's parameter grids over lags + estimator.
3. **Time-aware CV as data-ops plumbing**: `mark_as_X(cv=splitter)` + `freeze_after_fit` reproduces backtesting/rolling-refit declaratively.

## Tier 1 — complementary rewrites (skforecast keeps the engine; skrub handles the data and orchestration)

### 1. "Skforecast: time series forecasting with ML" (intro, py27) → DataOps orchestration around a skforecast forecaster
- The `Forecaster*` stays the engine; the skrub graph takes over the manual scaffolding the tutorial hand-codes: lag/window features (`AggJoiner` + `skb.apply_func`), calendar features (`DatetimeEncoder`), train/prediction matrix creation, and backtesting as a custom time-ordered splitter (`mark_as_X(cv=...)` + `freeze_after_fit`) around the wrapped forecaster.
- The fully tabular branch (shared with the aeon/tslearn candidates) enters `choose_from` alongside the skforecast node, so "which paradigm fits this series" becomes a searched option, not a rewrite of the library.

### 2. "Forecasting energy demand" (py29) → skrub builds the `exog` skforecast assumes
- skforecast's exogenous frame must be clean, typed and time-aligned — precisely the artifact skrub produces: fuzzy-join weather-station tables (`Joiner`), aggregate to the target grid (`AggJoiner`/`MultiAggJoiner`), encode holidays and descriptions (`GapEncoder`/`TextEncoder`), sanitize dtypes (`TableVectorizer`) — then hand the resulting `exog` to the skforecast forecaster.
- Uses skrub's own `fetch_electricity_forecasting` (or `fetch_bike_sharing`) so the demo stays self-contained for both libraries.

### 3. Global models series (py44, py53, py59, py66 sticker sales, py61 M5) → skrub prepares the per-entity inputs `ForecasterRecursiveMultiSeries` needs
- The global forecaster stays; skrub supplies the entity side: attribute tables (store, item, brand) cleaned and encoded via `TableVectorizer`/`GapEncoder` (high-cardinality text is where skrub's encoders beat integer IDs), plus `AggJoiner` history stats as covariates and `choose_from` for per-cluster configurations.
- py64 (clustering to improve forecasting) → select clustering features with `DropSimilar`/`column_associations`, then hand the groups to per-cluster skforecast models.

### 4. "Forecasting of categorical time series" (py72) → skrub encodes the history skforecast predicts from
- Lagged category strings arrive dirty and high-cardinality: `ToCategorical`/`Cleaner`/`SimilarityEncoder` modelize the history (run-lengths via `AggTarget`) for skforecast's classification forecaster; free-text categories become encodable — a variant the dict-mapping forecaster cannot express alone.

### 5. "Forecasting web traffic" (py37) → events as text into aligned exog
- Campaign/outage descriptions live in an auxiliary table with fuzzy keys; skrub joins (`Joiner`) and embeds them (`TextEncoder`/`LLMEncoder`/`GapEncoder`) onto the hourly/daily exog grid the skforecast forecaster consumes — unstructured exogenous data skforecast's gallery simply does not handle.

## Tier 2 — partial rewrite / contrast
- **Missing values (py46)** → skrub imputation built into `TableVectorizer` + `Cleaner` + `ToFloat` (separator-tolerant parsing); simpler demo of the same strategies.
- **Stacking ensembles (py52)** → DataOps `concat` of prediction branches + meta-estimator, wrapped in `choose_from` so stacking choices become searched structure.
- **Probabilistic (py42, py60) + CRPS (py74)** → no skrub equivalent; rewrite only by swapping the estimator for `HistGradientBoostingRegressor(loss="quantile", ...)` inside the same graph — good showcase that the graph is estimator-agnostic.
- **COVID weighting (py45)** → `skb.if_else`/`match` on date ranges to build sample weights inside the graph.
- **Interpretability/SHAP (py57)** → skrub contrast: `column_associations`, `full_report`, `describe_steps` (which featurization branch won) rather than post-hoc SHAP.

## Tier 3 — minor / skip
- Trend-with-trees & differentiation (py49), ARAR/ARIMA/ETS (py51/73/76/77): statistical modeling skrub deliberately doesn't do; mention-only in a "when to use which library" note.
- Foundation models (py79), leakage in pre-training (py63), GPU/intelex acceleration (py65/75), agentic/Studio tools (py80/78), intermittent demand (py48), drift/anomaly detection (py70/62): no meaningful skrub leverage.

## Suggested order of work
1. Intro orchestration rewrite (#1) — pairs with aeon/tslearn rewrites; the three-gallery comparison write-up sells itself.
2. Energy-demand exog factory (#2) — most differentiated, uses skrub-native data.
3. Categorical-series history encoding (#4) — cheapest to build, squarely skrub's strengths.

## Cross-library note
skforecast shares skrub's tabular paradigm most closely, but the posture is complementary: skforecast remains the forecasting engine, skrub builds the inputs it assumes (aligned exog, entity tables, encoded histories, calendar features) and orchestrates splits/search around it; the pure-tabular skrub pipeline survives only as a `choose_from` branch. A coherent doc story: "skforecast builds the tabular dataset for you from one series; skrub builds it for you from fifty dirty tables — and hands it to skforecast."

# aeon Gallery → skrub Rewrite Candidates

Companion to `aeon-gallery-summary.md`. Gallery: https://www.aeon-toolkit.org/en/stable/examples.html
Additional context: `/Users/rcap/work/sessionization/main.py` — a working DataOps pipeline on timestamped event data (game heartbeats → monthly churn prediction) proving out: custom CV splitter (`mark_as_X(cv=Splitter())`), `SessionEncoder` sessionization, lag/rolling feature tables built outside the fold split, structural hyperparameter search (`choose_from`/`choose_bool` over feature-engineering choices), Optuna `make_randomized_search` with sqlite storage, multi-table env (`skrub.var` file path → lazy polars load via `apply_func`).

## Tier 1 — complementary rewrites (aeon keeps the models; skrub is the data & orchestration layer)

### 1. `forecasting/forecasting.html` → skrub prepares what aeon's forecasters assume
- **Complementary play**: aeon forecasters expect one clean, aligned series (+ exog). The skrub DataOps graph produces those inputs from messy sources — `Joiner`/fuzzy joins for auxiliary tables, `DatetimeEncoder`/`to_datetime`/`DurationToFloat` for calendars and durations — and wraps the aeon forecaster as a graph node (`skb.apply`/`apply_func`) with time-aware splits (`mark_as_X(cv=...)`) and `freeze_after_fit` rolling refit.
- **Optional rival branch**: the tabular path from the sessionization script (lags via `AggJoiner` + supervised HGB) enters the same graph inside `choose_from` next to the aeon node — "dedicated forecaster vs supervised tabular" becomes a *searched choice*, not a library decision.
- **Data**: skrub's `fetch_electricity_forecasting`/`fetch_bike_sharing` exercise both branches.

### 2. `benchmarking/benchmarking.html` → DataOps as the benchmarking harness around aeon estimators
- The aeon estimators stay; skrub supplies the harness: `choose_from(RocketClassifier, Arsenal, tabular branch, ...)` searches structure rather than fixed grids, `skb.cross_validate` + `full_report` + `describe_param_grid` produce the comparison, and Optuna + sqlite persistence (as in `random_search()` of main.py) replaces hand-rolled result tables.

### 3. `datasets/data_loading.html` + `data_unequal.html` → skrub as the front-end for aeon's collection formats
- aeon's loaders and collection formats remain the target; skrub cleans what loaders cannot swallow — `SessionEncoder` sessionizes raw event streams, `Cleaner`/`DropUninformative` sanitize, `ToDatetime` repairs timestamps — then reshapes into 3D/df-list/nested via `skb.apply_func`.
- **Bonus**: preview/subsampling (`skb.preview`, `SubsamplePreviews`) makes the wrangling interactive on huge collections, nothing like it in the aeon gallery.

## Tier 2 — hybrid examples (skrub wraps aeon, showcases extensibility)

### 4. `transformations/rocket.html` / `tsfresh.html` / `catch22.html` → series-in-a-dataframe-cell
- **Recipe**: a dataframe column of numpy arrays (one series per row/entity), encoded by an aeon transformer wrapped in `ApplyToSubFrame`/`wrap_transformer` inside `TableVectorizer`/`tabular_pipeline`; as a DataOps node it slots into `choose_from` alongside skrub-native encoders.
- **Why**: strongest cross-library story — "your time-series featurizer is just another column transformer".

### 5. `transformations/channel_selection.html` → redundant-feature dropping
- **Recipe**: `DropSimilar` + `column_associations` on the featurized output; `selectors` (`s.numeric()`, `s.has_dtype(...)`) for declarative selection; `if_else`/`match` for conditional feature blocks.

## Tier 3 — conceptual parallels (rewrite only if aiming for contrast)

### 6. `distances/sklearn_distances.html` → skrub analogue: `Joiner`/`fuzzy_join`/`MinHashEncoder`/`deduplicate` (custom similarity + sklearn). Different problem domain (strings, not series); useful as a "matching without distances" counterpart.
### 7. `base/base_classes.html` → skrub analogue: `describe_steps`, `draw_data_op_graph`, `SingleColumnTransformer`/`RejectColumn` for writing column-level estimators. A "how it's built" doc, not an analysis example.

## Not recommended
- Classification family examples (convolution/dictionary/distance/interval/shapelet/hybrid/deep-learning), clustering, segmentation, similarity-search internals, early classification: these operate on numeric series collections where skrub adds nothing; rewriting would be misleading (skrub is not a time-series toolkit).

## Suggested order of work
1. Forecasting front-end + harness around aeon forecasters (electricity dataset) — clearest complementary value.
2. Rocket/tsfresh-as-a-column-transformer hybrid — best cross-library demo.
3. Benchmarking/structure-search — reuses sessionization patterns almost verbatim.

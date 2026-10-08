# skrub Example Opportunities from Time-Series & Explainability Galleries — Conclusions

Synthesis of the five candidate analyses (2026-09-22):
- `aeon-to-skrub-rewrite-candidates.md`
- `tslearn-to-skrub-rewrite-candidates.md`
- `skforecast-to-skrub-rewrite-candidates.md`
- `mapie-to-skrub-rewrite-candidates.md`
- `shapash-to-skrub-rewrite-candidates.md`

## Libraries fall into three relationship classes

1. **Forecasting engines skrub complements** — skforecast (same tabular paradigm), plus the forecasting slices of aeon and tslearn. The libraries' models stay as the prediction nodes; skrub's role is building the tables they assume (aligned exog, calendar features, entity attributes, encoded histories) from messy multi-table sources, plus time-aware splits and structural search around them. A pure skrub tabular pipeline appears only as a rival branch inside `choose_from` — searched alongside the native model, never the headline replacement.
2. **Algorithm-internals libraries** — aeon, tslearn. skrub should *not* rewrite their clustering/shapelets/metrics/segmentation material; the honest plays are wrappers (aeon rocket / SAX strings / tsfresh inside `ApplyToSubFrame`, `choose_from` structure search) and conceptual contrasts.
3. **Downstream layers skrub feeds** — MAPIE (conformal prediction), shapash (attribution). Integration pattern is `Mapie*(SkrubLearner)` and `shapash.compile(postprocess=...)` over skrub encoders; skrub contributes the data wrangling and the encoder→raw-column bridge.

Positioning principle for all three classes: skrub is the **data & orchestration layer** — join, clean, encode, split, search, report — and the numeric model (aeon forecaster, tslearn VARIMA, skforecast engine, HGB) stays whatever library it came from.

## Cross-cutting finding: one script rules them all
The **DataOps timestamped-forecasting pipeline** (lags/rolling features via `AggJoiner`/`apply_func`, calendar via `DatetimeEncoder`, time-aware splits via `mark_as_X(cv=Splitter())`, `freeze_after_fit`, search via `ParamSearch` — all already proven in `/Users/rcap/work/sessionization/main.py`) is the shared backbone inside *all five* galleries (aeon forecasting, tslearn VARIMA, skforecast intro/energy/global, MAPIE ts-tutorial/EnbPI/Zaffran, shapash domain05). Build it once, cut five gallery-flavored variants — each keeps the gallery's native model as the head node (rocket baseline, DTW-kNN/VARIMA, skforecast engine, MapieRegressor, SHAP layer) with the skrub tabular path demoted to one `choose_from` branch among them.

## Shortlist (ranked across all galleries)

| # | Example | Uses | Why it wins |
|---|---------|------|-------------|
| 1 | Forecasting with DataOps on `fetch_electricity_forecasting`/`fetch_bike_sharing` | AggJoiner, DatetimeEncoder, custom splitter, `make_randomized_search` | Anchors 5 galleries; swappable head estimator — native forecaster (aeon/skforecast) or tabular branch, chosen by `choose_from` |
| 2 | SAX/shapelets/rocket/tsfresh → series-as-a-column via `ApplyToSubFrame` | wrap_transformer, tabular_pipeline, choose_from | Unique cross-library story ("your TS featurizer is a column transformer") |
| 3 | What-if / recourse by re-`skb.eval` on modified graphs | DataOps graph editing | Only skrub can do it; shapash structurally can't; genuinely novel |
| 4 | Structural pipeline search vs GridSearchCV/backtesting loops | ParamSearch/Optuna, cross_validate, full_report | Upgrades aeon `knnts_sklearn`, MAPIE `nested-cv`, skforecast tuning in one stroke |
| 5 | Multi-table exogenous data (energy demand + weather + holidays) | Joiner/fuzzy_join, MultiAggJoiner, TextEncoder/LLMEncoder | skrub's home turf; skforecast explicitly can't do dirty exog |
| 6 | Explaining original columns through encoder postprocess | TableVectorizer + shapash postprocess, column_associations | Required bridge for any skrub+shapash user; reuses #1 |
| 7 | Group-conditional CP with groups mined from messy columns | GapEncoder/deduplicate + Mondrian/ResNormalised | Novel: conditioning variables derived from free text |
| 8 | Categorical-series forecasting + Titanic end-to-end | ToCategorical, Cleaner, encoders | Cheapest to build; squarely skrub's strengths |

## Explicit non-goals
- Rewrites of aeon classification families, tslearn metrics/clustering/matrix-profile/barycenters, MAPIE paper reproductions and exchangeability tests, shapash backends (SHAP/LIME/ShapIQ) and plot mechanics: pure algorithmics or synthetic-data theory demos where skrub adds nothing. Attempting them would misrepresent skrub as a numeric time-series or attribution library.

## Suggested production order
1–2 from the shortlist (forecasting script + series-as-column wrapper), then reuse their scaffolding for 4 (search) and 6/7 (the two MAPIE/shapash integrations); 3 is the standalone differentiator worth doing early; 5 and 8 are cheap fillers once 1 exists.

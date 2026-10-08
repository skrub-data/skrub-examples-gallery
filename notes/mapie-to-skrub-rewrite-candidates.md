# MAPIE Gallery → skrub Rewrite Candidates

Source: https://mapie.readthedocs.io/en/stable/content/all-examples/ (fetched 2026-09-22; 6 sub-galleries, ~50 examples)
Same exercise as `aeon-`, `tslearn-`, `skforecast-to-skrub-rewrite-candidates.md`.

## Gallery overview
- **Regression (15+1 nb)**: quickstarts (homoscedastic/heteroscedastic 1D, toy intervals, prefit, **time-series tutorial**); advanced (CQR, cross-CQR variants, coverage-width criterion, ResidualNormalisedScore, local/conditional coverage, predictive distribution, coverage validity, **nested-CV tuning**, **EnbPI for time series**); scientific reproductions (jackknife+/Barber2020, JaB/Kim2020, Jaber2025 GP surrogates, **Zaffran2022 adaptive CP for time series**); external notebook: EnbPI+ACI changepoints.
- **Classification (5+1 nb)**: prediction sets quickstart, LAC/APS on 2D, cross-conformal, binary set prediction, Sadinle2019; Cifar10 DL notebook.
- **Conditional CP (8)**: group-conditional intervals/sets (quickstart + advanced), conditional coverage metrics, Mondrian fairness tutorial, Gibbs2023 reproduction.
- **Calibration (4)**: hypothesis testing, Venn-ABERS binary/multiclass, p-value convergence.
- **Risk control (11)**: precision/recall control (binary, multi-risk, custom risk, multi-param, multi-label), FWER comparisons, split fixed-sequence testing, semantic segmentation (×2), **LLM-as-a-judge with abstention**, Blot2025 adaptive CRC.
- **Exchangeability testing (7)**: fixed-dataset & online-stream tests, martingale tests (classifier/regressor), permutation tests, RiskMonitoring drift detection.

## Key structural difference vs previous three libraries
MAPIE is a *wrapper* layer, not a competitor: its estimators are model-agnostic and already accept any sklearn-compatible base. So the natural integration is **skrub inside MAPIE** (`MapieRegressor(SkrubLearner(...))`, `MapieClassifier(pipeline_with_TableVectorizer)`) — skrub contributes the data wrangling; MAPIE contributes coverage guarantees. "Rewrites" here mean: examples where skrub makes the *task* real (messy tabular data, time-aware graphs) instead of synthetic blobs, or where DataOps improves the *evaluation machinery* (splits, tuning, reporting).

## Tier 1 — rewrite directly

### 1. The time-series cluster: `regression/1-quickstart/plot_ts-tutorial`, `plot_timeseries_enbpi`, `plot_zaffran2022_comparison`, ts-changepoint notebook
- All tabularize forecasting (lags → supervised regression + rolling re-fit) — exactly the skforecast/aeon/tslearn shared candidate, so one skrub DataOps script anchors all four galleries.
- skrub mapping: lag/rolling features via `AggJoiner`/`apply_func`; rolling-origin backtest via custom `Splitter` + `mark_as_X(cv=...)`; EnbPI/ACI-style streaming refits via `freeze_after_fit` + `iter_cv_splits`; wrap the fitted graph in `MapieRegressor`/`MapieSubsetRegressor` for the intervals.
- Data: `fetch_electricity_forecasting` / `fetch_bike_sharing` replace synthetic series.

### 2. `regression/2-advanced-analysis/plot_nested-cv` → `ParamSearch` over a conformalized learner
- The example hand-builds nested CV (tune base model, then cross-conformalize). DataOps does it declaratively: `choose_float/choose_from` in the graph, MAPIE step as an `skb.apply` node, results via `skb.cross_validate` + `full_report`/parallel coordinates.
- Strongest "our evaluation plumbing is better" example of the four galleries.

### 3. Conditional CP family (`plot_conditional_conformal_*`, `plot_main-tutorial-mondrian-regression`, `plot_ResidualNormalisedScore_tutorial`, `plot_conditional_coverage`) → groups mined from messy data
- Every group-conditional method needs a *conditioning variable*; real ones live in dirty columns (free-text country names, product descriptions, user agents). skrub's `GapEncoder`/`SimilarityEncoder`/`deduplicate`/`Cleaner` turn those into the latent groups Mondrian/ResNormalised need — and `column_associations` helps pick which columns define groups.
- Rewrite: same coverage guarantees, but groups derived from a real dataframe instead of a supplied integer label.

## Tier 2 — skrub-enhanced variants
- **`risk_control/plot_risk_control_llm_as_a_judge`** → skrub's home turf adjacent: treat judge outputs/high-cardinality categories as text columns encoded with `TextEncoder`/`LLMEncoder`/`StringEncoder` before the classifier whose risk is controlled. Shows skrub + MAPIE on the hot "LLM judge with guarantees" use case.
- **`regression/1-quickstart/plot_heteroscedastic_1d_data`** (+ CQR quickstarts) → same story with a real heteroscedastic tabular dataset (skrub's `fetch_*` bunches) so `TableVectorizer`'s per-dtype routing is exercised; leave 1D-synthetic versions untouched (they teach method behavior, not data skills).
- **`classification/1-quickstart/plot_quickstart_classification` + multi-label risk control** → high-cardinality target/label handling: `ToCategorical`, `CleanCategories`, label-string cleanup feeding `MapieClassifier`/`MapieMultiLabelClassifier`.
- **`regression/1-quickstart/plot_prefit`** → skrub analogue: reuse a fitted `SkrubLearner`/frozen nodes (`get_data`/`set_data`, `freeze_after_fit`) as the "already trained base model".

## Tier 3 — conceptual parallels only
- **Exchangeability testing / martingale tests / RiskMonitoring** → skrub has no drift tooling; adjacent only via `SessionEncoder` (stream → sessions) and TableReport for monitoring tables. Skip as rewrites.
- **Calibration (Venn-ABERS, hypothesis tests)** → fully model-agnostic; a skrub version is just "swap the classifier", too thin for a standalone example.

## Not recommended
- Scientific-article reproductions (Barber2020 jackknife+, Kim2020 JaB, Jaber2025 GP, Gibbs2023, Blot2025), Cifar10/semantic-segmentation risk control, 2D LAC/APS visuals, method-comparison grids on synthetic data: these demonstrate CP theory where skrub adds nothing (and image/2D-CNN settings are out of skrub's scope entirely).

## Suggested order of work
1. TS-CP rewrite (#1) — reuses the skforecast/aeon/tslearn DataOps script; MAPIE supplies the intervals nobody else in the comparison series has. Good capstone: one dataset, four-library contrast (aeon / tslearn / skforecast / skrub+mapie).
2. Nested-CV → `ParamSearch` (#2) — small, clearly superior workflow demo.
3. Groups-from-messy-data conditional CP (#3) — the most novel conceptual pitch.

## Cross-library note (updating the running story)
- aeon/tslearn: skrub complements algorithmic internals (wrappers, structural search).
- skforecast: skrub substitutes the whole tabular-forecasting core, wins on relational exog data.
- MAPIE: skrub *feeds* the method — different libraries' strengths combine; candidate examples should ship with `Mapie*(SkrubLearner(...))` as the pattern to copy.

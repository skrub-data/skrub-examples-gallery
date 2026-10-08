# tslearn Gallery → skrub Rewrite Candidates

Source: https://tslearn.readthedocs.io/en/stable/auto_examples/index.html (fetched 2026-09-22)
Same exercise as `aeon-to-skrub-rewrite-candidates.md`. Context: skrub public API + DataOps patterns proven in `/Users/rcap/work/sessionization/main.py`.

## Gallery overview (28 examples, 7 sections)
- **Metrics (9)**: DTW (+custom metric), CTW, Fréchet, LB_Keogh, LCSS (+custom metric), sDTW, Soft-DTW — all low-level alignment algorithms.
- **Nearest neighbors (4)**: k-NN search, `KNeighborsTimeSeriesClassifier` sklearn-pipeline tuning, neighbors API, 1-NN with SAX+MINDIST.
- **Clustering & barycenters (6)**: DBSCAN, Soft-DTW barycenters, barycenter interpolation, kernel k-means, (s)k-means, KShape.
- **Classification (5)**: early classification, Learning Shapelets (3 views), SVM+GAK.
- **Forecasting (1)**: VARIMA.
- **Automatic differentiation (1)**: Soft-DTW loss for PyTorch.
- **Misc (4)**: matrix/distance profiles (2), PAA+SAX features, model persistence.

Key difference vs aeon: tslearn is a *building-block* library (distances, symbolic features, barycenters), not task pipelines. skrub's leverage points are therefore about **composing tslearn-style primitives into tabular pipelines**, plus its unique DataOps/search story.

## Tier 1 — complementary rewrites (tslearn keeps the primitives; skrub composes and orchestrates)

### 1. `neighbors/plot_knnts_sklearn.html` → DataOps structural search around the tslearn estimator
- `KNeighborsTimeSeriesClassifier` stays as the final estimator; skrub replaces the hand-written GridSearchCV setup: `skb.make_grid_search`/`make_randomized_search` with `choose_from`/`choose_bool` over the **featurization architecture around it** (DTW-kNN branch vs tabular-features branch), compared via `skb.cross_validate` + `full_report`.
- Shows what vanilla GridSearchCV can't — searching pipeline *structure* with tslearn as one of the searched branches, not an alternative to it.

### 2. `misc/plot_sax.html` + `neighbors/plot_sax_mindist_knn.html` → SAX strings meet skrub string encoders
- Complementary by construction: tslearn's SAX transform produces the symbolic word (`"aabccbbc"`) for each series; it lives in a dataframe string column encoded by skrub's `MinHashEncoder` (shared n-grams ≈ MINDIST-style similarity), `SimilarityEncoder`, `GapEncoder` (latent motif topics), `StringEncoder` → any tabular learner via `tabular_pipeline`.
- Narrative: "symbolic time-series features are just text — skrub's text machinery applies".

### 3. `forecasting/plot_VARIMA.html` → skrub builds the covariates, VARIMA stays the model
- VARIMA consumes skrub-prepared inputs: `AggJoiner` exogenous columns from auxiliary tables, `DatetimeEncoder` calendar features, `to_datetime` parsing, and the forecaster runs as a graph node with a custom time-based splitter (`mark_as_X(cv=...)`) and `freeze_after_fit`.
- Optional: the tabular-forecasting branch (VAR as joined lag columns + HGB) enters `choose_from` next to the VARIMA node — same searched-choice pattern as the aeon candidate; one script serves both galleries' comparisons.

## Tier 2 — hybrid / contrast examples

### 4. `classification/plot_shapelets*.html` → shapelet features as a joined table
- Wrap `ShapeletTransform`/`LearningShapelets` in `ApplyToSubFrame` (same wrapper recipe as the aeon rocket candidate); the resulting (distance-to-k-shapelets) features enter a `TableVectorizer` pipeline, or the discovered shapelets get deduplicated with `deduplicate`/`MinHashEncoder`.

### 5. `metrics/plot_dtw_custom_metric.html` + `plot_ctw.html` → "skrub matching" contrast
- skrub's analogue of custom-similarity alignment: `Joiner`/`fuzzy_join` (TF-IDF char n-grams + KNN + `_matching` rescaling) for *record* alignment vs DTW for *time* alignment. Side-by-side table-join vs series-alignment story.

## Tier 3 — minor parallels only
- `misc/plot_serialize_models.html` → skrub analogue: `SkrubLearner`/data-op graphs are picklable/cloudpickle-able including deferred envs; worth a docs note, not an example.
- Early classification, matrix profiles: event-stream cousins (`SessionEncoder` gap logic) but different goals; skip.

## Not recommended
- Metrics internals (Fréchet, LB_Keogh, LCSS, sDTW, Soft-DTW), barycenters, all clustering (KShape, kernel k-means, DBSCAN), SVM+GAK, PyTorch autodiff: pure numeric-series algorithmics; skrub adds nothing and rewriting would misrepresent both libraries.

## Suggested order of work
1. SAX→string-encoders example (#2) — unique to skrub, cheap to build, strong narrative.
2. Pipeline-tuning → DataOps structural search (#1) — direct upgrade of an existing demo.
3. VARIMA with skrub-built covariates (#3) — shared with the aeon list; one script serves both comparisons.

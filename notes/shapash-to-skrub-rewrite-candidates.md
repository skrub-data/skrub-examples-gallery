# Shapash Tutorials → skrub Rewrite Candidates

Source: https://shapash.readthedocs.io/en/latest/tutorials.html (fetched 2026-09-22; ~35 tutorials in 12 topic groups)
Same exercise as `aeon-`, `tslearn-`, `skforecast-`, `mapie-to-skrub-rewrite-candidates.md`.

## Gallery overview
- **Overview (2)**: launch web app on sample dataset (Titanic), Jupyter overview.
- **Common (1)**: groups of features (+ SmartPredictor variant), custom colors.
- **Debug & what-if (3)**: debugging a Titanic model, recourse/what-if simulation, counterfactual business scenarios.
- **Domain examples (6)**: Keras Titanic, GLM regression, TabICL regression, imbalanced Titanic, **time-series tabular forecasting**, NLP TF-IDF classification.
- **Explainability quality (1)**: building confidence in attributions.
- **Explainer/backends (6)**: SHAP, LIME (+faster), FastTreeSHAP, ShapIQ, custom backend.
- **Report (1)** / **Webapp (1)**: auto-report; adding extra data columns to the app.
- **Plots (7)**: local/contribution/importance/compare/interactions/scatter-prediction plots, data-distribution exploration.
- **Postprocess (1)**: mapping encoded-feature contributions back to original inputs at `compile()`.
- **Production (4)**: SmartPredictor, FastAPI serving, parquet batch scoring.

## Relationship to skrub
Like MAPIE, shapash sits *downstream* of the model, so the pattern is **skrub upstream** (featurization) + shapash (attribution). The friction point is exactly where skrub adds value: SHAP explains the *encoded matrix*; users want explanations of the *original columns/values* — shapash's `postprocess` and feature-groups exist precisely to undo skrub-style encoding, so a skrub-aware version of those tutorials is the natural pitch. skrub's independent counter-proposal for part of this space: `TableReport` (data exploration), `column_associations`, `full_report` (pipeline results).

## Tier 1 — rewrite directly

### 1. `postprocess01` + `common01 groups of features` → "explain the original columns, not the encoder output"
- Flagship integration tutorial: `TableVectorizer`/`GapEncoder`/`StringEncoder` pipeline → shapash `compile(postprocess=...)` aggregating one-hot/topic/SVD contributions back to raw values; skrub's `transformers_`/`get_feature_names_out` supply the mapping automatically instead of a hand-written dict.
- Variant: GapEncoder topics as named groups ("topic *addresses* drove the prediction") — genuinely better explanations than per-column SHAP.

### 2. `debug02 recourse/what-if` + `debug03 counterfactual business scenarios` → what-if through the DataOps graph
- shapash perturbs the final feature matrix; DataOps perturbs the *data*: re-`skb.eval` the graph with a modified env (different join table, session gap, imputation) and compare predictions/attributions. Feature-level counterfactuals shapash structurally cannot do. Most novel skrub-side contribution of all five galleries.

### 3. Titanic cluster: `tutorial01/02`, `debug01`, `domain01/04` → skrub-cleaned Titanic
- Titanic is skrub's canonical dirty table (Name free-text, Cabin codes, Ticket strings, mixed types): run identical shapash web-app/report flows on a `TableVectorizer`/`GapEncoder` pipeline. Shows the full stack end-to-end with minimal new code.

### 4. `domain05 time-series tabular forecasting` → the shared DataOps forecasting rewrite
- Fifth library whose gallery contains the same tabularize-the-series move; joins aeon/tslearn/skforecast/MAPIE around **one skrub script, five-way contrast** (shapash's angle there: explain the forecast's drivers, including categorical exog).

## Tier 2 — skrub-enhanced variants / analogues
- **`domain06 NLP TF-IDF explainability`** → replace TF-IDF+SGD with `StringEncoder`/`TextEncoder`/`LLMEncoder` + HGB; requires #1's contribution-aggregation to stay interpretable — good stretch goal, honest caveat that attributions on learned embeddings are weaker than on TF-IDF terms.
- **`webapp01 add features outside the model` + `plot07 data distributions`** → skrub-side twin: `TableReport` + `column_associations` for the same exploration step before any model exists.
- **`generate_report01`** → contrast with skrub `full_report` (pipeline/report story, different object: explains *results*, not attributions).
- **`prod03 batch scoring parquet`** → skrub on lazy polars: batch-score a parquet dataset with a pickled `SkrubLearner` (dispatch layer is the selling point); FastAPI tutorial transfers as-is, no rewrite needed.

## Tier 3 — skip (attribution internals, skrub adds nothing)
- explainer_and_backend group (SHAP/LIME/FastTreeSHAP/ShapIQ/custom backend), explainability_quality, colors, plot01–06 mechanics, Keras/GLM/TabICL model-backend demos (skrub touches data, not backends), SmartPredictor core API.

## Suggested order of work
1. Postprocess/groups over `TableVectorizer` (#1) — required reading for anyone combining the libraries; unlocks #5's credibility.
2. DataOps what-if/counterfactual (#2) — the differentiated "only skrub can do this" demo.
3. Titanic end-to-end (#3) — cheap, familiar, showcases the combo to both user bases.
4. Fold #4 into the cross-gallery time-series script.

## Cross-library note (final roll-up of all five galleries)
- **aeon / tslearn**: skrub complements numeric-series algorithmics (wrappers, structural search) — modest overlap.
- **skforecast**: skrub substitutes the whole tabular-forecasting core — direct competitor, wins on relational exog data.
- **MAPIE / shapash**: skrub *feeds* them (uncertainty, explanations) — compositional; skrub's encoder outputs are precisely the object their `postprocess`/coverage machinery must handle, so integration tutorials for both share the `TableVectorizer`-aware bridge built for #1.
- The five galleries converge on one skrub asset: the **DataOps timestamped-forecasting script** (sessionization-style), reusable as the backbone of 5 comparison examples.

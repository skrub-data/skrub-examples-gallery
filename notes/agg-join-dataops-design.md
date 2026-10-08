# DataOps `agg_join` — Design Note

Status: **design only, no implementation.** Captures the discussion about a
DataOps-level replacement for `AggJoiner` that does not freeze, how it should
split from a temporal variant, and how `SessionEncoder` composes with it.
Motivated by `examples/agg_joiner_fraud.py`, where `AggJoiner`'s freeze-after-fit
breaks held-out scoring when the `products` (child) table changes.

## The problem: `AggJoiner` freezes its auxiliary table

`skrub.AggJoiner` is a fitted scikit-learn transformer. At `fit` it computes the
auxiliary aggregation once and stores it (`aggregated_aux_table_`); `transform`
just left-joins that **frozen** table onto new rows (skrub docs, `_agg_joiner.py`
warning ~lines 183–195, and `transform` at ~line 375).

Consequences:
- Keys unseen at fit time get null aggregates. In the fraud example, scoring the
  dataset's separate test `products` files collapses to random (roc_auc ≈ 0.50)
  because test baskets aren't in the fitted train aggregation.
- Within cross-validation on a *shared* universe it is fine (the `products`
  variable is not split, so every fold's held-out baskets are covered), and the
  aggregation recipe stays *searchable* — which is why the node-based fraud
  pipeline works and scores well.
- It also cannot express time windows at all: in skrub 0.10.1 `AggJoiner` has
  **no `duration`/`offset`/`window` parameter** — it is a plain equality group-by
  then join. Rolling/lag features had to be hand-built with a `@skrub.deferred`
  function instead.

The fix is to make the aggregation+join a **DataOps node** (like `Apply`/`Concat`)
whose left/right inputs are re-evaluated from the environment on every
`eval`/`transform`/`predict`, rather than an estimator that bakes in `aux_table`.
Passing the evolving table as a `skrub.var("products")` then makes the whole
chain live.

## Two primitives, split by disjoint vs overlapping grain

The natural cut is not "forecasting vs not" but whether a row can belong to only
one group:

| Primitive | Grain | Example | Can a `group key` express it? |
|-----------|-------|---------|-------------------------------|
| `agg_join` | **disjoint** key each row falls into exactly one of | `basket_ID`, calendar bucket, **session id** | yes |
| `rolling_agg_join` | **overlapping** window (a row is in many windows) | "trailing 24 h up to prediction time, excluding now" | no — needs asof/rolling |

Splitting is principled: sessions can't subsume the overlapping case, and the
equi-join can't subsume sessions without first producing a key.

### `agg_join` (equi-join group aggregation, unfrozen)

```
skrub.agg_join(
    left, right,
    *,
    on=None,                     # shared key column name(s)
    left_on=None, right_on=None, # asymmetric keys (pandas-style)
    cols=None,                   # cols to aggregate: name, list, or skrub selector
                                 #   (e.g. s.numeric()); default = all non-key
    agg="mean",                  # op name, list, or {selector|col -> op(s)} map
    how="left",
    suffix="",
)
```
- `left` = main table (one row per entity, preserved); `right` = child table
  (collapsed then joined back). Returns a `DataOp`.
- Convenience mirror: `left.skb.agg_join(right, ...)` forwards `self` as `left`.
- Both sides accept a DataOp or a coercible frame (`as_data_op`); the **live**
  behaviour requires the evolving table to be a `var`.

The `cols=`/`agg=` spec — the main design fork, informed by the fraud example (the
biggest `AggJoiner` annoyance is that `sum/mean/std` cannot mix with `mode`/string
cols in one call; supported ops are `count, mode, min, max, sum, median, mean,
std`, with `sum/median/mean/std` numeric-only). The key insight: **selection should
use skrub selectors** (`from skrub import selectors as s`), the same composable
column-matching objects already accepted by `.skb.apply(cols=...)` and
`ApplyToCols`. Both `cols` and the keys of `agg` should accept a selector.

`cols=` — which child columns to aggregate:
- `cols="cash_price"` (a name) / `cols=["cash_price", "qty"]` (a list)
- **`cols=s.numeric()`** / `cols=s.string() | s.categorical()` (a selector)
- `cols=s.numeric() - s.cols("qty")` (combinators: `|`, `&`, `-`, `^`, `~`)

`agg=` — how to aggregate them, either uniform or routed by column/selector:
- `agg="mean"` → one op for all selected `cols`
- `agg=["mean","std"]` → cross-product cols × ops (== current `AggJoiner`)
- `agg={"cash_price":"sum", "make":"mode"}` → per-column map (explicit names)
- **`agg={s.numeric(): ["sum","mean","std"], s.string(): "mode"}`** → *dtype-routed*
  map: apply a set of ops to every column a selector matches. This is the natural
  fix for the numeric/string footgun — numeric columns get `sum/mean/std`, string
  columns get `mode`, in a single call — and it is exactly what
  `TableVectorizer`'s dtype routing does for encoding, reused here for aggregation.
- (advanced) named polars expressions for full control

Why selectors fit `agg_join` specifically:
- **Delayed resolution** — a selector matches against the child table's dtypes at
  evaluation time (`selector.expand(df)` / `s.select`), which is when we actually
  know the columns. This composes with the whole "unfrozen / live node" idea: if
  the evolving `products` table gains a column, `s.numeric()` picks it up without
  editing the pipeline.
- **Same vocabulary as the rest of skrub** — the reader already knows
  `s.numeric()`, `s.string()`, `s.cardinality_below(n)`, `s.has_nulls()`, and
  combinators; no new mini-language for naming which columns to aggregate.
- The keys of a `dict` selector-map must be selectors or column names (not
  arbitrary callables); collisions (a column matched by two selectors) resolve by
  union of the ops, or by an explicit `key_priority`/ordered-match rule — to be
  pinned down in the spec.
Result columns named `{col}_{agg}` unless an explicit alias is given.

Design rule: **keep `agg_join` key-agnostic.** No `session_gap=`, no `window=`.
Sessionizing and time-truncation stay as separate upstream nodes so `agg_join`
remains a clean reducer and every stage is independently searchable
(`agg=choose_from([...])`, `on=choose_from([...])`, `cols=s.numeric() | s.categorical()`).
A whole aggregation *policy* can also be a choice, e.g.
`agg=choose_from({"stats": {s.numeric(): ["mean","std"]}, "extremes": {s.numeric(): ["min","max"]}})`.

### `rolling_agg_join` (temporal, overlapping — phase 2)

Owns the time logic; leans on `join_asof`/rolling rather than on `agg_join`:
```
skrub.rolling_agg_join(
    predictions, history,
    *, left_on="prediction_time", right_on="t",
    agg={"cnt":"mean","cnt2":"std"},
    window="24h",          # aggregate right rows within a duration of each left key
    offset="1h",           # exclude the current bucket (no same-row leakage)
    min_periods=1,
)
```
This is the piece that finally makes causal lag/rolling features a first-class
DataOps node and replaces the hand-rolled `tabularize`/`build_feature_table`
helpers in `examples/forecast_bike_sharing.py` and the sessionization backbone —
leak-free by construction via `offset`.

## `SessionEncoder` composes with `agg_join` (does not replace it)

Read from `_session_encoder.py`: `SessionEncoder` is a **row-preserving key
generator, not an aggregator.** Params: `split_by` (entity col or list),
`timestamp_col`, `session_gap` (seconds), `suffix` (default `"session_id"`).
Output: same rows plus a `<timestamp_col>_<suffix>` column of per-entity,
gap-partitioned, monotonic integer session ids (`-1` for null timestamps).

So it is exactly the *grain* half that `agg_join` needs for burst/visit/session
aggregation — no new temporal code required:

```
sessions = history.skb.apply(
    SessionEncoder(split_by="user", timestamp_col="t", session_gap=gap_choice)
)
per_user = skrub.agg_join(entity_table, sessions, on="t_session_id", agg={...})
```
- `SessionEncoder` supplies the grain; `agg_join` supplies the reduction.
- It is a **stateless** transformer (output depends only on rows + gap), so as a
  live data-op node the whole chain stays unfrozen and re-segments evolving data.
- Bonus: `session_gap` becomes a **searchable choice** (`choose_from`/`choose_int`),
  so aggregation *granularity* is tuned — the sessionization `main.py` pattern
  generalized.

Limit: sessions partition the *child* stream. A causal "trailing sessions up to
prediction time, excluding now" still needs the left/prediction rows to reach the
history by time, not by a shared id — that is `rolling_agg_join`'s job, not the
session + equi-join's. Keep both primitives.

## What `AggJoiner`'s siblings imply
- `MultiAggJoiner` (several tables) becomes "call `agg_join` per table."
- The "aggregate the same day/hour" case = truncate the timestamp into a key,
  then `agg_join` — no dedicated primitive needed (see open question 1).

## Open questions
1. Disjoint-but-time-*derived* grain that isn't a session (e.g. "aggregate
   everything in the same day/hour"): add a small `time_trunc` helper, or is
   "truncate timestamp → `agg_join`" enough (no new primitive)?
2. Should `rolling_agg_join` be a **join** (`rolling_agg_join(main, history, …)`
   → main gains window columns, mirrors the forecasting backbone) or a **pure
   windowed aggregate** on one table (then joined manually via `agg_join`)? The
   former is more convenient; the latter more orthogonal.

## Suggested build order
1. `agg_join` (equi, per-column `agg=` dict, unfrozen) — directly fixes the fraud
   example and the disjoint multi-table cases.
2. `SessionEncoder` + `agg_join` composition examples (session/burst aggregation).
3. `rolling_agg_join` (phase 2) — then refactor the forecasting examples onto it.

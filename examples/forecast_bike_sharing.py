r"""
Forecasting bike demand with skrub DataOps
==========================================

A "skrubified" version of the tabular-forecasting recipe that is at the core of
the aeon, tslearn, skforecast, MAPIE and shapash example galleries: turn a time
series into a supervised learning table (lagged/rolling features + calendar
features), split it with a time-aware (rolling-origin) cross-validation scheme,
and fit a model -- then let *which* model (a dedicated forecasting rule vs. a
supervised tabular learner) be a **searched choice** rather than a library
decision.

Everything that those galleries hand-code with bespoke helper functions
(``add_lags``, ``add_time_features``, ``backtesting_splits`` ...) is expressed
here with skrub primitives:

- lags & rolling history of the target   -> ``@skrub.deferred`` + ``.skb.apply_func``
- seasonal profile from history          -> ``skrub.AggTarget`` (fit on the train fold)
- calendar features from timestamps      -> ``skrub.DatetimeEncoder`` (inside ``TableVectorizer``)
- time-aware, leakage-free CV            -> custom splitter via ``.skb.mark_as_X(cv=...)``
- "which model / which structure" search -> ``skrub.choose_from`` + ``make_randomized_search``

The dataset is skrub's own ``fetch_bike_sharing`` (hourly bike demand, 2011-2012,
with weather & holiday covariates), so the example is fully self-contained.

Extending this example
----------------------
This is the self-contained core of the "DataOps forecasting" backbone that the
notes show up across all five galleries. To specialize it per gallery you add a
node, you do not rewrite the pipeline:

- **aeon / tslearn / skforecast heads**: register the library's model as another
  outcome of the ``paradigm`` ``choose_from`` (wrap it with ``.skb.apply`` the
  same way ``SeasonalNaiveRegressor`` is wrapped) so a dedicated forecaster is
  *searched against* the tabular branch instead of replacing it.
- **multi-table exogenous data** (electricity + weather + holidays): the single
  ``fetch_bike_sharing`` table stands in for several; bring them in with
  ``skrub.Joiner`` / ``fuzzy_join`` / ``MultiAggJoiner`` at the ``tabularize``
  step. Dirty string keys (station names, event descriptions) are what
  ``fuzzy_join`` and the text encoders (``GapEncoder``/``StringEncoder``) are for.
- **uncertainty (MAPIE)**: wrap the ``prediction`` node in
  ``MapieRegressor(SkrubLearner(...))``; the same rolling-origin CV becomes the
  conformal calibration scheme.
- **attribution (shapash)**: the encoder->raw-column bridge (``TableVectorizer.
  transformers_`` / ``get_feature_names_out``) feeds shapash's ``postprocess`` so
  SHAP contributions are reported on the original columns.
- **true future / streaming refit**: this example back-tests over observed data
  (all features are computed before the split, so nothing leaks). When you
  forecast genuinely-unseen future steps, reach for ``.skb.freeze_after_fit()``
  to compute the history-derived tables once at ``fit`` and reuse them at
  ``predict``.

Scoring uses negative MAPE; note that MAPE is unstable on the near-zero nighttime
counts (hence some large values), but the relative ranking of the searched
branches is the point of the example.

Run:  ``python examples/forecast_bike_sharing.py``
"""

# %% [markdown]
# ## Imports & data

import datetime as dt

import polars as pl
import skrub
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import BaseEstimator, RegressorMixin

# Horizon (in hours) we forecast ahead: "day-ahead" demand, issued at the top
# of each hour. All the input features for a row use data observed *up to* that
# hour, so nothing from the prediction window leaks into the features.
HORIZON = 24

# Columns we will build as lag / rolling-history features of the target.
LAGS = (1, 2, 3, 6, 12, 24, 48, 168)  # 168 = same hour last week
ROLLING_WINDOWS = (24, 24 * 7)  # trailing mean/std over 1 day and 1 week


def load_bike_sharing(path):
    """Read the raw hourly bike-demand table and parse its timestamp."""
    return pl.read_csv(path).with_columns(
        date=pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S")
    )


def tabularize(df, horizon=HORIZON, lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    """Turn the (date, cnt, exog...) series into a supervised forecasting table.

    Row ``t`` predicts the demand ``horizon`` hours ahead using only information
    available at ``t``: the target's recent history (lags), trailing summaries
    (rolling mean/std over the past day/week) and the covariates observed at
    ``t``. This is exactly the dataset skforecast/aeon build internally before
    handing it to an estimator -- here it is a single lazy skrub data-op node.
    """
    out = df.sort("date")
    # Target: demand `horizon` hours in the future.
    out = out.with_columns(target=pl.col("cnt").shift(-horizon))
    # Calendar keys used later as aggregation keys (kept as plain integers so
    # they can serve as join keys; richer calendar features come from
    # DatetimeEncoder downstream).
    out = out.with_columns(
        hour=pl.col("date").dt.hour(),
        weekday=pl.col("date").dt.weekday(),
    )
    # Lagged history of the target (all strictly at-or-before t -> causal).
    for lag in lags:
        out = out.with_columns(pl.col("cnt").shift(lag).alias(f"lag_{lag}"))
    # Trailing rolling summaries of the observed history up to t.
    for width in rolling_windows:
        shifted = pl.col("cnt")
        out = out.with_columns(
            shifted.rolling_mean(width, min_samples=width).alias(f"roll_mean_{width}h"),
            shifted.rolling_std(width, min_samples=width).alias(f"roll_std_{width}h"),
        )
    # Drop warm-up rows (null lags/rolling) and the horizon rows at the tail
    # (null target). At real forecast time these are supplied instead.
    return out.drop_nulls(["target", "lag_168", "roll_std_168h"])


# %% [markdown]
# ## A leakage-free, rolling-origin time splitter
#
# This is skrub's replacement for each gallery's bespoke ``backtest`` loop: a
# plain scikit-learn-compatible splitter passed to ``.skb.mark_as_X(cv=...)``.
# Everything before ``mark_as_X`` runs once on the whole series (so lags see the
# full timeline); everything after is fit on the training window and scored on
# the strictly-future test window, fold by fold.


class TimeSeriesSplitter:
    def __init__(self, time_col="date", min_train_days=240, test_days=14):
        self.time_col = time_col
        self.min_train_days = min_train_days
        self.test_days = test_days

    def split(self, X, y=None, groups=None):
        t = X[self.time_col]
        start = t.min()
        min_train = start + dt.timedelta(days=self.min_train_days)
        step = dt.timedelta(days=self.test_days)
        gap = dt.timedelta(days=self.test_days)
        test_start = min_train + dt.timedelta(days=self.test_days)
        while test_start < t.max():
            test_end = min(test_start + gap, t.max())
            train = (
                X.with_row_index()
                .filter(pl.col(self.time_col) < test_start - dt.timedelta(days=1))["index"]
                .to_numpy()
            )
            test = (
                X.with_row_index()
                .filter(
                    (pl.col(self.time_col) >= test_start)
                    & (pl.col(self.time_col) < test_end)
                )["index"]
                .to_numpy()
            )
            if len(train) > 100 and len(test) > 100:
                yield train, test
            test_start = test_start + step

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self.split(X)) if X is not None else None


# %% [markdown]
# ## A dedicated forecasting rule, as a rival branch
#
# ``SeasonalNaiveRegressor`` encodes the classic time-series "seasonal naive"
# baseline (predict next day = value observed one week ago). It is deliberately
# *not* a learned model: it stands in for the dedicated forecasters the galleries
# put up front, and it competes -- inside ``choose_from`` -- with a supervised
# skrub tabular pipeline.


class SeasonalNaiveRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lag_col="lag_168"):
        self.lag_col = lag_col

    def fit(self, X, y):
        if self.lag_col not in X.columns:
            raise ValueError(f"column {self.lag_col!r} missing from features")
        return self

    def predict(self, X):
        return X.get_column(self.lag_col).to_numpy()


# %% [markdown]
# ## The DataOps pipeline

def neg_mape(estimator, X, y):
    """Custom scorer: negative MAPE (sklearn maximizes, forecasting wants small %err)."""
    return -mean_absolute_percentage_error(y, estimator.predict(X))


def make_data_op():
    # The raw table is a variable: no data is baked into the pipeline, so the
    # same graph can be cross-validated, searched, pickled and re-run on new
    # data by passing {"bike_path": ...} in the environment.
    bike_path = skrub.var("bike_path")
    bike = bike_path.skb.apply_func(load_bike_sharing)

    # Build the supervised forecasting table (lazy; computed before the split).
    features = bike.skb.apply_func(tabularize)

    # X = everything except the target & the raw observed count; y = the target.
    X = features.skb.drop(["target", "cnt"]).skb.mark_as_X(cv=TimeSeriesSplitter())
    y = features["target"].skb.mark_as_y()

    # Seasonal profile learned *from the training fold only* (leak-free because
    # steps after mark_as_X are fit on train and applied to the future test fold).
    seasonal = X.skb.apply(
        skrub.AggTarget(
            main_key=["hour", "weekday"],
            operations=["mean", "std"],
            suffix="_season",
        ),
        y=y,
    )

    # Encode the timestamp with DatetimeEncoder (circular hour/weekday, no
    # absolute-epoch trend) and clean the weather/holiday covariates.
    vectorizer = skrub.TableVectorizer(
        datetime=skrub.DatetimeEncoder(
            resolution="hour",
            add_weekday=False,
            add_total_seconds=False,
            periodic_encoding="circular",
        )
    )
    encoded = seasonal.skb.apply(vectorizer)

    # ---- The searched head -------------------------------------------------
    # Paradigm A: a dedicated seasonal-naive rule, applied to the raw frame.
    naive = X.skb.apply(SeasonalNaiveRegressor(), y=y)
    # Paradigm B: a supervised tabular pipeline, with its own learner choice.
    learner = skrub.choose_from(
        {
            "ridge": Ridge(alpha=skrub.choose_float(0.1, 100.0, log=True, name="ridge_alpha")),
            "hgb": HistGradientBoostingRegressor(
                max_leaf_nodes=skrub.choose_int(3, 64, log=True, name="max_leaf_nodes"),
                learning_rate=skrub.choose_float(0.02, 0.5, log=True, name="learning_rate"),
                random_state=0,
            ),
            "dummy": DummyRegressor(),
        },
        name="tabular_learner",
    )
    tabular = encoded.skb.apply(learner, y=y)

    # "dedicated forecaster vs supervised tabular" becomes a searched choice.
    prediction = skrub.choose_from(
        {"seasonal-naive": naive, "supervised-tabular": tabular},
        name="paradigm",
    ).as_data_op()

    return prediction.skb.with_scoring(neg_mape)


# %% [markdown]
# ## Inspect, cross-validate and search

ENV = {"bike_path": str(skrub.datasets.fetch_bike_sharing().path)}


def describe():
    data_op = make_data_op()
    print(data_op.skb.describe_param_grid())
    return data_op


def cross_validate():
    # The rolling-origin splitter is already attached via mark_as_X(cv=...);
    # cross_validate picks it up automatically.
    return make_data_op().skb.cross_validate(ENV)


def random_search(n_iter=24):
    search = make_data_op().skb.make_randomized_search(
        n_iter=n_iter,
        random_state=0,
        cv=TimeSeriesSplitter(),
        scoring=neg_mape,
    )
    search.fit(ENV)
    return search.results_.sort_values("mean_test_score", ascending=False)


if __name__ == "__main__":
    print("== parameter grid (the structure that gets searched) ==")
    describe()

    print("\n== cross-validation (rolling origin), default = seasonal-naive baseline ==")
    scores = cross_validate()["test_score"]
    print(f"{len(scores)} folds, mean -MAPE = {scores.mean():.3f}")

    print("\n== randomized search over paradigm + features + learner ==")
    results = random_search(n_iter=16)
    print(
        results[["paradigm", "tabular_learner", "mean_test_score"]]
        .head(10)
        .to_string(index=False)
    )

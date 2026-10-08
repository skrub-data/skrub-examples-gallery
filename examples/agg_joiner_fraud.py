r"""
Multi-table fraud detection with skrub DataOps + AggJoiner
==========================================================

A non-forecasting showcase of skrub's relational feature engineering. The task is
**binary fraud classification** on skrub's ``fetch_credit_fraud`` dataset, which
is deliberately split across two tables joined one-to-many:

* ``baskets``  -- the main table: one row per basket, whose *only* columns are
  ``ID`` and the ``fraud_flag`` target. There are **no features**.
* ``products`` -- the auxiliary table: several line-items per basket
  (``basket_ID`` -> ``ID``), with ``cash_price``, quantity, a brand ``make``, a
  high-cardinality ``item`` category and free-text ``model``.

Every feature must therefore be *manufactured by aggregating the child table and
joining the result onto the parent*. That is exactly what :class:`skrub.AggJoiner`
does, and wrapping it in DataOps turns the aggregation recipe itself -- which
columns to aggregate, which operations, whether to roll up categoricals, how to
encode the text roll-ups, which learner -- into a **searched choice** instead of a
hand-fixed pipeline.

Pipeline (``make_data_op``):

1. aggregate numeric product columns per basket    -> ``AggJoiner``
2. optionally roll up the categorical mode/brand    -> ``AggJoiner`` (searched on/off via ``if_else``)
3. encode the mixed frame (text via a searched path)-> ``TableVectorizer`` + ``StringEncoder``
4. classify the fraud flag (searched learner)        -> ``choose_from``
5. score with ``roc_auc`` / ``average_precision``    (fraud rate is only ~1.3%)

Run:  ``python examples/agg_joiner_fraud.py``
"""

# %% [markdown]
# ## Imports & data

import numpy as np
import polars as pl
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_PRODUCT_COLS = ["cash_price", "Nbr_of_prod_purchas"]
CATEGORICAL_PRODUCT_COLS = ["make", "item"]  # rolled up with "mode"
MAIN_KEY, AUX_KEY = "ID", "basket_ID"
SCORING = ["roc_auc", "average_precision"]


def load_credit_fraud():
    """Return (train, test) environments, each mapping the two raw tables."""
    d = skrub.datasets.fetch_credit_fraud()
    train = {
        "baskets": pl.read_csv(d["baskets_train_path"]),
        "products": pl.read_csv(d["products_train_path"]),
    }
    test = {
        "baskets": pl.read_csv(d["baskets_test_path"]),
        "products": pl.read_csv(d["products_test_path"]),
    }
    return train, test


# %% [markdown]
# ## The DataOps pipeline
#
# Both tables enter as *variables* (``skrub.var``), not baked-in dataframes.
# ``mark_as_X`` sits on the baskets node, so cross-validation splits the entities
# (baskets) *before* the aggregation -- the AggJoiner is then fitted on the
# training baskets and applied to the held-out ones within each fold.

def as_features(baskets):
    """Strip the label so the target column never reaches the model as a feature.

    This runs *upstream* of ``mark_as_X`` (it helps materialize the rows to be
    split), so it must be a plain, stateless function -- not a fitted
    transformer -- to be safely recomputed when scoring brand-new baskets.
    """
    return baskets.drop(["fraud_flag"])


def make_data_op():
    baskets = skrub.var("baskets")
    products = skrub.var("products")

    # ---- Choices that shape the design matrix ------------------------------
    numeric_ops = skrub.choose_from(
        {
            "basic": ["mean", "max"],
            "spread": ["mean", "std", "max"],
            "full": ["sum", "mean", "std", "min", "max"],
        },
        name="numeric_ops",
    )
    use_modes = skrub.choose_bool(name="product_modes")

    X = baskets.skb.apply_func(as_features).skb.mark_as_X()
    y = baskets["fraud_flag"].skb.mark_as_y()

    # ---- 1. numeric roll-ups (spend profile, basket size) ------------------
    numeric_features = X.skb.apply(
        skrub.AggJoiner(
            aux_table=products,
            main_key=MAIN_KEY,
            aux_key=AUX_KEY,
            cols=NUMERIC_PRODUCT_COLS,
            operations=numeric_ops,
        )
    )
    # ---- 2. optional categorical roll-ups (dominant brand / category) -----
    mode_features = numeric_features.skb.apply(
        skrub.AggJoiner(
            aux_table=products,
            main_key=MAIN_KEY,
            aux_key=AUX_KEY,
            cols=CATEGORICAL_PRODUCT_COLS,
            operations=["mode", "count"],
        )
    )
    # only the selected branch is ever fitted -- this is a structural choice.
    augmented = use_modes.as_data_op().skb.if_else(mode_features, numeric_features)

    # ---- 3. vectorize the aggregated frame ---------------------------------
    # The mode columns are high-cardinality text; StringEncoder embeds them. The
    # numeric aggregates need an imputer: a "std" over a single-item basket is
    # undefined (NaN), and more than half of these baskets have one item.
    encoded = augmented.skb.drop([MAIN_KEY]).skb.apply(
        skrub.TableVectorizer(
            numeric=make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler()
            ),
            high_cardinality=skrub.StringEncoder(
                n_components=skrub.choose_int(5, 48, log=True, name="se_components"),
                random_state=0,
            ),
            drop_if_unique=True,
        )
    )

    # ---- 4. searched classifier (imbalanced -> class_weight option) --------
    classifier = skrub.choose_from(
        {
            "logreg": LogisticRegression(
                class_weight=skrub.choose_from(
                    [None, "balanced"], name="class_weight"
                ),
                max_iter=2000,
            ),
            "hgb": HistGradientBoostingClassifier(random_state=0),
        },
        name="learner",
    )
    return encoded.skb.apply(classifier, y=y)


# %% [markdown]
# ## Orchestration: cross-validate, structurally search, evaluate on held-out data

def cross_validate():
    train, _ = load_credit_fraud()
    return make_data_op().skb.cross_validate(train, scoring=SCORING)


def held_out_report():
    """Score the best searched config on baskets the model never trained on.

    ``AggJoiner`` freezes its auxiliary table after ``fit`` (it is a fitted
    transformer, and the per-basket aggregates are computed once from the
    ``products`` universe). So the held-out split must draw its baskets from a
    universe whose products are already known -- we split the *training*
    baskets, keeping every product available. Scoring the dataset's separate
    test files instead would need the join re-run on the new products; that
    "recompute the aggregation on new data" case is exactly what ``what_if``
    illustrates below.
    """
    train, _ = load_credit_fraud()
    op = make_data_op()
    split = op.skb.train_test_split(train, random_state=0, test_size=0.3)
    search = make_data_op().skb.make_randomized_search(
        n_iter=20, random_state=0, scoring="average_precision", n_jobs=-1
    )
    search.fit(split["train"])
    learner = search.best_learner_
    y_true = np.asarray(split["y_test"])
    y_score = learner.predict_proba(split["test"])[:, 1]
    print("best config:", learner.describe_params())
    print(f"in-universe held-out  roc_auc={roc_auc_score(y_true, y_score):.3f}  "
          f"average_precision={average_precision_score(y_true, y_score):.3f}")


# %% [markdown]
# ## The DataOps differentiator: re-evaluate the aggregation on modified data
#
# A fitted sklearn pipeline lets you poke at the final numeric matrix. A DataOps
# graph lets you poke the *source tables* and watch it propagate through every
# join automatically. Here we double basket 1's line-item prices and re-run only
# the feature-building graph -- nothing is "fitted", the aggregate just recomputes.

def what_if():
    baskets = pl.DataFrame({"ID": [1, 2], "fraud_flag": [0, 0]})
    products = pl.DataFrame(
        {
            "basket_ID": [1, 1, 2],
            "cash_price": [100, 50, 30],
            "Nbr_of_prod_purchas": [1, 2, 1],
        }
    )
    features = (
        skrub.var("baskets")
        .skb.drop(["fraud_flag"])
        .skb.apply(
            skrub.AggJoiner(
                aux_table=skrub.var("products"),
                main_key=MAIN_KEY,
                aux_key=AUX_KEY,
                cols="cash_price",
                operations=["sum", "mean"],
            )
        )
    )
    print("original:")
    print(features.skb.eval({"baskets": baskets, "products": products}))

    bumped = products.with_columns(
        pl.when(pl.col("basket_ID") == 1)
        .then(pl.col("cash_price") * 2)
        .otherwise(pl.col("cash_price"))
        .alias("cash_price")
    )
    print("\nafter doubling basket 1's cash_price:")
    print(features.skb.eval({"baskets": baskets, "products": bumped}))


if __name__ == "__main__":
    print("== searched structure (param grid) ==")
    print(make_data_op().skb.describe_param_grid())

    print("\n== cross-validation (mean over folds) ==")
    cv = cross_validate()
    for metric in SCORING:
        print(f"  test_{metric}: {cv[f'test_{metric}'].mean():.3f}")

    print("\n== structural search + held-out test ==")
    held_out_report()

    print("\n== what-if: re-evaluate the aggregation on modified products ==")
    what_if()

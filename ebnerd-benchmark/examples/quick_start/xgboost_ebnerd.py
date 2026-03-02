"""
XGBoost hybrid recommendation model for the EB-NeRD dataset.

Trains a gradient-boosted decision tree combining three signal families:
  - Content features:       article metadata, popularity, sentiment, freshness
  - Collaborative features: user reading-history patterns, category overlap
  - Contextual features:    time of day, device type, impression size

Train on ebnerd_small/train, evaluate on ebnerd_small/validation.

Usage
-----
    .venv/bin/python examples/quick_start/xgboost_ebnerd.py
"""

from pathlib import Path
import polars as pl
import numpy as np
import xgboost as xgb
import gc

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
    DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_IS_SSO_USER_COL,
    DEFAULT_DEVICE_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_USER_COL,
    DEFAULT_AGE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_PREMIUM_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_TOTAL_READ_TIME_COL,
    DEFAULT_SENTIMENT_SCORE_COL,
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
)
from ebrec.utils._behaviors import create_binary_labels_column
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

# ======================== Configuration ========================

PATH = Path("~/ebnerd_data").expanduser()
DATASPLIT = "ebnerd_small"
HISTORY_SIZE = 30
SEED = 42

FEATURE_COLS = [
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_IS_SSO_USER_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_AGE_COL,
    DEFAULT_DEVICE_COL,
    "hour_of_day",
    "day_of_week",
    DEFAULT_CATEGORY_COL,
    DEFAULT_PREMIUM_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_TOTAL_READ_TIME_COL,
    DEFAULT_SENTIMENT_SCORE_COL,
    "article_age_hours",
    "history_len",
    "history_avg_read_time",
    "history_avg_scroll_pct",
    "num_history_categories",
    "category_match",
    "inview_len",
]


# ======================== Data Loading ========================


def load_split(path: Path, history_size: int = 30) -> pl.DataFrame:
    """Load behaviors joined with truncated user reading history."""
    df_history = pl.read_parquet(path / "history.parquet")
    for col in [
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_HISTORY_READ_TIME_COL,
        DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
    ]:
        print(df_history.columns)
        if col in df_history.columns:
            df_history = df_history.with_columns(
                pl.col(col).list.tail(history_size)
            )

    df_behaviors = pl.read_parquet(path / "behaviors.parquet")
    return df_behaviors.join(df_history, on=DEFAULT_USER_COL, how="left")


# ======================== Feature Engineering ========================


def build_user_category_data(
    df: pl.DataFrame,
    article_cat_map: dict[int, int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    From user reading histories, produce:
      1. Per-user count of distinct categories read  (for a 'breadth' feature)
      2. (user_id, category) pairs                   (for a join-based category_match)
    """
    cat_pair_uids: list[int] = []
    cat_pair_cats: list[int] = []
    n_cat_uids: list[int] = []
    n_cat_vals: list[int] = []

    has_hist = DEFAULT_HISTORY_ARTICLE_ID_COL in df.columns
    select_cols = [DEFAULT_USER_COL]
    if has_hist:
        select_cols.append(DEFAULT_HISTORY_ARTICLE_ID_COL)

    for row in df.select(select_cols).unique(subset=DEFAULT_USER_COL).iter_rows():
        uid = row[0]
        aids = row[1] if has_hist else None
        cats: set[int] = set()
        if aids:
            for a in aids:
                c = article_cat_map.get(a)
                if c is not None:
                    cats.add(c)

        n_cat_uids.append(uid)
        n_cat_vals.append(len(cats))
        for c in cats:
            cat_pair_uids.append(uid)
            cat_pair_cats.append(c)

    df_n_cats = pl.DataFrame(
        {
            DEFAULT_USER_COL: pl.Series(n_cat_uids, dtype=pl.UInt32),
            "num_history_categories": pl.Series(n_cat_vals, dtype=pl.Int32),
        }
    )

    df_cat_pairs = pl.DataFrame(
        {
            DEFAULT_USER_COL: pl.Series(cat_pair_uids, dtype=pl.UInt32),
            DEFAULT_CATEGORY_COL: pl.Series(cat_pair_cats, dtype=pl.Int16),
        }
    ).with_columns(pl.lit(1).cast(pl.Int8).alias("category_match"))

    return df_n_cats, df_cat_pairs


def build_feature_df(
    df: pl.DataFrame,
    df_articles: pl.DataFrame,
    article_cat_map: dict[int, int],
) -> pl.DataFrame:
    """
    Expand impression-level behaviours into a flat dataframe with one row per
    (impression, candidate article), carrying all hybrid features and a binary
    click label.
    """
    # ---- impression-level features ----
    df = df.with_columns(
        [
            pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.hour().alias("hour_of_day"),
            pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL)
            .dt.weekday()
            .alias("day_of_week"),
            pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias("inview_len"),
        ]
    )

    # history aggregates
    hist_exprs = []
    if DEFAULT_HISTORY_ARTICLE_ID_COL in df.columns:
        hist_exprs.append(
            pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL)
            .list.len()
            .fill_null(0)
            .alias("history_len")
        )
    if DEFAULT_HISTORY_READ_TIME_COL in df.columns:
        hist_exprs.append(
            pl.col(DEFAULT_HISTORY_READ_TIME_COL)
            .list.mean()
            .alias("history_avg_read_time")
        )
    if DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL in df.columns:
        hist_exprs.append(
            pl.col(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL)
            .list.mean()
            .alias("history_avg_scroll_pct")
        )
    if hist_exprs:
        df = df.with_columns(hist_exprs)

    # user history category breadth
    df_n_cats, df_cat_pairs = build_user_category_data(df, article_cat_map)
    df = df.join(df_n_cats, on=DEFAULT_USER_COL, how="left")

    # binary click labels
    df = create_binary_labels_column(df)

    # drop heavy list columns before the explode to save memory
    keep = [
        DEFAULT_IMPRESSION_ID_COL,
        DEFAULT_USER_COL,
        DEFAULT_IMPRESSION_TIMESTAMP_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
        DEFAULT_LABELS_COL,
        DEFAULT_IS_SUBSCRIBER_COL,
        DEFAULT_IS_SSO_USER_COL,
        DEFAULT_GENDER_COL,
        DEFAULT_AGE_COL,
        DEFAULT_DEVICE_COL,
        "hour_of_day",
        "day_of_week",
        "inview_len",
        "history_len",
        "history_avg_read_time",
        "history_avg_scroll_pct",
        "num_history_categories",
    ]
    df = df.select([c for c in keep if c in df.columns])

    # ---- explode to (impression, candidate) rows ----
    df = df.explode([DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_LABELS_COL])

    # ---- join article metadata ----
    article_cols = [
        DEFAULT_ARTICLE_ID_COL,
        DEFAULT_CATEGORY_COL,
        DEFAULT_PREMIUM_COL,
        DEFAULT_TOTAL_INVIEWS_COL,
        DEFAULT_TOTAL_PAGEVIEWS_COL,
        DEFAULT_TOTAL_READ_TIME_COL,
        DEFAULT_SENTIMENT_SCORE_COL,
        DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    ]
    df = df.join(
        df_articles.select([c for c in article_cols if c in df_articles.columns]),
        left_on=DEFAULT_INVIEW_ARTICLES_COL,
        right_on=DEFAULT_ARTICLE_ID_COL,
        how="left",
    )

    # article freshness (hours since publication)
    if DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL in df.columns:
        df = df.with_columns(
            (
                (
                    pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL)
                    - pl.col(DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL)
                )
                .dt.total_seconds()
                .cast(pl.Float64)
                / 3600.0
            ).alias("article_age_hours")
        )

    # category match: candidate's category appeared in user's history?
    df = df.join(
        df_cat_pairs,
        on=[DEFAULT_USER_COL, DEFAULT_CATEGORY_COL],
        how="left",
    ).with_columns(pl.col("category_match").fill_null(0))

    # cast booleans to int for XGBoost
    for col in [DEFAULT_IS_SUBSCRIBER_COL, DEFAULT_IS_SSO_USER_COL, DEFAULT_PREMIUM_COL]:
        if col in df.columns and df[col].dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Int8))

    return df


def features_to_numpy(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract feature columns as a float64 numpy matrix; nulls become NaN."""
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return np.empty((len(df), 0))
    return np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in cols])


# ======================== Main Pipeline ========================

if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Hybrid Recommender - EB-NeRD")
    print("=" * 60)

    # ---- 1. load ----
    print("\n[1/5] Loading data …")
    df_train = load_split(PATH / DATASPLIT / "train", HISTORY_SIZE)
    print(df_train.head())
    df_val = load_split(PATH / DATASPLIT / "validation", HISTORY_SIZE)
    df_articles = pl.read_parquet(PATH / "articles.parquet")

    article_cat_map = dict(
        zip(
            df_articles[DEFAULT_ARTICLE_ID_COL].to_list(),
            df_articles[DEFAULT_CATEGORY_COL].to_list(),
        )
    )
    print(f"  Train impressions : {df_train.shape[0]:>10,}")
    print(f"  Val impressions   : {df_val.shape[0]:>10,}")
    print(f"  Articles          : {df_articles.shape[0]:>10,}")

    # ---- 2. training features ----
    print("\n[2/5] Building training features …")
    df_train_flat = build_feature_df(df_train, df_articles, article_cat_map)
    del df_train
    gc.collect()

    feature_cols = [c for c in FEATURE_COLS if c in df_train_flat.columns]
    X_train = features_to_numpy(df_train_flat, feature_cols)
    y_train = df_train_flat[DEFAULT_LABELS_COL].to_numpy().astype(np.float32)
    del df_train_flat
    gc.collect()

    print(f"  Samples           : {X_train.shape[0]:>10,}")
    print(f"  Features          : {X_train.shape[1]:>10}")
    print(f"  Positive rate     : {y_train.mean():.4f}")

    # ---- 3. validation features ----
    print("\n[3/5] Building validation features …")
    df_val_flat = build_feature_df(df_val, df_articles, article_cat_map)
    del df_val
    gc.collect()

    X_val = features_to_numpy(df_val_flat, feature_cols)
    y_val = df_val_flat[DEFAULT_LABELS_COL].to_numpy().astype(np.float32)

    print(f"  Samples           : {X_val.shape[0]:>10,}")

    # ---- 4. train ----
    print("\n[4/5] Training XGBoost …")
    pos_weight = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)
    print(f"  scale_pos_weight  : {pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    del X_train, y_train
    gc.collect()

    # ---- 5. evaluate ----
    print("\n[5/5] Evaluating on validation set …")
    y_pred = model.predict_proba(X_val)[:, 1]
    del X_val
    gc.collect()

    # regroup per-candidate scores back to impression-level lists
    df_results = (
        df_val_flat.select(DEFAULT_IMPRESSION_ID_COL, DEFAULT_LABELS_COL)
        .with_columns(pl.Series("pred_score", y_pred))
        .group_by(DEFAULT_IMPRESSION_ID_COL, maintain_order=True)
        .agg(DEFAULT_LABELS_COL, "pred_score")
    )

    metrics = MetricEvaluator(
        labels=df_results[DEFAULT_LABELS_COL].to_list(),
        predictions=df_results["pred_score"].to_list(),
        metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
    )
    metrics.evaluate()

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(metrics)

    # feature importance
    print(f"\n{'=' * 60}")
    print("Feature Importance (gain)")
    print(f"{'=' * 60}")
    max_imp = max(model.feature_importances_)
    for name, imp in sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp / max_imp * 30) if max_imp > 0 else ""
        print(f"  {name:30s} {imp:.4f}  {bar}")

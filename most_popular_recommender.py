from __future__ import annotations

from pathlib import Path

import polars as pl

from prediction_io import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchWriter,
    ProgressTracker,
    iter_parquet_batches,
    parquet_row_count,
)
from utils import get_dataset_path, to_labeled_format


# PARAMETERS
m = 200
w_ctr = 0.6
w_click = 0.4
w_time = 0.0
w_scroll = 0.0
quantile_min_clicks = 0.10
TRAIN_COLUMNS = [
    "article_ids_inview",
    "article_ids_clicked",
    "article_id",
    "read_time",
    "scroll_percentage",
]
TARGET_COLUMNS = ["impression_id", "article_ids_inview", "article_ids_clicked"]


def minmax(col: str) -> pl.Expr:
    return (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())


def _to_lazyframe(dataframe: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return dataframe.lazy() if isinstance(dataframe, pl.DataFrame) else dataframe


def build_popularity_scores(
    train_behaviors: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    source = _to_lazyframe(train_behaviors)
    available_columns = set(source.schema.keys())

    impressions_count = (
        source.select(pl.col("article_ids_inview").alias("article_id"))
        .explode("article_id")
        .drop_nulls(["article_id"])
        .group_by("article_id")
        .len()
        .rename({"len": "impression_count"})
        .collect(streaming=True)
    )

    clicks_count = (
        source.select(pl.col("article_ids_clicked").alias("article_id"))
        .explode("article_id")
        .drop_nulls(["article_id"])
        .group_by("article_id")
        .len()
        .rename({"len": "clicks_count"})
        .collect(streaming=True)
    )

    ctr = impressions_count.join(clicks_count, on="article_id", how="left").with_columns(
        pl.col("clicks_count").fill_null(0)
    )

    ctr_global = ctr.sum()["clicks_count"].item() / ctr.sum()["impression_count"].item()
    min_clicks = int(ctr["clicks_count"].quantile(quantile_min_clicks))

    ctr_smooth = ctr.with_columns(
        ((pl.col("clicks_count") + m * ctr_global) / (pl.col("impression_count") + m)).alias("CTR")
    )

    ctr_smooth = ctr_smooth.with_columns(
        pl.when(pl.col("clicks_count") < min_clicks)
        .then(pl.col("CTR") * 0.5)
        .otherwise(pl.col("CTR"))
        .alias("CTR")
    )

    if {"article_id", "read_time", "scroll_percentage"}.issubset(available_columns):
        engagement = (
            source.select("article_id", "read_time", "scroll_percentage")
            .drop_nulls(["article_id", "read_time", "scroll_percentage"])
            .group_by("article_id")
            .agg(
                pl.col("read_time").mean().alias("mean_read_time"),
                pl.col("scroll_percentage").mean().alias("mean_scroll_percentage"),
            )
            .collect(streaming=True)
        )
        metrics = ctr_smooth.join(engagement, on="article_id", how="left")
    else:
        metrics = ctr_smooth

    normalization_expressions: list[pl.Expr] = [
        minmax("CTR").alias("CTR_norm"),
        minmax("clicks_count").alias("clicks_norm"),
    ]
    if "mean_read_time" in metrics.columns:
        normalization_expressions.append(
            minmax("mean_read_time").fill_null(0.0).alias("read_time_norm")
        )
    else:
        normalization_expressions.append(pl.lit(0.0).alias("read_time_norm"))

    if "mean_scroll_percentage" in metrics.columns:
        normalization_expressions.append(
            minmax("mean_scroll_percentage").fill_null(0.0).alias("scroll_norm")
        )
    else:
        normalization_expressions.append(pl.lit(0.0).alias("scroll_norm"))

    metrics = metrics.with_columns(*normalization_expressions)

    return metrics.with_columns(
        (
            w_ctr * pl.col("CTR_norm")
            + w_click * pl.col("clicks_norm")
            + w_time * pl.col("read_time_norm")
            + w_scroll * pl.col("scroll_norm")
        ).alias("popularity_score")
    ).select("article_id", "popularity_score")


def _predict_batch(
    target_behaviors: pl.DataFrame,
    popularity_scores: pl.DataFrame,
) -> pl.DataFrame:
    similarities = (
        target_behaviors
        .select("impression_id", pl.col("article_ids_inview").alias("article_id"))
        .explode("article_id")
        .join(popularity_scores, on="article_id", how="left")
        .select(
            "impression_id",
            "article_id",
            pl.col("popularity_score").fill_null(0.0).cast(pl.Float32).alias("score"),
        )
    )

    return to_labeled_format(similarities, behaviors=target_behaviors)


def predict(
    train_behaviors: pl.DataFrame,
    target_behaviors: pl.DataFrame,
    prediction_batch_size: int = DEFAULT_BATCH_SIZE,
) -> pl.DataFrame:
    popularity_scores = build_popularity_scores(train_behaviors)
    prediction_batches = [
        _predict_batch(batch, popularity_scores)
        for batch in target_behaviors.iter_slices(n_rows=prediction_batch_size)
    ]
    return pl.concat(prediction_batches, rechunk=False) if prediction_batches else pl.DataFrame()


def predict_to_parquet(
    train_behaviors_path: str | Path | None = None,
    target_behaviors_path: str | Path | None = None,
    output_path: str | Path = "predictions/most_popular.parquet",
    prediction_batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    train_behaviors_path = (
        Path(train_behaviors_path)
        if train_behaviors_path is not None
        else get_dataset_path("BEHAVIORS")
    )
    target_behaviors_path = (
        Path(target_behaviors_path)
        if target_behaviors_path is not None
        else get_dataset_path("BEHAVIORS")
    )

    popularity_scores = build_popularity_scores(
        pl.scan_parquet(train_behaviors_path).select(TRAIN_COLUMNS)
    )
    output_path = Path(output_path)
    total_rows = parquet_row_count(target_behaviors_path)
    wrote_batch = False

    with ParquetBatchWriter(output_path) as writer, ProgressTracker(
        f"Writing {output_path.name}",
        total=total_rows,
        unit="rows",
    ) as progress:
        for batch in iter_parquet_batches(
            target_behaviors_path,
            columns=TARGET_COLUMNS,
            batch_size=prediction_batch_size,
        ):
            prediction_batch = _predict_batch(batch, popularity_scores)
            writer.write(prediction_batch)
            progress.advance(prediction_batch.height)
            wrote_batch = True

    if not wrote_batch:
        raise ValueError("No target behavior rows were available for most-popular prediction.")

    return output_path


if __name__ == "__main__":
    prediction_path = predict_to_parquet()
    print(f"Wrote predictions to {prediction_path.resolve()}")

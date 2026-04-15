from pathlib import Path

import polars as pl
import numpy as np
from scipy.sparse import csr_matrix

from prediction_io import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchWriter,
    ProgressTracker,
    iter_parquet_batches,
    parquet_row_count,
)
from utils import get_dataset_path, to_labeled_format
from implicit.als import AlternatingLeastSquares

TRAIN_COLUMNS = ["user_id", "article_ids_clicked"]
TARGET_COLUMNS = ["impression_id", "user_id", "article_ids_inview", "article_ids_clicked"]


def collaborative_from_behaviors(
    behaviors_df: pl.DataFrame,
    factors: int,
    reg: float,
    iterations: int,
):
    interactions = (
        behaviors_df
        .select("user_id", pl.col("article_ids_clicked").alias("article_id"))
        .explode("article_id")
        .drop_nulls(["user_id", "article_id"])
    )

    user_codes = (
        interactions
        .select(pl.col("user_id").unique().sort())
        .with_row_index("user_idx")
        .select(["user_idx", "user_id"])
    )

    item_codes = (
        interactions
        .select(pl.col("article_id").unique().sort())
        .with_row_index("item_idx")
        .select(["item_idx", "article_id"])
    )

    indexed = (
        interactions
        .join(user_codes, on="user_id", how="inner")
        .join(item_codes, on="article_id", how="inner")
    )

    counts = (
        indexed
        .group_by(["user_idx", "item_idx"])
        .len()
        .rename({"len": "value"})
    )

    rows_u = counts["user_idx"].to_numpy()
    cols_i = counts["item_idx"].to_numpy()
    data = counts["value"].to_numpy().astype(np.float32)

    n_users = user_codes.height
    n_items = item_codes.height

    user_item_csr = csr_matrix((data, (rows_u, cols_i)), shape=(n_users, n_items)).tocsr()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
    )
    model.fit(user_item_csr, show_progress=True)

    return model, user_item_csr, user_codes, item_codes


def build_similarities_for_inviews(
    model: AlternatingLeastSquares,
    user_codes: pl.DataFrame,
    item_codes: pl.DataFrame,
    behaviors: pl.DataFrame,
    batch_rows: int = 1_000_000,
) -> pl.DataFrame:
    candidates = (
        behaviors
        .select(
            "impression_id",
            "user_id",
            pl.col("article_ids_inview").alias("article_id"),
        )
        .explode("article_id")
        .drop_nulls(["user_id", "article_id"])
    )

    mapped = (
        candidates
        .join(user_codes, on="user_id", how="left")
        .join(item_codes, on="article_id", how="left")
        .select(["impression_id", "user_id", "article_id", "user_idx", "item_idx"])
    )

    if mapped.height == 0:
        return pl.DataFrame({"impression_id": [], "article_id": [], "score": []})

    scorable = mapped.filter(
        pl.col("user_idx").is_not_null() & pl.col("item_idx").is_not_null()
    )

    if scorable.height > 0:
        user_idx = scorable.get_column("user_idx").cast(pl.UInt32).to_numpy()
        item_idx = scorable.get_column("item_idx").cast(pl.UInt32).to_numpy()

        U = model.user_factors
        V = model.item_factors

        scorable_scores = np.empty(scorable.height, dtype=np.float32)
        n = scorable.height

        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            u = user_idx[start:end]
            i = item_idx[start:end]
            scorable_scores[start:end] = (U[u] * V[i]).sum(axis=1).astype(np.float32)

        scorable = scorable.with_row_index("row_nr").with_columns(
            pl.Series(name="score", values=scorable_scores)
        )

        mapped = mapped.with_row_index("row_nr").join(
            scorable.select("row_nr", "score"),
            on="row_nr",
            how="left",
        )
    else:
        mapped = mapped.with_row_index("row_nr")

    mapped = mapped.with_columns(
        pl.col("score").fill_null(0.0).cast(pl.Float32)
    )

    return (
        mapped
        .select(["impression_id", "article_id", "score"])
        .group_by("impression_id", "article_id", maintain_order=True)
        .agg(pl.col("score").first())
    )


def predict(
    behaviors: pl.DataFrame,
    factors: int = 50,
    reg: float = 0.01,
    iterations: int = 20,
    batch_rows: int = 1_000_000,
    prediction_batch_size: int = DEFAULT_BATCH_SIZE,
) -> pl.DataFrame:
    """
    Returns the collaborative filtering predictions in labeled format.
    """

    model, _, user_codes, item_codes = collaborative_from_behaviors(
        behaviors_df=behaviors,
        factors=factors,
        reg=reg,
        iterations=iterations,
    )

    prediction_batches: list[pl.DataFrame] = []
    for batch in behaviors.iter_slices(n_rows=prediction_batch_size):
        similarities = build_similarities_for_inviews(
            model=model,
            user_codes=user_codes,
            item_codes=item_codes,
            behaviors=batch,
            batch_rows=batch_rows,
        )

        prediction_batches.append(
            to_labeled_format(similarities, batch).with_columns(
                pl.col("predicted_score")
                .list.eval(pl.element().fill_null(0.0))
                .alias("predicted_score")
            )
        )

    return pl.concat(prediction_batches, rechunk=False) if prediction_batches else pl.DataFrame()


def predict_to_parquet(
    train_behaviors_path: str | Path | None = None,
    target_behaviors_path: str | Path | None = None,
    output_path: str | Path = "predictions/collaborative.parquet",
    factors: int = 50,
    reg: float = 0.01,
    iterations: int = 20,
    batch_rows: int = 1_000_000,
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

    train_behaviors = pl.read_parquet(train_behaviors_path, columns=TRAIN_COLUMNS)
    model, _, user_codes, item_codes = collaborative_from_behaviors(
        behaviors_df=train_behaviors,
        factors=factors,
        reg=reg,
        iterations=iterations,
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
            similarities = build_similarities_for_inviews(
                model=model,
                user_codes=user_codes,
                item_codes=item_codes,
                behaviors=batch,
                batch_rows=batch_rows,
            )
            prediction_batch = to_labeled_format(similarities, batch).with_columns(
                pl.col("predicted_score")
                .list.eval(pl.element().fill_null(0.0))
                .alias("predicted_score")
            )
            writer.write(prediction_batch)
            progress.advance(prediction_batch.height)
            wrote_batch = True

    if not wrote_batch:
        raise ValueError("No target behavior rows were available for collaborative prediction.")

    return output_path


if __name__ == "__main__":
    prediction_path = predict_to_parquet(
        factors=50,
        reg=0.01,
        iterations=20,
        batch_rows=1_000_000,
    )
    print(f"Wrote predictions to {prediction_path.resolve()}")

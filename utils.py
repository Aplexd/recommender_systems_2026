import os
from pathlib import Path
from typing import Sequence

import dotenv
import polars as pl

# Loads variables to environment
dotenv.load_dotenv(".env")


def get_dataset_path(file_name: str) -> Path:
    path = os.getenv(file_name)

    if path is None:
        raise NameError(f"No path specified in .env-file for {file_name}")

    return Path(path)


def _load_parquet(file_name: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """
    Loads a parquet-file

    :param file_name: Name of file in .env-file
    :type file_name: str
    """
    return pl.read_parquet(get_dataset_path(file_name), columns=list(columns) if columns else None)


def _scan_parquet(file_name: str, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    scan = pl.scan_parquet(get_dataset_path(file_name))
    if columns is None:
        return scan

    return scan.select(list(columns))


def load_articles(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("ARTICLES", columns=columns)


def load_embeddings() -> pl.DataFrame:
    dataframe = _load_parquet("EMBEDDINGS")
    dataframe.columns = [dataframe.columns[0], "embedding"]

    return dataframe.with_columns(
        embedding_norm=pl.col("embedding").list.eval(pl.element().pow(2)).list.sum().sqrt()
    )


def load_behaviors(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("BEHAVIORS", columns=columns)


def load_test_behaviors(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("TEST_BEHAVIORS", columns=columns)


def load_history(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("HISTORY", columns=columns)


def load_validation_behaviors(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("VALIDATION_BEHAVIORS", columns=columns)


def load_validation_history(columns: Sequence[str] | None = None) -> pl.DataFrame:
    return _load_parquet("VALIDATION_HISTORY", columns=columns)


def binary_labels(behaviors: pl.DataFrame) -> pl.DataFrame:
    """
    Based on create_binary_labels_column from
    https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/utils/_behaviors.py
    """

    behaviors = behaviors.with_row_index("row_nr")

    labels = (
        behaviors
        .explode("article_ids_inview")
        .with_columns(
            pl.col("article_ids_inview")
            .is_in(pl.col("article_ids_clicked"))
            .cast(pl.Int8)
            .alias("clicked_labels")
        )
        .group_by("row_nr", maintain_order=True)
        .agg(pl.col("clicked_labels"))
    )

    return (
        behaviors
        .join(labels, on="row_nr", how="left")
        .drop("row_nr")
    )


def to_labeled_format(similarities: pl.DataFrame, behaviors: pl.DataFrame) -> pl.DataFrame:
    """
    Transform (impression_id, article_id, score) into
    (impression_id, clicked_labels, predicted_score)

    This version preserves the exact original article order per impression.
    """

    labeled = (
        binary_labels(behaviors=behaviors)
        .select("impression_id", "article_ids_inview", "clicked_labels")
        .with_row_index("imp_row")
    )

    exploded = (
        labeled
        .explode("article_ids_inview", "clicked_labels")
        .with_row_index("candidate_row")
        .rename({"article_ids_inview": "article_id"})
    )

    scored = (
        exploded
        .join(similarities, on=["impression_id", "article_id"], how="left")
        .with_columns(
            pl.col("score").fill_null(0.0).cast(pl.Float32)
        )
    )

    prediction = (
        scored
        .group_by("imp_row", "impression_id", maintain_order=True)
        .agg(
            pl.col("clicked_labels"),
            pl.col("score").alias("predicted_score"),
        )
        .select("impression_id", "clicked_labels", "predicted_score")
    )

    return prediction

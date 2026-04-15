from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from prediction_io import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchWriter,
    ProgressTracker,
    aligned_parquet_row_count,
    iter_aligned_parquet_batches,
)

MODEL_WEIGHTS = {
    "most_popular": 0.7,
    "content_based": 0.25,
    "collaborative": 0.05,
}


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    return [weight / total_weight for weight in weights]


def predict(
    *labeled_dataframes: pl.DataFrame, weights: Sequence[float] | None = None
) -> pl.DataFrame:
    """
    Takes in one or more labeled prediction dataframes with columns:
    - impression_id
    - clicked_labels
    - predicted_score

    Returns a dataframe with the same schema, where predicted_score
    is the elementwise weighted average of all input predicted_score lists.
    If weights are omitted, all inputs are weighted equally.
    """

    if len(labeled_dataframes) == 0:
        raise ValueError("predict() requires at least one dataframe")

    if weights is None:
        normalized_weights = [1 / len(labeled_dataframes)] * len(labeled_dataframes)
    else:
        if len(weights) != len(labeled_dataframes):
            raise ValueError("weights must have the same length as labeled_dataframes")
        normalized_weights = _normalize_weights(weights)

    if len(labeled_dataframes) == 1:
        return labeled_dataframes[0].select(
            "impression_id",
            "clicked_labels",
            pl.col("predicted_score").cast(pl.List(pl.Float32)),
        )

    base = labeled_dataframes[0].select(
        "impression_id",
        "clicked_labels",
        pl.col("predicted_score").alias("predicted_score0"),
    )

    for i, df in enumerate(labeled_dataframes[1:], start=1):
        base = base.join(
            df.select(
                "impression_id",
                pl.col("predicted_score").alias(f"predicted_score{i}"),
            ),
            on="impression_id",
            how="inner",
        )

    score_cols = [col for col in base.columns if col.startswith("predicted_score")]
    weighted_score = sum(
        pl.col(col) * weight for col, weight in zip(score_cols, normalized_weights)
    )

    result = (
        base
        .explode(score_cols)
        .with_columns(
            weighted_score.cast(pl.Float32).alias("weighted_score")
        )
        .group_by("impression_id", "clicked_labels", maintain_order=True)
        .agg(pl.col("weighted_score").alias("predicted_score"))
        .select("impression_id", "clicked_labels", "predicted_score")
        .sort("impression_id")
    )

    return result


def predict_named(
    named_labeled_dataframes: Sequence[tuple[str, pl.DataFrame]],
    model_weights: dict[str, float] = MODEL_WEIGHTS,
) -> pl.DataFrame:
    """
    Builds the hybrid prediction using named model outputs and configured weights.
    Unknown model names are ignored, but every weighted model must be present exactly once.
    """
    available_predictions = {name: dataframe for name, dataframe in named_labeled_dataframes}
    missing_models = [
        model_name for model_name in model_weights if model_name not in available_predictions
    ]
    if missing_models:
        raise ValueError(
            "Missing required prediction files for weighted hybrid: "
            + ", ".join(sorted(missing_models))
        )

    weighted_dataframes = [available_predictions[name] for name in model_weights]
    weights = [model_weights[name] for name in model_weights]
    return predict(*weighted_dataframes, weights=weights)


def _validate_aligned_prediction_batches(
    named_batches: Sequence[tuple[str, pl.DataFrame]],
) -> None:
    base_name, base_batch = named_batches[0]
    base_impression_ids = base_batch.get_column("impression_id")
    base_labels = base_batch.get_column("clicked_labels")

    for name, batch in named_batches[1:]:
        if not base_impression_ids.equals(batch.get_column("impression_id")):
            raise ValueError(
                f"Prediction batches '{base_name}' and '{name}' do not align on impression_id."
            )

        if not base_labels.equals(batch.get_column("clicked_labels")):
            raise ValueError(
                f"Prediction batches '{base_name}' and '{name}' do not align on clicked_labels."
            )


def _combine_prediction_batches(
    named_batches: Sequence[tuple[str, pl.DataFrame]],
    model_weights: dict[str, float],
) -> pl.DataFrame:
    _validate_aligned_prediction_batches(named_batches)

    ordered_names = [name for name, _ in named_batches]
    normalized_weights = _normalize_weights([model_weights[name] for name in ordered_names])
    score_rows_per_model = [
        batch.get_column("predicted_score").to_list() for _, batch in named_batches
    ]

    weighted_rows: list[list[float]] = []
    for score_row_group in zip(*score_rows_per_model):
        row_length = len(score_row_group[0])
        if any(len(score_row) != row_length for score_row in score_row_group[1:]):
            raise ValueError("Aligned prediction batches must have matching candidate counts.")

        weighted_scores = np.zeros(row_length, dtype=np.float32)
        for score_row, weight in zip(score_row_group, normalized_weights):
            weighted_scores += np.asarray(score_row, dtype=np.float32) * weight

        weighted_rows.append(weighted_scores.tolist())

    base_batch = named_batches[0][1]
    return base_batch.select("impression_id", "clicked_labels").with_columns(
        pl.Series(
            name="predicted_score",
            values=weighted_rows,
            dtype=pl.List(pl.Float32),
        )
    )


def predict_named_from_paths(
    named_prediction_paths: Sequence[tuple[str, Path]],
    output_path: str | Path = "./predictions/hybrid_average.parquet",
    model_weights: dict[str, float] = MODEL_WEIGHTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    available_paths = {name: Path(path) for name, path in named_prediction_paths}
    missing_models = [model_name for model_name in model_weights if model_name not in available_paths]
    if missing_models:
        raise ValueError(
            "Missing required prediction files for weighted hybrid: "
            + ", ".join(sorted(missing_models))
        )

    ordered_names = list(model_weights)
    ordered_paths = [available_paths[name] for name in ordered_names]
    output_path = Path(output_path)
    total_rows = aligned_parquet_row_count(ordered_paths)
    wrote_batch = False

    with ParquetBatchWriter(output_path) as writer, ProgressTracker(
        f"Writing {output_path.name}",
        total=total_rows,
        unit="rows",
    ) as progress:
        for batches in iter_aligned_parquet_batches(
            ordered_paths,
            columns=["impression_id", "clicked_labels", "predicted_score"],
            batch_size=batch_size,
        ):
            prediction_batch = _combine_prediction_batches(
                named_batches=list(zip(ordered_names, batches)),
                model_weights=model_weights,
            )
            writer.write(prediction_batch)
            progress.advance(prediction_batch.height)
            wrote_batch = True

    if not wrote_batch:
        raise FileNotFoundError("No aligned prediction rows were found for weighted hybrid.")

    print(f"Wrote weighted hybrid predictions to: {output_path}")
    return output_path


def read_directory_and_predict(
    predictions_dir: str = "./predictions",
    output_path: str = "./predictions/hybrid_average.parquet",
) -> pl.DataFrame:
    """
    Reads every parquet file in predictions_dir except hybrid/average outputs,
    computes the weighted hybrid prediction, and writes it to output_path.
    """
    directory = Path(predictions_dir)

    named_dataframes: list[tuple[str, pl.DataFrame]] = []
    for file in sorted(directory.rglob("*.parquet")):
        if file.name in {"hybrid_average.parquet", "average.parquet"}:
            continue
        print(f"Loading {file}")
        named_dataframes.append((file.stem, pl.read_parquet(file)))

    if not named_dataframes:
        raise FileNotFoundError(f"No usable parquet files found in {directory.resolve()}")

    result = predict_named(named_dataframes)
    result.write_parquet(output_path)
    print(f"Wrote weighted hybrid predictions to: {output_path}")

    return result


def _prediction_paths_from_directory(
    predictions_dir: str | Path,
) -> list[tuple[str, Path]]:
    directory = Path(predictions_dir)
    return [
        (file.stem, file)
        for file in sorted(directory.rglob("*.parquet"))
        if file.name not in {"hybrid_average.parquet", "average.parquet"}
    ]


if __name__ == "__main__":
    prediction_paths = _prediction_paths_from_directory("./predictions")
    if not prediction_paths:
        raise FileNotFoundError("No usable parquet files found in predictions.")
    predict_named_from_paths(prediction_paths)

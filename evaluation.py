from __future__ import annotations

import json
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import polars as pl

from hybrid_average import MODEL_WEIGHTS, predict_named_from_paths
from prediction_io import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchWriter,
    ProgressTracker,
    iter_parquet_batches,
    parquet_row_count,
)
from utils import get_dataset_path

PREDICTIONS_DIR = Path("predictions")
RESULTS_PATH = Path("evaluation_results.json")
REQUIRED_COLUMNS = {"impression_id", "clicked_labels", "predicted_score"}
GENERATED_PREDICTION_STEMS = {"average", "hybrid_average"}


@lru_cache(maxsize=None)
def _discounts(length: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(length, dtype=np.float64) + 2.0)


def _validate_prediction_frame(name: str, dataframe: pl.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction file '{name}' is missing required columns: {missing}")

    if dataframe.height == 0:
        raise ValueError(
            f"Prediction file '{name}' is empty. Regenerate it before evaluation."
        )


def _auc_score(
    labels: np.ndarray,
    scores: np.ndarray,
    positives: int,
    negatives: int,
) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int8, copy=False)
    positive_rank_sum = 0.0

    start = 0
    total = scores.size
    while start < total:
        end = start + 1
        score = sorted_scores[start]
        while end < total and sorted_scores[end] == score:
            end += 1

        average_rank = 0.5 * (start + end - 1) + 1.0
        positive_rank_sum += average_rank * float(sorted_labels[start:end].sum())
        start = end

    return (
        positive_rank_sum - (positives * (positives + 1) / 2.0)
    ) / (positives * negatives)


def _ndcg_from_sorted_labels(
    sorted_labels: np.ndarray,
    positives: int,
    k: int,
) -> float:
    top_k = min(sorted_labels.size, k)
    if top_k == 0:
        return 0.0

    discounts = _discounts(top_k)
    dcg = float(np.dot(sorted_labels[:top_k], discounts))
    ideal_k = min(positives, top_k)
    ideal_dcg = float(discounts[:ideal_k].sum())
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def _evaluate_prediction_batches(
    batches: Iterable[pl.DataFrame],
    progress: ProgressTracker | None = None,
) -> dict[str, float]:
    totals = {"auc": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    evaluated_rows = 0

    for batch in batches:
        label_rows = batch.get_column("clicked_labels").to_list()
        score_rows = batch.get_column("predicted_score").to_list()

        for labels_raw, scores_raw in zip(label_rows, score_rows):
            labels = np.asarray(labels_raw, dtype=np.int8)
            scores = np.asarray(scores_raw, dtype=np.float32)

            if labels.size != scores.size:
                raise ValueError(
                    "Predicted score lists must have the same length as clicked label lists."
                )

            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            if positives == 0 or negatives == 0:
                raise ValueError(
                    "Each evaluated impression must contain at least one clicked and one non-clicked article."
                )

            order_desc = np.argsort(scores, kind="mergesort")[::-1]
            sorted_labels = labels[order_desc].astype(np.float64, copy=False)
            positive_positions = np.flatnonzero(sorted_labels) + 1

            totals["auc"] += _auc_score(
                labels=labels,
                scores=scores,
                positives=positives,
                negatives=negatives,
            )
            totals["mrr"] += float(np.sum(1.0 / positive_positions) / positives)
            totals["ndcg@5"] += _ndcg_from_sorted_labels(sorted_labels, positives, k=5)
            totals["ndcg@10"] += _ndcg_from_sorted_labels(sorted_labels, positives, k=10)
            evaluated_rows += 1

        if progress is not None:
            progress.advance(batch.height)

    if evaluated_rows == 0:
        raise ValueError("No prediction rows were available for evaluation.")

    return {
        metric_name: metric_total / evaluated_rows
        for metric_name, metric_total in totals.items()
    }


def category_coverage(
    labeled_dataframe: pl.DataFrame,
    articles: pl.DataFrame,
    behaviors: pl.DataFrame,
    n_highest: int = 1,
):
    """
    Calculates how many different category-subcategory pairs each user sees and the
    fraction of the total number.

    Returns the average over all users.
    """
    categories_per_article = articles.explode("subcategory").select(
        "article_id", "category", "subcategory"
    )

    all_categories = categories_per_article.select("category", "subcategory").unique()
    number_of_distinct_categories = len(all_categories)

    if n_highest != 1:
        result = (
            labeled_dataframe.join(
                behaviors.select("impression_id", "user_id", "article_ids_inview"),
                on="impression_id",
            )
            .explode("article_ids_inview", "clicked_labels", "predicted_score")
            .group_by("user_id", "impression_id")
            .agg(
                pl.col("article_ids_inview")
                .sort_by("predicted_score", descending=True)
                .head(n_highest)
            )
            .select("user_id", pl.col("article_ids_inview").alias("article_id"))
            .explode("article_id")
            .join(categories_per_article, on="article_id")
            .group_by("user_id")
            .agg(pl.struct("category", "subcategory").unique().len().alias("n_categories"))
            .select(
                "n_categories",
                (pl.col("n_categories") / number_of_distinct_categories).alias("fraction"),
            )
            .mean()
        )
        return result.row(0)

    result = (
        labeled_dataframe.select("impression_id", "predicted_score")
        .join(
            behaviors.select("impression_id", "user_id", "article_ids_inview"),
            on="impression_id",
        )
        .with_columns(
            pl.col("article_ids_inview")
            .list.get(pl.col("predicted_score").list.arg_max())
            .alias("article_id")
        )
        .select("user_id", "article_id")
        .join(categories_per_article, on="article_id", how="inner")
        .group_by("user_id")
        .agg(pl.struct("category", "subcategory").unique().len().alias("n_categories"))
        .select(
            pl.col("n_categories").mean().alias("count"),
            (pl.col("n_categories") / number_of_distinct_categories)
            .mean()
            .alias("fraction"),
        )
    )

    return result.row(0)


def category_coverage_from_paths(
    prediction_path: Path,
    articles_path: Path,
    behaviors_path: Path,
    n_highest: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    categories_per_article = (
        pl.read_parquet(articles_path, columns=["article_id", "category", "subcategory"])
        .explode("subcategory")
        .drop_nulls(["article_id", "category", "subcategory"])
        .select("article_id", "category", "subcategory")
        .unique()
    )
    number_of_distinct_categories = categories_per_article.select(
        pl.struct("category", "subcategory").n_unique().alias("n_categories")
    ).item()

    if n_highest != 1:
        prediction = pl.read_parquet(prediction_path)
        articles = pl.read_parquet(articles_path)
        behaviors = pl.read_parquet(behaviors_path)
        return category_coverage(
            labeled_dataframe=prediction,
            articles=articles,
            behaviors=behaviors,
            n_highest=n_highest,
        )

    total_rows = parquet_row_count(prediction_path)
    prediction_batches = iter_parquet_batches(
        prediction_path,
        columns=["impression_id", "predicted_score"],
        batch_size=batch_size,
    )
    behavior_batches = iter_parquet_batches(
        behaviors_path,
        columns=["impression_id", "user_id", "article_ids_inview"],
        batch_size=batch_size,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pairs_path = Path(temp_dir) / f"{prediction_path.stem}_category_pairs.parquet"

        with ParquetBatchWriter(temp_pairs_path) as writer, ProgressTracker(
            f"Selecting top articles for {prediction_path.stem}",
            total=total_rows,
            unit="rows",
        ) as progress:
            for batch_index, (prediction_batch, behavior_batch) in enumerate(
                zip_longest(prediction_batches, behavior_batches),
                start=1,
            ):
                if prediction_batch is None or behavior_batch is None:
                    raise ValueError(
                        "Prediction and behavior files do not contain the same number of batches."
                    )

                if prediction_batch.height != behavior_batch.height:
                    raise ValueError(
                        f"Prediction and behavior batch {batch_index} do not have the same row count."
                    )

                if not prediction_batch.get_column("impression_id").equals(
                    behavior_batch.get_column("impression_id")
                ):
                    raise ValueError(
                        f"Prediction and behavior batch {batch_index} do not align on impression_id."
                    )

                top_article_pairs = (
                    behavior_batch
                    .with_columns(
                        prediction_batch
                        .get_column("predicted_score")
                        .cast(pl.List(pl.Float32))
                    )
                    .with_columns(
                        pl.col("article_ids_inview")
                        .list.get(pl.col("predicted_score").list.arg_max())
                        .alias("article_id")
                    )
                    .select("user_id", "article_id")
                    .drop_nulls(["article_id"])
                    .join(categories_per_article, on="article_id", how="inner")
                    .select("user_id", "category", "subcategory")
                    .unique()
                )
                progress.advance(behavior_batch.height)
                writer.write(top_article_pairs)

        result = (
            pl.scan_parquet(temp_pairs_path)
            .unique()
            .group_by("user_id")
            .agg(pl.len().alias("n_categories"))
            .select(
                pl.col("n_categories").mean().alias("count"),
                (pl.col("n_categories") / number_of_distinct_categories)
                .mean()
                .alias("fraction"),
            )
            .collect(streaming=True)
        )

    return result.row(0)


def evaluate(
    labeled_dataframe: pl.DataFrame,
    articles: pl.DataFrame,
    behaviors: pl.DataFrame,
) -> dict[str, Any]:
    """
    Takes a labeled dataframe as input and returns evaluations for:
    AUC, MRR, NDCG@5, NDCG@10, and category coverage.
    """
    _validate_prediction_frame("in_memory_prediction", labeled_dataframe)
    accuracy_metrics = _evaluate_prediction_batches([labeled_dataframe])

    count, fraction = category_coverage(
        labeled_dataframe=labeled_dataframe,
        articles=articles,
        behaviors=behaviors,
    )
    accuracy_metrics["category_coverage"] = {"count": count, "fraction": fraction}

    return accuracy_metrics


def evaluate_prediction_file(
    prediction_path: Path,
    articles_path: Path,
    behaviors_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    name = prediction_path.stem
    first_batch: pl.DataFrame | None = None
    total_rows = parquet_row_count(prediction_path)

    def _batches() -> Iterable[pl.DataFrame]:
        nonlocal first_batch
        for batch in iter_parquet_batches(
            prediction_path,
            columns=["impression_id", "clicked_labels", "predicted_score"],
            batch_size=batch_size,
        ):
            if first_batch is None:
                first_batch = batch
                _validate_prediction_frame(name, batch)
            yield batch

    with ProgressTracker(f"Scoring rows for {name}", total=total_rows, unit="rows") as progress:
        accuracy_metrics = _evaluate_prediction_batches(_batches(), progress=progress)

    if first_batch is None:
        raise ValueError(
            f"Prediction file '{name}' is empty. Regenerate it before evaluation."
        )

    print(f"Computing category coverage for {name}")
    count, fraction = category_coverage_from_paths(
        prediction_path=prediction_path,
        articles_path=articles_path,
        behaviors_path=behaviors_path,
        batch_size=batch_size,
    )
    accuracy_metrics["category_coverage"] = {"count": count, "fraction": fraction}
    return accuracy_metrics


def _saved_prediction_paths(
    predictions_dir: Path, excluded_stems: set[str] | None = None
) -> list[tuple[str, Path]]:
    excluded_stems = excluded_stems or set()
    prediction_paths = sorted(
        path for path in predictions_dir.glob("*.parquet") if path.stem not in excluded_stems
    )

    if not prediction_paths:
        suffix = ""
        if excluded_stems:
            suffix = f" after excluding {', '.join(sorted(excluded_stems))}"
        raise FileNotFoundError(
            f"No prediction parquet files found in {predictions_dir.resolve()}{suffix}"
        )

    return [(path.stem, path) for path in prediction_paths]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_to_builtin(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def _has_required_hybrid_inputs(saved_predictions: list[tuple[str, Path]]) -> bool:
    available_names = {name for name, _ in saved_predictions}
    missing_names = [
        model_name for model_name in MODEL_WEIGHTS if model_name not in available_names
    ]
    if missing_names:
        print(
            "Skipping hybrid_average because weighted inputs are missing: "
            + ", ".join(sorted(missing_names))
        )
        return False

    return True


def _write_results(results_path: Path, evaluations: dict[str, Any]) -> None:
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(_to_builtin(evaluations), file, indent=2)


def evaluate_saved_predictions(
    predictions_dir: Path = PREDICTIONS_DIR,
    results_path: Path = RESULTS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    articles_path = get_dataset_path("ARTICLES")
    behaviors_path = get_dataset_path("BEHAVIORS")
    base_prediction_paths = _saved_prediction_paths(
        predictions_dir, excluded_stems=GENERATED_PREDICTION_STEMS
    )

    evaluations: dict[str, Any] = {}
    latest_prediction_mtime = max(path.stat().st_mtime for _, path in base_prediction_paths)
    can_resume_from_results = (
        results_path.exists() and results_path.stat().st_mtime >= latest_prediction_mtime
    )
    if can_resume_from_results:
        try:
            with results_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            evaluations = loaded
    elif results_path.exists():
        print(
            f"Ignoring stale existing results at {results_path.resolve()} because prediction files are newer."
        )

    for name, prediction_path in base_prediction_paths:
        if name in evaluations:
            print(f"Skipping evaluation for {name} because it already exists in {results_path.resolve()}")
            continue

        print(f"Starting evaluation for {name}")
        evaluations[name] = evaluate_prediction_file(
            prediction_path=prediction_path,
            articles_path=articles_path,
            behaviors_path=behaviors_path,
            batch_size=batch_size,
        )
        _write_results(results_path, evaluations)
        print(f"Finished evaluation for {name}")

    if _has_required_hybrid_inputs(base_prediction_paths):
        print(
            "Building weighted hybrid prediction from saved base model outputs "
            "(most_popular=0.7, content_based=0.25, collaborative=0.05)"
        )
        hybrid_average_path = predictions_dir / "hybrid_average.parquet"
        predict_named_from_paths(
            named_prediction_paths=base_prediction_paths,
            output_path=hybrid_average_path,
            batch_size=batch_size,
        )
        print(f"Saved hybrid average prediction to {hybrid_average_path.resolve()}")

        if "hybrid_average" in evaluations:
            print(
                f"Skipping evaluation for hybrid_average because it already exists in {results_path.resolve()}"
            )
        else:
            print("Starting evaluation for hybrid_average")
            evaluations["hybrid_average"] = evaluate_prediction_file(
                prediction_path=hybrid_average_path,
                articles_path=articles_path,
                behaviors_path=behaviors_path,
                batch_size=batch_size,
            )
            _write_results(results_path, evaluations)
            print("Finished evaluation for hybrid_average")

    print(f"Wrote evaluation results to {results_path.resolve()}")
    return evaluations


if __name__ == "__main__":
    evaluate_saved_predictions()

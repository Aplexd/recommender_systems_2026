from __future__ import annotations

from collections.abc import Callable
from typing import Any

from content_based import predict_to_parquet as predict_content_based_to_parquet
from evaluation import evaluate_saved_predictions
from most_popular_recommender import predict_to_parquet as predict_most_popular_to_parquet

try:
    from collaborative import predict_to_parquet as predict_collaborative_to_parquet
except ImportError:
    predict_collaborative_to_parquet = None


def generate_evaluation_plots() -> list[str]:
    try:
        from plot_evaluation_results import generate_plots
    except SystemExit as exc:
        print(f"Skipping evaluation plots: {exc}")
        return []

    return [str(path) for path in generate_plots()]


def predict_all(include_collaborative: bool = True) -> dict[str, Any]:
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("Writing most_popular predictions", predict_most_popular_to_parquet),
        ("Writing content_based predictions", lambda: predict_content_based_to_parquet(verbose=True)),
    ]
    if include_collaborative:
        if predict_collaborative_to_parquet is None:
            print("Skipping collaborative because the 'implicit' package is not installed.")
        else:
            stages.append(("Writing collaborative predictions", predict_collaborative_to_parquet))

    stages.append(("Evaluating saved predictions", evaluate_saved_predictions))
    stages.append(("Creating evaluation plots", generate_evaluation_plots))

    result: dict[str, Any] | None = None
    total_stages = len(stages)
    for index, (label, action) in enumerate(stages, start=1):
        print(f"[{index}/{total_stages}] {label}")
        stage_result = action()
        if label == "Evaluating saved predictions":
            result = stage_result

    return result or {}


if __name__ == "__main__":
    predict_all()

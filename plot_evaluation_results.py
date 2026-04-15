from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard for CLI usage
    raise SystemExit(
        "matplotlib is required to generate plots. Install it with "
        "'pip install matplotlib' or add it to your environment."
    ) from exc


DEFAULT_INPUT_PATH = Path("evaluation_results.json")
DEFAULT_OUTPUT_DIR = Path("plots")
PRETTY_METRIC_NAMES = {
    "auc": "AUC",
    "mrr": "MRR",
    "ndcg@5": "NDCG@5",
    "ndcg@10": "NDCG@10",
    "category_coverage.count": "Category coverage count",
    "category_coverage.fraction": "Category coverage fraction",
}
PRETTY_MODEL_NAMES = {
    "collaborative": "Collaborative",
    "content_based": "Content-Based",
    "most_popular": "Most Popular",
    "hybrid_average": "Hybrid Average",
    "xgboost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create comparison plots from evaluation_results.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the JSON file containing evaluation metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated PNG files will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Attempt to display the plots after saving them.",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not data:
        raise ValueError("The input JSON file must contain a non-empty object.")

    for model_name, metrics in data.items():
        if not isinstance(model_name, str) or not isinstance(metrics, dict):
            raise ValueError(
                "Each top-level entry must map a model name to a metrics object."
            )

    return data


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}

    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=full_key))
        elif isinstance(value, (int, float)):
            flattened[full_key] = float(value)

    return flattened


def collect_metrics(
    results: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    model_names = list(results.keys())
    metrics_by_model = {
        model_name: flatten_metrics(metrics)
        for model_name, metrics in results.items()
    }

    metric_names: list[str] = []
    for model_name in model_names:
        for metric_name in metrics_by_model[model_name]:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    if not metric_names:
        raise ValueError("No numeric metrics were found in the input JSON file.")

    return model_names, metric_names, metrics_by_model


def format_metric_name(metric_name: str) -> str:
    if metric_name in PRETTY_METRIC_NAMES:
        return PRETTY_METRIC_NAMES[metric_name]

    label = metric_name.replace("_", " ").replace(".", " / ")
    return label.title()


def format_model_name(model_name: str) -> str:
    if model_name in PRETTY_MODEL_NAMES:
        return PRETTY_MODEL_NAMES[model_name]

    return model_name.replace("_", " ").title()


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return slug.strip("_")


def add_value_labels(ax: Any, values: list[float]) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02 if ymax > ymin else 0.02

    for index, value in enumerate(values):
        if math.isnan(value):
            continue

        ax.text(
            index,
            value + offset,
            format_value(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_metric(
    model_names: list[str],
    metric_name: str,
    metrics_by_model: dict[str, dict[str, float]],
    colors: list[Any],
    output_dir: Path,
    keep_open: bool,
) -> Path:
    display_model_names = [format_model_name(model_name) for model_name in model_names]
    values = [
        metrics_by_model[model_name].get(metric_name, float("nan"))
        for model_name in model_names
    ]
    figure, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(display_model_names, values, color=colors)
    ax.set_title(format_metric_name(metric_name))
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    ax.margins(y=0.15)
    add_value_labels(ax, values)

    figure.tight_layout()
    output_path = output_dir / f"{slugify(metric_name)}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if not keep_open:
        plt.close(figure)
    return output_path


def plot_overview(
    model_names: list[str],
    metric_names: list[str],
    metrics_by_model: dict[str, dict[str, float]],
    colors: list[Any],
    output_dir: Path,
    keep_open: bool,
) -> Path:
    display_model_names = [format_model_name(model_name) for model_name in model_names]
    columns = min(3, len(metric_names))
    rows = math.ceil(len(metric_names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.8 * rows),
        squeeze=False,
    )

    for ax, metric_name in zip(axes.flat, metric_names):
        values = [
            metrics_by_model[model_name].get(metric_name, float("nan"))
            for model_name in model_names
        ]
        ax.bar(display_model_names, values, color=colors)
        ax.set_title(format_metric_name(metric_name))
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=20)
        ax.margins(y=0.15)
        add_value_labels(ax, values)

    for ax in axes.flat[len(metric_names) :]:
        ax.remove()

    figure.suptitle("Evaluation results overview", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = output_dir / "evaluation_overview.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if not keep_open:
        plt.close(figure)
    return output_path


def build_colors(count: int) -> list[Any]:
    cmap = plt.get_cmap("tab10")
    return [cmap(index % cmap.N) for index in range(count)]


def generate_plots(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> list[Path]:
    results = load_results(input_path)
    model_names, metric_names, metrics_by_model = collect_metrics(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = build_colors(len(model_names))

    created_files = [
        plot_overview(
            model_names=model_names,
            metric_names=metric_names,
            metrics_by_model=metrics_by_model,
            colors=colors,
            output_dir=output_dir,
            keep_open=show,
        )
    ]

    for metric_name in metric_names:
        created_files.append(
            plot_metric(
                model_names=model_names,
                metric_name=metric_name,
                metrics_by_model=metrics_by_model,
                colors=colors,
                output_dir=output_dir,
                keep_open=show,
            )
        )

    return created_files


def main() -> None:
    args = parse_args()
    created_files = generate_plots(
        input_path=args.input,
        output_dir=args.output_dir,
        show=args.show,
    )

    print("Created plot files:")
    for path in created_files:
        print(f" - {path.resolve()}")

    if args.show:
        if plt.get_backend().lower() == "agg":
            print("Plots were saved, but --show is unavailable with the Agg backend.")
        else:
            plt.show()
            plt.close("all")


if __name__ == "__main__":
    main()

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterator

import polars as pl
import numpy as np
from scipy.sparse import csr_matrix

from prediction_io import (
    DEFAULT_BATCH_SIZE,
    ParquetBatchWriter,
    ProgressTracker,
    iter_parquet_batches,
)
from utils import binary_labels, get_dataset_path, load_embeddings, load_history

EPSILON = sys.float_info.epsilon
USER_BATCH_SIZE = 16_384
HISTORY_ITEM_CHUNK_SIZE = 20_000
IMPRESSION_BATCH_SIZE = 50_000
PAIR_CHUNK_SIZE = 100_000
STORE_EMBEDDING_DTYPE = np.float16
HISTORY_COLUMNS = ["user_id", "article_id_fixed"]


@dataclass
class ArticleStore:
    embeddings: np.ndarray
    norms: np.ndarray
    lookup: np.ndarray


@dataclass
class UserFeatureStore:
    user_ids: np.ndarray
    embeddings: np.ndarray
    norms: np.ndarray
    lookup: np.ndarray

    def to_frame(self, limit: int | None = None) -> pl.DataFrame:
        end = self.user_ids.size if limit is None else min(limit, self.user_ids.size)
        return pl.DataFrame(
            {
                "user_id": self.user_ids[:end],
                "embedding": self.embeddings[:end].tolist(),
                "embedding_norm": self.norms[:end],
            }
        )


def _build_lookup(ids: np.ndarray) -> np.ndarray:
    max_id = int(ids.max(initial=-1))
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    lookup[ids] = np.arange(ids.size, dtype=np.int32)
    return lookup


def _build_article_store(article_embeddings: pl.DataFrame) -> ArticleStore:
    article_ids = article_embeddings["article_id"].to_numpy()
    embeddings = np.asarray(
        article_embeddings["embedding"].to_list(),
        dtype=np.float32,
    ).astype(STORE_EMBEDDING_DTYPE, copy=False)
    norms = article_embeddings["embedding_norm"].to_numpy().astype(np.float32, copy=False)
    lookup = _build_lookup(article_ids)
    return ArticleStore(embeddings=embeddings, norms=norms, lookup=lookup)


def _flatten_list_column(list_values: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter(
        (len(values) for values in list_values),
        dtype=np.int32,
        count=len(list_values),
    )

    if lengths.sum(initial=0) == 0:
        return np.empty(0, dtype=np.int32), lengths

    flattened = np.concatenate(
        [np.asarray(values, dtype=np.int32) for values in list_values if values],
        axis=0,
    )
    return flattened, lengths


def _build_user_feature_store(
    article_store: ArticleStore,
    user_history: pl.DataFrame,
    target_users: pl.DataFrame | None = None,
    user_batch_size: int = USER_BATCH_SIZE,
    history_item_chunk_size: int = HISTORY_ITEM_CHUNK_SIZE,
    verbose: bool = False,
) -> UserFeatureStore:
    if target_users is not None:
        user_history = user_history.join(target_users.select("user_id").unique(), on="user_id", how="semi")

    user_ids = user_history["user_id"].to_numpy()
    history_lists = user_history["article_id_fixed"].to_list()

    if user_ids.size == 0:
        return UserFeatureStore(
            user_ids=np.empty(0, dtype=np.uint32),
            embeddings=np.empty((0, article_store.embeddings.shape[1]), dtype=STORE_EMBEDDING_DTYPE),
            norms=np.empty(0, dtype=np.float32),
            lookup=np.empty(0, dtype=np.int32),
        )

    embedding_dim = article_store.embeddings.shape[1]
    article_count = article_store.embeddings.shape[0]
    user_embeddings = np.zeros((user_ids.size, embedding_dim), dtype=STORE_EMBEDDING_DTYPE)
    user_norms = np.zeros(user_ids.size, dtype=np.float32)
    progress_context = (
        ProgressTracker("Building content-based user vectors", total=user_ids.size, unit="users")
        if verbose
        else nullcontext(None)
    )

    with progress_context as progress:
        for start in range(0, user_ids.size, user_batch_size):
            end = min(start + user_batch_size, user_ids.size)
            batch_size = end - start
            batch_histories = history_lists[start:end]
            flat_article_ids, lengths = _flatten_list_column(batch_histories)

            if flat_article_ids.size == 0:
                if progress is not None:
                    progress.advance(batch_size)
                continue

            batch_positions = np.repeat(np.arange(batch_size, dtype=np.int32), lengths)
            in_lookup_range = flat_article_ids < article_store.lookup.size

            if not np.any(in_lookup_range):
                if progress is not None:
                    progress.advance(batch_size)
                continue

            candidate_positions = batch_positions[in_lookup_range]
            candidate_rows = article_store.lookup[flat_article_ids[in_lookup_range]]
            has_embedding = candidate_rows >= 0

            if not np.any(has_embedding):
                if progress is not None:
                    progress.advance(batch_size)
                continue

            valid_positions = candidate_positions[has_embedding]
            valid_rows = candidate_rows[has_embedding]

            counts = np.bincount(valid_positions, minlength=batch_size).astype(np.float32)
            has_history = counts > 0
            weights = (1.0 / counts[valid_positions]).astype(np.float32, copy=False)
            interaction_matrix = csr_matrix(
                (weights, (valid_positions, valid_rows)),
                shape=(batch_size, article_count),
                dtype=np.float32,
            )
            batch_sums = interaction_matrix @ article_store.embeddings

            user_embeddings[start:end] = batch_sums.astype(STORE_EMBEDDING_DTYPE, copy=False)
            user_norms[start:end] = np.linalg.norm(batch_sums, axis=1).astype(np.float32, copy=False)
            user_norms[start:end][~has_history] = 0.0

            if progress is not None:
                progress.advance(batch_size)

    lookup = _build_lookup(user_ids.astype(np.int32, copy=False))

    return UserFeatureStore(
        user_ids=user_ids,
        embeddings=user_embeddings,
        norms=user_norms,
        lookup=lookup,
    )


def _user_store_from_frame(user_features: pl.DataFrame) -> UserFeatureStore:
    user_ids = user_features["user_id"].to_numpy()
    embeddings = np.asarray(user_features["embedding"].to_list(), dtype=np.float32)
    norms = user_features["embedding_norm"].to_numpy().astype(np.float32, copy=False)
    lookup = _build_lookup(user_ids.astype(np.int32, copy=False))
    return UserFeatureStore(user_ids=user_ids, embeddings=embeddings, norms=norms, lookup=lookup)


def _score_flat_pairs(
    user_ids: np.ndarray,
    article_ids: np.ndarray,
    user_store: UserFeatureStore,
    article_store: ArticleStore,
    pair_chunk_size: int,
) -> np.ndarray:
    scores = np.zeros(article_ids.size, dtype=np.float32)

    valid_user_id = user_ids < user_store.lookup.size
    valid_article_id = (article_ids >= 0) & (article_ids < article_store.lookup.size)

    user_rows = np.full(user_ids.size, -1, dtype=np.int32)
    article_rows = np.full(article_ids.size, -1, dtype=np.int32)
    user_rows[valid_user_id] = user_store.lookup[user_ids[valid_user_id]]
    article_rows[valid_article_id] = article_store.lookup[article_ids[valid_article_id]]

    valid_pairs = (user_rows >= 0) & (article_rows >= 0)
    if not np.any(valid_pairs):
        return scores

    valid_indices = np.flatnonzero(valid_pairs)

    for start in range(0, valid_indices.size, pair_chunk_size):
        end = min(start + pair_chunk_size, valid_indices.size)
        pair_idx = valid_indices[start:end]

        user_idx = user_rows[pair_idx]
        article_idx = article_rows[pair_idx]
        norms = user_store.norms[user_idx] * article_store.norms[article_idx]
        nonzero = norms > EPSILON

        if not np.any(nonzero):
            continue

        active_pairs = pair_idx[nonzero]
        active_users = user_rows[active_pairs]
        active_articles = article_rows[active_pairs]

        numerators = np.sum(
            user_store.embeddings[active_users] * article_store.embeddings[active_articles],
            axis=1,
            dtype=np.float32,
        )
        scores[active_pairs] = numerators / norms[nonzero]

    return scores


def _score_behavior_slice(
    behaviors: pl.DataFrame,
    user_store: UserFeatureStore,
    article_store: ArticleStore,
    pair_chunk_size: int,
) -> list[list[float]]:
    inview_lists = behaviors["article_ids_inview"].to_list()
    flat_article_ids, lengths = _flatten_list_column(inview_lists)

    if flat_article_ids.size == 0:
        return [[] for _ in range(behaviors.height)]

    user_ids = behaviors["user_id"].to_numpy()
    repeated_user_ids = np.repeat(user_ids, lengths)
    flat_scores = _score_flat_pairs(
        user_ids=repeated_user_ids,
        article_ids=flat_article_ids,
        user_store=user_store,
        article_store=article_store,
        pair_chunk_size=pair_chunk_size,
    )

    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return [flat_scores[offsets[i] : offsets[i + 1]].tolist() for i in range(lengths.size)]


def _iter_scored_behavior_batches(
    behaviors: pl.DataFrame,
    user_store: UserFeatureStore,
    article_store: ArticleStore,
    impression_batch_size: int = IMPRESSION_BATCH_SIZE,
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
    verbose: bool = False,
) -> Iterator[pl.DataFrame]:
    behavior_columns = ["impression_id", "user_id", "article_ids_inview"]
    has_clicked_labels = "article_ids_clicked" in behaviors.columns
    if has_clicked_labels:
        behavior_columns.append("article_ids_clicked")

    selected = behaviors.select(behavior_columns)
    progress_context = (
        ProgressTracker(
            "Scoring content-based impressions",
            total=selected.height,
            unit="rows",
        )
        if verbose
        else nullcontext(None)
    )

    with progress_context as progress:
        for batch in selected.iter_slices(n_rows=impression_batch_size):
            score_lists = _score_behavior_slice(
                behaviors=batch,
                user_store=user_store,
                article_store=article_store,
                pair_chunk_size=pair_chunk_size,
            )

            if has_clicked_labels:
                batch_result = binary_labels(batch).select("impression_id", "clicked_labels")
            else:
                batch_result = batch.select("impression_id").with_columns(
                    pl.Series(
                        name="clicked_labels",
                        values=[None] * batch.height,
                        dtype=pl.List(pl.Int8),
                    )
                )

            batch_result = batch_result.with_columns(
                pl.Series(
                    name="predicted_score",
                    values=score_lists,
                    dtype=pl.List(pl.Float32),
                )
            )

            if progress is not None:
                progress.advance(batch_result.height)

            yield batch_result


def _score_behaviors(
    behaviors: pl.DataFrame,
    user_store: UserFeatureStore,
    article_store: ArticleStore,
    impression_batch_size: int = IMPRESSION_BATCH_SIZE,
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
    verbose: bool = False,
) -> pl.DataFrame:
    result_batches = list(
        _iter_scored_behavior_batches(
            behaviors=behaviors,
            user_store=user_store,
            article_store=article_store,
            impression_batch_size=impression_batch_size,
            pair_chunk_size=pair_chunk_size,
            verbose=verbose,
        )
    )
    return pl.concat(result_batches, rechunk=False) if result_batches else pl.DataFrame()


def calculate_user_article_similarity(
    behaviors: pl.DataFrame,
    user_features: pl.DataFrame,
    article_embeddings: pl.DataFrame,
    impression_batch_size: int = IMPRESSION_BATCH_SIZE,
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
) -> pl.DataFrame:
    """
    Calculates user-article similarity for every article each user has seen.

    Note:
    - This returns one row per user/article pair and is therefore best suited for
      smaller slices of the dataset.
    """

    user_store = _user_store_from_frame(user_features)
    article_store = _build_article_store(article_embeddings)
    result_batches: list[pl.DataFrame] = []

    selected = behaviors.select("user_id", pl.col("article_ids_inview").alias("article_id"))

    for batch in selected.iter_slices(n_rows=impression_batch_size):
        exploded = batch.explode("article_id").drop_nulls(["article_id"])
        if exploded.height == 0:
            continue

        user_ids = exploded["user_id"].to_numpy()
        article_ids = exploded["article_id"].to_numpy()
        scores = _score_flat_pairs(
            user_ids=user_ids,
            article_ids=article_ids,
            user_store=user_store,
            article_store=article_store,
            pair_chunk_size=pair_chunk_size,
        )

        result_batches.append(
            pl.DataFrame(
                {
                    "user_id": user_ids,
                    "article_id": article_ids,
                    "similarity": scores,
                }
            )
        )

    return pl.concat(result_batches, rechunk=False) if result_batches else pl.DataFrame(
        {"user_id": [], "article_id": [], "similarity": []}
    )


def aggregate_user_features(
    article_features: pl.DataFrame,
    user_history: pl.DataFrame,
    target_users: pl.DataFrame | None = None,
    user_batch_size: int = USER_BATCH_SIZE,
    history_item_chunk_size: int = HISTORY_ITEM_CHUNK_SIZE,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Aggregates the embeddings of each user's history using a mean value.
    """

    article_store = _build_article_store(article_features)
    user_store = _build_user_feature_store(
        article_store=article_store,
        user_history=user_history,
        target_users=target_users,
        user_batch_size=user_batch_size,
        history_item_chunk_size=history_item_chunk_size,
        verbose=verbose,
    )
    return user_store.to_frame()


def predict(
    history: pl.DataFrame,
    articles: pl.DataFrame,  # Not in use
    behaviors: pl.DataFrame,  # Not in use
    test_behaviors: pl.DataFrame,
    article_embeddings: pl.DataFrame,
    user_batch_size: int = USER_BATCH_SIZE,
    history_item_chunk_size: int = HISTORY_ITEM_CHUNK_SIZE,
    impression_batch_size: int = IMPRESSION_BATCH_SIZE,
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Returns the content-based predictions in labeled format.
    """

    del articles, behaviors

    article_store = _build_article_store(article_embeddings)
    target_users = test_behaviors.select("user_id").unique()
    user_store = _build_user_feature_store(
        article_store=article_store,
        user_history=history,
        target_users=target_users,
        user_batch_size=user_batch_size,
        history_item_chunk_size=history_item_chunk_size,
        verbose=verbose,
    )
    return _score_behaviors(
        behaviors=test_behaviors,
        user_store=user_store,
        article_store=article_store,
        impression_batch_size=impression_batch_size,
        pair_chunk_size=pair_chunk_size,
        verbose=verbose,
    )


def predict_to_parquet(
    history_path: str | Path | None = None,
    target_behaviors_path: str | Path | None = None,
    output_path: str | Path = "predictions/content_based.parquet",
    article_embeddings: pl.DataFrame | None = None,
    user_batch_size: int = USER_BATCH_SIZE,
    history_item_chunk_size: int = HISTORY_ITEM_CHUNK_SIZE,
    impression_batch_size: int = IMPRESSION_BATCH_SIZE,
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
    verbose: bool = False,
) -> Path:
    history_path = Path(history_path) if history_path is not None else get_dataset_path("HISTORY")
    target_behaviors_path = (
        Path(target_behaviors_path)
        if target_behaviors_path is not None
        else get_dataset_path("BEHAVIORS")
    )
    article_embeddings = article_embeddings if article_embeddings is not None else load_embeddings()
    history = load_history(columns=HISTORY_COLUMNS) if history_path == get_dataset_path("HISTORY") else pl.read_parquet(
        history_path, columns=HISTORY_COLUMNS
    )

    target_scan = pl.scan_parquet(target_behaviors_path)
    target_schema = set(target_scan.schema.keys())
    target_columns = ["impression_id", "user_id", "article_ids_inview"]
    if "article_ids_clicked" in target_schema:
        target_columns.append("article_ids_clicked")

    target_users = target_scan.select("user_id").unique().collect(streaming=True)
    article_store = _build_article_store(article_embeddings)
    user_store = _build_user_feature_store(
        article_store=article_store,
        user_history=history,
        target_users=target_users,
        user_batch_size=user_batch_size,
        history_item_chunk_size=history_item_chunk_size,
        verbose=verbose,
    )

    output_path = Path(output_path)
    wrote_batch = False
    with ParquetBatchWriter(output_path) as writer:
        for batch in iter_parquet_batches(
            target_behaviors_path,
            columns=target_columns,
            batch_size=impression_batch_size,
        ):
            for scored_batch in _iter_scored_behavior_batches(
                behaviors=batch,
                user_store=user_store,
                article_store=article_store,
                impression_batch_size=impression_batch_size,
                pair_chunk_size=pair_chunk_size,
                verbose=verbose,
            ):
                writer.write(scored_batch)
                wrote_batch = True

    if not wrote_batch:
        raise ValueError("No target behavior rows were available for content-based prediction.")

    return output_path


if __name__ == "__main__":
    output_path = predict_to_parquet(verbose=True)
    print(f"Finished writing predictions to {output_path.resolve()}")

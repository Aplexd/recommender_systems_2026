import polars as pl
import numpy as np
from scipy.sparse import csr_matrix

from utils import load_behaviors, to_labeled_format
from implicit.als import AlternatingLeastSquares


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
    model.fit(user_item_csr)

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
            pl.col("user_id"),
            pl.col("article_ids_inview").alias("article_id"),
        )
        .explode("article_id")
        .drop_nulls(["user_id", "article_id"])
    )

    mapped = (
        candidates
        .join(user_codes, on="user_id", how="inner")
        .join(item_codes, on="article_id", how="inner")
        .select(["user_id", "article_id", "user_idx", "item_idx"])
    )

    if mapped.height == 0:
        return pl.DataFrame({"user_id": [], "article_id": [], "score": []})

    user_idx = mapped.get_column("user_idx").to_numpy()
    item_idx = mapped.get_column("item_idx").to_numpy()

    U = model.user_factors  # (n_users, factors)
    V = model.item_factors  # (n_items, factors)

    scores = np.empty(mapped.height, dtype=np.float32)
    n = mapped.height
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        u = user_idx[start:end]
        i = item_idx[start:end]
        scores[start:end] = (U[u] * V[i]).sum(axis=1).astype(np.float32)

    return (
        mapped
        .with_columns(pl.Series(name="score", values=scores))
        .select(["user_id", "article_id", "score"])
    )


if __name__ == "__main__":
    behaviors = load_behaviors()

    model, user_item_csr, user_codes, item_codes = collaborative_from_behaviors(
        behaviors, factors=50, reg=0.01, iterations=20
    )

    print("user_item_csr shape:", user_item_csr.shape)
    print("n_users:", user_codes.height, "n_items:", item_codes.height)
    print("schemas:")
    print("item_codes:", item_codes.schema)
    print("behaviors inview:", behaviors.select(["user_id", "article_ids_inview"]).schema)

    similarities = build_similarities_for_inviews(
        model=model,
        user_codes=user_codes,
        item_codes=item_codes,
        behaviors=behaviors,
        batch_rows=1_000_000,
    )

    print("similarities head:")
    print(similarities.head(20))

    ready_df = to_labeled_format(similarities, behaviors)

    ready_df = ready_df.with_columns(
        pl.col("predicted_score")
        .list.eval(pl.element().fill_null(0.0))
        .alias("predicted_score")
    )

    print("ready_df head:")
    print(ready_df.head(20))

    imp = int(ready_df["impression_id"][0])
    row = ready_df.filter(pl.col("impression_id") == imp).row(0)
    print("\nExample impression:", imp)
    print("clicked_labels:", row[1])
    print("predicted_score:", row[2])
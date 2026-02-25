from utils import load_behaviors, load_embeddings, load_articles, load_history
import numpy as np
import polars as pl
import polars_distance as pld

"""
Matrices:
- Features per article
- Features per user 

TODO: Take into account both the history and training dataset
"""


def extract_features(test: pl.DataFrame):

    pass


def predict():

    pass


def function(articles: pl.DataFrame,
             embeddings: pl.DataFrame):
    embeddings.columns = [embeddings.columns[0], "embedding"]
    articles.join(embeddings, on="article_id", how="left").select(
        ["article_id", "embedding"])
    return


def unpack_embeddings(embeddings: pl.DataFrame):

    # Assumes all embedding vectors are same size
    embedding_size = len(embeddings.item(0, 1))

    return embeddings.select(
        pl.col("article_id"),
        *(pl.col("embedding").list.get(i).alias(f"embedding_{i}") for i in range(embedding_size))
    )


def aggregate_user_features(article_features: pl.DataFrame,
                            user_history: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates the embeddings of each users history using mean value

    TODO: Generalize to all feature spaces, not just embeddings.
    """

    return (
        user_history
        # Unpack list of articles and join each with its embedding
        .explode("article_id_fixed")
        .join(article_features,
              left_on="article_id_fixed",
              right_on="article_id"
              )
        .select("user_id", "embedding")

        # Add indexes to each embedding, unpack, grouby index and take mean, then repack into embedding per user
        .with_columns(embedding_index=pl.int_ranges(0, pl.col("embedding").list.len()))
        .explode("embedding", "embedding_index")
        .group_by("user_id", "embedding_index", maintain_order=True)
        .mean()
        .group_by("user_id")
        .agg(pl.col("embedding").implode())

        # Add column with norm
        .with_columns(embedding_norm=pl.col("embedding").list.eval(pl.element().pow(2)).list.sum().sqrt())
    )


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     dot_product = np.dot(a, b)

#     denominator = np.linalg.norm(a) * np.linalg.norm(b)

#     return 0 if denominator == 0 else (dot_product / denominator)

# TODO: Abstract this in a way that works
def norm(column: pl.Expr) -> pl.Expr: 
    return column.list.eval(pl.element().pow(2)).list.sum().sqrt()

def cosine_similarity(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return  (a * b).sum() / (norm(a) * norm(b))


if __name__ == "__main__":
    print("Starting content based filtering test")
    behaviors = load_behaviors()
    articles = load_articles()

    history = load_history()

    article_embeddings = load_embeddings()

    print("Article embeddings: \n", article_embeddings)

    user_features = aggregate_user_features(article_embeddings, history)
    print("User features aggregated.")

    print(f"User features:\n", user_features)

    print(f"Behaviors:\n", behaviors)

    # Test prediction
    for i in range(10):
        user_id = user_features.item(i, 0)

        articles_seen_by_user = (
            behaviors
            .select("user_id", pl.col("article_ids_inview").alias("article_id"))
            .filter(pl.col("user_id") == user_id)
            .explode("article_id")
            # User might have been shown the same article multiple times.
            .unique()

        )

        user_article_similarities = (
            articles_seen_by_user

            # Join with user features
            .join(user_features
                  .select("user_id",
                          pl.col("embedding").alias("user_embedding"),
                          pl.col("embedding_norm").alias("user_norm")),
                  on="user_id")
            
            # Join with article features
            .join(article_embeddings
                  .select("article_id",
                          pl.col("embedding").alias("article_embedding"),
                          pl.col("embedding_norm").alias("article_norm")),
                  on="article_id")

            # Calulate cosine similarity
            .explode("article_embedding", "user_embedding")
            .group_by("user_id", "article_id")
            .agg(
                # similarity=cosine_similarity(pl.col("article_embedding"), pl.col("user_embedding"))
                similarity=(pl.col("article_embedding") * pl.col("user_embedding")).sum() / (pl.col("user_norm") * pl.col("article_norm")).first()
                )
                    
        )

        print(user_article_similarities)

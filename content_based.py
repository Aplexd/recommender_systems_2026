from utils import load_behaviors, load_embeddings, load_articles, load_history, to_labeled_format
import polars as pl


def calculate_user_article_similarity(behaviors: pl.DataFrame, 
            user_features: pl.DataFrame,
            article_embeddings: pl.DataFrame):
    """
    Calculates user-article similarity for every article each user has seen
    
    Result tuples: (user_id, article_id, similarity)
    """

    articles_seen_by_user = (
        behaviors
        .select("user_id", pl.col("article_ids_inview").alias("article_id"))
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
            similarity=(pl.col("article_embedding") * pl.col("user_embedding")).sum() / (pl.col("user_norm") * pl.col("article_norm")).first()
            )
                
    )

    return user_article_similarities


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

def predict(history: pl.DataFrame,
            behaviors: pl.DataFrame,
            article_embeddings: pl.DataFrame) -> pl.DataFrame:
    user_features = aggregate_user_features(article_embeddings, history)
    user_article_similarity = calculate_user_article_similarity(behaviors=behaviors, 
                                                                user_features=user_features, 
                                                                article_embeddings=article_embeddings)
    prediction = to_labeled_format(user_article_similarity.select("user_id", "article_id", pl.col("similarity").alias("score")), behaviors=behaviors)

    return prediction


# TODO: Abstract this in a way that works
def norm(column: pl.Expr) -> pl.Expr: 
    return column.list.eval(pl.element().pow(2)).list.sum().sqrt()

def cosine_similarity(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return  (a * b).sum() / (norm(a) * norm(b))


if __name__ == "__main__":
    print("Starting content based filtering test")
    behaviors = load_behaviors()

    history = load_history()

    article_embeddings = load_embeddings()

    print("Article embeddings: \n", article_embeddings)

    user_features = aggregate_user_features(article_embeddings, history)
    print("User features aggregated.")

    print(f"User features:\n", user_features)

    print(f"Behaviors:\n", behaviors)

    user_article_similarity = calculate_user_article_similarity(behaviors=behaviors, 
                                                                user_features=user_features, 
                                                                article_embeddings=article_embeddings)
    print(user_article_similarity)

    prediction = to_labeled_format(user_article_similarity.select("user_id", "article_id", pl.col("similarity").alias("score")), behaviors=behaviors)
    print(prediction)

    prediction.write_parquet("predictions/content_based.parquet", mkdir=True)

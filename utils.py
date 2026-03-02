import dotenv
import polars as pl
import os

# Loads variables to environment
dotenv.load_dotenv(".env")

def _load_parquet(file_name: str) -> pl.DataFrame:
    """
    Loads a parquet-file
    
    :param file_name: Name of file in .env-file
    :type file_name: str
    """
    path = os.getenv(file_name)
    
    if path is None:
        raise NameError(f"No path specified in .env-file for {file_name}")

    return pl.read_parquet(path)


# Loads file content from environment
def load_articles() -> pl.DataFrame:
    return _load_parquet("ARTICLES")

def load_embeddings() -> pl.DataFrame:
    dataframe = _load_parquet("EMBEDDINGS")

    # Rename columns to make switching between embeddings easier
    dataframe.columns = [dataframe.columns[0], "embedding"]
    

    return (dataframe
            # Add column with norm
            .with_columns(embedding_norm=pl.col("embedding").list.eval(pl.element().pow(2)).list.sum().sqrt()))

def load_behaviors() -> pl.DataFrame:
    return _load_parquet("BEHAVIORS")

def load_history() -> pl.DataFrame:
    return _load_parquet("HISTORY")


def binary_labels(behaviors: pl.DataFrame) -> pl.DataFrame:
    """
    Based on create_binary_labels_column from 
    https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/utils/_behaviors.py
    """

    behaviors = behaviors.with_row_index()

    labels = (
        behaviors.explode("article_ids_inview")
        .with_columns(pl.col("article_ids_inview")
                      .is_in(pl.col("article_ids_clicked"))
                      .cast(pl.Int8)
                      .alias("clicked_labels"))
        .group_by("index")
        .agg("clicked_labels")
    )

    return (behaviors
            .join(labels, on="index", how="left")
            .drop("index"))


def to_labeled_format(similarities: pl.DataFrame, 
            behaviors: pl.DataFrame) -> pl.DataFrame:
    """
    Transform the format (user_id, article_id, score) to labeled vector format
    (impression_id, label_vector, prediction_score_vector)
    """
    
    
    labeled = (binary_labels(behaviors=behaviors)
               .select("impression_id", "user_id", pl.col("article_ids_inview").alias("article_id"), "clicked_labels"))
    
    prediction = (labeled
     .explode("article_id", "clicked_labels")
     .join(similarities, on=("user_id", "article_id"), how="left")
     .group_by("impression_id", maintain_order=True)
     .agg("clicked_labels", pl.col("score").alias("predicted_score")))
    
    return prediction

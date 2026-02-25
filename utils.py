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

import dotenv
import polars as pl
import os

# Loads variables to environment
dotenv.load_dotenv()

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
    return _load_parquet("EMBEDDINGS")

def load_behaviors() -> pl.DataFrame:
    return _load_parquet("BEHAVIORS")

def load_history() -> pl.DataFrame:
    return _load_parquet("HISTORY")


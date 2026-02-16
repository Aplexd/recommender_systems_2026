import polars as pl
from utils import load_articles

if __name__ == "__main__":


    articles = load_articles()
    print(articles)
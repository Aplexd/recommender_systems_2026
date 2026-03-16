import polars as pl
from ebrec.evaluation import AucScore, MetricEvaluator, MrrScore, NdcgScore

from content_based import predict as predict_content_based
from utils import load_articles, load_behaviors, load_embeddings, load_history


def category_coverage(labeled_dataframe: pl.DataFrame,
                      articles: pl.DataFrame,
                      behaviors: pl.DataFrame,
                      n_highest: int = 1):
    """
    Calculates how many different category-subcategory pairs each user sees and the fraction of the total number

    Returns the average over all users

    args:
    - n_highest: how many of the highest scored articles per impression we should consider recommended

    return:
        (count, fraction)
    """
    categories_per_article = (articles
                              .explode("subcategory")
                              .select("article_id", "category", "subcategory"))

    all_categories = (categories_per_article
                      .select("category", "subcategory")
                      .unique())

    number_of_distinct_categories = len(all_categories)

    result = (labeled_dataframe
              .join(behaviors
                    .select("impression_id", "user_id", "article_ids_inview"),
                    on="impression_id")
              .explode("article_ids_inview", "clicked_labels", "predicted_score")
              .group_by("user_id", "impression_id")
              .agg(pl.col("article_ids_inview").sort_by("predicted_score", descending=True).head(n_highest))
              .select("user_id", pl.col("article_ids_inview").alias("article_id"))
              .explode("article_id")

              .join(categories_per_article, on="article_id")
              .group_by("user_id")
              .agg(pl.struct("category", "subcategory").unique().len().alias("n_categories"))

              .select("n_categories", (pl.col("n_categories") / number_of_distinct_categories).alias("fraction"))
              .mean()
              )

    return result.row(0)


def evaluate(labeled_dataframe: pl.DataFrame,
             articles: pl.DataFrame,
             behaviors: pl.DataFrame):
    """
    Takes a labeled dataframe as input and returns evaluations for 
    the following metrics: AUC, MRR, NDCG@5, NDCG@10, category_coverage
    TODO: Use intra-list-diversity?
    """
    print("Starting evaluation")

    metrics = MetricEvaluator(
        labels=labeled_dataframe["clicked_labels"].to_list(),
        predictions=labeled_dataframe["predicted_score"].to_list(),
        metric_functions=[AucScore(), MrrScore(),
                          NdcgScore(k=5), NdcgScore(k=10)],
    )

    # .evaluate() sier feilaktig at den gir ut en dict, men gir ut self
    accuracy_metrics = metrics.evaluate().evaluations  # type: ignore

    print("Evaluating category coverage")
    count, fraction = category_coverage(labeled_dataframe=labeled_dataframe,
                                        articles=articles,
                                        behaviors=behaviors)
    accuracy_metrics["category_coverage"] = {
        "count": count, "fraction": fraction}
    print("Evaluation done")

    return accuracy_metrics


if __name__ == "__main__":
    articles = load_articles()
    history = load_history()
    article_embeddings = load_embeddings()
    behaviors = load_behaviors()

    content_based_results = pl.read_parquet(
        "./predictions/content_based.parquet")

    evaluation = evaluate(content_based_results,
                          articles=articles, behaviors=behaviors)
    print(evaluation)

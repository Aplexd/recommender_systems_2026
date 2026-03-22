import polars as pl
import numpy as np
from utils import to_labeled_format


# PARAMETERS 
m = 200 # CTR parameter
w_ctr = 0.6     # weight for ctr
w_click = 0.4 # weight for number of clicks
w_time = 0.0    # weight for read_time
w_scroll = 0.0  # weight for scroll time
penalty_factor = 0.5  # reduce the score of the articles with low clicks
quantile_min_clicks = 0.10

df = pl.read_parquet("datasets/ebnerd_small/train/behaviors.parquet")
df_val = pl.read_parquet("datasets/ebnerd_small/validation/behaviors.parquet")

# impressions
impressions = df.explode("article_ids_inview")
impressions_count = impressions["article_ids_inview"].value_counts().rename({"count": "impression_count", "article_ids_inview": "article_ids"})

# clicks
clicks = df.explode("article_ids_clicked")
clicks_count = clicks["article_ids_clicked"].value_counts().rename({"count": "clicks_count", "article_ids_clicked": "article_ids"})
min_clicks = int(clicks_count['clicks_count'].quantile(0.05))

# CTR
ctr = impressions_count.join(clicks_count, on="article_ids")
ctr_global = ctr.sum()["clicks_count"].item() /ctr.sum()["impression_count"].item()
min_clicks = int(ctr['clicks_count'].quantile(quantile_min_clicks))
# print(min_clicks)

# ctr = ctr.with_columns((pl.col("clicks_count") / pl.col("impression_count")).alias("CTR"))
# ctr = ctr.fill_nan(0)
# print(ctr.sort("CTR",descending=True))


ctr_smooth = ctr.with_columns(((pl.col("clicks_count") + m * ctr_global)/ (pl.col("impression_count")+m)).alias("CTR"))
ctr_smooth = ctr_smooth.fill_nan(0)
ctr_smooth = ctr_smooth.with_columns(
    pl.when(pl.col("clicks_count") < min_clicks)
      .then(pl.col("CTR") * 0.5)  # apply the penalisation
      .otherwise(pl.col("CTR"))   # keep the same value
      .alias("CTR")
)


# Mean read time and scroll percentage per article_id
# Each row in behaviors represents a user reading article_id with those engagement metrics
df_cleaned = df.drop_nulls(["article_id", "read_time", "scroll_percentage"])
engagement = (
    df_cleaned
    .group_by("article_id")
    .agg([
        pl.col("read_time").mean().alias("mean_read_time"),
        pl.col("scroll_percentage").mean().alias("mean_scroll_percentage"),
    ])
    .rename({"article_id": "article_ids"})
)

# Join CTR with engagement metrics
metrics = ctr_smooth.join(engagement, on="article_ids", how="left")

# Normalize each metric to [0, 1] for weighted combination
def minmax(col):
    return (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())

metrics = metrics.with_columns([
    minmax("CTR").alias("CTR_norm"),
    minmax("clicks_count").alias("clicks_norm"),
    minmax("mean_read_time").fill_null(0).alias("read_time_norm"),
    minmax("mean_scroll_percentage").fill_null(0).alias("scroll_norm"),
])

# Weighted popularity score
metrics = metrics.with_columns(
    (w_ctr * pl.col("CTR_norm") + w_click * pl.col("clicks_norm") + w_time * pl.col("read_time_norm") + w_scroll * pl.col("scroll_norm"))
    .alias("popularity_score")
)

ranked = metrics.sort("popularity_score", descending=True)
print(ranked.head(10))

# Softmax sampling over popularity scores
# Temperature T: low = deterministic (top articles), high = uniform
T = 1.0

scores = ranked["popularity_score"].to_numpy()
exp_scores = np.exp((scores - scores.max()) / T)  # shift for numerical stability
softmax_probs = exp_scores / exp_scores.sum()

def softmax_sample(n: int = 10) -> pl.DataFrame:
    """Sample n articles without replacement using softmax probabilities."""
    indices = np.random.choice(len(ranked), size=n, replace=False, p=softmax_probs)
    return ranked[indices.tolist()]

print(softmax_sample(5))

# Save predictions in labeled format
similarities = (
    df_val.explode("article_ids_inview")
    .select("user_id", "article_ids_inview")
    .unique()
    .join(
        ranked.select("article_ids", "popularity_score"),
        left_on="article_ids_inview",
        right_on="article_ids",
        how="left"
    )
    .select(
        pl.col("user_id"),
        pl.col("article_ids_inview").alias("article_id"),
        pl.col("popularity_score").fill_null(0.0).alias("score")
    )
)

prediction = to_labeled_format(similarities, behaviors=df_val)
prediction.write_parquet("predictions/most_popular.parquet")
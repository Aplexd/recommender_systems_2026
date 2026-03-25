import polars as pl
from pathlib import Path


def predict(*labeled_dataframes: pl.DataFrame) -> pl.DataFrame:
    """
    Takes in a list of predictions in the labeled format
    (impression_id, clicked_labels, prediction_score) and
    computes the avereage prediction score.
    """

    def multi_way_join():
        """Joins together the list of dataframes"""
        if len(labeled_dataframes) == 1:
            return labeled_dataframes[0]

        def recurse(cumulated_chain: pl.DataFrame, rest: list[pl.DataFrame], index):
            if len(rest) == 0:
                return cumulated_chain

            next_datafrane = rest[0]
            return recurse(
                cumulated_chain.join(
                    next_datafrane.select(
                        "impression_id",
                        pl.col("predicted_score").alias(
                            f"predicted_score{index}"),
                    ),
                    on="impression_id",
                ),
                rest[1:],
                index=index + 1,
            )

        return recurse(
            labeled_dataframes[0].select(
                "impression_id",
                "clicked_labels",
                pl.col("predicted_score").alias("predicted_score0"),
            ),
            list(labeled_dataframes[1:]),
            index=1,
        )

    return (
        multi_way_join()
        .explode(pl.selectors.starts_with("predicted_score"))
        .group_by(
            pl.col("impression_id"),
            pl.col("clicked_labels"),
        )
        .agg(
            pl.mean_horizontal(pl.selectors.starts_with("predicted_score")).implode().alias(
                "predicted_score"
            )
        )

    )

def read_directory_and_predict():
    """Reads every parquet in the predictions-folder and makes an average prediction"""
    directory = Path("./predictions")
    dataframes = [pl.read_parquet(file) for file in directory.rglob("*.parquet")]

    return predict(*dataframes)


from polars.testing import assert_frame_equal

if __name__ == "__main__":
    content_based_prediction = pl.read_parquet(
        "./predictions/content_based.parquet")

    copy = content_based_prediction.clone()
    copy = copy.with_columns(predicted_score=pl.col("predicted_score").list.eval(pl.element() - 0.5))

    result = predict(content_based_prediction, copy).sort("impression_id")

    print(content_based_prediction)
    print(copy)
    print(result)
    assert_frame_equal(result, content_based_prediction.with_columns(predicted_score=pl.col("predicted_score").list.eval(pl.element() - 0.25)))

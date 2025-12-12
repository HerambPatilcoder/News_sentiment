import pandas as pd

def aggregate_daily_by_ticker(df, time_col="publishedAt", score_col="sentiment", ticker_col="ticker_query"):
    """
    Returns: DataFrame indexed by date + ticker with:
      - article_count
      - mean_sentiment
      - median_sentiment
      - sentiment_std
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[time_col]).dt.date
    agg = df.groupby([ticker_col, "date"])[score_col].agg(
        article_count="count",
        mean_sentiment="mean",
        median_sentiment="median",
        sentiment_std="std"
    ).reset_index()
    agg["mean_sentiment"] = agg["mean_sentiment"].fillna(0.0)
    agg["sentiment_z"] = agg.groupby(ticker_col)["mean_sentiment"].transform(
        lambda x: (x - x.rolling(7, min_periods=1).mean()) / (x.rolling(7, min_periods=1).std().replace(0,1))
    )
    return agg
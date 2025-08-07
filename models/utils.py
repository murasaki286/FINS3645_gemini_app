import pandas as pd

def load_and_merge_features(feature_path, sentiment_path, target_symbol="BTC", version="api"):
    """
    Load structured features and sentiment data, merge on date, and prepare training features and target.

    Parameters:
        feature_path (str): Path to the feature CSV file (API or CSV version).
        sentiment_path (str): Path to the sentiment index CSV file.
        target_symbol (str): Symbol to filter (e.g., 'BTC').
        version (str): 'api' to use API-based features (filters on 'base'),
                       'csv' to use CSV-based features (filters on 'symbol').

    Returns:
        X (DataFrame): Feature matrix including ['return','log_return','momentum','volatility',volume_col,'sentiment_lag1']
        y (Series): Next-period returns ('return_t+1').
        merged (DataFrame): Full merged DataFrame with all columns.
    """
    features = pd.read_csv(feature_path, parse_dates=["date"], low_memory=False)

    if version == "api":
        if "base" not in features.columns:
            raise KeyError("Expected 'base' column in API data.")
        features = features[features["base"] == target_symbol].copy()
    elif version == "csv":
        if "symbol" not in features.columns:
            raise KeyError("Expected 'symbol' column in CSV data.")
        features = features[features["symbol"] == target_symbol].copy()
    else:
        raise ValueError("Invalid version. Use 'api' or 'csv'.")

    features = features.sort_values("date").reset_index(drop=True)
    features = features[features["return"].notna() & (features["return"] != 0)]
    features["return_t+1"] = features["return"].shift(-1)
    features = features.dropna(subset=["return_t+1"])

    sentiment = pd.read_csv(sentiment_path, parse_dates=["date"], low_memory=False)
    sentiment = sentiment.sort_values("date").reset_index(drop=True)
    sentiment = sentiment.assign(sentiment_lag1=sentiment["vader_sentiment"].shift(1))

    merged = pd.merge(
        features,
        sentiment[["date", "sentiment_lag1"]],
        on="date",
        how="left"
    )

    merged = merged.assign(sentiment_lag1=merged["sentiment_lag1"].fillna(0))

    if "quote_volume" in merged.columns:
        volume_col = "quote_volume"
    elif "quote_vol" in merged.columns:
        volume_col = "quote_vol"
    else:
        raise KeyError("Volume column not found in merged DataFrame.")

    feature_cols = [
        "return", "log_return", "momentum", "volatility",
        volume_col, "sentiment_lag1"
    ]

    merged = merged.dropna(subset=feature_cols + ["return_t+1"])

    X = merged[feature_cols]
    y = merged["return_t+1"]
    return X, y, merged

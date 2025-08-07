import os
import pandas as pd
from utils import load_and_merge_features
from models.expanding_window import expanding_window_forecast


def run_pipeline(version="api", symbol="BTC"):
    """
    Executes the Ridge Regression forecasting pipeline for the given data version and symbol.
    """
    print(f"\nRunning Ridge Regression with [{version.upper()}] data for symbol: {symbol}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if version == "api":
        feature_filename = "crypto_features_api.csv"
    elif version == "csv":
        feature_filename = "crypto_features_csv.csv"
    else:
        raise ValueError("Invalid version. Choose 'api' or 'csv'.")
    feature_path = os.path.join(data_dir, feature_filename)
    output_filename = f"btc_predictions_{version}.csv"
    output_path = os.path.join(output_dir, output_filename)

    sentiment_path = os.path.join(data_dir, "crypto_sentiment_index.csv")

    print(f"Loading raw feature data from: {feature_path}")
    raw_feat = pd.read_csv(feature_path, parse_dates=["date"], low_memory=False)
    print(f"Feature date range: {raw_feat['date'].min()} -> {raw_feat['date'].max()}")
    print("Raw feature columns:", list(raw_feat.columns))
    if version == "api" and 'base' in raw_feat.columns:
        cnt_base = (raw_feat['base'] == symbol).sum()
        print(f"Rows where base == {symbol}: {cnt_base}")
    elif version == "csv" and 'symbol' in raw_feat.columns:
        cnt_sym = (raw_feat['symbol'] == symbol).sum()
        print(f"Rows where symbol == {symbol}: {cnt_sym}")

    print(f"Loading raw sentiment data from: {sentiment_path}")
    raw_sent = pd.read_csv(sentiment_path, parse_dates=["date"], low_memory=False)
    print(f"Sentiment date range: {raw_sent['date'].min()} -> {raw_sent['date'].max()}")
    print(f"Merging features and sentiment for symbol: {symbol}")
    X, y, df = load_and_merge_features(
        feature_path,
        sentiment_path,
        target_symbol=symbol,
        version=version
    )
    n = len(y)
    print(f"Merged rows (valid samples): {n}")

    if n < 2:
        print("Not enough data to perform forecasting (need at least 2 samples). Exiting.")
        pd.DataFrame().to_csv(output_path, index=False)
        print(f"\nEmpty results saved to: {output_path}")
        return

    if n <= 15:
        initial_train_size = max(5, n // 2)
        step_size = 1
    elif 15 < n <= 50:
        initial_train_size = n // 3
        step_size = 1
    else:
        initial_train_size = n // 2
        step_size = 1

    print(f"Initial training size: {initial_train_size}, Step size: {step_size}")

    forecast_df = expanding_window_forecast(
        X=X,
        y=y,
        dates=df["date"],
        initial_train_size=initial_train_size,
        step_size=step_size,
        alpha=1.0,
        verbose=True
    )

    forecast_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_pipeline(version="api", symbol="BTC")
    run_pipeline(version="csv", symbol="BTC")

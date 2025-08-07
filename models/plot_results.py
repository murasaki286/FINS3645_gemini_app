import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(df, title, output_path):
    """
    Plot predicted vs actual returns over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["actual_return"], label="Actual Return", linewidth=2)
    plt.plot(df["date"], df["predicted_return"], label="Predicted Return", linewidth=2)
    plt.title(f"{title}: Predicted vs Actual BTC Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved prediction plot to: {output_path}")

def plot_r2(df, title, output_path):
    """
    Plot R-squared values over time.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["r2"], marker="o", linestyle="-", label="R-squared")
    plt.title(f"{title}: R-squared Over Time")
    plt.xlabel("Date")
    plt.ylabel("R²")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved R² plot to: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    api_file = os.path.join(output_dir, "btc_predictions_api.csv")
    csv_file = os.path.join(output_dir, "btc_predictions_csv.csv")

    api_df = pd.read_csv(api_file, parse_dates=["date"])
    csv_df = pd.read_csv(csv_file, parse_dates=["date"])

    plot_predictions(api_df, "API Version", os.path.join(output_dir, "plot_api_predictions.png"))
    plot_r2(api_df, "API Version", os.path.join(output_dir, "plot_api_r2.png"))

    plot_predictions(csv_df, "CSV Version", os.path.join(output_dir, "plot_csv_predictions.png"))
    plot_r2(csv_df, "CSV Version", os.path.join(output_dir, "plot_csv_r2.png"))

if __name__ == "__main__":
    main()

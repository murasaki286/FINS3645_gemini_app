import os
import pandas as pd
import google.generativeai as genai


def load_predictions(predictions_path):
    """
    Load the last 10 rows of prediction CSV for summarization.
    """
    df = pd.read_csv(predictions_path, parse_dates=['date'])
    return df.tail(10)


def load_api_key():
    """
    Load the Gemini API key securely from environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables")
    return api_key


def generate_insight(prompt: str) -> str:
    """
    Generate textual insight using Google Gemini Pro model.
    """
    api_key = load_api_key()
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


def build_prompt(pred_df: pd.DataFrame, version: str) -> str:
    """
    Create a prompt with prediction table + version context.
    """
    df_text = pred_df.to_string(index=False)
    prompt = f"""
You're a FinTech analyst working on a university project.
Below is a Ridge Regression model forecast for BTC using {version.upper()} data (last 10 days):

{df_text}

Write a concise summary (3–5 lines) discussing:
1. Any noticeable prediction trend,
2. Forecast accuracy (visually),
3. Whether model seems stable or volatile.

Respond in a professional tone suitable for an academic report.
"""
    return prompt


def main():
    versions = ["api", "csv"]
    for version in versions:
        file_path = f"models/outputs/btc_predictions_{version}.csv"

        if not os.path.exists(file_path):
            print(f"[Skipped] File not found: {file_path}")
            continue

        print(f"\n--- {version.upper()} DATA SUMMARY ---")
        df = load_predictions(file_path)
        prompt = build_prompt(df, version)

        try:
            summary = generate_insight(prompt)
            print(summary)
        except Exception as e:
            print(f"[Error generating insight] {e}")


if __name__ == "__main__":
    main()

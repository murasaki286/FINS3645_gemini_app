import os
import streamlit as st
import pandas as pd
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not set")
    st.stop()

genai.configure(api_key=api_key)
MODEL_ID = "models/chat-bison-001"

def load_predictions(version: str) -> pd.DataFrame:
    path = os.path.join("models", "outputs", f"btc_predictions_{version}.csv")
    if not os.path.exists(path):
        st.warning(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])

def generate_insight(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_ID)
        chat = model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"[Error generating insight] {e}"

def build_prompt(df: pd.DataFrame, version: str) -> str:
    recent = df.tail(10)
    table_text = recent.to_string(index=False)
    return (
        f"You're an expert FinTech analyst. BTC prediction output using {version.upper()} data (last 10 days):\n\n"
        f"{table_text}\n\n"
        "Provide a concise, professional summary (3–5 lines) focusing on key trends and insights."
    )

st.title("BTC Forecast Summary")
version = st.selectbox("Data version:", ["api", "csv"])
df = load_predictions(version)
if df.empty:
    st.stop()

st.subheader("Recent Predictions")
st.dataframe(df.tail(10))

prompt = build_prompt(df, version)
st.subheader("Insight")
with st.spinner("Generating..."):
    summary = generate_insight(prompt)
st.markdown(summary)

if st.checkbox("Show full table"):
    st.subheader("Full Table")
    st.dataframe(df)

import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

st.set_page_config(
    page_title="BTC Forecast Summary",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
/* page padding */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* section title */
h1, h2, h3 {letter-spacing: .3px;}
/* "card" */
.card {border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 1rem 1.1rem; background: rgba(255,255,255,.03);}
.insight {border-left: 4px solid #7aa2ff; background: rgba(122,162,255,.08); padding: 1rem 1.1rem; border-radius: 8px;}
.small {opacity:.75; font-size:.9rem;}
/* compact dataframe */
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
/* hide footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not set")
    st.stop()
genai.configure(api_key=api_key)

def load_predictions(version: str) -> pd.DataFrame:
    path = os.path.join("models", "outputs", f"btc_predictions_{version}.csv")
    if not os.path.exists(path):
        st.warning(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])

def generate_insight(prompt: str, model_id: str, temperature: float, max_tokens: int = 600) -> str:
    try:
        model = genai.GenerativeModel(
            model_id,
            generation_config={
                "temperature": float(temperature),
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": int(max_tokens),
            },
        )
        chat = model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"[Error generating insight] {e}"

def build_prompt(df: pd.DataFrame, version: str) -> str:
    recent = df.tail(10).copy()
    table_text = recent.to_string(index=False)
    return (
        "You are a senior fintech analyst. Write a concise (3–6 lines), "
        "professional insight summary focusing on direction, confidence, and risks.\n\n"
        f"Data source: {version.upper()} | Last 10 rows of BTC forecast results:\n\n"
        f"{table_text}\n\n"
        "Use neutral tone. Avoid code blocks. Do not repeat the table."
    )

st.sidebar.header("Controls")
version = st.sidebar.selectbox("Data version", ["api", "csv"], index=0)
model_id = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-1.5-pro-latest", "gemini-1.5-flash"],
    index=0,
    help="Use Pro for best quality; Flash for speed/lower quota."
)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
st.sidebar.caption("Tip: lower temperature → more deterministic, report-like output.")

st.markdown("## 📈 BTC Forecast Summary")
st.caption("Expanding-window Ridge regression with API/CSV feature variants and Gemini-generated insights.")

df = load_predictions(version)
if df.empty:
    st.stop()

last = df.tail(1).iloc[0]
r2 = float(last.get("r2", 0))
mse = float(last.get("mse", 0))
pred = float(last.get("predicted_return", 0))
act = float(last.get("actual_return", 0))
date_str = last["date"].strftime("%Y-%m-%d") if "date" in last and not pd.isna(last["date"]) else "-"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Latest Date", date_str)
with c2:
    st.metric("R² (latest)", f"{r2:.4f}")
with c3:
    st.metric("MSE (latest)", f"{mse:.4f}")
with c4:
    st.metric("Pred vs Actual (latest)", f"{pred:.4f}", delta=f"{act:.4f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Overview", "Insight", "Full data"])

with tab1:
    st.markdown("#### Recent Predictions")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("#### Insight")
    prompt = build_prompt(df, version)
    with st.spinner("Generating Gemini summary..."):
        summary = generate_insight(prompt, model_id, temperature)
    st.markdown(f'<div class="insight">{summary}</div>', unsafe_allow_html=True)
    st.download_button(
        "Download insight as .txt",
        data=summary,
        file_name=f"btc_insight_{version}.txt",
        type="secondary",
    )

with tab3:
    st.markdown("#### Full Table")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="small">Model: Ridge (expanding window). '
    'LLM: Google Gemini. Data source selectable via sidebar.</div>',
    unsafe_allow_html=True
)

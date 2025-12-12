import streamlit as st
import pandas as pd
import yfinance as yf

from news.fetcher import fetch_news_for_tickers
from sentiment.analyzer import score_headlines_df_transformer
from sentiment.aggregate import aggregate_daily_by_ticker
from genai.summarizer import generate_news_sentiment_summary
from features.pdf_report import generate_pdf_report

st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="wide")
st.title("📰 Financial News Sentiment Analyzer")

# -------------------------
# Sidebar: user controls
# -------------------------
st.sidebar.header("Fetch & Analysis Settings")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,TSLA")
days = st.sidebar.number_input("Days back to fetch news", min_value=1, max_value=30, value=3)
model_name = st.sidebar.selectbox(
    "Transformer model (smaller = faster)",
    options=[
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment",
        "ProsusAI/finbert"
    ],
    index=0,
    help="Smaller models download faster on first run. Streamlit Cloud will download model weights."
)

st.sidebar.markdown("---")
st.sidebar.header("GenAI Settings")
genai_enabled = st.sidebar.checkbox("Enable AI summaries (Groq)", value=True)
genai_model = st.sidebar.text_input("Groq model name", value="llama-3.1-8b-instant")

st.sidebar.markdown("---")
st.sidebar.header("PDF / Misc")
report_author = st.sidebar.text_input("Report author", value="Portfolio Analyzer")

# -------------------------
# Helper functions
# -------------------------
def _ensure_session_state_keys():
    keys = ["df_scored", "agg", "tickers", "ai_summary", "selected_ticker", "pred_df"]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

_ensure_session_state_keys()

# -------------------------
# Main Controls
# -------------------------
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Fetch & Analyze News"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
        else:
            try:
                with st.spinner("Fetching news..."):
                    df = fetch_news_for_tickers(tickers, days=days)
                if df.empty:
                    st.warning("No articles found for the selected tickers and timeframe.")
                else:   
                    st.success(f"Fetched {len(df)} articles.")

                    # Score using transformer
                    with st.spinner("Scoring headlines with transformer model..."):
                        df_scored = score_headlines_df_transformer(df, text_col="title", model_name=model_name, batch_size=16)

                    # Aggregate
                    agg = aggregate_daily_by_ticker(df_scored)

                    # --- AUTOMATIC TOPIC CLASSIFICATION (hidden defaults) ---
                    try:
                        with st.spinner("Detecting topics for articles..."):
                            from features.topic_classifier import classify_df_articles_auto
                            df_classified = classify_df_articles_auto(df_scored, text_col="title")
                    except Exception as e:
                        st.error(f"Topic classification failed (continuing without topics): {e}")
                        # fall back to scored only
                        df_classified = df_scored.copy()
                        df_classified["topic_label"] = None
                        df_classified["topic_score"] = None

                    # Save into session_state
                    st.session_state["df_scored"] = df_scored
                    st.session_state["agg"] = agg
                    st.session_state["tickers"] = tickers
                    st.session_state["df_classified"] = df_classified

                    # Clear previous AI summary
                    st.session_state["ai_summary"] = None
                    st.session_state["pred_df"] = None
                    # Rendering sample (replace earlier df_scored display)
                display_df = st.session_state.get("df_classified", st.session_state.get("df_scored"))
                st.dataframe(display_df[["ticker_query", "publishedAt", "title", "sentiment", "topic_label"]].head(50), width="stretch")
                
            except Exception as e:
                st.error(f"Failed to fetch/score news: {e}")

with col2:
    st.write("")  # spacer
    st.write("")  # spacer
    st.write("Tips:")
    st.write("- First click **Fetch & Analyze News**")
    st.write("- Then generate AI summary or run prediction")
    st.write("- Transformer models download weights on first run")

st.markdown("---")

# -------------------------
# Display analysis if present
# -------------------------
if st.session_state.get("df_scored") is not None:
    df_scored = st.session_state["df_scored"]
    agg = st.session_state["agg"]
    tickers = st.session_state["tickers"]

    st.subheader("Sample of Scored Articles")
    st.dataframe(df_scored[["ticker_query", "publishedAt", "title", "sentiment", "sentiment_label", "sentiment_score"]].head(50), width="stretch")

    st.subheader("Daily Aggregated Sentiment (recent)")
    st.dataframe(agg.sort_values(["ticker_query", "date"], ascending=[True, False]).head(50), width="stretch")


else:
    st.info("No analysis available yet — click **Fetch & Analyze News** to start.")
    st.stop()

st.markdown("---")

# -------------------------
# GenAI Summary (Groq)
# -------------------------
st.subheader("🧠 AI Summary (Retail-friendly)")

selected_ticker = st.selectbox("Select ticker for summary", options=st.session_state["tickers"], index=0)

if st.button("Generate AI Summary"):
    if not genai_enabled:
        st.info("GenAI is disabled in sidebar.")
    else:
        try:
            df_classified = st.session_state.get("df_classified", st.session_state["df_scored"])
            agg = st.session_state["agg"]
            recent = df_scored[df_classified["ticker_query"] == selected_ticker].sort_values("publishedAt", ascending=False).head(10)
            agg_sub = agg[agg["ticker_query"] == selected_ticker]
            agg_row = agg_sub.sort_values("date").iloc[-1].to_dict() if not agg_sub.empty else None

            with st.spinner("Generating AI summary..."):
                summary = generate_news_sentiment_summary(
                    ticker=selected_ticker,
                    recent_news=recent,
                    agg_row=agg_row,
                    model=genai_model,
                )
            st.session_state["ai_summary"] = summary
            st.session_state["selected_ticker"] = selected_ticker
            st.success("AI summary generated.")
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            st.info("Check that GROQ_API_KEY is set in Streamlit secrets and model name is valid.")

# Show latest summary if present
if st.session_state.get("ai_summary"):
    st.markdown("**Latest AI Summary**")
    st.write(st.session_state["ai_summary"])

st.markdown("---")

# -------------------------
# PDF Report Generation + Download
# -------------------------
st.subheader("📄 Generate PDF Report")

# we will use the last generated summary or generate one on-the-fly
pdf_ticker = st.selectbox("Ticker for report", options=st.session_state["tickers"], index=0, key="report_ticker")

if st.button("Generate & Download PDF Report"):
    try:
        if st.session_state.get("ai_summary") and st.session_state.get("selected_ticker") == pdf_ticker:
            summary_text = st.session_state["ai_summary"]
        else:
            # generate a quick summary synchronously (if genai enabled)
            recent = st.session_state["df_scored"][st.session_state["df_scored"]["ticker_query"] == pdf_ticker].sort_values("publishedAt", ascending=False).head(10)
            agg_sub = st.session_state["agg"][st.session_state["agg"]["ticker_query"] == pdf_ticker]
            agg_row = agg_sub.sort_values("date").iloc[-1].to_dict() if not agg_sub.empty else None
            if genai_enabled:
                with st.spinner("Generating AI summary for report..."):
                    summary_text = generate_news_sentiment_summary(
                        ticker=pdf_ticker,
                        recent_news=recent,
                        agg_row=agg_row,
                        model=genai_model,
                    )
            else:
                summary_text = "AI summaries are disabled; no summary available."

        recent_news = st.session_state["df_scored"][st.session_state["df_scored"]["ticker_query"] == pdf_ticker].sort_values("publishedAt", ascending=False).head(10)
        agg_df_for_pdf = st.session_state["agg"]

        with st.spinner("Building PDF..."):
            pdf_bytes = generate_pdf_report(
                ticker=pdf_ticker,
                ai_summary=summary_text,
                recent_news_df=recent_news,
                agg_df=agg_df_for_pdf,
                author=report_author,
            )

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"{pdf_ticker}_news_report_{pd.Timestamp.utcnow().date()}.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")

st.markdown("---")
st.info("Workflow: Fetch & Analyze → (optional) Generate AI Summary → (optional) Run Prediction → Export PDF")
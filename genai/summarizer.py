import os
from typing import Optional, Dict, Any
from groq import Groq
import pandas as pd


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    try:
        import streamlit as st
        if not api_key and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit secrets or env vars")
    return Groq(api_key=api_key)


def _build_topics_summary(recent_news_df: Optional[pd.DataFrame]) -> str:
    """
    Build a short textual summary of detected topics (top 3) from recent_news_df,
    e.g. "Top topics today: Earnings (5), Regulation (2), Product Launch (1)."
    """
    if recent_news_df is None or recent_news_df.empty or "topic_label" not in recent_news_df.columns:
        return ""
    counts = recent_news_df["topic_label"].value_counts().head(5)
    items = [f"{lbl} ({cnt})" for lbl, cnt in counts.items()]
    return "Top topics: " + ", ".join(items) + "."


def generate_news_sentiment_summary(
    ticker: str,
    recent_news,
    agg_row: Optional[Dict[str, Any]] = None,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Creates a natural-language, retail-friendly summary and includes top topics.
    recent_news: DataFrame expected to contain 'title' and optionally 'topic_label' and 'sentiment'.
    agg_row: aggregated row for the ticker (dict)
    """
    client = _get_client()

    lines = []
    lines.append(f"Ticker: {ticker}")

    if agg_row is not None:
        lines.append(
            f"Daily Avg Sentiment: {agg_row.get('mean_sentiment', 0):.2f}, "
            f"Articles Today: {agg_row.get('article_count', 0)}"
        )
        lines.append(f"Sentiment Z-Score: {agg_row.get('sentiment_z', 0):.2f}")

    # topics summary
    topics_text = _build_topics_summary(recent_news)
    if topics_text:
        lines.append(topics_text)

    lines.append("\nRecent Headlines:")
    for _, row in recent_news.iterrows():
        title = (row.get("title") or "")[:200]
        topic = f" [{row.get('topic_label')}]" if row.get("topic_label") else ""
        score = f" (score={row.get('sentiment'):.2f})" if row.get("sentiment") is not None else ""
        lines.append(f"- {title}{topic}{score}")

    stats_text = "\n".join(lines)

    system_msg = (
        "You are a friendly financial news analyst. Explain sentiment in simple English for retail investors. "
        "Avoid jargon. Focus on why sentiment changed, which topics (e.g., Earnings, Regulation) influenced it, "
        "and what a regular person should understand. Give trading advice."
    )

    user_msg = (
        "Here is the news and topic information:\n\n"
        f"{stats_text}\n\n"
        "Write a clear, retail-friendly explanation that:\n"
        "- States whether sentiment today is positive, negative, or neutral\n"
        "- Explains the most important topics driving sentiment\n"
        "- Mentions 1–2 headlines that mattered most\n"
        "- Ends with a short 'What this means for you' bullet list\n"
        "Do not say you are an AI."
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=600,
        temperature=0.4,
    )

    return completion.choices[0].message.content.strip()
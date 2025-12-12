import os
import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news_newsapi(query, from_dt, to_dt, page_size=100, api_key=None, max_pages=2):
    """
    query: search string (e.g., 'Apple OR AAPL')
    from_dt, to_dt: ISO date strings 'YYYY-MM-DD'
    """
    api_key = api_key or st.secrets["NEWSAPI_KEY"]
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set (env or streamlit secrets).")

    all_articles = []
    for page in range(1, max_pages+1):
        params = {
            "q": query,
            "from": from_dt,
            "to": to_dt,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": api_key,
        }
        r = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break
        all_articles.extend(articles)
        time.sleep(0.2)
        if len(articles) < page_size:
            break

    df = pd.DataFrame(all_articles)
    # keep minimal columns
    if not df.empty:
        df = df[["source","author","title","description","url","publishedAt","content"]]
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df["fetched_at"] = pd.Timestamp.utcnow()
    return df

# Simple wrapper for fetching across tickers
def fetch_news_for_tickers(tickers, days=2, api_key=None):
    to_dt = datetime.utcnow().date()
    from_dt = to_dt - timedelta(days=days)
    results = []
    for t in tickers:
        q = f'"{t}" OR "{t}.NS" OR "{t} stock" OR "{t} share"'
        df = fetch_news_newsapi(q, from_dt.isoformat(), to_dt.isoformat(), api_key=api_key)
        if not df.empty:
            df["ticker_query"] = t
            results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

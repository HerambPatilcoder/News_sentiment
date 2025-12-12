# src/utils/pdf_report.py
import io
from datetime import datetime
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)


def _create_sentiment_chart_bytes(agg_df: pd.DataFrame, ticker: str):
    """
    Returns PNG bytes of a sentiment time-series chart for the given ticker.
    """
    buf = io.BytesIO()

    # If agg_df is empty, create a placeholder image
    if agg_df is None or agg_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No sentiment data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Plot
    df = agg_df.copy()
    df = df.sort_values("date")
    if "date" in df.columns:
        x = pd.to_datetime(df["date"])
    else:
        x = pd.date_range(start=0, periods=len(df))
    y = df["mean_sentiment"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{ticker} - Daily Mean Sentiment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Sentiment")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _df_to_table_data(df: pd.DataFrame, max_rows: int = 10):
    """
    Convert small DataFrame to list-of-lists suitable for ReportLab Table.
    Limit rows to max_rows.
    """
    if df is None or df.empty:
        return [["No data"]]

    # Take subset
    df_sub = df.head(max_rows).copy()
    # Ensure string values
    df_sub = df_sub.fillna("").astype(str)
    header = list(df_sub.columns)
    data = [header] + df_sub.values.tolist()
    return data


def generate_pdf_report(
    ticker: str,
    ai_summary: str,
    recent_news_df: Optional[pd.DataFrame],
    agg_df: Optional[pd.DataFrame],
    author: str = "Portfolio Analyzer",
) -> bytes:
    """
    Build a PDF report and return bytes.

    Arguments:
    - ticker: symbol string
    - ai_summary: retail-friendly text from Groq
    - recent_news_df: DataFrame with columns like ['title','publishedAt','sentiment']
    - agg_df: aggregated DataFrame (date, mean_sentiment, article_count, sentiment_z)
    - prediction_df: DataFrame with model predictions (optional)
    """
    buf = io.BytesIO()
    # Using landscape A4 for better table width
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading1"]
    small_bold = ParagraphStyle("small_bold", parent=styles["Normal"], fontSize=10, leading=12, spaceAfter=6, leftIndent=0, rightIndent=0)

    elements = []

    # Header
    title_text = f"News Sentiment Report — {ticker}"
    date_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    elements.append(Paragraph(title_text, styles["Title"]))
    elements.append(Paragraph(f"Generated: {date_text}", small_bold))
    elements.append(Paragraph(f"Author: {author}", small_bold))
    elements.append(Spacer(1, 12))

    # AI Summary (wrap into paragraphs)
    elements.append(Paragraph("AI Summary (retail-friendly):", styles["Heading2"]))
    for para in ai_summary.strip().split("\n\n"):
        elements.append(Paragraph(para.strip(), normal))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))

    # Chart
    elements.append(Paragraph("Sentiment Trend", styles["Heading2"]))
    chart_bytes = _create_sentiment_chart_bytes(agg_df[agg_df["ticker_query"] == ticker] if agg_df is not None else None, ticker)
    chart_buf = io.BytesIO(chart_bytes)
    img = RLImage(chart_buf, width=18*cm, height=6*cm)
    elements.append(img)
    elements.append(Spacer(1, 12))

    # Recent headlines table
    elements.append(Paragraph("Recent Headlines (top results):", styles["Heading2"]))
    if recent_news_df is None or recent_news_df.empty:
        elements.append(Paragraph("No recent headlines found.", normal))
    else:
        # keep only relevant columns and short titles
        df_head = recent_news_df.copy()
        keep_cols = []
        if "publishedAt" in df_head.columns:
            df_head["publishedAt"] = pd.to_datetime(df_head["publishedAt"]).dt.strftime("%Y-%m-%d %H:%M")
            keep_cols.append("publishedAt")
        if "title" in df_head.columns:
            df_head["title"] = df_head["title"].str[:240]
            keep_cols.append("title")
        if "sentiment" in df_head.columns:
            df_head["sentiment"] = df_head["sentiment"].astype(float).map("{:.3f}".format)
            keep_cols.append("sentiment")
        if "url" in df_head.columns:
            df_head["url"] = df_head["url"].astype(str)
            # Do not include url by default unless space allows

        table_data = _df_to_table_data(df_head[keep_cols], max_rows=10)
        t = Table(table_data, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(t)

    elements.append(Spacer(1, 12))

    # Aggregated sentiment table (small)
    elements.append(Paragraph("Aggregated Daily Sentiment (recent):", styles["Heading2"]))
    if agg_df is None or agg_df.empty:
        elements.append(Paragraph("No aggregated sentiment data.", normal))
    else:
        # select rows for the ticker and format
        agg_sub = agg_df[agg_df["ticker_query"] == ticker].sort_values("date", ascending=False).head(10)
        if not agg_sub.empty:
            # pick useful cols
            display_df = agg_sub[["date", "article_count", "mean_sentiment", "sentiment_z"]].copy()
            display_df["date"] = display_df["date"].astype(str)
            display_df["mean_sentiment"] = display_df["mean_sentiment"].map("{:.3f}".format)
            display_df["sentiment_z"] = display_df["sentiment_z"].map("{:.3f}".format)
            table_data = _df_to_table_data(display_df, max_rows=10)
            t2 = Table(table_data, repeatRows=1, hAlign="LEFT")
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
            elements.append(t2)
        else:
            elements.append(Paragraph("No recent aggregated rows for this ticker.", normal))

    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
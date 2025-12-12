# src/sentiment/analyzer.py
"""
Transformer-based headline/article sentiment scoring.

This module uses Hugging Face's transformers.pipeline("sentiment-analysis").
Default model: "distilbert-base-uncased-finetuned-sst-2-english" (fast & small).
You can change model_name when calling score_headlines_df_transformer.
"""

from typing import Optional, List
import pandas as pd

# Lazy imports to avoid heavy import costs at module import time
_transformer_pipeline = None
_pipeline_model_name = None

def _init_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    global _transformer_pipeline, _pipeline_model_name
    if _transformer_pipeline is not None and _pipeline_model_name == model_name:
        return _transformer_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        raise RuntimeError(
            "transformers not installed or failed to import. "
            "Install with `pip install transformers torch`."
        ) from e

    # Initialize pipeline (CPU). Streamlit Cloud will download weights on first run.
    _transformer_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        device=-1,  # CPU
    )
    _pipeline_model_name = model_name
    return _transformer_pipeline


def _label_score_to_signed(label: str, score: float) -> float:
    """
    Map pipeline label+score to signed float in [-1, 1].
    Common outputs: "POSITIVE", "NEGATIVE".
    """
    lab = label.upper()
    if "POS" in lab:
        return float(score)  # positive in (0,1]
    if "NEG" in lab:
        return -float(score)  # negative in [-1,0)
    # fallback (neutral / unknown)
    return 0.0


def score_headlines_df_transformer(
    df: pd.DataFrame,
    text_col: str = "title",
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Scores headlines/articles using a transformer sentiment pipeline.

    Returns a copy of the DataFrame with an added 'sentiment' column (float in [-1,1])
    and optional 'sentiment_label' and 'sentiment_score' columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a text column to score.
    text_col : str
        Column name containing text to score (title or content).
    model_name : str
        Hugging Face model identifier; default is a small SST-2 distilled model.
    batch_size : int
        Batch size for pipeline; tune for memory/performance.

    Raises
    ------
    RuntimeError if transformers not installed or pipeline fails to initialize.
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame.")

    df = df.copy().reset_index(drop=True)
    texts = df[text_col].fillna("").astype(str).tolist()

    pipe = _init_pipeline(model_name=model_name)

    results = []
    # run in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # pipeline returns list of dicts like {'label':'POSITIVE','score':0.98}
        outs = pipe(batch, truncation=True)
        for out in outs:
            label = out.get("label", "")
            score = out.get("score", 0.0)
            signed = _label_score_to_signed(label, score)
            results.append({"sentiment": signed, "sentiment_label": label, "sentiment_score": float(score)})

    # pad results if needed (shouldn't be necessary)
    while len(results) < len(texts):
        results.append({"sentiment": 0.0, "sentiment_label": "NEUTRAL", "sentiment_score": 0.0})

    res_df = pd.DataFrame(results)
    out = pd.concat([df, res_df], axis=1)
    return out
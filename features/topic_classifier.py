from typing import List, Optional
import pandas as pd


_pipeline = None
_pipeline_model_name = None
    

def _get_pipeline(model_name: str = "facebook/bart-large-mnli"):
    global _pipeline, _pipeline_model_name
    if _pipeline is not None and _pipeline_model_name == model_name:
        return _pipeline
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError(
            "transformers is not installed or failed to import. "
            "Install transformers (and torch) to use topic classification."
        ) from e
    _pipeline = pipeline("zero-shot-classification", model=model_name, device=-1)
    _pipeline_model_name = model_name
    return _pipeline


def _classify_texts(pipe, texts: List[str], candidate_labels: List[str], batch_size: int = 8, hypothesis_template: str = "This text is about {}."):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outs = pipe(batch, candidate_labels, hypothesis_template=hypothesis_template)
        if isinstance(outs, dict):
            outs = [outs]
        for out in outs:
            labels = out.get("labels", [])
            scores = out.get("scores", [])
            if labels:
                results.append({"label": labels[0], "score": float(scores[0])})
            else:
                results.append({"label": "Other", "score": 0.0})
    return results


def classify_df_articles_auto(
    df: pd.DataFrame,
    text_col: str = "title",
    model_name: str = "facebook/bart-large-mnli",
    candidate_labels: Optional[List[str]] = None,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Automatically classify articles in df and return a copy with
    'topic_label' and 'topic_score' columns added.
    Uses reasonable default labels and model; no UI needed.
    """
    if candidate_labels is None:
        candidate_labels = [
            "Earnings",
            "Product Launch",
            "Regulation",
            "Mergers & Acquisitions",
            "Partnership",
            "Analyst Rating/Target",
            "Financial Results",
            "Legal / Lawsuit",
            "Macro Economy",
            "Other",
        ]

    if text_col not in df.columns:
        raise ValueError(f"{text_col} not found in DataFrame")

    df = df.copy().reset_index(drop=True)
    texts = df[text_col].fillna("").astype(str).tolist()
    if len(texts) == 0:
        df["topic_label"] = []
        df["topic_score"] = []
        return df
        

    pipe = _get_pipeline(model_name=model_name)
    preds = _classify_texts(pipe, texts, candidate_labels, batch_size=batch_size)
    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]

            
    df["topic_label"] = labels
    df["topic_score"] = scores
    return df

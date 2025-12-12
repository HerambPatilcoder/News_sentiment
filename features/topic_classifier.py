from typing import List, Optional, Dict
import pandas as pd
import hashlib

_pipeline = None
_pipeline_model_name = None

def _hash_text(s: str) -> str:
    import hashlib as _h
    return _h.md5(s.encode("utf-8")).hexdigest()

def _keyword_label(text: str, candidate_labels: List[str]) -> Dict[str, float]:
    """
    Fast fallback: naive keyword matching. Returns best label and a low confidence.
    """
    txt = text.lower()
    scores = {}
    for lbl in candidate_labels:
        keywords = lbl.lower().split()  # simplistic mapping
        match_count = sum(1 for k in keywords if k in txt)
        scores[lbl] = match_count
    # choose label with highest matches (or Other)
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return {"label": "Other", "score": 0.0}
    # convert count->score (simple)
    return {"label": best[0], "score": float(best[1]) / max(1, len(txt.split()))}

def _get_pipeline(model_name: str = "typeform/distilbert-base-uncased-mnli"):
    """
    Safe pipeline initializer:
    - forces CPU
    - uses smaller default model
    - returns None if pipeline can't be created (caller will fallback)
    """
    global _pipeline, _pipeline_model_name
    if _pipeline is not None and _pipeline_model_name == model_name:
        return _pipeline
    try:
        from transformers import pipeline
        # force CPU device
        pipe = pipeline("zero-shot-classification", model=model_name, device=-1)
        _pipeline = pipe
        _pipeline_model_name = model_name
        return _pipeline
    except Exception as e:
        # common meta/tensor issues land here; return None so caller falls back
        print("Warning: zero-shot pipeline initialization failed:", str(e))
        _pipeline = None
        _pipeline_model_name = None
        return None

def _classify_batch_safe(pipe, texts: List[str], candidate_labels: List[str], batch_size: int = 32):
    results = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outs = pipe(batch, candidate_labels, hypothesis_template="This text is about {}.")
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
    except Exception as e:
        # pipeline failed during inference (e.g. meta tensor)
        print("Warning: pipeline failed during inference:", str(e))
        return None

def classify_df_articles_auto(
    df: pd.DataFrame,
    text_col: str = "title",
    model_name: str = "typeform/distilbert-base-uncased-mnli",
    candidate_labels: Optional[List[str]] = None,
    batch_size: int = 32,
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    Safe classifier: tries HF zero-shot with a small model; if that fails falls back to keyword matching.
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
        df["topic_label"] = None
        df["topic_score"] = None
        return df

    # optional dedupe
    if dedupe:
        hash_to_indices: Dict[str, List[int]] = {}
        uniq_texts: List[str] = []
        for idx, txt in enumerate(texts):
            h = _hash_text(txt)
            if h not in hash_to_indices:
                hash_to_indices[h] = [idx]
                uniq_texts.append(txt)
            else:
                hash_to_indices[h].append(idx)
    else:
        uniq_texts = texts
        hash_to_indices = { _hash_text(t): [i] for i, t in enumerate(texts) }

    # try HF pipeline
    pipe = _get_pipeline(model_name=model_name)
    results = None
    if pipe is not None:
        preds = _classify_batch_safe(pipe, uniq_texts, candidate_labels, batch_size=batch_size)
        results = preds

    # If HF failed, use keyword fallback
    if results is None:
        print("Using keyword fallback for topic classification (fast).")
        results = []
        for t in uniq_texts:
            res = _keyword_label(t, candidate_labels)
            results.append(res)

    # map back
    labels = [None] * len(texts)
    scores = [None] * len(texts)
    for uniq_idx, pred in enumerate(results):
        uniq_text = uniq_texts[uniq_idx]
        h = _hash_text(uniq_text)
        indices = hash_to_indices.get(h, [])
        for original_idx in indices:
            labels[original_idx] = pred["label"]
            scores[original_idx] = pred["score"]

    df["topic_label"] = labels
    df["topic_score"] = scores
    return df

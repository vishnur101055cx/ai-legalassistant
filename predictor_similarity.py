# predictor_similarity.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, re

EMBED_FILE = "case_embeddings.npy"
META_FILE = "case_metadata.pkl"
MODEL_NAME_FILE = "embed_model_name.pkl"

_model = None
_embeddings = None
_meta = None

def _load_resources():
    global _model, _embeddings, _meta
    if _model is None:
        model_name = joblib.load(MODEL_NAME_FILE)
        _model = SentenceTransformer(model_name)
        _embeddings = np.load(EMBED_FILE)
        _meta = pd.read_pickle(META_FILE)
    return _model, _embeddings, _meta

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_similar_cases(doc_text: str, top_k: int = 5):
    model, embeddings, meta = _load_resources()
    q_emb = model.encode([clean_text(doc_text)], convert_to_numpy=True)[0]
    sims = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]
    idx = sims.argsort()[-top_k:][::-1]
    results = []
    for i in idx:
        row = meta.iloc[i]
        results.append({
            "case_id": int(row["case_id"]) if "case_id" in row.index else int(i),
            "year": int(row["year"]) if "year" in row.index and pd.notna(row["year"]) else None,
            "text": row["text"],
            "outcome": row["outcome"],
            "similarity": float(sims[i])
        })
    return results

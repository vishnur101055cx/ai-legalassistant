# build_index.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_EMBED_FILE = "case_embeddings.npy"
OUT_META = "case_metadata.pkl"
OUT_MODEL_NAME = "embed_model_name.pkl"

def load_local_csv(path="legal_cases.csv"):
    df = pd.read_csv(path)
    if "text" not in df.columns or "outcome" not in df.columns:
        raise ValueError("CSV must have 'text' and 'outcome' columns.")
    df["text"] = df["text"].astype(str)
    return df

def main():
    print("Loading local CSV: legal_cases.csv")
    df = load_local_csv("legal_cases.csv")
    print("Num cases:", len(df))

    model = SentenceTransformer(EMBED_MODEL)
    texts = df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)

    np.save(OUT_EMBED_FILE, embeddings)
    df.to_pickle(OUT_META)
    joblib.dump(EMBED_MODEL, OUT_MODEL_NAME)

    print("Saved embeddings ->", OUT_EMBED_FILE)
    print("Saved metadata ->", OUT_META)

if __name__ == "__main__":
    main()

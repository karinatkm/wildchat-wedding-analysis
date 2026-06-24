"""
Phase 2a — Embed candidates ONCE and cache to disk.

Embedding is the CPU bottleneck (~13 min). Caching normalized embeddings lets us
iterate on prototype design (phase2_classify.py) in seconds.

Inputs : data/wedding_candidates.parquet
Outputs: data/cand_embeddings.npy  (float32, L2-normalized, row-aligned to parquet)

Run:  .venv/bin/python scripts/phase2_embed.py
"""
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(8)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CANDIDATES = os.path.join(DATA_DIR, "wedding_candidates.parquet")
OUT = os.path.join(DATA_DIR, "cand_embeddings.npy")
SBERT_MODEL = "all-MiniLM-L6-v2"


def main():
    df = pd.read_parquet(CANDIDATES)
    texts = df["user_text"].fillna("").tolist()
    print(f"Embedding {len(texts):,} candidates ({SBERT_MODEL})...")
    sbert = SentenceTransformer(SBERT_MODEL)
    embs = np.zeros((len(texts), 384), dtype=np.float32)
    BATCH = 128
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        e = sbert.encode(batch, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        embs[start:start + len(batch)] = e
        if start % (BATCH * 8) == 0:
            print(f"  {min(start + BATCH, len(texts)):,}/{len(texts):,}")
    np.save(OUT, embs)
    print(f"Saved {OUT}  shape={embs.shape}")


if __name__ == "__main__":
    main()

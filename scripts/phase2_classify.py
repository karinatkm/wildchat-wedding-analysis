"""
Phase 2b — Fast contrastive classification from cached embeddings.

Loads data/cand_embeddings.npy and classifies each candidate by nearest prototype
among competing categories. Adds travel + relationship prototypes to route the
common honeymoon-travel and divorce/anniversary false positives away from wedding.

Final wedding rule: nearest prototype == wedding_planning
                    AND sim_wedding >= MIN_WEDDING_SIM
                    AND has a STRONG lexical term (strict).

Inputs : data/wedding_candidates.parquet, data/cand_embeddings.npy
Outputs: data/wedding_scored.parquet (+ data/phase2_summary.json)

Run:  .venv/bin/python scripts/phase2_classify.py
"""
import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CANDIDATES = os.path.join(DATA_DIR, "wedding_candidates.parquet")
EMB = os.path.join(DATA_DIR, "cand_embeddings.npy")
OUT = os.path.join(DATA_DIR, "wedding_scored.parquet")
SBERT_MODEL = "all-MiniLM-L6-v2"
MIN_WEDDING_SIM = 0.30

PROTOTYPES = {
    "wedding_planning": [
        "Help me plan my wedding, timeline, and checklist.",
        "Write wedding vows for my partner.",
        "Write a best man speech or maid of honor toast for the wedding reception.",
        "How should I budget for our wedding and what does a wedding cost?",
        "Ideas for our wedding venue, ceremony, and reception decorations.",
        "Help me choose a wedding dress and wedding attire.",
        "Wedding invitation wording, RSVP reply, and save the date message.",
        "Wedding etiquette: seating chart, guest list, what to say at a wedding.",
        "Ideas for the bridal shower and bachelorette party.",
        "How to propose and plan our engagement; engagement message.",
        "Wedding cake ideas and flavors.",
        "Write an officiant script and father of the bride speech for a wedding ceremony.",
        "Wedding blog article and captions about planning a wedding.",
        "What should I reply to a wedding invitation from my friend.",
    ],
    "ecommerce_listing": [
        "1. Pendant Necklace For Women Fashion Gift 2. Bracelet Beads Bangle 3. Ring Set Jewelry",
        "Translate and optimize these product titles for my online store.",
        "Earrings, brooch, hair clips, scarf, decorative tape, gift tags product listing.",
        "Rewrite these Amazon product titles with keywords: necklace, ring, bracelet, set, pcs.",
    ],
    "fiction_roleplay": [
        "Write a story, fanfic, or crossover involving these characters.",
        "Continue this roleplay dialogue between the characters; there would be dialogue in this story.",
        "Write a script for a film or TV series with engaging dialogue.",
        "Summarize this chapter of the novel keeping the tone and details.",
        "Write a character backstory and lore for this fictional universe.",
        "Make a story of two characters getting married and living happily.",
        "In the school clubroom, MC says to Sayori my beloved wife; continue the Doki Doki Literature Club roleplay with Natsuki and Monika.",
        "[player]: dialogue with a female ghast in the Minecraft Nether; continue the roleplay scene.",
        "Roleplay scene: the vampire commander confesses feelings at the altar at night; continue the story.",
    ],
    "travel_guide": [
        "Write a travel guide article: best time to visit this destination for a honeymoon.",
        "Things to do and places to visit on a trip; tourist itinerary for my vacation.",
        "Write a constructive detailed guide for my travel guide book about this country.",
    ],
    "relationship_general": [
        "How do I end my marriage or get a divorce without hurting my spouse.",
        "Relationship advice for my wife and our anniversary after years of marriage.",
        "General questions about historical figures who married, or what marriage means.",
    ],
    "generic_task": [
        "Summarize the following text and preserve important details.",
        "Rewrite this email in a professional way to make it shorter.",
        "Write a business plan for a restaurant and bakery.",
        "Help me with this code or programming problem.",
        "Write an essay about a historical or cultural topic.",
        "As a prompt generator for Midjourney, create image prompts for a concept.",
    ],
}


def main():
    df = pd.read_parquet(CANDIDATES)
    embs = np.load(EMB)
    assert len(df) == len(embs), "row mismatch between parquet and embeddings"

    sbert = SentenceTransformer(SBERT_MODEL)
    names = list(PROTOTYPES.keys())
    cents = []
    for nm in names:
        e = sbert.encode(PROTOTYPES[nm], convert_to_numpy=True, normalize_embeddings=True)
        c = e.mean(axis=0)
        c = c / np.linalg.norm(c)
        cents.append(c)
    cents = np.stack(cents)  # [P, D]

    sims = embs @ cents.T  # cosine (both normalized) [N, P]
    for i, nm in enumerate(names):
        df[f"sim_{nm}"] = sims[:, i]
    df["sbert_sim"] = df["sim_wedding_planning"]
    best = sims.argmax(axis=1)
    df["nearest_proto"] = [names[i] for i in best]
    df["nearest_sim"] = sims.max(axis=1)

    df["is_wedding"] = (
        (df["nearest_proto"] == "wedding_planning")
        & (df["sim_wedding_planning"] >= MIN_WEDDING_SIM)
        & (df["strict"])
    )
    df.to_parquet(OUT, index=False)

    n_w = int(df["is_wedding"].sum())
    summary = {
        "candidates": len(df),
        "min_wedding_sim": MIN_WEDDING_SIM,
        "require_strict": True,
        "wedding_relevant": n_w,
        "nearest_proto_counts": df["nearest_proto"].value_counts().to_dict(),
        "model": SBERT_MODEL,
        "output": OUT,
    }
    with open(os.path.join(DATA_DIR, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("PHASE 2b (cached contrastive) COMPLETE")
    print("=" * 60)
    print(f"Nearest-proto counts: {summary['nearest_proto_counts']}")
    print(f"Wedding-relevant:     {n_w:,}  (nearest=wedding, sim>={MIN_WEDDING_SIM}, strict)")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()

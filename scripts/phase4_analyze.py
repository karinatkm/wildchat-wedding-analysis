"""
Phase 4 — Wedding subset analysis: sub-topics, intents, engagement, temporal.

Loads the validated wedding subset (is_wedding) and produces:
  - TF-IDF + KMeans sub-topic clusters with top terms
  - Rule-based intent taxonomy + frequencies
  - Engagement (turns) by intent
  - Temporal volume
  - Figures in figures/ and a machine-readable analysis_results.json

Run:  .venv/bin/python scripts/phase4_analyze.py
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
SCORED = os.path.join(DATA_DIR, "wedding_scored.parquet")

# Domain stopwords (too generic to be informative sub-topic terms)
EXTRA_STOP = {
    "wedding", "weddings", "write", "please", "chatgpt", "want", "need", "help",
    "make", "like", "good", "use", "using", "give", "list", "10", "20", "im",
    "ill", "id", "dont", "thats", "got", "just", "can", "could", "would", "also",
    "one", "get", "going", "day", "really", "things", "thing", "etc",
}

# ---------------------------------------------------------------------------
# Intent taxonomy (rule-based, ordered by specificity)
# ---------------------------------------------------------------------------
INTENT_RULES = [
    ("vows_speeches", r"\b(vow|vows|speech|speeches|toast|officiant|address|father of the bride|best man|maid of honor)\b"),
    ("invitations_messaging", r"\b(invitation|invite|rsvp|save the date|thank ?you|message|caption|card|wording|reply)\b"),
    ("budget_cost", r"\b(budget|cost|costs|afford|cheap|expensive|price|pricing|money|\$|rs\.?|dollars?)\b"),
    ("attire_beauty", r"\b(dress|gown|attire|suit|tux|makeup|hair|hairstyle|outfit|what to wear|fit into)\b"),
    ("planning_logistics", r"\b(plan|planner|checklist|timeline|schedule|organi[sz]e|guest list|seating|venue|date|coordinat)\b"),
    ("etiquette_advice", r"\b(etiquette|should i|is it ok|appropriate|rude|tradition|custom|advice|nervous|how do i)\b"),
    ("design_visual", r"\b(midjourney|image|photo|design|theme|colou?r|decor|decoration|prompt|logo|flower|cake)\b"),
    ("content_article", r"\b(article|blog|essay|write .* about|seo|paragraph|story about (our|my) wedding)\b"),
    ("recommendation", r"\b(best|recommend|recommendation|idea|ideas|suggest|suggestion|songs?|playlist|top \d+)\b"),
]


def classify_intent(text):
    t = (text or "").lower()
    for name, pat in INTENT_RULES:
        if re.search(pat, t):
            return name
    return "other"


def top_terms_per_cluster(tfidf, labels, feature_names, k, topn=12):
    out = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            out[c] = []
            continue
        centroid = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top = centroid.argsort()[::-1][:topn]
        out[c] = [feature_names[i] for i in top]
    return out


def main():
    df = pd.read_parquet(SCORED)
    w = df[df["is_wedding"]].copy().reset_index(drop=True)
    print(f"Wedding subset: {len(w):,} conversations")

    results = {"n_wedding": int(len(w))}

    # ---- Intent classification ----
    w["intent"] = w["user_text"].apply(classify_intent)
    intent_counts = w["intent"].value_counts()
    results["intent_counts"] = intent_counts.to_dict()
    print("Intents:\n", intent_counts)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=intent_counts.values, y=intent_counts.index, color="#7b2d8e")
    plt.title(f"Wedding conversations by intent (n={len(w)})")
    plt.xlabel("conversations"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "intents.png"), dpi=130); plt.close()

    # ---- Sub-topic clustering (TF-IDF + KMeans) ----
    K = 6
    # sklearn 'english' stopwords; domain stopwords (EXTRA_STOP) removed post-hoc from cluster terms
    vec = TfidfVectorizer(stop_words="english", max_df=0.6, min_df=3,
                          ngram_range=(1, 2), token_pattern=r"[a-zA-Z][a-zA-Z]+")
    X = vec.fit_transform(w["user_text"].fillna(""))
    # drop extra stopwords post-hoc by zeroing columns
    feats = np.array(vec.get_feature_names_out())
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    w["cluster"] = labels
    cluster_terms = top_terms_per_cluster(X, labels, feats, K)
    cluster_terms = {c: [t for t in terms if t not in EXTRA_STOP][:10] for c, terms in cluster_terms.items()}
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    results["cluster_terms"] = {int(c): t for c, t in cluster_terms.items()}
    results["cluster_sizes"] = {int(c): int(n) for c, n in cluster_sizes.items()}
    print("\nClusters:")
    for c in range(K):
        print(f"  [{c}] n={cluster_sizes.get(c,0)}: {', '.join(cluster_terms[c][:8])}")

    # ---- Engagement: turns by intent ----
    results["turns_overall"] = {
        "mean": round(float(w["turn"].mean()), 2),
        "median": float(w["turn"].median()),
        "p90": float(w["turn"].quantile(0.9)),
    }
    turns_by_intent = w.groupby("intent")["turn"].mean().sort_values(ascending=False)
    results["turns_by_intent"] = turns_by_intent.round(2).to_dict()
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=turns_by_intent.values, y=turns_by_intent.index, color="#2d8e7b")
    plt.title("Avg conversation turns by intent"); plt.xlabel("avg turns"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "turns_by_intent.png"), dpi=130); plt.close()

    # ---- Temporal volume ----
    w["ts"] = pd.to_datetime(w["timestamp"], errors="coerce", utc=True)
    by_day = w.set_index("ts").resample("D").size()
    results["date_range"] = {"min": str(w["ts"].min()), "max": str(w["ts"].max())}
    plt.figure(figsize=(9, 3.8))
    by_day.plot(color="#8e5a2d")
    plt.title("Wedding conversation volume over time"); plt.ylabel("conversations/day"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "temporal.png"), dpi=130); plt.close()

    # ---- Top matched terms ----
    term_counter = Counter()
    for s in w["matched_terms"].dropna():
        for t in str(s).split(";"):
            term_counter[t] += 1
    results["top_matched_terms"] = dict(term_counter.most_common(20))

    with open(os.path.join(DATA_DIR, "analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    w.to_parquet(os.path.join(DATA_DIR, "wedding_final.parquet"), index=False)
    print(f"\nWrote figures/*.png, data/analysis_results.json, data/wedding_final.parquet")


if __name__ == "__main__":
    main()

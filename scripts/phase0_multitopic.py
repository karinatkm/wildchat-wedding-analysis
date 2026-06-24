"""
Phase 0 (multi-topic) — feasibility scan across brand-relevant industries.

Streams allenai/WildChat (English only) once and measures, per topic, the
Stage-1 STRICT lexical prevalence (>=1 strong/unambiguous term in user text).
Reports per-topic match counts, rates, projected full-English volume, and a few
example prompts so we can pick a domain with a large-enough corpus.

Run:  .venv/bin/python scripts/phase0_multitopic.py
"""
import json
import re
import os
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_ENGLISH = 80_000      # English convs to profile
MAX_STREAM = 170_000         # cap raw records streamed (English ~51%)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

FULL_CONVS = 650_000
ENGLISH_SHARE = 0.53
FULL_ENGLISH = FULL_CONVS * ENGLISH_SHARE

# ---------------------------------------------------------------------------
# Strong (precision-oriented, mostly multi-word) lexicons per topic.
# A conversation counts for a topic if ANY strong term appears in user text.
# ---------------------------------------------------------------------------
TOPICS = {
    "wedding": [
        "wedding", "weddings", "getting married", "nuptials", "elopement",
        "bride", "bridal", "groom", "fiance", "fiancee", "fiancé", "fiancée",
        "bridesmaid", "groomsman", "maid of honor", "best man", "officiant",
        "wedding planner", "rehearsal dinner", "bachelorette", "bridal shower",
        "honeymoon", "wedding vows", "wedding dress", "wedding cake",
        "wedding speech", "wedding registry", "wedding invitation",
    ],
    "health_fitness": [
        "workout", "workout plan", "exercise routine", "weight loss", "lose weight",
        "diet plan", "meal plan", "nutrition", "calorie", "calories", "protein intake",
        "muscle gain", "bodybuilding", "cardio", "gym", "yoga", "weight training",
        "intermittent fasting", "body fat",
    ],
    "mental_health": [
        "mental health", "anxiety", "depression", "panic attack", "therapy",
        "therapist", "self-esteem", "stress management", "burnout", "mindfulness",
        "coping mechanism", "emotional support",
    ],
    "finance_investing": [
        "investing", "investment", "stock market", "stocks", "portfolio",
        "cryptocurrency", "bitcoin", "ethereum", "mortgage", "budgeting", "budget plan",
        "retirement", "401k", "savings account", "dividend", "financial plan",
        "personal finance", "interest rate", "compound interest",
    ],
    "travel": [
        "itinerary", "travel guide", "vacation", "tourist", "sightseeing",
        "things to do in", "backpacking", "visa application", "road trip", "cruise",
        "flight booking", "hotel recommendation", "travel plan", "places to visit",
    ],
    "beauty_skincare": [
        "skincare", "skin care", "makeup", "moisturizer", "serum", "acne",
        "wrinkles", "cosmetics", "lipstick", "mascara", "haircare", "anti-aging",
        "skincare routine", "foundation makeup",
    ],
    "food_cooking": [
        "recipe", "cooking", "baking", "meal prep", "marinade", "homemade",
        "ingredients for", "how to cook", "how to bake", "dessert recipe", "dinner recipe",
    ],
    "real_estate": [
        "real estate", "buying a house", "home buying", "apartment for rent",
        "landlord", "tenant", "lease agreement", "realtor", "property investment",
        "house hunting", "down payment",
    ],
    "career_job": [
        "resume", "cover letter", "job interview", "job application", "linkedin profile",
        "career advice", "salary negotiation", "job search", "curriculum vitae",
        "interview questions", "professional summary",
    ],
    "marketing": [
        "marketing strategy", "advertising", "seo", "social media marketing", "ad copy",
        "email campaign", "content marketing", "google ads", "facebook ads",
        "marketing plan", "brand strategy", "landing page", "copywriting",
    ],
    "fashion": [
        "outfit", "fashion", "wardrobe", "style guide", "dress code", "apparel",
        "what to wear", "fashion trends", "capsule wardrobe", "streetwear",
    ],
    "parenting": [
        "parenting", "toddler", "newborn", "pregnancy", "breastfeeding", "diaper",
        "potty training", "baby sleep", "child development", "raising a child",
    ],
    "automotive": [
        "car repair", "car insurance", "used car", "buying a car", "engine problem",
        "dealership", "vehicle maintenance", "car model", "tire", "automotive",
    ],
}

# Precompile a regex per topic (word boundaries; literal phrases)
TOPIC_PATTERNS = {
    name: re.compile(r"\b(" + "|".join(re.escape(s) for s in terms) + r")\b", re.IGNORECASE)
    for name, terms in TOPICS.items()
}


def user_text(conversation):
    parts = [t.get("content", "") for t in conversation if t.get("role") == "user"]
    return "\n".join(p for p in parts if p)


def main():
    print(f"Streaming WildChat (cap={MAX_STREAM}, target English={TARGET_ENGLISH})...")
    ds = load_dataset("allenai/WildChat", split="train", streaming=True)

    n_seen = 0
    n_english = 0
    topic_counts = Counter()
    topic_terms = defaultdict(Counter)
    topic_examples = defaultdict(list)

    for rec in ds:
        n_seen += 1
        if rec.get("language") != "English":
            if n_seen >= MAX_STREAM:
                break
            continue
        n_english += 1
        utext = user_text(rec.get("conversation", []))
        if not utext:
            if n_english >= TARGET_ENGLISH or n_seen >= MAX_STREAM:
                break
            continue
        for name, pat in TOPIC_PATTERNS.items():
            hits = pat.findall(utext)
            if hits:
                topic_counts[name] += 1
                for h in hits:
                    topic_terms[name][h.lower()] += 1
                if len(topic_examples[name]) < 12:
                    topic_examples[name].append(utext[:200].replace("\n", " "))
        if n_english >= TARGET_ENGLISH or n_seen >= MAX_STREAM:
            break

    # Build results
    rows = []
    for name in TOPICS:
        c = topic_counts[name]
        rate = c / n_english if n_english else 0
        rows.append({
            "topic": name,
            "matches": c,
            "rate_pct": round(rate * 100, 3),
            "proj_full_english": int(rate * FULL_ENGLISH),
            "top_terms": dict(topic_terms[name].most_common(8)),
        })
    rows.sort(key=lambda r: r["matches"], reverse=True)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records_streamed": n_seen,
        "english_profiled": n_english,
        "note": "STRICT lexical (>=1 strong term). Raw recall; precision refined in Stage 2 (~50-70%).",
        "topics": rows,
    }
    with open(os.path.join(DATA_DIR, "phase0_multitopic_summary.json"), "w") as f:
        json.dump({**summary, "examples": {k: v for k, v in topic_examples.items()}}, f, indent=2)

    # Console report
    print("\n" + "=" * 78)
    print(f"MULTI-TOPIC FEASIBILITY  (English profiled: {n_english:,} of {n_seen:,} streamed)")
    print("=" * 78)
    print(f"{'topic':<20}{'matches':>9}{'rate%':>9}{'proj_full_EN':>15}   top terms")
    print("-" * 78)
    for r in rows:
        tt = ", ".join(list(r["top_terms"].keys())[:4])
        print(f"{r['topic']:<20}{r['matches']:>9,}{r['rate_pct']:>9}{r['proj_full_english']:>15,}   {tt}")
    print("=" * 78)
    print("Wrote: data/phase0_multitopic_summary.json")
    print("\nReminder: proj_full_EN is raw lexical recall; multiply by ~0.5-0.7 for")
    print("true-positive corpus size after Stage 2 semantic verification.")


if __name__ == "__main__":
    main()

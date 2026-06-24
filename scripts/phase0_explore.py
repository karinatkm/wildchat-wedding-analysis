"""
Phase 0 — WildChat exploration (English-only) + Stage-1 wedding-volume estimate.

Streams allenai/WildChat via HuggingFace (no full download), inspects schema,
profiles an English-only sample, and estimates wedding-topic prevalence using the
Stage-1 lexical filter. Writes a JSON summary + sample CSVs to data/.

Run:  .venv/bin/python scripts/phase0_explore.py
"""
import json
import re
import os
from collections import Counter
from datetime import datetime

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 30_000        # English convs to profile in Phase 0
MAX_STREAM = 120_000        # cap raw records streamed (English ~53% -> ~30k)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage 1 lexical seed lexicon (grouped, auditable)
# ---------------------------------------------------------------------------
# STRONG = unambiguous wedding terms; a single hit qualifies a conversation.
STRONG_SEEDS = [
    "wedding", "weddings", "getting married", "nuptials", "elopement", "elope",
    "bride", "bridal", "groom", "fiance", "fiancee", "fiancé", "fiancée",
    "bridesmaid", "bridesmaids", "groomsman", "groomsmen", "maid of honor",
    "best man", "officiant", "wedding planner", "rehearsal dinner",
    "bachelorette", "bridal shower", "honeymoon", "wedding vows",
    "save the date", "wedding dress", "bridal gown", "wedding cake",
    "wedding speech", "wedding toast", "seating chart", "wedding registry",
    "wedding invitation",
]
# WEAK = polysemous terms; only count when co-occurring with a STRONG term.
WEAK_SEEDS = [
    "engagement", "engaged", "marriage", "married", "betrothed",
    "ceremony", "reception", "registry", "vows", "bouquet", "rsvp",
]
ALL_SEEDS = sorted(set(STRONG_SEEDS) | set(WEAK_SEEDS))

_strong_pat = re.compile(r"\b(" + "|".join(re.escape(s) for s in STRONG_SEEDS) + r")\b", re.IGNORECASE)
_weak_pat = re.compile(r"\b(" + "|".join(re.escape(s) for s in WEAK_SEEDS) + r")\b", re.IGNORECASE)
_pattern = re.compile(r"\b(" + "|".join(re.escape(s) for s in ALL_SEEDS) + r")\b", re.IGNORECASE)


def user_text(conversation):
    """Concatenate user turns only."""
    parts = [t.get("content", "") for t in conversation if t.get("role") == "user"]
    return "\n".join(p for p in parts if p)


def lexical_hits(text):
    return _pattern.findall(text or "")


# ---------------------------------------------------------------------------
# Stream + profile
# ---------------------------------------------------------------------------
def main():
    print(f"Streaming allenai/WildChat (cap={MAX_STREAM}, target English sample={SAMPLE_SIZE})...")
    ds = load_dataset("allenai/WildChat", split="train", streaming=True)

    # profiling accumulators
    n_seen = 0
    n_english = 0
    models = Counter()
    turns_list = []
    timestamps = []
    toxic_count = 0
    redacted_count = 0
    lang_counter = Counter()

    # wedding accumulators
    wedding_rows = []
    wedding_hit_terms = Counter()
    n_wedding = 0          # loose: any seed (strong OR weak)
    n_wedding_strict = 0   # strict: >=1 strong term (single hit qualifies)

    schema_printed = False

    for rec in ds:
        n_seen += 1
        lang = rec.get("language")
        lang_counter[lang] += 1

        if not schema_printed:
            print("\n=== SCHEMA (first record keys) ===")
            print(list(rec.keys()))
            conv = rec.get("conversation", [])
            if conv:
                print("Utterance keys:", list(conv[0].keys()))
            schema_printed = True

        if lang != "English":
            if n_seen >= MAX_STREAM:
                break
            continue

        n_english += 1
        models[rec.get("model")] += 1
        turns_list.append(rec.get("turn"))
        if rec.get("timestamp") is not None:
            timestamps.append(str(rec.get("timestamp")))
        if rec.get("toxic"):
            toxic_count += 1
        if rec.get("redacted"):
            redacted_count += 1

        # Stage 1 lexical filter on user text
        utext = user_text(rec.get("conversation", []))
        hits = lexical_hits(utext)
        strong_hits = _strong_pat.findall(utext or "")
        is_strict = len(strong_hits) > 0
        if hits:
            n_wedding += 1
            if is_strict:
                n_wedding_strict += 1
            for h in hits:
                wedding_hit_terms[h.lower()] += 1
            if len(wedding_rows) < 800:  # cap stored examples
                wedding_rows.append({
                    "conversation_id": rec.get("conversation_id"),
                    "model": rec.get("model"),
                    "turn": rec.get("turn"),
                    "timestamp": str(rec.get("timestamp")),
                    "n_hits": len(hits),
                    "strict": is_strict,
                    "unique_terms": ";".join(sorted(set(h.lower() for h in hits))),
                    "first_user_msg": (utext[:500].replace("\n", " ")),
                })

        if n_english >= SAMPLE_SIZE or n_seen >= MAX_STREAM:
            break

    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------
    turns_series = pd.Series([t for t in turns_list if t is not None])
    wedding_rate = n_wedding / n_english if n_english else 0
    wedding_rate_strict = n_wedding_strict / n_english if n_english else 0

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records_streamed": n_seen,
        "english_records_profiled": n_english,
        "english_share_of_streamed": round(n_english / n_seen, 4) if n_seen else None,
        "model_split": dict(models),
        "turns": {
            "mean": round(float(turns_series.mean()), 3) if len(turns_series) else None,
            "median": float(turns_series.median()) if len(turns_series) else None,
            "p90": float(turns_series.quantile(0.9)) if len(turns_series) else None,
            "max": int(turns_series.max()) if len(turns_series) else None,
            "multi_turn_share": round(float((turns_series > 1).mean()), 4) if len(turns_series) else None,
        },
        "timestamp_range": {
            "min": min(timestamps) if timestamps else None,
            "max": max(timestamps) if timestamps else None,
        },
        "toxic_rate": round(toxic_count / n_english, 4) if n_english else None,
        "redacted_rate": round(redacted_count / n_english, 4) if n_english else None,
        "top_languages_streamed": dict(lang_counter.most_common(12)),
        "wedding_stage1": {
            "n_wedding_loose": n_wedding,
            "n_wedding_strict": n_wedding_strict,
            "wedding_rate_loose": round(wedding_rate, 5),
            "wedding_rate_strict": round(wedding_rate_strict, 5),
            "projected_full_english_loose": None,  # filled below
            "projected_full_english_strict": None,
            "top_hit_terms": dict(wedding_hit_terms.most_common(30)),
        },
    }

    # Project to full English set. WildChat ~650k convs, English ~53% -> ~344k.
    FULL_CONVS = 650_000
    ENGLISH_SHARE = 0.53
    full_english = FULL_CONVS * ENGLISH_SHARE
    summary["wedding_stage1"]["projected_full_english_loose"] = int(wedding_rate * full_english)
    summary["wedding_stage1"]["projected_full_english_strict"] = int(wedding_rate_strict * full_english)

    # ---------------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------------
    with open(os.path.join(DATA_DIR, "phase0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(wedding_rows).to_csv(
        os.path.join(DATA_DIR, "phase0_wedding_candidates_sample.csv"), index=False
    )

    # ---------------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY (English-only)")
    print("=" * 60)
    print(f"Records streamed:        {n_seen:,}")
    print(f"English profiled:        {n_english:,} ({summary['english_share_of_streamed']:.1%} of streamed)")
    print(f"Model split:             {dict(models)}")
    print(f"Turns mean/median/p90:   {summary['turns']['mean']} / {summary['turns']['median']} / {summary['turns']['p90']}")
    print(f"Multi-turn share:        {summary['turns']['multi_turn_share']:.1%}")
    print(f"Timestamp range:         {summary['timestamp_range']['min']} -> {summary['timestamp_range']['max']}")
    print(f"Toxic / redacted rate:   {summary['toxic_rate']:.1%} / {summary['redacted_rate']:.1%}")
    print("-" * 60)
    print(f"WEDDING (Stage 1 lexical):")
    print(f"  LOOSE  (any seed):     {n_wedding:,}  ({wedding_rate:.3%})  -> proj ~{summary['wedding_stage1']['projected_full_english_loose']:,}")
    print(f"  STRICT (>=1 strong):   {n_wedding_strict:,}  ({wedding_rate_strict:.3%})  -> proj ~{summary['wedding_stage1']['projected_full_english_strict']:,}")
    print(f"  top terms:             {dict(wedding_hit_terms.most_common(12))}")
    print("=" * 60)
    print(f"Wrote: data/phase0_summary.json, data/phase0_wedding_candidates_sample.csv")


if __name__ == "__main__":
    main()

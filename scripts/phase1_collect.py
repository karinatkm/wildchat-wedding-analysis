"""
Phase 1 — Full-dataset scan to build the wedding candidate pool (English only).

Streams ALL of allenai/WildChat, keeps English conversations whose user text
matches the Stage-1 LOOSE wedding lexicon (high recall), and persists full
conversations + metadata to data/wedding_candidates.parquet for Stage 2.

Run:  .venv/bin/python scripts/phase1_collect.py
"""
import json
import re
import os
import time
from collections import Counter
from datetime import datetime

import pandas as pd
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT = os.path.join(DATA_DIR, "wedding_candidates.parquet")

# Reuse the strong/weak lexicon from Phase 0
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
WEAK_SEEDS = [
    "engagement", "engaged", "marriage", "married", "betrothed",
    "ceremony", "reception", "registry", "vows", "bouquet", "rsvp",
]
ALL_SEEDS = sorted(set(STRONG_SEEDS) | set(WEAK_SEEDS))
_strong_pat = re.compile(r"\b(" + "|".join(re.escape(s) for s in STRONG_SEEDS) + r")\b", re.IGNORECASE)
_loose_pat = re.compile(r"\b(" + "|".join(re.escape(s) for s in ALL_SEEDS) + r")\b", re.IGNORECASE)

FLUSH_EVERY = 5000   # write partial parquet periodically


def user_text(conversation):
    parts = [t.get("content", "") for t in conversation if t.get("role") == "user"]
    return "\n".join(p for p in parts if p)


def main():
    print(f"[{datetime.utcnow().isoformat()}Z] Full-dataset scan starting...")
    ds = load_dataset("allenai/WildChat", split="train", streaming=True)

    rows = []
    n_seen = 0
    n_english = 0
    n_cand = 0
    n_strict = 0
    t0 = time.time()

    for rec in ds:
        n_seen += 1
        if rec.get("language") != "English":
            continue
        n_english += 1
        conv = rec.get("conversation", [])
        utext = user_text(conv)
        if not utext:
            continue
        loose_hits = _loose_pat.findall(utext)
        if not loose_hits:
            continue
        strong_hits = _strong_pat.findall(utext)
        is_strict = len(strong_hits) > 0
        n_cand += 1
        if is_strict:
            n_strict += 1
        rows.append({
            "conversation_id": rec.get("conversation_id"),
            "model": rec.get("model"),
            "timestamp": str(rec.get("timestamp")),
            "turn": rec.get("turn"),
            "redacted": rec.get("redacted"),
            "n_user_chars": len(utext),
            "matched_terms": ";".join(sorted(set(h.lower() for h in loose_hits))),
            "n_hits": len(loose_hits),
            "strict": is_strict,
            "user_text": utext,
            "conversation_json": json.dumps(
                [{"role": t.get("role"), "content": t.get("content")} for t in conv],
                ensure_ascii=False,
            ),
        })

        if n_cand % FLUSH_EVERY == 0 and n_cand > 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            rate = n_seen / max(time.time() - t0, 1)
            print(f"  seen={n_seen:,} english={n_english:,} cand={n_cand:,} "
                  f"strict={n_strict:,} | {rate:.0f} rec/s | flushed")

    # Final write
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    elapsed = time.time() - t0

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records_streamed": n_seen,
        "english_records": n_english,
        "candidates_loose": n_cand,
        "candidates_strict": n_strict,
        "loose_rate_in_english": round(n_cand / n_english, 5) if n_english else None,
        "strict_rate_in_english": round(n_strict / n_english, 5) if n_english else None,
        "elapsed_sec": round(elapsed, 1),
        "output": OUT,
    }
    with open(os.path.join(DATA_DIR, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"Streamed:           {n_seen:,}")
    print(f"English:            {n_english:,}")
    print(f"Loose candidates:   {n_cand:,}  ({summary['loose_rate_in_english']:.2%})")
    print(f"Strict candidates:  {n_strict:,}  ({summary['strict_rate_in_english']:.2%})")
    print(f"Elapsed:            {elapsed/60:.1f} min")
    print(f"Wrote:              {OUT}")


if __name__ == "__main__":
    main()

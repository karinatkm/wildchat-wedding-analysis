"""
Phase 3 (Stage 3) — Threshold calibration & validation for the wedding filter.

- Applies the decision rule (SBERT gate + NLI entailment) to produce the final
  wedding-relevant set, and a real-planning vs creative-writing split.
- Emits a stratified review sample for manual labeling.
- If data/validation_labels.csv exists (cols: conversation_id,label in {1,0}),
  reports precision / recall / F1 at the chosen thresholds.

Inputs : data/wedding_scored.parquet
Outputs: data/wedding_final.parquet, data/validation_sample.csv (+ metrics)

Run:  .venv/bin/python scripts/phase3_calibrate.py [--nli-threshold 0.5]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SCORED = os.path.join(DATA_DIR, "wedding_scored.parquet")
FINAL = os.path.join(DATA_DIR, "wedding_final.parquet")
VAL_SAMPLE = os.path.join(DATA_DIR, "validation_sample.csv")
VAL_LABELS = os.path.join(DATA_DIR, "validation_labels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-threshold", type=float, default=0.5)
    ap.add_argument("--sample-size", type=int, default=160)
    args = ap.parse_args()

    df = pd.read_parquet(SCORED)
    df["nli_wedding"] = df["nli_wedding"].fillna(0.0)
    df["nli_real_planning_p"] = df["nli_real_planning_p"].fillna(0.0)

    # Decision rule: NLI entailment is the precise cut (SBERT already gated upstream)
    df["is_wedding"] = df["nli_wedding"] >= args.nli_threshold
    final = df[df["is_wedding"]].copy()
    final.to_parquet(FINAL, index=False)

    n_real = int((final["mode"] == "real_planning").sum())
    n_creative = int((final["mode"] == "creative_writing").sum())

    print("=" * 60)
    print("PHASE 3 — decision + split")
    print("=" * 60)
    print(f"Scored candidates:     {len(df):,}")
    print(f"Wedding-relevant (NLI>={args.nli_threshold}): {len(final):,}")
    print(f"  real planning:       {n_real:,}")
    print(f"  creative writing:    {n_creative:,}")

    # Stratified review sample across NLI score bins (for manual labeling)
    bins = pd.cut(df["nli_wedding"], [0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    sample = (df.groupby(bins, observed=True, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.sample_size // 6), random_state=42)))
    cols = ["conversation_id", "sbert_sim", "nli_wedding", "nli_real_planning_p",
            "mode", "matched_terms", "user_text"]
    sample_out = sample[cols].copy()
    sample_out["user_text"] = sample_out["user_text"].str.slice(0, 300).str.replace("\n", " ", regex=False)
    sample_out["label"] = ""  # to be filled manually: 1=wedding, 0=not
    sample_out.to_csv(VAL_SAMPLE, index=False)
    print(f"\nWrote review sample: {VAL_SAMPLE} ({len(sample_out)} rows)")

    # Validation metrics if labels provided
    if os.path.exists(VAL_LABELS):
        labels = pd.read_csv(VAL_LABELS)[["conversation_id", "label"]]
        m = df.merge(labels, on="conversation_id", how="inner")
        m = m.dropna(subset=["label"])
        if len(m):
            y_true = (m["label"].astype(int) == 1)
            y_pred = m["is_wedding"]
            tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            print("\n--- VALIDATION (labeled sample) ---")
            print(f"n={len(m)}  TP={tp} FP={fp} FN={fn} TN={tn}")
            print(f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
            metrics = {"n": len(m), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": prec, "recall": rec, "f1": f1,
                       "nli_threshold": args.nli_threshold}
            with open(os.path.join(DATA_DIR, "phase3_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
    else:
        print(f"\n(No {VAL_LABELS} yet — fill the 'label' column in the review sample to compute metrics.)")


if __name__ == "__main__":
    main()

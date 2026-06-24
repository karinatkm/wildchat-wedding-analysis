"""Build the reproducible analysis notebook (analysis.ipynb) with nbformat."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# WildChat × Weddings — How people use an LLM for weddings

**Question:** Within a large, real-world corpus of human–ChatGPT conversations
([allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat)), what do
people actually do with the assistant when the topic is *weddings* — and what does
that imply for a wedding-adjacent brand?

This notebook is fully reproducible from the cached artifacts in `data/`. The heavy
extraction (full-dataset stream + embedding) is done by the `scripts/phase*.py`
pipeline; here we load its outputs and analyze.

**Pipeline (3-stage precision funnel):**
1. **Stage 0 — Profiling & feasibility:** stream-sample WildChat, confirm schema,
   measure wedding prevalence vs. other industries.
2. **Stage 1 — Lexical recall:** full-dataset scan; keep English conversations whose
   *user* text hits a wedding lexicon (strong vs. weak terms).
3. **Stage 2 — Semantic precision:** embed candidates (SBERT) and classify by
   **nearest competing prototype** (wedding / fiction-roleplay / e-commerce /
   travel / relationship / generic). Keep only wedding-nearest, strong-term hits.
4. **Stage 3 — Validation:** manual spot-check + cluster inspection.
""")

code("""import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd())=="notebooks" else os.getcwd()
DATA = os.path.join(ROOT, "data")
def load_json(name):
    with open(os.path.join(DATA, name)) as f: return json.load(f)
phase0 = load_json("phase0_summary.json")
multi  = load_json("phase0_multitopic_summary.json")
phase1 = load_json("phase1_summary.json")
phase2 = load_json("phase2_summary.json")
res    = load_json("analysis_results.json")
print("Loaded pipeline artifacts.")""")

md("""## 1. Dataset profile (English sample)

Schema and high-level stats from the streamed English sample.""")
code("""print("English share of streamed:", phase0["english_share_of_streamed"])
print("Model split:", phase0["model_split"])
print("Turns (mean/median/p90):", phase0["turns"]["mean"], phase0["turns"]["median"], phase0["turns"]["p90"])
print("Multi-turn share:", phase0["turns"]["multi_turn_share"])
print("Timestamp range (sample):", phase0["timestamp_range"])""")

md("""## 2. Why weddings? — feasibility across industries

Before committing, we measured strict-lexical prevalence of several brand-relevant
domains over an 80k English sample. Weddings are a mid-volume, high-commerce-intent
niche.""")
code("""topics = pd.DataFrame(multi["topics"]).sort_values("matches", ascending=False)
plt.figure(figsize=(8,4.5))
sns.barplot(data=topics, x="proj_full_english", y="topic", color="#4c72b0")
plt.title("Projected full-English conversations by industry (strict lexical, raw recall)")
plt.xlabel("projected conversations"); plt.ylabel("")
plt.tight_layout(); plt.show()
topics[["topic","matches","rate_pct","proj_full_english"]]""")

md("""## 3. The funnel — recall then precision

The full-dataset scan found many lexical candidates, but most are **not** real wedding
usage. The contrastive classifier routes them to the right bucket.""")
code("""print(f"Full scan: {phase1['records_streamed']:,} convs streamed, "
      f"{phase1['english_records']:,} English")
print(f"Stage 1 lexical candidates: {phase1['candidates_loose']:,} loose / "
      f"{phase1['candidates_strict']:,} strict")
counts = phase2["nearest_proto_counts"]
s = pd.Series(counts).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=s.values, y=s.index, palette="rocket")
plt.title("Where do wedding-lexicon candidates actually belong? (nearest prototype)")
plt.xlabel("candidate conversations"); plt.ylabel("")
plt.tight_layout(); plt.show()
print(f"Wedding-relevant after Stage 2 (+strict, sim>=0.30): {phase2['wedding_relevant']:,}")""")

md("""### Key finding #1 — Wedding *words* mostly aren't wedding *usage*

Of the wedding-lexicon candidates, the **plurality are fiction/roleplay** (characters
who "get married" — e.g. Doki Doki Literature Club and Minecraft roleplay), plus large
shares of **e-commerce product listings** ("bridal" jewelry to translate/optimize),
**travel guides** (honeymoon destinations), and **generic tasks**. Only a minority are
genuine wedding planning/advice/content. *Lexical filters alone would be ~30% precise.*""")

md("""## 4. The validated wedding subset — what people ask for

Intent taxonomy over the high-precision subset.""")
code("""w = pd.read_parquet(os.path.join(DATA, "wedding_final.parquet"))
print("Wedding conversations:", len(w))
ic = pd.Series(res["intent_counts"]).sort_values(ascending=False)
plt.figure(figsize=(8,4.5))
sns.barplot(x=ic.values, y=ic.index, color="#7b2d8e")
plt.title(f"Wedding conversations by intent (n={len(w)})")
plt.xlabel("conversations"); plt.ylabel(""); plt.tight_layout(); plt.show()
ic""")

md("""### Key finding #2 — It's a **text-generation** tool, not a planner

The dominant intents are **writing tasks**: vows/speeches/toasts, invitation &
thank-you messaging, and captions/blog content. Operational planning (budgets,
checklists, logistics, etiquette) is comparatively rare. People bring the assistant
in to *write wedding words*, not to *project-manage the wedding*.""")

md("""## 5. Sub-topics (TF-IDF + KMeans)""")
code("""for cid, terms in res["cluster_terms"].items():
    print(f"Cluster {cid} (n={res['cluster_sizes'][cid]}): {', '.join(terms[:8])}")""")

md("""## 6. Engagement — where conversations get deep""")
code("""tbi = pd.Series(res["turns_by_intent"]).sort_values(ascending=False)
plt.figure(figsize=(8,4.5))
sns.barplot(x=tbi.values, y=tbi.index, color="#2d8e7b")
plt.title("Avg conversation turns by intent"); plt.xlabel("avg turns"); plt.ylabel("")
plt.tight_layout(); plt.show()
print("Overall turns:", res["turns_overall"])""")

md("""### Key finding #3 — Iteration happens on *personal, high-stakes* text

The longest back-and-forths are **messaging/invitations, planning, and budgets** —
tasks where wording is personal or details must be right. One-shot intents
(design/Midjourney prompts, short articles, recommendations) end quickly. Depth is a
proxy for *where the assistant is trusted to get it right* and where users invest effort.""")

md("""## 7. Temporal volume""")
code("""w["ts"] = pd.to_datetime(w["timestamp"], errors="coerce", utc=True)
by_week = w.set_index("ts").resample("W").size()
plt.figure(figsize=(9,3.8)); by_week.plot(color="#8e5a2d")
plt.title("Wedding conversation volume over time (weekly)"); plt.ylabel("conversations/week"); plt.xlabel("")
plt.tight_layout(); plt.show()
print("Date range:", res["date_range"])""")

md("""## 8. Example conversations (qualitative)""")
code("""for t in w.sort_values("sim_wedding_planning", ascending=False)["user_text"].head(6):
    print("•", " ".join(str(t).split())[:160], "\\n")""")

md("""## 9. Brand-lens takeaways

See `REPORT.md` for the full write-up. Headlines:

1. **Wedding intent is rare and noisy in the wild** — fiction/roleplay, product
   listings, and travel dominate the wedding *vocabulary*. A brand listening for
   "wedding" signal must filter semantically, not lexically.
2. **The killer job-to-be-done is writing** — vows, speeches, toasts, invitations,
   thank-yous, captions. A wedding brand's most-used AI feature would be a
   *guided wedding-writing assistant*, not a generic planner.
3. **Depth = trust + stakes** — messaging/budget/planning drive multi-turn iteration;
   that's where an assistant can add real value and where users will tolerate friction.
4. **Underserved, commerce-adjacent moments** — budget/cost and vendor/venue
   recommendations are low-volume here (2023, generic ChatGPT), signaling a *product
   gap*: people don't yet trust a general LLM for transactional wedding decisions.
""")

nb["cells"] = c
out = os.path.join(ROOT, "notebooks", "analysis.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out)

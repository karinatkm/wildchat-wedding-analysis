# WildChat × Weddings

How do real people use a general-purpose LLM (ChatGPT) when the topic is **weddings**?
This project mines the [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat)
corpus of real human-assistant conversations, builds a validated wedding subset with a
recall-to-precision funnel, and analyzes intents, sub-topics, and engagement, framed for a
wedding-adjacent brand.

**Read first:** [`REPORT.md`](REPORT.md) (insights) · [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) (reproducible analysis + charts).

> **Note on notebook rendering:** GitHub's `.ipynb` preview can occasionally show "Unable to render code block" due to browser extensions or transient renderer issues. If the notebook doesn't load, use the [nbviewer mirror](https://nbviewer.org/github/karinatkm/wildchat-wedding-analysis/blob/main/notebooks/analysis.ipynb) for a reliable view.

## Headline findings

1. **Wedding *words* mostly aren't wedding *usage***: ~49% of wedding-lexicon hits are fiction/roleplay, plus large shares of e-commerce listings, honeymoon travel, and generic tasks. Keyword-only filtering is ~30% precise.
2. **The job-to-be-done is writing, not planning**: vows/speeches, invitation & thank-you messaging, and captions dominate; budgets/checklists/logistics are the long tail. Transactional intents (budget, vendor/venue) are low-volume, signaling an open commerce opportunity for a trusted wedding-specific assistant.
3. **Depth tracks personal stakes**: messaging, planning, and budget conversations run longest (~4.9 turns).

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 0. Profile + feasibility | `scripts/phase0_explore.py`, `scripts/phase0_multitopic.py` | `data/phase0_*.json` |
| 1. Lexical recall (full scan) | `scripts/phase1_collect.py` | `data/wedding_candidates.parquet` |
| 2a. Embed (cache) | `scripts/phase2_embed.py` | `data/cand_embeddings.npy` |
| 2b. Contrastive classify | `scripts/phase2_classify.py` | `data/wedding_scored.parquet` |
| 3. Calibrate/validate | `scripts/phase3_calibrate.py` | `data/validation_sample.csv` |
| 4. Analyze | `scripts/phase4_analyze.py` | `figures/*.png`, `data/analysis_results.json`, `data/wedding_final.parquet` |
| 5. Notebook | `scripts/build_notebook.py` | `notebooks/analysis.ipynb` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned because torch 2.2.x is incompatible with numpy 2.x.
> An optional `HF_TOKEN` env var greatly speeds up dataset streaming (avoids rate limits).

## Run

```bash
# Stage 0 — profiling (streamed samples; minutes)
.venv/bin/python scripts/phase0_explore.py
.venv/bin/python scripts/phase0_multitopic.py

# Stage 1 — full-dataset lexical scan (~15-30 min, network-bound)
.venv/bin/python scripts/phase1_collect.py

# Stage 2 — embed once (CPU ~13 min), then classify (seconds; iterate freely)
.venv/bin/python scripts/phase2_embed.py
.venv/bin/python scripts/phase2_classify.py

# Stage 3/4 — validate + analyze
.venv/bin/python scripts/phase3_calibrate.py
.venv/bin/python scripts/phase4_analyze.py

# Build + execute the notebook
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The analysis/notebook steps run from cached artifacts in `data/`, so you can re-run them
without repeating the expensive stream/embed.

## Method note

Stage 2 uses **contrastive multi-prototype classification**: each candidate is embedded
(SBERT `all-MiniLM-L6-v2`) and assigned to its **nearest** category among competing
prototypes (`wedding_planning`, `fiction_roleplay`, `ecommerce_listing`, `travel_guide`,
`relationship_general`, `generic_task`). This cleanly diverts the dominant noise
(fan-fiction, product listings, honeymoon travel) that a single keyword or weak NLI
classifier lets through. Final precision ≈ 70–75% (manual spot-check).

## Repo layout

```
scripts/    pipeline stages (phase0 → build_notebook)
data/        cached artifacts (parquet, npy, json) — large files gitignored
figures/     exported charts
notebooks/   analysis.ipynb (reproducible)
REPORT.md    insights write-up
```

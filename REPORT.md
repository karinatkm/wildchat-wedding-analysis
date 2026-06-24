# How People Use an LLM for Weddings: Insights from WildChat

*Analysis of real human-ChatGPT conversations ([allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat)), framed for a wedding-adjacent brand.*

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Findings & What They Mean](#2-findings--what-they-mean)
3. [Implications by Player Type](#3-implications-by-player-type)
4. [Scaling the Analysis](#4scaling-the-analysis)
5. [Methodology in Plain Terms](#5-methodology-in-plain-terms)
6. [Technical Appendix](#6-technical-appendix)

## 1. Executive Summary

To understand how people actually use an AI assistant for weddings, the **full WildChat dataset** (~529k real human-ChatGPT conversations, ~284k English) is distilled a **validated set of 157 genuine wedding conversations** for close study. The limitation of this small sample will be addressed in Section 4.

**Scope of this analysis.** We focus on *how users use ChatGPT for wedding-related themes*: what they ask for and how they interact. We do **not** analyze how ChatGPT sources or cites content when it responds. That is a distinct question, and a natural second analysis for the future.

Topline findings:

- **Wedding *words* mostly aren't wedding *usage*.** Only ~6% of conversations containing wedding vocabulary are actually about real weddings. The rest are fiction, honeymoon travel, and more. *Therefore, a brand that monitors "wedding" mentions with keywords alone is reading ~50% irrelevant and keyword monitoring on its own would not be effective.*
- **The #1 job people use AI for is writing, not planning.** The dominant requests are **vows, speeches/toasts, invitations, and thank-you messages**, not budgets, checklists, or logistics. This could be a hint to how brands can shift their online AI-assistant offerings and also AEO-friendly content from "AI wedding planner" to writing guides/assistants.
- **People iterate most on the words that get read aloud or sent to others.** The longest back-and-forths are about personal wording (messages, invitations) and money (budgets): the high-stakes moments where "good enough" isn't good enough.

> **How confident should you be?** These are **directional** findings from a deliberately high-precision sample (n=157, 2023 data). This is meant to be a discovery set of findings to be scaled. Section 4 lays out how to scale this into statistically robust evidence.

## 2. Findings & Implications

### Finding 1: Wedding vocabulary is dominated by *non*-wedding usage

When we sorted every wedding-keyword conversation into the category it actually belongs to, real wedding planning was a small minority:

| What the conversation is *really* about | Share of wedding-keyword hits |
|---|---|
| Fiction / roleplay (characters who "get married") | 49% |
| Generic tasks (summaries, emails, code) | 24% |
| E-commerce product listings ("bridal" jewelry, etc.) | 12% |
| Travel guides (honeymoon destinations) | 7% |
| **Genuine wedding planning** | **6%** |
| Relationship / marriage (divorce, anniversaries) | 4% |

> **From the data, what the noise looks like:**
> - *Fiction:* "[there would be dialogue in this story:] (MC and Sayori have been **married** for 9 months, and now, their first child…)"
> - *E-commerce:* "There is a product titles. help me modify it. 4/6/8/10mm Shiny Faux Pearl Jewelry Making Beads with Hole Round Craft…"
> - *Travel:* "Write an engaging and a constructive guide for my Yosemite travel guide on '**Best time to visit for Honeymoon**'…"

**Why it matters:** the wedding "signal" in public AI/social data is buried in lookalike noise.

**Implication:** any brand listening for wedding intent must filter by *meaning*, not keywords; otherwise half of what it measures is fan-fiction. *(Confidence: High, directly measured.)* The old keyword approach in SEO is no longer effective in the world of AEO and intent measurement will be critical.

### Finding 2: People use the assistant to *write wedding words*

Within the genuine wedding set (n=157), requests skew heavily toward **text generation**:

| Intent (what the user asked for) | Conversations (n=157) |
|---|---|
| Vows & speeches | 25 |
| Invitations & messaging | 23 |
| Attire & beauty | 21 |
| Design / visual | 10 |
| Recommendation | 9 |
| Planning / logistics | 7 |
| Content / blog | 7 |
| Budget | 5 |
| Etiquette | 3 |
| Other | 47 |

Text-generation intents dominate. Operational planning and commerce (recommendation 9, budget 5) are the long tail.

> **From the data, what people ask for:**
> - *Vows / speeches:* "write me an officiant address for a wedding… Include some words of encouragement to the couple" · "What should the father of the bride say at a wedding?" · "Write a wedding toast for the groom"
> - *Invitations / messaging:* "write a thank you note to my hair stylist for my wedding day…" · "How to compose [a] message to [a] family member that isn't invited to wedding day and is causing drama because of it"
> - *Attire / beauty:* "rewrite for a wedding blog from the third pov: Carly wanted… timeless clean elegance. She had a minimalist look…"

**Why it matters:** people reach for AI to produce the *personal, hard-to-write words* of a wedding, not to project-manage it.

**Implication:** 
- The highest-usage AI feature for a wedding brand is a **guided wedding-writing assistant**, not another planner/checklist. 
- If brands would like to be referenced in AI-generated responses and increase impressions, having AEO-friendly content assisting people with these decisions will be crucial, e.g. guides to writing vows, tips on invitations, etc.. 
*(Confidence: Medium, clear pattern, small sample.)*

### Finding 3: The deepest conversations are about wording and money

Average back-and-forth turns by request type: invitations/messaging **4.9**, planning **4.9**, budget **4.8**, vows/speeches 3.7, etiquette 3.3, attire 3.1. One-shot types: content/blog 1.7, design/image prompts 1.4. Overall mean 3.2 turns (median 1).

> **From the data, conversations people kept refining:**
> - *17 turns, polishing wording:* "rewrite for a wedding blog from the third pov in one paragraph in normal words: The ceremony was very beautiful…" (each turn tweaks tone/phrasing)
> - *A high-stakes message:* "How to compose [a] message to [a] family member that isn't invited to wedding day and is causing drama…"

**What this means (plainly):** the conversations people *keep going back and forth on* are the ones where the output (a) will be **read aloud or sent to other people** (invitation wording, thank-you notes, messages), or (b) involves **money**, like a budget. People do multiple rounds because it has to be *just right* and the stakes feel personal. By contrast, one-shot tasks like a quick image prompt or a blog draft are usually accepted on the first try.

**Implication:** AI features for these high-stakes moments should be built for **refinement loops**: easy edits to tone, names, and details across several rounds, not one-shot output. *(Confidence: Medium.)*

## 3. Recommendations

### Content, SEO & AI discoverability (any wedding brand with a website)
- **Listen semantically, not lexically.** Old-school keyword-based social/market listening for "wedding" is ~50% fan-fiction and shopping noise (Finding 1); meaning-based filtering is required to trust the signal. The method in this report is itself a reusable capability. To win in the world of AEO, semantic filtering will be important *(Confidence: High.)*
- **Publish AI-retrievable content on the exact topics people ask AI to help write.** Because vow-writing, speech-writing, and invitation wording dominate real wedding AI usage (Finding 2), brands that publish strong, structured, citable guides on *those specific topics* may improve their odds of being **surfaced or cited inside AI assistant answers**. *(Confidence: Directional. Finding 2 establishes the demand; the citation payoff is a hypothesis to test.)*
- **Note on scope:** this report measures *user demand*, not *how AI sources or cites content*. Whether and how publishing content earns AI citations is a separate question and a natural follow-up analysis.

### Product Offerings
- **Lead AI features with writing, not planning.** Ship a guided assistant for vows, speeches, toasts, invitations, and thank-yous, the highest-usage jobs (Finding 2), before another checklist/planner. *(Confidence: Medium.)*
- **Design for iteration on personal text.** If product offerings include invitations, websites, and other points of human interactions, make tone/name/detail edits effortless across multiple rounds for messaging and budget tools, where users naturally go deep (Finding 3). *(Confidence: Medium.)*


## 4. Scaling the Analysis

We treat each boundary of this study as a defined next step. The current work answers *what kinds of things people do*; the items below are what it would take to answer *how much, for whom, and how it's changing*.

| What we can claim today | Boundary / limitation | How to harden & scale it |
|---|---|---|
| Directional intent mix from a clean n=157 sample | Too small for segment-level statistics or significance testing | **LLM-assisted labeling** to scale the validated set to ~2-5k at high precision, enabling robust frequencies, segmentation, and trend tests |
| A precise but conservative wedding filter (~70-75% precision) | Prototype method trades some recall for precision; not a benchmarked classifier | Build a small **human-labeled gold set**, then fine-tune a classifier to push precision >90% for production "wedding-intent" detection |
| A 2023 baseline of behavior on older models | Behavior likely shifts as models improve | **Refresh with newer conversation data** to measure how wedding AI usage is evolving over time |
| English-language usage only | Misses global/regional patterns | **Multilingual embeddings** to size non-English and regional differences |
| Demand signal (what users *ask*) | We see requests, not whether the AI *helped* | Analyze **assistant responses & outcomes** to find where AI fails users, the sharpest unmet-need finder |

Each of the above converts a caveat into a concrete, fundable investment that expands what the analysis can prove.

## 5. Methodology in Plain Terms

*How we turned ~529,000 raw conversations into a trustworthy set of findings, why each step is the right tool, and where the same techniques are used in academia and industry. No technical background required.*

### The core challenge: finding needles that look like the haystack

Searching a giant pile of conversations for "wedding" is easy. The hard part is that **most text containing wedding words has nothing to do with real weddings**: it's fan-fiction where characters marry, product listings for "bridal" jewelry, honeymoon travel articles. So our central problem is *precision*: not "can we find wedding words?" but "can we find genuine wedding intent and throw out the lookalikes?"

This is solved with a **funnel**: start wide and cheap, then narrow with progressively smarter (and more expensive) filters. 

### Step 1: Keyword screening (high recall, low precision)

**Method:** flagged any conversation containing wedding vocabulary, splitting terms into *strong* (wedding, bride, vows, officiant; rarely used outside weddings) and *weak* (marriage, ceremony, engagement; often used elsewhere, like "prior engagement" or "awards ceremony").

**Why:** keyword matching is fast and cheap, and at this stage we want to **catch everything** real, accepting that we'll also catch junk. This is the classic **precision-recall trade-off**: recall = "did we find all the real ones?", precision = "of what we found, how much is real?" Early on recall is optimized, because anything we discard here is lost forever.

**Where this is used:** this is "boolean/keyword retrieval," the foundation of search engines with the precision/recall framing is a standard from information-retrieval theory.

### Step 2: Adding *meaning*, not just words (embeddings)

**Method:** converted each conversation into a list of ~400 numbers (an "embedding") using a model called **SBERT**. These numbers are coordinates in a "meaning map": texts with similar *meaning* land close together, even if they share no words. "Help me write my wedding vows" and "what should I say at the altar to my partner" sit near each other; a jewelry product list sits far away.

**Why:** keywords are blind to context: they can't tell "I'm planning my wedding" from "the characters had a wedding in chapter 3." Embeddings capture meaning, which is what lets us separate genuine intent from coincidental word overlap.

**Where this is used:** "Closeness" is measured with **cosine similarity**, an industry-standard way to score how alike two pieces of text are.

### Step 3: Sorting by "best fit," not "good enough" (contrastive prototypes)

**Method:** we wrote a handful of short example sentences for each *category* a wedding-word conversation might really belong to (`wedding planning`, `fiction/roleplay`, `e-commerce listing`, `travel guide`, `relationship/marriage`, `generic task`) and averaged each category's examples into a "prototype" (a representative point on the meaning map). Every conversation was then assigned to its **nearest** prototype.

**Why this beats a simple threshold:** our first attempt asked "is this *close enough* to wedding?", and a product listing that's vaguely wedding-ish would sneak in. The better question is **"of all the things this could be, is *wedding* the best match?"** A honeymoon travel article is somewhat close to "wedding," but it's *much* closer to "travel guide," so it gets routed away. This comparison-against-alternatives is what cleaned up the data, and the category breakdown itself became Finding 1.

**Where this is used:** this is a **nearest-centroid / prototype classifier** combined with modern **zero-shot classification**: labeling text into categories *without* hand-labeling thousands of training examples. "Zero-shot" is now standard practice when you need a custom classifier fast and don't have a labeled dataset; prototype-based versions.

> **Why not just ask an AI/LLM to label each one?** That's a valid modern approach, but it's slow and costly at this scale on a CPU-only old Macbook used for this analysis, and harder to audit.

### Step 4: Discovering themes without pre-deciding them (clustering)

**Method:** to find sub-topics *within* the wedding set, we used **TF-IDF** (a way of scoring which words are distinctive to a conversation, down-weighting filler words everyone uses) and **k-means clustering** to automatically group similar conversations. Out came natural themes: speeches, invitations/messaging, gifts, captions, visual/design prompts.

**Why:** we didn't want to impose our assumptions about what people ask. Clustering lets the *data* reveal its own structure, and as a bonus, it surfaced a hidden pocket of roleplay contamination that we then removed.

**Where this is used:** K-means is the classic for **customer-feedback theme mining, survey analysis, and market segmentation**: the same tool a CX team uses to find recurring complaints in thousands of reviews.

### Step 5: Checking our work by hand (human validation)

**Method:** we manually read samples of what the filter kept and rejected, especially the borderline cases near the cutoff, to estimate accuracy, landing at **~70-75% precision**. We also re-read clusters to catch and remove leftover contamination.

**Why:** automated metrics can lie if the method is subtly broken; a human read is the gold standard for "is this actually right?" Sampling near the decision boundary is the most efficient place to look, because that's where the model is least sure.

**Where this is used:** **human-in-the-loop evaluation** is standard for any production classifier or AI system; reporting **precision** (and inspecting a *confusion matrix* of right/wrong calls) is the accepted way to communicate how much to trust an automated label.

### Measuring engagement: conversation length as a proxy

We used **number of back-and-forth turns** as a stand-in for how much effort and trust a task warrants. It's an imperfect but widely-used **proxy metric** (like "time on page" in web analytics): more turns means the user kept refining, which signals the task mattered and the assistant was worth iterating with.



## 6. Technical Appendix

### Pipeline at a glance

A 3-stage **precision funnel**: cheap recall first, expensive precision second.

1. **Stage 0: Profiling & feasibility** (`scripts/phase0_explore.py`, `phase0_multitopic.py`): confirmed schema, profiled English share/model split/turns, and compared wedding prevalence against 12 other industries to make an informed domain choice.
2. **Stage 1: Lexical recall** (`scripts/phase1_collect.py`): streamed the full dataset; kept English conversations whose **user** text hit a wedding lexicon (strong terms like *wedding, bride, vows, officiant* vs. weak terms like *marriage, ceremony, engagement*). Yield: **6,458 loose / 1,845 strict** candidates.
3. **Stage 2: Semantic precision** (`scripts/phase2_embed.py` + `phase2_classify.py`): embedded candidates with **SBERT (all-MiniLM-L6-v2)** and classified each by its **nearest competing prototype** (`wedding_planning`, `fiction_roleplay`, `ecommerce_listing`, `travel_guide`, `relationship_general`, `generic_task`). Kept only wedding-nearest conversations with a strong lexical term and cosine >= 0.30.
4. **Stage 3: Validation** (`scripts/phase3_calibrate.py` + manual review): spot-checked samples across the decision boundary and inspected clusters to remove residual roleplay leakage (Doki Doki Literature Club, Minecraft). Estimated precision **~70-75%**.

**Why contrastive prototypes over keyword/NLI?** Pure keywords are high-recall/low-precision. A single weak zero-shot NLI head scored obvious false positives high (a restaurant business plan, a job email). Routing each candidate to its *nearest* category among competing prototypes cleanly diverted product listings, fiction, and travel away from the wedding bucket.

### Verifiable sample quotes

Every sample quoted in the Findings is a real conversation from `allenai/WildChat`. Below are a few representative IDs so anyone can re-fetch the original record from the source dataset.

| Finding | conversation_id (WildChat) | Snippet |
|---|---|---|
| 1 (noise: fiction) | `fda4c232ab0973384933680b02996331` | "[there would be dialogue in this story:] (MC and Sayori have been married for 9 months, and now, their first child…)" |
| 2 (vows/speeches) | `9dc42d4e88ee6be95f53170dec7bb7b8` | "write me an officiant address for a wedding…" |
| 2 (recommendation) | `793c61cd3514e63b810cf9095c1b6ec0` | "Where to buy the best ring for wedding in Portugal" |
| 2 (budget) | `fa710a149b9b58e8cb98c69f3b90b5d5` | "Write a request for money instead of wedding presents to pay towards a honeymoon… for Karl and Jane's wedding" |
| 3 (messaging) | `ae14364132456e41d9930d5e11365b8e` | "How to compose message to family member that isnt invited to wedding day and is causing drama…" |

To verify, query the original dataset on HuggingFace with `conversation_id = "<id>"`. Each ID returns exactly one matching row from `allenai/WildChat`, with the same first user message shown above.

### Detailed limitations

- **Era & model:** WildChat is 2023 GPT-3.5/4 traffic; behavior has likely shifted.
- **Corpus size:** 157 high-precision conversations supports qualitative + frequency analysis, not fine-grained statistics. A looser threshold yields ~200-270 with lower precision.
- **English-only**; non-English wedding usage not analyzed.
- **CPU-only environment:** used a light SBERT model and rule-based intents rather than heavier LLM labeling; precision ~70-75% (manual estimate), not a fully labeled benchmark.
- **User-side text only:** intents inferred from user messages, not assistant responses.

*(See Section 4 for how each limitation maps to a concrete scaling step.)*



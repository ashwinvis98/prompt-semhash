# Evaluation results

An honest, reproducible evaluation of the similarity digest on public data. Numbers are
about public datasets (HackAPrompt, JailbreakBench, HarmBench, WildJailbreak); no data
is committed to this repo. Lexical digest = `ppl1`; semantic digest = `pps1` (SimHash
over embeddings), `pps1c` when mean-centered.

**Datasets:** HackAPrompt (CC), JailbreakBench (MIT), HarmBench (MIT), WildJailbreak
(AI2, ODC-BY). **Models:** `BAAI/bge-*`, `mixedbread-ai/mxbai-embed-large-v1`,
`intfloat/multilingual-e5-large`, and the domain-tuned
`0dinai/jailbreak-embeddings-base-onnx` (multilingual-e5 fine-tuned for jailbreak /
prompt-injection duplicate detection). Runner: `fastembed` / `onnxruntime` (no torch).

> **Correction (read this first).** An earlier version of §3 reported recall@1 of
> 0.68–0.88 for the semantic digest. Those numbers were **raw cosine similarity on the
> full embedding vector** — the *ceiling* an embedding can reach — not the compact
> `pps1`/`pps1c` digest this library actually emits. Hashing the embedding down to a
> 256-bit digest costs 11–21 points of recall@1. The corrected §3 below reports both:
> the embedding ceiling **and** the shipping digest. The digest still beats the lexical
> baseline and the domain model still wins, but the honest deployable number is
> ~0.55–0.71, not 0.82–0.88.

---

## 1. Redundancy in a real corpus (lexical)

`eval/cluster_corpus.py` on HackAPrompt `user_input` (579,887 attack inputs):

| metric | value |
|---|---|
| total attack inputs | 579,887 |
| unique after normalisation | 249,484 |
| **exact-duplicate rate (full set)** | **57.0%** |
| exact-dup by model | text-davinci-003 38% · FlanT5-XXL 55% · gpt-3.5-turbo 58% |
| exact-dup by challenge level | 32% – 100% |

On a random 40k slice (exact-dup rate is sample-size dependent): 32% exact duplicates,
plus ~59% of the *unique* prompts pulled into near-duplicate clusters — an overall ~3x
collapse. Real attack corpora carry substantial redundancy, so de-duplicating by digest
materially cuts what an analyst reviews. Factor depends on the feed (HackAPrompt is a
competition, so its redundancy is on the high side).

## 2. Cross-org correlation, digests only (lexical)

`eval/*` (see `_scratch`): random 40k HackAPrompt split into two "orgs" (A, B), each
holding ~14.5k unique prompts. Sharing **only digests** (not raw prompts):

| method | Org-A prompts found in Org B |
|---|---|
| exact match (verbatim) | 12.6% |
| digest (near-duplicate) | **55.0%** (4.4x) |

The digest finds **42 percentage points** more cross-org overlap than exact matching —
reworded variants exact matching misses — while exchanging only fingerprints. (Caveat:
A/B are random splits of one high-redundancy corpus, so this shows the exact-vs-digest
gap on shared source material, not a universal rate. A genuinely cross-corpus test is
future work.)

## 3. Same-attack matching (semantic) — ceiling vs shipping digest

`eval/family_recovery.py` style, on WildJailbreak `adversarial_harmful` pairs. Each row
has a `vanilla` request (~113 chars) and its jailbroken `adversarial` rewrite (~979
chars) — same intent, very different surface. Task: for each vanilla request, is its true
rewrite the top match among N candidates (recall@1)?

Four columns, from weakest to strongest correlation signal:

- **lexical (`ppl1`)** — the dependency-free baseline.
- **cosine ceiling** — raw cosine on the full float embedding. Not a digest; the upper
  bound of what the embedding could achieve if you shipped and compared full vectors.
- **digest (`pps1`)** — the 256-bit SimHash digest this library emits.
- **digest (`pps1c`)** — the same, mean-centered.

**bge-small (general model — clean out-of-distribution reference):**

| N candidates | lexical | cosine ceiling | digest `pps1` | digest `pps1c` |
|---|---|---|---|---|
| 400 | 0.460 | 0.767 | 0.560 | 0.613 |
| 1000 | 0.407 | 0.683 | 0.455 | 0.534 |

**0din (domain-tuned model):**

| N candidates | lexical | cosine ceiling | digest `pps1` | digest `pps1c` |
|---|---|---|---|---|
| 400 | 0.460 | 0.820 | 0.645 | 0.708 |
| 1000 | 0.407 | 0.759 | 0.551 | 0.580 |

Reading the numbers honestly:

- The shipping digest **beats the lexical baseline** at every pool size (e.g. 0din
  centered 0.708 vs lexical 0.460 at N=400) — semantic correlation of heavily-reworded
  attacks is real and the digest captures a large part of it.
- Hashing to 256 bits **costs 11–21 points** versus the embedding ceiling. That is the
  price of a compact, portable, privacy-preserving digest instead of shipping and
  comparing full float vectors. If you can exchange full embeddings, do — you get the
  ceiling. The digest is for when you cannot.
- The **domain-tuned model wins** at both ceiling and digest.
- **Centering (`pps1c`) helps the digest** by ~5–6 points and is essentially free.

**Caveats:** the 0din model was fine-tuned on WildJailbreak-derived data, so its numbers
are mildly in-distribution; `bge-small` (a general model, no such exposure) is the clean
reference. Recall@1 degrades as the candidate pool grows (expected for any nearest-
neighbour retrieval). These are matching rates on one dataset's paraphrase pairs, not a
detection benchmark.

## 4. Which model? (semantic, category recovery)

`eval/family_recovery.py` on JailbreakBench (10 categories) and HarmBench (6). Raw-cosine
separation (mean intra − inter) and best-F1:

| model | JBB sep | HarmBench sep |
|---|---|---|
| bge-small | 0.064 | 0.078 |
| bge-large | 0.064 | 0.078 |
| mxbai-large | 0.075 | 0.088 |
| 0din domain | 0.104 | 0.156 |

Bigger *general* models barely help — the bottleneck is that general embeddings cluster
all "harmful text" together. The **domain-tuned** model roughly doubles raw separation.
(Note: broad *category* recovery plateaus around F1 0.5 for everything; it's the wrong
task for this tool — §3 same-attack matching is the right one.)

## 5. Calibration: mean-centering

Subtracting a shared reference mean before hashing (scheme `pps1c`) removes the dominant
"all-adversarial-text" direction. Raw-cosine separation, every model roughly doubles:

| model | JBB raw → centered | HarmBench raw → centered |
|---|---|---|
| bge-small | 0.064 → 0.145 | 0.078 → 0.154 |
| bge-large | 0.064 → 0.142 | 0.078 → 0.159 |
| mxbai-large | 0.075 → 0.151 | 0.088 → 0.158 |

Robust and model-agnostic. Whitening, by contrast, destroys separation. Centering helps
both same-attack recall (§3, +5–6 points on the digest) and *thresholded* detection
(cleaner separation). It requires a *shared* mean, so centered digests use a distinct
scheme (`pps1c`) and are never compared to uncentered ones. Because comparability depends
on **both** the embedding model and the exact reference mean, a centered digest is only
meaningful alongside those identifiers — see the versioning note in the README.

## 6. Lexical digest characterisation

`eval/` parameter sweep on controlled perturbations of real prompts (ground truth known):

- **Rewording tolerance:** at 20% of words substituted, similarity ~0.37 (3-word
  shingles), ~0.51 (2-word). Smaller shingles are more robust to substitution but less
  specific.
- **By perturbation:** insert 0.49, delete 0.44, substitute 0.37, **reorder 0.004** —
  word-shingles are order-sensitive; reshuffling evades the lexical digest entirely.
- **num_perm:** mean similarity is stable ~0.38 regardless; variance falls as perms rise
  (stdev 0.137 → 0.101 from 32 → 256). The shipping default is **64** (compact digests);
  raising to 128 trades digest size for lower variance, with diminishing returns past ~128.

---

## Verdict

- **Lexical digest (`ppl1`)** — proven for near-duplicate correlation: removes >half of a
  real corpus as exact duplicates, catches typos/rewording, order-sensitive. Cheap, offline,
  dependency-free.
- **Cross-org correlation** — the digest finds ~4.4x what exact matching does on shared
  source material, exchanging only fingerprints (privacy-preserving).
- **Semantic digest (`pps1`/`pps1c`)** — recovers the majority of heavily-reworded attacks
  (0din centered ~0.71 recall@1 at N=400; clean general model ~0.61), beating the lexical
  baseline but trailing the full-embedding ceiling by 11–21 points — the cost of a compact
  portable digest. A general model already gives a usable result, so **no fine-tuning is
  required**; a domain-tuned model is a modest further gain.
- **Centering (`pps1c`)** — a robust, cheap calibration that improves both recall and
  thresholded separation.

## Reproduce

```bash
pip install -e . && pip install promptprint[fastembed]   # or [onnx] for the domain model
python eval/cluster_corpus.py hackaprompt.parquet --column user_input
python eval/family_recovery.py labelled.csv --text-col Goal --label-col Category --semantic
python eval/semantic_eval.py
```

## Credits & prior art

Datasets: HackAPrompt; JailbreakBench (Chao et al.); HarmBench (Mazeika et al.);
WildJailbreak / WildTeaming (AI2, arXiv:2406.18510). Domain model:
`0dinai/jailbreak-embeddings-base-onnx` (Mozilla 0din; a `multilingual-e5-base`
fine-tune). Runner: `fastembed` (Qdrant), `onnxruntime`.

Method builds on classic similarity hashing — `ssdeep`, `TLSH`, `sdhash`, Broder's
MinHash, and Charikar's SimHash — and on the NLP dedup / clustering line of work:
MinishLab `semhash`, SemDeDup, and the embedding-based clustering of in-the-wild
jailbreaks by Shen et al. ("Do Anything Now", arXiv:2308.03825). The contribution here
is not the hashing math but packaging it as a **portable, deterministic digest attachable
to a threat-intel observable** so independent parties correlate reworded prompt attacks
without exchanging raw text. See the README "Prior art" section.

# Evaluation results

An honest, reproducible evaluation of the similarity digest on public data. Numbers are
about public datasets (HackAPrompt, JailbreakBench, HarmBench, WildJailbreak); no data
is committed to this repo. Lexical digest = `psh1`; semantic digest = `pse1` (SimHash
over embeddings), `pse1c` when mean-centered.

**Datasets:** HackAPrompt (CC), JailbreakBench (MIT), HarmBench (MIT), WildJailbreak
(AI2, ODC-BY). **Models:** `BAAI/bge-*`, `mixedbread-ai/mxbai-embed-large-v1`,
`intfloat/multilingual-e5-large`, and the domain-tuned
`0dinai/jailbreak-embeddings-base-onnx` (multilingual-e5 fine-tuned for jailbreak /
prompt-injection duplicate detection). Runner: `fastembed` / `onnxruntime` (no torch).

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
gap, not a universal rate.)

## 3. Same-attack matching — the headline (semantic)

`eval/family_recovery.py` style, on WildJailbreak `adversarial_harmful` pairs. Each row
has a `vanilla` request (~113 chars) and its jailbroken `adversarial` rewrite (~979
chars) — same intent, very different surface. Task: for each vanilla, is its true
rewrite the top match among N candidates (recall@1)?

| N candidates | lexical | bge-small | 0din domain |
|---|---|---|---|
| 200 | 0.535 | 0.82 | 0.88 |
| 400 | 0.460 | 0.77 | 0.82 |
| 1000 | 0.407 | 0.68 | **0.76** |

The semantic digest matches an attack to its heavily-reworded jailbreak **~2x better
than lexical** at every pool size, degrading gracefully as the haystack grows. This is
the digest's core use case, and it works.

**Caveats:** the 0din model was fine-tuned on WildJailbreak-derived data, so its numbers
are mildly in-distribution; `bge-small` (a general model, no such exposure) is the clean
reference and still reaches 68–82%. Mean-centering barely changes recall@1 (its benefit
is separation/thresholding, §5), so these use the raw digest.

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

Subtracting a shared reference mean before hashing (scheme `pse1c`) removes the dominant
"all-adversarial-text" direction. Raw-cosine separation, every model roughly doubles:

| model | JBB raw → centered | HarmBench raw → centered |
|---|---|---|
| bge-small | 0.064 → 0.145 | 0.078 → 0.154 |
| bge-large | 0.064 → 0.142 | 0.078 → 0.159 |
| mxbai-large | 0.075 → 0.151 | 0.088 → 0.158 |

Robust and model-agnostic. Whitening, by contrast, destroys separation. Centering helps
*thresholded* detection (cleaner separation) more than top-1 retrieval. It requires a
*shared* mean, so centered digests use a distinct scheme (`pse1c`) and are never compared
to uncentered ones.

## 6. Lexical digest characterisation

`eval/` parameter sweep on controlled perturbations of real prompts (ground truth known):

- **Rewording tolerance:** at 20% of words substituted, similarity ~0.37 (3-word
  shingles), ~0.51 (2-word). Smaller shingles are more robust to substitution but less
  specific.
- **By perturbation:** insert 0.49, delete 0.44, substitute 0.37, **reorder 0.004** —
  word-shingles are order-sensitive; reshuffling evades the lexical digest entirely.
- **num_perm:** mean similarity stable ~0.38; variance falls as perms rise (stdev
  0.137 → 0.101 from 32 → 256). Diminishing returns past ~128.

---

## Verdict

- **Lexical digest** — proven for near-duplicate correlation: removes >half of a real
  corpus as exact duplicates, catches typos/rewording, order-sensitive. Cheap, offline.
- **Cross-org correlation** — the digest finds ~4.4x what exact matching does, sharing
  only fingerprints (privacy-preserving).
- **Semantic digest** — strong on same-attack matching (68–88% recall@1, ~2x lexical),
  best with a domain-tuned model. A general model is already good, so **no fine-tuning
  is required** for a strong result.
- **Centering** — a robust, cheap calibration that improves separation/thresholding.

## Reproduce

```bash
pip install -e . && pip install prompt-semhash[fastembed]   # or [onnx] for the domain model
python eval/cluster_corpus.py hackaprompt.parquet --column user_input
python eval/family_recovery.py labelled.csv --text-col Goal --label-col Category --semantic
python eval/semantic_eval.py
```

## Credits

Datasets: HackAPrompt; JailbreakBench (Chao et al.); HarmBench (Mazeika et al.);
WildJailbreak / WildTeaming (AI2). Domain model: `0dinai/jailbreak-embeddings`. Runner:
`fastembed` (Qdrant), `onnxruntime`. Method builds on classic similarity hashing
(`ssdeep`, `TLSH`, MinHash, SimHash).

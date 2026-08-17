# Evaluation results

An honest, reproducible evaluation of the similarity digest on public data. Numbers are
about public datasets (HackAPrompt, JailbreakBench, HarmBench, WildJailbreak); no data
is committed to this repo. Lexical digest = `ppl1`; semantic digest = `pps1` (SimHash
over embeddings), `pps1c` when mean-centered. Semantic digests encode the embedding
model id (and, when centered, a reference-mean id), so comparing mismatched models or
means fails loudly rather than returning a plausible-looking number.

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

> **Correction (0.2.0).** These numbers dropped after the 0.2.0 tokenizer fix. The
> pre-0.2.0 tokenizer matched only `[a-z0-9]+`, so **7.4%** of HackAPrompt prompts (those
> with no ASCII alphanumerics) collapsed to an identical empty digest, and a further
> ~13% with material non-Latin content were under-differentiated. That inflated the
> measured redundancy. The figures below are re-run with the Unicode tokenizer and the
> new 128-perm default (which also makes near-duplicate clustering stricter).

`eval/cluster_corpus.py` on HackAPrompt `user_input` (579,953 attack inputs):

| metric | value | (was, pre-0.2.0) |
|---|---|---|
| total attack inputs | 579,953 | 579,887 |
| unique after normalisation | 274,804 | 249,484 |
| **exact-duplicate rate (full set)** | **52.6%** | 57.0% |
| exact-dup by model | text-davinci-003 36% · FlanT5-XXL 51% · gpt-3.5-turbo 52% | 38% / 55% / 58% |
| exact-dup by challenge level | 31% – 100% | 32% – 100% |

On a **seeded random 40k slice** (seed 42; exact-dup rate is sample-size dependent):
25% exact duplicates, plus ~35% of the *unique* prompts pulled into near-duplicate
clusters — an overall **~1.8x** collapse (down from a pre-fix ~3x, both because non-Latin
prompts no longer spuriously collapse and because the 128-perm default clusters more
strictly). Real attack corpora still carry substantial redundancy, so de-duplicating by
digest materially cuts what an analyst reviews — just less dramatically than first
reported. Factor depends on the feed (HackAPrompt is a competition, so its redundancy is
on the high side).

## 2. Cross-org correlation, digests only (lexical)

`eval/*` (see `_scratch`): random 40k HackAPrompt split into two "orgs" (A, B), each
holding ~14.5k unique prompts. Sharing **only digests** (not raw prompts):

| method | Org-A prompts found in Org B |
|---|---|
| exact match (verbatim) | 12.6% |
| digest (near-duplicate) | **55.0%** (4.4x) |

The digest finds **42 percentage points** more cross-org overlap than exact matching —
reworded variants exact matching misses — while exchanging only fingerprints. (Caveats:
A/B are random splits of one high-redundancy corpus, so this shows the exact-vs-digest
gap on shared source material, not a universal rate — a genuinely cross-corpus test is
future work; and these figures predate the 0.2.0 tokenizer/`num_perm` changes, so a
re-run would shift them modestly.)

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

| N candidates | lexical | cosine ceiling | int8-quant | digest `pps1` | digest `pps1c` |
|---|---|---|---|---|---|
| 400 | 0.537 | 0.767 | 0.767 | 0.560 | 0.613 |
| 1000 | 0.477 | 0.683 | 0.684 | 0.455 | 0.534 |

**0din (domain-tuned model):**

| N candidates | lexical | cosine ceiling | int8-quant | digest `pps1` | digest `pps1c` |
|---|---|---|---|---|---|
| 400 | 0.537 | 0.820 | 0.820 | 0.645 | 0.708 |
| 1000 | 0.477 | 0.759 | 0.760 | 0.551 | 0.580 |

(Lexical here uses the 128-perm default, hence higher than the ~0.46 reported for 64-perm
in earlier drafts. `int8-quant` = per-vector 8-bit quantised embedding, cosine-compared.)

Reading the numbers honestly:

- The shipping digest **beats the lexical baseline** at every pool size (e.g. 0din
  centered 0.708 vs lexical 0.460 at N=400) — semantic correlation of heavily-reworded
  attacks is real and the digest captures a large part of it.
- Hashing to 256 bits **costs 11–21 points** versus the embedding ceiling. That is the
  price of a 32-byte digest instead of shipping and comparing full float vectors.
- **An 8-bit quantised embedding keeps essentially the whole ceiling** (0.767 vs 0.767,
  0.820 vs 0.820) at ~384 bytes/vector. So the honest tradeoff is a size/fidelity curve,
  not "digest vs nothing": full embedding (≈1.5 KB, ceiling) → int8-quant (≈384 B,
  ~ceiling) → SimHash `pps1`/`pps1c` (32 B, −11–21 pts). Pick by your byte budget and how
  much you care about not shipping something near-invertible to an embedding.
- The **domain-tuned model wins** at both ceiling and digest.
- **Centering (`pps1c`) helps the digest** by ~5–6 points and is essentially free.

**What the hashing buys, measured (the "why SimHash and not a quantised embedding?"
question).** If you can exchange ~384 bytes per prompt, ship the int8-quantised embedding —
it is within noise of the ceiling and far ahead of the SimHash digest. The 256-bit
`pps1`/`pps1c` digest earns its place only where a 32-byte fingerprint matters, or where
you specifically want a lossy bit-signature rather than a near-recoverable embedding on the
wire. `promptprint` ships the lexical `ppl1` (dependency-free) and the SimHash `pps1`
digests; a quantised-embedding scheme is a sensible future addition for the mid-size point
on that curve, now that we've measured it's worth having.

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
  (stdev 0.137 → 0.101 from 32 → 256). The shipping default is **128** (lower-variance
  estimates); 64 halves the digest size at the cost of higher variance, with diminishing
  returns past ~128.

---

## Verdict

- **Lexical digest (`ppl1`)** — proven for near-duplicate correlation: removes >half of a
  real corpus as exact duplicates, catches typos/rewording, order-sensitive. Cheap, offline,
  dependency-free.
- **Cross-org correlation** — the digest finds ~4.4x what exact matching does on shared
  source material, exchanging only digests rather than raw prompt text (data
  minimisation, not a formal privacy guarantee — see the README).
- **Semantic digest (`pps1`/`pps1c`)** — recovers the majority of heavily-reworded attacks
  (0din centered ~0.71 recall@1 at N=400; clean general model ~0.61), beating the lexical
  baseline but trailing the full-embedding ceiling by 11–21 points — the cost of a 32-byte
  portable digest. A general model already gives a usable result, so **no fine-tuning is
  required**; a domain-tuned model is a modest further gain.
- **Size/fidelity is a curve, not a cliff.** An int8-quantised embedding (~384 B) holds the
  ceiling; the 256-bit SimHash digest (32 B) trades 11–21 points for compactness. Choose by
  byte budget — a quantised-embedding scheme is worth adding for the middle of that curve.
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

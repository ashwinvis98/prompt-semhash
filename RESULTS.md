# Evaluation results

An honest, reproducible evaluation of the similarity digest on public data. Numbers
below are about public datasets (HackAPrompt, JailbreakBench); no data is committed to
this repo. The lexical digest is `psh1`; the semantic digest is `pse1` over fastembed
embeddings (`BAAI/bge-small-en-v1.5`, and `bge-base` where noted).

## 1. Redundancy in a real corpus (lexical)

`eval/cluster_corpus.py` on a 20,000-prompt sample of HackAPrompt `user_input`:

| metric | value |
|---|---|
| total prompts | 20,000 |
| unique after normalisation | 3,132 (15.7%) |
| **exact-duplicate rate** | **84.3%** |
| near-duplicate clusters (LSH, 16 bands) | 2,026 |
| unique prompts pulled into multi-member clusters | 46.5% |
| largest cluster | 364 variants of an "I have been pwned" injection |

Net: **20,000 attempts collapse to ~2,026 distinct behaviours (~10x)**. The digest
groups reworded variants — including a `grammar`/`grammer` typo — that exact matching
treats as unrelated. (HackAPrompt is a competition, so redundancy is unusually high;
this shows the mechanism works, not that every feed is 84% duplicates.)

## 2. Paraphrase fixtures (lexical vs semantic)

`eval/semantic_eval.py` on the bundled fixtures (separation = mean intra − inter;
higher is better):

| fixture | lexical separation | semantic separation |
|---|---|---|
| Lexical families (reworded) | 0.685 (F1 1.00) | 0.306 |
| Semantic families (same intent, different words) | 0.000 | 0.054 |

The lexical digest is best at reworded near-duplicates and fails entirely on
paraphrase (0.000). The semantic digest lifts paraphrase intra-similarity from ~0 to
~0.73, but its *separation* is small — the raw-embedding ceiling for these fixtures is
only 0.092, because both fixture families are jailbreak-flavoured and sit close
together in embedding space.

## 3. Family recovery on distinct intents (JailbreakBench)

`eval/family_recovery.py` on JailbreakBench's 100 harmful goals across 10 distinct
categories (Malware, Harassment, Privacy, Physical harm, …):

| digest | intra | inter | separation | best F1 |
|---|---|---|---|---|
| lexical | 0.004 | 0.002 | 0.003 | 0.167 |
| semantic (bge-small) | 0.696 | 0.668 | 0.028 | 0.276 |
| semantic (bge-base) | 0.704 | 0.677 | 0.026 | 0.246 |

Here the lexical digest is useless — plainly-worded distinct goals share almost no word
shingles. The semantic digest is clearly better (F1 0.276 vs 0.167), confirming it
"sees" meaning the lexical one cannot. But absolute separation is still weak, and a
**larger model does not help** (bge-base ≈ bge-small): general embedding models place
all harmful-request text close together (inter ≈ 0.67), which caps category separation.

## Verdict

- **Lexical digest — proven value.** Strong, cheap, dependency-free near-duplicate
  correlation. On real data it collapses reworded attacks ~10x. This is a genuine,
  usable capability today.
- **Semantic digest — works, but not yet strong.** It consistently beats the lexical
  digest when wording differs, so the mechanism is real. With off-the-shelf general
  embeddings, though, separation is weak because such models cluster all adversarial /
  harmful text together, and a bigger model does not fix it.

## What would make the semantic digest strong

- A **domain-adapted embedding** (fine-tuned on prompt attacks) rather than a general
  model, so distinct attack intents spread out instead of clustering.
- **Calibration** — centering/whitening embeddings before hashing to spread the
  similarity distribution.
- A **cleaner ground truth** — pairs that are the *same attack reworded* (not just the
  same broad category), which is the case correlation actually targets.

## Reproduce

```bash
pip install -e . && pip install prompt-semhash[fastembed]
python eval/cluster_corpus.py hackaprompt.parquet --column user_input --limit 20000
python eval/semantic_eval.py
python eval/family_recovery.py jbb.csv --text-col Goal --label-col Category --semantic
```

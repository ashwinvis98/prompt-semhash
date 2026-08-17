"""Evaluation harness for the similarity digest.

Given a labelled set of prompts — ``(prompt_text, family_label)`` pairs — this measures
whether the digest can tell same-family pairs from different-family pairs:

- :func:`pair_similarities` splits all pairs into intra-family and inter-family
  (computing each digest once).
- :func:`summary` reports the two distributions (a good digest has intra >> inter).
- :func:`threshold_metrics` treats "same family" as a binary prediction at a similarity
  threshold and reports precision / recall / F1.
- :func:`best_threshold` sweeps thresholds and returns the best-F1 operating point.

The ``*_from_sims`` helpers work on precomputed similarity lists, so a caller with an
expensive ``digest_fn`` (an embedding model) computes digests once and then sweeps
thresholds for free.

Every function takes an optional ``digest_fn`` / ``sim_fn`` so the *same* harness works
for the lexical digest (default) and the semantic digest — pass
``digest_fn=hasher.digest, sim_fn=semantic_similarity``.
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean

from .digest import digest as _lexical_digest
from .digest import similarity as _lexical_similarity


def _digested(labelled, digest_fn):
    return [(label, digest_fn(text)) for text, label in labelled]


def pair_similarities(labelled, digest_fn=_lexical_digest, sim_fn=_lexical_similarity):
    """Return ``(intra_family_sims, inter_family_sims)`` over all unordered pairs."""
    digs = _digested(labelled, digest_fn)
    intra: list[float] = []
    inter: list[float] = []
    for (label_a, dig_a), (label_b, dig_b) in combinations(digs, 2):
        sim = sim_fn(dig_a, dig_b)
        (intra if label_a == label_b else inter).append(sim)
    return intra, inter


def summary(labelled, digest_fn=_lexical_digest, sim_fn=_lexical_similarity) -> dict:
    """Report the intra- vs inter-family similarity distributions."""
    intra, inter = pair_similarities(labelled, digest_fn, sim_fn)
    return summary_from_sims(intra, inter, n_items=len(labelled))


def summary_from_sims(intra, inter, n_items: int | None = None) -> dict:
    mi = mean(intra) if intra else 0.0
    me = mean(inter) if inter else 0.0
    return {
        "n_items": n_items,
        "n_pairs_intra": len(intra),
        "n_pairs_inter": len(inter),
        "mean_intra": mi,
        "mean_inter": me,
        "separation": mi - me,
    }


def metrics_from_sims(intra, inter, threshold: float) -> dict:
    """Precision/recall/F1 from precomputed intra/inter similarity lists."""
    tp = sum(1 for s in intra if s >= threshold)
    fn = len(intra) - tp
    fp = sum(1 for s in inter if s >= threshold)
    tn = len(inter) - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold_from_sims(intra, inter, steps: int = 100) -> dict:
    """Sweep thresholds over precomputed similarity lists; return the best-F1 metrics."""
    best = metrics_from_sims(intra, inter, 0.0)
    for i in range(1, steps + 1):
        m = metrics_from_sims(intra, inter, i / steps)
        if m["f1"] > best["f1"]:
            best = m
    return best


def threshold_metrics(
    labelled, threshold, digest_fn=_lexical_digest, sim_fn=_lexical_similarity
) -> dict:
    """Precision/recall/F1 of predicting "same family" when similarity >= *threshold*."""
    intra, inter = pair_similarities(labelled, digest_fn, sim_fn)
    return metrics_from_sims(intra, inter, threshold)


def best_threshold(
    labelled, steps: int = 100, digest_fn=_lexical_digest, sim_fn=_lexical_similarity
) -> dict:
    """Compute digests once, then sweep thresholds for the best-F1 operating point."""
    intra, inter = pair_similarities(labelled, digest_fn, sim_fn)
    return best_threshold_from_sims(intra, inter, steps)

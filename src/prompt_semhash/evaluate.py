"""Evaluation harness for the similarity digest.

Given a labelled set of prompts — ``(prompt_text, family_label)`` pairs — this measures
whether the digest can tell same-family pairs from different-family pairs:

- :func:`pair_similarities` splits all pairs into intra-family and inter-family.
- :func:`summary` reports the two distributions (a good digest has intra >> inter).
- :func:`threshold_metrics` treats "same family" as a binary prediction at a similarity
  threshold and reports precision / recall / F1.
- :func:`best_threshold` sweeps thresholds and returns the best-F1 operating point.

The harness is metric-only and corpus-agnostic: point it at the bundled fixture, or at
any list of labelled prompts (e.g. a public corpus loaded from CSV).
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean

from .digest import digest, similarity


def _digested(labelled: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return ``(family_label, digest)`` for each ``(text, family_label)`` pair."""
    return [(label, digest(text)) for text, label in labelled]


def pair_similarities(labelled: list[tuple[str, str]]) -> tuple[list[float], list[float]]:
    """Return ``(intra_family_sims, inter_family_sims)`` over all unordered pairs."""
    digs = _digested(labelled)
    intra: list[float] = []
    inter: list[float] = []
    for (label_a, dig_a), (label_b, dig_b) in combinations(digs, 2):
        sim = similarity(dig_a, dig_b)
        (intra if label_a == label_b else inter).append(sim)
    return intra, inter


def summary(labelled: list[tuple[str, str]]) -> dict:
    """Report the intra- vs inter-family similarity distributions."""
    intra, inter = pair_similarities(labelled)
    return {
        "n_items": len(labelled),
        "n_pairs_intra": len(intra),
        "n_pairs_inter": len(inter),
        "mean_intra": mean(intra) if intra else 0.0,
        "mean_inter": mean(inter) if inter else 0.0,
        "separation": (mean(intra) if intra else 0.0) - (mean(inter) if inter else 0.0),
    }


def threshold_metrics(labelled: list[tuple[str, str]], threshold: float) -> dict:
    """Precision/recall/F1 of predicting "same family" when similarity >= *threshold*."""
    intra, inter = pair_similarities(labelled)
    tp = sum(1 for s in intra if s >= threshold)
    fn = sum(1 for s in intra if s < threshold)
    fp = sum(1 for s in inter if s >= threshold)
    tn = sum(1 for s in inter if s < threshold)
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


def best_threshold(labelled: list[tuple[str, str]], steps: int = 100) -> dict:
    """Sweep thresholds in ``[0, 1]`` and return the metrics dict with the highest F1."""
    best = threshold_metrics(labelled, 0.0)
    for i in range(1, steps + 1):
        metrics = threshold_metrics(labelled, i / steps)
        if metrics["f1"] > best["f1"]:
            best = metrics
    return best

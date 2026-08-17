"""Family-recovery evaluation: does the digest group same-family prompts?

Reads a labelled CSV (``--text-col``, ``--label-col``) and reports, for the lexical
digest and (with ``--semantic``) an embedding digest, the mean intra-family vs
inter-family similarity and the best-F1 threshold for predicting "same family".

    python eval/family_recovery.py data.csv --text-col Goal --label-col Category \
        --semantic --model BAAI/bge-base-en-v1.5

A digest that recovers families well has intra >> inter and a high best F1.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from promptlsh.evaluate import (  # noqa: E402
    best_threshold_from_sims,
    pair_similarities,
    summary_from_sims,
)


def load(path: str, text_col: str | None, label_col: str | None, limit: int | None):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        tc = text_col or reader.fieldnames[0]
        lc = label_col or reader.fieldnames[1]
        for row in reader:
            text = (row.get(tc) or "").strip()
            label = (row.get(lc) or "").strip()
            if text and label:
                rows.append((text, label))
    return rows[:limit] if limit else rows


def _report(tag: str, intra, inter) -> None:
    s = summary_from_sims(intra, inter)
    b = best_threshold_from_sims(intra, inter)
    print(
        f"  {tag:10s} intra {s['mean_intra']:.3f}  inter {s['mean_inter']:.3f}  "
        f"separation {s['separation']:.3f}  bestF1 {b['f1']:.3f} @ {b['threshold']:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    data = load(args.path, args.text_col, args.label_col, args.limit)
    n_families = len({label for _, label in data})
    print(f"{len(data)} prompts, {n_families} families")

    intra, inter = pair_similarities(data)
    _report("lexical", intra, inter)

    if args.semantic:
        from promptlsh.backends import fastembed_hasher
        from promptlsh.embedding import semantic_similarity

        hasher = fastembed_hasher(args.model)
        s_intra, s_inter = pair_similarities(
            data, digest_fn=hasher.digest, sim_fn=semantic_similarity
        )
        _report("semantic", s_intra, s_inter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

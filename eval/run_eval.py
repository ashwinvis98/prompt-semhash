"""Run the evaluation harness on the bundled fixtures (or an external CSV).

    python eval/run_eval.py
    python eval/run_eval.py --corpus my_prompts.csv     # columns: text,label

The report shows, for each labelled set, the mean intra- vs inter-family similarity
and the best-F1 threshold for predicting "same family".
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from promptprint.evaluate import best_threshold, summary  # noqa: E402

from fixtures import LEXICAL_FAMILIES, SEMANTIC_FAMILIES  # noqa: E402


def _report(name: str, labelled: list[tuple[str, str]]) -> None:
    s = summary(labelled)
    best = best_threshold(labelled)
    print(f"\n## {name}  ({s['n_items']} prompts)")
    print(f"  mean intra-family similarity : {s['mean_intra']:.3f}")
    print(f"  mean inter-family similarity : {s['mean_inter']:.3f}")
    print(f"  separation (intra - inter)   : {s['separation']:.3f}")
    print(
        f"  best threshold F1 = {best['f1']:.3f} "
        f"(threshold={best['threshold']:.2f}, precision={best['precision']:.3f}, "
        f"recall={best['recall']:.3f})"
    )


def _load_csv(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [(row["text"], row["label"]) for row in reader]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="CSV with columns text,label")
    args = parser.parse_args(argv)

    if args.corpus:
        _report(os.path.basename(args.corpus), _load_csv(args.corpus))
    else:
        _report("LEXICAL families (same wording, lightly tweaked)", LEXICAL_FAMILIES)
        _report("SEMANTIC families (same intent, different words)", SEMANTIC_FAMILIES)
        print(
            "\nReading: the lexical digest separates reworded near-duplicates well, but "
            "collapses on semantic paraphrase (intra ~ inter). That gap is what the "
            "embedding-based digest targets."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

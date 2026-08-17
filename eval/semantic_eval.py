"""Compare the lexical vs semantic digest on the labelled fixtures.

Runs the same evaluation harness with the lexical digest and with the fastembed-backed
semantic digest, on both the lexical-variant families and the semantic-paraphrase
families. Shows where each digest separates families and where it does not.

    pip install promptlsh[fastembed]
    python eval/semantic_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import LEXICAL_FAMILIES, SEMANTIC_FAMILIES  # noqa: E402

from promptlsh.backends import fastembed_hasher  # noqa: E402
from promptlsh.embedding import semantic_similarity  # noqa: E402
from promptlsh.evaluate import summary  # noqa: E402


def _report(name: str, data, hasher) -> None:
    lex = summary(data)
    sem = summary(data, digest_fn=hasher.digest, sim_fn=semantic_similarity)
    print(f"\n## {name}")
    print(f"  lexical  : intra {lex['mean_intra']:.3f}  inter {lex['mean_inter']:.3f}  "
          f"separation {lex['separation']:.3f}")
    print(f"  semantic : intra {sem['mean_intra']:.3f}  inter {sem['mean_inter']:.3f}  "
          f"separation {sem['separation']:.3f}")


def main() -> int:
    hasher = fastembed_hasher()
    _report("LEXICAL families (same wording, lightly tweaked)", LEXICAL_FAMILIES, hasher)
    _report("SEMANTIC families (same intent, different words)", SEMANTIC_FAMILIES, hasher)
    print(
        "\nHigher separation = better at telling same-family from different-family. "
        "The semantic digest should hold up on the SEMANTIC families where the lexical "
        "one collapses to ~0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

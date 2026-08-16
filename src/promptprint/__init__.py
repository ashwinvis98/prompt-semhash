"""promptprint: a similarity digest (fingerprint) for adversarial prompts.

Two digest schemes share one compare interface:

- ``ppl1`` — lexical MinHash over word-shingles (``digest``, dependency-free).
- ``pps1`` — semantic SimHash over an embedding (``semantic_digest``, optional extra);
  ``pps1c`` is the mean-centered variant.

Use :func:`compare` to score two digests of the *same* scheme without caring which.
"""

from .digest import (
    SemHasher,
    digest,
    normalize,
    parse_digest,
    similarity,
    similarity_text,
)
from .embedding import (
    SemanticHasher,
    fit_reference_mean,
    parse_semantic_digest,
    semantic_digest,
    semantic_similarity,
)

__version__ = "0.1.0"


def compare(digest_a: str, digest_b: str) -> float:
    """Score two digests of the same scheme (``ppl1`` lexical or ``pps1`` semantic)."""
    scheme_a = digest_a.split(":", 1)[0]
    scheme_b = digest_b.split(":", 1)[0]
    if scheme_a != scheme_b:
        raise ValueError(f"digests use different schemes: {scheme_a!r} vs {scheme_b!r}")
    if scheme_a == "ppl1":
        return similarity(digest_a, digest_b)
    if scheme_a in ("pps1", "pps1c"):
        return semantic_similarity(digest_a, digest_b)
    raise ValueError(f"unknown digest scheme: {scheme_a!r}")


__all__ = [
    "SemHasher",
    "SemanticHasher",
    "__version__",
    "compare",
    "digest",
    "fit_reference_mean",
    "normalize",
    "parse_digest",
    "parse_semantic_digest",
    "semantic_digest",
    "semantic_similarity",
    "similarity",
    "similarity_text",
]

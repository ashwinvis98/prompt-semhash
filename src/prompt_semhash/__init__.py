"""prompt-semhash: a similarity digest for adversarial prompts.

Two digest schemes share one compare interface:

- ``psh1`` — lexical MinHash over word-shingles (``digest``, dependency-free).
- ``pse1`` — semantic SimHash over an embedding (``semantic_digest``, optional extra).

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
    parse_semantic_digest,
    semantic_digest,
    semantic_similarity,
)

__version__ = "0.2.0"


def compare(digest_a: str, digest_b: str) -> float:
    """Score two digests of the same scheme (``psh1`` lexical or ``pse1`` semantic)."""
    scheme_a = digest_a.split(":", 1)[0]
    scheme_b = digest_b.split(":", 1)[0]
    if scheme_a != scheme_b:
        raise ValueError(f"digests use different schemes: {scheme_a!r} vs {scheme_b!r}")
    if scheme_a == "psh1":
        return similarity(digest_a, digest_b)
    if scheme_a == "pse1":
        return semantic_similarity(digest_a, digest_b)
    raise ValueError(f"unknown digest scheme: {scheme_a!r}")


__all__ = [
    "SemHasher",
    "digest",
    "normalize",
    "parse_digest",
    "similarity",
    "similarity_text",
    "SemanticHasher",
    "semantic_digest",
    "semantic_similarity",
    "parse_semantic_digest",
    "compare",
    "__version__",
]

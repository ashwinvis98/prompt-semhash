"""A lexical similarity digest for adversarial prompts.

``prompt-semhash`` computes a MinHash signature over word-shingles so that
near-duplicate prompts produce *comparable* digests: two prompts that share phrasing
land close together, and a platform can cluster them without re-reading the raw text.

This is the **lexical baseline**. It catches rewording that preserves shared word
sequences (the common case for copy-paste-and-tweak jailbreaks); it does **not**
capture full semantic paraphrase where the wording changes but the meaning does not.
That is the embedding-based direction described in the README, and it is an open
problem rather than a solved one. This module is deliberately dependency-free so the
baseline is trivial to run and audit.

Digest format (``psh1``):

    psh1:<num_perm>:<hex>:<hex>:...:<hex>

where each ``<hex>`` is one 32-bit MinHash slot. Two digests of equal length are
compared slot-by-slot; the fraction of matching slots estimates the Jaccard
similarity of the two prompts' shingle sets.
"""

from __future__ import annotations

import random
import re
from hashlib import blake2b

_MERSENNE = (1 << 61) - 1          # large prime for the (a*h + b) mod p permutation
_MAX_HASH = 1 << 32                # slots are reduced to 32 bits for compact serialization
_DEFAULT_NUM_PERM = 64
_DEFAULT_SHINGLE = 3
_SCHEME = "psh1"

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> list[str]:
    """Lowercase and tokenise to alphanumeric words. Punctuation and case are dropped."""
    return _WORD_RE.findall((text or "").lower())


def _shingles(tokens: list[str], k: int) -> set[str]:
    """Return the set of k-word shingles, falling back to the whole token span if short."""
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _base_hash(shingle: str) -> int:
    return int.from_bytes(blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


class SemHasher:
    """Deterministic MinHash digester.

    The permutation coefficients are seeded from a fixed value so that any two
    installations produce **identical** digests for the same text — a requirement
    for cross-instance correlation.
    """

    def __init__(
        self,
        num_perm: int = _DEFAULT_NUM_PERM,
        shingle_size: int = _DEFAULT_SHINGLE,
        seed: int = 1,
    ) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def signature(self, text: str) -> list[int]:
        """Return the MinHash signature (a list of ``num_perm`` 32-bit ints)."""
        shingles = _shingles(normalize(text), self.shingle_size)
        if not shingles:
            return [0] * self.num_perm
        mins = [_MAX_HASH - 1] * self.num_perm
        for shingle in shingles:
            h = _base_hash(shingle)
            for i in range(self.num_perm):
                v = ((self._a[i] * h + self._b[i]) % _MERSENNE) % _MAX_HASH
                if v < mins[i]:
                    mins[i] = v
        return mins

    def digest(self, text: str) -> str:
        """Return the serialised ``psh1`` digest string for *text*."""
        sig = self.signature(text)
        return f"{_SCHEME}:{self.num_perm}:" + ":".join(format(v, "08x") for v in sig)


def parse_digest(digest_str: str) -> list[int]:
    """Parse a ``psh1`` digest back into its list of slot values."""
    parts = digest_str.split(":")
    if len(parts) < 3 or parts[0] != _SCHEME:
        raise ValueError(f"not a {_SCHEME} digest: {digest_str!r}")
    return [int(x, 16) for x in parts[2:]]


def similarity(digest_a: str, digest_b: str) -> float:
    """Estimate Jaccard similarity (0.0–1.0) from two ``psh1`` digests."""
    sa, sb = parse_digest(digest_a), parse_digest(digest_b)
    if len(sa) != len(sb):
        raise ValueError("digest length mismatch (different num_perm)")
    if not sa:
        return 0.0
    return sum(1 for x, y in zip(sa, sb) if x == y) / len(sa)


_default = SemHasher()


def digest(text: str) -> str:
    """Compute the default ``psh1`` digest for *text*."""
    return _default.digest(text)


def similarity_text(text_a: str, text_b: str) -> float:
    """Convenience: estimate similarity directly from two prompt strings."""
    return similarity(_default.digest(text_a), _default.digest(text_b))

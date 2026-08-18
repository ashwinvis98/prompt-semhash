"""A lexical similarity digest for adversarial prompts.

``promptlsh`` computes a MinHash signature over shingles so that near-duplicate
prompts produce *comparable* digests: two prompts that share phrasing land close
together, and a platform can cluster them without re-reading the raw text.

This is the **lexical baseline** (``plm1`` = promptlsh lexical v1). It catches
rewording that preserves shared word sequences (the common case for
copy-paste-and-tweak jailbreaks); it does **not** capture full semantic paraphrase
where the wording changes but the meaning does not. That is the embedding-based
direction (``pls1``, see ``embedding.py``). This module is deliberately
dependency-free so the baseline is trivial to run and audit.

Tokenisation is Unicode-aware (``\\w+`` with case folding), so non-Latin scripts
(CJK, Cyrillic, Arabic, Devanagari) are tokenised rather than stripped. Input with no
word characters at all (emoji-only, punctuation-only) falls back to character n-grams,
so distinct inputs always get distinct, non-empty shingle sets. An all-zero signature
can therefore only come from genuinely empty input, and :func:`similarity` treats it
as non-comparable (score 0.0) rather than letting empties collide at 1.0.

Digest format (``plm1``):

    plm1:<num_perm>:<hex>:<hex>:...:<hex>

where each ``<hex>`` is one 32-bit MinHash slot. Two digests of equal length are
compared slot-by-slot; the fraction of matching slots estimates the Jaccard
similarity of the two prompts' shingle sets.
"""

from __future__ import annotations

import re
from hashlib import blake2b

_MERSENNE = (1 << 61) - 1          # large prime for the (a*h + b) mod p permutation
_MAX_HASH = 1 << 32                # slots are reduced to 32 bits for compact serialization
_DEFAULT_NUM_PERM = 128
_DEFAULT_SHINGLE = 3
# Character n-grams (for unsegmented scripts + emoji/punctuation-only input) use bigrams:
# CJK "words" are typically 1-2 characters, so bigrams are the standard granularity and
# retain materially more near-duplicate signal than 3-grams (on a reworded zh
# prompt-injection pair: bigram ~0.43 vs trigram ~0.27; exact Jaccard 0.45 vs 0.30).
_CHAR_SHINGLE = 2
_SCHEME = "plm1"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Scripts with no whitespace word boundaries (CJK ideographs, Japanese kana, Thai).
# ``\w+`` swallows a whole sentence in these into one or two tokens, so word-shingling
# degenerates into an exact-match hash and near-duplicates score 0. Text containing these
# is shingled at the **character** level instead, which restores near-duplicate sensitivity.
_UNSEGMENTED_RE = re.compile(
    "[\u3040-\u30ff"   # Hiragana + Katakana
    "\u3400-\u4dbf"    # CJK Extension A
    "\u4e00-\u9fff"    # CJK Unified Ideographs
    "\uf900-\ufaff"    # CJK Compatibility Ideographs
    "\u0e00-\u0e7f]"   # Thai
)


def normalize(text: str) -> list[str]:
    """Case-fold and tokenise to Unicode word tokens. Punctuation and case are dropped.

    ``\\w+`` matches alphanumerics across scripts (Latin, Cyrillic, Arabic, CJK,
    Devanagari, ...), so non-Latin text is tokenised rather than discarded.
    """
    return _WORD_RE.findall((text or "").casefold())


def _char_shingles(text: str) -> set[str]:
    """Character n-grams over the case-folded, whitespace-stripped text."""
    chars = "".join((text or "").casefold().split())
    if not chars:
        return set()
    if len(chars) < _CHAR_SHINGLE:
        return {chars}
    return {chars[i : i + _CHAR_SHINGLE] for i in range(len(chars) - _CHAR_SHINGLE + 1)}


def _shingles(text: str, k: int) -> set[str]:
    """Return the shingle set for *text*.

    - Word k-shingles for whitespace-segmented scripts (Latin, Cyrillic, Arabic, ...).
    - **Character** n-grams for unsegmented scripts (CJK, Japanese, Thai), where a whole
      sentence is one ``\\w+`` token and word-shingling would collapse to an exact-match
      hash — so near-duplicates would score 0. Character n-grams restore near-dup
      sensitivity there.
    - Character n-grams also cover input with no word tokens at all (emoji-only,
      punctuation-only), so distinct inputs never share an empty shingle set.
    """
    if _UNSEGMENTED_RE.search(text or ""):
        return _char_shingles(text)
    tokens = normalize(text)
    if len(tokens) >= k:
        return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}
    if tokens:
        return {" ".join(tokens)}
    return _char_shingles(text)


def _base_hash(shingle: str) -> int:
    return int.from_bytes(blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


def _coeff(seed: int, salt: str, i: int, *, nonzero: bool) -> int:
    """Derive a permutation coefficient deterministically from a cryptographic hash.

    Using blake2b over ``(seed, salt, index)`` rather than the ``random`` module makes
    the coefficients — and therefore every digest — reproducible across *any* Python
    version. ``random.randrange``/``gauss`` carry no cross-version stability guarantee;
    blake2b does.
    """
    h = int.from_bytes(
        blake2b(f"{seed}:{salt}:{i}".encode("utf-8"), digest_size=8).digest(), "big"
    )
    return (h % (_MERSENNE - 1)) + 1 if nonzero else h % _MERSENNE


class LexicalHasher:
    """Deterministic MinHash digester.

    Permutation coefficients are derived from a fixed seed via blake2b, so any two
    installations — on any Python version — produce **identical** digests for the same
    text. That reproducibility is a hard requirement for cross-instance correlation.
    """

    def __init__(
        self,
        num_perm: int = _DEFAULT_NUM_PERM,
        shingle_size: int = _DEFAULT_SHINGLE,
        seed: int = 1,
    ) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.seed = seed
        self._a = [_coeff(seed, "a", i, nonzero=True) for i in range(num_perm)]
        self._b = [_coeff(seed, "b", i, nonzero=False) for i in range(num_perm)]

    def signature(self, text: str) -> list[int]:
        """Return the MinHash signature (a list of ``num_perm`` 32-bit ints)."""
        shingles = _shingles(text, self.shingle_size)
        if not shingles:
            return [0] * self.num_perm
        mins = [_MAX_HASH - 1] * self.num_perm
        for shingle in shingles:
            h = _base_hash(shingle)
            for i in range(self.num_perm):
                v = ((self._a[i] * h + self._b[i]) % _MERSENNE) % _MAX_HASH
                mins[i] = min(mins[i], v)
        return mins

    def digest(self, text: str) -> str:
        """Return the serialised ``plm1`` digest string for *text*."""
        sig = self.signature(text)
        return f"{_SCHEME}:{self.num_perm}:" + ":".join(format(v, "08x") for v in sig)


# Backwards-compatible alias for the pre-0.1 name. Deprecated; use LexicalHasher.
SemHasher = LexicalHasher


def parse_digest(digest_str: str) -> list[int]:
    """Parse a ``plm1`` digest back into its list of slot values.

    Validates that the declared ``num_perm`` matches the actual number of slots, so a
    truncated or malformed digest fails loudly rather than comparing as a shorter one.
    """
    parts = digest_str.split(":")
    if len(parts) < 3 or parts[0] != _SCHEME:
        raise ValueError(f"not a {_SCHEME} digest: {digest_str!r}")
    declared = int(parts[1])
    slots = [int(x, 16) for x in parts[2:]]
    if len(slots) != declared:
        raise ValueError(f"declared num_perm {declared} != {len(slots)} slots")
    return slots


def similarity(digest_a: str, digest_b: str) -> float:
    """Estimate Jaccard similarity (0.0–1.0) from two ``plm1`` digests.

    An all-zero signature only arises from genuinely empty input; it is treated as
    non-comparable (0.0) so degenerate inputs never collide at 1.0.
    """
    sa, sb = parse_digest(digest_a), parse_digest(digest_b)
    if len(sa) != len(sb):
        raise ValueError("digest length mismatch (different num_perm)")
    if not sa or not any(sa) or not any(sb):
        return 0.0
    return sum(1 for x, y in zip(sa, sb) if x == y) / len(sa)


_default = LexicalHasher()


def digest(text: str) -> str:
    """Compute the default ``plm1`` digest for *text*."""
    return _default.digest(text)


def similarity_text(text_a: str, text_b: str) -> float:
    """Convenience: estimate similarity directly from two prompt strings."""
    return similarity(_default.digest(text_a), _default.digest(text_b))

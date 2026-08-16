"""Embedding-backed semantic digest (SimHash / random-hyperplane LSH).

Where the lexical digest (`digest.py`) hashes word-shingles and therefore only sees
shared *wording*, this digest hashes an *embedding* of the prompt, so two prompts with
the same meaning but different words land close together.

The mechanism is SimHash: project the embedding onto a fixed set of random hyperplanes
and record the sign of each projection as one bit. Similar vectors (small angle) agree
on most bits; the fraction of agreeing bits estimates ``1 - angle/pi`` and is monotonic
in cosine similarity. The digest scheme is ``pse1``:

    pse1:<n_bits>:<hex>

The embedding function is **injectable**. The default lazily loads a
``sentence-transformers`` model (an optional dependency: ``pip install
prompt-semhash[semantic]``); tests and custom setups can pass any
``text -> vector`` callable, so the LSH logic has no heavy dependency of its own.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

_SCHEME = "pse1"
_DEFAULT_BITS = 256
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EmbedFn = Callable[[str], Sequence[float]]


def _bits_to_hex(bits: list[int]) -> str:
    padded = bits + [0] * ((-len(bits)) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for bit in padded[i : i + 8]:
            byte = (byte << 1) | (bit & 1)
        out.append(byte)
    return out.hex()


def _hex_to_bits(hex_str: str, n_bits: int) -> list[int]:
    bits: list[int] = []
    for byte in bytes.fromhex(hex_str):
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return bits[:n_bits]


class SemanticHasher:
    """SimHash digester over an injectable embedding function.

    Hyperplanes are seeded deterministically and built lazily once the embedding
    dimensionality is known, so two installations using the same model and seed
    produce identical digests.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        n_bits: int = _DEFAULT_BITS,
        seed: int = 1,
    ) -> None:
        self.embed_fn = embed_fn or _default_embed
        self.n_bits = n_bits
        self._seed = seed
        self._planes: list[list[float]] | None = None
        self._dim: int | None = None

    def _ensure_planes(self, dim: int) -> None:
        if self._planes is None:
            rng = random.Random(self._seed)
            self._planes = [
                [rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(self.n_bits)
            ]
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(
                f"embedding dim changed ({self._dim} -> {dim}); use one model per hasher"
            )

    def signature_bits(self, vector: Sequence[float]) -> list[int]:
        vec = list(vector)
        self._ensure_planes(len(vec))
        assert self._planes is not None
        bits: list[int] = []
        for plane in self._planes:
            dot = 0.0
            for p, v in zip(plane, vec):
                dot += p * v
            bits.append(1 if dot >= 0.0 else 0)
        return bits

    def digest(self, text: str) -> str:
        bits = self.signature_bits(self.embed_fn(text))
        return f"{_SCHEME}:{self.n_bits}:{_bits_to_hex(bits)}"


def parse_semantic_digest(digest_str: str) -> tuple[int, list[int]]:
    """Parse a ``pse1`` digest into ``(n_bits, bits)``."""
    parts = digest_str.split(":")
    if len(parts) != 3 or parts[0] != _SCHEME:
        raise ValueError(f"not a {_SCHEME} digest: {digest_str!r}")
    n_bits = int(parts[1])
    return n_bits, _hex_to_bits(parts[2], n_bits)


def semantic_similarity(digest_a: str, digest_b: str) -> float:
    """Estimate similarity (0.0-1.0) from two ``pse1`` digests via bit agreement."""
    na, ba = parse_semantic_digest(digest_a)
    nb, bb = parse_semantic_digest(digest_b)
    if na != nb:
        raise ValueError("digest length mismatch (different n_bits)")
    if na == 0:
        return 0.0
    return sum(1 for x, y in zip(ba, bb) if x == y) / na


# --- default embedding (optional dependency) -------------------------------- #

_model = None


def _default_embed(text: str) -> Sequence[float]:
    """Embed text with a sentence-transformers model (loaded lazily and cached)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "The semantic digest needs the optional 'semantic' extra. "
                "Install it with:  pip install prompt-semhash[semantic]"
            ) from exc
        _model = SentenceTransformer(_DEFAULT_MODEL)
    return _model.encode(text).tolist()


_default_hasher: SemanticHasher | None = None


def semantic_digest(text: str) -> str:
    """Compute the default ``pse1`` digest (loads the default model on first use)."""
    global _default_hasher
    if _default_hasher is None:
        _default_hasher = SemanticHasher()
    return _default_hasher.digest(text)

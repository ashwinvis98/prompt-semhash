"""Embedding-backed semantic digest (SimHash / random-hyperplane LSH).

Where the lexical digest (`digest.py`) hashes word-shingles and only sees shared
*wording*, this digest hashes an *embedding* of the prompt, so two prompts with the
same meaning but different words land close together.

The mechanism is SimHash: project the embedding onto a fixed set of random hyperplanes
and record the sign of each projection as one bit. Similar vectors agree on most bits;
the fraction of agreeing bits estimates ``1 - angle/pi`` and is monotonic in cosine.

Optional **centering**: subtracting a shared reference mean before hashing removes the
dominant "all-adversarial-text" direction and improves separation / thresholding
(see RESULTS.md). Centering requires the *same* reference mean on both sides, so
centered digests use a distinct scheme tag (``pps1c``) and are never comparable to
uncentered (``pps1``) ones.

Scheme tags: ``pps1`` = promptprint semantic v1 (raw); ``pps1c`` = the centered
variant. The embedding function is injectable (see ``backends.py`` for fastembed / ONNX).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

_BITS = 256
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

    Hyperplanes are seeded deterministically, so two installations using the same
    model, seed, and (optional) reference mean produce identical digests.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        n_bits: int = _BITS,
        seed: int = 1,
        mean: Sequence[float] | None = None,
    ) -> None:
        self.embed_fn = embed_fn or _default_embed
        self.n_bits = n_bits
        self._seed = seed
        self.mean = list(mean) if mean is not None else None
        self._planes: list[list[float]] | None = None
        self._dim: int | None = None

    @property
    def scheme(self) -> str:
        return "pps1c" if self.mean is not None else "pps1"

    def _ensure_planes(self, dim: int) -> None:
        if self._planes is None:
            rng = random.Random(self._seed)
            self._planes = [
                [rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(self.n_bits)
            ]
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(f"embedding dim changed ({self._dim} -> {dim})")

    def signature_bits(self, vector: Sequence[float]) -> list[int]:
        vec = list(vector)
        if self.mean is not None:
            if len(self.mean) != len(vec):
                raise ValueError("reference mean dim does not match embedding dim")
            vec = [v - m for v, m in zip(vec, self.mean)]
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
        return f"{self.scheme}:{self.n_bits}:{_bits_to_hex(bits)}"


def fit_reference_mean(embed_fn: EmbedFn, texts: Sequence[str]) -> list[float]:
    """Compute a reference mean embedding over a representative corpus (for centering).

    Publish/share this mean so all parties produce comparable centered digests.
    """
    vecs = [list(embed_fn(t)) for t in texts]
    if not vecs:
        raise ValueError("no texts to fit a mean")
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def parse_semantic_digest(digest_str: str) -> tuple[str, int, list[int]]:
    """Parse a ``pps1``/``pps1c`` digest into ``(scheme, n_bits, bits)``."""
    parts = digest_str.split(":")
    if len(parts) != 3 or parts[0] not in ("pps1", "pps1c"):
        raise ValueError(f"not a pps1/pps1c digest: {digest_str!r}")
    n_bits = int(parts[1])
    return parts[0], n_bits, _hex_to_bits(parts[2], n_bits)


def semantic_similarity(digest_a: str, digest_b: str) -> float:
    """Estimate similarity (0-1) from two semantic digests of the *same* scheme."""
    sa, na, ba = parse_semantic_digest(digest_a)
    sb, nb, bb = parse_semantic_digest(digest_b)
    if sa != sb:
        raise ValueError(f"digest schemes differ ({sa} vs {sb}); centered vs uncentered?")
    if na != nb:
        raise ValueError("digest length mismatch (different n_bits)")
    if na == 0:
        return 0.0
    return sum(1 for x, y in zip(ba, bb) if x == y) / na


# --- default embedding (optional dependency) -------------------------------- #

_model = None


def _default_embed(text: str) -> Sequence[float]:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The default semantic digest needs the 'semantic' extra, or use "
                "backends.fastembed_hasher / backends.onnx_hasher."
            ) from exc
        _model = SentenceTransformer(_MODEL)
    return _model.encode(text).tolist()


_default_hasher: SemanticHasher | None = None


def semantic_digest(text: str) -> str:
    """Compute the default ``pps1`` digest (loads the default model on first use)."""
    global _default_hasher
    if _default_hasher is None:
        _default_hasher = SemanticHasher()
    return _default_hasher.digest(text)

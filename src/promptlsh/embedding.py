"""Embedding-backed semantic digest (SimHash / random-hyperplane LSH).

Where the lexical digest (`digest.py`) hashes shingles and only sees shared *wording*,
this digest hashes an *embedding* of the prompt, so two prompts with the same meaning
but different words land close together.

The mechanism is SimHash: project the embedding onto a fixed set of random hyperplanes
and record the sign of each projection as one bit. Similar vectors agree on most bits;
the fraction of agreeing bits estimates ``1 - angle/pi`` and is monotonic in cosine.

The hyperplanes are derived deterministically from a cryptographic hash of the seed and
plane/dimension index (not the ``random`` module), so two installations produce
identical hyperplanes on any Python version.

**Comparability is enforced, not just documented.** A semantic digest is only meaningful
against another produced with the *same embedding model* and — for the centered variant
— the *same reference mean*. Both identities are encoded in the digest string, and
:func:`semantic_similarity` refuses to compare mismatched ones:

    pls1:<model_id>:<n_bits>:<hex>                (raw)
    pls1c:<model_id>:<ref_id>:<n_bits>:<hex>      (mean-centered)

``model_id`` is a caller-supplied label for the embedding model; ``ref_id`` is a short
hash of the reference mean, so two parties can confirm they centered on the same vector.

Optional **centering** (subtracting a shared reference mean before hashing) removes the
dominant "all-adversarial-text" direction and improves separation/thresholding (see
RESULTS.md). It uses the distinct scheme tag ``pls1c`` and is never comparable to ``pls1``.
The embedding function is injectable (see ``backends.py`` for fastembed / ONNX).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from hashlib import blake2b

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


def _det_gauss(seed: int, plane: int, dim: int) -> float:
    """A deterministic N(0,1) sample from blake2b over ``(seed, plane, dim)``.

    Box-Muller over two hash-derived uniforms. Version-independent, unlike
    ``random.gauss`` (which has no cross-version stability guarantee).
    """
    h = blake2b(f"{seed}:{plane}:{dim}".encode("utf-8"), digest_size=16).digest()
    u1 = (int.from_bytes(h[:8], "big") + 1) / (2**64 + 1)   # (0, 1), never 0
    u2 = int.from_bytes(h[8:], "big") / (2**64)             # [0, 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _sanitize_model_id(model_id: str) -> str:
    """Model ids appear in the ``:``-delimited digest, so strip delimiters/whitespace."""
    return "".join((model_id or "unknown").split()).replace(":", "_")


def _mean_ref_id(mean: Sequence[float]) -> str:
    """A short, stable id for a reference mean, so mismatched means fail loudly."""
    packed = b"".join(struct.pack("<d", float(m)) for m in mean)
    return blake2b(packed, digest_size=4).hexdigest()


class SemanticHasher:
    """SimHash digester over an injectable embedding function.

    Hyperplanes are derived deterministically from the seed, so two installations using
    the same model, seed, and (optional) reference mean produce identical digests.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        n_bits: int = _BITS,
        seed: int = 1,
        mean: Sequence[float] | None = None,
        model_id: str = "default",
    ) -> None:
        self.embed_fn = embed_fn or _default_embed
        self.n_bits = n_bits
        self._seed = seed
        self.mean = list(mean) if mean is not None else None
        self.model_id = _sanitize_model_id(model_id)
        self.ref_id = _mean_ref_id(self.mean) if self.mean is not None else None
        self._planes: list[list[float]] | None = None
        self._dim: int | None = None

    @property
    def scheme(self) -> str:
        return "pls1c" if self.mean is not None else "pls1"

    def _ensure_planes(self, dim: int) -> None:
        if self._planes is None:
            self._planes = [
                [_det_gauss(self._seed, p, d) for d in range(dim)]
                for p in range(self.n_bits)
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
        hexbits = _bits_to_hex(bits)
        if self.mean is not None:
            return f"pls1c:{self.model_id}:{self.ref_id}:{self.n_bits}:{hexbits}"
        return f"pls1:{self.model_id}:{self.n_bits}:{hexbits}"


def fit_reference_mean(embed_fn: EmbedFn, texts: Sequence[str]) -> list[float]:
    """Compute a reference mean embedding over a representative corpus (for centering).

    Publish/share this mean so all parties produce comparable centered digests; its
    ``ref_id`` (see :func:`_mean_ref_id`) then matches across parties.
    """
    vecs = [list(embed_fn(t)) for t in texts]
    if not vecs:
        raise ValueError("no texts to fit a mean")
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def parse_semantic_digest(digest_str: str) -> tuple[str, str, str | None, int, list[int]]:
    """Parse a semantic digest into ``(scheme, model_id, ref_id, n_bits, bits)``.

    ``ref_id`` is ``None`` for the uncentered ``pls1`` scheme.
    """
    parts = digest_str.split(":")
    scheme = parts[0]
    if scheme == "pls1" and len(parts) == 4:
        _, model_id, n_bits_s, hexbits = parts
        ref_id: str | None = None
    elif scheme == "pls1c" and len(parts) == 5:
        _, model_id, ref_id, n_bits_s, hexbits = parts
    else:
        raise ValueError(f"not a pls1/pls1c digest: {digest_str!r}")
    n_bits = int(n_bits_s)
    return scheme, model_id, ref_id, n_bits, _hex_to_bits(hexbits, n_bits)


def semantic_similarity(digest_a: str, digest_b: str) -> float:
    """Estimate similarity (0-1) from two semantic digests.

    Refuses to compare digests from different schemes, embedding models, reference
    means, or bit lengths — a meaningless comparison fails loudly instead of returning
    a plausible-looking number.
    """
    sa, ma, ra, na, ba = parse_semantic_digest(digest_a)
    sb, mb, rb, nb, bb = parse_semantic_digest(digest_b)
    if sa != sb:
        raise ValueError(f"digest schemes differ ({sa} vs {sb}); centered vs uncentered?")
    if ma != mb:
        raise ValueError(f"different embedding models ({ma} vs {mb}); not comparable")
    if ra != rb:
        raise ValueError(f"different reference means ({ra} vs {rb}); not comparable")
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
    """Compute the default ``pls1`` digest (loads the default model on first use)."""
    global _default_hasher
    if _default_hasher is None:
        _default_hasher = SemanticHasher(model_id=_MODEL)
    return _default_hasher.digest(text)

"""Optional embedding backends for the semantic digest.

These wire a real embedding model into :class:`SemanticHasher`. They are optional:
each imports its dependency lazily and raises a clear error if it is missing.
"""

from __future__ import annotations

from .embedding import SemanticHasher

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def fastembed_hasher(
    model_name: str = _DEFAULT_MODEL,
    n_bits: int = 256,
    seed: int = 1,
) -> SemanticHasher:
    """Return a SemanticHasher backed by fastembed (ONNX, no torch).

    Install with ``pip install prompt-semhash[fastembed]``.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fastembed is not installed. Install it with: pip install prompt-semhash[fastembed]"
        ) from exc

    model = TextEmbedding(model_name)

    def embed(text: str):
        return list(next(iter(model.embed([text]))))

    return SemanticHasher(embed_fn=embed, n_bits=n_bits, seed=seed)

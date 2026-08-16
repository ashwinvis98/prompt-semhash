"""prompt-semhash: a lexical similarity digest for adversarial prompts."""

from .digest import (
    SemHasher,
    digest,
    normalize,
    parse_digest,
    similarity,
    similarity_text,
)

__version__ = "0.1.0"

__all__ = [
    "SemHasher",
    "digest",
    "normalize",
    "parse_digest",
    "similarity",
    "similarity_text",
    "__version__",
]

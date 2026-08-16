"""Optional embedding backends for the semantic digest.

Each imports its dependency lazily and raises a clear error if it is missing.

- ``fastembed_hasher`` — general models via fastembed/ONNX (no torch).
- ``onnx_hasher`` — any sentence-transformers model exported to ONNX, e.g. a
  domain-tuned jailbreak/prompt-injection embedding. Also torch-free.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .embedding import SemanticHasher

_FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"


def fastembed_hasher(
    model_name: str = _FASTEMBED_DEFAULT,
    n_bits: int = 256,
    seed: int = 1,
    mean: Sequence[float] | None = None,
) -> SemanticHasher:
    """SemanticHasher backed by fastembed (ONNX). Install: ``promptprint[fastembed]``."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install fastembed: pip install promptprint[fastembed]") from exc
    model = TextEmbedding(model_name)

    def embed(text: str):
        return list(next(iter(model.embed([text]))))

    return SemanticHasher(embed_fn=embed, n_bits=n_bits, seed=seed, mean=mean)


def onnx_hasher(
    repo_id: str,
    n_bits: int = 256,
    seed: int = 1,
    max_length: int = 512,
    mean: Sequence[float] | None = None,
) -> SemanticHasher:
    """SemanticHasher backed by an ONNX sentence-transformers model on the HF Hub.

    Works with domain-tuned embeddings such as ``0dinai/jailbreak-embeddings-base-onnx``
    (a multilingual-e5 fine-tuned for jailbreak/prompt-injection duplicate detection).
    Requires ``onnxruntime``, ``tokenizers``, ``huggingface_hub`` (all torch-free).
    Mean-pools the last hidden state over the attention mask.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnx_hasher needs: pip install onnxruntime tokenizers huggingface_hub numpy"
        ) from exc

    local = snapshot_download(repo_id, allow_patterns=["onnx/*", "*.json"])
    tok = Tokenizer.from_file(os.path.join(local, "tokenizer.json"))
    pad = tok.token_to_id("<pad>")
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=1 if pad is None else pad, pad_token="<pad>")
    sess = ort.InferenceSession(
        os.path.join(local, "onnx", "model.onnx"), providers=["CPUExecutionProvider"]
    )
    in_names = {i.name for i in sess.get_inputs()}

    def embed(text: str):
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        outs = sess.run(None, feed)
        tok_emb = next((o for o in outs if o.ndim == 3), None)
        if tok_emb is not None:
            m = mask[:, :, None].astype(np.float32)
            pooled = (tok_emb * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        else:
            pooled = next(o for o in outs if o.ndim == 2)
        return pooled[0].tolist()

    return SemanticHasher(embed_fn=embed, n_bits=n_bits, seed=seed, mean=mean)

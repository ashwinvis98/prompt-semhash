"""Tests for the semantic (SimHash) digest.

Uses a fake, deterministic embedding function so the LSH logic is tested without the
optional sentence-transformers dependency. Runnable with pytest or directly.
"""

from promptlsh import SemanticHasher, compare, digest, semantic_similarity
from promptlsh.embedding import parse_semantic_digest

# Fixed vectors keyed by label (dim 8). No model needed.
_VEC = {
    "a": [1.0, 2.0, 3.0, -1.0, 0.5, -2.0, 1.5, 0.4],
    "a_scaled": [2.0, 4.0, 6.0, -2.0, 1.0, -4.0, 3.0, 0.8],   # 2 * a  (same direction)
    "neg_a": [-1.0, -2.0, -3.0, 1.0, -0.5, 2.0, -1.5, -0.4],  # -a     (opposite)
}

_H = SemanticHasher(embed_fn=lambda label: _VEC[label], n_bits=64, seed=7)


def test_digest_is_deterministic():
    assert _H.digest("a") == _H.digest("a")


def test_identical_vector_scores_one():
    assert semantic_similarity(_H.digest("a"), _H.digest("a")) == 1.0


def test_simhash_is_scale_invariant():
    # scaling a vector preserves every projection sign -> identical signature
    assert semantic_similarity(_H.digest("a"), _H.digest("a_scaled")) == 1.0


def test_opposite_vector_flips_all_bits():
    assert semantic_similarity(_H.digest("a"), _H.digest("neg_a")) < 0.05


def test_digest_format():
    # scheme : model_id : n_bits : hex
    assert _H.digest("a").startswith("pls1:default:64:")


def test_different_models_refuse_to_compare():
    h_bge = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7, model_id="bge")
    h_e5 = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7, model_id="e5")
    try:
        semantic_similarity(h_bge.digest("a"), h_e5.digest("a"))
    except ValueError:
        return
    raise AssertionError("expected ValueError comparing digests from different models")


def test_different_reference_means_refuse_to_compare():
    h1 = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7, mean=[0.5] * 8)
    h2 = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7, mean=[0.9] * 8)
    try:
        semantic_similarity(h1.digest("a"), h2.digest("a"))
    except ValueError:
        return
    raise AssertionError("expected ValueError comparing digests with different means")


def test_parse_rejects_lexical_digest():
    try:
        parse_semantic_digest("plm1:64:deadbeef")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a plm1 digest")


def test_compare_dispatches_by_scheme():
    # semantic vs semantic
    assert compare(_H.digest("a"), _H.digest("a")) == 1.0
    # lexical vs lexical
    assert compare(digest("hello world foo"), digest("hello world foo")) == 1.0


def test_compare_rejects_mixed_schemes():
    try:
        compare(digest("hello world foo"), _H.digest("a"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for mixed schemes")


def test_centering_uses_distinct_scheme_and_blocks_mixed_compare():
    h_raw = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7)
    h_ctr = SemanticHasher(embed_fn=lambda t: _VEC[t], n_bits=64, seed=7, mean=[0.5] * 8)
    assert h_raw.digest("a").startswith("pls1:")
    assert h_ctr.digest("a").startswith("pls1c:")
    assert semantic_similarity(h_ctr.digest("a"), h_ctr.digest("a")) == 1.0
    try:
        compare(h_raw.digest("a"), h_ctr.digest("a"))
    except ValueError:
        return
    raise AssertionError("expected ValueError when mixing pls1 and pls1c")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()

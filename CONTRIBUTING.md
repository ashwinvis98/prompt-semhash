# Contributing

Thanks for your interest. This is a small, focused library; contributions that keep it
that way are the most welcome.

## Development setup

```bash
python -m pip install -e ".[dev]"        # editable install + pytest, ruff, mypy
# optional backends for the semantic digest:
python -m pip install -e ".[fastembed]"  # ONNX, no torch
python -m pip install -e ".[onnx]"       # domain-tuned ONNX models
```

## Before opening a pull request

```bash
ruff check src tests
pytest -q
```

Both must pass. CI runs the same on Python 3.10–3.12.

## The one rule that matters: never silently change a digest

A digest is only useful if two installations, on any machine and any Python version,
produce the **same** bytes for the same input. That is what lets independent parties
correlate without a shared service.

- Any change that alters the bytes a scheme produces (`plm1`, `pls1`, `pls1c`) is a
  **breaking** change. It must bump the scheme tag (`ppl2`, ...) — never change `plm1`'s
  output in place — and update `CHANGELOG.md`.
- `tests/test_digest.py::test_digest_is_pinned_across_versions` pins a known input to its
  exact digest string. If your change makes it fail, that is the guard working: either
  your change was unintended, or it is intentional and needs a new scheme tag.
- Determinism must not depend on unstable sources: no Python `hash()` on text, and no
  `random`-module output that isn't documented stable across versions. Derive from a
  cryptographic hash instead (see `digest.py::_coeff`, `embedding.py::_det_gauss`).

## Scope

`promptlsh` is a similarity **digest**, not a detection engine or a clustering service.
Retrieval/indexing strategy belongs in the consumer (see the connector in
`adversarial-ai-cti`). Keep the lexical baseline dependency-free.

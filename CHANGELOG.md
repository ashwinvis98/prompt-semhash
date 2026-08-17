# Changelog

All notable changes to `promptprint` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Digest stability.** Before 1.0.0 the digest formats (`ppl1`, `pps1`, `pps1c`) are
> **not** guaranteed stable across releases. A digest stored under one 0.x version may
> not be comparable to one produced by another. From 1.0.0, any incompatible change to a
> digest's bytes will come with a new scheme tag (`ppl2`, ...), never a silent change.

## [0.2.0] - 2026-08-16

Review-driven correctness and interoperability fixes. **The digest bytes changed; 0.1.0
digests are not comparable to 0.2.0 digests.**

### Fixed
- **Multilingual collapse (correctness).** The lexical tokenizer matched only
  `[a-z0-9]+`, so any prompt with no ASCII alphanumerics (CJK, Cyrillic, Arabic,
  Devanagari, emoji-only) produced an all-zero digest, and all such digests compared as
  identical (1.0). Appending non-Latin script to a prompt also left its digest unchanged,
  a trivial evasion. Tokenisation is now Unicode-aware (`\w+`, case-folded) with a
  character-n-gram fallback, and an all-zero signature (only possible from empty input)
  is treated as non-comparable (0.0).
- **Cross-version determinism.** Permutation coefficients (lexical) and SimHash
  hyperplanes (semantic) were seeded via Python's `random` module, whose
  `randrange`/`gauss` carry no cross-version stability guarantee. Both are now derived
  from blake2b over the seed and index, so digests are reproducible on any Python version.

### Changed
- **Default `num_perm` 64 → 128** for lower-variance similarity estimates.
- **Semantic digest format now encodes model and reference-mean identity** and enforces
  it: `pps1:<model_id>:<n_bits>:<hex>` and `pps1c:<model_id>:<ref_id>:<n_bits>:<hex>`.
  `semantic_similarity` raises when comparing digests from different models, reference
  means, schemes, or bit-lengths, instead of returning a meaningless number.
- `parse_digest` now validates the declared `num_perm` against the actual slot count.
- The lexical hasher class `SemHasher` is renamed `LexicalHasher`; `SemHasher` remains as
  a deprecated alias.
- Documentation: privacy framing corrected to data-minimisation (not a formal guarantee).
- `RESULTS.md` §1 re-run on HackAPrompt with the fixed tokenizer: full-set exact-dup rate
  57.0% → 52.6%, random-slice collapse ~3x → ~1.8x (7.4% of prompts were previously
  all-zero-colliding). §3 gains an int8-quantised-embedding column showing quantisation
  holds the ceiling while the SimHash digest trades fidelity for a 32-byte size.

## [0.1.0] - 2026-08-15

First public release, renamed from `prompt-semhash` (which collided with the established
MinishLab `semhash` semantic-dedup library).

### Added
- Lexical MinHash digest (`ppl1`), dependency-free, with a deterministic serialised format.
- Embedding-backed semantic SimHash digest (`pps1`, `pps1c` centered) behind a shared
  `compare` interface, with fastembed / ONNX backends.
- Evaluation harness and scripts; `RESULTS.md` with a full public-data evaluation.

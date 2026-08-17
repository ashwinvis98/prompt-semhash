"""Measure redundancy in a prompt corpus by clustering with the similarity digest.

Reads prompts from a ``.txt`` (one per line), ``.csv`` (``--column``), or ``.parquet``
(``--column``; needs pandas), and reports:

- the exact-duplicate rate (after normalisation), and
- near-duplicate clusters found with MinHash-LSH banding on the lexical digest,
  plus the largest clusters with example members.

This quantifies how much of a real feed is the *same attack reworded* — the thing a
similarity digest is meant to collapse.

    python eval/cluster_corpus.py data.parquet --column user_input --limit 50000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from promptprint.digest import LexicalHasher, normalize  # noqa: E402


def load_texts(path: str, column: str | None, limit: int | None) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path, columns=[column] if column else None)
        col = column or df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
    elif ext == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            col = column or (reader.fieldnames[0] if reader.fieldnames else None)
            texts = [row[col] for row in reader if row.get(col)]
    else:
        with open(path, encoding="utf-8") as fh:
            texts = [line.rstrip("\n") for line in fh]
    texts = [t for t in (s.strip() for s in texts) if t]
    if limit:
        texts = texts[:limit]
    return texts


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def lsh_cluster(signatures: list[list[int]], bands: int) -> list[list[int]]:
    """Cluster items whose MinHash signatures collide in at least one band."""
    n = len(signatures)
    if n == 0:
        return []
    rows = max(1, len(signatures[0]) // bands)
    uf = _UnionFind(n)
    buckets: dict = defaultdict(list)
    for i, sig in enumerate(signatures):
        for b in range(bands):
            buckets[(b, tuple(sig[b * rows : (b + 1) * rows]))].append(i)
    for members in buckets.values():
        for other in members[1:]:
            uf.union(members[0], other)
    groups: dict = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--column", help="column name for csv/parquet")
    ap.add_argument("--limit", type=int, help="cap number of prompts (for speed)")
    ap.add_argument("--bands", type=int, default=16, help="LSH bands (of 64-slot signature)")
    args = ap.parse_args(argv)

    texts = load_texts(args.path, args.column, args.limit)
    total = len(texts)
    if total == 0:
        print("no prompts loaded")
        return 1

    normalised = [" ".join(normalize(t)) for t in texts]
    unique = sorted({n for n in normalised if n})
    print(f"total prompts         : {total}")
    print(f"unique (normalised)   : {len(unique)}  ({100 * len(unique) / total:.1f}%)")
    print(f"exact-duplicate rate  : {100 * (total - len(unique)) / total:.1f}%")

    hasher = LexicalHasher()
    signatures = [hasher.signature(t) for t in unique]
    clusters = lsh_cluster(signatures, args.bands)
    multi = sorted((c for c in clusters if len(c) > 1), key=len, reverse=True)
    in_multi = sum(len(c) for c in multi)

    print(f"\nnear-duplicate clustering on {len(unique)} unique prompts (LSH, {args.bands} bands):")
    print(f"  total clusters       : {len(clusters)}")
    print(f"  multi-member clusters : {len(multi)}")
    print(f"  prompts in multi      : {in_multi}  ({100 * in_multi / len(unique):.1f}% of unique)")

    for cluster in multi[:3]:
        print(f"\n  cluster of {len(cluster)} near-duplicates — examples:")
        for idx in cluster[:3]:
            snippet = unique[idx][:100]
            print(f"    - {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

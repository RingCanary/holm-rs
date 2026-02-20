#!/usr/bin/env python3
"""GMR cleaning for microGPT-style corpus prep (pure Python).

This script implements Geometric Manifold Rectification (GMR) as a
preprocessing stage before language-model training.

Input format:
- docs file: one document per line
- labels file: one label per line (line-aligned with docs)
  - supports 0/1 or aliases like majority/minority, neg/pos

Pipeline:
1) bytes_tokens(doc) -> list[int]
2) hashed_bow_embedding(token_ids) -> dense vector
3) gmr_keep_mask(X, y) -> keep/drop per document
4) write kept docs to output file

Usage:
  python chapters/ch05_microgpt/scripts/gmr_clean.py \
      --docs docs.txt --labels labels.txt --output input_clean.txt
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path


def bytes_tokens(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore"))


def hashed_bow_embedding(token_ids: list[int], dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in token_ids:
        vec[token % dim] += 1.0

    sq = 0.0
    for value in vec:
        sq += value * value
    norm = math.sqrt(sq) + 1e-12
    for i in range(dim):
        vec[i] /= norm
    return vec


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _euclidean(a: list[float], b: list[float]) -> float:
    sq = 0.0
    for i in range(len(a)):
        delta = a[i] - b[i]
        sq += delta * delta
    return math.sqrt(sq)


def _cosine_dist(
    a: list[float], b: list[float], na: float | None = None, nb: float | None = None
) -> float:
    eps = 1e-12
    if na is None:
        na = _norm(a)
    if nb is None:
        nb = _norm(b)
    return 1.0 - (_dot(a, b) / (na * nb + eps))


def gmr_keep_mask(
    x: list[list[float]],
    y: list[int],
    k: int = 15,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.1,
    metric: str | None = None,
) -> list[bool]:
    """Pure-Python GMR implementation.

    Args:
      x: Dense vectors
      y: Binary labels (0=majority, 1=minority)
      k, alpha, beta, gamma: GMR hyperparameters
      metric: "euclidean", "cosine", or None (auto by dimension)
    """

    n = len(x)
    if n == 0:
        return []

    d = len(x[0])
    n_min = 0
    for label in y:
        if label == 1:
            n_min += 1

    if n_min < 10:
        return [True] * n

    if metric is None:
        metric = "cosine" if d > 100 else "euclidean"

    norms = None
    if metric == "cosine":
        norms = [_norm(vec) for vec in x]

    def dist(i: int, j: int) -> float:
        if metric == "cosine":
            assert norms is not None
            return _cosine_dist(x[i], x[j], norms[i], norms[j])
        return _euclidean(x[i], x[j])

    kk = min(k, n - 1)
    if kk <= 0:
        return [True] * n

    yhat = [0] * n
    v0 = [0.0] * n
    v1 = [0.0] * n
    eps = 1e-12

    for i in range(n):
        neighbors = heapq.nsmallest(
            kk,
            ((dist(i, j), j) for j in range(n) if j != i),
            key=lambda pair: pair[0],
        )

        weights: list[tuple[float, int]] = []
        wsum = 0.0
        for dij, j in neighbors:
            w = 1.0 / (dij + eps)
            weights.append((w, j))
            wsum += w

        s0 = 0.0
        s1 = 0.0
        inv = 1.0 / (wsum + eps)
        for w, j in weights:
            wn = w * inv
            if y[j] == 1:
                s1 += wn
            else:
                s0 += wn

        v0[i] = s0
        v1[i] = s1
        yhat[i] = 1 if s1 > s0 else 0

    remove = [False] * n

    for i in range(n):
        if y[i] == 0:
            conf = v0[i]
            if yhat[i] == 1 or conf < alpha:
                remove[i] = True

    candidates: list[int] = []
    for i in range(n):
        if y[i] == 1 and yhat[i] == 0 and v0[i] > beta:
            candidates.append(i)

    budget = int(math.floor(gamma * n_min))
    if budget > 0 and candidates:
        candidates.sort(key=lambda i: v0[i], reverse=True)
        for i in candidates[:budget]:
            remove[i] = True

    return [not flag for flag in remove]


def parse_label(raw: str) -> int:
    value = raw.strip().lower()
    if value in {"1", "minority", "min", "pos", "positive", "clean", "good"}:
        return 1
    if value in {"0", "majority", "maj", "neg", "negative", "noisy", "bad"}:
        return 0
    raise ValueError(
        f"Unsupported label '{raw}'. Use 0/1 or majority/minority aliases."
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="GMR cleaner for corpus documents")
    parser.add_argument(
        "--docs", required=True, help="Path to docs file (one line per doc)"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels file (one line per label, aligned)",
    )
    parser.add_argument(
        "--output", default="input_clean.txt", help="Output path for cleaned docs"
    )
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--k", type=int, default=15, help="GMR k")
    parser.add_argument("--alpha", type=float, default=0.3, help="GMR alpha")
    parser.add_argument("--beta", type=float, default=0.7, help="GMR beta")
    parser.add_argument("--gamma", type=float, default=0.1, help="GMR gamma")
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "auto"],
        default="auto",
        help="Distance metric",
    )
    args = parser.parse_args()

    docs_path = Path(args.docs)
    labels_path = Path(args.labels)
    out_path = Path(args.output)

    docs = read_lines(docs_path)
    labels_raw = read_lines(labels_path)

    if len(docs) != len(labels_raw):
        raise ValueError(
            f"Line mismatch: docs={len(docs)} labels={len(labels_raw)}. They must align."
        )

    labels = [parse_label(v) for v in labels_raw]

    x: list[list[float]] = []
    for doc in docs:
        token_ids = bytes_tokens(doc)
        x.append(hashed_bow_embedding(token_ids, dim=args.dim))

    metric = None if args.metric == "auto" else args.metric
    keep = gmr_keep_mask(
        x,
        labels,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        metric=metric,
    )

    kept_docs = [doc for doc, flag in zip(docs, keep) if flag]
    kept_labels = [label for label, flag in zip(labels, keep) if flag]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(kept_docs) + ("\n" if kept_docs else ""), encoding="utf-8"
    )

    kept_count = len(kept_docs)
    dropped_count = len(docs) - kept_count
    kept_min = sum(1 for v in kept_labels if v == 1)
    kept_maj = sum(1 for v in kept_labels if v == 0)

    print("GMR cleaning complete")
    print(f"- input docs: {len(docs)}")
    print(f"- kept docs: {kept_count}")
    print(f"- dropped docs: {dropped_count}")
    print(f"- kept minority/majority: {kept_min}/{kept_maj}")
    print(f"- output: {out_path}")


if __name__ == "__main__":
    main()

## Q1

Today's topic of research https://arxiv.org/abs/2602.13045  
https://youtu.be/8ihN1ToYtGo?si=Y5l0kVt5CsuanrGa  
  
Goal : So I plan to take learnings from here and try my hands on karpathy's recent github gist on microgpt microgpt · GitHub https://share.google/7U7IOKqNamtlCm7Py  
  
Or an unslothAI notebook with a tiny model or a some other neural network

## A1


Here’s what you’ve got in those links, and a clean way to “transfer the learning” into **microgpt** or a **tiny Unsloth notebook**.

## 1) What arXiv:2602.13045 actually proposes (GMR)

The paper is **Geometric Manifold Rectification (GMR)** for **imbalanced + noisy tabular classification**. Core idea: treat resampling as **geometry-aware data cleaning**—specifically, remove *majority* points that “intrude” into the minority manifold and blur the decision boundary. ([arXiv][1])

**Two moving parts:**

* **Geometric confidence** via **inverse-distance weighted kNN voting** (closer neighbors count more). ([arXiv][1])
* **Asymmetric cleaning:**

  * **Majority cleaned aggressively** (remove if kNN predicts minority or if same-class confidence < α)
  * **Minority cleaned conservatively** (only remove if deeply embedded in majority), plus a hard cap γ on minority removals. ([arXiv][2])

**Default hyperparams (the paper’s “no-tuning” recipe):**

* `k=15, α=0.3, β=0.7, γ=0.1`, metric switch: cosine if dim>100 else euclidean, skip cleaning if minority <10 samples. ([arXiv][2])

Algorithm 1 is very implementable straight from pseudocode. ([arXiv][2])

## 2) About the YouTube “Basin Repair” framing

The video’s title/description frames this as a “Basin Repair Surgery” method tied to arXiv:2602.13045. ([youtube.com][3])
Even if you ignore the metaphor, the concrete mechanism is: **use local neighborhood geometry to delete boundary-confusing points**, which effectively “repairs” the overlap region.

## 3) Your goal → three practical routes (pick one, or do them in order)

### Route A (fastest, most faithful): reproduce GMR on a toy dataset

Do this first if you want the “aha” quickly.

* Make an imbalanced binary dataset (plus label noise).
* Run baseline classifier vs baseline+GMR-cleaned data.
* Compare **AUPRC** (paper’s main metric). ([arXiv][2])

Minimal implementation sketch (binary y∈{0,1}):

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def gmr_clean_mask(X, y, k=15, alpha=0.3, beta=0.7, gamma=0.1, eps=1e-8):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n, d = X.shape

    # critical scarcity guard
    if (y == 1).sum() < 10:
        return np.ones(n, dtype=bool)

    metric = "cosine" if d > 100 else "euclidean"
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self

    w = 1.0 / (dist + eps)
    w /= w.sum(axis=1, keepdims=True)

    neigh_y = y[idx]
    v1 = (w * (neigh_y == 1)).sum(axis=1)
    v0 = (w * (neigh_y == 0)).sum(axis=1)

    yhat = (v1 > v0).astype(int)
    conf = np.where(y == 1, v1, v0)
    majconf = v0

    # majority removals
    Rmaj = (y == 0) & ((yhat == 1) | (conf < alpha))

    # minority candidates + cap
    Cmin = (y == 1) & (yhat == 0) & (majconf > beta)
    cidx = np.where(Cmin)[0]
    budget = int(np.floor(gamma * (y == 1).sum()))
    if budget > 0 and len(cidx) > 0:
        cidx = cidx[np.argsort(-majconf[cidx])]  # desc
        Rmin = cidx[: min(len(cidx), budget)]
    else:
        Rmin = np.array([], dtype=int)

    keep = np.ones(n, dtype=bool)
    keep[Rmaj] = False
    keep[Rmin] = False
    return keep
```

This is basically Algorithm 1. ([arXiv][2])

---

### Route B (your Karpathy goal): bolt the *idea* onto **microgpt**

Karpathy’s **microgpt** is a single-file, dependency-free GPT training + inference script (dataset → tokenizer → autograd → GPT-2-ish model → Adam → train loop). ([karpathy.github.io][4])

**Key reality check:** microgpt is *next-token prediction* on unlabeled text, while GMR is *labeled* imbalanced classification. So you need a bridge. Two good bridges:

**B1) “Source-quality as labels” (closest to GMR):**

* Create a mixed training set:

  * **minority (y=1):** high-quality docs (your curated set)
  * **majority (y=0):** low-quality/noisy docs (synthetic junk, scraped clutter, etc.)
* Embed each doc into a vector space (cheap option: char n-gram TF-IDF; nicer option: sentence embeddings).
* Run GMR to remove intrusive majority docs near the minority manifold.
* Train microgpt on the cleaned docs list.

**B2) “Outlier pruning” (GMR-inspired, unlabeled):**

* Treat “good manifold” as dense regions in embedding space.
* Drop docs with low geometric support (low average similarity to kNN). Not GMR exactly, but same spirit.

**Microgpt integration point:** the dataset is just `docs = [...]` near the top. ([karpathy.github.io][4])
So your workflow is: `docs_raw -> clean -> write back docs_clean -> run training`.

If you want to keep the **microgpt file pure**, do the cleaning in a separate notebook/script and just output a new `input.txt`.

---

### Route C (most “LLM-native”): Unsloth tiny-model notebook + GMR-style cleaning

Unsloth literally has a catalog of ready notebooks (“Run all, add dataset, train”). ([unsloth.ai][5])

A solid experiment that *matches the paper*:

* Pick a **tiny model** (e.g., TinyLlama-class) and a **binary classification** or **preference** dataset with imbalance.
* Embed samples (sentence embeddings) → run **GMR cleaning** in embedding space → fine-tune → evaluate with PR-AUC/F1.
* This mirrors their CIFAR “feature-space GMR” extension idea (apply GMR in a representation space before training). ([arXiv][2])

---

## What I’d do (quick win path)

1. **Route A** on a toy dataset (prove to yourself GMR works in 30 minutes).
2. Then **Route B1**: mix clean + noisy docs, clean them with GMR in embedding space, and feed the cleaned list into microgpt.

If you tell me which direction you’re leaning (**microgpt-first** vs **Unsloth-first**), I’ll outline the exact minimal experiment setup (dataset choice, embedding choice, and what metrics to log) without turning it into a huge project.

[1]: https://www.arxiv.org/abs/2602.13045 "[2602.13045] Geometric Manifold Rectification for Imbalanced Learning"
[2]: https://www.arxiv.org/pdf/2602.13045 "Geometric Manifold Rectification for Imbalanced Learning"
[3]: https://www.youtube.com/watch?v=8ihN1ToYtGo&utm_source=chatgpt.com "This New “Basin Repair” Method Might Unlock AGI (Full ..."
[4]: https://karpathy.github.io/2026/02/12/microgpt/ "microgpt"
[5]: https://unsloth.ai/docs/get-started/unsloth-notebooks?utm_source=chatgpt.com "Unsloth Notebooks"


## Q2

No libs

## A2


Yep — we can keep it **pure standard library Python**, microgpt-style.

Below is a **drop-in, no-numpy/no-sklearn** implementation of **GMR cleaning** (Algorithm 1 from the paper), plus a tiny “embedding” trick that works with microgpt-like token IDs.

---

## 1) Pure-Python GMR (no deps)

```python
import math
import heapq


def _dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _norm(a):
    return math.sqrt(_dot(a, a))


def _euclidean(a, b):
    # true distance (not squared) because GMR uses 1/(d+eps)
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def _cosine_dist(a, b, na=None, nb=None, eps=1e-12):
    # distance = 1 - cosine_similarity
    if na is None:
        na = _norm(a)
    if nb is None:
        nb = _norm(b)
    denom = (na * nb) + eps
    return 1.0 - (_dot(a, b) / denom)


def gmr_keep_mask(X, y, k=15, alpha=0.3, beta=0.7, gamma=0.1, eps=1e-12, metric=None):
    """
    Pure-Python Geometric Manifold Rectification (GMR) data cleaning.

    Args:
      X: list[list[float]]  (dense vectors, same length)
      y: list[int]          (0=majority, 1=minority)
      k, alpha, beta, gamma: GMR hyperparams
      metric: None | "euclidean" | "cosine"
        If None, follows paper: cosine if dim > 100 else euclidean.

    Returns:
      keep: list[bool] same length as X
    """
    n = len(X)
    if n == 0:
        return []

    d = len(X[0])
    y = [int(v) for v in y]

    n_min = 0
    for v in y:
        if v == 1:
            n_min += 1
    if n_min < 10:
        # paper: skip cleaning if minority count < 10
        return [True] * n

    if metric is None:
        metric = "cosine" if d > 100 else "euclidean"

    # precompute norms for cosine to save work
    norms = None
    if metric == "cosine":
        norms = [_norm(vec) for vec in X]

    def dist(i, j):
        if metric == "cosine":
            return _cosine_dist(X[i], X[j], norms[i], norms[j], eps=eps)
        return _euclidean(X[i], X[j])

    # Step 1: kNN weighted votes for each point
    yhat = [0] * n
    v0 = [0.0] * n
    v1 = [0.0] * n

    # k+1 to safely handle tiny n; we'll cap later
    kk = min(k, n - 1)
    if kk <= 0:
        return [True] * n

    for i in range(n):
        # find kk nearest neighbors excluding self
        # O(n log k) using nsmallest over generator; simple and dependency-free
        neigh = heapq.nsmallest(
            kk,
            ((dist(i, j), j) for j in range(n) if j != i),
            key=lambda t: t[0]
        )

        # inverse-distance weights
        weights = []
        wsum = 0.0
        for (dij, j) in neigh:
            w = 1.0 / (dij + eps)
            weights.append((w, j))
            wsum += w

        # normalize and vote
        s0 = 0.0
        s1 = 0.0
        inv = 1.0 / (wsum + eps)
        for (w, j) in weights:
            wn = w * inv
            if y[j] == 1:
                s1 += wn
            else:
                s0 += wn

        v0[i] = s0
        v1[i] = s1
        yhat[i] = 1 if s1 > s0 else 0

    # Step 2: majority removals
    remove = [False] * n
    for i in range(n):
        if y[i] == 0:
            conf = v0[i]
            if (yhat[i] == 1) or (conf < alpha):
                remove[i] = True

    # Step 3: minority candidate removals with cap gamma
    # candidates: minority predicted majority AND strongly embedded (v0 > beta)
    cand = []
    for i in range(n):
        if (y[i] == 1) and (yhat[i] == 0) and (v0[i] > beta):
            cand.append(i)

    budget = int(math.floor(gamma * n_min))
    if budget > 0 and cand:
        # remove those with largest v0 first
        cand.sort(key=lambda i: v0[i], reverse=True)
        for i in cand[:budget]:
            remove[i] = True

    return [not r for r in remove]
```

---

## 2) Microgpt-friendly “embedding” with *only* Python

To run GMR you need vectors. Without numpy/sklearn, a nice trick is a **hashed bag-of-tokens**:

* You already have token IDs (or bytes).
* Make a fixed-length dense vector by counting `token_id % dim`.
* L2-normalize so cosine distance is meaningful.

```python
def hashed_bow_embedding(token_ids, dim=256):
    v = [0.0] * dim
    for t in token_ids:
        v[t % dim] += 1.0

    # L2 normalize
    s = 0.0
    for x in v:
        s += x * x
    n = math.sqrt(s) + 1e-12
    for i in range(dim):
        v[i] /= n
    return v
```

If you don’t have token IDs yet and want the simplest possible tokenizer: **bytes**.

```python
def bytes_tokens(s):
    return list(s.encode("utf-8", errors="ignore"))
```

---

## 3) End-to-end: clean docs before feeding microgpt

```python
# docs: list[str]
# y:    list[int]  1=high-quality/minority, 0=noisy/majority

X = []
for doc in docs:
    toks = bytes_tokens(doc)              # or your microgpt token ids if you have them
    X.append(hashed_bow_embedding(toks, dim=256))

keep = gmr_keep_mask(X, y, k=15, alpha=0.3, beta=0.7, gamma=0.1)

docs_clean = [d for d, k in zip(docs, keep) if k]
y_clean    = [c for c, k in zip(y, keep) if k]

# write for microgpt training
with open("input_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(docs_clean))
```

That’s it: you now have an **offline, geometry-aware “data cleaning” stage** that stays true to microgpt’s “no dependencies” vibe.

---

If you tell me what your “labels” are going to be in practice (e.g., *curated notes vs scraped dump*, *good code vs bad code*, *signal vs noise*), I’ll suggest the cleanest way to construct `y` and keep the dataset format microgpt-friendly.

# microGPT Learning Notes

Notes from working through Andrej Karpathy's microGPT/nanoGPT materials and related implementations.

---

## GMR Summary (from 2026-02-19 chat)

### What is GMR?

**Geometric Manifold Rectification (GMR)** is a data cleaning method for imbalanced + noisy classification. Core idea: treat resampling as geometry-aware cleaning—remove *majority* points that intrude into the minority manifold and blur the decision boundary.

**Two mechanisms:**
1. **Geometric confidence** via inverse-distance weighted kNN voting (closer neighbors count more)
2. **Asymmetric cleaning:**
   - Majority cleaned aggressively (remove if kNN predicts minority OR same-class confidence < α)
   - Minority cleaned conservatively (only if deeply embedded in majority), capped at γ

### Default Hyperparams (no-tuning recipe)

| Param | Value | Meaning |
|-------|-------|---------|
| k | 15 | neighbors for voting |
| α | 0.3 | majority confidence threshold |
| β | 0.7 | minority embedding threshold |
| γ | 0.1 | max minority removal fraction |

Metric: cosine if dim > 100, else euclidean. Skip cleaning if minority < 10 samples.

### Three Practical Routes

| Route | Description | Effort |
|-------|-------------|--------|
| **A: Toy repro** | Implement GMR on synthetic imbalanced dataset, compare AUPRC vs baseline | Low |
| **B: microGPT bridge** | Clean docs via GMR in embedding space before training | Medium |
| **C: Unsloth route** | Fine-tune tiny model on GMR-cleaned classification/preference data | Medium |

### Action Plan

1. [ ] **Route A**: Prove GMR works on toy data (~30 min)
2. [ ] **Route B1**: Mix high-quality (y=1) + noisy (y=0) docs, clean with GMR, feed to microGPT
3. [ ] Log AUPRC/F1 metrics for each experiment

---

## Roadmap Overview

| Phase | Focus | Status |
|-------|-------|--------|
| 1. Tokenization | BPE, encodings, special tokens | TODO |
| 2. Attention | Self-attention, multi-head, causal mask | TODO |
| 3. Transformer Blocks | FFN, LayerNorm, residual connections | TODO |
| 4. GPT Architecture | Full model assembly, position encoding | TODO |
| 5. Training Loop | Loss, optimizer, gradient clipping | TODO |
| 6. Generation | Sampling, temperature, top-k/top-p | TODO |

---

## Code Snippets

### Hashed Bag-of-Words Embedding (no deps)

```python
import math

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

def bytes_tokens(s):
    return list(s.encode("utf-8", errors="ignore"))
```

### GMR Keep Mask (pure Python, no deps)

```python
import math
import heapq

def gmr_keep_mask(X, y, k=15, alpha=0.3, beta=0.7, gamma=0.1, eps=1e-12, metric=None):
    """
    Pure-Python GMR data cleaning.
    X: list[list[float]], y: list[int] (0=majority, 1=minority)
    Returns: keep mask as list[bool]
    """
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    n_min = sum(1 for v in y if v == 1)
    if n_min < 10:
        return [True] * n
    if metric is None:
        metric = "cosine" if d > 100 else "euclidean"
    # ... (full impl in gmr-chat.md)
```

---

## Implementation Checklist

- [ ] Route A: Toy GMR experiment on imbalanced synthetic data
- [ ] Route B: Document cleaning pipeline for microGPT
- [ ] Benchmark: Compare AUPRC before/after GMR
- [ ] Integrate `gmr_keep_mask` into data prep notebook

---

## Experiments Log

### Experiment: GMR on Toy Dataset
- **Date:** _TBD_
- **Hypothesis:** GMR cleaning improves AUPRC on imbalanced noisy data
- **Setup:** Synthetic binary classification, 10:1 imbalance, 5% label noise
- **Results:** _pending_
- **Learnings:** _pending_

### Experiment: GMR + microGPT
- **Date:** _TBD_
- **Hypothesis:** Geometry-cleaned docs improve microGPT loss convergence
- **Setup:** Mixed corpus (curated + noisy), hashed BoW embeddings, GMR filter
- **Results:** _pending_
- **Learnings:** _pending_

---

## Questions Queue

| # | Question | Priority | Status |
|---|----------|----------|--------|
| 1 | Why 4x expansion in FFN? | Medium | Open |
| 2 | Pre-LN vs Post-LN stability | High | Open |
| 3 | Best embedding dim for hashed BoW? | Medium | Open |
| 4 | Does GMR help next-token prediction tasks? | High | Open |

---

## References

- [microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - Karpathy's reference impl
- [GMR Paper (arXiv:2602.13045)](https://arxiv.org/abs/2602.13045) - Geometric Manifold Rectification
- [Basin Repair Surgery (YouTube)](https://youtu.be/8ihN1ToYtGo) - Video explaining GMR
- [nanoGPT repo](https://github.com/karpathy/nanoGPT) - Reference GPT training
- [Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Karpathy's video series
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Unsloth Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) - Ready-to-run fine-tuning notebooks
- [Colab GMR nb-1](https://colab.research.google.com/drive/1JGzycOsvAbRwAfMoRSTtz8hrmjJtEf6U) - Experiments notebook
- [Colab GMR nb-2](https://colab.research.google.com/drive/1X5dmV6QBVg_v25HDYIObRPi8On7HV5Me) - Experiments notebook
- [Colab GMR nb-3](https://colab.research.google.com/drive/1aL_0bBGG0mBP0Afx66qpCWwV8ge2L_Va) - Experiments notebook

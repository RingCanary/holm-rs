# microGPT Learning Notes

Notes from working through Andrej Karpathy's microGPT/nanoGPT materials and related implementations.

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

## Concepts Log

### 2025-XX-XX: [Topic Title]

**Key insight:** 

**Math essentials:**

**Code snippet:**
```python
# Placeholder
```

**Questions:**

---

## Implementation Checkpoints

### Checkpoint 1: Tokenizer
- [ ] Implement BPE training
- [ ] Handle special tokens (BOS, EOS, PAD)
- [ ] Test roundtrip encoding/decoding

### Checkpoint 2: Attention
- [ ] Single-head self-attention
- [ ] Causal masking
- [ ] Multi-head attention

### Checkpoint 3: Transformer Block
- [ ] LayerNorm before/after (Pre-LN vs Post-LN)
- [ ] Feed-forward network (4x expansion)
- [ ] Residual connections

### Checkpoint 4: Full GPT
- [ ] Embedding layer + positional encoding
- [ ] Stack of transformer blocks
- [ ] Output projection to vocabulary

### Checkpoint 5: Training
- [ ] Cross-entropy loss
- [ ] AdamW optimizer
- [ ] Gradient clipping
- [ ] Learning rate warmup + decay

### Checkpoint 6: Sampling
- [ ] Greedy decoding
- [ ] Temperature scaling
- [ ] Top-k filtering
- [ ] Top-p (nucleus) sampling

---

## Experiments Log

### Experiment: [Name]
**Date:** 
**Hypothesis:** 
**Setup:** 
**Results:** 
**Learnings:** 

---

## Questions Queue

| # | Question | Priority | Status |
|---|----------|----------|--------|
| 1 | Why 4x expansion in FFN? | Medium | Open |
| 2 | Pre-LN vs Post-LN stability | High | Open |
| 3 | TODO | - | - |

---

## References

- [nanoGPT repo](https://github.com/karpathy/nanoGPT) - Reference implementation
- [Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Karpathy's video series
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper

<!-- TODO: Add notes from HÖLM-RS ch02_tokens as you progress -->

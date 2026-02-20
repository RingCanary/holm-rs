# RWKV Worklog

## Goals & Scope

- Build a minimal RWKV chapter in pure Rust for learning.
- Keep implementation incremental and runnable at each step.
- Compare behavior against microgpt chapter patterns.

## Milestones

- [x] Research pass and source curation completed
- [x] Chapter scaffold created (`ch06_rwkv`)
- [x] Lesson 00: recurrence basics with toy state update
- [ ] Lesson 01: time-mix step implementation
- [ ] Lesson 02: channel-mix step implementation
- [ ] Lesson 03: single block forward + residual
- [ ] Lesson 04: tiny text generation loop
- [ ] Add tiny benchmark + sanity checks

## Session Log

### 2026-02-20

- Completed citation-grade research on GMR and architecture options.
- Reconfirmed RWKV as first post-microgpt target for pure-std Rust learning.
- Created chapter skeleton and research note files.

### 2026-02-20 (Session 2)

- **Implemented Lesson 00: Recurrence Basics**
  - Created real scalar WKV recurrence step function with parameters (num, den, k, v, decay, bonus)
  - Ran 5-step hardcoded sequence demonstration
  - Added compact per-step table output showing k, v, num, den, y
  - Included 3 deterministic sanity checks:
    1. y[0] equals v[0] when initial state is zero
    2. Manual verification of y[1] via explicit formula
    3. All outputs bounded by min/max values
  - All assertions pass with epsilon tolerance
  - Code is pure Rust, no external dependencies

## Experiments

| Date | Experiment | Setup | Result | Next |
|------|------------|-------|--------|------|
| 2026-02-20 | Scaffold only | chapter + docs | ready | implement Lesson 00 |
| 2026-02-20 | Lesson 00 implementation | scalar WKV + assertions | validated | Lesson 01 time-mix |

## Open Questions

- Which tokenizer path first: byte-level or BPE?
- Should Lesson 04 use fixed tiny corpus or input file path?
- Do we target RWKV-4 only first, then RWKV-5/6 extension chapter?

## TODO Next

1. Implement `01_time_mix_step` with vectorized WKV over channels.
2. Add tiny `Vec1D` helper ops (dot, add, mul, sigmoid) if needed.
3. Implement stable WKV update utility and unit checks.

## References

- https://arxiv.org/abs/2305.13048
- https://arxiv.org/abs/2404.05892
- https://github.com/BlinkDL/RWKV-LM
- https://johanwind.github.io/2023/03/23/rwkv_details.html

# RWKV Notes

Implementation-oriented notes for a microgpt-style RWKV build in pure Rust.

## Minimal Math Breakdown (RWKV-4 first)

Given token representation `x_t` and previous state:

1) Time interpolation:

`xk = x_t * mix_k + x_{t-1} * (1 - mix_k)`

`xv = x_t * mix_v + x_{t-1} * (1 - mix_v)`

`xr = x_t * mix_r + x_{t-1} * (1 - mix_r)`

2) Projections:

`k = W_k xk`, `v = W_v xv`, `r = W_r xr`

3) WKV recurrent aggregation (per channel):

`wkv_t = (num_{t-1} + exp(bonus + k) * v) / (den_{t-1} + exp(bonus + k))`

`num_t = exp(-exp(decay)) * num_{t-1} + exp(k) * v`

`den_t = exp(-exp(decay)) * den_{t-1} + exp(k)`

4) Time-mix output:

`out_t = W_out (sigmoid(r) * wkv_t)`

5) Channel-mix (gated FFN style):

`k_c = W_k xk_c`, `r_c = W_r xr_c`, `v_c = W_v (ReLU(k_c)^2)`

`out_c = sigmoid(r_c) * v_c`

## Complexity

- Inference: recurrent, constant-memory per layer state.
- Training: can be parallelized with scan-style formulation.
- Practical contrast vs GPT: avoids KV cache growth with context length.

## Reading Order

1. RWKV core paper (v4):
   - https://arxiv.org/abs/2305.13048
   - https://markdown.new/https://arxiv.org/abs/2305.13048
2. RWKV v5/v6 (Eagle/Finch):
   - https://arxiv.org/abs/2404.05892
   - https://markdown.new/https://arxiv.org/abs/2404.05892
3. Official codebase:
   - https://github.com/BlinkDL/RWKV-LM
4. Minimal explainer:
   - https://johanwind.github.io/2023/03/23/rwkv_details.html

Context references:

- RetNet: https://arxiv.org/abs/2307.08621
- Mamba: https://arxiv.org/abs/2312.00752
- Linear attention: https://arxiv.org/abs/2006.16236

## Implementation Checklist

- [ ] Build tiny RWKV state struct: `prev_x`, `num`, `den`
- [ ] Implement time-mix interpolation
- [ ] Implement stable WKV update (avoid exp overflow)
- [ ] Implement channel-mix with squared ReLU
- [ ] Compose one RWKV block with residual paths
- [ ] Run recurrent token loop for tiny text generation
- [ ] Validate logits against a small reference case

## Pitfalls

- Mixing direction bugs (`x_t` vs `x_{t-1}` weights swapped).
- Numerical overflow in `exp` during WKV updates.
- Forgetting per-layer persistent recurrent state between tokens.
- Using plain ReLU instead of squared ReLU in channel mix.

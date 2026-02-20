# Chapter 6: RWKV (Learning Skeleton)

This chapter is a minimal, educational scaffold for implementing RWKV in pure Rust.

## Goals

- Build intuition for RWKV recurrence vs transformer attention.
- Implement tiny forward pieces in isolated lesson binaries.
- Progress to a tiny text-generation demo.

## Layout

```text
ch06_rwkv/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs
    ├── rwkv/
    │   ├── mod.rs
    │   ├── tensor.rs
    │   ├── state.rs
    │   ├── time_mix.rs
    │   ├── channel_mix.rs
    │   ├── block.rs
    │   └── model.rs
    └── bin/
        ├── 00_recurrence_basics.rs
        ├── 01_time_mix_step.rs
        ├── 02_channel_mix_step.rs
        ├── 03_single_block_forward.rs
        ├── 04_tiny_rwkv_textgen.rs
        └── 05_dataset_prep.rs
```

## Run

```bash
cargo run -p ch06_rwkv
cargo run -p ch06_rwkv --bin 00_recurrence_basics
cargo run -p ch06_rwkv --bin 04_tiny_rwkv_textgen
cargo run -p ch06_rwkv --bin 05_dataset_prep
```

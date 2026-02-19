# AGENTS.md

Quick reference for coding agents. See `docs/contributor_guide.md` for full details.

## Project Overview

HÖLM-RS is an educational Rust workspace for hands-on language model concepts.

**Active workspace members:** `ch02_tokens`, `ch04_lmstudio_api`. Other chapter folders are placeholders.

## Core Commands

```bash
cargo build                  # Build all chapters
cargo build --release        # Optimized build
cargo test --workspace       # Run tests
cargo check --workspace      # Fast type-check
cargo fmt --all --check      # Check formatting
cargo clippy --workspace -- -D warnings  # Lint
cargo run -p ch02_tokens     # Run specific chapter
cargo run -p ch04_lmstudio_api
```

## Adding New Chapters

1. Create `chapters/chXX_name` and run `cargo new chXX_name` inside
2. Append to `Cargo.toml` workspace members (don't create `[workspace.members]` table):
   ```toml
   members = ["chapters/ch02_tokens", "chapters/ch04_lmstudio_api", "chapters/chXX_name"]
   ```
3. Verify: `cargo build -p chXX_name`

## LM Studio (Chapter 4)

- Endpoint: `http://localhost:1234/v1/chat/completions`
- Model: `gemma-3-270m-it` (adjust if different)
- Start LM Studio, load model, ensure server runs on port 1234

## Troubleshooting

- Build issues: `rustup update`, `cargo clean`
- LM Studio: verify server at `localhost:1234`, check model name
- Performance: use `cargo build --release`

## Code Style

- Rust 2024 edition, `cargo fmt`, clippy
- Prefer `anyhow` for errors, `serde` for JSON

# HÖLM-RS

Hands-On Language Models in Rust — an educational, chapter-based workspace exploring tokenization, attention, local LLM APIs, and more.

## Quick Start

- Build the workspace: `cargo build`
- Run a chapter: `cargo run -p ch02_tokens` or `cargo run -p ch04_lmstudio_api`
- Format and lint: `cargo fmt --all --check` and `cargo clippy --workspace -- -D warnings`

## LM Studio (Chapter 4)

- Start LM Studio and enable the OpenAI-compatible local server on `http://localhost:1234`
- Ensure `POST /v1/chat/completions` is available and a chat model (e.g., Gemma-3-270m-it) is loaded
- Then: `cargo run -p ch04_lmstudio_api`

## Docs for Contributors

See `AGENTS.md` for full guidance on workspace layout, commands, adding chapters, and LM Studio details.

# AGENTS.md

This file provides context and instructions for coding agents working on the HÖLM-RS (Hands-On Language Models in Rust) project.

## Project Overview

HÖLM-RS is an educational Rust workspace project designed to teach hands-on language model concepts through practical implementation. The project follows a chapter-based approach where each chapter focuses on specific aspects of language model development and usage.

Active workspace members today: `ch02_tokens`, `ch04_lmstudio_api`. Other chapter folders exist as placeholders and are not part of the workspace until explicitly added.

## Prerequisites

- Rust 2024 edition (stable toolchain)
- Cargo package manager
- For Chapter 4: LM Studio running locally on port 1234
- CPU with native optimization support (recommended)

## Development Environment Setup

1. **Install Rust:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup default stable
   ```

2. **Clone and setup:**
   ```bash
   git clone https://github.com/your-repo/holm-rs.git
   cd holm-rs
   ```

3. **Build the workspace:**
   ```bash
   cargo build --release
   ```

4. **For LM Studio integration (Chapter 4):**
   - Start LM Studio
   - Load Gemma-3-270m-it model (or similar)
   - Ensure local server is running on `localhost:1234`

## Project Structure

```
holm-rs/
├── Cargo.toml              # Workspace configuration
├── .cargo/config.toml      # Build optimization settings
├── chapters/               # Chapter implementations
│   ├── ch02_tokens/        # Tokenization basics
│   ├── ch03_attention/     # Attention mechanisms
│   ├── ch04_lmstudio_api/  # LM Studio API integration
│   ├── ch07_chains/        # Chain operations
│   ├── ch08_rag/           # RAG implementation
│   ├── ch10_embed_train/   # Embedding training
│   └── ch12_sft_tiny/      # Small model fine-tuning
└── target/                 # Build artifacts
```

## Build Commands

### Workspace Commands
```bash
# Build all chapters
cargo build

# Build with optimizations
cargo build --release

# Run tests across all chapters
cargo test --workspace

# Fast type-check without building
cargo check --workspace

# Check code formatting
cargo fmt --all --check

# Run clippy lints
cargo clippy --workspace -- -D warnings
```

### Chapter-Specific Commands
```bash
# Run a specific chapter
cargo run -p ch02_tokens
cargo run -p ch04_lmstudio_api

# Build specific chapter
cargo build -p ch02_tokens
```

## Testing Commands

When tests are implemented in chapters:
```bash
# Run all tests
cargo test

# Run tests for specific chapter
cargo test -p ch04_lmstudio_api

# Run tests with output
cargo test -- --nocapture
```

## Development Workflow

1. **Create a new branch for features:**
   ```bash
   git checkout -b feature/chapter-name
   ```

2. **Make changes and commit:**
  ```bash
  cargo check --workspace
  cargo fmt
  cargo clippy
  git add .
  git commit -m "feat: implement chapter functionality"
  ```

3. **Push and create PR:**
   ```bash
   git push origin feature/chapter-name
   ```

## Code Style

- Follow Rust 2024 edition standards
- Use `cargo fmt` for formatting
- Follow clippy linting rules
- Prefer `anyhow` for error handling
- Use `serde` derive macros for JSON serialization

## Adding New Chapters

1. Create new directory in `chapters/` with naming pattern `chXX_name`
2. Initialize with `cargo new chXX_name`
3. Add it to the root `Cargo.toml` `members` array under `[workspace]` (append to the list — do not create a `[workspace.members]` table):
   ```toml
   [workspace]
   members = [
       "chapters/ch02_tokens",
       "chapters/ch04_lmstudio_api",
       "chapters/chXX_name",
   ]
   resolver = "2"
   ```
4. Build or run the new chapter to verify:
   ```bash
   cargo build -p chXX_name
   # or
   cargo run -p chXX_name
   ```
5. Implement chapter functionality following existing patterns

## Dependencies

Key external dependencies:
- `reqwest`: HTTP client for API calls
- `serde` & `serde_json`: JSON serialization
- `anyhow`: Error handling
- `tokenizers`: Text tokenization
- `tokio`: Async runtime (when needed)

## Security Considerations

- API keys should not be committed to the repository
- LM Studio runs locally - no external API calls
- Ensure proper error handling for network requests
- Validate all external inputs in API implementations

## Troubleshooting

**Build Issues:**
- Ensure Rust is up to date: `rustup update`
- Clear build cache: `cargo clean`
- Check for missing system dependencies

**LM Studio Connection Issues:**
- Start LM Studio’s OpenAI-compatible local server and verify it is listening on `http://localhost:1234`.
- Ensure the chat endpoint `POST /v1/chat/completions` is available.
- Load a supported chat model (e.g., Gemma-3-270m-it) and ensure the name matches what the client code uses, or update the code to match your loaded model.
- Verify network connectivity to `localhost` and that no firewall is blocking port 1234.

**Performance Issues:**
- Use release builds for production: `cargo build --release`
- The `.cargo/config.toml` enables native CPU optimizations

Note: `-C target-cpu=native` optimizes for the local host CPU and can reduce portability (e.g., in CI runners). For CI, consider overriding or disabling this flag.

## LM Studio Integration Details (Chapter 4)

- Endpoint: `http://localhost:1234/v1/chat/completions` (OpenAI-compatible)
- Model name: default example uses `gemma-3-270m-it`; adjust if your LM Studio uses a different model identifier
- For classification-style tasks, prefer low temperatures (0–0.2) for stability
- Consider adding request timeouts and user-friendly errors if the server is unreachable

## Reproducibility & Toolchains

- Use stable Rust; the workspace targets the 2024 edition
- Optionally add a `rust-toolchain.toml` to pin the toolchain for CI consistency

## Extra Instructions

- Each chapter should be self-contained but follow workspace patterns
- Use descriptive commit messages following conventional commits
- Update documentation when adding new chapters
- Consider adding example usage in chapter READMEs when appropriate

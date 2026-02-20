# Chapter 5: MicroGPT - Character Bigram Language Model

A minimal character-level language model implementation demonstrating core concepts
of text generation without external dependencies.

## Rust Demo

### What it does

The `main.rs` implements a character-level bigram language model:

1. **Vocabulary Building**: Extracts unique characters from input text,
   including special BOS (beginning-of-sequence) and EOS (end-of-sequence) tokens.

2. **Bigram Counting**: Builds a transition count matrix where `counts[i][j]`
   represents how many times character `j` followed character `i`.

3. **Text Generation**: Uses a simple xorshift64 PRNG for deterministic
   sampling from the learned bigram distribution.

### Running the demo

```bash
# With built-in tiny corpus
cargo run -p ch05_microgpt

# With your own text file
cargo run -p ch05_microgpt -- path/to/your/text.txt
```

### Example output

```
=== Character Bigram Language Model ===

Input text length: 180 chars
Vocabulary size: 35 chars
Non-zero transitions: 142/1225 (11.6%)

--- Generated Sample (200 chars, seed=42) ---
hello world the quick brown fox jumps over the lazy dog a language...
```

## GMR Cleaner Script

### What it does

The `gmr_clean.py` script implements **Geometric Manifold Rectification (GMR)**
as a corpus-cleaning stage before training:

- **bytes_tokens**: byte-level tokenization
- **hashed_bow_embedding**: simple dense embedding (`token_id % dim`)
- **gmr_keep_mask**: inverse-distance weighted kNN voting with asymmetric
  majority/minority cleaning

### Default parameters (from the paper recipe)

- `k=15`: number of neighbors
- `alpha=0.3`: majority confidence threshold
- `beta=0.7`: minority embedding threshold
- `gamma=0.1`: minority removal budget fraction

### Running the script

```bash
# Basic usage
python chapters/ch05_microgpt/scripts/gmr_clean.py \
  --docs docs.txt \
  --labels labels.txt \
  --output input_clean.txt

# With custom parameters
python chapters/ch05_microgpt/scripts/gmr_clean.py \
  --docs docs.txt \
  --labels labels.txt \
  --output cleaned.txt \
  --k 20 --alpha 0.4 --beta 0.6 --gamma 0.1 --metric cosine

# View help
python chapters/ch05_microgpt/scripts/gmr_clean.py --help
```

### Input format

- `docs.txt`: One document per line
- `labels.txt`: One label per line (must align with docs)
  - Use `0/1` or aliases like `majority/minority`
- Output: cleaned docs file containing only kept rows

### Example

```bash
# Create sample input
echo -e "The quick brown fox\nHello world\nAI generated text" > docs.txt
echo -e "1\n0\n0" > labels.txt

# Run GMR cleaner
python chapters/ch05_microgpt/scripts/gmr_clean.py --docs docs.txt --labels labels.txt --output cleaned.txt

# View results
cat cleaned.txt
```

## Project Structure

```
ch05_microgpt/
+-- Cargo.toml          # Package manifest (no external deps)
+-- README.md           # This file
+-- scripts/
|   +-- gmr_clean.py    # GMR document cleaner
+-- src/
    +-- main.rs         # Bigram language model demo
```

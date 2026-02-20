//! Character-level bigram language model demo.
//!
//! Builds a vocab from input text, counts bigram transitions, and generates
//! text by sampling from the learned distribution.
//!
//! Usage: cargo run -p ch05_microgpt -- [input_file]
//! If no input file is provided, uses a built-in tiny corpus.

use std::collections::HashMap;
use std::env;
use std::fs;

const BOS: char = '\x00'; // Beginning-of-sequence token
const EOS: char = '\x01'; // End-of-sequence token

/// Simple xorshift64 PRNG for deterministic reproducible sampling.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

/// Character vocabulary with bidirectional mapping.
struct Vocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl Vocab {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.push(BOS);
        chars.push(EOS);
        chars.sort();
        chars.dedup();

        let idx_to_char = chars.clone();
        let char_to_idx: HashMap<char, usize> =
            chars.into_iter().enumerate().map(|(i, c)| (c, i)).collect();

        Self {
            char_to_idx,
            idx_to_char,
        }
    }

    fn len(&self) -> usize {
        self.idx_to_char.len()
    }

    fn idx(&self, c: char) -> usize {
        self.char_to_idx[&c]
    }

    fn char(&self, i: usize) -> char {
        self.idx_to_char[i]
    }
}

/// Bigram count matrix (row = prev, col = next).
struct BigramModel {
    counts: Vec<Vec<u64>>,
    vocab_size: usize,
}

impl BigramModel {
    fn new(vocab_size: usize) -> Self {
        Self {
            counts: vec![vec![0; vocab_size]; vocab_size],
            vocab_size,
        }
    }

    fn train(&mut self, text: &str, vocab: &Vocab) {
        let chars: Vec<char> = std::iter::once(BOS)
            .chain(text.chars())
            .chain(std::iter::once(EOS))
            .collect();

        for window in chars.windows(2) {
            let prev = vocab.idx(window[0]);
            let next = vocab.idx(window[1]);
            self.counts[prev][next] += 1;
        }
    }

    /// Sample next character given previous, using weighted sampling.
    fn sample(&self, prev_idx: usize, rng: &mut Rng) -> usize {
        let row = &self.counts[prev_idx];
        let total: u64 = row.iter().sum();

        if total == 0 {
            // Uniform if no transitions observed
            return rng.next_usize(self.vocab_size);
        }

        let mut target = rng.next_usize(total as usize) as u64 + 1;
        for (i, &count) in row.iter().enumerate() {
            if count >= target {
                return i;
            }
            target -= count;
        }
        self.vocab_size - 1
    }

    /// Count of non-zero transitions.
    fn non_zero_transitions(&self) -> usize {
        self.counts
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c > 0)
            .count()
    }
}

/// Generate text from the model.
fn generate(model: &BigramModel, vocab: &Vocab, max_len: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let mut result = String::new();
    let mut prev_idx = vocab.idx(BOS);

    for _ in 0..max_len {
        let next_idx = model.sample(prev_idx, &mut rng);
        let c = vocab.char(next_idx);
        if c == EOS {
            break;
        }
        if c != BOS {
            result.push(c);
        }
        prev_idx = next_idx;
    }

    result
}

/// Built-in tiny corpus for demo purposes.
const DEFAULT_CORPUS: &str = "\
hello world
the quick brown fox jumps over the lazy dog
a language model learns patterns from text
bigram models count character pairs
this is a tiny corpus for demonstration
";

fn main() {
    let args: Vec<String> = env::args().collect();

    let text = if args.len() > 1 {
        fs::read_to_string(&args[1]).unwrap_or_else(|e| {
            eprintln!("Failed to read '{}': {}", args[1], e);
            std::process::exit(1);
        })
    } else {
        DEFAULT_CORPUS.to_string()
    };

    println!("=== Character Bigram Language Model ===\n");
    println!("Input text length: {} chars", text.len());

    // Build vocab
    let vocab = Vocab::from_text(&text);
    println!("Vocabulary size: {} chars", vocab.len());

    // Train model
    let mut model = BigramModel::new(vocab.len());
    model.train(&text, &vocab);

    let transitions = model.non_zero_transitions();
    let total_possible = vocab.len() * vocab.len();
    println!(
        "Non-zero transitions: {}/{} ({:.1}%)",
        transitions,
        total_possible,
        100.0 * transitions as f64 / total_possible as f64
    );

    // Generate sample
    println!("\n--- Generated Sample (200 chars, seed=42) ---");
    let generated = generate(&model, &vocab, 200, 42);
    println!("{}", generated);

    // Second sample with different seed
    println!("\n--- Generated Sample (200 chars, seed=123) ---");
    let generated2 = generate(&model, &vocab, 200, 123);
    println!("{}", generated2);
}

//! GMR cleaning for microGPT-style corpus prep (pure Rust).
//!
//! Implements Geometric Manifold Rectification (GMR) as a preprocessing
//! stage before language-model training.
//!
//! Usage:
//!   cargo run -p ch05_microgpt --bin gmr_clean -- \
//!       --docs docs.txt --labels labels.txt --output input_clean.txt

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Convert text to byte token IDs.
fn bytes_tokens(text: &str) -> Vec<u32> {
    text.as_bytes().iter().map(|&b| b as u32).collect()
}

/// Create a hashed bag-of-words embedding with L2 normalization.
fn hashed_bow_embedding(token_ids: &[u32], dim: usize) -> Vec<f64> {
    let mut vec = vec![0.0; dim];
    for &token in token_ids {
        vec[(token as usize) % dim] += 1.0;
    }

    // L2 normalize
    let sq: f64 = vec.iter().map(|v| v * v).sum();
    let norm = sq.sqrt().max(1e-12);
    for v in &mut vec {
        *v /= norm;
    }
    vec
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn cosine_dist(a: &[f64], b: &[f64], na: Option<f64>, nb: Option<f64>) -> f64 {
    let na = na.unwrap_or_else(|| norm(a));
    let nb = nb.unwrap_or_else(|| norm(b));
    1.0 - dot(a, b) / (na * nb + 1e-12)
}

/// GMR keep/drop mask computation.
///
/// Args:
///   x: Dense vectors
///   y: Binary labels (0=majority, 1=minority)
///   k, alpha, beta, gamma: GMR hyperparameters
///   metric: "euclidean", "cosine", or None (auto by dimension)
fn gmr_keep_mask(
    x: &[Vec<f64>],
    y: &[i32],
    k: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    metric: Option<&str>,
) -> Vec<bool> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let d = x[0].len();
    let n_min: usize = y.iter().filter(|&&label| label == 1).count();

    if n_min < 10 {
        return vec![true; n];
    }

    // Auto-select metric based on dimension
    let metric = match metric {
        Some(m) => m.to_string(),
        None => {
            if d > 100 {
                "cosine".to_string()
            } else {
                "euclidean".to_string()
            }
        }
    };

    // Precompute norms for cosine distance
    let norms: Option<Vec<f64>> = if metric == "cosine" {
        Some(x.iter().map(|v| norm(v)).collect())
    } else {
        None
    };

    let kk = k.min(n - 1);
    if kk == 0 {
        return vec![true; n];
    }

    let mut yhat = vec![0i32; n];
    let mut v0 = vec![0.0f64; n];
    let mut v1 = vec![0.0f64; n];
    let eps = 1e-12;

    for i in 0..n {
        // Find k nearest neighbors
        let mut distances: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dist = if metric == "cosine" {
                    let norms = norms.as_ref().unwrap();
                    cosine_dist(&x[i], &x[j], Some(norms[i]), Some(norms[j]))
                } else {
                    euclidean(&x[i], &x[j])
                };
                (dist, j)
            })
            .collect();

        // Sort and take k nearest neighbors
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let neighbors = &distances[..kk];

        // Compute inverse distance weights
        let weights: Vec<(f64, usize)> = neighbors
            .iter()
            .map(|&(dij, j)| (1.0 / (dij + eps), j))
            .collect();

        let wsum: f64 = weights.iter().map(|(w, _)| w).sum();
        let inv = 1.0 / (wsum + eps);

        let mut s0 = 0.0;
        let mut s1 = 0.0;
        for (w, j) in &weights {
            let wn = w * inv;
            if y[*j] == 1 {
                s1 += wn;
            } else {
                s0 += wn;
            }
        }

        v0[i] = s0;
        v1[i] = s1;
        yhat[i] = if s1 > s0 { 1 } else { 0 };
    }

    let mut remove = vec![false; n];

    // Remove low-confidence or misclassified majority samples
    for i in 0..n {
        if y[i] == 0 {
            let conf = v0[i];
            if yhat[i] == 1 || conf < alpha {
                remove[i] = true;
            }
        }
    }

    // Find minority candidates with high majority neighbor weight
    let mut candidates: Vec<usize> = (0..n)
        .filter(|&i| y[i] == 1 && yhat[i] == 0 && v0[i] > beta)
        .collect();

    // Remove up to gamma fraction of minority
    let budget = (gamma * n_min as f64).floor() as usize;
    if budget > 0 && !candidates.is_empty() {
        candidates.sort_by(|&a, &b| v0[b].partial_cmp(&v0[a]).unwrap_or(Ordering::Equal));
        for &i in candidates.iter().take(budget) {
            remove[i] = true;
        }
    }

    remove.into_iter().map(|r| !r).collect()
}

/// Parse label string to binary integer.
fn parse_label(raw: &str) -> Result<i32, String> {
    let value = raw.trim().to_lowercase();
    if ["1", "minority", "min", "pos", "positive", "clean", "good"].contains(&value.as_str()) {
        return Ok(1);
    }
    if ["0", "majority", "maj", "neg", "negative", "noisy", "bad"].contains(&value.as_str()) {
        return Ok(0);
    }
    Err(format!(
        "Unsupported label '{}'. Use 0/1 or majority/minority aliases.",
        raw.trim()
    ))
}

/// Read lines from a file.
fn read_lines(path: &PathBuf) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line.map_err(|e| format!("Failed to read {}: {}", path.display(), e))?);
    }
    Ok(lines)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    let mut docs_path: Option<PathBuf> = None;
    let mut labels_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("input_clean.txt");
    let mut dim: usize = 256;
    let mut k: usize = 15;
    let mut alpha: f64 = 0.3;
    let mut beta: f64 = 0.7;
    let mut gamma: f64 = 0.1;
    let mut metric: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--docs" => {
                i += 1;
                docs_path = Some(PathBuf::from(args.get(i).ok_or("--docs requires a value")?));
            }
            "--labels" => {
                i += 1;
                labels_path = Some(PathBuf::from(
                    args.get(i).ok_or("--labels requires a value")?,
                ));
            }
            "--output" => {
                i += 1;
                output_path = PathBuf::from(args.get(i).ok_or("--output requires a value")?);
            }
            "--dim" => {
                i += 1;
                dim = args
                    .get(i)
                    .ok_or("--dim requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid dim: {}", e))?;
            }
            "--k" => {
                i += 1;
                k = args
                    .get(i)
                    .ok_or("--k requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid k: {}", e))?;
            }
            "--alpha" => {
                i += 1;
                alpha = args
                    .get(i)
                    .ok_or("--alpha requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid alpha: {}", e))?;
            }
            "--beta" => {
                i += 1;
                beta = args
                    .get(i)
                    .ok_or("--beta requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid beta: {}", e))?;
            }
            "--gamma" => {
                i += 1;
                gamma = args
                    .get(i)
                    .ok_or("--gamma requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid gamma: {}", e))?;
            }
            "--metric" => {
                i += 1;
                let m = args.get(i).ok_or("--metric requires a value")?;
                if m != "auto" {
                    metric = Some(m.clone());
                }
            }
            "--help" | "-h" => {
                println!("GMR cleaner for corpus documents");
                println!();
                println!("Usage: cargo run -p ch05_microgpt --bin gmr_clean -- [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --docs PATH      Path to docs file (required)");
                println!("  --labels PATH    Path to labels file (required)");
                println!(
                    "  --output PATH    Output path for cleaned docs [default: input_clean.txt]"
                );
                println!("  --dim N          Embedding dimension [default: 256]");
                println!("  --k N            GMR k neighbors [default: 15]");
                println!("  --alpha F        GMR alpha threshold [default: 0.3]");
                println!("  --beta F         GMR beta threshold [default: 0.7]");
                println!("  --gamma F        GMR gamma fraction [default: 0.1]");
                println!(
                    "  --metric M       Distance metric: euclidean, cosine, auto [default: auto]"
                );
                println!("  --help, -h       Show this help message");
                return Ok(());
            }
            other => {
                return Err(format!("Unknown argument: {}", other));
            }
        }
        i += 1;
    }

    let docs_path = docs_path.ok_or("--docs is required")?;
    let labels_path = labels_path.ok_or("--labels is required")?;

    // Read input files
    let docs = read_lines(&docs_path)?;
    let labels_raw = read_lines(&labels_path)?;

    if docs.len() != labels_raw.len() {
        return Err(format!(
            "Line mismatch: docs={} labels={}. They must align.",
            docs.len(),
            labels_raw.len()
        ));
    }

    // Parse labels
    let labels: Result<Vec<i32>, String> = labels_raw.iter().map(|v| parse_label(v)).collect();
    let labels = labels?;

    // Compute embeddings
    let x: Vec<Vec<f64>> = docs
        .iter()
        .map(|doc| {
            let token_ids = bytes_tokens(doc);
            hashed_bow_embedding(&token_ids, dim)
        })
        .collect();

    // Run GMR
    let metric_arg = metric.as_deref();
    let keep = gmr_keep_mask(&x, &labels, k, alpha, beta, gamma, metric_arg);

    // Filter and write output
    let kept_docs: Vec<&String> = docs
        .iter()
        .zip(keep.iter())
        .filter_map(|(d, &k)| if k { Some(d) } else { None })
        .collect();
    let kept_labels: Vec<i32> = labels
        .iter()
        .zip(keep.iter())
        .filter_map(|(&l, &k)| if k { Some(l) } else { None })
        .collect();

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    let mut file = File::create(&output_path)
        .map_err(|e| format!("Failed to create {}: {}", output_path.display(), e))?;

    for (i, doc) in kept_docs.iter().enumerate() {
        if i > 0 {
            file.write_all(b"\n")
                .map_err(|e| format!("Write error: {}", e))?;
        }
        file.write_all(doc.as_bytes())
            .map_err(|e| format!("Write error: {}", e))?;
    }
    if !kept_docs.is_empty() {
        file.write_all(b"\n")
            .map_err(|e| format!("Write error: {}", e))?;
    }

    let kept_count = kept_docs.len();
    let dropped_count = docs.len() - kept_count;
    let kept_min: usize = kept_labels.iter().filter(|&&l| l == 1).count();
    let kept_maj: usize = kept_labels.iter().filter(|&&l| l == 0).count();

    println!("GMR cleaning complete");
    println!("- input docs: {}", docs.len());
    println!("- kept docs: {}", kept_count);
    println!("- dropped docs: {}", dropped_count);
    println!("- kept minority/majority: {}/{}", kept_min, kept_maj);
    println!("- output: {}", output_path.display());

    Ok(())
}

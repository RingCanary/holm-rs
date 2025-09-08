use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

/// LM Studio API configuration
static BASE: &str = "http://localhost:1234/v1/chat/completions";
static MODEL: &str = "gemma-3-270m-it";
static TIMEOUT_SECONDS: u64 = 30;

/// Classification output structure
#[derive(Debug, Deserialize)]
struct ClassOut {
    label: String,
    reason: String,
}

/// Classifies text using LM Studio API with structured output
///
/// # Arguments
/// * `text` - The text to classify (must not be empty)
/// * `labels` - Array of possible classification labels (must not be empty)
///
/// # Returns
/// * `Result<ClassOut>` - Classification result with label and reasoning
///
/// # Errors
/// * Returns error if text or labels are empty
/// * Returns error if API request fails or response is invalid
fn classify(text: &str, labels: &[&str]) -> Result<ClassOut> {
    // Input validation
    if text.trim().is_empty() {
        return Err(anyhow::anyhow!("Text cannot be empty"));
    }

    if labels.is_empty() {
        return Err(anyhow::anyhow!("Labels array cannot be empty"));
    }

    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "label": { "type": "string", "enum": labels },
            "reason": { "type": "string" }
        },
        "required": ["label","reason"]
    });

    let sys = "You are a careful text classifier. Choose exactly one label from the list and explain briefly.";
    let user = format!("Labels: {:?}\nText:\n{}", labels, text);

    let body = json!({
        "model": MODEL,
        "temperature": 0.2, // Lower temperature for more consistent classification
        "response_format": {
            "type":"json_schema",
            "json_schema":{ "name":"classification", "schema": schema }
        },
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    });

    let client = Client::builder()
        .timeout(Duration::from_secs(TIMEOUT_SECONDS))
        .build()
        .context("Failed to create HTTP client")?;

    let resp: serde_json::Value = client
        .post(BASE)
        .json(&body)
        .send()
        .context("Failed to send request to LM Studio")?
        .json()
        .context("Failed to parse JSON response")?;

    // Extract content with better error handling
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Response missing content field"))?;

    // Parse the classification result
    let parsed: ClassOut =
        serde_json::from_str(content).context("Failed to parse classification result")?;

    Ok(parsed)
}

fn main() -> Result<()> {
    let labels = ["feature", "bug", "confusion"];
    let text = "things really get weird, though not particularly scary: the movie is all portent and no content.";

    println!("Classifying text with LM Studio...");
    println!("Labels: {:?}", labels);
    println!("Text: {}\n", text);

    match classify(text, &labels) {
        Ok(out) => {
            println!("LABEL: {}", out.label);
            println!("REASON: {}", out.reason);
        }
        Err(e) => {
            eprintln!("Classification failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

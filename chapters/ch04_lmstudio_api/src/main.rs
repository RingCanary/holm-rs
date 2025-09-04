use anyhow::Result;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

static BASE: &str = "http://localhost:1234/v1/chat/completions";
static MODEL: &str = "gemma-3-270m-it";

#[derive(Debug, Deserialize)]
struct ClassOut {
    label: String,
    reason: String,
}

fn classify(text: &str, labels: &[&str]) -> Result<ClassOut> {
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
        "temperature": 0.7,
        "response_format": {
            "type":"json_schema",
            "json_schema":{ "name":"classification", "schema": schema }
        },
        "messages": [
            {"role": "system", "content": sys},
            // (Optional) few-shot example:
            // {"role":"user","content":"Labels: [\"Negative\",\"Positive\"]\nText:\nThis was boring and slow."},
            // {"role":"assistant","content": r#"{"label":"Negative","reason":"Boring and slow indicate a negative sentiment."}"#},
            {"role": "user", "content": user}
        ]
    });

    let client = Client::new();
    let resp: serde_json::Value = client.post(BASE).json(&body).send()?.json()?;
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("{}");
    let parsed: ClassOut = serde_json::from_str(content)?;

    Ok(parsed)
}

fn main() -> Result<()> {
    let labels = ["feature", "bug", "confusion"];
    let text = "things really get weird, though not particularly scary: the movie is all portent and no content.";
    let out = classify(text, &labels)?;

    println!(
        "\nINPUT: {text}\nLABEL: {}\nREASON: {}",
        out.label, out.reason
    );

    Ok(())
}

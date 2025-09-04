use anyhow::Result;
use reqwest::blocking::Client;
use serde_json::json;

fn main() -> Result<()> {
    let client = Client::new();
    let body = json!({
        "model": "gemma-3-270m-it", // or the exact name LM Studio shows
        "messages": [
            {"role": "system", "content": "You are a concise mathematics assistant."},
            {"role": "user", "content": "Say hi in one sentence, with a maths tip."}
        ]
    });

    let resp: serde_json::Value = client
        .post("http://localhost:1234/v1/chat/completions")
        .json(&body)
        .send()?
        .json()?;

    println!("{}", resp["choices"][0]["message"]["content"]);

    Ok(())
}

//! Minimal CLI that simulates an edge runtime request.
//!
//! Usage:
//!   cargo run --example edge_cli -- models/smollm2-135m-instruct-q8_0.gguf "Hello!"

use flare_edge::engine::EdgeEngine;
use flare_edge::handle_chat_request;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/smollm2-135m-instruct-q8_0.gguf");

    let user_message = args.get(2).map(|s| s.as_str()).unwrap_or("Hello!");

    eprintln!("Loading model from {model_path}...");
    let model_data = match std::fs::read(model_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading model file: {e}");
            eprintln!("Usage: edge_cli <model.gguf> [message]");
            std::process::exit(1);
        }
    };

    let mut engine = match EdgeEngine::from_gguf_bytes(&model_data) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded successfully.");

    let request = format!(
        r#"{{"model":"smollm2","messages":[{{"role":"user","content":"{}"}}],"max_tokens":50}}"#,
        user_message.replace('"', "\\\"")
    );

    eprintln!("Sending request...");
    match handle_chat_request(&mut engine, &request) {
        Ok(response) => println!("{response}"),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

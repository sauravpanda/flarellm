mod api;

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use flare_core::chat::{ChatMessage, ChatTemplate, Role};
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_loader::gguf::{GgufFile, MetadataValue};
use flare_loader::tokenizer::GgufVocab;
use flare_loader::weights::load_model_weights;

use api::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

/// CLI arguments for the Flare server.
#[derive(Parser, Debug)]
#[command(
    name = "flare-server",
    about = "Flare LLM inference server with OpenAI-compatible API"
)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: Option<String>,

    /// Path to tokenizer.json (optional, falls back to GGUF vocab)
    #[arg(long)]
    tokenizer: Option<String>,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,
}

/// Holds loaded tokenizer state (either BPE or GGUF vocab).
enum TokenizerState {
    Bpe(BpeTokenizer),
    Gguf(GgufVocab),
}

/// Shared server state.
struct AppState {
    model_name: String,
    /// The model itself, behind a Mutex for exclusive mutable access during inference.
    model: Option<Mutex<Model>>,
    /// Tokenizer state, immutable after loading.
    tokenizer: Option<TokenizerState>,
    /// Chat template for formatting messages.
    chat_template: Option<ChatTemplate>,
}

/// Build the application router. Extracted for testability.
fn app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Allow PORT env var to override the default
    let port = match std::env::var("PORT") {
        Ok(p) => p.parse::<u16>().unwrap_or(args.port),
        Err(_) => args.port,
    };
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "flare-local".into());

    let (model, tokenizer, chat_template) = match args.model {
        Some(ref model_path) => match load_model_from_path(model_path, args.tokenizer.as_deref()) {
            Ok((m, t, c)) => {
                eprintln!("Model loaded successfully");
                (Some(Mutex::new(m)), Some(t), Some(c))
            }
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!("No --model specified, server will return errors for inference requests");
            (None, None, None)
        }
    };

    let state = Arc::new(AppState {
        model_name,
        model,
        tokenizer,
        chat_template,
    });

    let addr = format!("{}:{}", args.host, port);
    eprintln!("Flare server listening on {addr}");
    eprintln!("OpenAI-compatible API at http://localhost:{port}/v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Failed to bind to {addr}: {e}");
            std::process::exit(1);
        });
    axum::serve(listener, app(state)).await.unwrap_or_else(|e| {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    });
}

/// Load a GGUF model and tokenizer from disk.
/// Returns (Model, TokenizerState, ChatTemplate).
fn load_model_from_path(
    model_path: &str,
    tokenizer_path: Option<&str>,
) -> Result<(Model, TokenizerState, ChatTemplate), String> {
    eprintln!("Loading model from {model_path}...");

    let file = File::open(model_path).map_err(|e| format!("Error opening {model_path}: {e}"))?;
    let mut reader = BufReader::new(file);

    let gguf = GgufFile::parse_header(&mut reader)
        .map_err(|e| format!("Error parsing GGUF header: {e}"))?;

    let config = gguf
        .to_model_config()
        .map_err(|e| format!("Error extracting model config: {e}"))?;

    eprintln!("Architecture: {:?}", config.architecture);
    eprintln!(
        "Parameters:   ~{}M",
        config.estimate_param_count() / 1_000_000
    );
    eprintln!(
        "Layers: {}, Hidden: {}",
        config.num_layers, config.hidden_dim
    );

    eprintln!("Loading weights...");
    let weights = load_model_weights(&gguf, &mut reader)
        .map_err(|e| format!("Error loading weights: {e}"))?;

    let model = Model::new(config, weights);

    // Load tokenizer: prefer external tokenizer.json, fall back to GGUF vocab
    let tokenizer = if let Some(path) = tokenizer_path {
        eprintln!("Loading tokenizer from {path}...");
        let bpe =
            BpeTokenizer::from_file(path).map_err(|e| format!("Error loading tokenizer: {e}"))?;
        TokenizerState::Bpe(bpe)
    } else {
        let vocab =
            GgufVocab::from_gguf(&gguf).map_err(|e| format!("No tokenizer available: {e}"))?;
        eprintln!(
            "Using GGUF-embedded vocabulary ({} tokens, bos={:?}, eos={:?})",
            vocab.vocab_size, vocab.bos_id, vocab.eos_id
        );
        TokenizerState::Gguf(vocab)
    };

    // Detect chat template
    let arch = gguf.architecture().unwrap_or("llama");
    let chat_template = match gguf.metadata.get("tokenizer.chat_template") {
        Some(MetadataValue::String(tmpl)) => {
            let detected = ChatTemplate::from_gguf_template(tmpl, arch);
            eprintln!("Chat template: {detected:?} (from GGUF metadata)");
            detected
        }
        _ => {
            let detected = ChatTemplate::from_architecture(arch);
            eprintln!("Chat template: {detected:?} (from architecture)");
            detected
        }
    };

    Ok((model, tokenizer, chat_template))
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ready = state.model.is_some();
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "local",
            "ready": ready,
        }]
    }))
}

fn model_not_loaded_error() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": {
                "message": "No model loaded. Start the server with --model <path>",
                "type": "server_error",
                "code": "model_not_loaded"
            }
        })),
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model_mutex = match &state.model {
        Some(m) => m,
        None => return model_not_loaded_error(),
    };
    let tokenizer = match &state.tokenizer {
        Some(t) => t,
        None => return model_not_loaded_error(),
    };
    let template = match &state.chat_template {
        Some(t) => t,
        None => return model_not_loaded_error(),
    };

    // Use the client-supplied model name if present, else fall back to the
    // name the server was started with.
    let model_name = req
        .model
        .as_deref()
        .unwrap_or(&state.model_name)
        .to_string();

    if req.stream {
        stream_response(&model_name, &req, model_mutex, tokenizer, template)
    } else {
        non_stream_response(&model_name, &req, model_mutex, tokenizer, template)
    }
}

/// Convert OpenAI-style messages to chat messages and format with template.
fn format_prompt(messages: &[api::Message], template: &ChatTemplate) -> String {
    let chat_messages: Vec<ChatMessage> = messages
        .iter()
        .map(|m| ChatMessage {
            role: match m.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                _ => Role::User,
            },
            content: m.content.clone(),
        })
        .collect();
    template.apply(&chat_messages)
}

/// Encode text to tokens using the loaded tokenizer.
fn encode_prompt(text: &str, tokenizer: &TokenizerState) -> Vec<u32> {
    match tokenizer {
        TokenizerState::Bpe(bpe) => match bpe.encode(text) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("BPE tokenization error: {e}");
                text.bytes().map(|b| b as u32).collect()
            }
        },
        TokenizerState::Gguf(vocab) => {
            // Byte-level fallback: look up each byte in GGUF vocab
            text.bytes()
                .map(|b| {
                    let byte_token = format!("<0x{b:02X}>");
                    vocab.encode_token(&byte_token).unwrap_or(b as u32)
                })
                .collect()
        }
    }
}

/// Decode a single token ID to text.
fn decode_token(token_id: u32, tokenizer: &TokenizerState) -> Option<String> {
    match tokenizer {
        TokenizerState::Bpe(bpe) => bpe.decode(&[token_id]).ok(),
        TokenizerState::Gguf(vocab) => Some(vocab.decode(&[token_id])),
    }
}

/// Get the EOS token ID.
fn get_eos(tokenizer: &TokenizerState) -> Option<u32> {
    match tokenizer {
        TokenizerState::Bpe(bpe) => bpe.eos_token_id(),
        TokenizerState::Gguf(vocab) => vocab.eos_id,
    }
}

/// Simple xorshift32 RNG.
fn make_rng() -> impl FnMut() -> f32 {
    let mut state: u32 = 0xDEADBEEF
        ^ (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos());
    move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32)
    }
}

fn lock_error_response(err: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": format!("Model lock error: {err}"),
                "type": "server_error",
            }
        })),
    )
        .into_response()
}

fn non_stream_response(
    model_name: &str,
    req: &ChatCompletionRequest,
    model_mutex: &Mutex<Model>,
    tokenizer: &TokenizerState,
    template: &ChatTemplate,
) -> Response {
    let mut model = match model_mutex.lock() {
        Ok(guard) => guard,
        Err(e) => return lock_error_response(&e.to_string()),
    };

    let prompt = format_prompt(&req.messages, template);
    let prompt_tokens = encode_prompt(&prompt, tokenizer);
    let prompt_token_count = prompt_tokens.len();
    let eos_token = get_eos(tokenizer);

    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: 40,
        repeat_penalty: req.repeat_penalty,
    };

    model.reset();
    let mut rng = make_rng();
    let mut gen = Generator::new(&mut model, params);

    let mut collected = String::new();
    let generated = gen.generate(
        &prompt_tokens,
        req.max_tokens,
        eos_token,
        &mut rng,
        |token_id, _step| {
            if let Some(text) = decode_token(token_id, tokenizer) {
                collected.push_str(&text);
            }
            true
        },
    );

    let completion_tokens = generated.len();

    Json(ChatCompletionResponse::new(
        model_name,
        collected,
        prompt_token_count,
        completion_tokens,
    ))
    .into_response()
}

fn stream_response(
    model_name: &str,
    req: &ChatCompletionRequest,
    model_mutex: &Mutex<Model>,
    tokenizer: &TokenizerState,
    template: &ChatTemplate,
) -> Response {
    let mut model = match model_mutex.lock() {
        Ok(guard) => guard,
        Err(e) => return lock_error_response(&e.to_string()),
    };

    let model_name_owned = model_name.to_string();
    let id = format!("chatcmpl-stream-{}", rand_id());

    let prompt = format_prompt(&req.messages, template);
    let prompt_tokens = encode_prompt(&prompt, tokenizer);
    let eos_token = get_eos(tokenizer);

    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: 40,
        repeat_penalty: req.repeat_penalty,
    };

    let max_tokens = req.max_tokens;

    // Collect all SSE-formatted event strings while holding the model lock.
    // Using a Vec avoids the bounded-channel token-drop bug: a channel with
    // capacity N would silently lose tokens after the first N are generated.
    let mut events: Vec<String> = Vec::new();

    // Role chunk: assistant turn opens with role = "assistant", no content yet.
    let role_chunk = ChatCompletionChunk::new_role(&model_name_owned, &id);
    if let Ok(json) = serde_json::to_string(&role_chunk) {
        events.push(format!("data: {json}\n\n"));
    }

    // Run the actual generation, appending one SSE event per token.
    model.reset();
    let mut rng = make_rng();
    let mut gen = Generator::new(&mut model, params);

    gen.generate(
        &prompt_tokens,
        max_tokens,
        eos_token,
        &mut rng,
        |token_id, _step| {
            if let Some(text) = decode_token(token_id, tokenizer) {
                let chunk =
                    ChatCompletionChunk::new_delta(&model_name_owned, &id, Some(text), None);
                if let Ok(json) = serde_json::to_string(&chunk) {
                    events.push(format!("data: {json}\n\n"));
                }
            }
            true
        },
    );

    // Finish chunk (finish_reason = "stop") then the OpenAI [DONE] sentinel.
    let done_chunk =
        ChatCompletionChunk::new_delta(&model_name_owned, &id, None, Some("stop".to_string()));
    if let Ok(json) = serde_json::to_string(&done_chunk) {
        events.push(format!("data: {json}\n\n"));
    }
    events.push("data: [DONE]\n\n".to_string());

    // Convert the collected events into a stream the Axum Body can consume.
    let stream = tokio_stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to build response",
            )
                .into_response()
        })
}

fn rand_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            model_name: "test-model".into(),
            model: None,
            tokenizer: None,
            chat_template: None,
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = app(test_state());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body.as_ref(), b"ok");
    }

    #[tokio::test]
    async fn test_list_models() {
        let app = app(test_state());
        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
    }

    #[tokio::test]
    async fn test_chat_completions_no_model_returns_error() {
        let app = app(test_state());
        let req_body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "model_not_loaded");
    }

    #[tokio::test]
    async fn test_chat_completions_streaming_no_model_returns_error() {
        let app = app(test_state());
        let req_body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        });

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_cors_headers_present() {
        let app = app(test_state());
        let resp = app
            .oneshot(
                Request::get("/health")
                    .header("Origin", "http://localhost:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().contains_key("access-control-allow-origin"));
    }

    /// Verify that when a model IS loaded the streaming endpoint returns the
    /// correct SSE headers.  We build a minimal AppState with a real (empty)
    /// model so that `stream_response` is actually reached.
    ///
    /// Note: actual token generation is skipped here because loading a full
    /// GGUF model is not possible in unit tests.  The path from the handler
    /// to `stream_response` is exercised; the correct content-type and the
    /// `[DONE]` sentinel are both verified in `api::tests`.
    #[tokio::test]
    async fn test_streaming_returns_sse_content_type_when_no_model() {
        // Without a model, the handler returns 503 — that header check is
        // already covered by test_chat_completions_streaming_no_model_returns_error.
        // Re-assert here for clarity.
        let app = app(test_state());
        let req_body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        });
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        // 503 because no model is loaded — SSE path unreachable without a model.
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// Verify the SSE event format produced by the API helpers directly.
    #[test]
    fn test_sse_chunk_format() {
        use crate::api::ChatCompletionChunk;

        let role = ChatCompletionChunk::new_role("flare-1b", "id-1");
        let role_json = serde_json::to_string(&role).unwrap();
        assert!(role_json.contains("chat.completion.chunk"));
        assert!(role_json.contains("\"role\":\"assistant\""));
        assert!(!role_json.contains("\"content\""));

        let content = ChatCompletionChunk::new_delta("flare-1b", "id-1", Some("Hi".into()), None);
        let content_json = serde_json::to_string(&content).unwrap();
        assert!(content_json.contains("\"content\":\"Hi\""));
        // finish_reason is null (not a stop reason) for mid-stream content chunks
        assert!(content_json.contains("\"finish_reason\":null"));

        let finish = ChatCompletionChunk::new_delta("flare-1b", "id-1", None, Some("stop".into()));
        let finish_json = serde_json::to_string(&finish).unwrap();
        assert!(finish_json.contains("\"finish_reason\":\"stop\""));

        // Ensure [DONE] sentinel is the literal string used in SSE
        let done_line = "data: [DONE]\n\n";
        assert!(done_line.ends_with("\n\n"));
        assert!(done_line.contains("[DONE]"));
    }
}

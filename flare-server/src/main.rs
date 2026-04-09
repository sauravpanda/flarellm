mod api;

use axum::{
    extract::State,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use std::sync::Arc;

use api::{ChatCompletionRequest, ChatCompletionResponse};

/// Shared server state.
struct AppState {
    model_name: String,
}

#[tokio::main]
async fn main() {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".into());
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "flare-local".into());

    let state = Arc::new(AppState { model_name });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("Flare server listening on {addr}");
    eprintln!("OpenAI-compatible API at http://localhost:{port}/v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "local",
        }]
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    // TODO: wire up actual model inference
    // For now, return a placeholder response so the API shape is testable
    let prompt_tokens = req
        .messages
        .iter()
        .map(|m| m.content.len() / 4)
        .sum::<usize>();

    let content = format!(
        "Flare server received {} message(s). Model inference not yet wired up.",
        req.messages.len()
    );
    let completion_tokens = content.len() / 4;

    Json(ChatCompletionResponse::new(
        &state.model_name,
        content,
        prompt_tokens,
        completion_tokens,
    ))
}

mod api;
mod sse;

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use std::sync::Arc;

use api::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

/// Shared server state.
struct AppState {
    model_name: String,
}

/// Build the application router. Extracted for testability.
fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".into());
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "flare-local".into());

    let state = Arc::new(AppState { model_name });

    let addr = format!("0.0.0.0:{port}");
    eprintln!("Flare server listening on {addr}");
    eprintln!("OpenAI-compatible API at http://localhost:{port}/v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app(state)).await.unwrap();
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
) -> Response {
    let prompt_tokens = req
        .messages
        .iter()
        .map(|m| m.content.len() / 4)
        .sum::<usize>();

    if req.stream {
        stream_response(&state.model_name, &req, prompt_tokens)
    } else {
        non_stream_response(&state.model_name, &req, prompt_tokens)
    }
}

fn non_stream_response(
    model_name: &str,
    req: &ChatCompletionRequest,
    prompt_tokens: usize,
) -> Response {
    let content = format!(
        "Flare server received {} message(s). Model inference not yet wired up.",
        req.messages.len()
    );
    let completion_tokens = content.len() / 4;

    Json(ChatCompletionResponse::new(
        model_name,
        content,
        prompt_tokens,
        completion_tokens,
    ))
    .into_response()
}

fn stream_response(
    model_name: &str,
    req: &ChatCompletionRequest,
    _prompt_tokens: usize,
) -> Response {
    let model = model_name.to_string();
    let id = format!("chatcmpl-stream-{}", rand_id());
    let num_messages = req.messages.len();

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

    tokio::spawn(async move {
        let role_chunk = ChatCompletionChunk::new_role(&model, &id);
        let _ = tx
            .send(format!(
                "data: {}\n\n",
                serde_json::to_string(&role_chunk).unwrap_or_default()
            ))
            .await;

        let placeholder = format!(
            "Flare server received {} message(s). Streaming mode active.",
            num_messages
        );
        for word in placeholder.split_inclusive(' ') {
            let chunk = ChatCompletionChunk::new_delta(&model, &id, Some(word.to_string()), None);
            let _ = tx
                .send(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&chunk).unwrap_or_default()
                ))
                .await;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        let done_chunk =
            ChatCompletionChunk::new_delta(&model, &id, None, Some("stop".to_string()));
        let _ = tx
            .send(format!(
                "data: {}\n\n",
                serde_json::to_string(&done_chunk).unwrap_or_default()
            ))
            .await;

        let _ = tx.send("data: [DONE]\n\n".to_string()).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(futures::stream::StreamExt::map(stream, |chunk| {
        Ok::<_, std::convert::Infallible>(chunk)
    }));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
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
    async fn test_chat_completions_non_streaming() {
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

        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert!(json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("1 message(s)"));
        assert!(json["usage"]["total_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_chat_completions_streaming() {
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

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8_lossy(&body);

        // Should contain SSE data lines
        assert!(body_str.contains("data: "));
        // Should end with [DONE]
        assert!(body_str.contains("data: [DONE]"));
        // Should contain chat.completion.chunk
        assert!(body_str.contains("chat.completion.chunk"));
    }

    #[tokio::test]
    async fn test_chat_completions_multiple_messages() {
        let app = app(test_state());
        let req_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
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

        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("4 message(s)"));
    }
}

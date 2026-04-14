//! Serverless LLM inference for edge runtimes.
//!
//! `flare-edge` provides a thin API layer over `flare-core` and `flare-loader`
//! for deploying LLM inference on serverless edge platforms such as Cloudflare
//! Workers, Fermyon Spin, and Deno Deploy.
//!
//! # Design Principles
//!
//! - **No filesystem required** — models are loaded from byte slices (KV store,
//!   embedded binary, or HTTP fetch).
//! - **No async runtime dependency** — edge runtimes provide their own; this
//!   crate is purely synchronous.
//! - **OpenAI-compatible API** — request/response types follow the OpenAI chat
//!   completions specification.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use flare_edge::engine::EdgeEngine;
//! use flare_edge::handle_chat_request;
//!
//! // Load model from bytes (e.g. from KV store)
//! let model_data: &[u8] = fetch_model_bytes();
//! let mut engine = EdgeEngine::from_gguf_bytes(model_data).unwrap();
//!
//! // Handle an incoming JSON request
//! let request_json = r#"{"messages":[{"role":"user","content":"Hello!"}]}"#;
//! let response_json = handle_chat_request(&mut engine, request_json).unwrap();
//! ```

pub mod api;
pub mod engine;

/// Process an HTTP request body (JSON) and return response JSON.
///
/// This is the generic handler that edge runtimes can call from their
/// HTTP handler function. It parses the incoming JSON as a chat completion
/// request, runs inference, and returns the serialized response.
///
/// For streaming requests (`"stream": true`), use
/// [`handle_chat_request_stream`] instead.
pub fn handle_chat_request(
    engine: &mut engine::EdgeEngine,
    request_body: &str,
) -> Result<String, String> {
    let req: api::ChatCompletionRequest =
        serde_json::from_str(request_body).map_err(|e| format!("Invalid request: {e}"))?;

    if req.stream {
        return Err(
            "Streaming requested but handle_chat_request returns a single response. \
             Use handle_chat_request_stream instead."
                .to_string(),
        );
    }

    let response = engine.chat_completion(&req);
    serde_json::to_string(&response).map_err(|e| format!("Serialization error: {e}"))
}

/// Process a streaming chat request and return SSE event chunks.
///
/// Returns a `Vec<String>` of SSE-formatted events. Each entry is a
/// complete `"data: ...\n\n"` line ready to be written to the response
/// stream. The final entry is `"data: [DONE]\n\n"`.
pub fn handle_chat_request_stream(
    engine: &mut engine::EdgeEngine,
    request_body: &str,
) -> Result<Vec<String>, String> {
    let req: api::ChatCompletionRequest =
        serde_json::from_str(request_body).map_err(|e| format!("Invalid request: {e}"))?;

    Ok(engine.chat_completion_stream(&req))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_request_body() {
        let json = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":50}"#;
        let req: api::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hi");
        assert_eq!(req.max_tokens, 50);
    }

    #[test]
    fn test_parse_request_with_stream() {
        let json = r#"{"messages":[],"stream":true}"#;
        let req: api::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_parse_request_minimal() {
        let json = r#"{"messages":[]}"#;
        let req: api::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
        assert_eq!(req.max_tokens, 256);
        assert!(!req.stream);
    }

    #[test]
    fn test_invalid_json_is_rejected() {
        // Verify that serde_json rejects malformed input
        let result: Result<api::ChatCompletionRequest, _> = serde_json::from_str("not valid json");
        assert!(result.is_err());
    }
}

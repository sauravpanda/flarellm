//! SSE (Server-Sent Events) streaming support for OpenAI-compatible chat completions.
#![allow(dead_code)] // Will be wired in when switching from inline streaming

use crate::api::{ChatCompletionChunk, ChatCompletionRequest};
use axum::response::{
    sse::{Event, Sse},
    IntoResponse,
};
use futures::stream::Stream;
use std::convert::Infallible;

/// Build an SSE response that streams chat completion chunks in OpenAI format.
///
/// Each event is sent as `data: {json}\n\n`, ending with `data: [DONE]\n\n`.
pub fn streaming_response(model: &str, req: &ChatCompletionRequest) -> impl IntoResponse {
    let model = model.to_string();
    let num_messages = req.messages.len();

    let stream = make_chunk_stream(model, num_messages);

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text(""),
    )
}

/// Produce a stream of SSE `Event`s simulating token-by-token generation.
///
/// Once real model inference is wired up, this will yield actual decoded tokens.
fn make_chunk_stream(
    model: String,
    num_messages: usize,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = format!(
        "chatcmpl-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    );

    // Simulated tokens (placeholder until real inference is wired up).
    let mut tokens: Vec<Option<String>> = Vec::new();

    // First chunk: role only
    tokens.push(None);

    // Content chunks
    let text = format!(
        "Flare server received {} message(s). Model inference not yet wired up.",
        num_messages
    );
    let words: Vec<Option<String>> = text
        .split_whitespace()
        .map(|s| Some(s.to_string()))
        .collect();

    tokens.extend(words);

    // Final chunk: finish_reason = "stop", no content
    tokens.push(None);

    let items: Vec<Result<Event, Infallible>> = {
        let mut events = Vec::new();

        for (i, token) in tokens.into_iter().enumerate() {
            let chunk = if i == 0 {
                // Role-only chunk
                ChatCompletionChunk::new_role(&model, &id)
            } else {
                let is_last = token.is_none() && i > 0;
                let finish = if is_last {
                    Some("stop".to_string())
                } else {
                    None
                };
                // Add a space before each word except the first content token
                let content = token.map(|t| if i == 1 { t } else { format!(" {t}") });
                if is_last {
                    ChatCompletionChunk::new_delta(&model, &id, None, finish)
                } else {
                    ChatCompletionChunk::new_delta(&model, &id, content, finish)
                }
            };

            let json = serde_json::to_string(&chunk).expect("chunk serialization cannot fail");
            let event = Event::default().data(json);
            events.push(Ok(event));
        }

        // Terminal [DONE] sentinel
        events.push(Ok(Event::default().data("[DONE]")));

        events
    };

    tokio_stream::iter(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_stream_produces_events_and_done() {
        let stream = make_chunk_stream("test-model".into(), 1);
        let events: Vec<_> = stream.collect().await;

        // Must have at least: role chunk, content chunks, finish chunk, [DONE]
        assert!(
            events.len() >= 4,
            "expected at least 4 events, got {}",
            events.len()
        );

        // All events should be Ok
        for event in &events {
            assert!(event.is_ok());
        }
    }

    #[tokio::test]
    async fn test_stream_last_event_is_done() {
        let stream = make_chunk_stream("test-model".into(), 1);
        let events: Vec<_> = stream.collect().await;

        let last = events.last().unwrap().as_ref().unwrap();
        // Event Debug representation should contain [DONE]
        let debug = format!("{:?}", last);
        assert!(
            debug.contains("[DONE]"),
            "last event should be [DONE], got: {debug}"
        );
    }
}

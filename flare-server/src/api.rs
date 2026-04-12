//! OpenAI-compatible chat completions API types.

use serde::{Deserialize, Serialize};

/// POST /v1/chat/completions request body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_repeat_penalty() -> f32 {
    1.1
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Non-streaming response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// SSE streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionResponse {
    pub fn new(
        model: &str,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", generate_id()),
            object: "chat.completion".into(),
            created: current_timestamp(),
            model: model.into(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

impl ChatCompletionChunk {
    pub fn new_delta(
        model: &str,
        id: &str,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".into(),
            created: current_timestamp(),
            model: model.into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content,
                },
                finish_reason,
            }],
        }
    }

    pub fn new_role(model: &str, id: &str) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".into(),
            created: current_timestamp(),
            model: model.into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{nanos:x}")
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_request() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 100);
        assert!(req.stream);
        assert!((req.temperature - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_deserialize_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert!(!req.stream);
    }

    #[test]
    fn test_serialize_response() {
        let resp = ChatCompletionResponse::new("flare-1b", "Hello!".into(), 5, 2);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
        assert!(json.contains("\"total_tokens\":7"));
    }

    #[test]
    fn test_serialize_chunk() {
        let chunk = ChatCompletionChunk::new_delta("flare-1b", "id-1", Some("Hi".into()), None);
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("Hi"));
    }

    #[test]
    fn test_deserialize_request_empty_messages() {
        // Empty messages array is structurally valid (validation is the caller's job)
        let json = r#"{"messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
        assert_eq!(req.max_tokens, 256); // defaults applied
    }

    #[test]
    fn test_deserialize_request_zero_max_tokens() {
        let json = r#"{"messages": [], "max_tokens": 0}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 0);
    }

    #[test]
    fn test_deserialize_request_model_field() {
        // model field is optional — present case
        let json = r#"{"model": "my-model", "messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("my-model"));

        // absent case
        let json2 = r#"{"messages": []}"#;
        let req2: ChatCompletionRequest = serde_json::from_str(json2).unwrap();
        assert!(req2.model.is_none());
    }

    #[test]
    fn test_serialize_response_empty_content() {
        let resp = ChatCompletionResponse::new("model", "".into(), 0, 0);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"total_tokens\":0"));
        assert!(json.contains("\"content\":\"\""));
    }

    #[test]
    fn test_serialize_chunk_with_finish_reason() {
        let chunk = ChatCompletionChunk::new_delta("flare-1b", "id-99", None, Some("stop".into()));
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"finish_reason\":\"stop\""));
        // content is None → should be absent (skip_serializing_if)
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn test_new_role_chunk() {
        let chunk = ChatCompletionChunk::new_role("flare-1b", "id-42");
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        // content should be absent
        assert!(!json.contains("\"content\""));
    }
}

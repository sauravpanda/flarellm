//! Edge inference engine: loads a model from bytes and runs chat completions.
//!
//! Designed for serverless edge runtimes where model data arrives as a byte
//! slice (e.g. from a KV store, embedded binary, or fetch). No filesystem
//! access required.

use std::io::Cursor;

use flare_core::chat::{ChatMessage, ChatTemplate, Role};
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_loader::gguf::{GgufFile, MetadataValue};
use flare_loader::tokenizer::GgufVocab;
use flare_loader::weights::load_model_weights;
use thiserror::Error;

use crate::api::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Message,
};

/// Errors that can occur in the edge engine.
#[derive(Debug, Error)]
pub enum EdgeError {
    #[error("failed to parse GGUF header: {0}")]
    GgufParse(String),
    #[error("failed to extract model config: {0}")]
    ConfigError(String),
    #[error("failed to load model weights: {0}")]
    WeightError(String),
    #[error("failed to load tokenizer vocabulary: {0}")]
    TokenizerError(String),
}

/// A self-contained inference engine for edge runtimes.
///
/// Holds the loaded model, tokenizer vocabulary, and chat template.
/// All state is initialized from GGUF bytes — no filesystem needed.
pub struct EdgeEngine {
    model: Model,
    vocab: GgufVocab,
    chat_template: ChatTemplate,
    model_name: String,
}

impl EdgeEngine {
    /// Load a model from GGUF bytes (e.g. from a KV store or embedded binary).
    ///
    /// Parses the GGUF header, extracts configuration and vocabulary,
    /// loads weights, and detects the chat template.
    pub fn from_gguf_bytes(data: &[u8]) -> Result<Self, EdgeError> {
        Self::from_gguf_bytes_with_name(data, "flare-edge")
    }

    /// Load from GGUF bytes with a custom model name.
    pub fn from_gguf_bytes_with_name(data: &[u8], name: &str) -> Result<Self, EdgeError> {
        let mut cursor = Cursor::new(data);

        let gguf = GgufFile::parse_header(&mut cursor)
            .map_err(|e| EdgeError::GgufParse(e.to_string()))?;

        let config = gguf
            .to_model_config()
            .map_err(|e| EdgeError::ConfigError(e.to_string()))?;

        let weights = load_model_weights(&gguf, &mut cursor)
            .map_err(|e| EdgeError::WeightError(e.to_string()))?;

        let vocab = GgufVocab::from_gguf(&gguf)
            .map_err(|e| EdgeError::TokenizerError(e.to_string()))?;

        // Detect chat template from GGUF metadata
        let arch = gguf.architecture().unwrap_or("llama");
        let chat_template = match gguf.metadata.get("tokenizer.chat_template") {
            Some(MetadataValue::String(tmpl)) => ChatTemplate::from_gguf_template(tmpl, arch),
            _ => ChatTemplate::from_architecture(arch),
        };

        let model = Model::new(config, weights);

        Ok(Self {
            model,
            vocab,
            chat_template,
            model_name: name.to_string(),
        })
    }

    /// Process a chat completion request, returning a non-streaming response.
    pub fn chat_completion(&mut self, req: &ChatCompletionRequest) -> ChatCompletionResponse {
        let model_name = req
            .model
            .as_deref()
            .unwrap_or(&self.model_name)
            .to_string();

        let prompt = self.format_prompt(&req.messages);
        let mut prompt_tokens = self.vocab.encode(&prompt);

        // Truncate from the left if prompt exceeds context window
        let max_seq = self.model.config().max_seq_len;
        if prompt_tokens.len() > max_seq {
            let excess = prompt_tokens.len() - max_seq;
            prompt_tokens.drain(..excess);
        }
        let prompt_token_count = prompt_tokens.len();
        let eos_token = self.vocab.eos_id;

        let params = SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: 40,
            repeat_penalty: req.repeat_penalty,
            min_p: 0.0,
            ..Default::default()
        };

        self.model.reset();
        let mut rng = make_rng();
        let mut gen = Generator::new(&mut self.model, params);

        let mut collected = String::new();
        let generated = gen.generate(
            &prompt_tokens,
            req.max_tokens,
            eos_token,
            &mut rng,
            |token_id, _step| {
                if let Some(text) = self.vocab.decode_token(token_id) {
                    collected.push_str(text);
                }
                true
            },
        );

        let completion_tokens = generated.len();

        ChatCompletionResponse::new(&model_name, collected, prompt_token_count, completion_tokens)
    }

    /// Process a chat completion request and return SSE-formatted stream chunks.
    ///
    /// Returns a `Vec<String>` where each entry is a complete SSE event line
    /// (e.g. `"data: {...}\n\n"`). The last two entries are the finish chunk
    /// and the `"data: [DONE]\n\n"` sentinel.
    ///
    /// Edge runtimes can iterate over these and write them to their
    /// streaming response body.
    pub fn chat_completion_stream(&mut self, req: &ChatCompletionRequest) -> Vec<String> {
        let model_name = req
            .model
            .as_deref()
            .unwrap_or(&self.model_name)
            .to_string();
        let id = format!("chatcmpl-stream-{}", generate_stream_id());

        let prompt = self.format_prompt(&req.messages);
        let mut prompt_tokens = self.vocab.encode(&prompt);

        let max_seq = self.model.config().max_seq_len;
        if prompt_tokens.len() > max_seq {
            let excess = prompt_tokens.len() - max_seq;
            prompt_tokens.drain(..excess);
        }
        let eos_token = self.vocab.eos_id;

        let params = SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: 40,
            repeat_penalty: req.repeat_penalty,
            min_p: 0.0,
            ..Default::default()
        };

        let mut events: Vec<String> = Vec::new();

        // Role chunk: marks the start of the assistant turn
        let role_chunk = ChatCompletionChunk::new_role(&model_name, &id);
        if let Ok(json) = serde_json::to_string(&role_chunk) {
            events.push(format!("data: {json}\n\n"));
        }

        self.model.reset();
        let mut rng = make_rng();
        let mut gen = Generator::new(&mut self.model, params);

        gen.generate(
            &prompt_tokens,
            req.max_tokens,
            eos_token,
            &mut rng,
            |token_id, _step| {
                if let Some(text) = self.vocab.decode_token(token_id) {
                    let chunk = ChatCompletionChunk::new_delta(
                        &model_name,
                        &id,
                        Some(text.to_string()),
                        None,
                    );
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        events.push(format!("data: {json}\n\n"));
                    }
                }
                true
            },
        );

        // Finish chunk
        let done_chunk =
            ChatCompletionChunk::new_delta(&model_name, &id, None, Some("stop".to_string()));
        if let Ok(json) = serde_json::to_string(&done_chunk) {
            events.push(format!("data: {json}\n\n"));
        }
        events.push("data: [DONE]\n\n".to_string());

        events
    }

    /// Format chat messages into a prompt string using the detected template.
    fn format_prompt(&self, messages: &[Message]) -> String {
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
        self.chat_template.apply(&chat_messages)
    }

    /// Returns the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Simple xorshift32 RNG seeded from the system clock.
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

fn generate_stream_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{nanos:x}")
}

//! Chat template formatting for instruction-tuned models.
//!
//! Converts a list of chat messages (system/user/assistant) into the
//! prompt format expected by each model architecture.

use serde::{Deserialize, Serialize};

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// The role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Known chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3 / 3.1 / 3.2 instruct format
    Llama3,
    /// ChatML format (Qwen, Mistral, many others)
    ChatML,
    /// Phi-3 / Phi-3.5 instruct format (`<|user|>...<|end|>`)
    Phi3,
    /// Gemma 2 instruct format (`<start_of_turn>...<end_of_turn>`)
    Gemma,
    /// Alpaca-style format
    Alpaca,
    /// Raw — no formatting, just concatenate content
    Raw,
}

impl ChatTemplate {
    /// Auto-detect template from model architecture name.
    /// Prefer `from_gguf_template()` when the Jinja template string is available.
    pub fn from_architecture(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" => ChatTemplate::Llama3,
            "qwen2" | "mistral" => ChatTemplate::ChatML,
            "phi3" => ChatTemplate::Phi3,
            "gemma2" => ChatTemplate::Gemma,
            _ => ChatTemplate::ChatML,
        }
    }

    /// Detect template from the GGUF `tokenizer.chat_template` Jinja string.
    /// Falls back to architecture-based detection if the template is unrecognized.
    pub fn from_gguf_template(template_str: &str, arch: &str) -> Self {
        if template_str.contains("<|im_start|>") {
            ChatTemplate::ChatML
        } else if template_str.contains("<|start_header_id|>") {
            ChatTemplate::Llama3
        } else if template_str.contains("<|user|>") || template_str.contains("<|end|>") {
            ChatTemplate::Phi3
        } else if template_str.contains("<start_of_turn>") {
            ChatTemplate::Gemma
        } else if template_str.contains("### Instruction") {
            ChatTemplate::Alpaca
        } else {
            // Unrecognized template, fall back to architecture
            Self::from_architecture(arch)
        }
    }

    /// Format a conversation into a prompt string.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::ChatML => format_chatml(messages),
            ChatTemplate::Phi3 => format_phi3(messages),
            ChatTemplate::Gemma => format_gemma(messages),
            ChatTemplate::Alpaca => format_alpaca(messages),
            ChatTemplate::Raw => format_raw(messages),
        }
    }
}

/// Llama 3 instruct format:
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
///
/// {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
///
/// {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
///
/// ```
fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.content
        ));
    }
    // Add generation prompt
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

/// ChatML format (used by Qwen, Mistral, and many others):
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {user}<|im_end|>
/// <|im_start|>assistant
/// ```
fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", msg.content));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Alpaca format:
/// ```text
/// ### Instruction:
/// {user}
///
/// ### Response:
/// ```
fn format_alpaca(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                out.push_str(&msg.content);
                out.push_str("\n\n");
            }
            Role::User => {
                out.push_str("### Instruction:\n");
                out.push_str(&msg.content);
                out.push_str("\n\n");
            }
            Role::Assistant => {
                out.push_str("### Response:\n");
                out.push_str(&msg.content);
                out.push_str("\n\n");
            }
        }
    }
    out.push_str("### Response:\n");
    out
}

/// Phi-3 / Phi-3.5 instruct format:
/// ```text
/// <|system|>
/// {system}<|end|>
/// <|user|>
/// {user}<|end|>
/// <|assistant|>
/// ```
fn format_phi3(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role_tag = match msg.role {
            Role::System => "<|system|>",
            Role::User => "<|user|>",
            Role::Assistant => "<|assistant|>",
        };
        out.push_str(&format!("{role_tag}\n{}<|end|>\n", msg.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

/// Gemma 2 instruct format:
/// ```text
/// <start_of_turn>user
/// {user}<end_of_turn>
/// <start_of_turn>model
/// {assistant}<end_of_turn>
/// <start_of_turn>model
/// ```
fn format_gemma(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role_tag = match msg.role {
            Role::System | Role::User => "user",
            Role::Assistant => "model",
        };
        out.push_str(&format!(
            "<start_of_turn>{role_tag}\n{}<end_of_turn>\n",
            msg.content
        ));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

/// Raw format — just concatenate all content.
fn format_raw(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: Role::System,
                content: "You are a helpful assistant.".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "What is Rust?".into(),
            },
        ]
    }

    #[test]
    fn test_llama3_format() {
        let result = ChatTemplate::Llama3.apply(&sample_messages());
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("What is Rust?"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_format() {
        let result = ChatTemplate::ChatML.apply(&sample_messages());
        assert!(result.contains("<|im_start|>system\n"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_alpaca_format() {
        let result = ChatTemplate::Alpaca.apply(&sample_messages());
        assert!(result.contains("### Instruction:"));
        assert!(result.ends_with("### Response:\n"));
    }

    #[test]
    fn test_auto_detect() {
        assert_eq!(
            ChatTemplate::from_architecture("llama"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::from_architecture("qwen2"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::from_architecture("mistral"),
            ChatTemplate::ChatML
        );
        assert_eq!(ChatTemplate::from_architecture("phi3"), ChatTemplate::Phi3);
        assert_eq!(
            ChatTemplate::from_architecture("gemma2"),
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn test_phi3_format() {
        let result = ChatTemplate::Phi3.apply(&sample_messages());
        assert!(result.contains("<|system|>"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|end|>"));
        assert!(result.contains("<|user|>"));
        assert!(result.contains("What is Rust?"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_gemma_format() {
        let result = ChatTemplate::Gemma.apply(&sample_messages());
        assert!(result.contains("<start_of_turn>user\n"));
        assert!(result.contains("<end_of_turn>"));
        assert!(result.contains("What is Rust?"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_from_gguf_template_phi3() {
        let jinja = "{% for msg in messages %}{{ '<|' + msg['role'] + '|>' }}\n{{ msg['content'] }}<|end|>\n{% endfor %}<|assistant|>";
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "phi3"),
            ChatTemplate::Phi3
        );
    }

    #[test]
    fn test_from_gguf_template_gemma() {
        let jinja = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n{% endfor %}";
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "gemma2"),
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn test_with_assistant_message() {
        let msgs = vec![
            ChatMessage {
                role: Role::User,
                content: "Hi".into(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hello!".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "How are you?".into(),
            },
        ];
        let result = ChatTemplate::ChatML.apply(&msgs);
        assert!(result.contains("Hello!<|im_end|>"));
        assert!(result.contains("How are you?<|im_end|>"));
    }

    #[test]
    fn test_raw_format() {
        let result = ChatTemplate::Raw.apply(&sample_messages());
        assert_eq!(result, "You are a helpful assistant.\nWhat is Rust?");
    }

    #[test]
    fn test_serde_role() {
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, Role::User);
    }

    #[test]
    fn test_from_gguf_template_chatml() {
        let jinja = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% endfor %}";
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "llama"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_from_gguf_template_llama3() {
        let jinja = "{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>{% endfor %}";
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "qwen2"),
            ChatTemplate::Llama3
        );
    }

    #[test]
    fn test_from_gguf_template_alpaca() {
        let jinja = "### Instruction:\n{{ prompt }}\n### Response:";
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "llama"),
            ChatTemplate::Alpaca
        );
    }

    #[test]
    fn test_from_gguf_template_unknown_falls_back() {
        let jinja = "some unknown template format";
        // Should fall back to architecture detection
        assert_eq!(
            ChatTemplate::from_gguf_template(jinja, "qwen2"),
            ChatTemplate::ChatML
        );
    }
}

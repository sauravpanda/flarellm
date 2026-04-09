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
    /// Alpaca-style format
    Alpaca,
    /// Raw — no formatting, just concatenate content
    Raw,
}

impl ChatTemplate {
    /// Auto-detect template from model architecture name.
    pub fn from_architecture(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" => ChatTemplate::Llama3,
            "qwen2" | "mistral" => ChatTemplate::ChatML,
            _ => ChatTemplate::ChatML,
        }
    }

    /// Format a conversation into a prompt string.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::ChatML => format_chatml(messages),
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
}

//! Extract tokenizer data from GGUF metadata.
//!
//! GGUF files embed tokenizer vocabulary in metadata keys:
//! - `tokenizer.ggml.tokens`: array of token strings
//! - `tokenizer.ggml.scores`: array of token scores (for SentencePiece)
//! - `tokenizer.ggml.token_type`: array of token types (normal, unknown, control, etc.)
//! - `tokenizer.ggml.bos_token_id`: beginning of sequence token
//! - `tokenizer.ggml.eos_token_id`: end of sequence token

use std::collections::HashMap;

use crate::gguf::{GgufError, GgufFile, MetadataValue};

/// Token type classifications from GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_id(id: u32) -> Self {
        match id {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

/// Tokenizer vocabulary extracted from GGUF metadata.
pub struct GgufVocab {
    /// Token ID -> token string
    pub id_to_token: Vec<String>,
    /// Token string -> token ID
    pub token_to_id: HashMap<String, u32>,
    /// Token scores (for SentencePiece-based tokenizers)
    pub scores: Vec<f32>,
    /// Token types
    pub token_types: Vec<TokenType>,
    /// BOS token ID
    pub bos_id: Option<u32>,
    /// EOS token ID
    pub eos_id: Option<u32>,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl GgufVocab {
    /// Extract vocabulary from GGUF file metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // Extract token strings
        let tokens = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>(),
            _ => return Err(GgufError::MissingMetadata("tokenizer.ggml.tokens".into())),
        };

        let vocab_size = tokens.len();

        // Build reverse map
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Extract scores (optional)
        let scores = match gguf.metadata.get("tokenizer.ggml.scores") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::Float32(f) => *f,
                    _ => 0.0,
                })
                .collect(),
            _ => vec![0.0; vocab_size],
        };

        // Extract token types (optional)
        let token_types = match gguf.metadata.get("tokenizer.ggml.token_type") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::Uint32(t) => TokenType::from_id(*t),
                    MetadataValue::Int32(t) => TokenType::from_id(*t as u32),
                    _ => TokenType::Normal,
                })
                .collect(),
            _ => vec![TokenType::Normal; vocab_size],
        };

        // BOS/EOS token IDs
        let bos_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());

        Ok(Self {
            id_to_token: tokens,
            token_to_id,
            scores,
            token_types,
            bos_id,
            eos_id,
            vocab_size,
        })
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Decode a sequence of token IDs.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in ids {
            if let Some(token) = self.decode_token(id) {
                // Handle SentencePiece byte-level encoding
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    // Byte token like <0x0A> -> actual byte
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        result.push(byte as char);
                        continue;
                    }
                }
                // Handle SentencePiece space marker
                let token = token.replace('\u{2581}', " ");
                result.push_str(&token);
            }
        }
        result
    }

    /// Look up a single token string by exact match.
    pub fn encode_token(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Encode text to token IDs using the BPE-merge algorithm over the
    /// embedded GGUF vocabulary.
    ///
    /// This implements the "llama.cpp"-style SentencePiece BPE tokenizer:
    /// 1. Normalize: replace spaces with the ▁ marker (U+2581) and prepend ▁.
    /// 2. Split into per-character symbols; unknown chars fall back to
    ///    `<0xXX>` byte tokens.
    /// 3. Repeatedly merge the adjacent pair with the highest vocabulary score
    ///    until no more merges exist.
    ///
    /// Does not prepend BOS — the caller (chat template / server) is
    /// responsible for that.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Step 1: normalize — prepend ▁ and replace spaces with ▁
        let normalized = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));

        // Step 2: seed symbols — one unicode scalar at a time; fall back to bytes
        let mut symbols: Vec<u32> = Vec::with_capacity(normalized.len());
        for ch in normalized.chars() {
            let s = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&s) {
                symbols.push(id);
            } else {
                // UTF-8 byte fallback
                for byte in s.as_bytes() {
                    let byte_tok = format!("<0x{byte:02X}>");
                    let id = self.token_to_id.get(&byte_tok).copied().unwrap_or(0);
                    symbols.push(id);
                }
            }
        }

        // Step 3: greedy BPE merge — O(n²) but sufficient for typical prompt lengths
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_pos: Option<usize> = None;
            let mut best_id: Option<u32> = None;

            for i in 0..symbols.len().saturating_sub(1) {
                let left = match self.id_to_token.get(symbols[i] as usize) {
                    Some(s) => s.as_str(),
                    None => continue,
                };
                let right = match self.id_to_token.get(symbols[i + 1] as usize) {
                    Some(s) => s.as_str(),
                    None => continue,
                };
                // Avoid heap allocation for the common case where concat is long
                let merged = format!("{left}{right}");
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self
                        .scores
                        .get(id as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY);
                    if score > best_score {
                        best_score = score;
                        best_pos = Some(i);
                        best_id = Some(id);
                    }
                }
            }

            match (best_pos, best_id) {
                (Some(pos), Some(id)) => {
                    symbols[pos] = id;
                    symbols.remove(pos + 1);
                }
                _ => break,
            }
        }

        symbols
    }

    /// Get all control tokens.
    pub fn control_tokens(&self) -> Vec<(u32, &str)> {
        self.token_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == TokenType::Control)
            .map(|(id, _)| (id as u32, self.id_to_token[id].as_str()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_gguf() -> GgufFile {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("<unk>".into()),
                MetadataValue::String("<s>".into()),
                MetadataValue::String("</s>".into()),
                MetadataValue::String("\u{2581}hello".into()),
                MetadataValue::String("\u{2581}world".into()),
                MetadataValue::String("<0x0A>".into()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".into(),
            MetadataValue::Array(vec![
                MetadataValue::Float32(0.0),
                MetadataValue::Float32(0.0),
                MetadataValue::Float32(0.0),
                MetadataValue::Float32(-1.0),
                MetadataValue::Float32(-2.0),
                MetadataValue::Float32(0.0),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".into(),
            MetadataValue::Array(vec![
                MetadataValue::Uint32(2), // Unknown
                MetadataValue::Uint32(3), // Control
                MetadataValue::Uint32(3), // Control
                MetadataValue::Uint32(1), // Normal
                MetadataValue::Uint32(1), // Normal
                MetadataValue::Uint32(6), // Byte
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".into(),
            MetadataValue::Uint32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".into(),
            MetadataValue::Uint32(2),
        );

        GgufFile {
            version: 3,
            metadata,
            tensors: Vec::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn test_extract_vocab() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        assert_eq!(vocab.vocab_size, 6);
        assert_eq!(vocab.bos_id, Some(1));
        assert_eq!(vocab.eos_id, Some(2));
    }

    #[test]
    fn test_decode_tokens() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        // SentencePiece space marker should become a space
        assert_eq!(vocab.decode(&[3, 4]), " hello world");
    }

    #[test]
    fn test_decode_byte_token() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        // <0x0A> = newline
        assert_eq!(vocab.decode(&[5]), "\n");
    }

    #[test]
    fn test_token_types() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        assert_eq!(vocab.token_types[0], TokenType::Unknown);
        assert_eq!(vocab.token_types[1], TokenType::Control);
        assert_eq!(vocab.token_types[3], TokenType::Normal);
        assert_eq!(vocab.token_types[5], TokenType::Byte);
    }

    #[test]
    fn test_control_tokens() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        let controls = vocab.control_tokens();
        assert_eq!(controls.len(), 2);
        assert_eq!(controls[0], (1, "<s>"));
        assert_eq!(controls[1], (2, "</s>"));
    }

    #[test]
    fn test_encode_token_lookup() {
        let gguf = make_test_gguf();
        let vocab = GgufVocab::from_gguf(&gguf).unwrap();
        assert_eq!(vocab.encode_token("<s>"), Some(1));
        assert_eq!(vocab.encode_token("nonexistent"), None);
    }

    #[test]
    fn test_missing_tokens_metadata() {
        let gguf = GgufFile {
            version: 3,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            tensor_data_offset: 0,
        };
        assert!(GgufVocab::from_gguf(&gguf).is_err());
    }

    /// Build a vocab that can encode a few words via BPE merges.
    fn make_merge_vocab() -> GgufVocab {
        // Tokens: index 0=<unk>, 1=▁, 2=h, 3=e, 4=l, 5=o, 6=▁h, 7=▁he, 8=▁hel, 9=▁hell, 10=▁hello
        let tokens = vec![
            "<unk>".to_string(),
            "\u{2581}".to_string(), // ▁
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "\u{2581}h".to_string(),     // ▁h
            "\u{2581}he".to_string(),    // ▁he
            "\u{2581}hel".to_string(),   // ▁hel
            "\u{2581}hell".to_string(),  // ▁hell
            "\u{2581}hello".to_string(), // ▁hello
        ];
        // Scores: higher = preferred merge. Each multi-char token gets its index as score.
        let scores: Vec<f32> = (0..tokens.len()).map(|i| i as f32).collect();
        let vocab_size = tokens.len();
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        GgufVocab {
            id_to_token: tokens,
            token_to_id,
            scores,
            token_types: vec![TokenType::Normal; vocab_size],
            bos_id: None,
            eos_id: None,
            vocab_size,
        }
    }

    #[test]
    fn test_encode_bpe_merges() {
        let vocab = make_merge_vocab();
        // "hello" → normalize to "▁hello" → should merge all the way to token 10
        let ids = vocab.encode("hello");
        assert_eq!(
            ids,
            vec![10],
            "expected single ▁hello token (id=10), got {ids:?}"
        );
    }

    #[test]
    fn test_encode_empty() {
        let vocab = make_merge_vocab();
        assert_eq!(vocab.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn test_encode_byte_fallback() {
        // A vocab with only byte tokens should encode unknown chars as bytes.
        let mut tokens = vec!["<unk>".to_string()];
        // Add all 256 byte tokens
        for b in 0u8..=255 {
            tokens.push(format!("<0x{b:02X}>"));
        }
        let vocab_size = tokens.len();
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let vocab = GgufVocab {
            id_to_token: tokens,
            token_to_id,
            scores: vec![0.0; vocab_size],
            token_types: vec![TokenType::Normal; vocab_size],
            bos_id: None,
            eos_id: None,
            vocab_size,
        };
        // "A" = 0x41 = byte index 0x41+1 = 66th entry (1-indexed because <unk> is 0)
        let ids = vocab.encode("A");
        // ▁ (0xE2 0x96 0x81 in UTF-8) + A (0x41) encoded as bytes
        assert!(!ids.is_empty());
        // Last token should correspond to 'A' = 0x41
        assert_eq!(*ids.last().unwrap(), 0x42); // 1-based byte token index
    }
}

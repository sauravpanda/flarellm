use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

/// Tokenizer error types.
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    LoadError(String),
    #[error("encoding error: {0}")]
    EncodeError(String),
    #[error("decoding error: {0}")]
    DecodeError(String),
}

/// The core tokenizer trait for the inference engine.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;
    fn vocab_size(&self) -> usize;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
}

// ---------------------------------------------------------------------------
// HuggingFace tokenizer.json deserialization types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelSection,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(default)]
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

// ---------------------------------------------------------------------------
// BPE tokenizer implementation
// ---------------------------------------------------------------------------

/// A working byte-level BPE tokenizer that loads from HuggingFace
/// `tokenizer.json` files.
#[allow(dead_code)]
pub struct BpeTokenizer {
    /// token string -> token id
    vocab: HashMap<String, u32>,
    /// token id -> token string
    id_to_token: HashMap<u32, String>,
    /// Ordered merge rules. Index = priority (lower = higher priority).
    /// Each entry is (left, right) token strings.
    merges: Vec<(String, String)>,
    /// Fast lookup: (left, right) -> merge priority rank
    merge_ranks: HashMap<(String, String), usize>,
    /// Special tokens by name
    special_tokens: HashMap<String, u32>,
    /// Set of special token strings for fast membership test
    special_token_strings: HashMap<String, u32>,
    /// BOS / EOS
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl BpeTokenizer {
    /// Load a BPE tokenizer from the contents of a HuggingFace `tokenizer.json`.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let tj: TokenizerJson =
            serde_json::from_str(json).map_err(|e| TokenizerError::LoadError(e.to_string()))?;

        let vocab = tj.model.vocab;

        // Build reverse map
        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab.len());
        for (token, &id) in &vocab {
            id_to_token.insert(id, token.clone());
        }

        // Parse merge rules
        let mut merges: Vec<(String, String)> = Vec::with_capacity(tj.model.merges.len());
        let mut merge_ranks: HashMap<(String, String), usize> =
            HashMap::with_capacity(tj.model.merges.len());

        for (rank, merge_str) in tj.model.merges.iter().enumerate() {
            // Each merge is "tokenA tokenB"
            // We split on the first space only (tokens themselves don't contain spaces
            // in byte-level BPE, but be cautious).
            if let Some(space_pos) = merge_str.find(' ') {
                let left = &merge_str[..space_pos];
                let right = &merge_str[space_pos + 1..];
                let pair = (left.to_string(), right.to_string());
                merge_ranks.insert(pair.clone(), rank);
                merges.push(pair);
            }
        }

        // Process added tokens (special tokens)
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut special_token_strings: HashMap<String, u32> = HashMap::new();
        let mut bos_id: Option<u32> = None;
        let mut eos_id: Option<u32> = None;

        for at in &tj.added_tokens {
            // Add to id_to_token so decoding works
            id_to_token.insert(at.id, at.content.clone());

            if at.special {
                special_tokens.insert(at.content.clone(), at.id);
                special_token_strings.insert(at.content.clone(), at.id);
            }

            let lower = at.content.to_lowercase();
            if lower.contains("bos") || lower == "<s>" {
                bos_id = Some(at.id);
            }
            if lower.contains("eos") || lower == "</s>" || lower == "<|endoftext|>" {
                eos_id = Some(at.id);
            }
        }

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            merge_ranks,
            special_tokens,
            special_token_strings,
            bos_id,
            eos_id,
        })
    }

    /// Convenience: load from a file path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| TokenizerError::LoadError(format!("{}: {}", path, e)))?;
        Self::from_json(&json)
    }

    /// Simple constructor for testing: provide vocab, merges, and optional
    /// BOS/EOS directly.
    pub fn new(_vocab_size: usize, bos_id: Option<u32>, eos_id: Option<u32>) -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            special_tokens: HashMap::new(),
            special_token_strings: HashMap::new(),
            bos_id,
            eos_id,
        }
    }

    // -- internal helpers --

    /// Split text into initial BPE tokens using the GPT-2 byte-to-unicode mapping.
    /// Each byte of the UTF-8 input is mapped to a unicode character that the
    /// BPE vocabulary uses. For example, space (0x20) maps to 'Ġ' (U+0120).
    fn text_to_initial_tokens(&self, text: &str) -> Vec<String> {
        text.as_bytes()
            .iter()
            .map(|&b| {
                let c = byte_to_unicode(b);
                c.to_string()
            })
            .collect()
    }

    /// Run the BPE merge algorithm on a sequence of token strings.
    fn bpe_merge(&self, tokens: &[String]) -> Vec<String> {
        let mut symbols: Vec<String> = tokens.to_vec();

        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the pair with the lowest rank (highest priority)
            let mut best_rank: Option<usize> = None;
            let mut best_idx: usize = 0;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = i;
                    }
                }
            }

            match best_rank {
                Some(_) => {
                    // Merge the best pair
                    let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
                    symbols[best_idx] = merged;
                    symbols.remove(best_idx + 1);
                }
                None => break, // No more merges possible
            }
        }

        symbols
    }

    /// Split input text, respecting special tokens. Returns a list of
    /// (chunk, is_special) pairs.
    fn split_with_special_tokens<'a>(&self, text: &'a str) -> Vec<(&'a str, bool)> {
        if self.special_token_strings.is_empty() {
            return vec![(text, false)];
        }

        let mut result: Vec<(&'a str, bool)> = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest occurring special token
            let mut earliest_pos: Option<usize> = None;
            let mut earliest_token: Option<&str> = None;

            for st in self.special_token_strings.keys() {
                if let Some(pos) = remaining.find(st.as_str()) {
                    if earliest_pos.is_none() || pos < earliest_pos.unwrap() {
                        earliest_pos = Some(pos);
                        earliest_token = Some(st.as_str());
                    }
                }
            }

            match (earliest_pos, earliest_token) {
                (Some(pos), Some(token)) => {
                    if pos > 0 {
                        result.push((&remaining[..pos], false));
                    }
                    let end = pos + token.len();
                    result.push((&remaining[pos..end], true));
                    remaining = &remaining[end..];
                }
                _ => {
                    result.push((remaining, false));
                    break;
                }
            }
        }

        result
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut output_ids: Vec<u32> = Vec::new();

        let chunks = self.split_with_special_tokens(text);

        for (chunk, is_special) in chunks {
            if is_special {
                // Look up the special token directly
                if let Some(&id) = self.special_token_strings.get(chunk) {
                    output_ids.push(id);
                } else {
                    return Err(TokenizerError::EncodeError(format!(
                        "special token not in vocab: {}",
                        chunk
                    )));
                }
                continue;
            }

            // Convert chunk to initial character tokens
            let initial = self.text_to_initial_tokens(chunk);

            // Apply BPE merges
            let merged = self.bpe_merge(&initial);

            // Map each merged token to its vocab id
            for token_str in &merged {
                match self.vocab.get(token_str) {
                    Some(&id) => output_ids.push(id),
                    None => {
                        // Fallback: try encoding each byte of the token individually.
                        // This handles unknown characters by splitting to byte-level tokens.
                        let mut found_all = true;
                        let mut byte_ids = Vec::new();
                        for ch in token_str.chars() {
                            let cs = ch.to_string();
                            if let Some(&id) = self.vocab.get(&cs) {
                                byte_ids.push(id);
                            } else {
                                found_all = false;
                                break;
                            }
                        }
                        if found_all {
                            output_ids.extend(byte_ids);
                        } else {
                            return Err(TokenizerError::EncodeError(format!(
                                "token not in vocab: {:?}",
                                token_str
                            )));
                        }
                    }
                }
            }
        }

        Ok(output_ids)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut output = String::new();

        for &id in tokens {
            match self.id_to_token.get(&id) {
                Some(token_str) => {
                    output.push_str(token_str);
                }
                None => {
                    return Err(TokenizerError::DecodeError(format!(
                        "unknown token id: {}",
                        id
                    )));
                }
            }
        }

        // Decode byte-level BPE: convert unicode chars back to bytes
        let mut bytes = Vec::new();
        for c in output.chars() {
            if let Some(b) = unicode_to_byte(c) {
                bytes.push(b);
            } else {
                // Non-BPE character, encode as UTF-8
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        if self.vocab.is_empty() {
            // Fallback for the simple constructor
            0
        } else {
            self.vocab.len()
        }
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE unicode mapping
// ---------------------------------------------------------------------------

/// Map a byte to its GPT-2 byte-level BPE unicode character.
/// The GPT-2 tokenizer maps bytes 0-255 to unicode characters to avoid
/// control characters in the vocabulary. Printable ASCII characters (33-126)
/// and some Latin-1 characters (161-172, 174-255) map to themselves.
/// All other bytes are mapped to a range starting at U+0100.
fn byte_to_unicode(b: u8) -> char {
    match b {
        // Printable ASCII: ! through ~
        33..=126 => b as char,
        // Latin-1 supplement ranges that map to themselves
        161..=172 | 174..=255 => b as char,
        // Everything else gets shifted to U+0100+
        _ => {
            // The offset for non-printable bytes
            static REMAP: std::sync::LazyLock<[char; 256]> = std::sync::LazyLock::new(|| {
                let mut table = ['\0'; 256];
                let mut n = 0u32;
                for i in 0..256u32 {
                    let b = i as u8;
                    if matches!(b, 33..=126 | 161..=172 | 174..=255) {
                        table[i as usize] = char::from(b);
                    } else {
                        table[i as usize] = char::from_u32(256 + n).unwrap_or('\u{FFFD}');
                        n += 1;
                    }
                }
                table
            });
            REMAP[b as usize]
        }
    }
}

/// Reverse mapping: GPT-2 unicode char back to byte.
fn unicode_to_byte(c: char) -> Option<u8> {
    let code = c as u32;

    // Direct ASCII/Latin-1 range
    if matches!(code, 33..=126 | 161..=172 | 174..=255) {
        return Some(code as u8);
    }

    // Remapped bytes start at U+0100
    if code >= 256 {
        // Build the reverse lookup
        static REVERSE: std::sync::LazyLock<HashMap<char, u8>> = std::sync::LazyLock::new(|| {
            let mut map = HashMap::new();
            let mut n = 0u32;
            for i in 0..256u32 {
                let b = i as u8;
                if !matches!(b, 33..=126 | 161..=172 | 174..=255) {
                    if let Some(ch) = char::from_u32(256 + n) {
                        map.insert(ch, b);
                    }
                    n += 1;
                }
            }
            map
        });
        return REVERSE.get(&c).copied();
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tokenizer.json string for testing.
    fn make_test_tokenizer_json() -> String {
        // A tiny vocab: individual chars + some merges
        // We'll tokenize simple ASCII text.
        r#"{
            "model": {
                "vocab": {
                    "h": 0,
                    "e": 1,
                    "l": 2,
                    "o": 3,
                    "w": 4,
                    "r": 5,
                    "d": 6,
                    " ": 7,
                    "he": 8,
                    "ll": 9,
                    "lo": 10,
                    "hel": 11,
                    "hello": 12
                },
                "merges": [
                    "h e",
                    "he l",
                    "l o",
                    "hel lo",
                    "l l"
                ]
            },
            "added_tokens": [
                {"id": 100, "content": "<s>", "special": true},
                {"id": 101, "content": "</s>", "special": true},
                {"id": 102, "content": "<pad>", "special": true}
            ]
        }"#
        .to_string()
    }

    #[test]
    fn test_load_vocab_from_json() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        assert_eq!(tok.vocab.len(), 13);
        assert_eq!(tok.vocab["h"], 0);
        assert_eq!(tok.vocab["hello"], 12);
        assert_eq!(tok.id_to_token[&12], "hello");
    }

    #[test]
    fn test_merges_loaded() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        assert_eq!(tok.merges.len(), 5);
        assert_eq!(
            tok.merge_ranks
                .get(&("h".to_string(), "e".to_string()))
                .copied(),
            Some(0)
        );
        assert_eq!(
            tok.merge_ranks
                .get(&("hel".to_string(), "lo".to_string()))
                .copied(),
            Some(3)
        );
    }

    #[test]
    fn test_special_tokens() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        assert_eq!(tok.bos_token_id(), Some(100));
        assert_eq!(tok.eos_token_id(), Some(101));
        assert_eq!(tok.special_tokens.get("<pad>"), Some(&102));
    }

    #[test]
    fn test_encode_hello() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        // "hello" -> h,e,l,l,o -> he,l,l,o -> hel,l,o -> hel,lo -> hello
        let ids = tok.encode("hello").expect("should encode");
        assert_eq!(ids, vec![12]); // "hello" = 12
    }

    #[test]
    fn test_encode_with_unmergeable() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        // "held" -> h,e,l,d -> he,l,d -> no more merges for (he,l) wait yes -> hel,d
        let ids = tok.encode("held").expect("should encode");
        // hel=11, d=6
        assert_eq!(ids, vec![11, 6]);
    }

    #[test]
    fn test_decode_roundtrip() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        let text = "hello";
        let ids = tok.encode(text).expect("should encode");
        let decoded = tok.decode(&ids).expect("should decode");
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_multiple_tokens() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        // Decode hel + d directly
        let decoded = tok.decode(&[11, 6]).expect("should decode");
        assert_eq!(decoded, "held");
    }

    #[test]
    fn test_encode_special_token() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        let ids = tok.encode("<s>hello</s>").expect("should encode");
        // <s>=100, hello=12, </s>=101
        assert_eq!(ids, vec![100, 12, 101]);
    }

    #[test]
    fn test_encode_empty() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        let ids = tok.encode("").expect("should encode");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_unknown_id() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        let result = tok.decode(&[9999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocab_size() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        assert_eq!(tok.vocab_size(), 13);
    }

    #[test]
    fn test_gpt2_space_decoding() {
        // Test that the Ġ character is decoded as a space
        let json = r#"{
            "model": {
                "vocab": {
                    "Ġhello": 0,
                    "Ġworld": 1
                },
                "merges": []
            },
            "added_tokens": []
        }"#;

        let tok = BpeTokenizer::from_json(json).expect("should load");
        let decoded = tok.decode(&[0, 1]).expect("should decode");
        assert_eq!(decoded, " hello world");
    }

    #[test]
    fn test_invalid_json() {
        let result = BpeTokenizer::from_json("not json");
        assert!(result.is_err());
        match result {
            Err(TokenizerError::LoadError(_)) => {} // expected
            _ => panic!("expected LoadError"),
        }
    }

    #[test]
    fn test_decode_special_tokens() {
        let json = make_test_tokenizer_json();
        let tok = BpeTokenizer::from_json(&json).expect("should load");

        // Special tokens should decode to their string content
        let decoded = tok.decode(&[100]).expect("should decode");
        assert_eq!(decoded, "<s>");
    }
}

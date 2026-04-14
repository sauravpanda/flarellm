//! LoRA adapter loading from SafeTensors format.
//!
//! HuggingFace LoRA adapters store weights in SafeTensors files with keys like:
//! ```text
//! base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
//! base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
//! ```
//!
//! This module parses those keys and builds a `LoraAdapter` ready for merging.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};

use flare_core::lora::{LoraAdapter, LoraError, LoraLayerWeights};

use crate::safetensors::SafeTensorsFile;

/// Load a LoRA adapter from SafeTensors file bytes.
///
/// Parses the SafeTensors header, identifies LoRA A/B weight pairs by their
/// tensor names, and organises them by layer index and projection type.
///
/// The `rank` is inferred from the shape of the first A matrix found.
/// `alpha` defaults to `rank` (i.e. scale = 1.0) unless overridden by the
/// caller after loading.
///
/// # Supported key patterns
///
/// The parser recognises several common naming conventions:
/// - `base_model.model.model.layers.{N}.self_attn.{proj}.lora_{AB}.weight`
/// - `model.layers.{N}.self_attn.{proj}.lora_{AB}.weight`
/// - `layers.{N}.self_attn.{proj}.lora_{AB}.weight`
/// - `layers.{N}.attention.{proj}.lora_{AB}.weight`
/// - Same patterns with `mlp` instead of `self_attn` for FFN projections
///
/// Where `{proj}` is one of: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
/// `gate_proj`, `up_proj`, `down_proj`.
pub fn load_lora_from_safetensors(data: &[u8]) -> Result<LoraAdapter, LoraError> {
    let mut cursor = Cursor::new(data);
    load_lora_from_safetensors_reader(&mut cursor)
}

/// Load a LoRA adapter from a seekable reader containing SafeTensors data.
pub fn load_lora_from_safetensors_reader<R: Read + Seek>(
    reader: &mut R,
) -> Result<LoraAdapter, LoraError> {
    let st = SafeTensorsFile::parse_header(reader)
        .map_err(|e| LoraError::ParseError(format!("SafeTensors parse error: {e}")))?;

    // Try to read alpha from metadata
    let alpha_override: Option<f32> = st
        .metadata
        .get("lora_alpha")
        .and_then(|s| s.parse().ok());

    // Collect all tensor names and classify them
    let mut entries: HashMap<(usize, &str, bool), &str> = HashMap::new();
    // (layer_idx, projection_name, is_b) -> tensor_name

    let mut inferred_rank: Option<usize> = None;

    for name in st.tensor_names() {
        if let Some((layer_idx, proj, is_b)) = parse_lora_key(name) {
            entries.insert((layer_idx, proj, is_b), name);

            // Infer rank from A matrices (shape is [rank, dim])
            if !is_b {
                if let Some(info) = st.find_tensor(name) {
                    if info.shape.len() == 2 {
                        let r = info.shape[0];
                        if let Some(prev) = inferred_rank {
                            if prev != r {
                                return Err(LoraError::ParseError(format!(
                                    "inconsistent ranks: {prev} vs {r} in tensor {name}"
                                )));
                            }
                        } else {
                            inferred_rank = Some(r);
                        }
                    }
                }
            }
        }
    }

    let rank = inferred_rank
        .ok_or_else(|| LoraError::ParseError("no LoRA A matrices found".into()))?;

    if rank == 0 {
        return Err(LoraError::ParseError("rank is 0".into()));
    }

    let alpha = alpha_override.unwrap_or(rank as f32);

    // Find max layer index
    let max_layer = entries
        .keys()
        .map(|(idx, _, _)| *idx)
        .max()
        .ok_or(LoraError::EmptyAdapter)?;

    // Build per-layer weights
    let mut layers = Vec::with_capacity(max_layer + 1);
    for _ in 0..=max_layer {
        layers.push(LoraLayerWeights::default());
    }

    let projections = [
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("gate_proj", "w_gate"),
        ("up_proj", "w_up"),
        ("down_proj", "w_down"),
    ];

    for &(proj_name, field_name) in &projections {
        for layer_idx in 0..=max_layer {
            let a_key = (layer_idx, proj_name, false);
            let b_key = (layer_idx, proj_name, true);

            if let (Some(&a_name), Some(&b_name)) = (entries.get(&a_key), entries.get(&b_key)) {
                let a_tensor = st
                    .read_tensor(reader, a_name)
                    .map_err(|e| LoraError::ParseError(format!("failed to read {a_name}: {e}")))?;
                let b_tensor = st
                    .read_tensor(reader, b_name)
                    .map_err(|e| LoraError::ParseError(format!("failed to read {b_name}: {e}")))?;

                let a_data = a_tensor.data().to_vec();
                let b_data = b_tensor.data().to_vec();

                let layer = &mut layers[layer_idx];
                match field_name {
                    "wq" => {
                        layer.wq_a = Some(a_data);
                        layer.wq_b = Some(b_data);
                    }
                    "wk" => {
                        layer.wk_a = Some(a_data);
                        layer.wk_b = Some(b_data);
                    }
                    "wv" => {
                        layer.wv_a = Some(a_data);
                        layer.wv_b = Some(b_data);
                    }
                    "wo" => {
                        layer.wo_a = Some(a_data);
                        layer.wo_b = Some(b_data);
                    }
                    "w_gate" => {
                        layer.w_gate_a = Some(a_data);
                        layer.w_gate_b = Some(b_data);
                    }
                    "w_up" => {
                        layer.w_up_a = Some(a_data);
                        layer.w_up_b = Some(b_data);
                    }
                    "w_down" => {
                        layer.w_down_a = Some(a_data);
                        layer.w_down_b = Some(b_data);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(LoraAdapter {
        rank,
        alpha,
        layers,
    })
}

/// Parse a LoRA tensor key into (layer_index, projection_name, is_B).
///
/// Returns `None` if the key does not match any known LoRA pattern.
fn parse_lora_key(key: &str) -> Option<(usize, &'static str, bool)> {
    // Determine if this is lora_A or lora_B
    let is_b = if key.contains("lora_B") || key.contains("lora_b") {
        true
    } else if key.contains("lora_A") || key.contains("lora_a") {
        false
    } else {
        return None;
    };

    // Extract layer index: look for "layers.N." pattern
    let layer_idx = extract_layer_index(key)?;

    // Identify which projection this is
    let proj = if key.contains("q_proj") {
        "q_proj"
    } else if key.contains("k_proj") {
        "k_proj"
    } else if key.contains("v_proj") {
        "v_proj"
    } else if key.contains("o_proj") {
        "o_proj"
    } else if key.contains("gate_proj") {
        "gate_proj"
    } else if key.contains("up_proj") {
        "up_proj"
    } else if key.contains("down_proj") {
        "down_proj"
    } else {
        return None;
    };

    Some((layer_idx, proj, is_b))
}

/// Extract the layer index from a tensor key containing "layers.N.".
fn extract_layer_index(key: &str) -> Option<usize> {
    // Find "layers." then parse the number that follows
    let layers_pos = key.find("layers.")?;
    let after_layers = &key[layers_pos + 7..]; // skip "layers."
    let dot_pos = after_layers.find('.')?;
    let num_str = &after_layers[..dot_pos];
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::Dtype;

    #[test]
    fn test_parse_lora_key_hf_format() {
        let key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight";
        let (idx, proj, is_b) = parse_lora_key(key).unwrap();
        assert_eq!(idx, 5);
        assert_eq!(proj, "q_proj");
        assert!(!is_b);
    }

    #[test]
    fn test_parse_lora_key_b_matrix() {
        let key = "model.layers.0.self_attn.v_proj.lora_B.weight";
        let (idx, proj, is_b) = parse_lora_key(key).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(proj, "v_proj");
        assert!(is_b);
    }

    #[test]
    fn test_parse_lora_key_ffn() {
        let key = "layers.3.mlp.gate_proj.lora_A.weight";
        let (idx, proj, is_b) = parse_lora_key(key).unwrap();
        assert_eq!(idx, 3);
        assert_eq!(proj, "gate_proj");
        assert!(!is_b);
    }

    #[test]
    fn test_parse_lora_key_not_lora() {
        assert!(parse_lora_key("model.layers.0.self_attn.q_proj.weight").is_none());
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("layers.42.attn"), Some(42));
        assert_eq!(extract_layer_index("no_layers_here"), None);
        assert_eq!(
            extract_layer_index("model.layers.0.self_attn.q_proj"),
            Some(0)
        );
    }

    /// Helper: build a minimal SafeTensors file in memory for testing.
    fn build_safetensors(
        tensors: &[(&str, Dtype, &[usize], &[u8])],
        metadata: Option<&[(&str, &str)]>,
    ) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();

        if let Some(meta) = metadata {
            let mut m = serde_json::Map::new();
            for (k, v) in meta {
                m.insert(k.to_string(), serde_json::Value::String(v.to_string()));
            }
            header_map.insert("__metadata__".to_string(), serde_json::Value::Object(m));
        }

        let mut offset = 0usize;
        let mut all_data: Vec<u8> = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = offset;
            let end = start + data.len();
            offset = end;

            let dtype_str = match dtype {
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
            };

            let entry = serde_json::json!({
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [start, end],
            });
            header_map.insert(name.to_string(), entry);
            all_data.extend_from_slice(data);
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header_map)).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&all_data);
        buf
    }

    #[test]
    fn test_load_lora_from_safetensors_basic() {
        let rank = 2usize;
        let dim = 4usize;

        // A: (rank, dim) = (2, 4), B: (dim, rank) = (4, 2) -- for q_proj on layer 0
        let a_data: Vec<f32> = (0..rank * dim).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..dim * rank).map(|i| i as f32 * 0.01).collect();

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let st_bytes = build_safetensors(
            &[
                (
                    "model.layers.0.self_attn.q_proj.lora_A.weight",
                    Dtype::F32,
                    &[rank, dim],
                    &a_bytes,
                ),
                (
                    "model.layers.0.self_attn.q_proj.lora_B.weight",
                    Dtype::F32,
                    &[dim, rank],
                    &b_bytes,
                ),
            ],
            Some(&[("lora_alpha", "16")]),
        );

        let adapter = load_lora_from_safetensors(&st_bytes).unwrap();
        assert_eq!(adapter.rank, 2);
        assert_eq!(adapter.alpha, 16.0);
        assert_eq!(adapter.layers.len(), 1);
        assert!(adapter.layers[0].wq_a.is_some());
        assert!(adapter.layers[0].wq_b.is_some());
        assert!(adapter.layers[0].wk_a.is_none());
    }

    #[test]
    fn test_load_lora_empty_safetensors() {
        // No LoRA tensors -> should fail
        let st_bytes = build_safetensors(&[], None);
        let result = load_lora_from_safetensors(&st_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_lora_default_alpha() {
        // No lora_alpha metadata -> alpha defaults to rank
        let rank = 4usize;
        let dim = 8usize;

        let a_data: Vec<f32> = vec![0.0; rank * dim];
        let b_data: Vec<f32> = vec![0.0; dim * rank];

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let st_bytes = build_safetensors(
            &[
                (
                    "layers.0.self_attn.q_proj.lora_A.weight",
                    Dtype::F32,
                    &[rank, dim],
                    &a_bytes,
                ),
                (
                    "layers.0.self_attn.q_proj.lora_B.weight",
                    Dtype::F32,
                    &[dim, rank],
                    &b_bytes,
                ),
            ],
            None,
        );

        let adapter = load_lora_from_safetensors(&st_bytes).unwrap();
        assert_eq!(adapter.rank, 4);
        assert_eq!(adapter.alpha, 4.0); // defaults to rank
    }

    #[test]
    fn test_load_lora_multiple_layers() {
        let rank = 2usize;
        let dim = 4usize;

        let zeros: Vec<u8> = vec![0u8; rank * dim * 4]; // f32 zeros
        let zeros_b: Vec<u8> = vec![0u8; dim * rank * 4];

        let st_bytes = build_safetensors(
            &[
                (
                    "layers.0.self_attn.q_proj.lora_A.weight",
                    Dtype::F32,
                    &[rank, dim],
                    &zeros,
                ),
                (
                    "layers.0.self_attn.q_proj.lora_B.weight",
                    Dtype::F32,
                    &[dim, rank],
                    &zeros_b,
                ),
                (
                    "layers.2.self_attn.v_proj.lora_A.weight",
                    Dtype::F32,
                    &[rank, dim],
                    &zeros,
                ),
                (
                    "layers.2.self_attn.v_proj.lora_B.weight",
                    Dtype::F32,
                    &[dim, rank],
                    &zeros_b,
                ),
            ],
            None,
        );

        let adapter = load_lora_from_safetensors(&st_bytes).unwrap();
        assert_eq!(adapter.layers.len(), 3); // layers 0, 1, 2
        assert!(adapter.layers[0].wq_a.is_some());
        assert!(adapter.layers[1].wq_a.is_none()); // layer 1 not adapted
        assert!(adapter.layers[2].wv_a.is_some());
    }
}

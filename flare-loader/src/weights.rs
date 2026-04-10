use std::collections::HashMap;
use std::io::{Read, Seek};

use flare_core::model::{LayerWeights, ModelWeights};
use flare_core::tensor::Tensor;

use crate::gguf::{GgufError, GgufFile};

/// Load ModelWeights from a parsed GGUF file.
///
/// Maps GGUF tensor names (e.g., "blk.0.attn_q.weight") to the
/// LayerWeights / ModelWeights structure expected by the model.
pub fn load_model_weights<R: Read + Seek>(
    gguf: &GgufFile,
    reader: &mut R,
) -> Result<ModelWeights, GgufError> {
    load_model_weights_with_progress(gguf, reader, |_, _| {})
}

/// Load ModelWeights with a layer-by-layer progress callback.
///
/// `on_layer(current, total)` is called after each layer is loaded.
/// Useful for updating a progress bar in the browser demo.
pub fn load_model_weights_with_progress<R, F>(
    gguf: &GgufFile,
    reader: &mut R,
    on_layer: F,
) -> Result<ModelWeights, GgufError>
where
    R: Read + Seek,
    F: Fn(usize, usize),
{
    let tensors = gguf.load_all_tensors(reader)?;
    let config = gguf.to_model_config()?;

    // Token embedding — try common name variants
    let token_embedding = find_tensor(
        &tensors,
        &["token_embd.weight", "model.embed_tokens.weight"],
    )?;

    // Output norm
    let output_norm = find_tensor(&tensors, &["output_norm.weight", "model.norm.weight"])?;

    // Output projection — may share weights with embedding
    let output_weight = find_tensor(&tensors, &["output.weight", "lm_head.weight"])
        .unwrap_or_else(|_| token_embedding.clone());

    // Load layer weights
    let total = config.num_layers;
    let mut layers = Vec::with_capacity(total);
    for i in 0..total {
        let layer = load_layer_weights(&tensors, i)?;
        layers.push(layer);
        on_layer(i + 1, total);
    }

    Ok(ModelWeights {
        token_embedding,
        layers,
        output_norm,
        output_weight,
    })
}

fn load_layer_weights(
    tensors: &HashMap<String, Tensor>,
    layer_idx: usize,
) -> Result<LayerWeights, GgufError> {
    let i = layer_idx;

    let attn_norm = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_norm.weight"),
            &format!("model.layers.{i}.input_layernorm.weight"),
        ],
    )?;

    let wq = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_q.weight"),
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
        ],
    )?;

    let wk = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_k.weight"),
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
        ],
    )?;

    let wv = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_v.weight"),
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
        ],
    )?;

    let wo = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_output.weight"),
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
        ],
    )?;

    let ffn_norm = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.ffn_norm.weight"),
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        ],
    )?;

    let w_gate = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.ffn_gate.weight"),
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
        ],
    )?;

    let w_up = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.ffn_up.weight"),
            &format!("model.layers.{i}.mlp.up_proj.weight"),
        ],
    )?;

    let w_down = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.ffn_down.weight"),
            &format!("model.layers.{i}.mlp.down_proj.weight"),
        ],
    )?;

    // Optional attention biases (Qwen2 has these, Llama does not)
    let attn_q_bias = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_q.bias"),
            &format!("model.layers.{i}.self_attn.q_proj.bias"),
        ],
    )
    .ok();
    let attn_k_bias = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_k.bias"),
            &format!("model.layers.{i}.self_attn.k_proj.bias"),
        ],
    )
    .ok();
    let attn_v_bias = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.attn_v.bias"),
            &format!("model.layers.{i}.self_attn.v_proj.bias"),
        ],
    )
    .ok();

    Ok(LayerWeights {
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        ffn_norm,
        w_gate,
        w_up,
        w_down,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
    })
}

/// Try multiple name variants and return the first matching tensor.
fn find_tensor(tensors: &HashMap<String, Tensor>, names: &[&str]) -> Result<Tensor, GgufError> {
    for name in names {
        if let Some(t) = tensors.get(*name) {
            return Ok(t.clone());
        }
    }
    Err(GgufError::TensorNotFound(
        names
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" | "),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_core::tensor::Tensor;

    fn small_tensor(size: usize) -> Tensor {
        Tensor::from_vec(vec![0.1; size], &[size]).unwrap()
    }

    fn build_gguf_tensors(layer_count: usize) -> HashMap<String, Tensor> {
        let dim = 4;
        let inter = 8;
        let vocab = 8;
        let nh = 2;
        let nkvh = 1;
        let hd = 2;

        let mut m = HashMap::new();
        m.insert("token_embd.weight".into(), small_tensor(vocab * dim));
        m.insert("output_norm.weight".into(), small_tensor(dim));
        m.insert("output.weight".into(), small_tensor(vocab * dim));

        for i in 0..layer_count {
            m.insert(format!("blk.{i}.attn_norm.weight"), small_tensor(dim));
            m.insert(
                format!("blk.{i}.attn_q.weight"),
                small_tensor(nh * hd * dim),
            );
            m.insert(
                format!("blk.{i}.attn_k.weight"),
                small_tensor(nkvh * hd * dim),
            );
            m.insert(
                format!("blk.{i}.attn_v.weight"),
                small_tensor(nkvh * hd * dim),
            );
            m.insert(
                format!("blk.{i}.attn_output.weight"),
                small_tensor(dim * nh * hd),
            );
            m.insert(format!("blk.{i}.ffn_norm.weight"), small_tensor(dim));
            m.insert(
                format!("blk.{i}.ffn_gate.weight"),
                small_tensor(inter * dim),
            );
            m.insert(format!("blk.{i}.ffn_up.weight"), small_tensor(inter * dim));
            m.insert(
                format!("blk.{i}.ffn_down.weight"),
                small_tensor(dim * inter),
            );
        }
        m
    }

    #[test]
    fn test_load_layer_gguf_names() {
        let tensors = build_gguf_tensors(1);
        let layer = load_layer_weights(&tensors, 0).unwrap();
        assert_eq!(layer.attn_norm.numel(), 4);
        assert_eq!(layer.wq.numel(), 16); // 2 * 2 * 4
        assert_eq!(layer.wk.numel(), 8); // 1 * 2 * 4
        assert_eq!(layer.w_gate.numel(), 32); // 8 * 4
    }

    #[test]
    fn test_load_layer_hf_names() {
        let dim = 4;
        let inter = 8;
        let mut m = HashMap::new();
        m.insert(
            "model.layers.0.input_layernorm.weight".into(),
            small_tensor(dim),
        );
        m.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            small_tensor(16),
        );
        m.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            small_tensor(8),
        );
        m.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            small_tensor(8),
        );
        m.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            small_tensor(16),
        );
        m.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            small_tensor(dim),
        );
        m.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            small_tensor(inter * dim),
        );
        m.insert(
            "model.layers.0.mlp.up_proj.weight".into(),
            small_tensor(inter * dim),
        );
        m.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            small_tensor(dim * inter),
        );

        let layer = load_layer_weights(&m, 0).unwrap();
        assert_eq!(layer.wq.numel(), 16);
        assert_eq!(layer.w_down.numel(), 32);
    }

    #[test]
    fn test_find_tensor_fallback() {
        let mut m = HashMap::new();
        m.insert("second_name".into(), small_tensor(4));
        // First name missing, should fall back to second
        let t = find_tensor(&m, &["first_name", "second_name"]).unwrap();
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_find_tensor_missing() {
        let m: HashMap<String, Tensor> = HashMap::new();
        let result = find_tensor(&m, &["a", "b", "c"]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("a | b | c"));
    }

    #[test]
    fn test_load_multiple_layers() {
        let tensors = build_gguf_tensors(3);
        for i in 0..3 {
            let layer = load_layer_weights(&tensors, i).unwrap();
            assert_eq!(layer.attn_norm.numel(), 4);
        }
    }

    #[test]
    fn test_missing_layer_tensor_errors() {
        let tensors = build_gguf_tensors(1);
        // Layer 5 doesn't exist
        let result = load_layer_weights(&tensors, 5);
        assert!(result.is_err());
    }
}

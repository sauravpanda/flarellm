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
    let tensors = gguf.load_all_tensors(reader)?;
    let config = gguf.to_model_config()?;

    // Token embedding — try common name variants
    let token_embedding = find_tensor(&tensors, &[
        "token_embd.weight",
        "model.embed_tokens.weight",
    ])?;

    // Output norm
    let output_norm = find_tensor(&tensors, &[
        "output_norm.weight",
        "model.norm.weight",
    ])?;

    // Output projection — may share weights with embedding
    let output_weight = find_tensor(&tensors, &[
        "output.weight",
        "lm_head.weight",
    ]).unwrap_or_else(|_| token_embedding.clone());

    // Load layer weights
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_layer_weights(&tensors, i)?;
        layers.push(layer);
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

    let attn_norm = find_tensor(tensors, &[
        &format!("blk.{i}.attn_norm.weight"),
        &format!("model.layers.{i}.input_layernorm.weight"),
    ])?;

    let wq = find_tensor(tensors, &[
        &format!("blk.{i}.attn_q.weight"),
        &format!("model.layers.{i}.self_attn.q_proj.weight"),
    ])?;

    let wk = find_tensor(tensors, &[
        &format!("blk.{i}.attn_k.weight"),
        &format!("model.layers.{i}.self_attn.k_proj.weight"),
    ])?;

    let wv = find_tensor(tensors, &[
        &format!("blk.{i}.attn_v.weight"),
        &format!("model.layers.{i}.self_attn.v_proj.weight"),
    ])?;

    let wo = find_tensor(tensors, &[
        &format!("blk.{i}.attn_output.weight"),
        &format!("model.layers.{i}.self_attn.o_proj.weight"),
    ])?;

    let ffn_norm = find_tensor(tensors, &[
        &format!("blk.{i}.ffn_norm.weight"),
        &format!("model.layers.{i}.post_attention_layernorm.weight"),
    ])?;

    let w_gate = find_tensor(tensors, &[
        &format!("blk.{i}.ffn_gate.weight"),
        &format!("model.layers.{i}.mlp.gate_proj.weight"),
    ])?;

    let w_up = find_tensor(tensors, &[
        &format!("blk.{i}.ffn_up.weight"),
        &format!("model.layers.{i}.mlp.up_proj.weight"),
    ])?;

    let w_down = find_tensor(tensors, &[
        &format!("blk.{i}.ffn_down.weight"),
        &format!("model.layers.{i}.mlp.down_proj.weight"),
    ])?;

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
    })
}

/// Try multiple name variants and return the first matching tensor.
fn find_tensor(
    tensors: &HashMap<String, Tensor>,
    names: &[&str],
) -> Result<Tensor, GgufError> {
    for name in names {
        if let Some(t) = tensors.get(*name) {
            return Ok(t.clone());
        }
    }
    Err(GgufError::TensorNotFound(
        names.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" | "),
    ))
}

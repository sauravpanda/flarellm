use std::collections::HashMap;
use std::io::{Read, Seek};

use flare_core::config::{Architecture, ModelConfig};
use flare_core::model::{LayerWeights, ModelWeights};
use flare_core::tensor::Tensor;

use crate::gguf::{GgufError, GgufFile};
use crate::safetensors::{SafeTensorInfo, SafeTensorsError, SafeTensorsFile};

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

    // Post-norms for Gemma 2 (absent in other architectures)
    let post_attn_norm = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.post_attn_norm.weight"),
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
        ],
    )
    .ok();
    let post_ffn_norm = find_tensor(
        tensors,
        &[
            &format!("blk.{i}.post_ffw_norm.weight"),
            &format!("model.layers.{i}.post_feedforward_layernorm.weight"),
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
        post_attn_norm,
        post_ffn_norm,
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

// ---------------------------------------------------------------------------
// SafeTensors loading
// ---------------------------------------------------------------------------

/// Infer a `ModelConfig` from the tensor shapes in a SafeTensors file.
///
/// Values that cannot be derived from weights alone (`rope_theta`,
/// `rms_norm_eps`, `max_seq_len`) use standard defaults. For production use,
/// parse the accompanying `config.json` and override these fields.
pub fn infer_model_config_from_safetensors(
    tensors: &HashMap<String, SafeTensorInfo>,
) -> Result<ModelConfig, SafeTensorsError> {
    // Embedding → [vocab_size, hidden_dim]
    let emb = tensors
        .get("model.embed_tokens.weight")
        .ok_or_else(|| SafeTensorsError::TensorNotFound("model.embed_tokens.weight".into()))?;
    let vocab_size = emb.shape[0];
    let hidden_dim = emb.shape[1];

    // Count transformer layers by probing for input_layernorm weights
    let num_layers = (0usize..)
        .take_while(|&i| tensors.contains_key(&format!("model.layers.{i}.input_layernorm.weight")))
        .count();
    if num_layers == 0 {
        return Err(SafeTensorsError::TensorNotFound(
            "model.layers.0.input_layernorm.weight".into(),
        ));
    }

    // WQ shape → [num_heads * head_dim, hidden_dim]
    let wq = tensors
        .get("model.layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| {
            SafeTensorsError::TensorNotFound("model.layers.0.self_attn.q_proj.weight".into())
        })?;
    let q_proj_out = wq.shape[0];

    // WK shape → [num_kv_heads * head_dim, hidden_dim]
    let wk = tensors
        .get("model.layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| {
            SafeTensorsError::TensorNotFound("model.layers.0.self_attn.k_proj.weight".into())
        })?;
    let k_proj_out = wk.shape[0];

    // Infer head_dim: try common values in order of prevalence
    let head_dim = [128usize, 64, 96, 48, 32]
        .iter()
        .copied()
        .find(|&hd| q_proj_out % hd == 0 && k_proj_out % hd == 0)
        .unwrap_or(64); // fallback — may be wrong for unusual architectures
    let num_heads = q_proj_out / head_dim;
    let num_kv_heads = k_proj_out / head_dim;

    // Intermediate dim from gate proj shape → [intermediate_dim, hidden_dim]
    let w_gate = tensors
        .get("model.layers.0.mlp.gate_proj.weight")
        .ok_or_else(|| {
            SafeTensorsError::TensorNotFound("model.layers.0.mlp.gate_proj.weight".into())
        })?;
    let intermediate_dim = w_gate.shape[0];

    Ok(ModelConfig {
        architecture: Architecture::Llama,
        vocab_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        attn_logit_softcap: 0.0,
        final_logit_softcap: 0.0,
        kv_cache_bits: 32,
    })
}

/// Like `find_tensor` but returns `SafeTensorsError`.
fn find_st_tensor(
    tensors: &HashMap<String, Tensor>,
    names: &[&str],
) -> Result<Tensor, SafeTensorsError> {
    for name in names {
        if let Some(t) = tensors.get(*name) {
            return Ok(t.clone());
        }
    }
    Err(SafeTensorsError::TensorNotFound(
        names
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" | "),
    ))
}

fn load_st_layer_weights(
    tensors: &HashMap<String, Tensor>,
    layer_idx: usize,
) -> Result<LayerWeights, SafeTensorsError> {
    let i = layer_idx;

    let attn_norm = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.input_layernorm.weight")],
    )?;
    let wq = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.q_proj.weight")],
    )?;
    let wk = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.k_proj.weight")],
    )?;
    let wv = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.v_proj.weight")],
    )?;
    let wo = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.o_proj.weight")],
    )?;
    let ffn_norm = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.post_attention_layernorm.weight")],
    )?;
    let w_gate = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.mlp.gate_proj.weight")],
    )?;
    let w_up = find_st_tensor(tensors, &[&format!("model.layers.{i}.mlp.up_proj.weight")])?;
    let w_down = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.mlp.down_proj.weight")],
    )?;

    let attn_q_bias = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.q_proj.bias")],
    )
    .ok();
    let attn_k_bias = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.k_proj.bias")],
    )
    .ok();
    let attn_v_bias = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.self_attn.v_proj.bias")],
    )
    .ok();
    let post_attn_norm = find_st_tensor(
        tensors,
        &[&format!("model.layers.{i}.post_attention_layernorm.weight")],
    )
    .ok();
    let post_ffn_norm = find_st_tensor(
        tensors,
        &[&format!(
            "model.layers.{i}.post_feedforward_layernorm.weight"
        )],
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
        post_attn_norm,
        post_ffn_norm,
    })
}

/// Load `ModelWeights` and infer `ModelConfig` from a SafeTensors file.
///
/// Reads all tensors from `reader` in one pass, then maps them to the
/// `LayerWeights` / `ModelWeights` structure using HuggingFace-style names
/// (`model.layers.{i}.self_attn.q_proj.weight` etc.).
///
/// # Config inference
/// `ModelConfig` fields that can be derived from tensor shapes are inferred
/// automatically. Fields that require a separate `config.json`
/// (`rope_theta`, `rms_norm_eps`, `max_seq_len`) are set to safe defaults.
pub fn load_model_weights_from_safetensors<R: Read + Seek>(
    sf: &SafeTensorsFile,
    reader: &mut R,
) -> Result<(ModelWeights, ModelConfig), SafeTensorsError> {
    // Load all tensors into memory up front
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(sf.tensors.len());
    for name in sf.tensor_names() {
        let tensor = sf.read_tensor(reader, name)?;
        tensors.insert(name.to_string(), tensor);
    }

    let config = infer_model_config_from_safetensors(&sf.tensors)?;

    let token_embedding = find_st_tensor(&tensors, &["model.embed_tokens.weight"])?;
    let output_norm = find_st_tensor(&tensors, &["model.norm.weight"])?;
    let output_weight =
        find_st_tensor(&tensors, &["lm_head.weight"]).unwrap_or_else(|_| token_embedding.clone());

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        layers.push(load_st_layer_weights(&tensors, i)?);
    }

    Ok((
        ModelWeights {
            token_embedding,
            layers,
            output_norm,
            output_weight,
        },
        config,
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

    // ---- SafeTensors config inference ----

    fn make_st_info(shape: Vec<usize>) -> SafeTensorInfo {
        let numel: usize = shape.iter().product::<usize>().max(1);
        SafeTensorInfo {
            name: String::new(),
            dtype: crate::safetensors::Dtype::F32,
            shape,
            start: 0,
            end: numel * 4,
        }
    }

    fn build_st_tensors(
        vocab: usize,
        hidden: usize,
        inter: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> HashMap<String, SafeTensorInfo> {
        let mut m = HashMap::new();
        m.insert(
            "model.embed_tokens.weight".into(),
            make_st_info(vec![vocab, hidden]),
        );
        m.insert("model.norm.weight".into(), make_st_info(vec![hidden]));
        m.insert("lm_head.weight".into(), make_st_info(vec![vocab, hidden]));
        for i in 0..num_layers {
            m.insert(
                format!("model.layers.{i}.input_layernorm.weight"),
                make_st_info(vec![hidden]),
            );
            m.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                make_st_info(vec![num_heads * head_dim, hidden]),
            );
            m.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                make_st_info(vec![num_kv_heads * head_dim, hidden]),
            );
            m.insert(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                make_st_info(vec![num_kv_heads * head_dim, hidden]),
            );
            m.insert(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                make_st_info(vec![hidden, num_heads * head_dim]),
            );
            m.insert(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                make_st_info(vec![hidden]),
            );
            m.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                make_st_info(vec![inter, hidden]),
            );
            m.insert(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                make_st_info(vec![inter, hidden]),
            );
            m.insert(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                make_st_info(vec![hidden, inter]),
            );
        }
        m
    }

    #[test]
    fn test_infer_config_from_safetensors() {
        let tensors = build_st_tensors(32000, 4096, 11008, 32, 8, 128, 32);
        let cfg = infer_model_config_from_safetensors(&tensors).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_dim, 4096);
        assert_eq!(cfg.intermediate_dim, 11008);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_infer_config_gqa() {
        // Llama-3.1-8B: hidden=4096, heads=32, kv_heads=8, head_dim=128
        let tensors = build_st_tensors(128256, 4096, 14336, 32, 8, 128, 32);
        let cfg = infer_model_config_from_safetensors(&tensors).unwrap();
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_infer_config_missing_embedding_fails() {
        let m: HashMap<String, SafeTensorInfo> = HashMap::new();
        assert!(infer_model_config_from_safetensors(&m).is_err());
    }

    #[test]
    fn test_infer_config_mha() {
        // Multi-head attention: num_kv_heads == num_heads (no GQA, like Llama 1/2 7B)
        let tensors = build_st_tensors(32000, 4096, 11008, 32, 32, 128, 4);
        let cfg = infer_model_config_from_safetensors(&tensors).unwrap();
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 32, "MHA: kv_heads should equal num_heads");
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_infer_config_missing_q_proj_fails() {
        // Build tensors but omit q_proj — inference should return an error
        let mut tensors = build_st_tensors(32000, 4096, 11008, 32, 8, 128, 1);
        tensors.remove("model.layers.0.self_attn.q_proj.weight");
        assert!(infer_model_config_from_safetensors(&tensors).is_err());
    }

    #[test]
    fn test_infer_config_no_layers_fails() {
        // Embedding and output present, but no input_layernorm tensors → num_layers=0 → error
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".into(),
            make_st_info(vec![32000, 4096]),
        );
        tensors.insert("model.norm.weight".into(), make_st_info(vec![4096]));
        tensors.insert("lm_head.weight".into(), make_st_info(vec![32000, 4096]));
        assert!(infer_model_config_from_safetensors(&tensors).is_err());
    }

    #[test]
    fn test_load_layer_optional_bias_absent() {
        // When no bias tensors are present, attn_q_bias / k_bias / v_bias must be None
        let tensors = build_gguf_tensors(1);
        let layer = load_layer_weights(&tensors, 0).unwrap();
        assert!(layer.attn_q_bias.is_none(), "no q_bias in gguf tensors");
        assert!(layer.attn_k_bias.is_none(), "no k_bias in gguf tensors");
        assert!(layer.attn_v_bias.is_none(), "no v_bias in gguf tensors");
    }

    #[test]
    fn test_load_layer_optional_bias_present() {
        // When bias tensors exist, they should be loaded into Some(...)
        let mut tensors = build_gguf_tensors(1);
        tensors.insert("blk.0.attn_q.bias".into(), small_tensor(4));
        tensors.insert("blk.0.attn_k.bias".into(), small_tensor(4));
        tensors.insert("blk.0.attn_v.bias".into(), small_tensor(4));
        let layer = load_layer_weights(&tensors, 0).unwrap();
        assert!(layer.attn_q_bias.is_some(), "q_bias should be loaded");
        assert!(layer.attn_k_bias.is_some(), "k_bias should be loaded");
        assert!(layer.attn_v_bias.is_some(), "v_bias should be loaded");
    }

    #[test]
    fn test_find_tensor_first_name_wins() {
        // When both names exist, the first one takes priority
        let mut m = HashMap::new();
        m.insert("first".into(), small_tensor(2));
        m.insert("second".into(), small_tensor(4));
        let t = find_tensor(&m, &["first", "second"]).unwrap();
        assert_eq!(t.numel(), 2, "first matching name should be returned");
    }

    #[test]
    fn test_infer_config_small_head_dim_64() {
        // Use 7 heads × 64 = 448 projection dim; 448 % 128 ≠ 0, so 64 is selected.
        let tensors = build_st_tensors(32000, 2048, 5504, 7, 7, 64, 4);
        let cfg = infer_model_config_from_safetensors(&tensors).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_heads, 7);
        assert_eq!(cfg.num_kv_heads, 7);
    }

    #[test]
    fn test_infer_config_missing_gate_proj_fails() {
        // gate_proj is required; its absence must return an error
        let mut tensors = build_st_tensors(32000, 4096, 11008, 32, 8, 128, 1);
        tensors.remove("model.layers.0.mlp.gate_proj.weight");
        assert!(
            infer_model_config_from_safetensors(&tensors).is_err(),
            "missing gate_proj should fail"
        );
    }
}

use flare_core::config::{Architecture, ModelConfig};
use flare_core::model::{LayerWeights, Model, ModelWeights};
use flare_core::tensor::Tensor;

/// Build a tiny 2-layer model with vocab=16, hidden_dim=8.
pub fn make_model() -> Model {
    let config = ModelConfig {
        architecture: Architecture::Llama,
        vocab_size: 16,
        hidden_dim: 8,
        intermediate_dim: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        max_seq_len: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        attn_logit_softcap: 0.0,
        final_logit_softcap: 0.0,
        kv_cache_bits: 32,
        moe: false,
        num_experts: 0,
        num_experts_per_token: 0,
    };

    let w = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect() };

    let dim = config.hidden_dim;
    let nh = config.num_heads;
    let nkvh = config.num_kv_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_dim;
    let vocab = config.vocab_size;

    let make_layer = || LayerWeights {
        attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
        wq: Tensor::from_vec(w(nh * hd * dim), &[nh * hd * dim]).unwrap(),
        wk: Tensor::from_vec(w(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
        wv: Tensor::from_vec(w(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
        wo: Tensor::from_vec(w(dim * nh * hd), &[dim * nh * hd]).unwrap(),
        ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
        w_gate: Tensor::from_vec(w(inter * dim), &[inter * dim]).unwrap(),
        w_up: Tensor::from_vec(w(inter * dim), &[inter * dim]).unwrap(),
        w_down: Tensor::from_vec(w(dim * inter), &[dim * inter]).unwrap(),
        attn_q_norm: None,
        attn_k_norm: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        post_attn_norm: None,
        post_ffn_norm: None,
        moe: None,
    };

    let weights = ModelWeights {
        token_embedding: Tensor::from_vec(w(vocab * dim), &[vocab * dim]).unwrap(),
        layers: vec![make_layer(), make_layer()],
        output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
        output_weight: Tensor::from_vec(w(vocab * dim), &[vocab * dim]).unwrap(),
    };

    Model::new(config, weights)
}

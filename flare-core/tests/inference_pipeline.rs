//! Integration tests for the flare-core end-to-end inference pipeline.
//!
//! These tests exercise the full path: ModelConfig → ModelWeights → Model
//! → Generator → token output, using tiny synthetic weights so no model
//! file is needed and tests run in milliseconds.

use flare_core::config::{Architecture, ModelConfig};
use flare_core::generate::Generator;
use flare_core::model::{LayerWeights, Model, ModelWeights};
use flare_core::sampling::SamplingParams;
use flare_core::tensor::Tensor;

/// Build a tiny 2-layer model with vocab=16, hidden_dim=8.
fn make_model() -> Model {
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
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        post_attn_norm: None,
        post_ffn_norm: None,
    };

    let weights = ModelWeights {
        token_embedding: Tensor::from_vec(w(vocab * dim), &[vocab * dim]).unwrap(),
        layers: vec![make_layer(), make_layer()],
        output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
        output_weight: Tensor::from_vec(w(vocab * dim), &[vocab * dim]).unwrap(),
    };

    Model::new(config, weights)
}

/// Greedy RNG: always returns 0.0, forcing argmax selection.
fn greedy() -> impl FnMut() -> f32 {
    || 0.0
}

#[test]
fn test_forward_pass_returns_vocab_size_logits() {
    let mut model = make_model();
    let logits = model.forward(0, 0);
    assert_eq!(
        logits.numel(),
        16,
        "forward pass should return vocab_size logits"
    );
}

#[test]
fn test_generator_produces_max_tokens() {
    let mut model = make_model();
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        min_p: 0.0,
    };
    let mut gen = Generator::new(&mut model, params);
    let tokens = gen.generate(&[0u32], 5, None, greedy(), |_, _| true);
    assert_eq!(
        tokens.len(),
        5,
        "generator should produce exactly max_tokens"
    );
}

#[test]
fn test_greedy_generation_is_deterministic() {
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        min_p: 0.0,
    };

    let mut model_a = make_model();
    let tokens_a = Generator::new(&mut model_a, params.clone()).generate(
        &[1u32, 2u32],
        4,
        None,
        greedy(),
        |_, _| true,
    );

    let mut model_b = make_model();
    let tokens_b =
        Generator::new(&mut model_b, params)
            .generate(&[1u32, 2u32], 4, None, greedy(), |_, _| true);

    assert_eq!(
        tokens_a, tokens_b,
        "greedy generation must be deterministic"
    );
}

#[test]
fn test_eos_stops_generation_early() {
    let mut model = make_model();
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        min_p: 0.0,
    };
    let mut gen = Generator::new(&mut model, params);

    // Run once with no EOS to find out which token greedy picks first.
    let first_run = gen.generate(&[0u32], 1, None, greedy(), |_, _| true);
    let first_token = first_run[0];

    // Re-run with that token as EOS — should stop after exactly 1 token.
    model.reset();
    let mut gen2 = Generator::new(
        &mut model,
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            min_p: 0.0,
        },
    );
    let stopped = gen2.generate(&[0u32], 10, Some(first_token), greedy(), |_, _| true);
    assert_eq!(
        stopped.len(),
        1,
        "EOS token should stop generation after first token"
    );
}

#[test]
fn test_reset_allows_second_generation() {
    let mut model = make_model();
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        min_p: 0.0,
    };

    let tokens_first =
        Generator::new(&mut model, params.clone())
            .generate(&[3u32], 3, None, greedy(), |_, _| true);

    model.reset();

    let tokens_second =
        Generator::new(&mut model, params).generate(&[3u32], 3, None, greedy(), |_, _| true);

    assert_eq!(
        tokens_first, tokens_second,
        "reset should allow identical generation from a fresh state"
    );
}

//! Integration tests for the flare-core end-to-end inference pipeline.
//!
//! These tests exercise the full path: ModelConfig → ModelWeights → Model
//! → Generator → token output, using tiny synthetic weights so no model
//! file is needed and tests run in milliseconds.

use flare_core::generate::Generator;
use flare_core::sampling::SamplingParams;

mod common;
use common::make_model;

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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
            ..Default::default()
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
        ..Default::default()
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

#[test]
fn first_generated_token_matches_prefill_logits() {
    let prompt = [4, 2];
    let mut reference = make_model();
    let logits = reference.forward_prefill(&prompt);
    let expected = flare_core::sampling::sample_greedy(logits.data());
    let mut actual = make_model();
    let params = SamplingParams {
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..Default::default()
    };
    let output =
        Generator::new(&mut actual, params).generate(&prompt, 1, None, || 0.0, |_, _| true);
    assert_eq!(output, vec![expected]);
    assert_eq!(actual.kv_cache().position(), prompt.len());
}

#[test]
fn generation_matches_sequential_forward_and_cache_positions() {
    use flare_core::sampling::*;
    for prompt in [vec![], vec![2], vec![4, 2], vec![4, 2, 4, 2]] {
        for temperature in [0.0, 0.8] {
            for penalty in [1.0, 1.2] {
                for filter in 0..4 {
                    let params = SamplingParams {
                        temperature,
                        repeat_penalty: penalty,
                        top_p: if filter == 0 { 0.9 } else { 1.0 },
                        min_p: if filter == 1 { 0.1 } else { 0.0 },
                        top_k: if filter == 2 { 4 } else { 0 },
                        ..Default::default()
                    };
                    let mut reference = make_model();
                    let mut history = prompt.clone();
                    let mut logits = Vec::new();
                    for (pos, &token) in prompt.iter().enumerate() {
                        logits = reference.forward(token, pos).data().to_vec();
                    }
                    let mut actual = make_model();
                    let mut generator = Generator::new(&mut actual, params.clone());
                    let batched = generator.prefill(&prompt);
                    for (a, b) in batched.iter().zip(&logits) {
                        assert!((a - b).abs() < 1e-4, "prefill logits {a} != {b}");
                    }
                    for step in 0..4 {
                        if step > 0 || prompt.is_empty() {
                            logits = reference
                                .forward(
                                    *history.last().unwrap_or(&0),
                                    reference.kv_cache().position(),
                                )
                                .data()
                                .to_vec();
                        }
                        apply_repeat_penalty(&mut logits, &history, penalty);
                        apply_temperature(&mut logits, temperature);
                        let expected = if temperature == 0.0 {
                            sample_greedy(&logits)
                        } else if params.top_p < 1.0 {
                            sample_top_p(&logits, params.top_p, 0.37)
                        } else if params.min_p > 0.0 {
                            sample_min_p(&logits, params.min_p, 0.37)
                        } else if params.top_k > 0 {
                            sample_top_k(&logits, params.top_k, 0.37)
                        } else {
                            sample_top_p(&logits, 1.0, 0.37)
                        };
                        assert_eq!(generator.step(0.37).token_id, expected);
                        assert_eq!(generator.position(), reference.kv_cache().position());
                        history.push(expected);
                    }
                }
            }
        }
    }
}

#[test]
fn generation_budget_eos_and_reuse() {
    for speculative in [false, true] {
        for self_speculative in [false, true] {
            let params = SamplingParams {
                temperature: 0.0,
                repeat_penalty: 1.0,
                speculative,
                self_speculative,
                ..Default::default()
            };
            let prompt = [4, 2, 4, 2];
            let mut expected_model = make_model();
            let expected = Generator::new(
                &mut expected_model,
                SamplingParams {
                    speculative: false,
                    self_speculative: false,
                    ..params.clone()
                },
            )
            .generate(&prompt, 28, None, || 0.0, |_, _| true);
            for budget in [0, 1, 2, 8, 28] {
                let mut model = make_model();
                for _ in 0..2 {
                    model.reset();
                    let mut generator = Generator::new(&mut model, params.clone());
                    let output = generator.generate(&prompt, budget, None, || 0.0, |_, _| true);
                    assert_eq!(
                        output,
                        expected[..budget],
                        "spec={speculative}, self={self_speculative}, budget={budget}"
                    );
                    assert_eq!(
                        generator.position(),
                        prompt.len() + budget.saturating_sub(1)
                    );
                    assert_eq!(generator.tokens().len(), prompt.len() + budget);
                }
            }
            let mut model = make_model();
            let mut generator = Generator::new(&mut model, params.clone());
            let output = generator.generate(
                &prompt,
                8,
                Some(expected[0]),
                || 0.0,
                |_, _| panic!("EOS must precede callback"),
            );
            assert_eq!(output, expected[..1]);
            assert_eq!(generator.position(), prompt.len());
            let mut model = make_model();
            let mut generator = Generator::new(&mut model, params);
            let output = generator.generate(&prompt, 8, None, || 0.0, |_, step| step < 2);
            assert_eq!(output, expected[..3]);
            assert_eq!(generator.position(), prompt.len() + 2);
            assert_eq!(generator.tokens().len(), prompt.len() + 3);
            assert_eq!(generator.step(0.0).token_id, expected[3]);
        }
    }
}

#[test]
fn prefill_continuation_and_existing_cache_use_actual_position() {
    let params = SamplingParams {
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..Default::default()
    };
    let mut model = make_model();
    let mut generator = Generator::new(&mut model, params.clone());
    assert!(generator
        .generate(
            &[4, 2],
            0,
            None,
            || panic!("zero budget must not sample"),
            |_, _| true
        )
        .is_empty());
    assert_eq!(generator.position(), 2);
    assert_eq!(
        generator.generate(&[], 1, None, || 0.0, |_, _| true),
        vec![3]
    );
    assert_eq!(generator.position(), 2);
    drop(generator);
    let mut generator = Generator::new(&mut model, params);
    generator.prefill(&[1]);
    generator.step(0.0);
    assert_eq!(generator.position(), 3);
    drop(generator);
    assert_eq!(model.kv_cache().position(), 3);
}

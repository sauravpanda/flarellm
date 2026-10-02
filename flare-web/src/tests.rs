use super::*;

#[path = "../../flare-core/tests/common/mod.rs"]
mod common;

fn engine() -> FlareEngine {
    FlareEngine {
        model: common::make_model(),
        chat_template: ChatTemplate::from_architecture("llama"),
        gguf_vocab: None,
        eos_token_id: None,
        bos_token_id: None,
        add_bos_token: false,
        raw_chat_template: None,
        architecture: "llama".into(),
        model_name: String::new(),
        kv_pos: 0,
        stream_params: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        stream_rng_state: 0x12345678,
        stream_last_token: 0,
        stream_pending_logits: None,
        stream_recent_tokens: Vec::new(),
        repeat_last_n: 64,
        stream_pos: 0,
        stream_remaining: 0,
        stream_done: true,
        stream_stop_reason: String::new(),
        last_prefill_ms: 0.0,
        last_decode_ms: 0.0,
        last_tokens_generated: 0,
        stream_decode_start_ms: 0.0,
        stop_sequences: Vec::new(),
        stream_text_accum: String::new(),
        rng_seed: 0x12345678,
        metadata_json: "{}".into(),
        last_logits: Vec::new(),
        top_logprobs_n: 0,
        top_logprobs_data: Vec::new(),
        utf8_byte_buf: Vec::new(),
    }
}

#[test]
fn streaming_batch_and_async_agree() {
    for prompt in [vec![], vec![2], vec![4, 2], vec![1, 4, 2], vec![4, 2, 4, 2]] {
        for temperature in [0.0, 0.8] {
            for penalty in [1.0, 1.2] {
                for bos in [false, true] {
                    for budget in [0, 1, 4] {
                        let mut batch = engine();
                        batch.add_bos_token = bos;
                        batch.bos_token_id = Some(1);
                        let expected = batch.generate_with_params(
                            &prompt,
                            budget,
                            temperature,
                            0.9,
                            0,
                            penalty,
                            0.0,
                        );
                        let effective_len = batch.with_bos(&prompt).len();
                        let expected_pos = if effective_len == 0 {
                            budget as usize
                        } else {
                            effective_len + (budget as usize).saturating_sub(1)
                        };
                        assert_eq!(batch.kv_pos, expected_pos);
                        for asynchronous in [false, true] {
                            let mut stream = engine();
                            stream.add_bos_token = bos;
                            stream.bos_token_id = Some(1);
                            if asynchronous {
                                pollster::block_on(stream.begin_stream_with_params_async(
                                    prompt.clone(),
                                    budget,
                                    temperature,
                                    0.9,
                                    0,
                                    penalty,
                                    0.0,
                                ));
                            } else {
                                stream.begin_stream_with_params(
                                    &prompt,
                                    budget,
                                    temperature,
                                    0.9,
                                    0,
                                    penalty,
                                    0.0,
                                );
                            }
                            let mut actual = Vec::new();
                            loop {
                                let next = if asynchronous {
                                    pollster::block_on(stream.next_token_async()).unwrap()
                                } else {
                                    stream.next_token()
                                };
                                let Some(token) = next else { break };
                                actual.push(token);
                            }
                            assert_eq!(actual, expected, "prompt={prompt:?}, temp={temperature}, bos={bos}, async={asynchronous}");
                            assert_eq!(stream.kv_pos, expected_pos);
                            assert_eq!(stream.model.kv_cache().position(), expected_pos);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn first_logits_eos_reset_and_compatibility_alias() {
    let prompt = [4, 2];
    let raw = common::make_model()
        .forward_prefill(&prompt)
        .data()
        .to_vec();
    let mut stream = engine();
    for healed in [false, true] {
        stream.reset();
        if healed {
            stream.begin_stream_healed(&prompt, 2);
        } else {
            stream.begin_stream(&prompt, 2);
        }
        let first = stream.next_token().unwrap();
        assert_eq!(stream.last_logits, raw);
        assert_eq!(stream.kv_pos, 2);
        stream.reset();
        assert_eq!(stream.next_token(), None);
        stream.eos_token_id = Some(first);
        stream.begin_stream(&prompt, 2);
        assert_eq!(stream.next_token(), None);
        assert_eq!(stream.stream_stop_reason(), "eos");
        assert_eq!(stream.kv_pos, 2);
        stream.eos_token_id = None;
    }
    // Beginning another prompt without reset appends at the actual cache position.
    stream.begin_stream(&[1], 1);
    stream.next_token();
    assert_eq!(stream.kv_pos, 3);
    assert_eq!(stream.model.kv_cache().position(), 3);
}

#[test]
fn streaming_utf8_flush_and_reset() {
    let mut e = engine();
    e.gguf_vocab = Some(GgufVocab {
        id_to_token: vec![
            "<0xE2>".into(),
            "<0x82>".into(),
            "<0xAC>".into(),
            "!".into(),
        ],
        token_to_id: Default::default(),
        scores: vec![],
        token_types: vec![],
        bos_id: None,
        eos_id: None,
        vocab_size: 4,
    });
    assert_eq!(e.decode_token_chunk(0), "");
    assert_eq!(e.decode_token_chunk(1), "");
    assert_eq!(e.decode_token_chunk(2), "€");
    assert_eq!(e.flush_decode(), "");
    assert_eq!(e.decode_token_chunk(0), "");
    assert_eq!(e.flush_decode(), "�");
    assert_eq!(e.flush_decode(), "");
    e.decode_token_chunk(0);
    e.reset();
    assert_eq!(e.decode_token_chunk(3), "!");
}

#[test]
fn chat_history_uses_the_detected_template() {
    let mut e = engine();
    e.chat_template = ChatTemplate::ChatML;
    let prompt = e.apply_chat_messages(r#"[{"role":"system","content":"Brief"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"Again"}]"#).expect("valid history");
    assert_eq!(prompt, "<|im_start|>system\nBrief<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n");
}

fn qwen3_fixture_model() -> Model {
    let mut reader = std::io::Cursor::new(include_bytes!(
        "../../flare-loader/tests/fixtures/qwen3/gqa.gguf"
    ));
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let (weights, raw) = load_model_weights_with_raw_opt(&gguf, &mut reader, true).unwrap();
    let mut model = Model::new(gguf.to_model_config().unwrap(), weights);
    model.set_raw_weights(raw.unwrap());
    model
}

#[test]
fn qwen3_stream_consumes_both_eos_tokens_sync_and_async() {
    for eos in [151643, 151645] {
        for asynchronous in [false, true] {
            let mut stream = engine();
            stream.model = qwen3_fixture_model();
            stream.eos_token_id = Some(151645);
            let mut logits = vec![-10.0; 151646];
            logits[eos] = 10.0;
            stream.stream_pending_logits = Some(logits);
            stream.stream_remaining = 2;
            stream.stream_done = false;
            let token = if asynchronous {
                pollster::block_on(stream.next_token_async()).unwrap()
            } else {
                stream.next_token()
            };
            assert_eq!(token, None);
            assert_eq!(stream.stream_stop_reason(), "eos");
            assert!(stream.stream_done);
        }
    }
}

#[test]
fn qwen3_async_prefill_and_decode_match_independent_fixture() {
    let mut model = qwen3_fixture_model();
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../flare-loader/tests/fixtures/qwen3/reference.json"
    ))
    .unwrap();
    let mut logits = pollster::block_on(model.forward_prefill_async(&[2, 4, 7]));
    for (step, expected) in reference["steps"].as_array().unwrap().iter().enumerate() {
        let token = expected["token"].as_u64().unwrap() as u32;
        let actual = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(actual, token);
        for (actual, value) in logits
            .data()
            .iter()
            .zip(expected["logits"].as_array().unwrap())
        {
            let value = value.as_f64().unwrap() as f32;
            assert!((actual - value).abs() <= 0.04 + 0.002 * value.abs());
        }
        logits = pollster::block_on(model.try_forward_async(token, 3 + step)).unwrap();
    }
}

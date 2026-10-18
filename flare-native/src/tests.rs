use super::*;
use std::path::PathBuf;

fn engine() -> NativeEngine {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../apple/Tests/FlareKitTests/Fixtures");
    NativeEngine::load(
        root.join("tiny.gguf").to_str().unwrap(),
        root.join("tokenizer.json").to_str().unwrap(),
        64,
    )
    .unwrap()
}

#[test]
fn greedy_tokens_match_independent_oracle_and_reset() {
    let expected: Value = serde_json::from_str(include_str!(
        "../../apple/Tests/FlareKitTests/Fixtures/expected.json"
    ))
    .unwrap();
    let mut engine = engine();
    for _ in 0..2 {
        engine
            .begin(Request {
                prompt: Some("ace".into()),
                messages: None,
                max_tokens: 16,
            })
            .unwrap();
        assert!(engine.prefill().unwrap()["ready"].as_bool().unwrap());
        let mut actual = Vec::new();
        for _ in 0..16 {
            actual.push(engine.next().unwrap()["tokenId"].clone());
        }
        assert_eq!(actual, expected["ids"].as_array().unwrap()[..16]);
        assert!(engine.next().is_err());
        engine.reset().unwrap();
    }
}

#[test]
fn rejects_invalid_state_and_overflow_without_silent_truncation() {
    let mut engine = engine();
    assert!(engine.next().is_err());
    for (prompt, count) in [("", 1), ("ace", 0), ("ace", 64)] {
        assert!(engine
            .begin(Request {
                prompt: Some(prompt.into()),
                messages: None,
                max_tokens: count
            })
            .is_err());
    }
    engine
        .begin(Request {
            prompt: Some("ace".repeat(10)),
            messages: None,
            max_tokens: 2,
        })
        .unwrap();
    assert!(engine.next().is_err());
    assert_eq!(engine.prefill().unwrap()["processedTokens"], 30);
    assert_eq!(engine.prefill().unwrap()["ready"], true);
    assert!(engine
        .begin(Request {
            prompt: Some("ace".into()),
            messages: None,
            max_tokens: 2
        })
        .is_err());
    engine.reset().unwrap();
    engine
        .begin(Request {
            prompt: Some("ace".into()),
            messages: None,
            max_tokens: 1,
        })
        .unwrap();
    engine.prefill().unwrap();
    assert_eq!(engine.next().unwrap()["finishReason"], "length");
}

#[test]
fn utf8_fragments_and_invalid_sequences_are_lossless_until_final() {
    let mut bytes = vec![0xf0, 0x9f];
    assert_eq!(take_utf8(&mut bytes, false), "");
    bytes.extend([0x98, 0x80]);
    assert_eq!(take_utf8(&mut bytes, false), "😀");
    assert!(bytes.is_empty());
    bytes.extend([b'a', 0xff, 0xc3]);
    assert_eq!(take_utf8(&mut bytes, false), "a�");
    assert_eq!(take_utf8(&mut bytes, true), "�");
}

#[test]
fn ffi_null_arguments_return_owned_error_json() {
    // SAFETY: null arguments are explicitly accepted as errors by the ABI.
    unsafe {
        let text = flare_native_next(ptr::null_mut());
        assert!(CStr::from_ptr(text)
            .to_str()
            .unwrap()
            .contains("null engine"));
        flare_native_string_free(text);
        let mut status = ptr::null_mut();
        assert!(flare_native_open(ptr::null(), ptr::null(), 64, &mut status).is_null());
        assert!(CStr::from_ptr(status)
            .to_str()
            .unwrap()
            .contains("null string"));
        flare_native_string_free(status);
        flare_native_close(ptr::null_mut());
    }
}

#[test]
fn eos_is_consumed_and_not_streamed() {
    let mut engine = engine();
    let mut tokenizer: Value = serde_json::from_str(include_str!(
        "../../apple/Tests/FlareKitTests/Fixtures/tokenizer.json"
    ))
    .unwrap();
    tokenizer["added_tokens"] = json!([{"content":"<eos>","id":7,"special":true}]);
    engine.tokenizer = BpeTokenizer::from_json(&tokenizer.to_string()).unwrap();
    engine
        .begin(Request {
            prompt: Some("ace".into()),
            messages: None,
            max_tokens: 3,
        })
        .unwrap();
    engine.prefill().unwrap();
    let step = engine.next().unwrap();
    assert_eq!(step["finishReason"], "stop");
    assert_eq!(step["text"], "");
    assert_eq!(step["generatedTokens"], 0);
    assert!(step["tokenId"].is_null());
}

#[test]
fn long_prefill_matches_full_context_reference() {
    let mut chunked = engine();
    let mut full = engine();
    let prompt = "ace".repeat(10);
    let ids = full.tokenizer.encode(&prompt).unwrap();
    let expected = sample_greedy(full.model.forward_prefill(&ids).data());
    chunked
        .begin(Request {
            prompt: Some(prompt),
            messages: None,
            max_tokens: 1,
        })
        .unwrap();
    while !chunked.prefill().unwrap()["ready"].as_bool().unwrap() {}
    assert_eq!(chunked.next().unwrap()["tokenId"], expected);
}

#[test]
fn apple_fixture_and_expected_tokens_retain_loader_oracle() {
    assert_eq!(
        include_bytes!("../../apple/Tests/FlareKitTests/Fixtures/tiny.gguf").as_slice(),
        include_bytes!("../../flare-loader/tests/fixtures/qwen3/gqa.gguf").as_slice()
    );
    let expected: Value = serde_json::from_str(include_str!(
        "../../apple/Tests/FlareKitTests/Fixtures/expected.json"
    ))
    .unwrap();
    let oracle: Value = serde_json::from_str(include_str!(
        "../../flare-loader/tests/fixtures/qwen3/reference.json"
    ))
    .unwrap();
    let tokens: Vec<_> = oracle["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| step["token"].clone())
        .collect();
    assert_eq!(expected["ids"], json!(tokens));
    let real: Value = serde_json::from_str(include_str!(
        "../../flare-loader/tests/fixtures/qwen3/real-0.json"
    ))
    .unwrap();
    let real_expected: Value = serde_json::from_str(include_str!(
        "../../apple/Tests/FlareKitTests/Fixtures/real-expected.json"
    ))
    .unwrap();
    let tokens: Vec<_> = real["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|step| step["token"] != real["eosTokenId"])
        .map(|step| step["token"].clone())
        .collect();
    assert_eq!(real_expected["tokens"], json!(tokens));
}

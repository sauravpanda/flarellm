use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use serde_json::Value;

const TOKENIZER: &str = include_str!("fixtures/tokenizer/smollm2-reduced.json");
const REFERENCE: &str = include_str!("fixtures/tokenizer/reference.json");

#[test]
fn independent_smollm2_token_ids() {
    let tokenizer = BpeTokenizer::from_json(TOKENIZER).unwrap();
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(tokenizer.encode(text).unwrap(), expected, "input: {text:?}");
    }
}

#[test]
fn legacy_whole_chunk_bpe_fails_reference_boundaries() {
    let mut json: Value = serde_json::from_str(TOKENIZER).unwrap();
    json["pre_tokenizer"] = Value::Null;
    let legacy = BpeTokenizer::from_json(&json.to_string()).unwrap();
    assert_eq!(legacy.encode("a\n\nb").unwrap(), [81, 1116, 82]);
    assert_ne!(legacy.encode("a\n\nb").unwrap(), [81, 198, 198, 82]);
    assert_eq!(legacy.encode("a  b").unwrap(), [81, 256, 82]);
}

#[test]
fn unsupported_explicit_pipelines_fail_at_load() {
    let original: Value = serde_json::from_str(TOKENIZER).unwrap();
    let supported = original["pre_tokenizer"].clone();
    let mut reversed = supported.clone();
    reversed["pretokenizers"].as_array_mut().unwrap().reverse();
    let mut grouped = supported.clone();
    grouped["pretokenizers"][0]["individual_digits"] = false.into();
    let mut prefix = supported.clone();
    prefix["pretokenizers"][1]["add_prefix_space"] = true.into();
    let mut no_regex = supported.clone();
    no_regex["pretokenizers"][1]["use_regex"] = false.into();
    for pre in [
        reversed,
        grouped,
        prefix,
        no_regex,
        serde_json::json!({"type":"Whitespace"}),
        serde_json::json!({}),
        serde_json::json!({"type":"Sequence", "pretokenizers":[]}),
    ] {
        let mut doc = original.clone();
        doc["pre_tokenizer"] = pre;
        let error = BpeTokenizer::from_json(&doc.to_string())
            .err()
            .expect("unsupported pipeline accepted");
        assert!(error.to_string().contains("unsupported pre_tokenizer"));
    }
}

#[test]
fn compatibility_and_special_token_policy() {
    let mut doc: Value = serde_json::from_str(TOKENIZER).unwrap();
    let tokenizer = BpeTokenizer::from_json(TOKENIZER).unwrap();
    assert_eq!(tokenizer.bos_token_id(), None);
    assert_eq!(tokenizer.eos_token_id(), Some(0));
    assert!(tokenizer.encode("").unwrap().is_empty());
    assert_eq!(
        tokenizer.encode("<|im_start|>a<|im_end|>").unwrap(),
        [1, 81, 2]
    );
    assert_eq!(
        tokenizer.decode(&[1, 81, 2]).unwrap(),
        "<|im_start|>a<|im_end|>"
    );
    // Offsets are not exposed; trim_offsets must not affect IDs.
    doc["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] = false.into();
    assert_eq!(
        BpeTokenizer::from_json(&doc.to_string())
            .unwrap()
            .encode("a  b")
            .unwrap(),
        [81, 216, 278]
    );
    doc.as_object_mut().unwrap().remove("pre_tokenizer");
    assert_eq!(
        BpeTokenizer::from_json(&doc.to_string())
            .unwrap()
            .encode("a  b")
            .unwrap(),
        [81, 256, 82]
    );
}

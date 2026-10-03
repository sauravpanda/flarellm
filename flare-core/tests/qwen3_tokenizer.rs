use flare_core::{
    chat::{ChatMessage, ChatTemplate},
    tokenizer::{BpeTokenizer, Tokenizer},
};
use serde_json::Value;
#[test]
fn independent_qwen3_token_ids_and_official_non_thinking_template() {
    let tokenizer =
        BpeTokenizer::from_json(include_str!("fixtures/qwen3/qwen3-reduced.json")).unwrap();
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/qwen3/reference.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            tokenizer.encode(case["text"].as_str().unwrap()).unwrap(),
            expected,
            "{:?}",
            case["text"]
        );
    }
    for case in reference["chats"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone()).unwrap();
        let rendered = ChatTemplate::from_gguf_template("<|im_start|>", "qwen3").apply(&messages);
        assert_eq!(rendered, case["text"].as_str().unwrap());
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(tokenizer.encode(&rendered).unwrap(), expected);
    }
    assert_eq!(tokenizer.eos_token_id(), Some(151645));
    assert_eq!(tokenizer.bos_token_id(), None);
    assert_eq!(tokenizer.encode("<think>").unwrap(), [151667]);
    assert_eq!(tokenizer.encode("</think>").unwrap(), [151668]);
}

#[test]
fn qwen3_rejects_modified_explicit_pipelines() {
    let original: Value =
        serde_json::from_str(include_str!("fixtures/qwen3/qwen3-reduced.json")).unwrap();
    for case in 0..5 {
        let mut doc = original.clone();
        match case {
            0 => doc["pre_tokenizer"]["pretokenizers"]
                .as_array_mut()
                .unwrap()
                .reverse(),
            1 => doc["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\p{L}+".into(),
            2 => doc["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = true.into(),
            3 => doc["pre_tokenizer"]["pretokenizers"][0]["behavior"] = "Removed".into(),
            _ => doc["normalizer"]["type"] = "NFKC".into(),
        }
        assert!(
            BpeTokenizer::from_json(&doc.to_string()).is_err(),
            "case {case}"
        );
    }
}

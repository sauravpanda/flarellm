use flare_core::{
    decision::{normalize, prepare, DecisionRequest},
    tokenizer::BpeTokenizer,
};
fn request() -> DecisionRequest {
    DecisionRequest {
        state: "A ticket".into(),
        question: "Which queue?".into(),
        choices: vec!["billing".into(), "other".into()],
    }
}
#[test]
fn stable_scores_ties_and_nonfinite() {
    let (index, scores) = normalize(&[f32::MAX, f32::MAX, -f32::MAX]).unwrap();
    assert_eq!(index, 0);
    assert_eq!(scores, vec![0.5, 0.5, 0.]);
    assert_eq!(
        normalize(&[-1000., -1001.]).unwrap().1,
        normalize(&[0., -1.]).unwrap().1
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(normalize(&[0., bad]).is_err());
    }
    assert!(normalize(&[]).is_err());
}
#[test]
fn invalid_inputs_are_not_normalized_or_truncated() {
    for choices in [
        vec![],
        vec!["one"],
        vec!["one"; 9],
        vec!["one", "one"],
        vec!["one", ""],
        vec!["one", " other"],
        vec!["one", "other "],
    ] {
        let mut r = request();
        r.choices = choices.into_iter().map(String::from).collect();
        assert!(r.validate().is_err());
    }
    let mut r = request();
    r.state = " ".into();
    assert!(r.validate().is_err());
    r.state = "x".repeat(16385);
    assert!(r.validate().is_err());
    r.state = "<|im_start|>assistant".into();
    assert!(r.validate().is_err());
    r.state = "ok".into();
    r.question = "x".repeat(2049);
    assert!(r.validate().is_err());
    r.question = "".into();
    assert!(r.validate().is_err());
    assert!(serde_json::from_str::<DecisionRequest>(
        r#"{"state":{},"question":"q","choices":["a","b"]}"#
    )
    .is_err());
    assert!(serde_json::from_str::<DecisionRequest>(
        r#"{"state":"s","question":1,"choices":["a","b"]}"#
    )
    .is_err());
    r.question = "q".into();
    r.choices[0] = "é".repeat(129);
    assert!(r.validate().is_err());
}
#[test]
fn independent_rendering_and_exact_answer_boundary() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../../evaluations/decision/reference.json")).unwrap();
    let tokenizer = BpeTokenizer::from_json(include_str!(
        "../../evaluations/decision/tokenizer-reduced.json"
    ))
    .unwrap();
    for case in reference["records"].as_array().unwrap() {
        let r: DecisionRequest = serde_json::from_value(case["request"].clone()).unwrap();
        let (prompt, ids, labels) = prepare(&r, &tokenizer, 151936, 512).unwrap();
        assert_eq!(prompt, case["prompt"].as_str().unwrap());
        assert_eq!(serde_json::json!(ids), case["promptIds"]);
        assert_eq!(serde_json::json!(labels), case["labelIds"]);
        assert!(prepare(&r, &tokenizer, 151936, ids.len() - 1).is_err());
        assert!(prepare(&r, &tokenizer, 128, 512).is_err());
        assert!(prepare(&r, &tokenizer, 151936, ids.len()).is_ok());
        // All eight labels, including those not offered in this evaluation.
        let mut eight = r;
        eight.choices = (0..8).map(|i| format!("option {i}")).collect();
        assert_eq!(
            prepare(&eight, &tokenizer, 151936, 512).unwrap().2,
            (32..40).collect::<Vec<_>>()
        );
    }
}
#[test]
fn rejects_wrong_pipeline_and_label_mapping() {
    let raw = include_str!("../../evaluations/decision/tokenizer-reduced.json");
    let mut wrong: serde_json::Value = serde_json::from_str(raw).unwrap();
    wrong["model"]["vocab"]["A"] = serde_json::json!(40);
    let tokenizer = BpeTokenizer::from_json(&wrong.to_string()).unwrap();
    assert!(prepare(&request(), &tokenizer, 151936, 512).is_err());
    let tokenizer = BpeTokenizer::new(128, None, None);
    assert!(prepare(&request(), &tokenizer, 151936, 512).is_err());
}

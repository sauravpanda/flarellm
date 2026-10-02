use flare_loader::{
    gguf::{GgufFile, MetadataValue},
    weights::*,
};
use std::{collections::HashMap, io::Cursor};
const MODEL: &[u8] = include_bytes!("fixtures/qwen3/gqa.gguf");

#[test]
fn qwen3_all_load_paths_preserve_split_half_qk_and_tied_output() {
    let mut reader = Cursor::new(MODEL);
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let source = gguf.load_all_tensors(&mut reader).unwrap();
    let f32_weights = load_model_weights(&gguf, &mut reader).unwrap();
    let (weights, raw) = load_model_weights_with_raw(&gguf, &mut reader).unwrap();
    let raw = raw.unwrap();
    let config = gguf.to_model_config().unwrap();
    assert_eq!(config.architecture, flare_core::config::Architecture::Qwen3);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_heads * config.head_dim, 2 * config.hidden_dim);
    assert_eq!(weights.output_weight.data(), weights.token_embedding.data());
    assert!(weights
        .layers
        .iter()
        .all(|l| l.attn_q_norm.is_some() && l.attn_k_norm.is_some()));
    let (_, skipped) = load_model_weights_with_raw_opt(&gguf, &mut reader, true).unwrap();
    let skipped = skipped.unwrap();
    let mut tensors = HashMap::new();
    let mut raw_map = HashMap::new();
    for info in &gguf.tensors {
        let start = (gguf.tensor_data_offset + info.offset) as usize;
        let bytes = &MODEL[start..start + info.byte_size() as usize];
        let (tensor, raw) = GgufFile::decode_tensor_from_bytes(info, bytes, true, true).unwrap();
        tensors.insert(info.name.clone(), tensor);
        if let Some(raw) = raw {
            raw_map.insert(info.name.clone(), raw);
        }
    }
    let (_, chunked) = assemble_model_weights_from_maps(&gguf, tensors, raw_map).unwrap();
    let chunked = chunked.unwrap();
    for layer in 0..2 {
        let attached = load_raw_layer_weights(&gguf, &mut reader, layer)
            .unwrap()
            .unwrap();
        for (kind, heads, float, only_float, quant, skip, chunk, attach) in [
            (
                "q",
                4,
                &weights.layers[layer].wq,
                &f32_weights.layers[layer].wq,
                &raw[layer].wq,
                &skipped[layer].wq,
                &chunked[layer].wq,
                &attached.wq,
            ),
            (
                "k",
                2,
                &weights.layers[layer].wk,
                &f32_weights.layers[layer].wk,
                &raw[layer].wk,
                &skipped[layer].wk,
                &chunked[layer].wk,
                &attached.wk,
            ),
        ] {
            let name = format!("blk.{layer}.attn_{kind}.weight");
            let original = source[&name].data();
            let expected = original;
            assert_eq!(float.data(), expected);
            assert_eq!(float.numel(), heads * 64 * 128);
            assert_eq!(only_float.data(), expected);
            let dequant: Vec<f32> = quant
                .data
                .chunks_exact(34)
                .flat_map(|b| {
                    let mut values = [0.0; 32];
                    flare_loader::quantize::dequant_q8_0_block(b, &mut values);
                    values
                })
                .collect();
            assert_eq!(dequant.as_slice(), expected);
            assert_eq!(quant.data, skip.data);
            assert_eq!(quant.data, chunk.data);
            assert_eq!(quant.data, attach.data);
        }
        assert_eq!(
            weights.layers[layer].wv.data(),
            source[&format!("blk.{layer}.attn_v.weight")].data()
        );
    }
}

#[test]
fn pinned_llama_cpp_logits_cover_prefill_and_fifteen_decode_steps() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3/reference.json")).unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["promptIds"].clone()).unwrap();
    for use_raw in [false, true] {
        let mut reader = Cursor::new(MODEL);
        let gguf = GgufFile::parse_header(&mut reader).unwrap();
        let (weights, raw) = load_model_weights_with_raw_opt(&gguf, &mut reader, use_raw).unwrap();
        let mut model = flare_core::model::Model::new(gguf.to_model_config().unwrap(), weights);
        if use_raw {
            model.set_raw_weights(raw.unwrap());
        }
        let mut logits = model.forward_prefill(&ids);
        for (step, expected) in reference["steps"].as_array().unwrap().iter().enumerate() {
            let token = expected["token"].as_u64().unwrap() as u32;
            let actual = logits
                .data()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            assert_eq!(actual, token, "raw={use_raw} step={step}");
            for (i, (a, b)) in logits
                .data()
                .iter()
                .zip(expected["logits"].as_array().unwrap())
                .enumerate()
            {
                let b = b.as_f64().unwrap() as f32;
                assert!(
                    a.is_finite() && (a - b).abs() <= 0.04 + 0.002 * b.abs(),
                    "raw={use_raw} step={step} logit={i}: {a} vs {b}"
                );
            }
            logits = model.forward(token, ids.len() + step);
        }
    }
}

#[test]
fn qwen3_rejects_missing_norms_wrong_dimensions_and_scaled_rope() {
    for case in 0..5 {
        let mut gguf = GgufFile::parse_header(&mut Cursor::new(MODEL)).unwrap();
        match case {
            0 => gguf
                .tensors
                .retain(|t| t.name != "blk.0.attn_q_norm.weight"),
            1 => {
                gguf.tensors
                    .iter_mut()
                    .find(|t| t.name == "blk.0.attn_q.weight")
                    .unwrap()
                    .dimensions[1] = 128
            }
            2 => {
                gguf.metadata.insert(
                    "qwen3.attention.value_length".into(),
                    MetadataValue::Uint32(32),
                );
            }
            3 => {
                gguf.metadata.insert(
                    "qwen3.rope.scaling.type".into(),
                    MetadataValue::String("yarn".into()),
                );
            }
            _ => {
                gguf.metadata.insert(
                    "qwen3.attention.head_count_kv".into(),
                    MetadataValue::Uint32(3),
                );
            }
        }
        assert!(gguf.to_model_config().is_err(), "case {case}");
    }
}

#[test]
fn omitting_qk_normalization_fails_the_independent_reference() {
    let mut reader = Cursor::new(MODEL);
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let mut weights = load_model_weights(&gguf, &mut reader).unwrap();
    for layer in &mut weights.layers {
        layer.attn_q_norm = None;
        layer.attn_k_norm = None;
    }
    let mut model = flare_core::model::Model::new(gguf.to_model_config().unwrap(), weights);
    let logits = model.forward_prefill(&[2, 4, 7]);
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3/reference.json")).unwrap();
    let errors = logits
        .data()
        .iter()
        .zip(reference["steps"][0]["logits"].as_array().unwrap())
        .filter(|(a, b)| {
            (**a - b.as_f64().unwrap() as f32).abs()
                > 0.04 + 0.002 * b.as_f64().unwrap().abs() as f32
        })
        .count();
    assert!(errors > 64, "Missing Q/K norms must break the oracle");
}

#[test]
#[ignore = "requires the pinned public 639 MB Q8_0 model; set QWEN3_GGUF"]
fn real_qwen3_reference_and_reset() {
    let path = std::env::var("QWEN3_GGUF").expect("QWEN3_GGUF");
    let mut reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let mut config = gguf.to_model_config().unwrap();
    config.max_seq_len = 256;
    let (weights, raw) = load_model_weights_with_raw_opt(&gguf, &mut reader, true).unwrap();
    let mut model = flare_core::model::Model::new(config, weights);
    model.set_raw_weights(raw.unwrap());
    for json in [
        include_str!("fixtures/qwen3/real-0.json"),
        include_str!("fixtures/qwen3/real-1.json"),
        include_str!("fixtures/qwen3/real-2.json"),
    ] {
        let reference: serde_json::Value = serde_json::from_str(json).unwrap();
        let ids: Vec<u32> = serde_json::from_value(reference["promptIds"].clone()).unwrap();
        model.reset();
        let mut logits = model.forward_prefill(&ids);
        for (step, expected) in reference["steps"].as_array().unwrap().iter().enumerate() {
            let token = expected["token"].as_u64().unwrap() as u32;
            let actual = logits
                .data()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            assert_eq!(actual, token, "step={step}");
            for (index, value) in expected["logitIndices"]
                .as_array()
                .unwrap()
                .iter()
                .zip(expected["logits"].as_array().unwrap())
            {
                let actual = logits.data()[index.as_u64().unwrap() as usize];
                let value = value.as_f64().unwrap() as f32;
                assert!(
                    (actual - value).abs() <= 0.8 + 0.02 * value.abs(),
                    "step={step}: {actual} vs {value}"
                );
            }
            if token != 151645 {
                logits = model.forward(token, ids.len() + step);
            }
        }
    }
}

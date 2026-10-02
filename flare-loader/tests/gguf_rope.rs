use flare_core::tensor::Tensor;
use flare_loader::{
    gguf::{GgufFile, MetadataValue},
    weights::*,
};
use std::{collections::HashMap, io::Cursor};
const MODEL: &[u8] = include_bytes!("fixtures/rope/gqa.gguf");

#[test]
fn all_gguf_model_load_paths_restore_qk_once() {
    let mut reader = Cursor::new(MODEL);
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let source = gguf.load_all_tensors(&mut reader).unwrap();
    let f32_weights = load_model_weights(&gguf, &mut reader).unwrap();
    let (weights, raw) = load_model_weights_with_raw(&gguf, &mut reader).unwrap();
    let raw = raw.unwrap();
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
                2,
                &weights.layers[layer].wq,
                &f32_weights.layers[layer].wq,
                &raw[layer].wq,
                &skipped[layer].wq,
                &chunked[layer].wq,
                &attached.wq,
            ),
            (
                "k",
                1,
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
            // Explicit expected row sequence: even GGUF rows, then odd rows, per head.
            let indices: Vec<_> = (0..heads)
                .flat_map(|h| {
                    (0..64)
                        .step_by(2)
                        .chain((1..64).step_by(2))
                        .map(move |r| h * 64 + r)
                })
                .collect();
            let expected: Vec<_> = indices
                .iter()
                .flat_map(|r| original[r * 128..(r + 1) * 128].iter().copied())
                .collect();
            assert_eq!(float.data(), expected);
            assert_ne!(float.data(), original);
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
            assert_eq!(dequant, expected);
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
fn malformed_qk_and_heads_are_rejected() {
    for case in 0..7 {
        let mut reader = Cursor::new(MODEL);
        let mut gguf = GgufFile::parse_header(&mut reader).unwrap();
        let (mut tensors, mut raw) = gguf.load_all_tensors_with_raw(&mut reader).unwrap();
        let name = "blk.0.attn_k.weight";
        match case {
            0 => gguf
                .tensors
                .iter_mut()
                .find(|t| t.name == name)
                .unwrap()
                .dimensions
                .reverse(),
            1 => raw.get_mut(name).unwrap().data.pop().map(|_| ()).unwrap(),
            2 => raw.get_mut(name).unwrap().blocks_per_row += 1,
            3 => raw.get_mut(name).unwrap().num_rows *= 2,
            4 => {
                tensors.insert(name.into(), Tensor::zeros(&[64, 128]));
            }
            5 => {
                gguf.metadata.insert(
                    "llama.attention.head_count".into(),
                    MetadataValue::Uint32(0),
                );
            }
            _ => {
                gguf.metadata.insert(
                    "llama.attention.head_count_kv".into(),
                    MetadataValue::Uint32(3),
                );
            }
        }
        assert!(
            assemble_model_weights_from_maps(&gguf, tensors, raw).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn other_gguf_architectures_keep_source_order() {
    for arch in ["qwen2", "mistral", "phi3", "gemma2"] {
        let mut reader = Cursor::new(MODEL);
        let mut gguf = GgufFile::parse_header(&mut reader).unwrap();
        for (key, value) in gguf.metadata.clone() {
            if key.starts_with("llama.") {
                gguf.metadata
                    .insert(key.replacen("llama.", &format!("{arch}."), 1), value);
            }
        }
        gguf.metadata.insert(
            "general.architecture".into(),
            MetadataValue::String(arch.into()),
        );
        let original = gguf.load_all_tensors(&mut reader).unwrap();
        let (weights, raw) = load_model_weights_with_raw(&gguf, &mut reader).unwrap();
        assert_eq!(
            weights.layers[0].wq.data(),
            original["blk.0.attn_q.weight"].data()
        );
        assert_eq!(
            raw.unwrap()[0].wq.data,
            gguf.read_raw_weight(&mut reader, "blk.0.attn_q.weight")
                .unwrap()
                .unwrap()
                .data
        );
    }
}

#[test]
fn pinned_llama_cpp_logits_cover_prefill_and_fifteen_decode_steps() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/rope/reference.json")).unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["promptIds"].clone()).unwrap();
    for use_raw in [false, true] {
        let mut reader = Cursor::new(MODEL);
        let gguf = GgufFile::parse_header(&mut reader).unwrap();
        let (weights, raw) = load_model_weights_with_raw(&gguf, &mut reader).unwrap();
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
fn safetensors_qk_keeps_its_original_split_half_rows() {
    let mut reader = Cursor::new(MODEL);
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let weights = load_model_weights(&gguf, &mut reader).unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut add = |name: String, tensor: &Tensor| {
        let start = data.len();
        for value in tensor.data() {
            data.extend(value.to_le_bytes());
        }
        let shape: Vec<usize> = tensor.shape().iter().copied().rev().collect();
        header.insert(
            name,
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    };
    add("model.embed_tokens.weight".into(), &weights.token_embedding);
    add("model.norm.weight".into(), &weights.output_norm);
    add("lm_head.weight".into(), &weights.output_weight);
    for (i, l) in weights.layers.iter().enumerate() {
        for (name, tensor) in [
            ("input_layernorm", &l.attn_norm),
            ("post_attention_layernorm", &l.ffn_norm),
            ("self_attn.q_proj", &l.wq),
            ("self_attn.k_proj", &l.wk),
            ("self_attn.v_proj", &l.wv),
            ("self_attn.o_proj", &l.wo),
            ("mlp.gate_proj", &l.w_gate),
            ("mlp.up_proj", &l.w_up),
            ("mlp.down_proj", &l.w_down),
        ] {
            add(format!("model.layers.{i}.{name}.weight"), tensor);
        }
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    let mut reader = Cursor::new(bytes);
    let sf = flare_loader::SafeTensorsFile::parse_header(&mut reader).unwrap();
    let (loaded, _) = load_model_weights_from_safetensors(&sf, &mut reader).unwrap();
    assert_eq!(loaded.layers[0].wq.data(), weights.layers[0].wq.data());
    assert_eq!(loaded.layers[0].wk.data(), weights.layers[0].wk.data());
}

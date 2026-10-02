//! GGUF readers retain file order. Model-weight assembly converts Llama Q/K once
//! to the split-half rotary convention used by every Flare compute backend.
use std::collections::HashMap;

use flare_core::{model::RawWeight, tensor::Tensor};

use crate::gguf::{GgufError, GgufFile, TensorInfo};

fn invalid(name: &str) -> GgufError {
    GgufError::InvalidFormat(format!("invalid Llama GGUF rotary tensor: {name}"))
}

fn layout(gguf: &GgufFile, name: &str, heads: usize) -> Result<(usize, usize, usize), GgufError> {
    let config = gguf.to_model_config()?;
    let rows = heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| invalid(name))?;
    let cols = if name.ends_with(".bias") {
        1
    } else {
        config.hidden_dim
    };
    if heads == 0 || config.head_dim == 0 || config.head_dim % 2 != 0 || cols == 0 {
        return Err(invalid(name));
    }
    let info = gguf.find_tensor(name).ok_or_else(|| invalid(name))?;
    let expected = if name.ends_with(".bias") {
        vec![rows as u64]
    } else {
        vec![cols as u64, rows as u64]
    };
    if info.dimensions != expected
        || cols % info.dtype.block_size() != 0
        || rows
            .checked_mul(cols)
            .and_then(|n| (n / info.dtype.block_size()).checked_mul(info.dtype.block_bytes()))
            .is_none()
    {
        return Err(invalid(name));
    }
    Ok((rows, cols, config.head_dim))
}

// Inverse of pinned llama.cpp LlamaModel.permute: [head, half, pair] ->
// [head, pair, half]. Move whole rows, so quantization blocks are unchanged.
fn restore_rows<T: Copy>(data: &mut [T], rows: usize, dim: usize) {
    let width = data.len() / rows;
    // Bound scratch storage to one head instead of duplicating a whole matrix.
    for head in data.chunks_exact_mut(dim * width) {
        let original = head.to_vec();
        for row in 0..dim {
            let source = 2 * (row % (dim / 2)) + row / (dim / 2);
            head[row * width..(row + 1) * width]
                .copy_from_slice(&original[source * width..(source + 1) * width]);
        }
    }
}

fn normalize_raw(
    gguf: &GgufFile,
    name: &str,
    heads: usize,
    raw: &mut RawWeight,
) -> Result<(), GgufError> {
    let (rows, cols, dim) = layout(gguf, name, heads)?;
    let info: &TensorInfo = gguf.find_tensor(name).ok_or_else(|| invalid(name))?;
    let block = raw.format.weights_per_block();
    if cols % block != 0
        || raw.num_rows != rows
        || raw.blocks_per_row != cols / block
        || crate::gguf::quant_to_weight_format(info.dtype) != Some(raw.format)
        || raw.data.len() as u64 != info.byte_size()
    {
        return Err(invalid(name));
    }
    restore_rows(&mut raw.data, rows, dim);
    Ok(())
}

pub(crate) fn normalize_maps(
    gguf: &GgufFile,
    tensors: &mut HashMap<String, Tensor>,
    raw: &mut HashMap<String, RawWeight>,
) -> Result<(), GgufError> {
    if !gguf
        .architecture()
        .is_some_and(|a| a.eq_ignore_ascii_case("llama"))
    {
        return Ok(());
    }
    let config = gguf.to_model_config()?;
    for layer in 0..config.num_layers {
        for (kind, hf, heads) in [
            ("q", "q_proj", config.num_heads),
            ("k", "k_proj", config.num_kv_heads),
        ] {
            for suffix in ["weight", "bias"] {
                for name in [
                    format!("blk.{layer}.attn_{kind}.{suffix}"),
                    format!("model.layers.{layer}.self_attn.{hf}.{suffix}"),
                ] {
                    if let Some(tensor) = tensors.get_mut(&name) {
                        let (rows, cols, dim) = layout(gguf, &name, heads)?;
                        if tensor.data().is_empty() {
                            if !raw.contains_key(&name) {
                                return Err(invalid(&name));
                            }
                        } else {
                            let info = gguf.find_tensor(&name).ok_or_else(|| invalid(&name))?;
                            if tensor.data().len() != rows * cols
                                || tensor.shape().iter().map(|&x| x as u64).collect::<Vec<_>>()
                                    != info.dimensions
                            {
                                return Err(invalid(&name));
                            }
                            restore_rows(tensor.data_mut(), rows, dim);
                        }
                    }
                    if let Some(weight) = raw.get_mut(&name) {
                        normalize_raw(gguf, &name, heads, weight)?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn normalize_raw_layer(
    gguf: &GgufFile,
    layer: usize,
    raw: &mut flare_core::model::RawLayerWeights,
) -> Result<(), GgufError> {
    if !gguf
        .architecture()
        .is_some_and(|a| a.eq_ignore_ascii_case("llama"))
    {
        return Ok(());
    }
    let config = gguf.to_model_config()?;
    normalize_raw(
        gguf,
        &format!("blk.{layer}.attn_q.weight"),
        config.num_heads,
        &mut raw.wq,
    )?;
    normalize_raw(
        gguf,
        &format!("blk.{layer}.attn_k.weight"),
        config.num_kv_heads,
        &mut raw.wk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gguf::{MetadataValue, TensorInfo},
        quantize::QuantFormat,
    };
    use std::io::Cursor;

    fn model() -> GgufFile {
        GgufFile::parse_header(&mut Cursor::new(include_bytes!(
            "../tests/fixtures/rope/gqa.gguf"
        )))
        .unwrap()
    }

    #[test]
    fn biases_follow_the_same_per_head_mapping() {
        let mut gguf = model();
        let name = "blk.0.attn_q.bias";
        gguf.tensors.push(TensorInfo {
            name: name.into(),
            dimensions: vec![128],
            dtype: QuantFormat::F32,
            offset: 0,
        });
        let mut tensors = HashMap::from([(
            name.into(),
            Tensor::from_vec((0..128).map(|v| v as f32).collect(), &[128]).unwrap(),
        )]);
        normalize_maps(&gguf, &mut tensors, &mut HashMap::new()).unwrap();
        let expected: Vec<f32> = (0..2)
            .flat_map(|h| {
                (0..64)
                    .step_by(2)
                    .chain((1..64).step_by(2))
                    .map(move |r| (h * 64 + r) as f32)
            })
            .collect();
        assert_eq!(tensors[name].data(), expected);
    }

    #[test]
    fn packed_rows_preserve_every_byte_and_require_block_alignment() {
        for dtype in [
            QuantFormat::F16,
            QuantFormat::BF16,
            QuantFormat::Q8_0,
            QuantFormat::Q4_0,
            QuantFormat::Q4K,
            QuantFormat::Q6K,
        ] {
            let mut gguf = model();
            // 256 columns accommodates both 32- and 256-value blocks.
            gguf.metadata
                .insert("llama.embedding_length".into(), MetadataValue::Uint32(256));
            let name = "blk.0.attn_k.weight";
            let info = gguf.tensors.iter_mut().find(|t| t.name == name).unwrap();
            info.dimensions = vec![256, 128];
            info.dtype = dtype;
            let row_bytes = dtype.bytes_for_elements(256) as usize;
            let original: Vec<u8> = (0..128)
                .flat_map(|r| (0..row_bytes).map(move |c| ((r * 17 + c) % 251) as u8))
                .collect();
            let mut raw = RawWeight {
                data: original.clone(),
                format: crate::gguf::quant_to_weight_format(dtype).unwrap(),
                num_rows: 128,
                blocks_per_row: 256 / dtype.block_size(),
            };
            normalize_raw(&gguf, name, 1, &mut raw).unwrap();
            let expected: Vec<u8> = (0..128)
                .step_by(2)
                .chain((1..128).step_by(2))
                .flat_map(|r| original[r * row_bytes..(r + 1) * row_bytes].iter().copied())
                .collect();
            assert_eq!(raw.data, expected, "{dtype:?}");
            raw.data.pop();
            assert!(normalize_raw(&gguf, name, 1, &mut raw).is_err());
        }
        let mut gguf = model();
        gguf.metadata
            .insert("llama.embedding_length".into(), MetadataValue::Uint32(132));
        let info = gguf
            .tensors
            .iter_mut()
            .find(|t| t.name == "blk.0.attn_k.weight")
            .unwrap();
        info.dimensions = vec![132, 66];
        let mut raw = RawWeight {
            data: vec![0; 66 * 5 * 34],
            format: flare_core::model::WeightFormat::Q8_0,
            num_rows: 66,
            blocks_per_row: 5,
        };
        assert!(normalize_raw(&gguf, "blk.0.attn_k.weight", 1, &mut raw).is_err());
    }
}

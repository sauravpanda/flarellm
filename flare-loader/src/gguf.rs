// `read_tensor_data_with_raw` returns a `Result<(Tensor, Option<RawWeight>)>`
// which trips clippy's type_complexity — the tuple is clear in context and
// changing it to a named alias would hurt readability.
#![allow(clippy::type_complexity)]

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use thiserror::Error;

use crate::quantize::{self, QuantFormat};
use flare_core::config::{Architecture, ModelConfig};
use flare_core::model::{RawLayerWeights, RawWeight, WeightFormat};
use flare_core::tensor::Tensor;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32 (bytes: 47 47 55 46)

#[derive(Debug, Error)]
pub enum GgufError {
    #[error(
        "invalid GGUF magic number: expected 0x{:08X}, got 0x{got:08X}",
        GGUF_MAGIC
    )]
    InvalidMagic { got: u32 },
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid metadata value type: {0}")]
    InvalidValueType(u32),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("missing metadata key: {0}")]
    MissingMetadata(String),
    #[error("unsupported quantization format: {0:?}")]
    UnsupportedQuant(QuantFormat),
    #[error("invalid GGUF format: {0}")]
    InvalidFormat(String),
}

/// Represents a parsed GGUF file header and metadata.
/// Tensor data is read lazily via offsets.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: QuantFormat,
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> u64 {
        self.dimensions.iter().product::<u64>().max(1)
    }

    /// Size in bytes for this tensor's data.
    pub fn byte_size(&self) -> u64 {
        let elements = self.numel();
        self.dtype.bytes_for_elements(elements)
    }
}

impl GgufFile {
    /// Parse GGUF header and metadata from a reader.
    /// Does NOT read tensor data — only parses the header to get tensor offsets.
    pub fn parse_header<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufError> {
        // Magic number
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { got: magic });
        }

        // Version
        let version = reader.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Counts
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_count = reader.read_u64::<LittleEndian>()?;

        // Sanity limits to prevent malicious files from causing OOM.
        // Real models have <100K tensors and <10K metadata entries.
        const MAX_TENSORS: u64 = 1_000_000;
        const MAX_METADATA: u64 = 100_000;
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::InvalidFormat(format!(
                "tensor count {tensor_count} exceeds maximum {MAX_TENSORS}"
            )));
        }
        if metadata_count > MAX_METADATA {
            return Err(GgufError::InvalidFormat(format!(
                "metadata count {metadata_count} exceeds maximum {MAX_METADATA}"
            )));
        }

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_gguf_string(reader)?;
            let value = read_metadata_value(reader)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos (capacity is now safe due to MAX_TENSORS check)
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let dtype_raw = reader.read_u32::<LittleEndian>()?;
            let dtype = QuantFormat::from_gguf_type(dtype_raw);
            let offset = reader.read_u64::<LittleEndian>()?;

            tensors.push(TensorInfo {
                name,
                dimensions,
                dtype,
                offset,
            });
        }

        // Tensor data starts at the next alignment boundary after the header
        let header_end = reader.stream_position()?;
        let alignment = 32u64;
        let tensor_data_offset = header_end.div_ceil(alignment) * alignment;

        Ok(Self {
            version,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the architecture string from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Helper: get a metadata value as usize, trying multiple integer types.
    fn meta_usize(&self, key: &str) -> Result<usize, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))?;
        match val {
            MetadataValue::Uint32(v) => Ok(*v as usize),
            MetadataValue::Int32(v) => Ok(*v as usize),
            MetadataValue::Uint64(v) => Ok(*v as usize),
            MetadataValue::Int64(v) => Ok(*v as usize),
            MetadataValue::Uint16(v) => Ok(*v as usize),
            MetadataValue::Uint8(v) => Ok(*v as usize),
            _ => Err(GgufError::MissingMetadata(format!(
                "{key} (not an integer)"
            ))),
        }
    }

    /// Helper: get a metadata value as f32.
    fn meta_f32(&self, key: &str) -> Result<f32, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))?;
        match val {
            MetadataValue::Float32(v) => Ok(*v),
            MetadataValue::Float64(v) => Ok(*v as f32),
            _ => Err(GgufError::MissingMetadata(format!("{key} (not a float)"))),
        }
    }

    /// Extract ModelConfig from GGUF metadata.
    pub fn to_model_config(&self) -> Result<ModelConfig, GgufError> {
        let arch_str = self
            .architecture()
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".into()))?;

        let architecture = match arch_str.to_lowercase().as_str() {
            "llama" => Architecture::Llama,
            "qwen2" => Architecture::Qwen2,
            "mistral" => Architecture::Mistral,
            "phi3" => Architecture::Phi3,
            "gemma2" => Architecture::Gemma2,
            other => return Err(GgufError::UnsupportedArchitecture(other.to_string())),
        };

        let prefix = arch_str.to_lowercase();

        // Vocab size: prefer embedding tensor shape (most reliable),
        // because some models have incorrect vocab_size metadata.
        let vocab_size = self
            .find_tensor("token_embd.weight")
            .or_else(|| self.find_tensor("model.embed_tokens.weight"))
            .and_then(|t| {
                if t.dimensions.len() == 2 {
                    Some(t.dimensions[1] as usize)
                } else {
                    None
                }
            })
            .or_else(|| self.meta_usize(&format!("{prefix}.vocab_size")).ok())
            .unwrap_or(32000);

        let hidden_dim = self.meta_usize(&format!("{prefix}.embedding_length"))?;
        let num_layers = self.meta_usize(&format!("{prefix}.block_count"))?;
        let num_heads = self.meta_usize(&format!("{prefix}.attention.head_count"))?;
        let num_kv_heads = self
            .meta_usize(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_heads);
        let head_dim = hidden_dim / num_heads;

        let intermediate_dim = self
            .meta_usize(&format!("{prefix}.feed_forward_length"))
            .unwrap_or(hidden_dim * 4);

        let max_seq_len = self
            .meta_usize(&format!("{prefix}.context_length"))
            .unwrap_or(2048);

        let rope_theta = self
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);

        let rms_norm_eps = self
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        // Gemma 2 logit soft-capping parameters (absent for other architectures)
        let attn_logit_softcap = self
            .meta_f32(&format!("{prefix}.attn_logit_softcapping"))
            .unwrap_or(0.0);
        let final_logit_softcap = self
            .meta_f32(&format!("{prefix}.final_logit_softcapping"))
            .unwrap_or(0.0);

        // MoE (Mixture-of-Experts) detection via GGUF metadata keys.
        // Mixtral and similar models set `llm.expert_count` and `llm.expert_used_count`.
        // Some models use the architecture-prefixed variant instead.
        let num_experts = self
            .meta_usize("llm.expert_count")
            .or_else(|_| self.meta_usize(&format!("{prefix}.expert_count")))
            .unwrap_or(0);
        let num_experts_per_token = self
            .meta_usize("llm.expert_used_count")
            .or_else(|_| self.meta_usize(&format!("{prefix}.expert_used_count")))
            .unwrap_or(0);
        let moe = num_experts > 0 && num_experts_per_token > 0;

        Ok(ModelConfig {
            architecture,
            vocab_size,
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            attn_logit_softcap,
            final_logit_softcap,
            kv_cache_bits: 32,
            moe,
            num_experts,
            num_experts_per_token,
        })
    }

    /// Read a single tensor's data from the file, dequantizing to f32.
    pub fn read_tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_info: &TensorInfo,
    ) -> Result<Tensor, GgufError> {
        let (tensor, _raw) = self.read_tensor_data_with_raw(reader, tensor_info, false)?;
        Ok(tensor)
    }

    /// Like `read_tensor_data` but also optionally returns the raw
    /// pre-dequantization bytes as a `RawWeight`.  When `keep_raw` is `true`
    /// and the tensor format maps to a supported `WeightFormat`, the raw
    /// bytes are kept alongside the dequantized `Tensor`.
    ///
    /// This exists so callers that need BOTH the f32 Tensor (for code paths
    /// that aren't SIMD-accelerated) AND the raw quantized bytes (for the
    /// int8 SIMD matvec kernels) can get both in a single pass over the
    /// tensor data — no second seek + read, and no double allocation of the
    /// read buffer.
    pub fn read_tensor_data_with_raw<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_info: &TensorInfo,
        keep_raw: bool,
    ) -> Result<(Tensor, Option<RawWeight>), GgufError> {
        let absolute_offset = self.tensor_data_offset + tensor_info.offset;
        reader.seek(SeekFrom::Start(absolute_offset))?;

        let byte_size = tensor_info.byte_size() as usize;
        let mut raw = vec![0u8; byte_size];
        reader.read_exact(&mut raw)?;

        let numel = tensor_info.numel() as usize;
        let f32_data = dequantize_tensor(&raw, tensor_info.dtype, numel)?;

        let raw_weight = if keep_raw {
            quant_to_weight_format(tensor_info.dtype).map(|format| {
                let num_rows = tensor_info.dimensions.last().copied().unwrap_or(1) as usize;
                let weights_per_block = format.weights_per_block();
                let col_elements = tensor_info
                    .dimensions
                    .iter()
                    .rev()
                    .skip(1)
                    .product::<u64>()
                    .max(1) as usize;
                let blocks_per_row = col_elements.div_ceil(weights_per_block);
                RawWeight {
                    data: raw,
                    format,
                    num_rows,
                    blocks_per_row,
                }
            })
        } else {
            None
        };

        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        let tensor = Tensor::from_vec(f32_data, &shape).map_err(|e| {
            GgufError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
        })?;
        Ok((tensor, raw_weight))
    }

    /// Load all tensor data from the file, returning a map of name → Tensor.
    pub fn load_all_tensors<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<HashMap<String, Tensor>, GgufError> {
        let mut tensors = HashMap::new();
        for info in &self.tensors {
            let tensor = self.read_tensor_data(reader, info)?;
            tensors.insert(info.name.clone(), tensor);
        }
        Ok(tensors)
    }

    /// Load all tensor data in a single pass, returning both the f32 Tensor
    /// map (for non-SIMD code paths) and a raw quantized-weight map (for
    /// int8 SIMD matvec kernels).
    ///
    /// Tensor bytes are read exactly once: dequantized to f32 for the Tensor
    /// and, for formats that map to a supported `WeightFormat`, retained as
    /// raw bytes for the SIMD path.  This is the path `flare-web` uses so
    /// that loading a quantized model doesn't pay for two full passes over
    /// the tensor data.
    pub fn load_all_tensors_with_raw<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<(HashMap<String, Tensor>, HashMap<String, RawWeight>), GgufError> {
        let mut tensors = HashMap::new();
        let mut raw_weights = HashMap::new();
        for info in &self.tensors {
            let (tensor, maybe_raw) = self.read_tensor_data_with_raw(reader, info, true)?;
            if let Some(raw) = maybe_raw {
                raw_weights.insert(info.name.clone(), raw);
            }
            tensors.insert(info.name.clone(), tensor);
        }
        Ok((tensors, raw_weights))
    }

    /// Like [`Self::load_all_tensors_with_raw`] but skips the f32 dequantization
    /// for tensors whose names appear in `skip_f32_for`, returning an
    /// empty-data `Tensor` with the correct shape in place of the f32 copy.
    ///
    /// Used by the browser load path to cut peak WASM memory during parse:
    /// the per-layer matmul weights (~270 MB of f32 on a 138 MB Q8_0 model)
    /// are served from the raw bytes directly via fused dequant+matvec, so
    /// their f32 representation is never actually read.  A skipped tensor
    /// whose format cannot be retained as raw (unsupported quant format)
    /// falls back to the normal f32 dequant path — the caller can detect
    /// this by checking the raw_weights map.
    pub fn load_all_tensors_with_raw_skipping_f32<R: Read + Seek>(
        &self,
        reader: &mut R,
        skip_f32_for: &std::collections::HashSet<String>,
    ) -> Result<(HashMap<String, Tensor>, HashMap<String, RawWeight>), GgufError> {
        let mut tensors = HashMap::new();
        let mut raw_weights = HashMap::new();
        for info in &self.tensors {
            let skip = skip_f32_for.contains(&info.name);
            let (tensor, maybe_raw) =
                self.read_tensor_data_with_raw_opt(reader, info, true, skip)?;
            if let Some(raw) = maybe_raw {
                raw_weights.insert(info.name.clone(), raw);
            }
            tensors.insert(info.name.clone(), tensor);
        }
        Ok((tensors, raw_weights))
    }

    /// Decode a single tensor from its pre-read bytes.
    ///
    /// This is the "no Reader" form of [`Self::read_tensor_data_with_raw_opt`]:
    /// callers stream the bulk GGUF from JS and hand WASM one tensor's bytes
    /// at a time, so there's no need for a seekable reader and no copy of the
    /// full file is held in WASM memory.  The return tuple mirrors
    /// `read_tensor_data_with_raw_opt`.
    pub fn decode_tensor_from_bytes(
        tensor_info: &TensorInfo,
        raw_bytes: &[u8],
        keep_raw: bool,
        skip_f32: bool,
    ) -> Result<(Tensor, Option<RawWeight>), GgufError> {
        let byte_size = tensor_info.byte_size() as usize;
        if raw_bytes.len() != byte_size {
            return Err(GgufError::InvalidFormat(format!(
                "tensor {} byte length mismatch: got {}, expected {}",
                tensor_info.name,
                raw_bytes.len(),
                byte_size
            )));
        }

        let format = quant_to_weight_format(tensor_info.dtype);
        let can_skip = skip_f32 && keep_raw && format.is_some();
        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        let numel = tensor_info.numel() as usize;

        let tensor = if can_skip {
            Tensor::from_vec(Vec::new(), &shape)
                .unwrap_or_else(|_| Tensor::from_vec(Vec::new(), &[0]).expect("0-el tensor"))
        } else {
            let f32_data = dequantize_tensor(raw_bytes, tensor_info.dtype, numel)?;
            Tensor::from_vec(f32_data, &shape).map_err(|e| {
                GgufError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            })?
        };

        let raw_weight = if keep_raw {
            format.map(|format| {
                let num_rows = tensor_info.dimensions.last().copied().unwrap_or(1) as usize;
                let weights_per_block = format.weights_per_block();
                let col_elements = tensor_info
                    .dimensions
                    .iter()
                    .rev()
                    .skip(1)
                    .product::<u64>()
                    .max(1) as usize;
                let blocks_per_row = col_elements.div_ceil(weights_per_block);
                RawWeight {
                    data: raw_bytes.to_vec(),
                    format,
                    num_rows,
                    blocks_per_row,
                }
            })
        } else {
            None
        };
        Ok((tensor, raw_weight))
    }

    /// Like [`Self::read_tensor_data_with_raw`] but with an extra flag that
    /// skips the f32 dequantization step when the raw bytes are retained.
    /// Returns an empty-data `Tensor` with the correct shape when skipped.
    pub fn read_tensor_data_with_raw_opt<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_info: &TensorInfo,
        keep_raw: bool,
        skip_f32: bool,
    ) -> Result<(Tensor, Option<RawWeight>), GgufError> {
        let absolute_offset = self.tensor_data_offset + tensor_info.offset;
        reader.seek(SeekFrom::Start(absolute_offset))?;

        let byte_size = tensor_info.byte_size() as usize;
        let mut raw = vec![0u8; byte_size];
        reader.read_exact(&mut raw)?;

        // If we're both keeping the raw bytes AND the caller has opted out of
        // the f32 representation, don't allocate or decode the f32 buffer.
        // Fall through to the normal path if the quant format isn't one we
        // can keep as raw (i.e. `quant_to_weight_format` returns `None`).
        let format = quant_to_weight_format(tensor_info.dtype);
        let can_skip = skip_f32 && keep_raw && format.is_some();

        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        let numel = tensor_info.numel() as usize;

        let tensor = if can_skip {
            // Placeholder: zero-length f32 data, real shape.  The model code
            // uses raw_weights instead of .data() for any tensor we skipped.
            Tensor::from_vec(Vec::new(), &shape).unwrap_or_else(|_| {
                // Shape parsing shouldn't fail for valid GGUF, but keep a
                // total fallback that hands back a dummy 0-element tensor.
                Tensor::from_vec(Vec::new(), &[0]).expect("0-element tensor")
            })
        } else {
            let f32_data = dequantize_tensor(&raw, tensor_info.dtype, numel)?;
            Tensor::from_vec(f32_data, &shape).map_err(|e| {
                GgufError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
            })?
        };

        let raw_weight = if keep_raw {
            format.map(|format| {
                let num_rows = tensor_info.dimensions.last().copied().unwrap_or(1) as usize;
                let weights_per_block = format.weights_per_block();
                let col_elements = tensor_info
                    .dimensions
                    .iter()
                    .rev()
                    .skip(1)
                    .product::<u64>()
                    .max(1) as usize;
                let blocks_per_row = col_elements.div_ceil(weights_per_block);
                RawWeight {
                    data: raw,
                    format,
                    num_rows,
                    blocks_per_row,
                }
            })
        } else {
            None
        };
        Ok((tensor, raw_weight))
    }

    /// Read a single tensor's raw bytes without dequantizing, along with its
    /// shape and quantization format.
    ///
    /// Returns `None` if the tensor name is not found or the format is not one
    /// of the GPU-accelerated formats supported by `WeightFormat`.
    pub fn read_raw_weight<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Option<RawWeight>, GgufError> {
        let info = match self.tensors.iter().find(|t| t.name == name) {
            Some(i) => i,
            None => return Ok(None),
        };
        let format = match quant_to_weight_format(info.dtype) {
            Some(f) => f,
            None => return Ok(None),
        };

        let absolute_offset = self.tensor_data_offset + info.offset;
        reader.seek(SeekFrom::Start(absolute_offset))?;

        let byte_size = info.byte_size() as usize;
        let mut raw = vec![0u8; byte_size];
        reader.read_exact(&mut raw)?;

        // GGUF stores dimensions in row-major order: [col, row] for 2D tensors.
        // The last dimension is the number of output rows (out_features).
        let num_rows = info.dimensions.last().copied().unwrap_or(1) as usize;
        let weights_per_block = format.weights_per_block();
        let col_elements = info.dimensions.iter().rev().skip(1).product::<u64>().max(1) as usize;
        let blocks_per_row = col_elements.div_ceil(weights_per_block);

        Ok(Some(RawWeight {
            data: raw,
            format,
            num_rows,
            blocks_per_row,
        }))
    }

    /// Load raw (non-dequantized) layer weights for a single transformer layer.
    ///
    /// Returns `None` if any of the seven required weight tensors (wq, wk, wv,
    /// wo, w_gate, w_up, w_down) is missing or uses an unsupported format.
    /// This allows callers to gracefully fall back to the f32 dequantized path
    /// for models that mix quantization formats.
    pub fn load_raw_layer_weights<R: Read + Seek>(
        &self,
        reader: &mut R,
        layer_idx: usize,
    ) -> Result<Option<RawLayerWeights>, GgufError> {
        let i = layer_idx;
        let wq = self.read_raw_weight(reader, &format!("blk.{i}.attn_q.weight"))?;
        let wk = self.read_raw_weight(reader, &format!("blk.{i}.attn_k.weight"))?;
        let wv = self.read_raw_weight(reader, &format!("blk.{i}.attn_v.weight"))?;
        let wo = self.read_raw_weight(reader, &format!("blk.{i}.attn_output.weight"))?;
        let w_gate = self.read_raw_weight(reader, &format!("blk.{i}.ffn_gate.weight"))?;
        let w_up = self.read_raw_weight(reader, &format!("blk.{i}.ffn_up.weight"))?;
        let w_down = self.read_raw_weight(reader, &format!("blk.{i}.ffn_down.weight"))?;

        match (wq, wk, wv, wo, w_gate, w_up, w_down) {
            (Some(wq), Some(wk), Some(wv), Some(wo), Some(w_gate), Some(w_up), Some(w_down)) => {
                Ok(Some(RawLayerWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    w_gate,
                    w_up,
                    w_down,
                }))
            }
            _ => Ok(None),
        }
    }
}

/// Map a `QuantFormat` to a GPU-accelerated `WeightFormat`, if one exists.
fn quant_to_weight_format(q: QuantFormat) -> Option<WeightFormat> {
    match q {
        QuantFormat::BF16 => Some(WeightFormat::BF16),
        QuantFormat::F16 => Some(WeightFormat::F16),
        QuantFormat::Q4_1 => Some(WeightFormat::Q4_1),
        QuantFormat::Q8_1 => Some(WeightFormat::Q8_1),
        QuantFormat::Q4_0 => Some(WeightFormat::Q4_0),
        QuantFormat::Q5_0 => Some(WeightFormat::Q5_0),
        QuantFormat::Q5_1 => Some(WeightFormat::Q5_1),
        QuantFormat::Q8_0 => Some(WeightFormat::Q8_0),
        QuantFormat::Q2K => Some(WeightFormat::Q2K),
        QuantFormat::Q3K => Some(WeightFormat::Q3K),
        QuantFormat::Q4K => Some(WeightFormat::Q4K),
        QuantFormat::Q5K => Some(WeightFormat::Q5K),
        QuantFormat::Q6K => Some(WeightFormat::Q6K),
        QuantFormat::IQ4NL => Some(WeightFormat::IQ4NL),
        QuantFormat::IQ4XS => Some(WeightFormat::IQ4XS),
        QuantFormat::IQ3S => Some(WeightFormat::IQ3S),
        QuantFormat::IQ2XXS => Some(WeightFormat::IQ2XXS),
        QuantFormat::IQ2XS => Some(WeightFormat::IQ2XS),
        QuantFormat::IQ3XXS => Some(WeightFormat::IQ3XXS),
        QuantFormat::IQ2S => Some(WeightFormat::IQ2S),
        QuantFormat::IQ1S => Some(WeightFormat::IQ1S),
        QuantFormat::Ternary => Some(WeightFormat::Ternary),
        _ => None,
    }
}

/// Dequantize raw bytes to f32 based on quantization format.
pub(crate) fn dequantize_tensor(
    raw: &[u8],
    dtype: QuantFormat,
    numel: usize,
) -> Result<Vec<f32>, GgufError> {
    // Reject unknown formats early
    if matches!(dtype, QuantFormat::Unknown(_)) {
        return Err(GgufError::UnsupportedQuant(dtype));
    }
    // Validate that we have enough bytes for the requested element count.
    // This prevents panics on truncated/malformed tensor data.
    let required_bytes = dtype.bytes_for_elements(numel as u64) as usize;
    if raw.len() < required_bytes {
        return Err(GgufError::InvalidFormat(format!(
            "dequantize {dtype:?}: have {} bytes, need {} for {} elements",
            raw.len(),
            required_bytes,
            numel
        )));
    }
    match dtype {
        QuantFormat::F32 => {
            let mut data = vec![0.0f32; numel];
            for (i, chunk) in raw.chunks_exact(4).enumerate() {
                data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(data)
        }
        QuantFormat::F16 => {
            let mut data = vec![0.0f32; numel];
            for (i, chunk) in raw.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                data[i] = quantize::f16_to_f32(bits);
            }
            Ok(data)
        }
        QuantFormat::BF16 => {
            let mut data = vec![0.0f32; numel];
            for (i, chunk) in raw.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                data[i] = quantize::bf16_to_f32(bits);
            }
            Ok(data)
        }
        QuantFormat::Q8_0 => {
            let block_size = 32;
            let block_bytes = 34; // 2 (scale) + 32 (data)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q8_0_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                data[dst_start..dst_end].copy_from_slice(&out[..dst_end - dst_start]);
            }
            Ok(data)
        }
        QuantFormat::Q8_1 => {
            let block_size = 32;
            let block_bytes = 36; // 2 (scale) + 2 (sum) + 32 (int8)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q8_1_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                data[dst_start..dst_end].copy_from_slice(&out[..dst_end - dst_start]);
            }
            Ok(data)
        }
        QuantFormat::Q4_0 => {
            let block_size = 32;
            let block_bytes = 18; // 2 (scale) + 16 (data)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q4_0_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                let copy_len = dst_end - dst_start;
                data[dst_start..dst_end].copy_from_slice(&out[..copy_len]);
            }
            Ok(data)
        }
        QuantFormat::IQ4NL => {
            let block_size = 32;
            let block_bytes = 18; // 2 (scale) + 16 (nibbles), same layout as Q4_0
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_iq4nl_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                data[dst_start..dst_end].copy_from_slice(&out[..dst_end - dst_start]);
            }
            Ok(data)
        }
        QuantFormat::Q4_1 => {
            let block_size = 32;
            let block_bytes = 20; // 2 (scale) + 2 (min) + 16 (nibbles)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q4_1_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                data[dst_start..dst_end].copy_from_slice(&out[..dst_end - dst_start]);
            }
            Ok(data)
        }
        QuantFormat::Q5_0 => {
            let block_size = 32;
            let block_bytes = 22; // 2 (scale) + 4 (high bits) + 16 (nibbles)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q5_0_block(block, &mut out);
                let dst_start = b * block_size;
                let dst_end = (dst_start + block_size).min(numel);
                data[dst_start..dst_end].copy_from_slice(&out[..dst_end - dst_start]);
            }
            Ok(data)
        }
        QuantFormat::Q2K => dequant_k_blocks(numel, 256, 84, raw, quantize::dequant_q2k_block),
        QuantFormat::Q3K => dequant_k_blocks(numel, 256, 110, raw, quantize::dequant_q3k_block),
        QuantFormat::Q6K => dequant_k_blocks(numel, 256, 210, raw, quantize::dequant_q6k_block),
        QuantFormat::Q4K => dequant_k_blocks(numel, 256, 144, raw, quantize::dequant_q4k_block),
        QuantFormat::Q5K => dequant_k_blocks(numel, 256, 176, raw, quantize::dequant_q5k_block),
        QuantFormat::Ternary => {
            // Ternary: 4 weights per byte, 2 bits each.
            // Encoding: 00=0, 01=+1, 10=-1, 11=unused (treated as 0).
            let mut data = vec![0.0f32; numel];
            for (i, val) in data.iter_mut().enumerate() {
                let byte_idx = i / 4;
                let bit_shift = (i % 4) * 2;
                if byte_idx < raw.len() {
                    let bits = (raw[byte_idx] >> bit_shift) & 0b11;
                    *val = match bits {
                        0b01 => 1.0,
                        0b10 => -1.0,
                        _ => 0.0,
                    };
                }
            }
            Ok(data)
        }
        other => Err(GgufError::UnsupportedQuant(other)),
    }
}

/// Dequantize K-quant blocks (Q4_K, Q5_K, Q6_K) with proper handling of
/// partial last blocks when numel is not a multiple of block_size.
fn dequant_k_blocks(
    numel: usize,
    block_size: usize,
    block_bytes: usize,
    raw: &[u8],
    dequant_fn: fn(&[u8], &mut [f32; 256]),
) -> Result<Vec<f32>, GgufError> {
    let num_blocks = numel.div_ceil(block_size);
    let mut data = vec![0.0f32; numel];
    for b in 0..num_blocks {
        let block = &raw[b * block_bytes..(b + 1) * block_bytes];
        let mut out = [0.0f32; 256];
        dequant_fn(block, &mut out);
        let start = b * block_size;
        let end = (start + block_size).min(numel);
        data[start..end].copy_from_slice(&out[..end - start]);
    }
    Ok(data)
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    // Cap string length to prevent OOM from malicious files
    const MAX_STRING_LEN: u64 = 1024 * 1024; // 1 MB
    let len = reader.read_u64::<LittleEndian>()?;
    if len > MAX_STRING_LEN {
        return Err(GgufError::InvalidFormat(format!(
            "string length {len} exceeds maximum {MAX_STRING_LEN}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue, GgufError> {
    let value_type = reader.read_u32::<LittleEndian>()?;
    match value_type {
        0 => Ok(MetadataValue::Uint8(reader.read_u8()?)),
        1 => Ok(MetadataValue::Int8(reader.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_gguf_string(reader)?)),
        9 => {
            // Array
            let element_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                // Read raw values of the element type
                let val = read_typed_value(reader, element_type)?;
                values.push(val);
            }
            Ok(MetadataValue::Array(values))
        }
        10 => Ok(MetadataValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(GgufError::InvalidValueType(value_type)),
    }
}

fn read_typed_value<R: Read>(reader: &mut R, type_id: u32) -> Result<MetadataValue, GgufError> {
    match type_id {
        0 => Ok(MetadataValue::Uint8(reader.read_u8()?)),
        1 => Ok(MetadataValue::Int8(reader.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_gguf_string(reader)?)),
        10 => Ok(MetadataValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(GgufError::InvalidValueType(type_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_gguf_header(buf: &mut Vec<u8>, tensor_count: u64, metadata_count: u64) {
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&metadata_count.to_le_bytes());
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_parse_empty_gguf() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 0);

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; 32];
        let mut cursor = Cursor::new(buf);
        let result = GgufFile::parse_header(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_metadata() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 1);

        // Write one string metadata entry
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        write_gguf_string(&mut buf, "llama");

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        assert_eq!(file.architecture(), Some("llama"));
    }

    // Helper: write a metadata key-value pair
    fn write_meta_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type: uint32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn write_meta_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes()); // type: float32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn write_meta_str(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        write_gguf_string(buf, value);
    }

    /// Build a complete GGUF buffer with Llama-style metadata for config extraction.
    fn build_llama_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        let meta_count = 8;
        write_gguf_header(&mut buf, 0, meta_count);

        write_meta_str(&mut buf, "general.architecture", "llama");
        write_meta_u32(&mut buf, "llama.embedding_length", 256);
        write_meta_u32(&mut buf, "llama.block_count", 4);
        write_meta_u32(&mut buf, "llama.attention.head_count", 8);
        write_meta_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_meta_u32(&mut buf, "llama.feed_forward_length", 512);
        write_meta_u32(&mut buf, "llama.context_length", 2048);
        write_meta_f32(&mut buf, "llama.rope.freq_base", 500000.0);

        buf
    }

    #[test]
    fn test_parse_multiple_metadata_types() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 3);

        write_meta_u32(&mut buf, "count", 42);
        write_meta_f32(&mut buf, "ratio", 1.234);
        write_meta_str(&mut buf, "name", "test_model");

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        assert_eq!(file.metadata["count"].as_u32(), Some(42));
        assert!((file.metadata["ratio"].as_f32().unwrap() - 1.234).abs() < 0.01);
        assert_eq!(file.metadata["name"].as_str(), Some("test_model"));
    }

    #[test]
    fn test_to_model_config_llama() {
        let buf = build_llama_gguf();
        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        let config = file.to_model_config().unwrap();

        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 32); // 256 / 8
        assert_eq!(config.intermediate_dim, 512);
        assert_eq!(config.max_seq_len, 2048);
        assert!((config.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_to_model_config_qwen2() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 4);

        write_meta_str(&mut buf, "general.architecture", "qwen2");
        write_meta_u32(&mut buf, "qwen2.embedding_length", 1024);
        write_meta_u32(&mut buf, "qwen2.block_count", 12);
        write_meta_u32(&mut buf, "qwen2.attention.head_count", 16);

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        let config = file.to_model_config().unwrap();

        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 16);
        // kv_heads defaults to num_heads when not specified
        assert_eq!(config.num_kv_heads, 16);
    }

    #[test]
    fn test_to_model_config_defaults() {
        // Only required fields, optional fields should use defaults
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 4);

        write_meta_str(&mut buf, "general.architecture", "llama");
        write_meta_u32(&mut buf, "llama.embedding_length", 128);
        write_meta_u32(&mut buf, "llama.block_count", 2);
        write_meta_u32(&mut buf, "llama.attention.head_count", 4);

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        let config = file.to_model_config().unwrap();

        assert_eq!(config.num_kv_heads, 4); // defaults to num_heads
        assert_eq!(config.intermediate_dim, 512); // defaults to hidden_dim * 4
        assert_eq!(config.max_seq_len, 2048); // default
        assert!((config.rope_theta - 10000.0).abs() < 1.0); // default
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-7); // default
    }

    #[test]
    fn test_to_model_config_missing_required() {
        // Missing embedding_length should error
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 2);

        write_meta_str(&mut buf, "general.architecture", "llama");
        write_meta_u32(&mut buf, "llama.block_count", 2);

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        assert!(file.to_model_config().is_err());
    }

    #[test]
    fn test_to_model_config_unsupported_arch() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 1);
        write_meta_str(&mut buf, "general.architecture", "gpt_neox");

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        let result = file.to_model_config();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("gpt_neox"));
    }

    #[test]
    fn test_parse_tensor_info() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 1, 0);

        // Write one tensor info
        write_gguf_string(&mut buf, "blk.0.attn_q.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&4096u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype: F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(file.tensors[0].dimensions, vec![4096, 4096]);
        assert_eq!(file.tensors[0].dtype, QuantFormat::F32);
        assert_eq!(file.tensors[0].numel(), 4096 * 4096);
    }

    #[test]
    fn test_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes()); // unsupported version
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(buf);
        let result = GgufFile::parse_header(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("99"));
    }

    #[test]
    fn test_dequant_k_blocks_partial_last_block() {
        // 300 elements with block_size=256 → 2 blocks, last has only 44 elements
        fn dummy_dequant(block: &[u8], out: &mut [f32; 256]) {
            let _ = block;
            for (i, v) in out.iter_mut().enumerate() {
                *v = i as f32;
            }
        }

        // 2 blocks × 144 bytes = 288 bytes of raw data
        let raw = vec![0u8; 288];
        let result = super::dequant_k_blocks(300, 256, 144, &raw, dummy_dequant).unwrap();

        assert_eq!(result.len(), 300);
        // First block: 0..255
        assert_eq!(result[0], 0.0);
        assert_eq!(result[255], 255.0);
        // Second block (partial): only 44 elements copied
        assert_eq!(result[256], 0.0); // out[0] of second block
        assert_eq!(result[299], 43.0); // out[43] of second block
    }

    // -----------------------------------------------------------------------
    // Helpers for building a synthetic GGUF with quantized layer tensors
    // -----------------------------------------------------------------------

    /// Write a tensor info entry into a GGUF buffer.
    ///
    /// `dtype_id`: raw GGUF type code (3 = Q4_1).
    /// `dims`: e.g. `&[num_rows as u64, num_cols as u64]`.
    /// `offset`: byte offset within the tensor-data section.
    fn write_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], dtype_id: u32, offset: u64) {
        write_gguf_string(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&dtype_id.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Build a synthetic GGUF buffer containing one transformer layer's weight
    /// tensors in Q4_1 format (dtype_id=3, 20 bytes/block, 32 weights/block).
    ///
    /// The model shape is:
    ///   - num_rows = 4, num_cols = 64 → 2 blocks/row → 160 bytes/tensor
    ///
    /// Returns the full buffer and the expected raw bytes for one tensor.
    fn build_q4_1_layer_gguf() -> (Vec<u8>, Vec<u8>) {
        // GGUF stores dimensions as [in_features, out_features].
        // So dims=[64, 4] means 4 output rows of 64 input columns each.
        const NUM_COLS: u64 = 64; // in_features (dimensions[0])
        const NUM_ROWS: u64 = 4; // out_features (dimensions[1])
        const BLOCKS_PER_ROW: usize = 2; // 64 / 32
        const BYTES_PER_BLOCK: usize = 20; // Q4_1
        const TENSOR_BYTES: usize = NUM_ROWS as usize * BLOCKS_PER_ROW * BYTES_PER_BLOCK; // 160

        let tensor_names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let num_tensors = tensor_names.len();

        let mut header = Vec::new();
        write_gguf_header(&mut header, num_tensors as u64, 0);

        let dims = [NUM_COLS, NUM_ROWS];
        for (i, name) in tensor_names.iter().enumerate() {
            write_tensor_info(
                &mut header,
                name,
                &dims,
                3, // Q4_1
                (i * TENSOR_BYTES) as u64,
            );
        }

        // Pad header to 32-byte alignment boundary
        let alignment = 32usize;
        let padded = header.len().div_ceil(alignment) * alignment;
        header.resize(padded, 0u8);

        // Build per-tensor raw data: fill with recognisable pattern so we can
        // verify the right bytes are returned.
        let expected_tensor: Vec<u8> = (0..TENSOR_BYTES).map(|i| (i & 0xFF) as u8).collect();
        let mut tensor_data = Vec::new();
        for _ in 0..num_tensors {
            tensor_data.extend_from_slice(&expected_tensor);
        }

        let mut buf = header;
        buf.extend_from_slice(&tensor_data);
        (buf, expected_tensor)
    }

    #[test]
    fn test_read_raw_weight_q4_1_metadata() {
        use flare_core::model::WeightFormat;

        let (buf, _) = build_q4_1_layer_gguf();
        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        let rw = file
            .read_raw_weight(&mut cursor, "blk.0.attn_q.weight")
            .unwrap()
            .expect("should find blk.0.attn_q.weight");

        assert_eq!(rw.format, WeightFormat::Q4_1);
        assert_eq!(rw.num_rows, 4);
        assert_eq!(rw.blocks_per_row, 2);
        assert_eq!(rw.data.len(), 160); // 4 rows × 2 blocks × 20 bytes
    }

    #[test]
    fn test_read_raw_weight_missing_returns_none() {
        let (buf, _) = build_q4_1_layer_gguf();
        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        let result = file
            .read_raw_weight(&mut cursor, "blk.99.attn_q.weight")
            .unwrap();
        assert!(result.is_none(), "missing tensor should return None");
    }

    #[test]
    fn test_read_raw_weight_bytes_match() {
        let (buf, expected) = build_q4_1_layer_gguf();
        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        // Third tensor (attn_v) starts at offset 2 * 160 = 320 within tensor data.
        let rw = file
            .read_raw_weight(&mut cursor, "blk.0.attn_v.weight")
            .unwrap()
            .unwrap();

        // The expected bytes are the same pattern regardless of which tensor we
        // pick, because build_q4_1_layer_gguf writes the same pattern for each.
        assert_eq!(rw.data, expected);
    }

    #[test]
    fn test_load_raw_layer_weights_complete() {
        use flare_core::model::WeightFormat;

        let (buf, _) = build_q4_1_layer_gguf();
        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        let layer = file
            .load_raw_layer_weights(&mut cursor, 0)
            .unwrap()
            .expect("all seven tensors should be present");

        for rw in [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
        ] {
            assert_eq!(rw.format, WeightFormat::Q4_1);
            assert_eq!(rw.num_rows, 4);
            assert_eq!(rw.blocks_per_row, 2);
            assert_eq!(rw.data.len(), 160);
        }
    }

    #[test]
    fn test_load_raw_layer_weights_missing_tensor_returns_none() {
        // Build a GGUF with only 6 of the 7 required tensors (omit ffn_down).
        const NUM_ROWS: u64 = 4;
        const NUM_COLS: u64 = 64;
        const TENSOR_BYTES: usize = 4 * 2 * 20;

        let six_names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            // ffn_down intentionally omitted
        ];

        let mut header = Vec::new();
        write_gguf_header(&mut header, six_names.len() as u64, 0);
        for (i, name) in six_names.iter().enumerate() {
            write_tensor_info(
                &mut header,
                name,
                &[NUM_COLS, NUM_ROWS],
                3,
                (i * TENSOR_BYTES) as u64,
            );
        }
        let padded = header.len().div_ceil(32) * 32;
        header.resize(padded, 0u8);
        header.extend(vec![0u8; six_names.len() * TENSOR_BYTES]);

        let mut cursor = Cursor::new(header);
        let file = GgufFile::parse_header(&mut cursor).unwrap();

        let result = file.load_raw_layer_weights(&mut cursor, 0).unwrap();
        assert!(
            result.is_none(),
            "missing ffn_down should cause load to return None"
        );
    }

    // ── dequantize_tensor direct tests ────────────────────────────────────────

    #[test]
    fn test_dequantize_tensor_f32_passthrough() {
        // F32: raw bytes are just little-endian f32 values
        let val: f32 = 42.5;
        let raw = val.to_le_bytes().to_vec();
        let result = super::dequantize_tensor(&raw, QuantFormat::F32, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - val).abs() < 1e-6,
            "f32 passthrough: got {}",
            result[0]
        );
    }

    #[test]
    fn test_dequantize_tensor_f32_multiple() {
        // F32: two values
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        raw.extend_from_slice(&(-2.0f32).to_le_bytes());
        let result = super::dequantize_tensor(&raw, QuantFormat::F32, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_tensor_f16() {
        // F16: 0x3C00 = 1.0, 0xBC00 = -1.0
        let raw = vec![0x00u8, 0x3C, 0x00, 0xBC];
        let result = super::dequantize_tensor(&raw, QuantFormat::F16, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6, "f16[0] = {}", result[0]);
        assert!((result[1] - (-1.0)).abs() < 1e-6, "f16[1] = {}", result[1]);
    }

    #[test]
    fn test_dequantize_tensor_bf16() {
        // BF16: 0x3F80 = 1.0, 0xBF80 = -1.0
        let raw = vec![0x80u8, 0x3F, 0x80, 0xBF];
        let result = super::dequantize_tensor(&raw, QuantFormat::BF16, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6, "bf16[0] = {}", result[0]);
        assert!((result[1] - (-1.0)).abs() < 1e-6, "bf16[1] = {}", result[1]);
    }

    #[test]
    fn test_dequantize_tensor_q8_0() {
        // Q8_0 block: scale f16=1.0 (0x3C00) + 32 bytes of i8=0 → all 0.0
        let mut raw = vec![0u8; 34];
        raw[0] = 0x00;
        raw[1] = 0x3C; // scale = 1.0
                       // qs[0..32] all 0 → weights all 0.0
        let result = super::dequantize_tensor(&raw, QuantFormat::Q8_0, 32).unwrap();
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "q8_0 zero: result[{i}] = {v}");
        }
    }

    #[test]
    fn test_dequantize_tensor_q8_0_nonzero() {
        // Q8_0 block: scale=1.0, qs[0]=5 → result[0]=5.0
        let mut raw = vec![0u8; 34];
        raw[0] = 0x00;
        raw[1] = 0x3C; // scale = 1.0
        raw[2] = 5i8 as u8; // qs[0] = 5
        let result = super::dequantize_tensor(&raw, QuantFormat::Q8_0, 32).unwrap();
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "q8_0 nonzero: result[0] = {}",
            result[0]
        );
    }

    #[test]
    fn test_dequantize_tensor_q4_0() {
        // Q4_0 block (18 bytes): scale=1.0, all nibbles=8 → weight=(8-8)*1.0=0.0
        let mut raw = vec![0u8; 18];
        raw[0] = 0x00;
        raw[1] = 0x3C; // scale = 1.0
        raw[2..18].fill(0x88); // all nibbles = 8 → (8-8)*scale = 0.0
        let result = super::dequantize_tensor(&raw, QuantFormat::Q4_0, 32).unwrap();
        assert_eq!(result.len(), 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "q4_0 zero: result[{i}] = {v}");
        }
    }

    #[test]
    fn test_dequantize_tensor_unsupported_quant() {
        // Unknown quant format should return UnsupportedQuant error
        let raw = vec![0u8; 18];
        let result = super::dequantize_tensor(&raw, QuantFormat::Unknown(999), 32);
        assert!(result.is_err());
        match result.unwrap_err() {
            GgufError::UnsupportedQuant(_) => {}
            e => panic!("expected UnsupportedQuant, got {e:?}"),
        }
    }

    #[test]
    fn test_tensor_info_numel_scalar() {
        // Empty dimensions → empty product = 1, max(1) = 1
        let ti = TensorInfo {
            name: "scalar".into(),
            dimensions: vec![],
            dtype: QuantFormat::F32,
            offset: 0,
        };
        assert_eq!(ti.numel(), 1);
    }

    #[test]
    fn test_tensor_info_numel_multidim() {
        let ti = TensorInfo {
            name: "weights".into(),
            dimensions: vec![4, 8, 2],
            dtype: QuantFormat::F32,
            offset: 0,
        };
        assert_eq!(ti.numel(), 64);
    }

    #[test]
    fn test_metadata_value_as_u32_non_numeric() {
        // as_u32() should return None for non-Uint32 variants
        assert_eq!(MetadataValue::String("hello".into()).as_u32(), None);
        assert_eq!(MetadataValue::Bool(true).as_u32(), None);
        assert_eq!(MetadataValue::Array(vec![]).as_u32(), None);
        assert_eq!(MetadataValue::Float32(1.0).as_u32(), None);
    }

    #[test]
    fn test_metadata_value_as_str_on_non_string() {
        // as_str() should return None for non-String variants
        assert_eq!(MetadataValue::Uint32(42).as_str(), None);
        assert_eq!(MetadataValue::Bool(false).as_str(), None);
    }

    #[test]
    fn test_gguf_file_no_architecture_returns_none() {
        let file = GgufFile {
            version: 3,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            tensor_data_offset: 0,
        };
        assert_eq!(file.architecture(), None);
    }

    #[test]
    fn test_parse_empty_string_metadata() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 0, 1);
        write_gguf_string(&mut buf, "some.key");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        write_gguf_string(&mut buf, ""); // empty string value

        let mut cursor = Cursor::new(buf);
        let file = GgufFile::parse_header(&mut cursor).unwrap();
        let val = file.metadata.get("some.key").unwrap();
        assert_eq!(val.as_str(), Some(""));
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    proptest! {
        // Random byte streams should never panic the parser
        #[test]
        fn parse_header_random_bytes_never_panics(
            data in proptest::collection::vec(any::<u8>(), 0..4096)
        ) {
            let mut cursor = Cursor::new(data);
            let _ = GgufFile::parse_header(&mut cursor);
        }

        // Streams with valid magic should also never panic
        #[test]
        fn parse_header_valid_magic_never_panics(
            tail in proptest::collection::vec(any::<u8>(), 0..2048)
        ) {
            let mut data = Vec::new();
            data.extend_from_slice(b"GGUF");
            data.extend_from_slice(&3u32.to_le_bytes());
            data.extend(tail);
            let mut cursor = Cursor::new(data);
            let _ = GgufFile::parse_header(&mut cursor);
        }

        // Random tensor count + metadata count should not OOM or panic
        #[test]
        fn parse_header_random_counts_never_panics(
            tensor_count in any::<u64>(),
            meta_count in any::<u64>(),
            tail in proptest::collection::vec(any::<u8>(), 0..512)
        ) {
            let mut data = Vec::new();
            data.extend_from_slice(b"GGUF");
            data.extend_from_slice(&3u32.to_le_bytes());
            data.extend_from_slice(&tensor_count.to_le_bytes());
            data.extend_from_slice(&meta_count.to_le_bytes());
            data.extend(tail);
            let mut cursor = Cursor::new(data);
            let _ = GgufFile::parse_header(&mut cursor);
        }

        // read_raw_weight with arbitrary names should never panic
        #[test]
        fn read_raw_weight_random_name_never_panics(
            data in proptest::collection::vec(any::<u8>(), 100..2000),
            name in "[a-z._0-9]{1,50}"
        ) {
            let mut cursor = Cursor::new(data);
            if let Ok(file) = GgufFile::parse_header(&mut cursor) {
                let _ = file.read_raw_weight(&mut cursor, &name);
            }
        }

        // Dequantize should never panic on random byte inputs
        #[test]
        fn dequantize_q8_0_random_never_panics(
            raw in proptest::collection::vec(any::<u8>(), 0..1024),
            numel in 0usize..256
        ) {
            let _ = dequantize_tensor(&raw, QuantFormat::Q8_0, numel);
        }

        #[test]
        fn dequantize_q4_0_random_never_panics(
            raw in proptest::collection::vec(any::<u8>(), 0..1024),
            numel in 0usize..256
        ) {
            let _ = dequantize_tensor(&raw, QuantFormat::Q4_0, numel);
        }

        #[test]
        fn dequantize_q4_1_random_never_panics(
            raw in proptest::collection::vec(any::<u8>(), 0..1024),
            numel in 0usize..256
        ) {
            let _ = dequantize_tensor(&raw, QuantFormat::Q4_1, numel);
        }

        #[test]
        fn dequantize_q5_0_random_never_panics(
            raw in proptest::collection::vec(any::<u8>(), 0..1024),
            numel in 0usize..256
        ) {
            let _ = dequantize_tensor(&raw, QuantFormat::Q5_0, numel);
        }
    }

    // Specific edge case tests
    #[test]
    fn empty_file_returns_error() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        assert!(GgufFile::parse_header(&mut cursor).is_err());
    }

    #[test]
    fn just_magic_bytes_returns_error() {
        let mut cursor = Cursor::new(b"GGUF".to_vec());
        assert!(GgufFile::parse_header(&mut cursor).is_err());
    }

    #[test]
    fn wrong_magic_returns_error() {
        let mut data = b"XXXX".to_vec();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        assert!(GgufFile::parse_header(&mut cursor).is_err());
    }

    #[test]
    fn truncated_after_magic_returns_error() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[0u8, 0, 0]); // only 3 bytes of version
        let mut cursor = Cursor::new(data);
        assert!(GgufFile::parse_header(&mut cursor).is_err());
    }

    #[test]
    fn massive_tensor_count_does_not_oom() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // huge tensor count
        data.extend_from_slice(&0u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        // Should error gracefully, not allocate u64::MAX entries
        let _ = GgufFile::parse_header(&mut cursor);
    }

    #[test]
    fn massive_metadata_count_does_not_oom() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // huge meta count
        let mut cursor = Cursor::new(data);
        let _ = GgufFile::parse_header(&mut cursor);
    }

    #[test]
    fn dequant_zero_elements() {
        let result = dequantize_tensor(&[], QuantFormat::Q8_0, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn dequant_truncated_data_errors_gracefully() {
        // Q8_0 needs 34 bytes per 32 elements, so 32 elements need 34 bytes
        // Provide only 10 bytes
        let result = dequantize_tensor(&[0u8; 10], QuantFormat::Q8_0, 32);
        // Should error or return zeros, not panic
        let _ = result;
    }
}

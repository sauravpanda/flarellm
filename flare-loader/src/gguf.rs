use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use thiserror::Error;

use crate::quantize::{self, QuantFormat};
use flare_core::config::{Architecture, ModelConfig};
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

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_gguf_string(reader)?;
            let value = read_metadata_value(reader)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
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
        })
    }

    /// Read a single tensor's data from the file, dequantizing to f32.
    pub fn read_tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        tensor_info: &TensorInfo,
    ) -> Result<Tensor, GgufError> {
        let absolute_offset = self.tensor_data_offset + tensor_info.offset;
        reader.seek(SeekFrom::Start(absolute_offset))?;

        let byte_size = tensor_info.byte_size() as usize;
        let mut raw = vec![0u8; byte_size];
        reader.read_exact(&mut raw)?;

        let numel = tensor_info.numel() as usize;
        let f32_data = dequantize_tensor(&raw, tensor_info.dtype, numel)?;

        let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        Tensor::from_vec(f32_data, &shape)
            .map_err(|e| GgufError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))
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
}

/// Dequantize raw bytes to f32 based on quantization format.
fn dequantize_tensor(raw: &[u8], dtype: QuantFormat, numel: usize) -> Result<Vec<f32>, GgufError> {
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
        QuantFormat::Q8_0 => {
            let block_size = 32;
            let block_bytes = 34; // 2 (scale) + 32 (data)
            let num_blocks = numel.div_ceil(block_size);
            let mut data = vec![0.0f32; numel];
            for b in 0..num_blocks {
                let block = &raw[b * block_bytes..(b + 1) * block_bytes];
                let mut out = [0.0f32; 32];
                quantize::dequant_q8_0_block(block, &mut out);
                data[b * block_size..(b + 1) * block_size].copy_from_slice(&out);
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
                data[b * block_size..(b + 1) * block_size].copy_from_slice(&out);
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
                data[b * block_size..(b + 1) * block_size].copy_from_slice(&out);
            }
            Ok(data)
        }
        QuantFormat::Q3K => dequant_k_blocks(numel, 256, 110, raw, quantize::dequant_q3k_block),
        QuantFormat::Q6K => dequant_k_blocks(numel, 256, 210, raw, quantize::dequant_q6k_block),
        QuantFormat::Q4K => dequant_k_blocks(numel, 256, 144, raw, quantize::dequant_q4k_block),
        QuantFormat::Q5K => dequant_k_blocks(numel, 256, 176, raw, quantize::dequant_q5k_block),
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
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
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
}

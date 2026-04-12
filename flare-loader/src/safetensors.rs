use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt};
use serde::Deserialize;
use thiserror::Error;

use flare_core::tensor::Tensor;

use crate::quantize::QuantFormat;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SafeTensorsError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON header parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("header size {0} exceeds sanity limit ({1} bytes)")]
    HeaderTooLarge(u64, u64),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("data size mismatch for tensor \"{name}\": expected {expected} bytes, got {got}")]
    DataSizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("tensor error: {0}")]
    Tensor(#[from] flare_core::tensor::TensorError),
}

// ---------------------------------------------------------------------------
// Dtype
// ---------------------------------------------------------------------------

/// Data types supported in SafeTensors files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    /// Size of a single element in bytes.
    pub fn element_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
        }
    }

    /// Map to the closest `QuantFormat`.
    pub fn to_quant_format(self) -> QuantFormat {
        match self {
            Dtype::F32 => QuantFormat::F32,
            Dtype::F16 | Dtype::BF16 => QuantFormat::F16,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::F32 => write!(f, "F32"),
            Dtype::F16 => write!(f, "F16"),
            Dtype::BF16 => write!(f, "BF16"),
        }
    }
}

// ---------------------------------------------------------------------------
// JSON header schema
// ---------------------------------------------------------------------------

/// Raw JSON entry for a single tensor in the header.
#[derive(Debug, Clone, Deserialize)]
struct RawTensorEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

// ---------------------------------------------------------------------------
// Parsed types
// ---------------------------------------------------------------------------

/// Metadata for a single tensor inside a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte offset of the first data byte, relative to the start of the
    /// tensor-data region (i.e. right after the JSON header).
    pub start: usize,
    /// Byte offset one past the last data byte (exclusive), relative to the
    /// start of the tensor-data region.
    pub end: usize,
}

impl SafeTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Expected byte length of the raw data.
    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

/// A parsed SafeTensors file header.  Tensor data is read lazily via offsets.
#[derive(Debug)]
pub struct SafeTensorsFile {
    /// Optional string-to-string metadata embedded in the header.
    pub metadata: HashMap<String, String>,
    /// Tensor descriptors, keyed by name.
    pub tensors: HashMap<String, SafeTensorInfo>,
    /// Absolute byte offset in the file where the tensor data region begins
    /// (immediately after the JSON header).
    pub data_offset: u64,
}

impl SafeTensorsFile {
    // Maximum header size we are willing to allocate (256 MiB).
    const MAX_HEADER_SIZE: u64 = 256 * 1024 * 1024;

    /// Parse the SafeTensors header from a reader.
    ///
    /// This only reads the 8-byte length prefix and the JSON header; it does
    /// NOT read any tensor data.
    pub fn parse_header<R: Read + Seek>(reader: &mut R) -> Result<Self, SafeTensorsError> {
        // 1. Read the 8-byte header size (u64 LE).
        let header_size = reader.read_u64::<LittleEndian>()?;
        if header_size > Self::MAX_HEADER_SIZE {
            return Err(SafeTensorsError::HeaderTooLarge(
                header_size,
                Self::MAX_HEADER_SIZE,
            ));
        }

        // 2. Read the JSON header bytes.
        let mut header_buf = vec![0u8; header_size as usize];
        reader.read_exact(&mut header_buf)?;

        // The tensor data starts right after the header.
        let data_offset = 8 + header_size;

        // 3. Deserialize as a map of string → raw JSON value so we can
        //    separate `__metadata__` from tensor entries.
        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)?;

        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();

        for (key, value) in raw_map {
            if key == "__metadata__" {
                // Parse the metadata object as string→string.
                if let serde_json::Value::Object(map) = value {
                    for (mk, mv) in map {
                        if let serde_json::Value::String(s) = mv {
                            metadata.insert(mk, s);
                        }
                    }
                }
            } else {
                // Parse as a tensor entry.
                let entry: RawTensorEntry = serde_json::from_value(value)?;
                tensors.insert(
                    key.clone(),
                    SafeTensorInfo {
                        name: key,
                        dtype: entry.dtype,
                        shape: entry.shape,
                        start: entry.data_offsets[0],
                        end: entry.data_offsets[1],
                    },
                );
            }
        }

        Ok(Self {
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Look up a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.tensors.get(name)
    }

    /// Return an iterator over all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Read raw bytes for a specific tensor from the reader.
    pub fn read_tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Vec<u8>, SafeTensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_owned()))?;

        let abs_start = self.data_offset + info.start as u64;
        reader.seek(SeekFrom::Start(abs_start))?;

        let len = info.byte_len();
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a tensor and convert it to a `flare_core::tensor::Tensor` (f32).
    ///
    /// F16 and BF16 data are dequantized to f32 on the fly.
    pub fn read_tensor<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<Tensor, SafeTensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_owned()))?;

        let numel = info.numel();
        let expected_bytes = numel * info.dtype.element_size();
        if info.byte_len() != expected_bytes {
            return Err(SafeTensorsError::DataSizeMismatch {
                name: name.to_owned(),
                expected: expected_bytes,
                got: info.byte_len(),
            });
        }

        let raw = self.read_tensor_data(reader, name)?;
        let f32_data = decode_to_f32(&raw, info.dtype, numel)?;
        let tensor = Tensor::from_vec(f32_data, &info.shape)?;
        Ok(tensor)
    }
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

/// Decode raw bytes into a Vec<f32> according to `dtype`.
fn decode_to_f32(data: &[u8], dtype: Dtype, numel: usize) -> Result<Vec<f32>, SafeTensorsError> {
    match dtype {
        Dtype::F32 => {
            let mut out = vec![0f32; numel];
            // Safe: we have verified that data.len() == numel * 4.
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(numel);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(numel);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16_to_f32(bits));
            }
            Ok(out)
        }
    }
}

/// Convert IEEE 754 half-precision (binary16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normalise.
        let mut m = mant;
        let mut e = 0u32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e = e.wrapping_add(1);
        }
        let exp32 = 127u32.wrapping_sub(15).wrapping_sub(e).wrapping_add(1);
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | mant32);
    }
    if exp == 31 {
        // Inf / NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let exp32 = exp + 127 - 15;
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}

/// Convert bfloat16 bits to f32.
///
/// bfloat16 is simply the upper 16 bits of an f32, so conversion is a
/// left-shift by 16.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal SafeTensors file in memory and return the bytes.
    fn build_safetensors(
        tensors: &[(&str, Dtype, &[usize], &[u8])],
        metadata: Option<&[(&str, &str)]>,
    ) -> Vec<u8> {
        // Compute data offsets and build the JSON header map.
        let mut header_map = serde_json::Map::new();

        if let Some(meta) = metadata {
            let mut m = serde_json::Map::new();
            for (k, v) in meta {
                m.insert(k.to_string(), serde_json::Value::String(v.to_string()));
            }
            header_map.insert("__metadata__".to_string(), serde_json::Value::Object(m));
        }

        let mut offset = 0usize;
        let mut all_data: Vec<u8> = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = offset;
            let end = start + data.len();
            offset = end;

            let dtype_str = match dtype {
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
            };

            let entry = serde_json::json!({
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [start, end],
            });
            header_map.insert(name.to_string(), entry);
            all_data.extend_from_slice(data);
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header_map)).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&all_data);
        buf
    }

    #[test]
    fn test_parse_empty() {
        let file_bytes = build_safetensors(&[], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();
        assert!(st.tensors.is_empty());
        assert!(st.metadata.is_empty());
    }

    #[test]
    fn test_parse_metadata() {
        let file_bytes = build_safetensors(&[], Some(&[("format", "pt"), ("version", "1")]));
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();
        assert_eq!(st.metadata.get("format").unwrap(), "pt");
        assert_eq!(st.metadata.get("version").unwrap(), "1");
    }

    #[test]
    fn test_read_f32_tensor() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let file_bytes = build_safetensors(&[("weight", Dtype::F32, &[2, 3], &raw)], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        assert!(st.find_tensor("weight").is_some());
        let info = st.find_tensor("weight").unwrap();
        assert_eq!(info.dtype, Dtype::F32);
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.numel(), 6);

        let tensor = st.read_tensor(&mut cursor, "weight").unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_f16_tensor() {
        // f16 1.0 = 0x3C00
        let raw: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // [1.0, 2.0] in f16
        let file_bytes = build_safetensors(&[("bias", Dtype::F16, &[2], &raw)], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        let tensor = st.read_tensor(&mut cursor, "bias").unwrap();
        assert_eq!(tensor.shape(), &[2]);
        assert!((tensor.data()[0] - 1.0).abs() < 1e-3);
        assert!((tensor.data()[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_read_bf16_tensor() {
        // bf16 is upper 16 bits of f32.
        // f32 1.0 = 0x3F80_0000 → bf16 = 0x3F80
        // f32 -2.0 = 0xC000_0000 → bf16 = 0xC000
        let raw: Vec<u8> = vec![0x80, 0x3F, 0x00, 0xC0];
        let file_bytes = build_safetensors(&[("x", Dtype::BF16, &[2], &raw)], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        let tensor = st.read_tensor(&mut cursor, "x").unwrap();
        assert_eq!(tensor.shape(), &[2]);
        assert!((tensor.data()[0] - 1.0).abs() < 1e-3);
        assert!((tensor.data()[1] - (-2.0)).abs() < 1e-3);
    }

    #[test]
    fn test_multiple_tensors() {
        let w_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_data: Vec<u8> = [0.5f32].iter().flat_map(|v| v.to_le_bytes()).collect();

        let file_bytes = build_safetensors(
            &[
                ("weight", Dtype::F32, &[2], &w_data),
                ("bias", Dtype::F32, &[1], &b_data),
            ],
            None,
        );
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();
        assert_eq!(st.tensors.len(), 2);

        let w = st.read_tensor(&mut cursor, "weight").unwrap();
        assert_eq!(w.data(), &[1.0, 2.0]);

        let b = st.read_tensor(&mut cursor, "bias").unwrap();
        assert_eq!(b.data(), &[0.5]);
    }

    #[test]
    fn test_tensor_not_found() {
        let file_bytes = build_safetensors(&[], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        let err = st.read_tensor(&mut cursor, "nope").unwrap_err();
        assert!(matches!(err, SafeTensorsError::TensorNotFound(_)));
    }

    #[test]
    fn test_header_too_large() {
        // Write a header size that exceeds the limit.
        let huge_size: u64 = 512 * 1024 * 1024;
        let mut buf = Vec::new();
        buf.extend_from_slice(&huge_size.to_le_bytes());
        // Don't need actual data; parse should fail early.
        buf.extend_from_slice(&[0u8; 64]);

        let mut cursor = Cursor::new(buf);
        let err = SafeTensorsFile::parse_header(&mut cursor).unwrap_err();
        assert!(matches!(err, SafeTensorsError::HeaderTooLarge(_, _)));
    }

    #[test]
    fn test_tensor_names() {
        let data: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let file_bytes = build_safetensors(
            &[
                ("a", Dtype::F32, &[1], &data),
                ("b", Dtype::F32, &[1], &data),
            ],
            None,
        );
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        let mut names: Vec<&str> = st.tensor_names().collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_f16_conversion_roundtrip() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_read_tensor_data_raw() {
        let values: Vec<f32> = vec![42.0, 99.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let file_bytes = build_safetensors(&[("t", Dtype::F32, &[2], &raw)], None);
        let mut cursor = Cursor::new(file_bytes);
        let st = SafeTensorsFile::parse_header(&mut cursor).unwrap();

        let data = st.read_tensor_data(&mut cursor, "t").unwrap();
        assert_eq!(data.len(), 8);
        assert_eq!(data, raw);
    }

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::BF16.element_size(), 2);
    }

    #[test]
    fn test_dtype_to_quant_format() {
        use crate::QuantFormat;
        assert_eq!(Dtype::F32.to_quant_format(), QuantFormat::F32);
        assert_eq!(Dtype::F16.to_quant_format(), QuantFormat::F16);
        assert_eq!(Dtype::BF16.to_quant_format(), QuantFormat::F16);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", Dtype::F32), "F32");
        assert_eq!(format!("{}", Dtype::F16), "F16");
        assert_eq!(format!("{}", Dtype::BF16), "BF16");
    }

    #[test]
    fn test_safe_tensor_info_numel() {
        let make_info = |shape: Vec<usize>| SafeTensorInfo {
            name: "x".into(),
            dtype: Dtype::F32,
            shape,
            start: 0,
            end: 0,
        };
        // Scalar (empty shape) → numel=1 (max(product,1))
        assert_eq!(make_info(vec![]).numel(), 1);
        // 1D
        assert_eq!(make_info(vec![5]).numel(), 5);
        // 2D
        assert_eq!(make_info(vec![2, 3]).numel(), 6);
        // 3D
        assert_eq!(make_info(vec![2, 3, 4]).numel(), 24);
    }

    #[test]
    fn test_safe_tensor_info_byte_len() {
        let info = SafeTensorInfo {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![2, 3],
            start: 10,
            end: 34, // 24 bytes = 2*3*4
        };
        assert_eq!(info.byte_len(), 24);
    }
}

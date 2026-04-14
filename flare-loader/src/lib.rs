//! Model weight loading for Flare LLM.
//!
//! Supports GGUF (primary) and SafeTensors formats with on-the-fly
//! dequantization from Q4_0, Q8_0, F16, and F32 formats.
//!
//! # Example
//!
//! ```rust,ignore
//! use flare_loader::gguf::GgufFile;
//! use flare_loader::weights::load_model_weights;
//!
//! let mut reader = BufReader::new(File::open("model.gguf")?);
//! let gguf = GgufFile::parse_header(&mut reader)?;
//! let config = gguf.to_model_config()?;
//! let weights = load_model_weights(&gguf, &mut reader)?;
//! ```

pub mod gguf;
pub mod lora;
pub mod progressive;
pub mod quantize;
pub mod safetensors;
pub mod tokenizer;
pub mod weights;

pub use gguf::{GgufError, GgufFile};
pub use lora::load_lora_from_safetensors;
pub use quantize::QuantFormat;
pub use safetensors::{SafeTensorsError, SafeTensorsFile};
pub use weights::{
    infer_model_config_from_safetensors, load_model_weights, load_model_weights_from_safetensors,
    load_model_weights_with_progress,
};

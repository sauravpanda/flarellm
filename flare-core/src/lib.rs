//! Core inference engine for Flare LLM.
//!
//! This crate contains everything needed to run LLM inference once model
//! weights are loaded: tensor operations, the Llama transformer forward pass,
//! KV cache, sampling strategies, tokenization, and the generation loop.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use flare_core::model::Model;
//! use flare_core::generate::Generator;
//! use flare_core::sampling::SamplingParams;
//!
//! let mut model = Model::new(config, weights);
//! let mut gen = Generator::new(&mut model, SamplingParams::default());
//! let tokens = gen.generate(&prompt, 128, None, || 0.5, |tok, _| true);
//! ```

pub mod chat;
pub mod config;
pub mod error;
pub mod generate;
pub mod kv_cache;
pub mod model;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;

pub use config::ModelConfig;
pub use error::{FlareError, ModelError};
pub use generate::Generator;
pub use tensor::Tensor;

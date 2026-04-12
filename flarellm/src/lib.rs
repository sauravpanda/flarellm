//! # flarellm
//!
//! A WASM-first LLM inference engine with WebGPU acceleration.
//!
//! Run large language models directly in the browser with zero server costs.
//! Built in pure Rust, compiles to both native and WebAssembly.
//!
//! See the [repository](https://github.com/sauravpanda/flarellm) for examples
//! and architecture documentation.
//!
//! ## Crate organization
//!
//! - [`core`] — model, tensor, sampling, generation (always available)
//! - [`loader`] — GGUF / SafeTensors loading (enabled by `loader` feature, default)
//! - `gpu` — WebGPU compute backend (enabled by `gpu` feature)
//!
//! ## Quick start
//!
//! ```no_run
//! use flarellm::core::generate::Generator;
//! use flarellm::core::sampling::SamplingParams;
//! use flarellm::loader::gguf::GgufFile;
//! use flarellm::loader::weights::load_model_weights;
//! use flarellm::core::model::Model;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut reader = BufReader::new(File::open("model.gguf")?);
//! let gguf = GgufFile::parse_header(&mut reader)?;
//! let config = gguf.to_model_config()?;
//! let weights = load_model_weights(&gguf, &mut reader)?;
//! let mut model = Model::new(config, weights);
//!
//! let mut gen = Generator::new(&mut model, SamplingParams::default());
//! let tokens = gen.generate(&[1, 2, 3], 50, None, || 0.5, |_, _| true);
//! # Ok(())
//! # }
//! ```

pub use flare_core as core;

#[cfg(feature = "loader")]
pub use flare_loader as loader;

#[cfg(feature = "gpu")]
pub use flare_gpu as gpu;

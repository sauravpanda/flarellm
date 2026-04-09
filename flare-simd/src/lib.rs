//! WASM SIMD128 CPU fallback backend for Flare LLM.
//!
//! Used when WebGPU is not available. Implements the same [`flare_core::model::ComputeBackend`]
//! trait as `flare-gpu`, providing tiled matrix multiply, RMSNorm, RoPE,
//! softmax, and SiLU activation on the CPU.

pub mod backend;
pub mod matmul;

pub use backend::SimdBackend;

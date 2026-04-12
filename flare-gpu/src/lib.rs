//! WebGPU/wgpu compute backend for Flare LLM.
//!
//! Provides GPU-accelerated inference using `wgpu`, which compiles to
//! native Vulkan/Metal/DX12 on desktop and WebGPU in the browser.
//!
//! WGSL compute shaders live in the `shaders/` directory:
//! `matmul.wgsl`, `rmsnorm.wgsl`, `rope.wgsl`, `silu_mul.wgsl`, `softmax.wgsl`.

pub mod backend;
pub mod buffers;
pub mod pipeline;

pub use backend::WebGpuBackend;

use flare_core::model::ComputeBackend;
use flare_core::tensor::Tensor;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(String),
    #[error("shader compilation error: {0}")]
    ShaderError(String),
    #[error("buffer operation error: {0}")]
    BufferError(String),
}

/// WebGPU/wgpu compute backend.
///
/// On native, this uses Vulkan/Metal/DX12 via wgpu.
/// In the browser (WASM), this uses WebGPU via wgpu's web backend.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WebGpuBackend {
    /// Create a new GPU backend. This is async because adapter/device
    /// request is async on both native and web.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flare-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl ComputeBackend for WebGpuBackend {
    fn matmul(&self, _a: &Tensor, _b: &Tensor, _output: &mut Tensor) {
        // TODO: dispatch matmul.wgsl compute shader
        todo!("GPU matmul not yet implemented")
    }

    fn rmsnorm(&self, _input: &Tensor, _weight: &Tensor, _eps: f32, _output: &mut Tensor) {
        todo!("GPU rmsnorm not yet implemented")
    }

    fn rope(&self, _q: &mut Tensor, _k: &mut Tensor, _pos: usize, _head_dim: usize, _theta: f32) {
        todo!("GPU RoPE not yet implemented")
    }

    fn softmax(&self, _input: &mut Tensor) {
        todo!("GPU softmax not yet implemented")
    }

    fn silu_mul(&self, _gate: &Tensor, _up: &Tensor, _output: &mut Tensor) {
        todo!("GPU SiLU*mul not yet implemented")
    }
}

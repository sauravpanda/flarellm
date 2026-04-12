use flare_core::model::ComputeBackend;
use flare_core::tensor::Tensor;
use thiserror::Error;

use crate::buffers::{self, BufferPool};
use crate::pipeline::{CachedPipeline, PipelineCache};

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

const MATMUL_SHADER: &str = include_str!("../shaders/matmul.wgsl");
const SILU_MUL_SHADER: &str = include_str!("../shaders/silu_mul.wgsl");

/// WebGPU/wgpu compute backend.
///
/// On native, this uses Vulkan/Metal/DX12 via wgpu.
/// In the browser (WASM), this uses WebGPU via wgpu's web backend.
///
/// Pipelines are cached on first use to avoid per-call shader compilation.
/// Buffers are pooled so that repeated calls with the same tensor sizes reuse
/// already-allocated GPU memory instead of allocating and freeing each time.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cache: PipelineCache,
    /// Buffer pool: eliminates ~1–2 ms per-allocation overhead on repeated calls.
    pool: BufferPool,
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

        Ok(Self {
            device,
            queue,
            cache: PipelineCache::new(),
            pool: BufferPool::new(),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Standard 4-binding layout: 2 read-only storage, 1 read-write storage, 1 uniform.
    fn standard_layout() -> [wgpu::BindGroupLayoutEntry; 4] {
        [
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_rw_entry(2),
            uniform_entry(3),
        ]
    }

    /// Run a compute shader with the given bind group and dispatch dimensions,
    /// then read back the output buffer to CPU.
    ///
    /// The staging buffer used for readback is pooled; it is returned to the
    /// pool after the GPU results are copied to the returned `Vec<f32>`.
    fn dispatch_and_readback(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dispatch: [u32; 3],
        output_buf: &wgpu::Buffer,
        output_size: u64,
    ) -> Vec<f32> {
        let staging = self.pool.get_staging(&self.device, output_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
        }

        encoder.copy_buffer_to_buffer(output_buf, 0, &staging, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Map and read back
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .expect("GPU readback channel closed")
            .expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // Return staging buffer to pool for the next call
        self.pool.return_staging(staging);

        result
    }

    /// Build a 4-entry bind group from the standard layout.
    fn make_bind_group(
        &self,
        cached: &CachedPipeline,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cached.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }
}

impl ComputeBackend for WebGpuBackend {
    fn matvec(&self, mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let m = rows as u32;
        let k = cols as u32;
        let n = 1u32;

        // Use pooled buffers to avoid per-call GPU allocation overhead.
        let a_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(mat));
        let b_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(vec));
        let output_size = m as u64 * 4;
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [m, n, k];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "matvec",
            MATMUL_SHADER,
            "matmul",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &a_buf, &b_buf, &out_buf, &params_buf);
                let tile = 16u32;
                let dispatch_x = m.div_ceil(tile);
                let dispatch_y = n.div_ceil(tile);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, dispatch_y, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        // Return buffers to pool — GPU work is synchronously polled in
        // dispatch_and_readback, so they are safe to reuse immediately.
        self.pool.return_storage(a_buf);
        self.pool.return_storage(b_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    fn silu_mul_vec(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        let size = gate.len() as u32;

        let gate_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(gate));
        let up_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(up));
        let output_size = size as u64 * 4;
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [size];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "silu_mul",
            SILU_MUL_SHADER,
            "silu_mul",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &gate_buf, &up_buf, &out_buf, &params_buf);
                let workgroup_size = 256u32;
                let dispatch_x = size.div_ceil(workgroup_size);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(gate_buf);
        self.pool.return_storage(up_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, output: &mut Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert!(a_shape.len() == 2 && b_shape.len() == 2);
        let m = a_shape[0] as u32;
        let k = a_shape[1] as u32;
        let n = b_shape[1] as u32;

        let a_buf = buffers::create_storage_buffer(
            &self.device,
            "matmul_a",
            bytemuck::cast_slice(a.data()),
        );
        let b_buf = buffers::create_storage_buffer(
            &self.device,
            "matmul_b",
            bytemuck::cast_slice(b.data()),
        );
        let output_size = (m * n) as u64 * 4;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_out"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params: [u32; 3] = [m, n, k];
        let params_buf = buffers::create_uniform_buffer(
            &self.device,
            "matmul_params",
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "matmul",
            MATMUL_SHADER,
            "matmul",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &a_buf, &b_buf, &out_buf, &params_buf);
                let tile = 16u32;
                let dispatch_x = m.div_ceil(tile);
                let dispatch_y = n.div_ceil(tile);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, dispatch_y, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        output.data_mut()[..result.len()].copy_from_slice(&result);
    }

    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32, output: &mut Tensor) {
        // CPU fallback for now — RMSNorm is memory-bound and fast on CPU
        let data = input.data();
        let w = weight.data();
        let out = output.data_mut();
        let dim = w.len();
        let num_rows = data.len() / dim;
        for row in 0..num_rows {
            let offset = row * dim;
            let row_data = &data[offset..offset + dim];
            let sum_sq: f32 = row_data.iter().map(|x| x * x).sum();
            let rms = (sum_sq / dim as f32 + eps).sqrt();
            for i in 0..dim {
                out[offset + i] = (row_data[i] / rms) * w[i];
            }
        }
    }

    fn rope(&self, q: &mut Tensor, k: &mut Tensor, pos: usize, head_dim: usize, theta: f32) {
        // CPU fallback — RoPE is a lightweight per-element operation
        let half = head_dim / 2;
        let num_q_heads = q.numel() / head_dim;
        for h in 0..num_q_heads {
            let offset = h * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();
                let q_data = q.data_mut();
                let q0 = q_data[offset + i];
                let q1 = q_data[offset + i + half];
                q_data[offset + i] = q0 * cos_val - q1 * sin_val;
                q_data[offset + i + half] = q0 * sin_val + q1 * cos_val;
            }
        }
        let num_k_heads = k.numel() / head_dim;
        for h in 0..num_k_heads {
            let offset = h * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();
                let k_data = k.data_mut();
                let k0 = k_data[offset + i];
                let k1 = k_data[offset + i + half];
                k_data[offset + i] = k0 * cos_val - k1 * sin_val;
                k_data[offset + i + half] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    fn softmax(&self, input: &mut Tensor) {
        // CPU fallback — softmax over a single vector is fast
        let data = input.data_mut();
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in data.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in data.iter_mut() {
            *v /= sum;
        }
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor, output: &mut Tensor) {
        let size = gate.numel() as u32;

        let gate_buf = buffers::create_storage_buffer(
            &self.device,
            "silu_gate",
            bytemuck::cast_slice(gate.data()),
        );
        let up_buf = buffers::create_storage_buffer(
            &self.device,
            "silu_up",
            bytemuck::cast_slice(up.data()),
        );
        let output_size = size as u64 * 4;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu_out"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params: [u32; 1] = [size];
        let params_buf = buffers::create_uniform_buffer(
            &self.device,
            "silu_params",
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "silu_mul",
            SILU_MUL_SHADER,
            "silu_mul",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &gate_buf, &up_buf, &out_buf, &params_buf);
                let workgroup_size = 256u32;
                let dispatch_x = size.div_ceil(workgroup_size);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        output.data_mut()[..result.len()].copy_from_slice(&result);
    }
}

// WASM is single-threaded; wgpu's JS-backed types don't impl Send/Sync but it's
// safe to assert them here because no other threads exist.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for WebGpuBackend {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WebGpuBackend {}

// Helper functions for bind group layout entries
fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

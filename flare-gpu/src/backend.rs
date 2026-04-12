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
const SOFTMAX_SHADER: &str = include_str!("../shaders/softmax.wgsl");
const ATTENTION_SHADER: &str = include_str!("../shaders/attention.wgsl");
const DEQUANT_Q4_1_SHADER: &str = include_str!("../shaders/dequant_q4_1.wgsl");
const DEQUANT_Q4K_SHADER: &str = include_str!("../shaders/dequant_q4k.wgsl");

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

    /// 3-binding layout: 1 read-only storage, 1 read-write storage, 1 uniform.
    /// Used by kernels that take a single input (e.g. softmax).
    fn single_input_layout() -> [wgpu::BindGroupLayoutEntry; 3] {
        [storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)]
    }

    /// Build a 3-entry bind group for single-input kernels.
    fn make_single_input_bind_group(
        &self,
        cached: &CachedPipeline,
        input: &wgpu::Buffer,
        out: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cached.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// 5-binding layout for the attention shader:
    /// q (read), k_cache (read), v_cache (read), output (read-write), params (uniform).
    fn attention_layout() -> [wgpu::BindGroupLayoutEntry; 5] {
        [
            storage_ro_entry(0),
            storage_ro_entry(1),
            storage_ro_entry(2),
            storage_rw_entry(3),
            uniform_entry(4),
        ]
    }

    /// Build a 5-entry bind group for the attention shader.
    #[allow(clippy::too_many_arguments)]
    fn make_attention_bind_group(
        &self,
        cached: &CachedPipeline,
        q: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cached.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_cache.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_cache.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Dequantize Q4_1 blocks on GPU.
    ///
    /// `raw_bytes` is the packed GGUF tensor data (num_blocks × 20 bytes).
    /// Returns a flat `Vec<f32>` of num_blocks × 32 dequantized weights.
    pub fn dequant_q4_1(&self, raw_bytes: &[u8], num_blocks: usize) -> Vec<f32> {
        let output_size = (num_blocks * 32) as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [num_blocks as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_q4_1",
            DEQUANT_Q4_1_SHADER,
            "dequant_q4_1",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_single_input_bind_group(cached, &raw_buf, &out_buf, &params_buf);
                let dispatch_x = (num_blocks as u32).div_ceil(64);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Dequantize Q4_K blocks on GPU.
    ///
    /// `raw_bytes` is the packed GGUF tensor data (num_blocks × 144 bytes).
    /// Returns a flat `Vec<f32>` of num_blocks × 256 dequantized weights.
    pub fn dequant_q4k(&self, raw_bytes: &[u8], num_blocks: usize) -> Vec<f32> {
        let output_size = (num_blocks * 256) as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [num_blocks as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_q4k",
            DEQUANT_Q4K_SHADER,
            "dequant_q4k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_single_input_bind_group(cached, &raw_buf, &out_buf, &params_buf);
                let dispatch_x = (num_blocks as u32).div_ceil(64);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

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
        let size = input.numel() as u32;

        let in_buf = self.pool.get_storage(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(input.data()),
        );
        let output_size = size as u64 * 4;
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [size];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "softmax",
            SOFTMAX_SHADER,
            "softmax",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_single_input_bind_group(cached, &in_buf, &out_buf, &params_buf);
                // The shader uses a single workgroup — all work is done cooperatively
                // via workgroup shared memory (thread 0 computes max and sum, then all
                // threads normalise in parallel).
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [1, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        input.data_mut()[..result.len()].copy_from_slice(&result);

        self.pool.return_storage(in_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);
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

    #[allow(clippy::too_many_arguments)]
    fn grouped_query_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        attn_softcap: f32,
    ) -> Vec<f32> {
        // Gemma-2 soft-cap is not implemented in the WGSL shader; fall back to CPU.
        if attn_softcap != 0.0 {
            return flare_core::model::cpu_grouped_query_attention(
                q,
                k_cache,
                v_cache,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                attn_softcap,
            );
        }

        let heads_per_kv = num_heads / num_kv_heads;
        let kv_stride = num_kv_heads * head_dim;
        let mut output = vec![0.0f32; num_heads * head_dim];
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let layout_entries = Self::attention_layout();

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Extract contiguous [seq_len, head_dim] slice for this KV head.
            let mut k_head: Vec<f32> = Vec::with_capacity(seq_len * head_dim);
            let mut v_head: Vec<f32> = Vec::with_capacity(seq_len * head_dim);
            for t in 0..seq_len {
                let offset = t * kv_stride + kv_head * head_dim;
                k_head.extend_from_slice(&k_cache[offset..offset + head_dim]);
                v_head.extend_from_slice(&v_cache[offset..offset + head_dim]);
            }

            let output_size = head_dim as u64 * 4;

            let q_buf =
                self.pool
                    .get_storage(&self.device, &self.queue, bytemuck::cast_slice(q_head));
            let k_buf =
                self.pool
                    .get_storage(&self.device, &self.queue, bytemuck::cast_slice(&k_head));
            let v_buf =
                self.pool
                    .get_storage(&self.device, &self.queue, bytemuck::cast_slice(&v_head));
            let out_buf = self.pool.get_output(&self.device, output_size);

            // Params: seq_len, head_dim, scale
            let params_data: [u32; 3] = [
                seq_len as u32,
                head_dim as u32,
                scale.to_bits(), // pass f32 as u32 raw bits; shader reads as f32
            ];
            let params_buf = self.pool.get_uniform(
                &self.device,
                &self.queue,
                bytemuck::cast_slice(&params_data),
            );

            let head_output = self.cache.with_pipeline(
                &self.device,
                "attention_scores",
                ATTENTION_SHADER,
                "attention_scores",
                &layout_entries,
                |cached| {
                    let bind_group = self.make_attention_bind_group(
                        cached,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        &out_buf,
                        &params_buf,
                    );
                    self.dispatch_and_readback(
                        &cached.pipeline,
                        &bind_group,
                        [1, 1, 1],
                        &out_buf,
                        output_size,
                    )
                },
            );

            output[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_output);

            self.pool.return_storage(q_buf);
            self.pool.return_storage(k_buf);
            self.pool.return_storage(v_buf);
            self.pool.return_output(out_buf);
            self.pool.return_uniform(params_buf);
        }

        output
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

#[cfg(test)]
mod tests {
    use super::*;
    use flare_core::tensor::Tensor;

    /// CPU reference softmax — used as the ground truth in comparison tests.
    fn cpu_softmax(data: &[f32]) -> Vec<f32> {
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut out: Vec<f32> = data.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = out.iter().sum();
        for v in out.iter_mut() {
            *v /= sum;
        }
        out
    }

    /// Compare GPU softmax against the CPU reference.
    ///
    /// Marked `#[ignore]` because it requires a GPU adapter (not available in CI).
    /// Run manually with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_softmax_matches_cpu_reference() {
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let expected = cpu_softmax(&data);

        let mut tensor = Tensor::from_vec(data, &[8]).unwrap();
        backend.softmax(&mut tensor);

        for (i, (&got, &exp)) in tensor.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "softmax mismatch at [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Verify GPU softmax output sums to 1.0.
    #[test]
    #[ignore]
    fn test_softmax_sums_to_one() {
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");

        let data: Vec<f32> = vec![0.5, -1.2, 3.3, 2.1, -0.4, 1.6, 0.0, 4.2];
        let mut tensor = Tensor::from_vec(data, &[8]).unwrap();
        backend.softmax(&mut tensor);

        let sum: f32 = tensor.data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax output sum should be 1.0, got {sum}"
        );
    }

    /// Verify GPU Q4_1 dequantization against a known reference.
    ///
    /// One block: d=1.0 (f16 0x3C00), m=0.0 (f16 0x0000), all qs bytes = 0x12
    /// → 16 pairs of (lo=2, hi=1) → output = [2, 1, 2, 1, …] (32 floats).
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q4_1_matches_cpu() {
        let mut raw = vec![0u8; 20];
        // d = 1.0 as f16 LE: 0x3C00 → bytes [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // m = 0.0 as f16 LE: 0x0000 → bytes [0x00, 0x00] (already zero)
        // qs[16]: lo nibble = 2, hi nibble = 1 → byte = 0x12
        for b in raw[4..20].iter_mut() {
            *b = 0x12;
        }

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q4_1(&raw, 1);

        assert_eq!(result.len(), 32, "expected 32 weights");
        for i in 0..16 {
            assert!(
                (result[i * 2] - 2.0).abs() < 1e-3,
                "q4_1 lo mismatch at [{}]: got {}, expected 2.0",
                i * 2,
                result[i * 2]
            );
            assert!(
                (result[i * 2 + 1] - 1.0).abs() < 1e-3,
                "q4_1 hi mismatch at [{}]: got {}, expected 1.0",
                i * 2 + 1,
                result[i * 2 + 1]
            );
        }
    }

    /// Verify GPU Q4_K dequantization against a known reference.
    ///
    /// One block: d=1.0, dmin=0.0, all sc[*]=1, mn[*]=0, all qs bytes = 0x12
    /// → output[0..128]=2.0, output[128..256]=1.0.
    ///
    /// Scale encoding: scales_raw[0..4] = 0x41 sets sc[0..4]=1 and sc[4..8]=1
    /// (upper 2 bits of 0x41 are 01, so (0x41>>6)=1; scales_raw[8..12]=0 so
    /// (0 & 0x0F)<<2=0; sc[i+4] = 1|0 = 1). mn[*]=0 since scales_raw[4..8]=0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q4k_matches_cpu() {
        let mut raw = vec![0u8; 144];
        // d = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // dmin = 0.0 (already zero)
        // scales_raw[0..4] = 0x41: sc[i]=1, bits[7:6]=01 → sc[i+4]=1
        for b in raw[4..8].iter_mut() {
            *b = 0x41;
        }
        // scales_raw[4..8] = 0x00: mn[*]=0 (already zero)
        // scales_raw[8..12] = 0x00 (already zero): lower nibble=0 → sc[i+4] unchanged
        // qs[128]: lo=2, hi=1 → byte = 0x12
        for b in raw[16..144].iter_mut() {
            *b = 0x12;
        }

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q4k(&raw, 1);

        assert_eq!(result.len(), 256, "expected 256 weights");
        for j in 0..128 {
            assert!(
                (result[j] - 2.0).abs() < 1e-3,
                "q4k lo mismatch at [{}]: got {}, expected 2.0",
                j,
                result[j]
            );
            assert!(
                (result[j + 128] - 1.0).abs() < 1e-3,
                "q4k hi mismatch at [{}]: got {}, expected 1.0",
                j + 128,
                result[j + 128]
            );
        }
    }
}

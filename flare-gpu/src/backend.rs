use std::sync::Mutex;

use flare_core::model::{ComputeBackend, RawWeight, WeightFormat};
use flare_core::tensor::Tensor;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::buffers::{self, BufferPool};
use crate::kv_cache::GpuKvCache;
use crate::pipeline::{try_create_wgpu_cache, CachedPipeline, PipelineCache};

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
const DEQUANT_MATVEC_BF16_SHADER: &str = include_str!("../shaders/dequant_matvec_bf16.wgsl");
const DEQUANT_MATVEC_F16_SHADER: &str = include_str!("../shaders/dequant_matvec_f16.wgsl");
const DEQUANT_MATVEC_Q2K_SHADER: &str = include_str!("../shaders/dequant_matvec_q2k.wgsl");
const DEQUANT_MATVEC_Q3K_SHADER: &str = include_str!("../shaders/dequant_matvec_q3k.wgsl");
const DEQUANT_MATVEC_Q4_0_SHADER: &str = include_str!("../shaders/dequant_matvec_q4_0.wgsl");
const DEQUANT_MATVEC_Q4_1_SHADER: &str = include_str!("../shaders/dequant_matvec_q4_1.wgsl");
const DEQUANT_MATVEC_Q5_0_SHADER: &str = include_str!("../shaders/dequant_matvec_q5_0.wgsl");
const DEQUANT_MATVEC_Q5_1_SHADER: &str = include_str!("../shaders/dequant_matvec_q5_1.wgsl");
const DEQUANT_MATVEC_Q8_0_SHADER: &str = include_str!("../shaders/dequant_matvec_q8_0.wgsl");
const DEQUANT_MATVEC_Q8_1_SHADER: &str = include_str!("../shaders/dequant_matvec_q8_1.wgsl");
const DEQUANT_MATVEC_Q4K_SHADER: &str = include_str!("../shaders/dequant_matvec_q4k.wgsl");
const DEQUANT_MATVEC_Q5K_SHADER: &str = include_str!("../shaders/dequant_matvec_q5k.wgsl");
const DEQUANT_MATVEC_Q6K_SHADER: &str = include_str!("../shaders/dequant_matvec_q6k.wgsl");
const DEQUANT_MATVEC_IQ4NL_SHADER: &str = include_str!("../shaders/dequant_matvec_iq4nl.wgsl");
const DEQUANT_MATVEC_IQ4XS_SHADER: &str = include_str!("../shaders/dequant_matvec_iq4xs.wgsl");
const DEQUANT_MATVEC_IQ3S_SHADER: &str = include_str!("../shaders/dequant_matvec_iq3s.wgsl");
const DEQUANT_MATVEC_IQ2XXS_SHADER: &str = include_str!("../shaders/dequant_matvec_iq2xxs.wgsl");
const DEQUANT_MATVEC_IQ2XS_SHADER: &str = include_str!("../shaders/dequant_matvec_iq2xs.wgsl");
const DEQUANT_MATVEC_IQ3XXS_SHADER: &str = include_str!("../shaders/dequant_matvec_iq3xxs.wgsl");
const DEQUANT_MATVEC_IQ2S_SHADER: &str = include_str!("../shaders/dequant_matvec_iq2s.wgsl");
const DEQUANT_MATVEC_IQ1S_SHADER: &str = include_str!("../shaders/dequant_matvec_iq1s.wgsl");
const DEQUANT_Q5K_SHADER: &str = include_str!("../shaders/dequant_q5k.wgsl");
const DEQUANT_Q6K_SHADER: &str = include_str!("../shaders/dequant_q6k.wgsl");
const PREFILL_ATTENTION_SHADER: &str = include_str!("../shaders/prefill_attention.wgsl");
const DEQUANT_Q3K_SHADER: &str = include_str!("../shaders/dequant_q3k.wgsl");
const BATCHED_MATVEC_SHADER: &str = include_str!("../shaders/batched_matvec.wgsl");
const BATCHED_RMSNORM_SHADER: &str = include_str!("../shaders/batched_rmsnorm.wgsl");
const BATCHED_ROPE_SHADER: &str = include_str!("../shaders/batched_rope.wgsl");
const ADD_RESIDUAL_SHADER: &str = include_str!("../shaders/add_residual.wgsl");

/// GPU-resident weight buffer for a single quantized weight matrix.
///
/// Holds the raw bytes (packed GGUF data) in a persistent GPU storage buffer,
/// uploaded once at model load time.  Used by `forward_single_token_gpu` to
/// avoid re-uploading weight data on every token.
struct GpuWeightBuffer {
    buf: wgpu::Buffer,
    format: WeightFormat,
    num_rows: usize,
    blocks_per_row: usize,
}

/// GPU-resident weights for a single transformer layer.
struct GpuLayerWeights {
    wq: GpuWeightBuffer,
    wk: GpuWeightBuffer,
    wv: GpuWeightBuffer,
    wo: GpuWeightBuffer,
    w_gate: GpuWeightBuffer,
    w_up: GpuWeightBuffer,
    w_down: GpuWeightBuffer,
    /// RMSNorm weights for the attention sub-block.
    attn_norm: wgpu::Buffer,
    /// RMSNorm weights for the FFN sub-block.
    ffn_norm: wgpu::Buffer,
    /// Optional post-attention RMSNorm (Gemma 2).
    post_attn_norm: Option<wgpu::Buffer>,
    /// Optional post-FFN RMSNorm (Gemma 2).
    post_ffn_norm: Option<wgpu::Buffer>,
}

/// A single shard of a large f32 matrix, stored row-major on the GPU.
///
/// Used to split buffers that would exceed wgpu's 256 MB per-buffer limit
/// (e.g. output_weight for models with large vocabularies).
struct GpuF32Shard {
    buf: wgpu::Buffer,
    /// Number of complete rows in this shard.
    num_rows: usize,
    /// Row offset within the full (unsplit) matrix.
    row_offset: usize,
}

/// All model weights uploaded to GPU once.
#[allow(dead_code)]
struct GpuResidentWeights {
    layers: Vec<GpuLayerWeights>,
    output_norm: wgpu::Buffer,
    /// Output projection (lm_head): kept as f32 on GPU.
    /// Sharded into multiple buffers to stay under the 256 MB device limit.
    output_weight_shards: Vec<GpuF32Shard>,
    /// Number of columns (== dim) in the output weight matrix.
    output_weight_cols: usize,
    /// Token embedding table: kept as f32 on GPU for embedding lookup.
    /// Sharded for the same reason.
    token_embedding_shards: Vec<GpuF32Shard>,
}

/// WebGPU/wgpu compute backend.
///
/// On native, this uses Vulkan/Metal/DX12 via wgpu.
/// In the browser (WASM), this uses WebGPU via wgpu's web backend.
///
/// Pipelines are cached on first use to avoid per-call shader compilation.
/// Buffers are pooled so that repeated calls with the same tensor sizes reuse
/// already-allocated GPU memory instead of allocating and freeing each time.
///
/// When `init_gpu_kv_cache` is called (automatically from `Model::set_backend`),
/// the backend allocates `GpuKvCache` — GPU-resident ring buffers for K/V.
/// Subsequent attention calls use those buffers directly, eliminating the
/// CPU→GPU re-upload of the entire KV cache on every generated token.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cache: PipelineCache,
    /// Buffer pool: eliminates ~1–2 ms per-allocation overhead on repeated calls.
    pool: BufferPool,
    /// GPU-resident KV cache.  `None` until `init_gpu_kv_cache` is called.
    /// `Mutex` provides interior mutability so trait methods can take `&self`.
    gpu_kv_cache: Mutex<Option<GpuKvCache>>,
    /// GPU-resident model weights.  `None` until `upload_weights_to_gpu` is called.
    gpu_weights: Mutex<Option<GpuResidentWeights>>,
}

impl WebGpuBackend {
    /// Create a new GPU backend. This is async because adapter/device
    /// request is async on both native and web.
    pub async fn new() -> Result<Self, GpuError> {
        Self::new_impl(None).await
    }

    /// Create a new GPU backend pre-loaded with serialised pipeline cache data.
    ///
    /// On backends that support `Features::PIPELINE_CACHE` (Vulkan), the driver
    /// can reuse compiled GPU machine code from a previous run, avoiding cold-start
    /// shader recompilation.  Pass bytes previously obtained from
    /// [`Self::pipeline_cache_data`].  On unsupported backends the data is silently
    /// ignored and the backend behaves identically to [`Self::new`].
    ///
    /// # Safety of cache data
    ///
    /// The data must originate from the same (or ABI-compatible) device and driver.
    /// `fallback: true` is set so the driver discards mismatched blobs without
    /// causing undefined behaviour.
    pub async fn new_with_cache(data: &[u8]) -> Result<Self, GpuError> {
        Self::new_impl(Some(data)).await
    }

    async fn new_impl(cache_data: Option<&[u8]>) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // Enable PIPELINE_CACHE if the backend supports it (Vulkan).
        // On WebGPU / Metal / DX12 the feature flag is absent; requesting an
        // absent feature is an error, so we check first.
        let adapter_features = adapter.features();
        let extra_features = if adapter_features.contains(wgpu::Features::PIPELINE_CACHE) {
            wgpu::Features::PIPELINE_CACHE
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flare-gpu"),
                    required_features: extra_features,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let pipeline_cache = match try_create_wgpu_cache(&device, adapter_features, cache_data) {
            Some(wgpu_cache) => {
                log::debug!("flare-gpu: driver pipeline cache enabled");
                PipelineCache::new_with_wgpu_cache(wgpu_cache)
            }
            None => PipelineCache::new(),
        };

        Ok(Self {
            device,
            queue,
            cache: pipeline_cache,
            pool: BufferPool::new(),
            gpu_kv_cache: Mutex::new(None),
            gpu_weights: Mutex::new(None),
        })
    }

    /// Serialise the driver-managed pipeline cache to bytes.
    ///
    /// Store these bytes and pass them to [`Self::new_with_cache`] on the next
    /// startup to skip shader recompilation on supported backends (Vulkan).
    ///
    /// Returns an empty `Vec` on backends that do not support pipeline caching
    /// or when the driver has not produced any serialisable data yet.
    pub fn get_pipeline_cache_bytes(&self) -> Vec<u8> {
        self.cache.get_data()
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

    /// Dequantize Q5_K blocks on GPU.
    ///
    /// `raw_bytes` is the packed GGUF tensor data (num_blocks × 176 bytes).
    /// Returns a flat `Vec<f32>` of num_blocks × 256 dequantized weights.
    pub fn dequant_q5k(&self, raw_bytes: &[u8], num_blocks: usize) -> Vec<f32> {
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
            "dequant_q5k",
            DEQUANT_Q5K_SHADER,
            "dequant_q5k",
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

    /// Dequantize Q6_K blocks on GPU.
    ///
    /// `raw_bytes` is the packed GGUF tensor data (num_blocks × 210 bytes).
    /// Returns a flat `Vec<f32>` of num_blocks × 256 dequantized weights.
    ///
    /// Q6_K blocks are 210 bytes each (not u32-aligned).  The raw bytes are
    /// padded to the nearest 4-byte multiple before upload so that the wgpu
    /// storage buffer size requirement is satisfied.
    pub fn dequant_q6k(&self, raw_bytes: &[u8], num_blocks: usize) -> Vec<f32> {
        let output_size = (num_blocks * 256) as u64 * 4;

        // Q6_K blocks are 210 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [num_blocks as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_q6k",
            DEQUANT_Q6K_SHADER,
            "dequant_q6k",
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

    /// Dequantize Q3_K blocks on the GPU.
    ///
    /// Runs `dequant_q3k.wgsl` with one thread per block (workgroup 64, dispatch
    /// ceil(num_blocks/64) × 1 × 1).  Returns `num_blocks * 256` f32 values.
    ///
    /// Q3_K blocks are 110 bytes each (not u32-aligned).  The raw bytes are
    /// padded to the nearest 4-byte multiple before upload so that the wgpu
    /// storage buffer size requirement is satisfied.
    pub fn dequant_q3k(&self, raw_bytes: &[u8], num_blocks: usize) -> Vec<f32> {
        let output_size = (num_blocks * 256) as u64 * 4;

        // Q3_K blocks are 110 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 1] = [num_blocks as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_q3k",
            DEQUANT_Q3K_SHADER,
            "dequant_q3k",
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

    /// Fused Q2_K dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q2_K weight data, dequantizes each block on-the-fly, and
    /// accumulates the dot product with `input` in the same kernel.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 84` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 256`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q2k(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q2k",
            DEQUANT_MATVEC_Q2K_SHADER,
            "dequant_matvec_q2k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused F16 dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed F16 (IEEE 754 half-precision) weight data, converts each
    /// value to F32 on-the-fly using WGSL's `unpack2x16float`, and accumulates
    /// the dot products with `input` in the same kernel.
    ///
    /// - `raw_bytes`: F16 weight data — `num_rows × num_cols × 2` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input matrix of length `batch × num_cols`
    /// - `num_blocks_per_row`: equals `num_cols` (weights_per_block = 1 for F16)
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_f16(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // F16 data may not be u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_f16",
            DEQUANT_MATVEC_F16_SHADER,
            "dequant_matvec_f16",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q4_0 dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q4_0 weight data, dequantizes each block on-the-fly, and
    /// accumulates the dot product with `input` in the same kernel.
    ///
    /// Q4_0 blocks are 18 bytes each (not u32-aligned). The raw bytes are
    /// padded to the nearest 4-byte multiple before upload so that the wgpu
    /// storage buffer size requirement is satisfied.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 18` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 32`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q4_0(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // Q4_0 blocks are 18 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q4_0",
            DEQUANT_MATVEC_Q4_0_SHADER,
            "dequant_matvec_q4_0",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ4_NL dequantize + batched matrix-vector multiply.
    ///
    /// IQ4_NL uses a 16-entry neural-network-optimized lookup table instead of
    /// the simple `q - 8` mapping used by Q4_0, giving better quality at the
    /// same 4-bit width.  Block layout is identical to Q4_0: 18 bytes per block
    /// of 32 weights (2 bytes f16 scale + 16 bytes packed nibbles).
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 18` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 32`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq4nl(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // IQ4_NL blocks are 18 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq4nl",
            DEQUANT_MATVEC_IQ4NL_SHADER,
            "dequant_matvec_iq4nl",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ4_XS dequantize + batched matrix-vector multiply.
    ///
    /// IQ4_XS (GGUF type 22) is a 4.25-bit quantization format using the 16-entry
    /// KVALUES_IQ4NL lookup table with sub-group scales.  Each super-block is 136 bytes
    /// (2 bytes f16 scale + 2 bytes scales_h + 4 bytes scales_l + 128 bytes qs)
    /// covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 136` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq4xs(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 136 bytes is u32-aligned — no padding needed.
        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq4xs",
            DEQUANT_MATVEC_IQ4XS_SHADER,
            "dequant_matvec_iq4xs",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ3_S dequantize + batched matrix-vector multiply.
    ///
    /// IQ3_S (GGUF type 26) is a 3.44-bit quantization format using a 512-entry
    /// grid lookup table.  Each super-block is 110 bytes (2 d + 64 qs + 8 qh + 32 signs
    /// + 4 scales) covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 110` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq3s(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 110 bytes is not u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq3s",
            DEQUANT_MATVEC_IQ3S_SHADER,
            "dequant_matvec_iq3s",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ2_XXS dequantize + batched matrix-vector multiply.
    ///
    /// IQ2_XXS (GGUF type 16) is a 2.06-bit quantization format using a 256-entry
    /// grid lookup table.  Each super-block is 66 bytes (2 bytes f16 scale + 64 bytes qs)
    /// covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 66` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq2xxs(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 66 bytes is not u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq2xxs",
            DEQUANT_MATVEC_IQ2XXS_SHADER,
            "dequant_matvec_iq2xxs",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ2_XS dequantize + batched matrix-vector multiply.
    ///
    /// IQ2_XS (GGUF type 17) is a 2.31-bit quantization format using a 512-entry
    /// grid lookup table.  Each super-block is 74 bytes (2 bytes f16 scale + 64 bytes qs
    /// + 8 bytes scales) covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 74` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq2xs(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 74 bytes is not u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq2xs",
            DEQUANT_MATVEC_IQ2XS_SHADER,
            "dequant_matvec_iq2xs",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ3_XXS dequantize + batched matrix-vector multiply.
    ///
    /// IQ3_XXS (GGUF type 18) is a 3.06-bit quantization format using a 256-entry
    /// grid lookup table.  Each super-block is 98 bytes (2 bytes f16 scale + 64 bytes qs
    /// + 32 bytes scales_and_signs) covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 98` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq3xxs(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 98 bytes is not u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq3xxs",
            DEQUANT_MATVEC_IQ3XXS_SHADER,
            "dequant_matvec_iq3xxs",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ2_S dequantize + batched matrix-vector multiply.
    ///
    /// IQ2_S (GGUF type 21) is a 2.5625-bit quantization format using a 1024-entry
    /// u64 grid lookup table. Each super-block is 82 bytes (padded to 84 for GPU)
    /// covering 256 weights.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 82` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_iq2s(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 82 bytes is not u32-aligned — pad to next multiple of 4 (84 bytes).
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq2s",
            DEQUANT_MATVEC_IQ2S_SHADER,
            "dequant_matvec_iq2s",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused IQ1_S dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed IQ1_S weight data (50 bytes/block, padded to 52), dequantizes
    /// using the 2048-entry signed-byte grid with 11-bit indices, and accumulates
    /// the dot product with `input` in the same kernel.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 50` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 256`
    /// - Returns `num_rows × batch` f32 dot products
    pub fn dequant_matvec_iq1s(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // 50 bytes is not u32-aligned — pad to next multiple of 4 (52 bytes).
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_iq1s",
            DEQUANT_MATVEC_IQ1S_SHADER,
            "dequant_matvec_iq1s",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q4_1 dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q4_1 weight data, dequantizes each block on-the-fly, and
    /// accumulates the dot product with `input` in the same kernel — halving
    /// the effective memory bandwidth compared to a separate dequant + matvec.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 20` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 32`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q4_1(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q4_1",
            DEQUANT_MATVEC_Q4_1_SHADER,
            "dequant_matvec_q4_1",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q5_1 dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed Q5_1 weight data, dequantizes each 24-byte block on-the-fly,
    /// and accumulates the dot products with `input` in the same kernel.
    ///
    /// Q5_1 blocks are 24 bytes (u32-aligned — no padding needed).
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 24` bytes
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 32`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_q5_1(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q5_1",
            DEQUANT_MATVEC_Q5_1_SHADER,
            "dequant_matvec_q5_1",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q8_0 dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed Q8_0 weight data, dequantizes each 34-byte block on-the-fly,
    /// and accumulates the dot products with `input` in the same kernel.
    ///
    /// Q8_0 blocks are 34 bytes each (not u32-aligned). The raw bytes are
    /// padded to the nearest 4-byte multiple before upload.
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 34` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 32`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_q8_0(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // Q8_0 blocks are 34 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q8_0",
            DEQUANT_MATVEC_Q8_0_SHADER,
            "dequant_matvec_q8_0",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused BF16 dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed BF16 weight data, converts each value to f32 on-the-fly
    /// (BF16 = upper 16 bits of F32, so conversion is a left-shift by 16),
    /// and accumulates the dot products with `input` in the same kernel.
    ///
    /// - `raw_bytes`: BF16 weight data — `num_rows × num_cols × 2` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input matrix of length `batch × num_cols`
    /// - `num_blocks_per_row`: equals `num_cols` (weights_per_block = 1 for BF16)
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_bf16(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // BF16 data may not be u32-aligned — pad to next multiple of 4.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_bf16",
            DEQUANT_MATVEC_BF16_SHADER,
            "dequant_matvec_bf16",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q8_1 dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q8_1 weight data, dequantizes each block on-the-fly, and
    /// accumulates the dot product with `input` in the same kernel — halving
    /// the effective memory bandwidth compared to a separate dequant + matvec.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 36` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 32`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q8_1(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q8_1",
            DEQUANT_MATVEC_Q8_1_SHADER,
            "dequant_matvec_q8_1",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    ///
    /// Reads packed Q4_K weight data, dequantizes each block on-the-fly, and
    /// accumulates the dot product with `input` in the same kernel — halving
    /// the effective memory bandwidth compared to a separate dequant + matvec.
    ///
    /// - `raw_bytes`: packed GGUF tensor data — `num_rows × num_blocks_per_row × 144` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 256`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q4k(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q4k",
            DEQUANT_MATVEC_Q4K_SHADER,
            "dequant_matvec_q4k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q5_K dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q5_K weight data, dequantizes each 176-byte block on-the-fly,
    /// and accumulates the dot product with `input` in the same kernel.
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 176` bytes
    /// - `input`: f32 input vector of length `num_blocks_per_row × 256`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q5k(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        let raw_buf = self.pool.get_storage(&self.device, &self.queue, raw_bytes);
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q5k",
            DEQUANT_MATVEC_Q5K_SHADER,
            "dequant_matvec_q5k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Batched matrix-vector multiply on the GPU.
    ///
    /// For each batch item `b` in `0..batch`:
    ///   `output[b * out_rows + i] = Σ_j weight[i * in_cols + j] * input[b * in_cols + j]`
    ///
    /// - `weight`: f32 matrix `[out_rows × in_cols]`, row-major — same for all items
    /// - `input`:  f32 batch `[batch × in_cols]`, row-major — one row per batch item
    /// - Returns:  f32 result `[batch × out_rows]`, row-major
    ///
    /// Dispatch: `[out_rows, batch, 1]`, workgroup size 64 with tree reduction.
    pub fn batched_matvec(
        &self,
        weight: &[f32],
        input: &[f32],
        out_rows: usize,
        in_cols: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = (batch * out_rows) as u64 * 4;

        let weight_buf =
            self.pool
                .get_storage(&self.device, &self.queue, bytemuck::cast_slice(weight));
        let input_buf =
            self.pool
                .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [out_rows as u32, in_cols as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "batched_matvec",
            BATCHED_MATVEC_SHADER,
            "batched_matvec",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &weight_buf, &input_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [out_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(weight_buf);
        self.pool.return_storage(input_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Apply RoPE to a batch of token vectors.
    ///
    /// `inp` is laid out as `[seq_len × num_heads × head_dim]`. Token `t` gets
    /// position `start_pos + t`. Returns a new buffer with the same shape.
    ///
    /// Dispatches `seq_len × num_heads × head_dim/2` threads; each thread
    /// handles one rotation pair (element i, i + head_dim/2) for one token.
    pub fn batched_rope(
        &self,
        inp: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        start_pos: usize,
        theta: f32,
    ) -> Vec<f32> {
        let total = seq_len * num_heads * head_dim;
        let output_size = total as u64 * 4;

        let inp_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(inp));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 5] = [
            num_heads as u32,
            head_dim as u32,
            seq_len as u32,
            start_pos as u32,
            theta.to_bits(),
        ];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::single_input_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "batched_rope",
            BATCHED_ROPE_SHADER,
            "batched_rope",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_single_input_bind_group(cached, &inp_buf, &out_buf, &params_buf);
                let half = (head_dim / 2) as u32;
                let dispatch_x = (seq_len as u32 * num_heads as u32 * half).div_ceil(64);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [dispatch_x, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(inp_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q3_K dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed Q3_K weight data, dequantizes each 110-byte block on-the-fly,
    /// and accumulates the dot products with `input` in the same kernel.
    ///
    /// Q3_K blocks are 110 bytes each (not u32-aligned). The raw bytes are
    /// padded to the nearest 4-byte multiple before upload.
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 110` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 256`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_q3k(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // Q3_K blocks are 110 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q3k",
            DEQUANT_MATVEC_Q3K_SHADER,
            "dequant_matvec_q3k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q5_0 dequantize + batched matrix-vector multiply.
    ///
    /// Reads packed Q5_0 weight data, dequantizes each 22-byte block on-the-fly,
    /// and accumulates the dot products with `input` in the same kernel.
    ///
    /// Q5_0 blocks are 22 bytes each (not u32-aligned). The raw bytes are
    /// padded to the nearest 4-byte multiple before upload.
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 22` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input matrix of length `batch × num_blocks_per_row × 32`
    /// - Returns `batch × num_rows` f32 dot products
    pub fn dequant_matvec_q5_0(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // Q5_0 blocks are 22 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q5_0",
            DEQUANT_MATVEC_Q5_0_SHADER,
            "dequant_matvec_q5_0",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Fused Q6_K dequantize + matrix-vector multiply.
    ///
    /// Reads packed Q6_K weight data, dequantizes each 210-byte block on-the-fly,
    /// and accumulates the dot product with `input` in the same kernel.
    ///
    /// - `raw_bytes`: packed GGUF data — `num_rows × num_blocks_per_row × 210` bytes
    ///   (padded to 4-byte multiple by caller if needed)
    /// - `input`: f32 input vector of length `num_blocks_per_row × 256`
    /// - Returns `num_rows` f32 dot products
    pub fn dequant_matvec_q6k(
        &self,
        raw_bytes: &[u8],
        input: &[f32],
        num_rows: usize,
        num_blocks_per_row: usize,
        batch: usize,
    ) -> Vec<f32> {
        let output_size = num_rows as u64 * batch as u64 * 4;

        // Q6_K blocks are 210 bytes — pad to next multiple of 4 for wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let raw_buf = if raw_bytes.len() % 4 == 0 {
            self.pool.get_storage(&self.device, &self.queue, raw_bytes)
        } else {
            let mut padded = raw_bytes.to_vec();
            padded.resize((padded.len() + 3) & !3, 0);
            self.pool.get_storage(&self.device, &self.queue, &padded)
        };
        let vec_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [num_rows as u32, num_blocks_per_row as u32, batch as u32];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "dequant_matvec_q6k",
            DEQUANT_MATVEC_Q6K_SHADER,
            "dequant_matvec_q6k",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &raw_buf, &vec_buf, &out_buf, &params_buf);
                // One workgroup per output row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [num_rows as u32, batch as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(raw_buf);
        self.pool.return_storage(vec_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    /// Apply RMSNorm to every row in `input` using the given `weight` vector.
    ///
    /// - `input`:  f32 batch `[batch × dim]`, row-major
    /// - `weight`: f32 norm weights `[dim]`
    /// - Returns:  f32 result `[batch × dim]`, row-major
    ///
    /// Dispatches one workgroup per batch row; 64 threads compute a parallel
    /// sum-of-squares via tree reduction, then each thread writes its outputs.
    pub fn batched_rmsnorm(
        &self,
        input: &[f32],
        weight: &[f32],
        dim: usize,
        batch: usize,
        eps: f32,
    ) -> Vec<f32> {
        let output_size = (batch * dim) as u64 * 4;

        let inp_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(input));
        let wt_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(weight));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let params: [u32; 3] = [dim as u32, batch as u32, eps.to_bits()];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::standard_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "batched_rmsnorm",
            BATCHED_RMSNORM_SHADER,
            "batched_rmsnorm",
            &layout_entries,
            |cached| {
                let bind_group =
                    self.make_bind_group(cached, &inp_buf, &wt_buf, &out_buf, &params_buf);
                // One workgroup per batch row.
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [batch as u32, 1, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(inp_buf);
        self.pool.return_storage(wt_buf);
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

    /// Batched causal prefill attention for all token positions in a sequence.
    ///
    /// Dispatches one workgroup per `(query_pos, head)` pair so all positions
    /// are processed in parallel.  Position `t` attends causally to keys at
    /// positions `0..=t`.
    ///
    /// - `q`: `[seq_len * num_heads * head_dim]`
    /// - `k`: `[seq_len * num_kv_heads * head_dim]`
    /// - `v`: `[seq_len * num_kv_heads * head_dim]`
    /// - Returns `[seq_len * num_heads * head_dim]`
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_softcap: f32,
    ) -> Vec<f32> {
        let q_dim = num_heads * head_dim;
        let output_size = (seq_len * q_dim) as u64 * 4;

        let q_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(q));
        let k_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(k));
        let v_buf = self
            .pool
            .get_storage(&self.device, &self.queue, bytemuck::cast_slice(v));
        let out_buf = self.pool.get_output(&self.device, output_size);

        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let params: [u32; 6] = [
            seq_len as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            scale.to_bits(),
            attn_softcap.to_bits(),
        ];
        let params_buf =
            self.pool
                .get_uniform(&self.device, &self.queue, bytemuck::cast_slice(&params));

        let layout_entries = Self::attention_layout();
        let result = self.cache.with_pipeline(
            &self.device,
            "prefill_attention",
            PREFILL_ATTENTION_SHADER,
            "prefill_attention",
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
                // Dispatch [seq_len, num_heads, 1]: one workgroup per (position, head).
                self.dispatch_and_readback(
                    &cached.pipeline,
                    &bind_group,
                    [seq_len as u32, num_heads as u32, 1],
                    &out_buf,
                    output_size,
                )
            },
        );

        self.pool.return_storage(q_buf);
        self.pool.return_storage(k_buf);
        self.pool.return_storage(v_buf);
        self.pool.return_output(out_buf);
        self.pool.return_uniform(params_buf);

        result
    }

    // -----------------------------------------------------------------------
    // GPU-resident weight upload
    // -----------------------------------------------------------------------

    /// Upload a single raw weight tensor to a persistent GPU storage buffer.
    fn upload_raw_weight(&self, raw: &RawWeight) -> GpuWeightBuffer {
        // Pad to 4-byte alignment as required by wgpu.
        #[allow(clippy::manual_is_multiple_of)]
        let data = if raw.data.len() % 4 == 0 {
            std::borrow::Cow::Borrowed(&raw.data[..])
        } else {
            let mut padded = raw.data.clone();
            padded.resize((padded.len() + 3) & !3, 0);
            std::borrow::Cow::Owned(padded)
        };

        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_weight"),
            contents: &data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        GpuWeightBuffer {
            buf,
            format: raw.format,
            num_rows: raw.num_rows,
            blocks_per_row: raw.blocks_per_row,
        }
    }

    /// Upload f32 data to a persistent GPU storage buffer.
    fn upload_f32_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_f32"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Maximum bytes per GPU buffer.  wgpu's default `max_buffer_size` is 256 MiB.
    /// We leave a small margin so row-alignment never pushes us over.
    const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024 - 4096; // ~256 MB minus 4 KiB headroom

    /// Upload a row-major f32 matrix, splitting it into multiple GPU buffers
    /// ("shards") as needed so that no single buffer exceeds the wgpu 256 MB
    /// device limit.  Each shard contains a whole number of rows.
    fn upload_f32_matrix_sharded(
        &self,
        data: &[f32],
        num_rows: usize,
        num_cols: usize,
    ) -> Vec<GpuF32Shard> {
        assert_eq!(data.len(), num_rows * num_cols, "data length mismatch");

        let row_bytes = num_cols * std::mem::size_of::<f32>();
        let max_rows_per_shard = Self::MAX_BUFFER_BYTES / row_bytes;
        // Ensure at least one row per shard (a single row should always fit
        // for any realistic model dimension).
        let max_rows_per_shard = max_rows_per_shard.max(1);

        let mut shards = Vec::new();
        let mut row_offset = 0usize;
        while row_offset < num_rows {
            let shard_rows = (num_rows - row_offset).min(max_rows_per_shard);
            let start = row_offset * num_cols;
            let end = start + shard_rows * num_cols;
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_f32_shard"),
                contents: bytemuck::cast_slice(&data[start..end]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            shards.push(GpuF32Shard {
                buf,
                num_rows: shard_rows,
                row_offset,
            });
            row_offset += shard_rows;
        }

        log::debug!(
            "flare-gpu: sharded {}×{} f32 matrix into {} buffer(s)",
            num_rows,
            num_cols,
            shards.len(),
        );
        shards
    }

    /// Upload all model weights to persistent GPU buffers (once).
    ///
    /// After this call, `forward_single_token_gpu` avoids all per-token weight
    /// uploads, eliminating ~90+ CPU-to-GPU transfers per token.
    pub fn upload_weights_to_gpu(
        &self,
        raw_layers: &[flare_core::model::RawLayerWeights],
        layer_norms: &[(
            &[f32],            // attn_norm
            &[f32],            // ffn_norm
            Option<&[f32]>,    // post_attn_norm
            Option<&[f32]>,    // post_ffn_norm
        )],
        output_norm: &[f32],
        output_weight: &[f32],
        token_embedding: &[f32],
    ) {
        let mut layers = Vec::with_capacity(raw_layers.len());
        for (i, raw_layer) in raw_layers.iter().enumerate() {
            let (attn_n, ffn_n, post_attn_n, post_ffn_n) = &layer_norms[i];
            layers.push(GpuLayerWeights {
                wq: self.upload_raw_weight(&raw_layer.wq),
                wk: self.upload_raw_weight(&raw_layer.wk),
                wv: self.upload_raw_weight(&raw_layer.wv),
                wo: self.upload_raw_weight(&raw_layer.wo),
                w_gate: self.upload_raw_weight(&raw_layer.w_gate),
                w_up: self.upload_raw_weight(&raw_layer.w_up),
                w_down: self.upload_raw_weight(&raw_layer.w_down),
                attn_norm: self.upload_f32_buffer(attn_n),
                ffn_norm: self.upload_f32_buffer(ffn_n),
                post_attn_norm: post_attn_n.map(|d| self.upload_f32_buffer(d)),
                post_ffn_norm: post_ffn_n.map(|d| self.upload_f32_buffer(d)),
            });
        }

        // output_weight is a (vocab_size × dim) matrix stored row-major.
        // For large vocabs this exceeds the 256 MB wgpu buffer limit, so we
        // shard it into multiple GPU buffers.
        let dim = if raw_layers.is_empty() {
            // Fallback: infer dim from the norm vector length.
            output_norm.len()
        } else {
            output_norm.len()
        };
        let vocab_size = output_weight.len() / dim;
        let output_weight_shards = self.upload_f32_matrix_sharded(output_weight, vocab_size, dim);

        // token_embedding is also (vocab_size × dim) and can be equally large.
        let emb_rows = token_embedding.len() / dim;
        let token_embedding_shards = self.upload_f32_matrix_sharded(token_embedding, emb_rows, dim);

        let weights = GpuResidentWeights {
            layers,
            output_norm: self.upload_f32_buffer(output_norm),
            output_weight_shards,
            output_weight_cols: dim,
            token_embedding_shards,
        };

        *self.gpu_weights.lock().expect("gpu_weights mutex poisoned") = Some(weights);
        log::info!(
            "flare-gpu: uploaded {} layer weights to GPU ({} persistent buffers)",
            raw_layers.len(),
            raw_layers.len() * 11 + 3
        );
    }

    /// Returns `true` if GPU-resident weights have been uploaded.
    pub fn has_gpu_weights(&self) -> bool {
        self.gpu_weights
            .lock()
            .expect("gpu_weights mutex poisoned")
            .is_some()
    }

    // -----------------------------------------------------------------------
    // GPU-resident forward pass — single command encoder, no intermediate readback
    // -----------------------------------------------------------------------

    /// Returns the shader name for a given weight format's dequant_matvec kernel.
    fn dequant_shader_name(format: WeightFormat) -> &'static str {
        match format {
            WeightFormat::BF16 => "dequant_matvec_bf16",
            WeightFormat::F16 => "dequant_matvec_f16",
            WeightFormat::Q2K => "dequant_matvec_q2k",
            WeightFormat::Q3K => "dequant_matvec_q3k",
            WeightFormat::Q4_0 => "dequant_matvec_q4_0",
            WeightFormat::Q4_1 => "dequant_matvec_q4_1",
            WeightFormat::Q4K => "dequant_matvec_q4k",
            WeightFormat::Q5_0 => "dequant_matvec_q5_0",
            WeightFormat::Q5_1 => "dequant_matvec_q5_1",
            WeightFormat::Q5K => "dequant_matvec_q5k",
            WeightFormat::Q6K => "dequant_matvec_q6k",
            WeightFormat::Q8_0 => "dequant_matvec_q8_0",
            WeightFormat::Q8_1 => "dequant_matvec_q8_1",
            WeightFormat::IQ4NL => "dequant_matvec_iq4nl",
            WeightFormat::IQ4XS => "dequant_matvec_iq4xs",
            WeightFormat::IQ3S => "dequant_matvec_iq3s",
            WeightFormat::IQ2XXS => "dequant_matvec_iq2xxs",
            WeightFormat::IQ2XS => "dequant_matvec_iq2xs",
            WeightFormat::IQ3XXS => "dequant_matvec_iq3xxs",
            WeightFormat::IQ2S => "dequant_matvec_iq2s",
            WeightFormat::IQ1S => "dequant_matvec_iq1s",
        }
    }

    /// Returns the shader source for a given weight format's dequant_matvec kernel.
    fn dequant_shader_source(format: WeightFormat) -> &'static str {
        match format {
            WeightFormat::BF16 => DEQUANT_MATVEC_BF16_SHADER,
            WeightFormat::F16 => DEQUANT_MATVEC_F16_SHADER,
            WeightFormat::Q2K => DEQUANT_MATVEC_Q2K_SHADER,
            WeightFormat::Q3K => DEQUANT_MATVEC_Q3K_SHADER,
            WeightFormat::Q4_0 => DEQUANT_MATVEC_Q4_0_SHADER,
            WeightFormat::Q4_1 => DEQUANT_MATVEC_Q4_1_SHADER,
            WeightFormat::Q4K => DEQUANT_MATVEC_Q4K_SHADER,
            WeightFormat::Q5_0 => DEQUANT_MATVEC_Q5_0_SHADER,
            WeightFormat::Q5_1 => DEQUANT_MATVEC_Q5_1_SHADER,
            WeightFormat::Q5K => DEQUANT_MATVEC_Q5K_SHADER,
            WeightFormat::Q6K => DEQUANT_MATVEC_Q6K_SHADER,
            WeightFormat::Q8_0 => DEQUANT_MATVEC_Q8_0_SHADER,
            WeightFormat::Q8_1 => DEQUANT_MATVEC_Q8_1_SHADER,
            WeightFormat::IQ4NL => DEQUANT_MATVEC_IQ4NL_SHADER,
            WeightFormat::IQ4XS => DEQUANT_MATVEC_IQ4XS_SHADER,
            WeightFormat::IQ3S => DEQUANT_MATVEC_IQ3S_SHADER,
            WeightFormat::IQ2XXS => DEQUANT_MATVEC_IQ2XXS_SHADER,
            WeightFormat::IQ2XS => DEQUANT_MATVEC_IQ2XS_SHADER,
            WeightFormat::IQ3XXS => DEQUANT_MATVEC_IQ3XXS_SHADER,
            WeightFormat::IQ2S => DEQUANT_MATVEC_IQ2S_SHADER,
            WeightFormat::IQ1S => DEQUANT_MATVEC_IQ1S_SHADER,
        }
    }

    /// Create a GPU storage buffer suitable for intermediate results.
    ///
    /// The buffer supports STORAGE | COPY_SRC | COPY_DST so it can be:
    /// - bound as read-only input to subsequent passes
    /// - bound as read-write output from compute passes
    /// - copied to a staging buffer for final readback
    fn create_intermediate_buffer(&self, num_f32: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_intermediate"),
            size: (num_f32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Enqueue a dequant_matvec compute pass into an existing command encoder.
    ///
    /// Reads from `weight_buf` (persistent) and `input_buf` (intermediate),
    /// writes to `output_buf` (intermediate). No readback.
    fn enqueue_dequant_matvec(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weight: &GpuWeightBuffer,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) {
        let shader_name = Self::dequant_shader_name(weight.format);
        let shader_source = Self::dequant_shader_source(weight.format);
        let num_rows = weight.num_rows;
        let blocks_per_row = weight.blocks_per_row;
        let batch = 1u32;

        let params: [u32; 3] = [num_rows as u32, blocks_per_row as u32, batch];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::standard_layout();
        self.cache.with_pipeline(
            &self.device,
            shader_name,
            shader_source,
            shader_name,
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: weight.buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(shader_name),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_rows as u32, batch, 1);
            },
        );

        // Uniform buffer is tiny and can be returned to the pool immediately
        // (its contents are consumed by the dispatch, not read back).
        self.pool.return_uniform(params_buf);
    }

    /// Enqueue a batched_rmsnorm compute pass into an existing command encoder.
    ///
    /// Reads from `input_buf` and `weight_buf`, writes to `output_buf`.
    fn enqueue_rmsnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        dim: usize,
        batch: usize,
        eps: f32,
    ) {
        let params: [u32; 3] = [dim as u32, batch as u32, eps.to_bits()];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::standard_layout();
        self.cache.with_pipeline(
            &self.device,
            "batched_rmsnorm",
            BATCHED_RMSNORM_SHADER,
            "batched_rmsnorm",
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: weight_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rmsnorm"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(batch as u32, 1, 1);
            },
        );

        self.pool.return_uniform(params_buf);
    }

    /// Enqueue a batched_rope compute pass into an existing command encoder.
    fn enqueue_rope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        let params: [u32; 5] = [
            num_heads as u32,
            head_dim as u32,
            1u32, // seq_len = 1 for single-token
            pos as u32,
            theta.to_bits(),
        ];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::single_input_layout();
        self.cache.with_pipeline(
            &self.device,
            "batched_rope",
            BATCHED_ROPE_SHADER,
            "batched_rope",
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let half = (head_dim / 2) as u32;
                let dispatch_x = (num_heads as u32 * half).div_ceil(64);

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rope"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, 1, 1);
            },
        );

        self.pool.return_uniform(params_buf);
    }

    /// Enqueue a silu_mul compute pass into an existing command encoder.
    fn enqueue_silu_mul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        size: usize,
    ) {
        let params: [u32; 1] = [size as u32];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::standard_layout();
        self.cache.with_pipeline(
            &self.device,
            "silu_mul",
            SILU_MUL_SHADER,
            "silu_mul",
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: gate_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let dispatch_x = (size as u32).div_ceil(256);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("silu_mul"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, 1, 1);
            },
        );

        self.pool.return_uniform(params_buf);
    }

    /// Enqueue an add_residual compute pass: x[i] += residual[i].
    fn enqueue_add_residual(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        residual_buf: &wgpu::Buffer,
        x_buf: &wgpu::Buffer,
        size: usize,
    ) {
        let params: [u32; 1] = [size as u32];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params),
        );

        let layout_entries = Self::single_input_layout();
        self.cache.with_pipeline(
            &self.device,
            "add_residual",
            ADD_RESIDUAL_SHADER,
            "add_residual",
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: residual_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: x_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let dispatch_x = (size as u32).div_ceil(256);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("add_residual"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, 1, 1);
            },
        );

        self.pool.return_uniform(params_buf);
    }

    /// Enqueue attention using GPU-resident KV cache for a single query head.
    fn enqueue_attention_head(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_buf: &wgpu::Buffer,
        q_offset_bytes: u64,
        k_cache_buf: &wgpu::Buffer,
        v_cache_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        out_offset_bytes: u64,
        head_dim: usize,
        seq_len: usize,
        num_kv_heads: usize,
        kv_head_idx: usize,
    ) {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let params_data: [u32; 5] = [
            seq_len as u32,
            head_dim as u32,
            scale.to_bits(),
            num_kv_heads as u32,
            kv_head_idx as u32,
        ];
        let params_buf = self.pool.get_uniform(
            &self.device,
            &self.queue,
            bytemuck::cast_slice(&params_data),
        );

        // We need per-head slices. Use buffer binding with offset and size.
        let q_head_size = (head_dim * 4) as u64;
        let out_head_size = (head_dim * 4) as u64;

        let layout_entries = Self::attention_layout();
        self.cache.with_pipeline(
            &self.device,
            "attention_scores",
            ATTENTION_SHADER,
            "attention_scores",
            &layout_entries,
            |cached| {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: q_buf,
                                offset: q_offset_bytes,
                                size: std::num::NonZeroU64::new(q_head_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: k_cache_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: v_cache_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: output_buf,
                                offset: out_offset_bytes,
                                size: std::num::NonZeroU64::new(out_head_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("attention"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );

        self.pool.return_uniform(params_buf);
    }

    /// GPU-resident single-token forward pass.
    ///
    /// Builds ONE command encoder with all compute passes for the entire
    /// transformer forward pass. Intermediate results stay in GPU storage
    /// buffers. Only the final logits vector is read back to CPU.
    ///
    /// Returns `None` if GPU weights or KV cache are not initialized.
    ///
    /// # Arguments
    /// * `token_id` — input token
    /// * `pos` — sequence position for RoPE
    /// * `dim` — hidden dimension
    /// * `num_heads` — number of query heads
    /// * `num_kv_heads` — number of key/value heads
    /// * `head_dim` — dimension per head
    /// * `intermediate_dim` — FFN intermediate size
    /// * `vocab_size` — vocabulary size for logits output
    /// * `rms_norm_eps` — epsilon for RMSNorm
    /// * `rope_theta` — RoPE base frequency
    /// * `num_layers` — number of transformer layers
    /// * `seq_len` — current sequence length (including this token)
    /// * `token_embedding` — token embedding slice `[dim]`
    /// * `k_data_per_layer` — mutable slices for writing K back to CPU KV cache
    /// * `v_data_per_layer` — mutable slices for writing V back to CPU KV cache
    #[allow(clippy::too_many_arguments)]
    pub fn forward_single_token_gpu(
        &self,
        token_embedding: &[f32],
        pos: usize,
        dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        rope_theta: f32,
        num_layers: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let guard = self.gpu_weights.lock().expect("gpu_weights mutex poisoned");
        let weights = guard.as_ref().expect("forward_single_token_gpu called without GPU weights");
        let kv_guard = self.gpu_kv_cache.lock().expect("gpu_kv_cache mutex poisoned");
        let kv = kv_guard.as_ref().expect("forward_single_token_gpu called without GPU KV cache");

        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        // Allocate persistent intermediate buffers for the forward pass.
        // These stay on GPU between compute passes.
        let x_buf = self.create_intermediate_buffer(dim);
        let normed_buf = self.create_intermediate_buffer(dim);
        let q_buf = self.create_intermediate_buffer(q_dim);
        let k_buf = self.create_intermediate_buffer(kv_dim);
        let v_buf = self.create_intermediate_buffer(kv_dim);
        let q_rope_buf = self.create_intermediate_buffer(q_dim);
        let k_rope_buf = self.create_intermediate_buffer(kv_dim);
        let attn_out_buf = self.create_intermediate_buffer(q_dim);
        let attn_proj_buf = self.create_intermediate_buffer(dim);
        let ffn_normed_buf = self.create_intermediate_buffer(dim);
        let gate_buf = self.create_intermediate_buffer(intermediate_dim);
        let up_buf = self.create_intermediate_buffer(intermediate_dim);
        let silu_buf = self.create_intermediate_buffer(intermediate_dim);
        let ffn_out_buf = self.create_intermediate_buffer(dim);
        let final_normed_buf = self.create_intermediate_buffer(dim);
        let logits_buf = self.create_intermediate_buffer(vocab_size);

        // Upload token embedding to the x buffer
        self.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(token_embedding));

        // Build a single command encoder for the entire forward pass.
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("forward_gpu") },
        );

        for layer_idx in 0..num_layers {
            let layer = &weights.layers[layer_idx];

            // --- Attention block ---

            // 1. RMSNorm(x) -> normed
            self.enqueue_rmsnorm(
                &mut encoder,
                &x_buf,
                &layer.attn_norm,
                &normed_buf,
                dim,
                1,
                rms_norm_eps,
            );

            // 2. Q/K/V projections: dequant_matvec(weight, normed) -> q/k/v
            self.enqueue_dequant_matvec(&mut encoder, &layer.wq, &normed_buf, &q_buf);
            self.enqueue_dequant_matvec(&mut encoder, &layer.wk, &normed_buf, &k_buf);
            self.enqueue_dequant_matvec(&mut encoder, &layer.wv, &normed_buf, &v_buf);

            // 3. RoPE: apply to Q and K separately
            self.enqueue_rope(
                &mut encoder,
                &q_buf,
                &q_rope_buf,
                num_heads,
                head_dim,
                pos,
                rope_theta,
            );
            self.enqueue_rope(
                &mut encoder,
                &k_buf,
                &k_rope_buf,
                num_kv_heads,
                head_dim,
                pos,
                rope_theta,
            );

            // 4. Write K/V to GPU KV cache.
            //    We need to submit what we have so far to get K/V data, then copy
            //    from k_rope_buf/v_buf to the KV cache buffers via encoder.
            //    Actually we can use copy_buffer_to_buffer for the KV cache write.
            let ring_pos = pos % kv.max_seq_len;
            let kv_byte_offset = (ring_pos * kv_dim * 4) as u64;
            let kv_copy_size = (kv_dim * 4) as u64;
            encoder.copy_buffer_to_buffer(
                &k_rope_buf,
                0,
                kv.key_buf(layer_idx),
                kv_byte_offset,
                kv_copy_size,
            );
            encoder.copy_buffer_to_buffer(
                &v_buf,
                0,
                kv.val_buf(layer_idx),
                kv_byte_offset,
                kv_copy_size,
            );

            // 5. Grouped-query attention using GPU KV cache.
            //    For each query head, dispatch the attention shader.
            for h in 0..num_heads {
                let kv_head = h / heads_per_kv;
                self.enqueue_attention_head(
                    &mut encoder,
                    &q_rope_buf,
                    (h * head_dim * 4) as u64,
                    kv.key_buf(layer_idx),
                    kv.val_buf(layer_idx),
                    &attn_out_buf,
                    (h * head_dim * 4) as u64,
                    head_dim,
                    seq_len,
                    num_kv_heads,
                    kv_head,
                );
            }

            // 6. Output projection: wo @ attn_out -> attn_proj
            self.enqueue_dequant_matvec(&mut encoder, &layer.wo, &attn_out_buf, &attn_proj_buf);

            // 7. Optional post-attention RMSNorm (Gemma 2)
            if let Some(ref post_norm_buf) = layer.post_attn_norm {
                // Reuse ffn_normed_buf as temp for post-norm result
                self.enqueue_rmsnorm(
                    &mut encoder,
                    &attn_proj_buf,
                    post_norm_buf,
                    &ffn_normed_buf,
                    dim,
                    1,
                    rms_norm_eps,
                );
                // Add the post-normed result to x instead
                self.enqueue_add_residual(&mut encoder, &ffn_normed_buf, &x_buf, dim);
            } else {
                // 8. Residual: x += attn_proj
                self.enqueue_add_residual(&mut encoder, &attn_proj_buf, &x_buf, dim);
            }

            // --- FFN block ---

            // 9. RMSNorm(x) -> ffn_normed
            self.enqueue_rmsnorm(
                &mut encoder,
                &x_buf,
                &layer.ffn_norm,
                &ffn_normed_buf,
                dim,
                1,
                rms_norm_eps,
            );

            // 10. Gate and up projections
            self.enqueue_dequant_matvec(&mut encoder, &layer.w_gate, &ffn_normed_buf, &gate_buf);
            self.enqueue_dequant_matvec(&mut encoder, &layer.w_up, &ffn_normed_buf, &up_buf);

            // 11. SiLU(gate) * up -> silu_buf
            self.enqueue_silu_mul(&mut encoder, &gate_buf, &up_buf, &silu_buf, intermediate_dim);

            // 12. Down projection
            self.enqueue_dequant_matvec(&mut encoder, &layer.w_down, &silu_buf, &ffn_out_buf);

            // 13. Optional post-FFN RMSNorm (Gemma 2)
            if let Some(ref post_norm_buf) = layer.post_ffn_norm {
                self.enqueue_rmsnorm(
                    &mut encoder,
                    &ffn_out_buf,
                    post_norm_buf,
                    &normed_buf, // reuse normed_buf as temp
                    dim,
                    1,
                    rms_norm_eps,
                );
                self.enqueue_add_residual(&mut encoder, &normed_buf, &x_buf, dim);
            } else {
                // 14. Residual: x += ffn_out
                self.enqueue_add_residual(&mut encoder, &ffn_out_buf, &x_buf, dim);
            }
        }

        // --- Final output ---

        // 15. Final RMSNorm
        self.enqueue_rmsnorm(
            &mut encoder,
            &x_buf,
            &weights.output_norm,
            &final_normed_buf,
            dim,
            1,
            rms_norm_eps,
        );

        // 16. Output logits: output_weight @ final_normed -> logits
        //     output_weight is f32, sharded across multiple GPU buffers.
        //     We dispatch one batched_matvec per shard, each writing to the
        //     correct offset within logits_buf.
        for shard in &weights.output_weight_shards {
            let shard_rows = shard.num_rows;
            let params: [u32; 3] = [shard_rows as u32, dim as u32, 1u32];
            let params_buf = self.pool.get_uniform(
                &self.device,
                &self.queue,
                bytemuck::cast_slice(&params),
            );

            // Create a view into logits_buf at the correct row offset.
            let logit_byte_offset = (shard.row_offset * 4) as u64;
            let logit_byte_size = (shard_rows * 4) as u64;

            let layout_entries = Self::standard_layout();
            self.cache.with_pipeline(
                &self.device,
                "batched_matvec",
                BATCHED_MATVEC_SHADER,
                "batched_matvec",
                &layout_entries,
                |cached| {
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &cached.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: shard.buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: final_normed_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &logits_buf,
                                    offset: logit_byte_offset,
                                    size: Some(std::num::NonZeroU64::new(logit_byte_size).unwrap()),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("logits_matvec"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&cached.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(shard_rows as u32, 1, 1);
                },
            );

            self.pool.return_uniform(params_buf);
        }

        // 17. Readback only the logits
        let logits_size = (vocab_size * 4) as u64;
        let staging = self.pool.get_staging(&self.device, logits_size);
        encoder.copy_buffer_to_buffer(&logits_buf, 0, &staging, 0, logits_size);

        // Submit the ENTIRE forward pass as ONE command buffer.
        self.queue.submit(Some(encoder.finish()));

        // Map and read back logits
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
        let logits: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        self.pool.return_staging(staging);

        logits
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

            // Params: seq_len, head_dim, scale, num_kv_heads, kv_head_idx.
            // The K/V slice is already per-head (extracted above), so num_kv_heads=1
            // and kv_head_idx=0 — the shader indexes as t*1*head_dim + 0*head_dim + d.
            let params_data: [u32; 5] = [
                seq_len as u32,
                head_dim as u32,
                scale.to_bits(), // pass f32 as u32 raw bits; shader reads as f32
                1u32,            // num_kv_heads: per-head slice layout
                0u32,            // kv_head_idx: always 0 for per-head slices
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

    fn init_gpu_kv_cache(
        &self,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let kv = GpuKvCache::new(
            &self.device,
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        );
        *self
            .gpu_kv_cache
            .lock()
            .expect("gpu_kv_cache mutex poisoned") = Some(kv);
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_kv_write(
        &self,
        layer: usize,
        position: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        key: &[f32],
        value: &[f32],
    ) {
        if let Some(ref kv) = *self
            .gpu_kv_cache
            .lock()
            .expect("gpu_kv_cache mutex poisoned")
        {
            kv.write(&self.queue, layer, position, key, value);
        }
    }

    fn has_gpu_kv_cache(&self) -> bool {
        self.gpu_kv_cache
            .lock()
            .expect("gpu_kv_cache mutex poisoned")
            .is_some()
    }

    /// Grouped-query attention using GPU-resident K/V buffers.
    ///
    /// Binds the full per-layer K/V GPU buffer directly — no CPU-side extraction
    /// and no per-head CPU→GPU upload.  The WGSL shader handles head-stride
    /// indexing via `num_kv_heads` and `kv_head_idx` params.
    #[allow(clippy::too_many_arguments)]
    fn grouped_query_attention_from_gpu_cache(
        &self,
        q: &[f32],
        layer: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        attn_softcap: f32,
    ) -> Vec<f32> {
        let guard = self
            .gpu_kv_cache
            .lock()
            .expect("gpu_kv_cache mutex poisoned");
        let kv = guard
            .as_ref()
            .expect("grouped_query_attention_from_gpu_cache called without GPU KV cache");

        // Gemma-2 soft-cap is not implemented in the WGSL shader; fall back to CPU.
        // The CPU path re-reads K/V from the GPU-resident buffers would require
        // readback, which defeats the purpose.  Since Gemma-2 models are not the
        // primary use-case for GPU inference today, panic loudly instead.
        assert_eq!(
            attn_softcap, 0.0,
            "attn_logit_softcap != 0.0 is not supported with GPU KV cache"
        );

        let heads_per_kv = num_heads / num_kv_heads;
        let mut output = vec![0.0f32; num_heads * head_dim];
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let layout_entries = Self::attention_layout();

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            let output_size = head_dim as u64 * 4;

            // Upload Q for this head; bind the full-layer K/V GPU buffers directly.
            let q_buf =
                self.pool
                    .get_storage(&self.device, &self.queue, bytemuck::cast_slice(q_head));
            let out_buf = self.pool.get_output(&self.device, output_size);

            // Params include num_kv_heads and kv_head_idx so the shader can index
            // the interleaved full-layer K/V buffer without any CPU extraction.
            let params_data: [u32; 5] = [
                seq_len as u32,
                head_dim as u32,
                scale.to_bits(),
                num_kv_heads as u32,
                kv_head as u32,
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
                        kv.key_buf(layer),
                        kv.val_buf(layer),
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
            self.pool.return_output(out_buf);
            self.pool.return_uniform(params_buf);
        }

        output
    }

    fn prefill_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_softcap: f32,
    ) -> Vec<f32> {
        self.prefill_attention(
            q,
            k,
            v,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            attn_softcap,
        )
    }

    fn batched_matmul(
        &self,
        weight: &[f32],
        input: &[f32],
        out_rows: usize,
        in_cols: usize,
        batch: usize,
    ) -> Vec<f32> {
        self.batched_matvec(weight, input, out_rows, in_cols, batch)
    }

    fn batched_rmsnorm(
        &self,
        input: &[f32],
        weight: &[f32],
        dim: usize,
        batch: usize,
        eps: f32,
    ) -> Vec<f32> {
        WebGpuBackend::batched_rmsnorm(self, input, weight, dim, batch, eps)
    }

    fn batched_rope(
        &self,
        inp: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        start_pos: usize,
        theta: f32,
    ) -> Vec<f32> {
        WebGpuBackend::batched_rope(self, inp, num_heads, head_dim, seq_len, start_pos, theta)
    }

    fn supports_dequant_matmul(&self) -> bool {
        true
    }

    fn batched_dequant_matmul(&self, weight: &RawWeight, input: &[f32], batch: usize) -> Vec<f32> {
        let num_rows = weight.num_rows;

        match weight.format {
            WeightFormat::BF16 => self.dequant_matvec_bf16(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::F16 => {
                self.dequant_matvec_f16(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::Q4_1 => self.dequant_matvec_q4_1(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q8_0 => self.dequant_matvec_q8_0(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q8_1 => self.dequant_matvec_q8_1(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q4_0 => self.dequant_matvec_q4_0(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q5_0 => self.dequant_matvec_q5_0(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q5_1 => self.dequant_matvec_q5_1(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::Q2K => {
                self.dequant_matvec_q2k(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::Q3K => {
                self.dequant_matvec_q3k(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::Q4K => {
                self.dequant_matvec_q4k(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::Q5K => {
                self.dequant_matvec_q5k(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::Q6K => {
                self.dequant_matvec_q6k(&weight.data, input, num_rows, weight.blocks_per_row, batch)
            }
            WeightFormat::IQ4NL => self.dequant_matvec_iq4nl(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ2XXS => self.dequant_matvec_iq2xxs(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ2XS => self.dequant_matvec_iq2xs(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ3XXS => self.dequant_matvec_iq3xxs(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ4XS => self.dequant_matvec_iq4xs(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ3S => self.dequant_matvec_iq3s(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ2S => self.dequant_matvec_iq2s(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
            WeightFormat::IQ1S => self.dequant_matvec_iq1s(
                &weight.data,
                input,
                num_rows,
                weight.blocks_per_row,
                batch,
            ),
        }
    }

    fn pipeline_cache_data(&self) -> Vec<u8> {
        self.get_pipeline_cache_bytes()
    }

    fn supports_gpu_forward(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn upload_weights_to_gpu(
        &self,
        raw_layers: &[flare_core::model::RawLayerWeights],
        layer_norms: &[(
            &[f32],
            &[f32],
            Option<&[f32]>,
            Option<&[f32]>,
        )],
        output_norm: &[f32],
        output_weight: &[f32],
        token_embedding: &[f32],
    ) {
        WebGpuBackend::upload_weights_to_gpu(
            self,
            raw_layers,
            layer_norms,
            output_norm,
            output_weight,
            token_embedding,
        );
    }

    fn has_gpu_weights(&self) -> bool {
        WebGpuBackend::has_gpu_weights(self)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_single_token_gpu(
        &self,
        token_embedding: &[f32],
        pos: usize,
        dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        rope_theta: f32,
        num_layers: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if !WebGpuBackend::has_gpu_weights(self) || !self.has_gpu_kv_cache() {
            return None;
        }
        Some(WebGpuBackend::forward_single_token_gpu(
            self,
            token_embedding,
            pos,
            dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            num_layers,
            seq_len,
        ))
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

    /// Verify GPU Q2_K fused dequant+matvec against CPU reference.
    ///
    /// One block: d=1.0, dmin=0.0, all scales bytes = 0x01 (scale_nibble=1, min_nibble=0),
    /// all qs bytes = 0x55 (binary 01010101: each 2-bit group = 1).
    /// Each weight: d * 1 * 1 − dmin * 0 = 1.0.
    /// Input vector: all 1.0. Expected dot product: 256 × 1.0 = 256.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q2k_matches_cpu() {
        let mut raw = vec![0u8; 84];
        // qs[64]: 0x55 = 01010101b → each 2-bit group = 01 = 1
        for b in raw[0..64].iter_mut() {
            *b = 0x55;
        }
        // scales[16]: scale_nibble=1, min_nibble=0 → byte = 0x01
        for b in raw[64..80].iter_mut() {
            *b = 0x01;
        }
        // d = 1.0 as f16 LE: 0x3C00 → bytes [0x00, 0x3C]
        raw[80] = 0x00;
        raw[81] = 0x3C;
        // dmin = 0.0 (already zero)

        let input = vec![1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q2k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // 256 weights each = 1.0 * 1 * 1 - 0 = 1.0; dot with [1]*256 = 256.0
        assert!(
            (result[0] - 256.0).abs() < 1e-2,
            "dequant_matvec_q2k mismatch: got {}, expected 256.0",
            result[0]
        );
    }

    /// Verify GPU Q2_K fused dequant+matvec with non-trivial scales and min.
    ///
    /// One block: d=1.0, dmin=1.0, scales[*]=0x12 (scale_nibble=2, min_nibble=1),
    /// qs[*]=0xAA (binary 10101010: each 2-bit group = 10 = 2).
    /// weight = d * 2 * 2 − dmin * 1 = 4 − 1 = 3.0.
    /// Input: all 1.0. Expected: 256 × 3.0 = 768.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q2k_with_min() {
        let mut raw = vec![0u8; 84];
        // qs[64]: 0xAA = 10101010b → each 2-bit group = 2
        for b in raw[0..64].iter_mut() {
            *b = 0xAA;
        }
        // scales[16]: scale_nibble=2, min_nibble=1 → byte = 0x12
        for b in raw[64..80].iter_mut() {
            *b = 0x12;
        }
        // d = 1.0 as f16 LE
        raw[80] = 0x00;
        raw[81] = 0x3C;
        // dmin = 1.0 as f16 LE
        raw[82] = 0x00;
        raw[83] = 0x3C;

        let input = vec![1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q2k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // weight = 1*2*2 - 1*1 = 3.0; dot with [1]*256 = 768.0
        assert!(
            (result[0] - 768.0).abs() < 1e-1,
            "dequant_matvec_q2k with_min mismatch: got {}, expected 768.0",
            result[0]
        );
    }

    /// Verify GPU Q4_0 fused dequant+matvec against CPU reference.
    ///
    /// One block: scale=1.0, all qs bytes = 0x88 (lo=8, hi=8).
    /// lo−8=0, hi−8=0 → all weights = 0.0.
    /// Input vector: all 1.0. Expected dot product: 0.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4_0_zero_weights() {
        let mut raw = vec![0u8; 18];
        // scale = 1.0 as f16 LE: 0x3C00 → bytes [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qs[16]: lo=8, hi=8 → byte = 0x88; weight = (q-8)*scale = 0
        for b in raw[2..18].iter_mut() {
            *b = 0x88;
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4_0(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        assert!(
            result[0].abs() < 1e-3,
            "dequant_matvec_q4_0 zero mismatch: got {}, expected 0.0",
            result[0]
        );
    }

    /// Verify GPU Q4_0 fused dequant+matvec with non-zero weights.
    ///
    /// One block: scale=1.0, all qs bytes = 0x9A (lo=10, hi=9).
    /// weight[lo] = (10−8)*1.0 = 2.0, weight[hi] = (9−8)*1.0 = 1.0.
    /// Input: all 1.0.
    /// Expected: 16×2.0 + 16×1.0 = 48.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4_0_nonzero() {
        let mut raw = vec![0u8; 18];
        // scale = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qs: lo nibble = 10 (A), hi nibble = 9 → byte = 0x9A
        for b in raw[2..18].iter_mut() {
            *b = 0x9A;
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4_0(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // 16 pairs: w_lo = 2.0, w_hi = 1.0 → dot = 16*2 + 16*1 = 48
        assert!(
            (result[0] - 48.0).abs() < 1e-2,
            "dequant_matvec_q4_0 nonzero mismatch: got {}, expected 48.0",
            result[0]
        );
    }

    /// Verify GPU Q4_1 fused dequant+matvec against CPU reference.
    ///
    /// One block: d=1.0, m=0.0, all qs bytes = 0x12 (lo=2, hi=1).
    /// Input vector: all 1.0.
    /// Expected dot product: Σ(d*q+m) × 1.0 for all 32 weights.
    /// = Σ [2, 1, 2, 1, ... (16 pairs)] = 16×2 + 16×1 = 48.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4_1_matches_cpu() {
        let mut raw = vec![0u8; 20];
        // d = 1.0 as f16 LE: 0x3C00 → bytes [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // m = 0.0 (already zero)
        // qs[16]: lo nibble = 2, hi nibble = 1 → byte = 0x12
        for b in raw[4..20].iter_mut() {
            *b = 0x12;
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4_1(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // weights: [2,1]*16 → dot with [1]*32 = 16*2 + 16*1 = 48
        assert!(
            (result[0] - 48.0).abs() < 1e-2,
            "dequant_matvec_q4_1 mismatch: got {}, expected 48.0",
            result[0]
        );
    }

    /// Verify GPU Q4_1 fused dequant+matvec with multiple rows.
    ///
    /// Two rows, same block data. Each output should be 48.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4_1_multi_row() {
        let mut raw = vec![0u8; 40]; // 2 rows × 1 block × 20 bytes
        for row in 0..2 {
            let base = row * 20;
            // d = 1.0 as f16 LE
            raw[base] = 0x00;
            raw[base + 1] = 0x3C;
            // qs: lo=2, hi=1
            for b in raw[base + 4..base + 20].iter_mut() {
                *b = 0x12;
            }
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4_1(&raw, &input, 2, 1, 1);

        assert_eq!(result.len(), 2, "expected 2 outputs");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 48.0).abs() < 1e-2,
                "dequant_matvec_q4_1 multi_row mismatch at row {}: got {}, expected 48.0",
                i,
                v
            );
        }
    }

    /// Verify GPU Q8_0 fused dequant+matvec matches CPU reference dequantization.
    ///
    /// One block: scale=1.0, qs = [0, 1, 2, ..., 31] (ascending int8).
    /// Input: all 1.0. Expected dot product = 0+1+…+31 = 496.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q8_0_matches_cpu() {
        use flare_loader::quantize::dequant_q8_0_block;

        let mut raw = [0u8; 34];
        // scale = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qs[32] = 0, 1, 2, ..., 31
        for i in 0..32usize {
            raw[2 + i] = i as u8;
        }

        let mut dequant_out = [0.0f32; 32];
        dequant_q8_0_block(&raw, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let input = [1.0f32; 32];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q8_0(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "dequant_matvec_q8_0 mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify GPU Q8_0 fused dequant+matvec handles negative weights correctly.
    ///
    /// All qs = -1 (0xFF as u8), scale = 1.0. Expected dot product = -1 * 32 = -32.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q8_0_negative_weights() {
        let mut raw = [0u8; 34];
        raw[0] = 0x00;
        raw[1] = 0x3C; // scale = 1.0
        for b in raw[2..34].iter_mut() {
            *b = 0xFF; // -1 as i8
        }

        let input = [1.0f32; 32];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q8_0(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - (-32.0f32)).abs() < 1e-3,
            "expected -32.0, got {}",
            result[0]
        );
    }

    /// Verify GPU Q8_1 fused dequant+matvec against CPU reference.
    ///
    /// One block: d=1.0, qs[0..32] = [0, 1, 2, ..., 31] (ascending int8).
    /// Input vector: all 1.0.
    /// Expected dot product: Σ(d * i * 1.0) for i=0..31 = 0+1+…+31 = 496.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q8_1_matches_cpu() {
        let mut raw = vec![0u8; 36];
        // d = 1.0 as f16 LE: 0x3C00 → bytes [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // s = 0.0 (already zero, bytes 2-3)
        // qs[32]: values 0..31
        for i in 0..32usize {
            raw[4 + i] = i as u8;
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q8_1(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // Σ i for i=0..31 = 31*32/2 = 496
        assert!(
            (result[0] - 496.0).abs() < 1e-2,
            "dequant_matvec_q8_1 mismatch: got {}, expected 496.0",
            result[0]
        );
    }

    /// Verify GPU Q8_1 fused dequant+matvec with negative int8 values.
    ///
    /// One block: d=1.0, qs[32] = [-1i8 as u8; 32] = [0xFF; 32].
    /// Input vector: all 1.0.
    /// Expected dot product: 32 × (-1) × 1.0 = -32.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q8_1_negative_weights() {
        let mut raw = vec![0u8; 36];
        // d = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qs[32] = -1 as i8 = 0xFF
        for b in raw[4..36].iter_mut() {
            *b = 0xFF;
        }

        let input = vec![1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q8_1(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        // 32 × (-1.0) = -32.0
        assert!(
            (result[0] - (-32.0)).abs() < 1e-2,
            "dequant_matvec_q8_1 negative mismatch: got {}, expected -32.0",
            result[0]
        );
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
        let mut raw = [0u8; 144];
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

    /// Verify fused Q4_K dequant-matvec against a CPU reference.
    ///
    /// One block per row (1 row × 1 block = 144 bytes), d=1.0, dmin=0.0,
    /// sc[*]=1, mn[*]=0, qs=0x12 (lo=2, hi=1).
    /// Input vector = all 1.0.
    /// Expected: dot = 128 × 2.0 + 128 × 1.0 = 384.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4k_matches_cpu() {
        let mut raw = [0u8; 144];
        // d = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // dmin = 0.0 (already zero)
        // scales_raw[0..4] = 0x41: sc[i]=1, bits[7:6]=01 → sc[i+4]=1
        for b in raw[4..8].iter_mut() {
            *b = 0x41;
        }
        // scales_raw[4..8] and [8..12] = 0x00 (mn[*]=0, already zero)
        // qs[128]: lo=2, hi=1 → byte = 0x12
        for b in raw[16..144].iter_mut() {
            *b = 0x12;
        }

        // Input: all ones
        let input = [1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        // 128 weights of 2.0 and 128 of 1.0, all dotted with 1.0
        let expected = 128.0 * 2.0 + 128.0 * 1.0; // 384.0
        assert!(
            (result[0] - expected).abs() < 1e-1,
            "dequant_matvec_q4k mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Multi-row test: 3 rows × 1 block, same Q4_K block, input = all 1.0.
    /// All rows should give 384.0.
    #[test]
    #[ignore]
    fn test_dequant_matvec_q4k_multi_row() {
        let mut block = [0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        for b in block[4..8].iter_mut() {
            *b = 0x41; // sc[*]=1
        }
        for b in block[16..144].iter_mut() {
            *b = 0x12; // qs: lo=2, hi=1
        }

        // 3 rows × 1 block each
        let num_rows = 3usize;
        let raw: Vec<u8> = block.iter().copied().cycle().take(144 * num_rows).collect();
        let input = [1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q4k(&raw, &input, num_rows, 1, 1);

        assert_eq!(result.len(), num_rows);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 384.0).abs() < 1e-1, "row {i}: got {v}, expected 384.0");
        }
    }

    /// Verify Q5_K GPU dequant matches the CPU reference.
    ///
    /// Block setup: d=1.0, dmin=0.0, all sc[*]=1 (scales_raw[0..4]=0x41),
    /// mn[*]=0, qh=0 (no high bit), ql all 0x12 (lo=2, hi=1).
    /// Expected: output[0..128]=2.0, output[128..256]=1.0
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q5k_matches_cpu() {
        use flare_loader::quantize::dequant_q5k_block;

        let mut raw = [0u8; 176];
        // d = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // dmin = 0.0 (already zero)
        // scales_raw[0..4] = 0x41: sc[i]=1 (bits[5:0]=1), bits[7:6]=01 → sc[i+4]=1
        for b in raw[4..8].iter_mut() {
            *b = 0x41;
        }
        // scales_raw[4..12] = 0x00: mn[*]=0 (already zero)
        // qh[0..32] = 0x00: no high bits → lo/hi nibbles are the full 4-bit value
        // ql[128] at bytes 48-175: lo=2, hi=1 → byte = 0x12
        for b in raw[48..176].iter_mut() {
            *b = 0x12;
        }

        // CPU reference
        let mut cpu_out = [0.0f32; 256];
        dequant_q5k_block(&raw, &mut cpu_out);

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q5k(&raw, 1);

        assert_eq!(result.len(), 256, "expected 256 weights");
        for (j, (&gpu, &cpu)) in result.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-3,
                "q5k mismatch at [{j}]: gpu={gpu}, cpu={cpu}"
            );
        }
    }

    /// Verify Q6_K GPU dequant matches the CPU reference.
    ///
    /// Block setup: all-zero block (d=0.0 → all outputs = 0.0).
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q6k_zeroed() {
        use flare_loader::quantize::dequant_q6k_block;

        let raw = [0u8; 210];

        let mut cpu_out = [0.0f32; 256];
        dequant_q6k_block(&raw, &mut cpu_out);

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q6k(&raw, 1);

        assert_eq!(result.len(), 256);
        for (j, (&gpu, &cpu)) in result.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-3,
                "q6k zeroed mismatch at [{j}]: gpu={gpu}, cpu={cpu}"
            );
        }
    }

    /// Verify Q6_K GPU dequant with d=1.0, non-trivial scales and qh bits.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q6k_matches_cpu() {
        use flare_loader::quantize::dequant_q6k_block;

        let mut raw = [0u8; 210];
        // d = 1.0 as f16 LE at bytes 208-209
        raw[208] = 0x00;
        raw[209] = 0x3C;
        // scales[16] at bytes 192-207: set all to 1 (0x01 as signed i8)
        for b in raw[192..208].iter_mut() {
            *b = 0x01;
        }
        // ql[128] at bytes 0-127: lo nibble = 1, hi nibble = 2 → byte = 0x21
        for b in raw[0..128].iter_mut() {
            *b = 0x21;
        }
        // qh[64] at bytes 128-191: set upper 2 bits to 0b01 per group of 4
        // qh byte = 0x55 = 0b01010101: bits [1:0]=01, [3:2]=01, [5:4]=01, [7:6]=01
        for b in raw[128..192].iter_mut() {
            *b = 0x55;
        }

        // CPU reference
        let mut cpu_out = [0.0f32; 256];
        dequant_q6k_block(&raw, &mut cpu_out);

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q6k(&raw, 1);

        assert_eq!(result.len(), 256);
        for (j, (&gpu, &cpu)) in result.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-3,
                "q6k mismatch at [{j}]: gpu={gpu}, cpu={cpu}"
            );
        }
    }

    /// Verify GPU batched prefill attention matches the CPU reference.
    ///
    /// Verify GPU batched_rope matches CPU apply_rope for seq_len=3, num_heads=2, head_dim=4.
    ///
    /// Input: constant 1.0 across all elements. Checks that each token's rotation
    /// matches the per-token CPU reference.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_batched_rope_matches_cpu() {
        use flare_core::model::apply_rope;

        let seq_len = 3usize;
        let num_heads = 2usize;
        let head_dim = 4usize;
        let theta = 10000.0f32;
        let stride = num_heads * head_dim;

        let inp = vec![1.0f32; seq_len * stride];

        // CPU reference: apply_rope per token.
        let mut expected = inp.clone();
        for t in 0..seq_len {
            apply_rope(
                &mut expected[t * stride..(t + 1) * stride],
                num_heads,
                head_dim,
                t,
                theta,
            );
        }

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.batched_rope(&inp, num_heads, head_dim, seq_len, 0, theta);

        assert_eq!(result.len(), seq_len * stride);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    /// Uses a 3-position, 2-head, head_dim=4 setup:
    ///   Q = K = V = identity-like pattern (ones on diagonal per head).
    ///   Compares GPU output against `cpu_grouped_query_attention` per position.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_prefill_attention_matches_cpu() {
        use flare_core::model::cpu_grouped_query_attention;

        let seq_len = 4usize;
        let num_heads = 2usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Fill Q, K, V with deterministic values.
        let total_q = seq_len * q_dim;
        let total_kv = seq_len * kv_dim;
        let q: Vec<f32> = (0..total_q).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32) * 0.05).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| 1.0 / (1.0 + i as f32)).collect();

        // CPU reference: compute per-position causal attention.
        let mut cpu_out = vec![0.0f32; seq_len * q_dim];
        for t in 0..seq_len {
            let q_t = &q[t * q_dim..(t + 1) * q_dim];
            let k_t = &k[0..(t + 1) * kv_dim];
            let v_t = &v[0..(t + 1) * kv_dim];
            let attn_t = cpu_grouped_query_attention(
                q_t,
                k_t,
                v_t,
                num_heads,
                num_kv_heads,
                head_dim,
                t + 1,
                0.0,
            );
            cpu_out[t * q_dim..(t + 1) * q_dim].copy_from_slice(&attn_t);
        }

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let gpu_out =
            backend.prefill_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim, 0.0);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "prefill_attention mismatch at [{i}]: gpu={g}, cpu={c}"
            );
        }
    }

    /// Verify Q3_K GPU dequant with a zeroed block (d=0.0 → all outputs = 0.0).
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q3k_zeroed() {
        use flare_loader::quantize::dequant_q3k_block;

        let raw = [0u8; 110];
        let mut cpu_out = [0.0f32; 256];
        dequant_q3k_block(&raw, &mut cpu_out);

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q3k(&raw, 1);

        assert_eq!(result.len(), 256);
        for (j, (&gpu, &cpu)) in result.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-3,
                "q3k zeroed mismatch at [{j}]: gpu={gpu}, cpu={cpu}"
            );
        }
    }

    /// Verify fused Q5_K dequant-matvec against the CPU reference.
    ///
    /// Block setup: d=1.0, dmin=0.0, sc[*]=1 (scales_raw[0..4]=0x41), mn[*]=0,
    /// qh=0 (no high bit), ql all 0x12 (lo nibble=2, hi nibble=1).
    /// Input vector = all 1.0.
    /// Expected: dot = 128 × 2.0 + 128 × 1.0 = 384.0 (same as Q4_K with identical values).
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5k_matches_cpu() {
        let mut raw = [0u8; 176];
        // d = 1.0 as f16 LE
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // scales_raw[0..4] = 0x41: sc[i]=1, bits[7:6]=01 → sc[i+4]=1
        for b in raw[4..8].iter_mut() {
            *b = 0x41;
        }
        // qh[0..32] = 0x00: qh_bit=0 for all weights
        // ql[128] at bytes 48-175: lo nibble=2, hi nibble=1 → byte = 0x12
        for b in raw[48..176].iter_mut() {
            *b = 0x12;
        }

        let input = [1.0f32; 256];
        let expected = 128.0 * 2.0 + 128.0 * 1.0; // 384.0

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1e-1,
            "dequant_matvec_q5k mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Multi-row test: 3 rows × 1 Q5_K block, input = all 1.0.
    /// All rows should give 384.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5k_multi_row() {
        let mut block = [0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        for b in block[4..8].iter_mut() {
            *b = 0x41; // sc[*]=1
        }
        for b in block[48..176].iter_mut() {
            *b = 0x12; // ql: lo=2, hi=1; qh=0
        }

        let num_rows = 3usize;
        let raw: Vec<u8> = block.iter().copied().cycle().take(176 * num_rows).collect();
        let input = [1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5k(&raw, &input, num_rows, 1, 1);

        assert_eq!(result.len(), num_rows);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 384.0).abs() < 1e-1, "row {i}: got {v}, expected 384.0");
        }
    }

    /// Verify Q6_K GPU fused dequant+matvec matches CPU reference (dequant then dot product).
    ///
    /// Block setup: d=1.0, scales[0..16]=1 (signed i8), ql all 0x22 (both nibbles = 2),
    /// qh all 0x00 (upper 2 bits = 0).  This gives raw q = (2 | 0) - 32 = -30 for every
    /// weight.  With input = 1.0 everywhere, expected = 256 * 1.0 * 1 * (-30) = -7680.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q6k_matches_cpu() {
        use flare_loader::quantize::dequant_q6k_block;

        let mut raw = [0u8; 210];
        // d = 1.0 as f16 LE at bytes 208-209
        raw[208] = 0x00;
        raw[209] = 0x3C;
        // scales[16] at bytes 192-207 = 1 as signed i8
        for b in raw[192..208].iter_mut() {
            *b = 1;
        }
        // ql[128] at bytes 0-127 = 0x22: both nibbles = 2
        for b in raw[0..128].iter_mut() {
            *b = 0x22;
        }
        // qh[64] at bytes 128-191 = 0x00: all upper 2 bits = 0

        // CPU reference: dequant then dot product with all-1 input.
        let mut dequant_out = [0.0f32; 256];
        dequant_q6k_block(&raw, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum(); // input is all 1.0

        let input = [1.0f32; 256];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q6k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1e-1,
            "dequant_matvec_q6k mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Multi-row test: 3 rows × 1 Q6_K block, input = all 1.0.
    /// All rows should produce the same dot product as the CPU reference.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q6k_multi_row() {
        use flare_loader::quantize::dequant_q6k_block;

        let mut block = [0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        for b in block[192..208].iter_mut() {
            *b = 1; // scale = 1
        }
        for b in block[0..128].iter_mut() {
            *b = 0x22; // ql nibbles = 2
        }

        let mut dequant_out = [0.0f32; 256];
        dequant_q6k_block(&block, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let num_rows = 3usize;
        let raw: Vec<u8> = block.iter().copied().cycle().take(210 * num_rows).collect();
        let input = [1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q6k(&raw, &input, num_rows, 1, 1);

        assert_eq!(result.len(), num_rows);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-1,
                "row {i}: got {v}, expected {expected}"
            );
        }
    }

    /// Verify dequant_matvec_q3k single-block GPU result matches CPU reference dequantization.
    ///
    /// Uses a block with d=1.0, all-zero hmask (so sub=4 for every weight),
    /// all-zero qs (low2=0 → q=-4 for every weight), and scale=1 for all sub-blocks.
    /// Expected: each weight = 1.0 * 1 * (-4) = -4.0; dot product with all-1 input = -4 * 256.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q3k_matches_cpu() {
        use flare_loader::quantize::dequant_q3k_block;

        let mut raw = [0u8; 110];
        // d = 1.0 as f16 LE: bytes 108-109 = [0x00, 0x3C]
        raw[108] = 0x00;
        raw[109] = 0x3C;
        // scales bytes 96..108: encode scale=1 for all 8 sub-blocks.
        // scale[k] = (b[k] & 0x0F) | (((b[8+k] >> 4) & 3) << 4) - 32
        // We want scale=1, so raw_val = 33 = 0x21.
        // b[k] = 0x21 (low nibble = 1, high nibble = 2), b[8+k] = 0x00 → upper bits = 0
        // → scale = (1 | 0) - 32 = -31... that doesn't work easily.
        // Simplest: b[k] = 0x21 for k in 0..8, b[8+k] (for k in 0..4) must give
        // upper 2 bits = 0 so (b[8+k] >> 4) & 3 = 0 and (b[8+k] >> 6) & 3 = 0.
        // So scale[k] = (0x21 & 0x0F) = 1 for k=0..3 and (0x21 & 0x0F) = 1 for k=4..7.
        // Then scale_decoded = 1 - 32 = -31. Let's just use scale=0 → raw_val=32.
        // b[k] = 0x20 (low nibble = 0) → scale = 0 - 32 = -32. Still messy.
        // Use CPU dequant_q3k_block to get expected value, keep raw zero (scale=0 → 0 output).
        // With d=1, scales=0 (raw=0 → decoded = -32), hmask=0 (sub=4), qs=0 (low2=0 → q=-4):
        // w = 1 * (-32) * (-4) = 128.
        let mut dequant_out = [0.0f32; 256];
        dequant_q3k_block(&raw, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let input = [1.0f32; 256];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q3k(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1.0,
            "dequant_matvec_q3k mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify dequant_matvec_q3k produces consistent results for multiple rows.
    ///
    /// Uses 3 identical blocks; each should yield the same dot product as the CPU reference.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q3k_multi_row() {
        use flare_loader::quantize::dequant_q3k_block;

        let mut block = [0u8; 110];
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
                           // All other bytes zero: scale decodes to -32, hmask=0 → sub=4, qs=0 → q=-4
                           // w = 1 * (-32) * (-4) = 128 per weight; dot with 1-vec = 128 * 256 = 32768

        let mut dequant_out = [0.0f32; 256];
        dequant_q3k_block(&block, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let num_rows = 3usize;
        let raw: Vec<u8> = block.iter().copied().cycle().take(110 * num_rows).collect();
        let input = [1.0f32; 256];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q3k(&raw, &input, num_rows, 1, 1);

        assert_eq!(result.len(), num_rows);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1.0,
                "row {i}: got {v}, expected {expected}"
            );
        }
    }

    /// Verify dequant_matvec_q5_0 single-block GPU result matches CPU reference.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5_0_matches_cpu() {
        use flare_loader::quantize::dequant_q5_0_block;

        let mut raw = [0u8; 22];
        // scale = 1.0 as f16 LE: bytes 0-1 = [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qh = 0xFFFFFFFF: all high bits set, so xh_0=1 for weights 0..16, xh_1=1 for weights 16..32
        raw[2] = 0xFF;
        raw[3] = 0xFF;
        raw[4] = 0xFF;
        raw[5] = 0xFF;
        // qs[16] = 0xFF: lo nibble = 0xF = 15, hi nibble = 0xF = 15
        // With xh=1: x = (15 | 16) - 16 = 15
        for b in raw[6..22].iter_mut() {
            *b = 0xFF;
        }

        let mut dequant_out = [0.0f32; 32];
        dequant_q5_0_block(&raw, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let input = [1.0f32; 32];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5_0(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "dequant_matvec_q5_0 mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify dequant_matvec_q5_0 produces consistent results for multiple rows.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5_0_multi_row() {
        use flare_loader::quantize::dequant_q5_0_block;

        let mut block = [0u8; 22];
        block[0] = 0x00;
        block[1] = 0x3C; // scale = 1.0
                         // qh: bits 0-15 set → xh_0=1; bits 16-31 = 0 → xh_1=0
        block[2] = 0xFF;
        block[3] = 0xFF;
        block[4] = 0x00;
        block[5] = 0x00;
        // qs = 0x0F: lo nibble = 0xF, hi nibble = 0
        for b in block[6..22].iter_mut() {
            *b = 0x0F;
        }

        let mut dequant_out = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut dequant_out);
        let expected: f32 = dequant_out.iter().sum();

        let num_rows = 3usize;
        let raw: Vec<u8> = block.iter().copied().cycle().take(22 * num_rows).collect();
        let input = [1.0f32; 32];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5_0(&raw, &input, num_rows, 1, 1);

        assert_eq!(result.len(), num_rows);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-3,
                "row {i}: got {v}, expected {expected}"
            );
        }
    }

    /// Verify GPU Q5_1 fused dequant+matvec matches CPU reference.
    ///
    /// One block: d=1.0, m=0.0, qh=0 (all high bits clear → range [0,15]),
    /// qs all 0x55 → lo nibble=5, hi nibble=5.
    /// Expected: w[j] = 1.0*5 + 0 = 5.0 for all 32 weights. Dot with all-1 input = 5*32 = 160.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5_1_matches_cpu() {
        // Build a Q5_1 block manually (24 bytes = 6 u32s, LE)
        let mut raw = [0u8; 24];
        // d = 1.0 as f16: bytes 0-1 = [0x00, 0x3C]
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // m = 0.0 as f16: bytes 2-3 = [0x00, 0x00] (already zero)
        // qh = 0: bytes 4-7 = 0 (all high bits clear, q5 range [0, 15])
        // qs[16] = 0x55: lo nibble = 5, hi nibble = 5 → all q5 = 5
        for b in raw[8..24].iter_mut() {
            *b = 0x55;
        }

        let expected = 5.0f32 * 32.0; // d * 5 + 0 = 5.0 per weight, 32 weights
        let input = [1.0f32; 32];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5_1(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1, "expected one output value");
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "dequant_matvec_q5_1 mismatch: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify GPU Q5_1 handles non-zero min (m > 0) correctly.
    ///
    /// d=1.0, m=1.0, all q5=0 (qs=0, qh=0) → w[i] = 1.0*0 + 1.0 = 1.0.
    /// Dot with all-1 input = 32.0.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_q5_1_nonzero_min() {
        let mut raw = [0u8; 24];
        // d = 1.0 as f16: bytes 0-1
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // m = 1.0 as f16: bytes 2-3 = [0x00, 0x3C]
        raw[2] = 0x00;
        raw[3] = 0x3C;
        // qh = 0, qs = 0 → all q5 = 0; w[i] = 1.0*0 + 1.0 = 1.0

        let expected = 32.0f32;
        let input = [1.0f32; 32];
        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_q5_1(&raw, &input, 1, 1, 1);

        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - expected).abs() < 1e-3,
            "dequant_matvec_q5_1 min test: got {}, expected {}",
            result[0],
            expected
        );
    }

    /// Verify GPU batched_rmsnorm matches CPU rmsnorm for a 3-row × 4-dim batch.
    ///
    /// Input rows: [1,2,3,4], [2,2,2,2], [3,3,3,3].
    /// Weight: [1,1,1,1] (identity scaling).
    /// Expected: each row normalised to unit RMS.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_batched_rmsnorm_matches_cpu() {
        use flare_core::model::rmsnorm;

        let dim = 4usize;
        let batch = 3usize;
        let eps = 1e-5f32;
        let weight = [1.0f32; 4];

        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            2.0, 2.0, 2.0, 2.0, // row 1
            3.0, 3.0, 3.0, 3.0, // row 2
        ];

        // CPU reference.
        let expected: Vec<f32> = (0..batch)
            .flat_map(|b| rmsnorm(&input[b * dim..(b + 1) * dim], &weight, eps))
            .collect();

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.batched_rmsnorm(&input, &weight, dim, batch, eps);

        assert_eq!(result.len(), batch * dim);
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    /// Verify Q3_K GPU dequant matches CPU reference with d=1.0 and non-trivial weights.
    ///
    /// Block setup: d=1.0, all scales=1 (using kmask encoding), hmask=0xFF (all high bits set,
    /// so no subtraction), qs all 0x55 (low2=1 for all shift positions).
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_q3k_matches_cpu() {
        use flare_loader::quantize::dequant_q3k_block;

        let mut raw = [0u8; 110];
        // d = 1.0 as f16 LE at bytes 108-109
        raw[108] = 0x00;
        raw[109] = 0x3C;
        // hmask[32] = 0xFF: all high bits set → sub=0 for all weights
        for b in raw[0..32].iter_mut() {
            *b = 0xFF;
        }
        // qs[64] at bytes 32..96 = 0x55 = 0b01010101:
        //   shift 0 → bits[1:0]=01=1, shift 2 → bits[3:2]=01=1,
        //   shift 4 → bits[5:4]=01=1, shift 6 → bits[7:6]=01=1
        for b in raw[32..96].iter_mut() {
            *b = 0x55;
        }
        // scales_raw[12] at bytes 96..108:
        //   set b[0..8]=0x21 so (0x21 & 0x0F)=1, b[8..12]=0x00 → scales[0..4]=1-32=-31
        //   For a simple non-zero test, use values that produce known output.
        //   Instead: set all scales_raw to encode scales[i]=1:
        //     scales[i] = (b[i] & 0x0F) | extra_bits - 32 = 1
        //     → raw_val = 33 = 0x21 for b[0..4], b[8..12] bits[7:4]=0
        for b in raw[96..104].iter_mut() {
            *b = 0x21; // b[0..7]: low nibble=1 → after extra bits=0: scales[0..7]=1-32=-31
        }
        // b[8..12] = 0x00 → extra bits for scales[0..7] = 0

        // CPU reference
        let mut cpu_out = [0.0f32; 256];
        dequant_q3k_block(&raw, &mut cpu_out);

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_q3k(&raw, 1);

        assert_eq!(result.len(), 256);
        for (j, (&gpu, &cpu)) in result.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-3,
                "q3k mismatch at [{j}]: gpu={gpu}, cpu={cpu}"
            );
        }
    }

    /// Verify GPU BF16 fused dequant+matvec against CPU reference.
    ///
    /// One row, 4 BF16 weights: 1.0, 2.0, 3.0, 4.0.
    /// BF16 bit representation (same as upper 16 bits of F32):
    ///   1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040, 4.0 = 0x4080
    /// Input vector: all 1.0. Expected dot product = 1+2+3+4 = 10.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_bf16_matches_cpu() {
        // 4 BF16 values: 1.0, 2.0, 3.0, 4.0 (LE u16)
        // BF16 1.0 = 0x3F80 → bytes [0x80, 0x3F]
        // BF16 2.0 = 0x4000 → bytes [0x00, 0x40]
        // BF16 3.0 = 0x4040 → bytes [0x40, 0x40]
        // BF16 4.0 = 0x4080 → bytes [0x80, 0x40]
        let raw: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
        let input = vec![1.0f32; 4];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        // num_rows=1, num_blocks_per_row=4 (num_cols), batch=1
        let result = backend.dequant_matvec_bf16(&raw, &input, 1, 4, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        assert!(
            (result[0] - 10.0).abs() < 1e-3,
            "dequant_matvec_bf16 mismatch: got {}, expected 10.0",
            result[0]
        );
    }

    /// Verify GPU BF16 fused dequant+matvec with negative values.
    ///
    /// One row, 2 BF16 weights: -1.0, -2.0.
    /// BF16 -1.0 = 0xBF80 → bytes [0x80, 0xBF]
    /// BF16 -2.0 = 0xC000 → bytes [0x00, 0xC0]
    /// Input: all 1.0. Expected: -1 + -2 = -3.
    ///
    /// Requires a GPU adapter; run with: `cargo test -p flarellm-gpu -- --ignored`
    #[test]
    #[ignore]
    fn test_dequant_matvec_bf16_negative_weights() {
        let raw: Vec<u8> = vec![0x80, 0xBF, 0x00, 0xC0];
        let input = vec![1.0f32; 2];

        let backend = pollster::block_on(WebGpuBackend::new()).expect("GPU backend unavailable");
        let result = backend.dequant_matvec_bf16(&raw, &input, 1, 2, 1);

        assert_eq!(result.len(), 1, "expected 1 output");
        assert!(
            (result[0] - (-3.0)).abs() < 1e-3,
            "dequant_matvec_bf16 negative mismatch: got {}, expected -3.0",
            result[0]
        );
    }
}

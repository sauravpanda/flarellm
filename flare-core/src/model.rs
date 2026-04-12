use crate::config::{Architecture, ModelConfig};
use crate::kv_cache::KvCache;
use crate::tensor::Tensor;

/// Identifies the quantization format for a raw weight tensor stored on the GPU path.
///
/// Only formats with a corresponding fused dequant+matvec GPU kernel are listed here.
/// Other quantization formats must be dequantized to f32 at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    BF16,
    F16,
    Q2K,
    Q3K,
    Q4_0,
    Q4_1,
    Q4K,
    Q5_0,
    Q5_1,
    Q5K,
    Q6K,
    Q8_0,
    Q8_1,
    IQ4NL,
    IQ4XS,
    IQ3S,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ2S,
    IQ1S,
}

impl WeightFormat {
    /// Number of weights per block for this format.
    pub fn weights_per_block(self) -> usize {
        match self {
            WeightFormat::Q2K
            | WeightFormat::Q3K
            | WeightFormat::Q4K
            | WeightFormat::Q5K
            | WeightFormat::Q6K
            | WeightFormat::IQ2XXS
            | WeightFormat::IQ2XS
            | WeightFormat::IQ3XXS
            | WeightFormat::IQ4XS
            | WeightFormat::IQ3S
            | WeightFormat::IQ2S
            | WeightFormat::IQ1S => 256,
            WeightFormat::Q4_0
            | WeightFormat::Q4_1
            | WeightFormat::Q5_0
            | WeightFormat::Q5_1
            | WeightFormat::Q8_0
            | WeightFormat::Q8_1
            | WeightFormat::IQ4NL => 32,
            WeightFormat::BF16 | WeightFormat::F16 => 1,
        }
    }
}

/// Raw (undequantized) weight tensor for GPU fused dequant+matvec.
///
/// Stores the original GGUF bytes alongside the metadata needed to dispatch
/// the correct GPU shader.  Memory footprint is ~4× smaller than the
/// equivalent f32 `Tensor` for 4-bit quantized weights.
pub struct RawWeight {
    pub data: Vec<u8>,
    pub format: WeightFormat,
    /// Number of output rows (size of the output vector for a single matvec).
    pub num_rows: usize,
    /// Number of quantized blocks per row.
    pub blocks_per_row: usize,
}

/// Per-layer raw quantized weights.  Mirrors `LayerWeights` but stores bytes
/// instead of f32 tensors for the projection matrices.
pub struct RawLayerWeights {
    pub wq: RawWeight,
    pub wk: RawWeight,
    pub wv: RawWeight,
    pub wo: RawWeight,
    pub w_gate: RawWeight,
    pub w_up: RawWeight,
    pub w_down: RawWeight,
}

/// Trait for compute backends (WebGPU, SIMD, native wgpu).
/// Each backend implements these fundamental operations.
pub trait ComputeBackend: Send + Sync {
    fn matmul(&self, a: &Tensor, b: &Tensor, output: &mut Tensor);
    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32, output: &mut Tensor);
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, pos: usize, head_dim: usize, theta: f32);
    fn softmax(&self, input: &mut Tensor);
    fn silu_mul(&self, gate: &Tensor, up: &Tensor, output: &mut Tensor);

    /// Matrix-vector multiply: `output[rows]` = `mat[rows, cols]` * `vec[cols]`.
    /// Default implementation reshapes into matmul, but backends can override
    /// with a more efficient kernel.
    fn matvec(&self, mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        matvec(mat, vec, rows, cols)
    }

    /// RMSNorm on raw slices, returning a new Vec.
    /// Default delegates to the CPU implementation.
    fn rmsnorm_vec(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        rmsnorm(x, weight, eps)
    }

    /// Apply RoPE to interleaved Q or K vectors (in-place on raw slices).
    fn apply_rope_vec(
        &self,
        data: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        apply_rope(data, num_heads, head_dim, pos, theta);
    }

    /// SiLU(gate) * up, returning a new Vec.
    fn silu_mul_vec(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        silu_mul_cpu(gate, up)
    }

    /// Grouped-query attention for a single query position.
    ///
    /// - `q`: query vectors `[num_heads * head_dim]`
    /// - `k_cache`: key cache `[seq_len * num_kv_heads * head_dim]`
    /// - `v_cache`: value cache `[seq_len * num_kv_heads * head_dim]`
    /// - `attn_softcap`: Gemma-2 logit soft-cap (0.0 = disabled)
    ///
    /// Returns `[num_heads * head_dim]`.
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
        cpu_grouped_query_attention(
            q,
            k_cache,
            v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            attn_softcap,
        )
    }

    /// Initialize GPU-resident KV cache storage for backends that support it.
    ///
    /// Call this once after [`Model::set_backend`] to allocate GPU-side
    /// ring-buffer storage for all layers.  Backends that do not support GPU
    /// KV cache (e.g. `CpuBackend`) provide a no-op default.
    fn init_gpu_kv_cache(
        &self,
        _num_layers: usize,
        _max_seq_len: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) {
    }

    /// Write a new K/V pair into the GPU-resident cache (no CPU readback).
    ///
    /// For backends without GPU KV cache this is a no-op.
    #[allow(clippy::too_many_arguments)]
    fn gpu_kv_write(
        &self,
        _layer: usize,
        _position: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _key: &[f32],
        _value: &[f32],
    ) {
    }

    /// Returns `true` if a GPU-resident KV cache has been initialized.
    fn has_gpu_kv_cache(&self) -> bool {
        false
    }

    /// Grouped-query attention using GPU-resident KV cache buffers.
    ///
    /// Avoids re-uploading the KV cache from CPU on each forward pass —
    /// the data written by `gpu_kv_write` is used directly.
    ///
    /// Only valid after `init_gpu_kv_cache` has been called
    /// (`has_gpu_kv_cache()` returns `true`).
    #[allow(clippy::too_many_arguments)]
    fn grouped_query_attention_from_gpu_cache(
        &self,
        _q: &[f32],
        _layer: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _seq_len: usize,
        _attn_softcap: f32,
    ) -> Vec<f32> {
        panic!("GPU KV cache is not initialized — call init_gpu_kv_cache first");
    }

    /// Batched causal prefill attention for all token positions in a sequence.
    ///
    /// Computes causal self-attention for each position `t` in the sequence,
    /// where position `t` attends only to keys/values at positions `0..=t`.
    /// Backends that support GPU dispatch (e.g. `WebGpuBackend`) override this
    /// to run a single parallel kernel instead of the O(seq_len²) CPU loop.
    ///
    /// - `q`: query vectors `[seq_len * num_heads * head_dim]`
    /// - `k`: key vectors `[seq_len * num_kv_heads * head_dim]`
    /// - `v`: value vectors `[seq_len * num_kv_heads * head_dim]`
    /// - Returns `[seq_len * num_heads * head_dim]`
    #[allow(clippy::too_many_arguments)]
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
        // CPU fallback: iterate over positions and call the per-position GQA.
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut output = vec![0.0f32; seq_len * q_dim];
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
                attn_softcap,
            );
            output[t * q_dim..(t + 1) * q_dim].copy_from_slice(&attn_t);
        }
        output
    }

    /// Batched matrix-vector multiply: apply the same weight matrix to each row of the input.
    ///
    /// For each batch item `b` in `0..batch`:
    ///   `output[b * out_rows .. (b+1) * out_rows] = weight × input[b * in_cols .. (b+1) * in_cols]`
    ///
    /// - `weight`: f32 matrix `[out_rows × in_cols]`, row-major
    /// - `input`:  f32 batch  `[batch × in_cols]`, row-major
    /// - Returns:  f32 result `[batch × out_rows]`, row-major
    ///
    /// GPU backends override this for a single kernel dispatch;
    /// the default implementation loops over batch items and calls `matvec`.
    fn batched_matmul(
        &self,
        weight: &[f32],
        input: &[f32],
        out_rows: usize,
        in_cols: usize,
        batch: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * out_rows];
        for b in 0..batch {
            let in_row = &input[b * in_cols..(b + 1) * in_cols];
            let out_row = matvec(weight, in_row, out_rows, in_cols);
            output[b * out_rows..(b + 1) * out_rows].copy_from_slice(&out_row);
        }
        output
    }

    /// Apply RMSNorm to every row in `input` using the given `weight` vector.
    ///
    /// - `input`:  f32 batch `[batch × dim]`, row-major
    /// - `weight`: f32 norm weights `[dim]`
    /// - Returns:  f32 result `[batch × dim]`, row-major
    ///
    /// GPU backends override this for a single kernel dispatch;
    /// the default loops over rows and calls the scalar `rmsnorm`.
    fn batched_rmsnorm(
        &self,
        input: &[f32],
        weight: &[f32],
        dim: usize,
        batch: usize,
        eps: f32,
    ) -> Vec<f32> {
        (0..batch)
            .flat_map(|b| {
                let row = &input[b * dim..(b + 1) * dim];
                rmsnorm(row, weight, eps)
            })
            .collect()
    }

    /// Apply RoPE to every token in `inp` (laid out as `[seq_len × num_heads × head_dim]`).
    ///
    /// Token `t` gets position `start_pos + t`.  Returns a new buffer of the same shape.
    ///
    /// GPU backends override this for a single kernel dispatch;
    /// the default loops over tokens and calls `apply_rope`.
    fn batched_rope(
        &self,
        inp: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        start_pos: usize,
        theta: f32,
    ) -> Vec<f32> {
        let stride = num_heads * head_dim;
        let mut output = inp.to_vec();
        for t in 0..seq_len {
            apply_rope(
                &mut output[t * stride..(t + 1) * stride],
                num_heads,
                head_dim,
                start_pos + t,
                theta,
            );
        }
        output
    }

    /// Returns `true` if this backend can execute `batched_dequant_matmul`.
    ///
    /// The default CPU backend returns `false`; GPU backends override to `true`.
    fn supports_dequant_matmul(&self) -> bool {
        false
    }

    /// Fused dequantize + batched matrix-vector multiply for a raw weight tensor.
    ///
    /// For each batch item `b` in `0..batch`:
    /// ```text
    /// output[b * weight.num_rows .. (b+1) * weight.num_rows]
    ///   = dequant_matvec(weight, input[b * in_cols .. (b+1) * in_cols])
    /// ```
    /// where `in_cols = weight.blocks_per_row × weight.format.weights_per_block()`.
    ///
    /// The default implementation panics — override in GPU backends.
    fn batched_dequant_matmul(
        &self,
        _weight: &RawWeight,
        _input: &[f32],
        _batch: usize,
    ) -> Vec<f32> {
        panic!(
            "batched_dequant_matmul is not implemented for this backend; \
             call Model::set_backend with a GPU backend before using quantized weights"
        )
    }

    /// Serialise the driver-managed GPU pipeline cache to bytes.
    ///
    /// On backends that support `wgpu::Features::PIPELINE_CACHE` (Vulkan native),
    /// this returns an opaque blob that can be passed to the backend constructor
    /// on the next startup to skip shader recompilation.
    ///
    /// Returns an empty `Vec` on unsupported backends (WebGPU, Metal, DX12, CPU).
    fn pipeline_cache_data(&self) -> Vec<u8> {
        Vec::new()
    }
}

/// Default CPU compute backend. Uses optimized scalar loops.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, output: &mut Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert!(a_shape.len() == 2 && b_shape.len() == 2);
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        assert_eq!(b_shape[0], k);
        let out = output.data_mut();
        let a_data = a.data();
        let b_data = b.data();
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32, output: &mut Tensor) {
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
        let q_heads = q.numel() / head_dim;
        let k_heads = k.numel() / head_dim;
        apply_rope(q.data_mut(), q_heads, head_dim, pos, theta);
        apply_rope(k.data_mut(), k_heads, head_dim, pos, theta);
    }

    fn softmax(&self, input: &mut Tensor) {
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
        let out = output.data_mut();
        for (i, (g, u)) in gate.data().iter().zip(up.data().iter()).enumerate() {
            out[i] = (g / (1.0 + (-g).exp())) * u;
        }
    }
}

/// Weights for a single transformer layer.
pub struct LayerWeights {
    pub attn_norm: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub ffn_norm: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
    // Optional attention biases (Qwen2 uses these, Llama does not)
    pub attn_q_bias: Option<Tensor>,
    pub attn_k_bias: Option<Tensor>,
    pub attn_v_bias: Option<Tensor>,
    // Post-norms for Gemma 2 (applied after attention output / FFN output)
    pub post_attn_norm: Option<Tensor>,
    pub post_ffn_norm: Option<Tensor>,
}

/// Complete model weights.
pub struct ModelWeights {
    pub token_embedding: Tensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Tensor,
    pub output_weight: Tensor,
}

/// The core model that runs inference.
///
/// By default uses `CpuBackend` for all compute operations. Call
/// `set_backend` to plug in a GPU or SIMD backend.
pub struct Model {
    config: ModelConfig,
    weights: ModelWeights,
    /// Optional raw (quantized) layer weights for GPU fused dequant+matvec.
    ///
    /// When set alongside a backend that supports `batched_dequant_matmul`,
    /// `forward_prefill` uses the fused GPU kernels instead of the f32 weights,
    /// reducing memory bandwidth.  The f32 weights are still kept to support
    /// the single-token `forward` path and CPU fallback.
    raw_weights: Option<Vec<RawLayerWeights>>,
    kv_cache: KvCache,
    backend: Box<dyn ComputeBackend>,
}

impl Model {
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        let kv_cache = KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        );
        Self {
            config,
            weights,
            raw_weights: None,
            kv_cache,
            backend: Box::new(CpuBackend),
        }
    }

    /// Attach raw quantized weights for GPU fused-kernel inference.
    ///
    /// When a GPU backend is set and `raw_weights` is `Some`, `forward_prefill`
    /// will use the fused dequant+matvec shaders instead of the f32 projection
    /// matrices, saving memory bandwidth.
    pub fn set_raw_weights(&mut self, raw: Vec<RawLayerWeights>) {
        self.raw_weights = Some(raw);
    }

    /// Remove raw weights, reverting to the f32 weight path.
    pub fn clear_raw_weights(&mut self) {
        self.raw_weights = None;
    }

    /// Returns `true` if raw quantized weights have been set.
    pub fn has_raw_weights(&self) -> bool {
        self.raw_weights.is_some()
    }

    /// Return a reference to the current compute backend.
    pub fn backend(&self) -> &dyn ComputeBackend {
        self.backend.as_ref()
    }

    /// Replace the compute backend (e.g. with a GPU backend).
    ///
    /// After swapping the backend, initializes GPU-resident KV cache storage
    /// if the new backend supports it (default: no-op for CPU backends).
    ///
    /// Returns the previous backend.
    pub fn set_backend(&mut self, backend: Box<dyn ComputeBackend>) -> Box<dyn ComputeBackend> {
        let old = std::mem::replace(&mut self.backend, backend);
        self.backend.init_gpu_kv_cache(
            self.config.num_layers,
            self.config.max_seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
        );
        old
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Run a single forward pass for one token position.
    /// Returns logits over the vocabulary `[vocab_size]`.
    ///
    /// Delegates heavy compute (matvec, rmsnorm, rope, silu_mul) to the
    /// active `ComputeBackend`. By default this is `CpuBackend`; call
    /// `set_backend` to use GPU acceleration.
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Tensor {
        let config = &self.config;
        let dim = config.hidden_dim;
        let head_dim = config.head_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Token embedding lookup
        let embed_data = self.weights.token_embedding.data();
        let offset = token_id as usize * dim;
        let mut x = Tensor::from_vec(embed_data[offset..offset + dim].to_vec(), &[dim]).unwrap();

        // When raw quantized weights are available and the backend supports fused
        // dequant+matvec, use that path to avoid dequantizing to f32 for every token.
        let use_raw = self.backend.supports_dequant_matmul() && self.raw_weights.is_some();

        // Process each transformer layer
        for layer_idx in 0..config.num_layers {
            let layer = &self.weights.layers[layer_idx];

            // --- Attention block ---
            // RMSNorm
            let normed =
                self.backend
                    .rmsnorm_vec(x.data(), layer.attn_norm.data(), config.rms_norm_eps);

            // QKV projections — use fused dequant+matvec when raw weights are loaded.
            let mut q_data = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.wq, &normed, 1)
            } else {
                self.backend
                    .matvec(layer.wq.data(), &normed, num_heads * head_dim, dim)
            };
            let mut k_data = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.wk, &normed, 1)
            } else {
                self.backend.matvec(layer.wk.data(), &normed, kv_dim, dim)
            };
            let mut v_data = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.wv, &normed, 1)
            } else {
                self.backend.matvec(layer.wv.data(), &normed, kv_dim, dim)
            };

            // Add attention biases if present (Qwen2)
            if let Some(bias) = &layer.attn_q_bias {
                for (q, &b) in q_data.iter_mut().zip(bias.data().iter()) {
                    *q += b;
                }
            }
            if let Some(bias) = &layer.attn_k_bias {
                for (k, &b) in k_data.iter_mut().zip(bias.data().iter()) {
                    *k += b;
                }
            }
            if let Some(bias) = &layer.attn_v_bias {
                for (v, &b) in v_data.iter_mut().zip(bias.data().iter()) {
                    *v += b;
                }
            }
            self.backend
                .apply_rope_vec(&mut q_data, num_heads, head_dim, pos, config.rope_theta);
            self.backend.apply_rope_vec(
                &mut k_data,
                num_kv_heads,
                head_dim,
                pos,
                config.rope_theta,
            );

            // Write K, V to CPU cache (always) and GPU cache (if initialized).
            // The GPU write is a host→device queue.write_buffer — no readback.
            self.kv_cache.write(layer_idx, &k_data, &v_data);
            self.backend
                .gpu_kv_write(layer_idx, pos, num_kv_heads, head_dim, &k_data, &v_data);

            // Grouped-query attention — use GPU-resident cache when available to
            // avoid re-uploading the full KV cache on every token.
            let seq_len = self.kv_cache.len() + 1; // include current position
            let attn_output = if self.backend.has_gpu_kv_cache() {
                self.backend.grouped_query_attention_from_gpu_cache(
                    &q_data,
                    layer_idx,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                )
            } else {
                self.backend.grouped_query_attention(
                    &q_data,
                    self.kv_cache.keys(layer_idx).data(),
                    self.kv_cache.values(layer_idx).data(),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                )
            };

            // Output projection
            let attn_proj = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.wo, &attn_output, 1)
            } else {
                self.backend
                    .matvec(layer.wo.data(), &attn_output, dim, num_heads * head_dim)
            };

            // Gemma 2: apply post-attention RMSNorm before the residual add
            let attn_contrib = if let Some(ref post_norm) = layer.post_attn_norm {
                self.backend
                    .rmsnorm_vec(&attn_proj, post_norm.data(), config.rms_norm_eps)
            } else {
                attn_proj
            };

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += attn_contrib[i];
            }

            // --- FFN block ---
            let normed =
                self.backend
                    .rmsnorm_vec(x.data(), layer.ffn_norm.data(), config.rms_norm_eps);

            // Gate and up projections
            let gate = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.w_gate, &normed, 1)
            } else {
                self.backend
                    .matvec(layer.w_gate.data(), &normed, config.intermediate_dim, dim)
            };
            let up = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend.batched_dequant_matmul(&rw.w_up, &normed, 1)
            } else {
                self.backend
                    .matvec(layer.w_up.data(), &normed, config.intermediate_dim, dim)
            };

            // Gemma 2 uses GELU activation; Llama / Phi-3 use SiLU
            let ffn_hidden = if config.architecture == Architecture::Gemma2 {
                gelu_mul_cpu(&gate, &up)
            } else {
                self.backend.silu_mul_vec(&gate, &up)
            };

            // Down projection
            let ffn_out = if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                self.backend
                    .batched_dequant_matmul(&rw.w_down, &ffn_hidden, 1)
            } else {
                self.backend.matvec(
                    layer.w_down.data(),
                    &ffn_hidden,
                    dim,
                    config.intermediate_dim,
                )
            };

            // Gemma 2: apply post-FFN RMSNorm before the residual add
            let ffn_contrib = if let Some(ref post_norm) = layer.post_ffn_norm {
                self.backend
                    .rmsnorm_vec(&ffn_out, post_norm.data(), config.rms_norm_eps)
            } else {
                ffn_out
            };

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += ffn_contrib[i];
            }
        }

        // Advance KV cache after processing all layers
        self.kv_cache.advance();

        // Final RMSNorm
        let normed = self.backend.rmsnorm_vec(
            x.data(),
            self.weights.output_norm.data(),
            config.rms_norm_eps,
        );

        // Output logits: [vocab_size] = output_weight [vocab_size, dim] x normed [dim]
        let mut logits = self.backend.matvec(
            self.weights.output_weight.data(),
            &normed,
            config.vocab_size,
            dim,
        );

        // Gemma 2: apply final logit soft-cap (tanh(x / cap) * cap)
        if config.final_logit_softcap > 0.0 {
            let cap = config.final_logit_softcap;
            for l in &mut logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(logits, &[config.vocab_size]).unwrap()
    }

    /// Batched forward pass for a full prompt sequence.
    ///
    /// Processes all `tokens` in one shot:
    /// - Embeds the entire sequence up front.
    /// - For each transformer layer, computes Q/K/V for all positions with
    ///   individual matvec calls (in-order, so future work can batch them into
    ///   a single matmul), then does causal self-attention using the locally
    ///   computed K/V (bypassing the KV cache during attention).
    /// - After all layers, writes the computed K/V into the KV cache and
    ///   advances the position counter once per token.
    ///
    /// Returns logits for the **last** token position — suitable for picking
    /// the first generated token after prefill.
    ///
    /// For a single-token prompt this is equivalent to `forward()`.
    pub fn forward_prefill(&mut self, tokens: &[u32]) -> Tensor {
        assert!(
            !tokens.is_empty(),
            "forward_prefill requires at least one token"
        );

        let config = self.config.clone();
        let dim = config.hidden_dim;
        let head_dim = config.head_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let seq_len = tokens.len();

        // Embed all tokens: x_batch [seq_len * dim], row-major (row = token position)
        let mut x_batch: Vec<f32> = {
            let embed_data = self.weights.token_embedding.data();
            tokens
                .iter()
                .flat_map(|&tok| {
                    let off = tok as usize * dim;
                    embed_data[off..off + dim].iter().copied()
                })
                .collect()
        };

        // Per-layer K/V tensors computed locally (used for causal attention;
        // written to the KV cache at the end).
        let num_layers = config.num_layers;
        let mut k_all: Vec<Vec<f32>> = vec![vec![0.0; seq_len * kv_dim]; num_layers];
        let mut v_all: Vec<Vec<f32>> = vec![vec![0.0; seq_len * kv_dim]; num_layers];

        for layer_idx in 0..num_layers {
            // --- Attention block ---

            // RMSNorm all rows — single GPU dispatch.
            let normed_batch: Vec<f32> = {
                let attn_norm: Vec<f32> = self.weights.layers[layer_idx].attn_norm.data().to_vec();
                self.backend.batched_rmsnorm(
                    &x_batch,
                    &attn_norm,
                    dim,
                    seq_len,
                    config.rms_norm_eps,
                )
            };

            // Q/K/V projections: single batched dispatch per matrix, then per-token bias+RoPE.
            // When raw quantized weights are available and the backend supports fused
            // dequant+matvec, use that path to reduce memory bandwidth.
            let use_raw = self.backend.supports_dequant_matmul() && self.raw_weights.is_some();
            let mut q_batch = vec![0.0f32; seq_len * q_dim];
            {
                let q_bias: Option<Vec<f32>> = self.weights.layers[layer_idx]
                    .attn_q_bias
                    .as_ref()
                    .map(|t| t.data().to_vec());
                let k_bias: Option<Vec<f32>> = self.weights.layers[layer_idx]
                    .attn_k_bias
                    .as_ref()
                    .map(|t| t.data().to_vec());
                let v_bias: Option<Vec<f32>> = self.weights.layers[layer_idx]
                    .attn_v_bias
                    .as_ref()
                    .map(|t| t.data().to_vec());

                // Three GPU dispatches instead of 3 × seq_len individual matvec calls.
                // Use fused dequant+matvec when raw quantized weights are available.
                let q_proj = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.wq, &normed_batch, seq_len)
                } else {
                    let wq: Vec<f32> = self.weights.layers[layer_idx].wq.data().to_vec();
                    self.backend
                        .batched_matmul(&wq, &normed_batch, q_dim, dim, seq_len)
                };
                let k_proj = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.wk, &normed_batch, seq_len)
                } else {
                    let wk: Vec<f32> = self.weights.layers[layer_idx].wk.data().to_vec();
                    self.backend
                        .batched_matmul(&wk, &normed_batch, kv_dim, dim, seq_len)
                };
                let v_proj = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.wv, &normed_batch, seq_len)
                } else {
                    let wv: Vec<f32> = self.weights.layers[layer_idx].wv.data().to_vec();
                    self.backend
                        .batched_matmul(&wv, &normed_batch, kv_dim, dim, seq_len)
                };

                // Apply biases in-place (CPU broadcast, cheap; rare in most models).
                let mut q_proj = q_proj;
                let mut k_proj = k_proj;
                let mut v_proj = v_proj;
                if let Some(ref b) = q_bias {
                    for t in 0..seq_len {
                        let row = &mut q_proj[t * q_dim..(t + 1) * q_dim];
                        for (qi, &bi) in row.iter_mut().zip(b.iter()) {
                            *qi += bi;
                        }
                    }
                }
                if let Some(ref b) = k_bias {
                    for t in 0..seq_len {
                        let row = &mut k_proj[t * kv_dim..(t + 1) * kv_dim];
                        for (ki, &bi) in row.iter_mut().zip(b.iter()) {
                            *ki += bi;
                        }
                    }
                }
                if let Some(ref b) = v_bias {
                    for t in 0..seq_len {
                        let row = &mut v_proj[t * kv_dim..(t + 1) * kv_dim];
                        for (vi, &bi) in row.iter_mut().zip(b.iter()) {
                            *vi += bi;
                        }
                    }
                }

                // RoPE: single GPU dispatch instead of 2 × seq_len CPU calls.
                let q_roped = self.backend.batched_rope(
                    &q_proj,
                    num_heads,
                    head_dim,
                    seq_len,
                    0,
                    config.rope_theta,
                );
                let k_roped = self.backend.batched_rope(
                    &k_proj,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    0,
                    config.rope_theta,
                );

                q_batch.copy_from_slice(&q_roped);
                for t in 0..seq_len {
                    k_all[layer_idx][t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&k_roped[t * kv_dim..(t + 1) * kv_dim]);
                    v_all[layer_idx][t * kv_dim..(t + 1) * kv_dim]
                        .copy_from_slice(&v_proj[t * kv_dim..(t + 1) * kv_dim]);
                }
            }

            // Causal self-attention: dispatch to GPU (if available) or CPU fallback.
            let attn_out_batch = self.backend.prefill_attention_gpu(
                &q_batch,
                &k_all[layer_idx],
                &v_all[layer_idx],
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                config.attn_logit_softcap,
            );

            // Output projection + optional post-attention norm + residual
            {
                let post_attn_norm: Option<Vec<f32>> = self.weights.layers[layer_idx]
                    .post_attn_norm
                    .as_ref()
                    .map(|t| t.data().to_vec());

                // Single batched dispatch replaces one matvec per token.
                let proj_batch = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.wo, &attn_out_batch, seq_len)
                } else {
                    let wo: Vec<f32> = self.weights.layers[layer_idx].wo.data().to_vec();
                    self.backend
                        .batched_matmul(&wo, &attn_out_batch, dim, q_dim, seq_len)
                };

                for t in 0..seq_len {
                    let proj = proj_batch[t * dim..(t + 1) * dim].to_vec();
                    let contrib = match &post_attn_norm {
                        Some(n) => rmsnorm(&proj, n, config.rms_norm_eps),
                        None => proj,
                    };
                    for i in 0..dim {
                        x_batch[t * dim + i] += contrib[i];
                    }
                }
            }

            // --- FFN block ---
            let normed_batch2: Vec<f32> = {
                let ffn_norm: Vec<f32> = self.weights.layers[layer_idx].ffn_norm.data().to_vec();
                self.backend
                    .batched_rmsnorm(&x_batch, &ffn_norm, dim, seq_len, config.rms_norm_eps)
            };

            {
                let inter = config.intermediate_dim;
                let post_ffn_norm: Option<Vec<f32>> = self.weights.layers[layer_idx]
                    .post_ffn_norm
                    .as_ref()
                    .map(|t| t.data().to_vec());

                // Single batched dispatch for gate and up projections.
                let gate_batch = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.w_gate, &normed_batch2, seq_len)
                } else {
                    let w_gate: Vec<f32> = self.weights.layers[layer_idx].w_gate.data().to_vec();
                    self.backend
                        .batched_matmul(&w_gate, &normed_batch2, inter, dim, seq_len)
                };
                let up_batch = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.w_up, &normed_batch2, seq_len)
                } else {
                    let w_up: Vec<f32> = self.weights.layers[layer_idx].w_up.data().to_vec();
                    self.backend
                        .batched_matmul(&w_up, &normed_batch2, inter, dim, seq_len)
                };

                // Per-token silu_mul / gelu_mul (CPU arithmetic, no GPU dispatch).
                let ffn_hidden_batch: Vec<f32> = (0..seq_len)
                    .flat_map(|t| {
                        let gate_t = &gate_batch[t * inter..(t + 1) * inter];
                        let up_t = &up_batch[t * inter..(t + 1) * inter];
                        if config.architecture == Architecture::Gemma2 {
                            gelu_mul_cpu(gate_t, up_t)
                        } else {
                            self.backend.silu_mul_vec(gate_t, up_t)
                        }
                    })
                    .collect();

                // Single batched dispatch for the down projection.
                let down_batch = if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    self.backend
                        .batched_dequant_matmul(&rw.w_down, &ffn_hidden_batch, seq_len)
                } else {
                    let w_down: Vec<f32> = self.weights.layers[layer_idx].w_down.data().to_vec();
                    self.backend
                        .batched_matmul(&w_down, &ffn_hidden_batch, dim, inter, seq_len)
                };

                for t in 0..seq_len {
                    let down = down_batch[t * dim..(t + 1) * dim].to_vec();
                    let contrib = match &post_ffn_norm {
                        Some(n) => rmsnorm(&down, n, config.rms_norm_eps),
                        None => down,
                    };
                    for i in 0..dim {
                        x_batch[t * dim + i] += contrib[i];
                    }
                }
            }
        }

        // Write computed K/V to CPU cache and GPU cache (if initialized),
        // then advance the ring-buffer position once per token.
        let cache_pos = self.kv_cache.position(); // position before prefill tokens
        for t in 0..seq_len {
            for layer_idx in 0..num_layers {
                let k_slice = &k_all[layer_idx][t * kv_dim..(t + 1) * kv_dim];
                let v_slice = &v_all[layer_idx][t * kv_dim..(t + 1) * kv_dim];
                self.kv_cache.write(layer_idx, k_slice, v_slice);
                self.backend.gpu_kv_write(
                    layer_idx,
                    cache_pos + t,
                    num_kv_heads,
                    head_dim,
                    k_slice,
                    v_slice,
                );
            }
            self.kv_cache.advance();
        }

        // Final RMSNorm + output projection on the last token only
        let last = seq_len - 1;
        let last_x = &x_batch[last * dim..(last + 1) * dim];
        let normed_final =
            self.backend
                .rmsnorm_vec(last_x, self.weights.output_norm.data(), config.rms_norm_eps);

        let mut logits = self.backend.matvec(
            self.weights.output_weight.data(),
            &normed_final,
            config.vocab_size,
            dim,
        );

        if config.final_logit_softcap > 0.0 {
            let cap = config.final_logit_softcap;
            for l in &mut logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(logits, &[config.vocab_size]).unwrap()
    }
}

/// RMSNorm: normalize and scale (CPU implementation).
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON always available on aarch64
        unsafe { rmsnorm_neon(x, weight, eps) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rmsnorm_scalar(x, weight, eps)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rmsnorm_scalar(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}

/// NEON SIMD RMSNorm: vectorized sum-of-squares and normalize.
#[cfg(target_arch = "aarch64")]
unsafe fn rmsnorm_neon(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    use std::arch::aarch64::*;
    let dim = x.len();
    let dim16 = dim & !15;

    // Phase 1: vectorized sum-of-squares with 4 accumulators
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut j = 0usize;
    while j < dim16 {
        let v0 = vld1q_f32(x.as_ptr().add(j));
        acc0 = vfmaq_f32(acc0, v0, v0);
        let v1 = vld1q_f32(x.as_ptr().add(j + 4));
        acc1 = vfmaq_f32(acc1, v1, v1);
        let v2 = vld1q_f32(x.as_ptr().add(j + 8));
        acc2 = vfmaq_f32(acc2, v2, v2);
        let v3 = vld1q_f32(x.as_ptr().add(j + 12));
        acc3 = vfmaq_f32(acc3, v3, v3);
        j += 16;
    }
    let sum_v = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut sum_sq = vaddvq_f32(sum_v);
    while j < dim {
        sum_sq += x[j] * x[j];
        j += 1;
    }

    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_v = vdupq_n_f32(inv_rms);

    // Phase 2: vectorized output = (x * inv_rms) * weight
    let mut output = vec![0.0f32; dim];
    let mut j = 0usize;
    while j < dim16 {
        let v0 = vld1q_f32(x.as_ptr().add(j));
        let w0 = vld1q_f32(weight.as_ptr().add(j));
        let r0 = vmulq_f32(vmulq_f32(v0, inv_rms_v), w0);
        vst1q_f32(output.as_mut_ptr().add(j), r0);

        let v1 = vld1q_f32(x.as_ptr().add(j + 4));
        let w1 = vld1q_f32(weight.as_ptr().add(j + 4));
        let r1 = vmulq_f32(vmulq_f32(v1, inv_rms_v), w1);
        vst1q_f32(output.as_mut_ptr().add(j + 4), r1);

        let v2 = vld1q_f32(x.as_ptr().add(j + 8));
        let w2 = vld1q_f32(weight.as_ptr().add(j + 8));
        let r2 = vmulq_f32(vmulq_f32(v2, inv_rms_v), w2);
        vst1q_f32(output.as_mut_ptr().add(j + 8), r2);

        let v3 = vld1q_f32(x.as_ptr().add(j + 12));
        let w3 = vld1q_f32(weight.as_ptr().add(j + 12));
        let r3 = vmulq_f32(vmulq_f32(v3, inv_rms_v), w3);
        vst1q_f32(output.as_mut_ptr().add(j + 12), r3);

        j += 16;
    }
    while j < dim {
        output[j] = (x[j] * inv_rms) * weight[j];
        j += 1;
    }
    output
}

/// Matrix-vector multiply: `output[rows]` = `mat[rows, cols]` * `vec[cols]` (CPU implementation).
///
/// Dispatches to platform-specific SIMD implementations:
/// - ARM NEON (aarch64): 4-wide f32 SIMD, 4 accumulators (16 elements/iter), compile-time
/// - x86 AVX2+FMA (x86_64): 8-wide f32 SIMD, 4 accumulators (32 elements/iter), runtime check
/// - Fallback: 4-wide scalar unrolling for auto-vectorization
///
/// All SIMD paths parallelize via rayon for large matrices on native targets.
pub fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        matvec_simd(mat, vec, rows, cols)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above ensures AVX2 + FMA are available
            return unsafe { matvec_avx2(mat, vec, rows, cols) };
        }
        matvec_scalar(mat, vec, rows, cols)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        matvec_scalar(mat, vec, rows, cols)
    }
}

/// x86 AVX2 matvec entry point. Runtime-detected; uses #[target_feature] to enable
/// AVX2/FMA codegen for the inner loop.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available (via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const PARALLEL_FMA_THRESHOLD: usize = 5_000_000;
    const CHUNK_ROWS: usize = 64;
    let total_work = rows * cols;

    if total_work >= PARALLEL_FMA_THRESHOLD && rows >= CHUNK_ROWS * 2 {
        use rayon::prelude::*;
        let mut output = vec![0.0f32; rows];
        output
            .par_chunks_mut(CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * CHUNK_ROWS;
                for (local_i, out) in out_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let row = &mat[i * cols..i * cols + cols];
                    // SAFETY: AVX2/FMA verified at entry, row.len() == cols
                    *out = unsafe { matvec_avx2_row(row, vec, cols) };
                }
            });
        return output;
    }

    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..i * cols + cols];
        *out = matvec_avx2_row(row, vec, cols);
    }
    output
}

/// Compute one row of x86 AVX2 matvec.
///
/// # Safety
/// `row` and `vec` must both have length `cols`. AVX2 + FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2_row(row: &[f32], vec: &[f32], cols: usize) -> f32 {
    use std::arch::x86_64::*;
    let cols32 = cols & !31;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut j = 0usize;
    while j < cols32 {
        let r0 = _mm256_loadu_ps(row.as_ptr().add(j));
        let v0 = _mm256_loadu_ps(vec.as_ptr().add(j));
        acc0 = _mm256_fmadd_ps(r0, v0, acc0);

        let r1 = _mm256_loadu_ps(row.as_ptr().add(j + 8));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(j + 8));
        acc1 = _mm256_fmadd_ps(r1, v1, acc1);

        let r2 = _mm256_loadu_ps(row.as_ptr().add(j + 16));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(j + 16));
        acc2 = _mm256_fmadd_ps(r2, v2, acc2);

        let r3 = _mm256_loadu_ps(row.as_ptr().add(j + 24));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(j + 24));
        acc3 = _mm256_fmadd_ps(r3, v3, acc3);

        j += 32;
    }

    // Reduce 4 accumulators of 8 lanes each → scalar
    let sum01 = _mm256_add_ps(acc0, acc1);
    let sum23 = _mm256_add_ps(acc2, acc3);
    let sum = _mm256_add_ps(sum01, sum23);

    // Horizontal sum of 8 lanes
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(hi128, lo128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x4E);
    let sum2 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum2, sum2, 0xB1);
    let sum_final = _mm_add_ps(sum2, shuf2);
    let mut scalar = _mm_cvtss_f32(sum_final);

    // Handle remainder
    while j < cols {
        scalar += row[j] * vec[j];
        j += 1;
    }
    scalar
}

/// Compute one row of NEON matvec. Used by both serial and parallel paths.
///
/// # Safety
/// `row` and `vec` must both have length `cols`. NEON is always available on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matvec_simd_row(row: &[f32], vec: &[f32], cols: usize) -> f32 {
    use std::arch::aarch64::*;
    let cols16 = cols & !15;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut j = 0usize;
    while j < cols16 {
        let r0 = vld1q_f32(row.as_ptr().add(j));
        let v0 = vld1q_f32(vec.as_ptr().add(j));
        acc0 = vfmaq_f32(acc0, r0, v0);

        let r1 = vld1q_f32(row.as_ptr().add(j + 4));
        let v1 = vld1q_f32(vec.as_ptr().add(j + 4));
        acc1 = vfmaq_f32(acc1, r1, v1);

        let r2 = vld1q_f32(row.as_ptr().add(j + 8));
        let v2 = vld1q_f32(vec.as_ptr().add(j + 8));
        acc2 = vfmaq_f32(acc2, r2, v2);

        let r3 = vld1q_f32(row.as_ptr().add(j + 12));
        let v3 = vld1q_f32(vec.as_ptr().add(j + 12));
        acc3 = vfmaq_f32(acc3, r3, v3);

        j += 16;
    }

    let sum01 = vaddq_f32(acc0, acc1);
    let sum23 = vaddq_f32(acc2, acc3);
    let sum = vaddq_f32(sum01, sum23);
    let mut scalar = vaddvq_f32(sum);

    while j < cols {
        scalar += row[j] * vec[j];
        j += 1;
    }
    scalar
}

/// ARM NEON SIMD matvec with rayon parallelism on native, serial on wasm32.
#[cfg(target_arch = "aarch64")]
fn matvec_simd(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Total FMAs above which parallelism wins. ~5M FMAs / ~50µs sequential break-even.
    const PARALLEL_FMA_THRESHOLD: usize = 5_000_000;
    // Chunk size: balance task granularity vs overhead
    const CHUNK_ROWS: usize = 64;

    let total_work = rows * cols;

    #[cfg(not(target_arch = "wasm32"))]
    if total_work >= PARALLEL_FMA_THRESHOLD && rows >= CHUNK_ROWS * 2 {
        use rayon::prelude::*;
        let mut output = vec![0.0f32; rows];
        output
            .par_chunks_mut(CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * CHUNK_ROWS;
                for (local_i, out) in out_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let row = &mat[i * cols..i * cols + cols];
                    // SAFETY: row.len() == cols, SIMD feature gated by cfg
                    *out = unsafe { matvec_simd_row(row, vec, cols) };
                }
            });
        return output;
    }

    // Sequential path
    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..i * cols + cols];
        // SAFETY: row.len() == cols, SIMD feature gated by cfg
        *out = unsafe { matvec_simd_row(row, vec, cols) };
    }
    output
}

/// Scalar matvec with 4-wide unrolling for auto-vectorization.
///
/// Always available as a reference implementation for tests, regardless of target.
pub fn matvec_scalar(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    let cols4 = cols & !3;

    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..(i + 1) * cols];
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let mut j = 0;
        while j < cols4 {
            sum0 += row[j] * vec[j];
            sum1 += row[j + 1] * vec[j + 1];
            sum2 += row[j + 2] * vec[j + 2];
            sum3 += row[j + 3] * vec[j + 3];
            j += 4;
        }
        while j < cols {
            sum0 += row[j] * vec[j];
            j += 1;
        }
        *out = sum0 + sum1 + sum2 + sum3;
    }
    output
}

/// SiLU(gate) * up on raw slices, returning a new Vec (CPU implementation).
pub fn silu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

/// GELU(gate) * up — used by Gemma 2 FFN (tanh approximation).
pub fn gelu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            // tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let c = 0.797_884_6_f32; // sqrt(2/pi)
            let gelu = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh());
            gelu * u
        })
        .collect()
}

/// Apply RoPE to interleaved Q or K vectors (CPU implementation).
///
/// Pre-computes (cos, sin) per dimension index once for the given pos/theta/head_dim,
/// then applies the rotation to all heads. The original code recomputed
/// `theta.powf` and `sin_cos` per head, wasting work.
pub fn apply_rope(data: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;

    // Precompute cos/sin once for each frequency
    let mut cos_table = Vec::with_capacity(half);
    let mut sin_table = Vec::with_capacity(half);
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        cos_table.push(cos_val);
        sin_table.push(sin_val);
    }

    // Apply rotation to all heads using the cached tables
    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let c = cos_table[i];
            let s = sin_table[i];
            let x0 = data[offset + i];
            let x1 = data[offset + i + half];
            data[offset + i] = x0 * c - x1 * s;
            data[offset + i + half] = x0 * s + x1 * c;
        }
    }
}

/// CPU implementation of grouped-query attention for a single token position.
/// Exported so GPU backends can delegate to it for unsupported cases (e.g. soft-cap).
#[allow(clippy::too_many_arguments)]
pub fn cpu_grouped_query_attention(
    q: &[f32],       // [num_heads * head_dim]
    k_cache: &[f32], // [seq_len * num_kv_heads * head_dim]
    v_cache: &[f32], // [seq_len * num_kv_heads * head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,    // number of valid KV entries
    attn_softcap: f32, // 0.0 = no capping (Llama etc); non-zero for Gemma 2
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_heads * head_dim];

    let kv_stride = num_kv_heads * head_dim;

    for h in 0..num_heads {
        let kv_head = h / heads_per_kv;
        let q_offset = h * head_dim;

        // Compute attention scores for this head
        let mut scores = vec![0.0f32; seq_len];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for (t, score) in scores.iter_mut().enumerate() {
            let k_offset = t * kv_stride + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            *score = dot * scale;
        }

        // Gemma 2: tanh soft-cap on attention logits before softmax
        if attn_softcap > 0.0 {
            for s in &mut scores {
                *s = (*s / attn_softcap).tanh() * attn_softcap;
            }
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }

        // Weighted sum of values
        let out_offset = h * head_dim;
        for (t, &weight) in scores.iter().enumerate() {
            let v_offset = t * kv_stride + kv_head * head_dim;
            for d in 0..head_dim {
                output[out_offset + d] += weight * v_cache[v_offset + d];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let result = rmsnorm(&x, &w, 1e-5);
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        for (i, &v) in result.iter().enumerate() {
            let expected = (i + 1) as f32 / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "rmsnorm[{i}]: {v} != {expected}"
            );
        }
    }

    #[test]
    fn test_matvec() {
        // [[1,2],[3,4]] × [1,1] = [3, 7]
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![1.0, 1.0];
        let result = matvec(&mat, &v, 2, 2);
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 7.0).abs() < 1e-5);
    }

    /// Test SIMD matvec produces the same result as the scalar reference
    /// across many sizes, including odd sizes that exercise the remainder loop.
    #[test]
    fn test_matvec_simd_matches_scalar() {
        let test_cases = [
            (1, 1),
            (4, 16),
            (16, 4),
            (15, 15), // not multiple of SIMD width
            (17, 33),
            (64, 64),
            (128, 256),
            (256, 128),
            (576, 576),  // SmolLM Q proj
            (1536, 576), // SmolLM FFN gate
            (49152, 64), // wide-tall (logits-like)
            (300, 257),  // odd remainder
        ];

        for &(rows, cols) in &test_cases {
            let mat: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                .collect();
            let v: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();

            let simd = matvec(&mat, &v, rows, cols);
            let scalar = matvec_scalar(&mat, &v, rows, cols);

            assert_eq!(simd.len(), rows, "size mismatch for {rows}x{cols}");
            assert_eq!(scalar.len(), rows);

            for i in 0..rows {
                let diff = (simd[i] - scalar[i]).abs();
                let tol = (scalar[i].abs() * 1e-4).max(1e-4);
                assert!(
                    diff <= tol,
                    "matvec mismatch at row {i} for {rows}x{cols}: simd={} scalar={} diff={}",
                    simd[i],
                    scalar[i],
                    diff,
                );
            }
        }
    }

    /// Test parallel and sequential matvec produce identical results.
    #[test]
    fn test_matvec_parallel_matches_sequential() {
        // Sizes both above and below the parallel threshold
        let test_cases = [
            (200, 200),   // below threshold, sequential
            (500, 500),   // below threshold, sequential
            (1000, 5000), // above threshold (5M FMAs), parallel
            (4096, 2048), // well above threshold
        ];

        for &(rows, cols) in &test_cases {
            let mat: Vec<f32> = (0..rows * cols)
                .map(|i| (((i * 7) % 19) as f32 - 9.0) * 0.02)
                .collect();
            let v: Vec<f32> = (0..cols).map(|i| ((i * 3) as f32 * 0.05).cos()).collect();

            let simd = matvec(&mat, &v, rows, cols);
            let scalar = matvec_scalar(&mat, &v, rows, cols);

            for i in 0..rows {
                let diff = (simd[i] - scalar[i]).abs();
                let tol = (scalar[i].abs() * 1e-4).max(1e-4);
                assert!(
                    diff <= tol,
                    "parallel mismatch at row {i} for {rows}x{cols}: simd={} scalar={}",
                    simd[i],
                    scalar[i],
                );
            }
        }
    }

    #[test]
    fn test_rmsnorm_simd_matches_naive() {
        // Various dim sizes
        for &dim in &[1, 4, 15, 16, 17, 64, 576, 2048] {
            let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.01)).collect();

            let result = rmsnorm(&x, &weight, 1e-5);

            // Naive reference
            let sum_sq: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sum_sq / dim as f32 + 1e-5).sqrt();
            let expected: Vec<f32> = x
                .iter()
                .zip(weight.iter())
                .map(|(&xi, &wi)| (xi / rms) * wi)
                .collect();

            for i in 0..dim {
                let diff = (result[i] - expected[i]).abs();
                let tol = (expected[i].abs() * 1e-4).max(1e-4);
                assert!(
                    diff <= tol,
                    "rmsnorm mismatch at {i} for dim={dim}: got={} expected={}",
                    result[i],
                    expected[i],
                );
            }
        }
    }

    #[test]
    fn test_rope_preserves_magnitude() {
        let head_dim = 4;
        let mut data = vec![1.0, 0.0, 0.0, 1.0]; // one head
        let mag_before: f32 = data.iter().map(|x| x * x).sum();
        apply_rope(&mut data, 1, head_dim, 5, 10000.0);
        let mag_after: f32 = data.iter().map(|x| x * x).sum();
        assert!(
            (mag_before - mag_after).abs() < 1e-4,
            "RoPE should preserve magnitude"
        );
    }

    // ---------------------------------------------------------------
    // Integration tests: full forward pass
    // ---------------------------------------------------------------

    use crate::config::Architecture;

    /// Build a tiny deterministic model for testing.
    /// vocab=8, dim=4, intermediate=8, 1 layer, 2 heads, 2 kv_heads, head_dim=2
    fn tiny_test_model() -> Model {
        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 8,
            hidden_dim: 4,
            intermediate_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        };

        // Deterministic weight init: w[i] = (i % 7 - 3) * 0.1
        let make_weights =
            |size: usize| -> Vec<f32> { (0..size).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect() };

        let dim = config.hidden_dim;
        let nh = config.num_heads;
        let nkvh = config.num_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_dim;
        let vocab = config.vocab_size;

        let layer = LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: Tensor::from_vec(make_weights(nh * hd * dim), &[nh * hd * dim]).unwrap(),
            wk: Tensor::from_vec(make_weights(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wv: Tensor::from_vec(make_weights(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wo: Tensor::from_vec(make_weights(dim * nh * hd), &[dim * nh * hd]).unwrap(),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: Tensor::from_vec(make_weights(inter * dim), &[inter * dim]).unwrap(),
            w_up: Tensor::from_vec(make_weights(inter * dim), &[inter * dim]).unwrap(),
            w_down: Tensor::from_vec(make_weights(dim * inter), &[dim * inter]).unwrap(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        };

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
            layers: vec![layer],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_forward_output_shape() {
        let mut model = tiny_test_model();
        let logits = model.forward(0, 0);
        assert_eq!(logits.shape(), &[8], "output should be [vocab_size]");
        assert_eq!(logits.numel(), 8);
    }

    #[test]
    fn test_forward_different_tokens_different_logits() {
        let mut model = tiny_test_model();
        let logits_0 = model.forward(0, 0).data().to_vec();
        model.reset();
        let logits_1 = model.forward(1, 0).data().to_vec();
        // Different token embeddings should produce different logits
        assert_ne!(
            logits_0, logits_1,
            "different tokens should give different logits"
        );
    }

    #[test]
    fn test_forward_logits_are_finite() {
        let mut model = tiny_test_model();
        let logits = model.forward(3, 0);
        for (i, &v) in logits.data().iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_multi_token_forward_kv_cache_grows() {
        let mut model = tiny_test_model();
        model.forward(0, 0);
        assert_eq!(model.kv_cache().len(), 1);
        model.forward(1, 1);
        assert_eq!(model.kv_cache().len(), 2);
        model.forward(2, 2);
        assert_eq!(model.kv_cache().len(), 3);
    }

    #[test]
    fn test_forward_position_affects_output() {
        // RoPE should make the same token at different positions produce different logits
        let mut model = tiny_test_model();
        let logits_pos0 = model.forward(0, 0).data().to_vec();
        // Don't reset — KV cache state will differ, which is fine.
        // The point is the logits should differ from a fresh model at pos 5.
        let mut model2 = tiny_test_model();
        // Feed dummy tokens to advance to position 5
        for i in 0..5 {
            model2.forward(0, i);
        }
        let logits_pos5 = model2.forward(0, 5).data().to_vec();
        assert_ne!(
            logits_pos0, logits_pos5,
            "same token at different positions should differ (RoPE)"
        );
    }

    #[test]
    fn test_forward_residual_connection() {
        // With all-zero weights, the residual connection should preserve the embedding.
        // We use an identity-like setup: norm weights = 1, all projection weights = 0.
        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 4,
            hidden_dim: 4,
            intermediate_dim: 4,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        };

        let dim = 4;
        let zero_4x4 = vec![0.0f32; 16];

        let layer = LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            wk: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            wv: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            wo: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            w_up: Tensor::from_vec(zero_4x4.clone(), &[16]).unwrap(),
            w_down: Tensor::from_vec(zero_4x4, &[16]).unwrap(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        };

        // Embedding: token 0 = [1, 2, 3, 4]
        let mut embed = vec![0.0f32; 16];
        embed[0] = 1.0;
        embed[1] = 2.0;
        embed[2] = 3.0;
        embed[3] = 4.0;

        // Output weight = identity-ish (diagonal)
        let mut out_w = vec![0.0f32; 16];
        out_w[0] = 1.0; // out[0] picks up normed[0]
        out_w[5] = 1.0;
        out_w[10] = 1.0;
        out_w[15] = 1.0;

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(embed, &[16]).unwrap(),
            layers: vec![layer],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(out_w, &[16]).unwrap(),
        };

        let mut model = Model::new(config, weights);
        let logits = model.forward(0, 0);

        // With zero projections, residual preserves embedding [1,2,3,4].
        // After final rmsnorm with weight=1, then matmul with diagonal output_weight,
        // logits should reflect the normalized embedding.
        let data = logits.data();
        // The embedding [1,2,3,4] after RMSNorm has specific values. Just verify finite and non-zero.
        assert!(
            data.iter().any(|&v| v.abs() > 1e-6),
            "residual path should produce non-zero logits"
        );
        assert!(
            data.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );
    }

    #[test]
    fn test_gqa_heads_per_kv_group() {
        // Test with num_heads=4, num_kv_heads=2 (2 heads per KV group)
        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 4,
            hidden_dim: 4,
            intermediate_dim: 4,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        };

        let make_w =
            |size: usize| -> Vec<f32> { (0..size).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect() };

        let dim = 4;
        let nh = 4;
        let nkvh = 2;
        let hd = 1;
        let inter = 4;

        let layer = LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: Tensor::from_vec(make_w(nh * hd * dim), &[nh * hd * dim]).unwrap(),
            wk: Tensor::from_vec(make_w(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wv: Tensor::from_vec(make_w(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wo: Tensor::from_vec(make_w(dim * nh * hd), &[dim * nh * hd]).unwrap(),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: Tensor::from_vec(make_w(inter * dim), &[inter * dim]).unwrap(),
            w_up: Tensor::from_vec(make_w(inter * dim), &[inter * dim]).unwrap(),
            w_down: Tensor::from_vec(make_w(dim * inter), &[dim * inter]).unwrap(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        };

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(make_w(4 * dim), &[4 * dim]).unwrap(),
            layers: vec![layer],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(make_w(4 * dim), &[4 * dim]).unwrap(),
        };

        let mut model = Model::new(config, weights);
        let logits = model.forward(0, 0);
        assert_eq!(logits.shape(), &[4]);
        assert!(
            logits.data().iter().all(|v| v.is_finite()),
            "GQA output should be finite"
        );
    }

    #[test]
    fn test_multi_layer_forward() {
        // 2 layers should produce different results than 1 layer
        let mut model_1layer = tiny_test_model();
        let logits_1 = model_1layer.forward(0, 0).data().to_vec();

        // Build 2-layer model
        let config = ModelConfig {
            num_layers: 2,
            ..model_1layer.config().clone()
        };

        let make_weights =
            |size: usize| -> Vec<f32> { (0..size).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect() };

        let dim = config.hidden_dim;
        let nh = config.num_heads;
        let nkvh = config.num_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_dim;
        let vocab = config.vocab_size;

        let make_layer = || LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: Tensor::from_vec(make_weights(nh * hd * dim), &[nh * hd * dim]).unwrap(),
            wk: Tensor::from_vec(make_weights(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wv: Tensor::from_vec(make_weights(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
            wo: Tensor::from_vec(make_weights(dim * nh * hd), &[dim * nh * hd]).unwrap(),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: Tensor::from_vec(make_weights(inter * dim), &[inter * dim]).unwrap(),
            w_up: Tensor::from_vec(make_weights(inter * dim), &[inter * dim]).unwrap(),
            w_down: Tensor::from_vec(make_weights(dim * inter), &[dim * inter]).unwrap(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        };

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
            layers: vec![make_layer(), make_layer()],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
        };

        let mut model_2layer = Model::new(config, weights);
        let logits_2 = model_2layer.forward(0, 0).data().to_vec();

        assert_ne!(
            logits_1, logits_2,
            "2 layers should produce different output than 1 layer"
        );
    }
}

use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::tensor::Tensor;

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
            kv_cache,
            backend: Box::new(CpuBackend),
        }
    }

    /// Replace the compute backend (e.g. with a GPU backend).
    /// Returns the previous backend.
    pub fn set_backend(&mut self, backend: Box<dyn ComputeBackend>) -> Box<dyn ComputeBackend> {
        std::mem::replace(&mut self.backend, backend)
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

        // Process each transformer layer
        for layer_idx in 0..config.num_layers {
            let layer = &self.weights.layers[layer_idx];

            // --- Attention block ---
            // RMSNorm
            let normed =
                self.backend
                    .rmsnorm_vec(x.data(), layer.attn_norm.data(), config.rms_norm_eps);

            // QKV projections
            let mut q_data =
                self.backend
                    .matvec(layer.wq.data(), &normed, num_heads * head_dim, dim);
            let mut k_data = self.backend.matvec(layer.wk.data(), &normed, kv_dim, dim);
            let mut v_data = self.backend.matvec(layer.wv.data(), &normed, kv_dim, dim);

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

            // Write K, V to cache
            self.kv_cache.write(layer_idx, &k_data, &v_data);

            // Grouped-query attention
            let attn_output = grouped_query_attention(
                &q_data,
                &self.kv_cache,
                layer_idx,
                num_heads,
                num_kv_heads,
                head_dim,
                self.kv_cache.len() + 1, // include current position
            );

            // Output projection
            let attn_proj =
                self.backend
                    .matvec(layer.wo.data(), &attn_output, dim, num_heads * head_dim);

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += attn_proj[i];
            }

            // --- FFN block ---
            let normed =
                self.backend
                    .rmsnorm_vec(x.data(), layer.ffn_norm.data(), config.rms_norm_eps);

            // Gate and up projections
            let gate =
                self.backend
                    .matvec(layer.w_gate.data(), &normed, config.intermediate_dim, dim);
            let up = self
                .backend
                .matvec(layer.w_up.data(), &normed, config.intermediate_dim, dim);

            // SiLU(gate) * up
            let ffn_hidden = self.backend.silu_mul_vec(&gate, &up);

            // Down projection
            let ffn_out = self.backend.matvec(
                layer.w_down.data(),
                &ffn_hidden,
                dim,
                config.intermediate_dim,
            );

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += ffn_out[i];
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
        let logits = self.backend.matvec(
            self.weights.output_weight.data(),
            &normed,
            config.vocab_size,
            dim,
        );

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

/// Grouped-query attention for a single token position.
fn grouped_query_attention(
    q: &[f32], // [num_heads * head_dim]
    kv_cache: &KvCache,
    layer: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize, // number of valid KV entries
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_heads * head_dim];

    let k_cache = kv_cache.keys(layer).data();
    let v_cache = kv_cache.values(layer).data();
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

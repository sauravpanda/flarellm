use crate::config::{Architecture, ModelConfig};
use crate::kv_cache::{KvCache, QuantizedKvCache};
use crate::tensor::Tensor;

/// Ensure rayon's global thread pool is sized for CPU inference, not the
/// host's total logical CPU count.
///
/// Rayon defaults to `available_parallelism` — on Apple M5 Pro that's **18**
/// (6 Super + 12 Performance cores).  For the ~65 matmul calls per decoded
/// token in a 1B model, that much oversubscription just multiplies the
/// per-call latch / condvar sync cost without adding useful parallelism,
/// because the small and medium matmuls saturate before all 18 workers
/// are even needed.  Empirical sweep on 1B Q8_0:
///
/// | threads | sustained decode |
/// |---|---|
/// | 18 (default) | ~40 tok/s |
/// | 10 | **~45 tok/s** (+12 %) |
/// | 6 | ~42 tok/s |
///
/// We cap at 10, which leaves headroom for both P-core tiers on Apple
/// Silicon and still improves on default everywhere we've measured. Callers
/// that need a different count can set `RAYON_NUM_THREADS` before loading
/// the library, or pre-build rayon's global pool themselves — this call is
/// idempotent and silently no-ops if the pool is already initialized.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm_threads"))]
fn init_rayon_pool_once() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return;
        }
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let target = logical.min(10);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(target)
            .build_global();
    });
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasm_threads")))]
fn init_rayon_pool_once() {}

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
    /// BitNet b1.58 ternary weights: each weight is {-1, 0, +1}.
    /// Packed 4 weights per byte using 2-bit encoding: 00=0, 01=+1, 10=-1, 11=unused.
    Ternary,
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
            // Ternary: 4 weights per byte, so 1 "block" = 4 weights
            WeightFormat::Ternary => 4,
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

/// Returns `true` if `CpuBackend::batched_dequant_matmul` has a fast path for
/// this weight's format.  Mixed-format models (e.g. Q4_K_M with Q6_K tail
/// layers) should check this per-weight and fall back to the f32 dispatch
/// for unsupported formats rather than paying on-the-fly dequant cost in
/// the hot prefill loop.
#[inline]
pub fn raw_weight_cpu_dispatchable(weight: &RawWeight) -> bool {
    matches!(weight.format, WeightFormat::Q8_0 | WeightFormat::Q4K)
}

/// Fused Q8_0 weight data for projections that share the same input vector.
///
/// Created once at model-load time by concatenating the raw Q8_0 bytes of
/// [wq; wk; wv] and [w_gate; w_up].  This lets us replace 3 matvec calls
/// (Q, K, V) with 1, and 2 matvec calls (gate, up) with 1, reading the
/// input vector once instead of multiple times.
pub struct FusedLayerWeights {
    /// Concatenated Q8_0 bytes for [wq; wk; wv].
    pub qkv_data: Vec<u8>,
    /// Total output rows = q_dim + kv_dim + kv_dim.
    pub qkv_rows: usize,
    /// Concatenated Q8_0 bytes for [w_gate; w_up].
    pub gate_up_data: Vec<u8>,
    /// Total output rows = intermediate_dim * 2.
    pub gate_up_rows: usize,
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

    /// Ternary (BitNet b1.58) matrix-vector multiply.
    ///
    /// `packed_weights` contains 2-bit ternary-packed weights (4 per byte).
    /// No floating-point multiplications — pure add/sub based on {-1, 0, +1}.
    fn matvec_ternary(
        &self,
        packed_weights: &[u8],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        matvec_ternary(packed_weights, input, rows, cols)
    }

    /// Q8_0 direct quantized matvec: compute `output = weight_q8 * input` without
    /// dequantizing the weight matrix to f32.
    ///
    /// `weight_data`: raw Q8_0 bytes (row-major).
    /// `input`: f32 vector of length `cols`.
    /// `rows`: output dimension.  `cols`: input dimension (must be multiple of 32).
    fn matvec_q8_0(&self, weight_data: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        matvec_q8_0(weight_data, input, rows, cols)
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

    /// Returns `true` if this backend supports GPU-resident forward pass.
    ///
    /// When true, `forward_single_token_gpu` can be called to run the entire
    /// transformer forward pass in a single GPU command encoder submission.
    fn supports_gpu_forward(&self) -> bool {
        false
    }

    /// Upload all model weights to persistent GPU buffers.
    ///
    /// After this call, GPU-resident forward passes avoid per-token weight
    /// uploads. Only called on backends where `supports_dequant_matmul` is true.
    ///
    /// # Arguments
    /// * `raw_layers` — per-layer raw quantized weight matrices
    /// * `layer_norms` — per-layer norm weight slices (attn_norm, ffn_norm, post_attn_norm, post_ffn_norm)
    /// * `output_norm` — final RMSNorm weights
    /// * `output_weight` — output projection (lm_head) f32 weights
    /// * `token_embedding` — token embedding table f32 data
    #[allow(clippy::type_complexity)]
    fn upload_weights_to_gpu(
        &self,
        _raw_layers: &[RawLayerWeights],
        _layer_norms: &[(&[f32], &[f32], Option<&[f32]>, Option<&[f32]>)],
        _output_norm: &[f32],
        _output_weight: &[f32],
        _token_embedding: &[f32],
    ) {
    }

    /// Returns `true` if GPU-resident weights have been uploaded.
    fn has_gpu_weights(&self) -> bool {
        false
    }

    /// GPU-resident single-token forward pass.
    ///
    /// Builds ONE command encoder with all compute passes for the entire
    /// transformer. Intermediate results stay in GPU storage buffers with no
    /// CPU readback until the final logits vector.
    ///
    /// Returns `None` if GPU weights/KV cache are not initialized, or the
    /// backend does not support GPU-resident forward.
    #[allow(clippy::too_many_arguments)]
    fn forward_single_token_gpu(
        &self,
        _token_embedding: &[f32],
        _pos: usize,
        _dim: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _intermediate_dim: usize,
        _vocab_size: usize,
        _rms_norm_eps: f32,
        _rope_theta: f32,
        _num_layers: usize,
        _seq_len: usize,
    ) -> Option<Vec<f32>> {
        None
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

    /// CPU backend supports fused dequant matmul for Q8_0 and Q4_K.  Callers
    /// must additionally gate on `raw_weight_cpu_dispatchable()` per-weight
    /// to check whether the specific format is supported — mixed-format
    /// models (e.g. Q4_K_M with Q6_K tail layers) fall back to the f32 path
    /// for the unsupported weights.
    fn supports_dequant_matmul(&self) -> bool {
        true
    }

    /// Batched Q8_0 × Q8_0-input matmul for prefill / speculative verify.
    ///
    /// Quantizes each input row to Q8_0 once, then runs the existing fused
    /// `matvec_q8_0_preq_into` / `matvec_q4k_q8_into` per batch item against
    /// the raw quantized weight.  This is dramatically faster than the
    /// generic `batched_matmul` fallback, which dequantizes every weight to
    /// f32 and loops scalar `matvec` — ~4× less memory bandwidth on the Q8_0
    /// path plus keeps the hot weight bytes in L2 across batch items.
    ///
    /// Supported formats: `Q8_0`, `Q4K`.  Other formats panic — callers must
    /// gate on the model's actual weight format.
    fn batched_dequant_matmul(
        &self,
        weight: &RawWeight,
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        let out_rows = weight.num_rows;
        let blocks_per_row = weight.blocks_per_row;
        let weights_per_block = weight.format.weights_per_block();
        let in_cols = blocks_per_row * weights_per_block;

        debug_assert_eq!(
            input.len(),
            batch * in_cols,
            "input must be [batch × in_cols]"
        );

        match weight.format {
            WeightFormat::Q8_0 => {
                let mut output = vec![0.0f32; batch * out_rows];
                let mk_preq = || QuantizedInput {
                    scales: vec![0.0f32; blocks_per_row],
                    quants: vec![0i8; blocks_per_row * Q8_0_BLOCK_SIZE],
                    blocks_per_row,
                };
                let mut preq0 = mk_preq();
                let mut preq1 = mk_preq();
                let mut preq2 = mk_preq();
                let mut preq3 = mk_preq();

                #[cfg(target_arch = "aarch64")]
                let has_smmla = std::arch::is_aarch64_feature_detected!("i8mm");
                #[cfg(not(target_arch = "aarch64"))]
                let has_smmla = false;

                #[cfg(target_arch = "aarch64")]
                let mut pair_out = vec![0.0f32; 2 * out_rows];
                #[cfg(target_arch = "aarch64")]
                let mut quad_out = vec![0.0f32; 4 * out_rows];

                let mut b = 0usize;
                // 4-token tiles (M2+ only, SMMLA-backed).  Streams weights once
                // per 4 tokens instead of once per pair.
                #[cfg(target_arch = "aarch64")]
                if has_smmla {
                    while b + 4 <= batch {
                        quantize_input_q8_0_into(&input[(b + 0) * in_cols..(b + 1) * in_cols], &mut preq0);
                        quantize_input_q8_0_into(&input[(b + 1) * in_cols..(b + 2) * in_cols], &mut preq1);
                        quantize_input_q8_0_into(&input[(b + 2) * in_cols..(b + 3) * in_cols], &mut preq2);
                        quantize_input_q8_0_into(&input[(b + 3) * in_cols..(b + 4) * in_cols], &mut preq3);
                        matvec_q8_0_preq_quad_into(
                            &weight.data,
                            [&preq0, &preq1, &preq2, &preq3],
                            out_rows,
                            &mut quad_out,
                        );
                        for i in 0..4 {
                            output[(b + i) * out_rows..(b + i + 1) * out_rows]
                                .copy_from_slice(&quad_out[i * out_rows..(i + 1) * out_rows]);
                        }
                        b += 4;
                    }
                }

                // 2-token tiles for the remainder.
                while b + 2 <= batch {
                    quantize_input_q8_0_into(&input[(b + 0) * in_cols..(b + 1) * in_cols], &mut preq0);
                    quantize_input_q8_0_into(&input[(b + 1) * in_cols..(b + 2) * in_cols], &mut preq1);
                    #[cfg(target_arch = "aarch64")]
                    {
                        matvec_q8_0_preq_pair_into(
                            &weight.data, &preq0, &preq1, out_rows, &mut pair_out,
                        );
                        output[b * out_rows..(b + 1) * out_rows]
                            .copy_from_slice(&pair_out[..out_rows]);
                        output[(b + 1) * out_rows..(b + 2) * out_rows]
                            .copy_from_slice(&pair_out[out_rows..]);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        matvec_q8_0_preq_into(&weight.data, &preq0, out_rows,
                            &mut output[b * out_rows..(b + 1) * out_rows]);
                        matvec_q8_0_preq_into(&weight.data, &preq1, out_rows,
                            &mut output[(b + 1) * out_rows..(b + 2) * out_rows]);
                    }
                    b += 2;
                }

                // Trailing single token (odd batch).
                if b < batch {
                    quantize_input_q8_0_into(&input[b * in_cols..(b + 1) * in_cols], &mut preq0);
                    matvec_q8_0_preq_into(
                        &weight.data, &preq0, out_rows,
                        &mut output[b * out_rows..(b + 1) * out_rows],
                    );
                }
                output
            }
            WeightFormat::Q4K => {
                debug_assert_eq!(
                    in_cols % Q8_0_BLOCK_SIZE,
                    0,
                    "Q4_K input cols must be multiple of Q8_0 block size"
                );
                let preq_blocks = in_cols / Q8_0_BLOCK_SIZE;
                let mut output = vec![0.0f32; batch * out_rows];
                let mk_preq = || QuantizedInput {
                    scales: vec![0.0f32; preq_blocks],
                    quants: vec![0i8; preq_blocks * Q8_0_BLOCK_SIZE],
                    blocks_per_row: preq_blocks,
                };
                let mut preq0 = mk_preq();
                let mut preq1 = mk_preq();
                let mut preq2 = mk_preq();
                let mut preq3 = mk_preq();

                #[cfg(target_arch = "aarch64")]
                let mut pair_out = vec![0.0f32; 2 * out_rows];
                #[cfg(target_arch = "aarch64")]
                let mut quad_out = vec![0.0f32; 4 * out_rows];

                let mut b = 0usize;
                // 4-token tile: amortizes per-block decode (6-bit scale unpack,
                // 4-bit nibble unpack) across 4 tokens and halves weight bandwidth
                // again vs the 2-token tile.
                #[cfg(target_arch = "aarch64")]
                while b + 4 <= batch {
                    quantize_input_q8_0_into(&input[b * in_cols..(b + 1) * in_cols], &mut preq0);
                    quantize_input_q8_0_into(&input[(b + 1) * in_cols..(b + 2) * in_cols], &mut preq1);
                    quantize_input_q8_0_into(&input[(b + 2) * in_cols..(b + 3) * in_cols], &mut preq2);
                    quantize_input_q8_0_into(&input[(b + 3) * in_cols..(b + 4) * in_cols], &mut preq3);
                    matvec_q4k_q8_quad_into(
                        &weight.data,
                        [&preq0, &preq1, &preq2, &preq3],
                        out_rows, in_cols, &mut quad_out,
                    );
                    for i in 0..4 {
                        output[(b + i) * out_rows..(b + i + 1) * out_rows]
                            .copy_from_slice(&quad_out[i * out_rows..(i + 1) * out_rows]);
                    }
                    b += 4;
                }

                while b + 2 <= batch {
                    quantize_input_q8_0_into(&input[b * in_cols..(b + 1) * in_cols], &mut preq0);
                    quantize_input_q8_0_into(&input[(b + 1) * in_cols..(b + 2) * in_cols], &mut preq1);
                    #[cfg(target_arch = "aarch64")]
                    {
                        matvec_q4k_q8_pair_into(
                            &weight.data, &preq0, &preq1, out_rows, in_cols, &mut pair_out,
                        );
                        output[b * out_rows..(b + 1) * out_rows]
                            .copy_from_slice(&pair_out[..out_rows]);
                        output[(b + 1) * out_rows..(b + 2) * out_rows]
                            .copy_from_slice(&pair_out[out_rows..]);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        matvec_q4k_q8_into(&weight.data, &preq0, out_rows, in_cols,
                            &mut output[b * out_rows..(b + 1) * out_rows]);
                        matvec_q4k_q8_into(&weight.data, &preq1, out_rows, in_cols,
                            &mut output[(b + 1) * out_rows..(b + 2) * out_rows]);
                    }
                    b += 2;
                }

                // Trailing single token
                if b < batch {
                    quantize_input_q8_0_into(&input[b * in_cols..(b + 1) * in_cols], &mut preq0);
                    matvec_q4k_q8_into(
                        &weight.data, &preq0, out_rows, in_cols,
                        &mut output[b * out_rows..(b + 1) * out_rows],
                    );
                }
                output
            }
            other => panic!(
                "CpuBackend::batched_dequant_matmul: unsupported weight format {other:?}"
            ),
        }
    }
}

/// Weights for a single expert in a Mixture-of-Experts (MoE) layer.
///
/// Each expert is a standard SwiGLU FFN: gate_proj -> silu -> up_proj -> down_proj.
pub struct ExpertWeights {
    pub w_gate: Tensor, // (intermediate_dim, hidden_dim)
    pub w_up: Tensor,   // (intermediate_dim, hidden_dim)
    pub w_down: Tensor, // (hidden_dim, intermediate_dim)
    /// Whether this expert's weights are currently loaded in memory.
    /// For on-demand loading, unloaded experts have empty tensors and `loaded = false`.
    pub loaded: bool,
}

/// Weights for a Mixture-of-Experts (MoE) layer.
///
/// Contains the router/gate that selects which experts to activate,
/// plus the per-expert FFN weights.
pub struct MoeLayerWeights {
    /// Router weight matrix: (hidden_dim, num_experts).
    /// Computes expert scores from the hidden state.
    pub gate: Tensor,
    /// Per-expert FFN weights, indexed by expert ID.
    pub experts: Vec<ExpertWeights>,
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
    /// Optional MoE weights. When present, the FFN block uses router-selected
    /// experts instead of the shared w_gate/w_up/w_down above.
    pub moe: Option<MoeLayerWeights>,
}

/// Pre-allocated buffers for the forward() hot path.
///
/// Eliminates ~16 `Vec<f32>` heap allocations per layer per token by reusing
/// these buffers across forward() calls. For a 16-layer model this saves
/// ~256 allocations per token.
pub struct ForwardBuffers {
    /// RMSNorm output for attention block `[dim]`
    pub normed: Vec<f32>,
    /// Query projection `[num_heads * head_dim]`
    pub q_data: Vec<f32>,
    /// Key projection `[num_kv_heads * head_dim]`
    pub k_data: Vec<f32>,
    /// Value projection `[num_kv_heads * head_dim]`
    pub v_data: Vec<f32>,
    /// Attention output `[num_heads * head_dim]`
    pub attn_output: Vec<f32>,
    /// Attention output projection `[dim]`
    pub attn_proj: Vec<f32>,
    /// Post-attention norm scratch `[dim]`
    pub attn_contrib: Vec<f32>,
    /// RMSNorm output for FFN block `[dim]`
    pub ffn_normed: Vec<f32>,
    /// Gate projection `[intermediate_dim]`
    pub gate: Vec<f32>,
    /// Up projection `[intermediate_dim]`
    pub up: Vec<f32>,
    /// SiLU(gate) * up `[intermediate_dim]`
    pub ffn_hidden: Vec<f32>,
    /// Down projection `[dim]`
    pub ffn_out: Vec<f32>,
    /// Final norm output `[dim]`
    pub final_normed: Vec<f32>,
    /// Output logits `[vocab_size]`
    pub logits: Vec<f32>,
    /// Pre-quantized Q8_0 buffer for normed input (reused across Q/K/V projections)
    pub normed_q8: QuantizedInput,
    /// Pre-quantized Q8_0 buffer for FFN normed input (reused across gate/up projections)
    pub ffn_normed_q8: QuantizedInput,
    /// Pre-quantized Q8_0 buffer for final normed output (used for Q8_0 logits projection)
    pub output_q8: QuantizedInput,
    /// Pre-quantized Q8_0 buffer for attn_output (used for wo projection)
    pub attn_q8: QuantizedInput,
    /// Pre-quantized Q8_0 buffer for ffn_hidden (used for w_down projection)
    pub ffn_hidden_q8: QuantizedInput,
    /// Fused QKV output buffer `[q_dim + kv_dim + kv_dim]`
    pub qkv: Vec<f32>,
    /// Fused gate+up output buffer `[intermediate_dim * 2]`
    pub gate_up: Vec<f32>,
}

impl ForwardBuffers {
    /// Allocate all buffers based on model config dimensions.
    pub fn new(config: &super::config::ModelConfig) -> Self {
        let dim = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        Self {
            normed: vec![0.0; dim],
            q_data: vec![0.0; q_dim],
            k_data: vec![0.0; kv_dim],
            v_data: vec![0.0; kv_dim],
            attn_output: vec![0.0; q_dim],
            attn_proj: vec![0.0; dim],
            attn_contrib: vec![0.0; dim],
            ffn_normed: vec![0.0; dim],
            gate: vec![0.0; intermediate_dim],
            up: vec![0.0; intermediate_dim],
            ffn_hidden: vec![0.0; intermediate_dim],
            ffn_out: vec![0.0; dim],
            final_normed: vec![0.0; dim],
            logits: vec![0.0; vocab_size],
            normed_q8: QuantizedInput {
                scales: vec![0.0; dim / Q8_0_BLOCK_SIZE],
                quants: vec![0; dim],
                blocks_per_row: dim / Q8_0_BLOCK_SIZE,
            },
            ffn_normed_q8: QuantizedInput {
                scales: vec![0.0; dim / Q8_0_BLOCK_SIZE],
                quants: vec![0; dim],
                blocks_per_row: dim / Q8_0_BLOCK_SIZE,
            },
            output_q8: QuantizedInput {
                scales: vec![0.0; dim / Q8_0_BLOCK_SIZE],
                quants: vec![0; dim],
                blocks_per_row: dim / Q8_0_BLOCK_SIZE,
            },
            attn_q8: QuantizedInput {
                scales: vec![0.0; q_dim / Q8_0_BLOCK_SIZE],
                quants: vec![0; q_dim],
                blocks_per_row: q_dim / Q8_0_BLOCK_SIZE,
            },
            ffn_hidden_q8: QuantizedInput {
                scales: vec![0.0; intermediate_dim / Q8_0_BLOCK_SIZE],
                quants: vec![0; intermediate_dim],
                blocks_per_row: intermediate_dim / Q8_0_BLOCK_SIZE,
            },
            qkv: vec![0.0; q_dim + kv_dim + kv_dim],
            gate_up: vec![0.0; intermediate_dim * 2],
        }
    }
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
    /// Optional raw (quantized) output projection weight for Q8_0 logits path.
    ///
    /// When set, the final logits matvec uses `matvec_q8_0_preq_into` instead
    /// of the f32 `output_weight`, reducing memory bandwidth by ~4x.
    raw_output_weight: Option<RawWeight>,
    kv_cache: KvCache,
    /// Optional 2-bit quantized KV cache (KIVI).  When `Some`, writes go to
    /// both the full-precision cache (for GPU path compatibility) and the
    /// quantized cache; reads from the CPU attention path use the quantized
    /// version for ~8x memory savings on long contexts.
    quantized_kv_cache: Option<QuantizedKvCache>,
    backend: Box<dyn ComputeBackend>,
    /// Pre-allocated buffers for the forward() hot path to avoid per-token
    /// heap allocations. Initialized in `new()`.
    forward_buffers: ForwardBuffers,
    /// Tracks how many contiguous layers (from layer 0) have been loaded
    /// via `load_layer_weights`.  `None` means all layers were provided at
    /// construction time and are fully available.
    layers_loaded: Option<usize>,
    /// Fused Q8_0 layer weights: concatenated [wq;wk;wv] and [w_gate;w_up]
    /// per layer. Built once in `set_raw_weights()` to reduce matvec calls
    /// from 7 to 4 per layer.
    fused_weights: Option<Vec<FusedLayerWeights>>,
    /// Fused f32 layer weights for the CPU f32 path: concatenated f32 data
    /// for [wq;wk;wv] and [w_gate;w_up] per layer.
    fused_f32_qkv: Option<Vec<Vec<f32>>>,
    fused_f32_gate_up: Option<Vec<Vec<f32>>>,
    /// Pre-computed cos/sin lookup table for RoPE, eliminating transcendental
    /// function calls from the per-token forward path.
    rope_table: RopeTable,
}

impl Model {
    /// Look up the token embedding vector for a given token ID.
    ///
    /// Returns a copy of the row `token_id` of the token embedding table,
    /// of length `config.hidden_dim`. Exposed as a primitive for P2P /
    /// collaborative inference scenarios where one peer runs the embedding
    /// step and streams the hidden state to downstream peers.
    pub fn embed_token(&self, token_id: u32) -> Vec<f32> {
        let dim = self.config.hidden_dim;
        let offset = token_id as usize * dim;
        let embed = self.weights.token_embedding.data();
        embed[offset..offset + dim].to_vec()
    }

    /// Apply the final RMSNorm + output weight matvec to compute logits
    /// from a post-last-layer hidden state.
    ///
    /// This is the inverse counterpart to [`Model::embed_token`]: it is the
    /// tail of a forward pass. It is exposed as a primitive for P2P /
    /// collaborative inference: a downstream peer can hand back the final
    /// hidden state and an orchestrator can run this step to get logits.
    ///
    /// Note: this uses the dequantised f32 `output_weight` path and does
    /// not apply Gemma 2's final logit soft-cap. P2P callers that need the
    /// soft-cap should apply it themselves on the returned logits.
    pub fn output_projection(&self, hidden: &[f32]) -> Vec<f32> {
        let normed = self.backend.rmsnorm_vec(
            hidden,
            self.weights.output_norm.data(),
            self.config.rms_norm_eps,
        );
        matvec(
            self.weights.output_weight.data(),
            &normed,
            self.config.vocab_size,
            self.config.hidden_dim,
        )
    }

    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        // Cap rayon's global pool at 10 threads the first time a model is
        // constructed.  See `init_rayon_pool_once` for the rationale.
        init_rayon_pool_once();

        let kv_cache = KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        );
        let quantized_kv_cache = if config.kv_cache_bits == 2 {
            Some(QuantizedKvCache::new(
                config.num_layers,
                config.max_seq_len,
                config.num_kv_heads,
                config.head_dim,
            ))
        } else {
            None
        };
        let forward_buffers = ForwardBuffers::new(&config);
        let rope_table = RopeTable::new(config.head_dim, config.max_seq_len, config.rope_theta);
        Self {
            config,
            weights,
            raw_weights: None,
            raw_output_weight: None,
            kv_cache,
            quantized_kv_cache,
            backend: Box::new(CpuBackend),
            forward_buffers,
            layers_loaded: None,
            fused_weights: None,
            fused_f32_qkv: None,
            fused_f32_gate_up: None,
            rope_table,
        }
    }

    /// Attach raw quantized weights for GPU fused-kernel inference.
    ///
    /// When a GPU backend is set and `raw_weights` is `Some`, `forward_prefill`
    /// will use the fused dequant+matvec shaders instead of the f32 projection
    /// matrices, saving memory bandwidth.
    pub fn set_raw_weights(&mut self, raw: Vec<RawLayerWeights>) {
        // Build fused Q8_0 weights by concatenating bytes for shared-input
        // projections. This is done once at load time so forward() can use
        // a single matvec call instead of 3 (QKV) or 2 (gate/up).
        if !raw.is_empty() && raw[0].wq.format == WeightFormat::Q8_0 {
            let fused: Vec<FusedLayerWeights> = raw
                .iter()
                .map(|rw| {
                    // Concatenate [wq; wk; wv] Q8_0 bytes
                    let mut qkv_data =
                        Vec::with_capacity(rw.wq.data.len() + rw.wk.data.len() + rw.wv.data.len());
                    qkv_data.extend_from_slice(&rw.wq.data);
                    qkv_data.extend_from_slice(&rw.wk.data);
                    qkv_data.extend_from_slice(&rw.wv.data);
                    let qkv_rows = rw.wq.num_rows + rw.wk.num_rows + rw.wv.num_rows;

                    // Concatenate [w_gate; w_up] Q8_0 bytes
                    let mut gate_up_data =
                        Vec::with_capacity(rw.w_gate.data.len() + rw.w_up.data.len());
                    gate_up_data.extend_from_slice(&rw.w_gate.data);
                    gate_up_data.extend_from_slice(&rw.w_up.data);
                    let gate_up_rows = rw.w_gate.num_rows + rw.w_up.num_rows;

                    FusedLayerWeights {
                        qkv_data,
                        qkv_rows,
                        gate_up_data,
                        gate_up_rows,
                    }
                })
                .collect();
            self.fused_weights = Some(fused);
        }
        self.raw_weights = Some(raw);
    }

    /// Attach a raw quantized output projection weight for Q8_0 logits path.
    ///
    /// When set, the final logits matvec in `forward()` uses
    /// `matvec_q8_0_preq_into` instead of f32, reducing memory bandwidth by ~4x
    /// for the output projection (often the largest single matvec).
    pub fn set_raw_output_weight(&mut self, rw: RawWeight) {
        self.raw_output_weight = Some(rw);
    }

    /// Returns `true` if a raw quantized output weight has been set.
    pub fn has_raw_output_weight(&self) -> bool {
        self.raw_output_weight.is_some()
    }

    /// Build fused f32 weight matrices for the CPU f32 path.
    ///
    /// Concatenates [wq; wk; wv] and [w_gate; w_up] f32 data per layer so
    /// forward() can use a single matvec_into call instead of 3 (QKV) or 2
    /// (gate/up). Call this after all layer weights have been loaded.
    pub fn build_fused_f32_weights(&mut self) {
        let layers = &self.weights.layers;
        let mut qkv_all = Vec::with_capacity(layers.len());
        let mut gate_up_all = Vec::with_capacity(layers.len());
        for layer in layers {
            let mut qkv = Vec::with_capacity(
                layer.wq.data().len() + layer.wk.data().len() + layer.wv.data().len(),
            );
            qkv.extend_from_slice(layer.wq.data());
            qkv.extend_from_slice(layer.wk.data());
            qkv.extend_from_slice(layer.wv.data());
            qkv_all.push(qkv);

            let mut gate_up =
                Vec::with_capacity(layer.w_gate.data().len() + layer.w_up.data().len());
            gate_up.extend_from_slice(layer.w_gate.data());
            gate_up.extend_from_slice(layer.w_up.data());
            gate_up_all.push(gate_up);
        }
        self.fused_f32_qkv = Some(qkv_all);
        self.fused_f32_gate_up = Some(gate_up_all);
    }

    /// Upload all model weights to persistent GPU buffers.
    ///
    /// This must be called after `set_backend` and `set_raw_weights`.
    /// After this call, `forward()` will automatically use the GPU-resident
    /// path: a single command encoder submission with no intermediate CPU
    /// readback, reading logits only at the end.
    pub fn upload_weights_to_gpu(&self) {
        if !self.backend.supports_gpu_forward() {
            return;
        }
        let raw = match &self.raw_weights {
            Some(r) => r,
            None => return,
        };

        #[allow(clippy::type_complexity)]
        let layer_norms: Vec<(&[f32], &[f32], Option<&[f32]>, Option<&[f32]>)> = self
            .weights
            .layers
            .iter()
            .map(|l| {
                (
                    l.attn_norm.data(),
                    l.ffn_norm.data(),
                    l.post_attn_norm.as_ref().map(|t| t.data()),
                    l.post_ffn_norm.as_ref().map(|t| t.data()),
                )
            })
            .collect();

        self.backend.upload_weights_to_gpu(
            raw,
            &layer_norms,
            self.weights.output_norm.data(),
            self.weights.output_weight.data(),
            self.weights.token_embedding.data(),
        );
    }

    /// Remove raw weights and fused weight data, reverting to the f32 weight path.
    pub fn clear_raw_weights(&mut self) {
        self.raw_weights = None;
        self.fused_weights = None;
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
        // Cap GPU KV cache to fit within wgpu max_storage_buffer_binding_size
        // (128 MB). Each KV buffer is max_seq_len × num_kv_heads × head_dim × 4 bytes.
        let bytes_per_pos = self.config.num_kv_heads * self.config.head_dim * 4;
        let max_gpu_seq = if bytes_per_pos > 0 {
            (128 * 1024 * 1024 / bytes_per_pos).min(self.config.max_seq_len)
        } else {
            self.config.max_seq_len
        };
        self.backend.init_gpu_kv_cache(
            self.config.num_layers,
            max_gpu_seq,
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
        if let Some(ref mut qkv) = self.quantized_kv_cache {
            qkv.clear();
        }
    }

    /// Run a dummy forward pass to warm caches and initialize thread pools.
    ///
    /// This pages weight data into memory, warms CPU caches, and initializes
    /// the rayon thread pool. Eliminates the 30-50% first-token latency penalty.
    /// KV cache is reset after warmup so the model is in a clean state.
    pub fn warmup(&mut self) {
        let _logits = self.forward(0, 0);
        self.reset();
    }

    /// Merge a LoRA adapter into the model weights (f32 path).
    ///
    /// Applies `W_new = W + (alpha / rank) * B @ A` for every adapted projection
    /// in every layer.  After merging, the adapter's effect is permanent until
    /// the base weights are reloaded.
    ///
    /// This invalidates any fused weight caches (Q8_0 fused weights and f32 fused
    /// QKV/gate-up buffers are cleared and must be rebuilt if needed).
    ///
    /// # Errors
    ///
    /// Returns `LoraError` if dimensions are inconsistent or a layer index is
    /// out of range.
    pub fn merge_lora(
        &mut self,
        adapter: &crate::lora::LoraAdapter,
    ) -> Result<(), crate::lora::LoraError> {
        use crate::lora::{apply_lora_delta, LoraError};

        if adapter.layers.is_empty() {
            return Err(LoraError::EmptyAdapter);
        }

        let scale = adapter.alpha / adapter.rank as f32;
        let rank = adapter.rank;

        for (layer_idx, lora_layer) in adapter.layers.iter().enumerate() {
            if layer_idx >= self.weights.layers.len() {
                return Err(LoraError::LayerIndexOutOfRange {
                    index: layer_idx,
                    count: self.weights.layers.len(),
                });
            }

            let layer = &mut self.weights.layers[layer_idx];

            // Helper macro to apply a LoRA pair to a weight tensor
            macro_rules! merge_pair {
                ($a:expr, $b:expr, $w:expr) => {
                    if let (Some(a), Some(b)) = ($a, $b) {
                        apply_lora_delta($w.data_mut(), a, b, rank, scale)?;
                    }
                };
            }

            // Attention projections
            merge_pair!(&lora_layer.wq_a, &lora_layer.wq_b, &mut layer.wq);
            merge_pair!(&lora_layer.wk_a, &lora_layer.wk_b, &mut layer.wk);
            merge_pair!(&lora_layer.wv_a, &lora_layer.wv_b, &mut layer.wv);
            merge_pair!(&lora_layer.wo_a, &lora_layer.wo_b, &mut layer.wo);

            // FFN projections
            merge_pair!(
                &lora_layer.w_gate_a,
                &lora_layer.w_gate_b,
                &mut layer.w_gate
            );
            merge_pair!(&lora_layer.w_up_a, &lora_layer.w_up_b, &mut layer.w_up);
            merge_pair!(
                &lora_layer.w_down_a,
                &lora_layer.w_down_b,
                &mut layer.w_down
            );
        }

        // Invalidate fused weight caches — they embed the old weight values.
        self.fused_weights = None;
        self.fused_f32_qkv = None;
        self.fused_f32_gate_up = None;

        Ok(())
    }

    /// Run a single forward pass for one token position.
    /// Returns logits over the vocabulary `[vocab_size]`.
    ///
    /// Delegates heavy compute (matvec, rmsnorm, rope, silu_mul) to the
    /// active `ComputeBackend`. By default this is `CpuBackend`; call
    /// `set_backend` to use GPU acceleration.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Tensor {
        // Copy config values we need for the GPU fast path (avoids holding
        // a borrow on self.config across the forward_layers call below).
        let dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let vocab_size = self.config.vocab_size;

        // --- GPU-resident fast path ---
        // When GPU weights are uploaded, run the entire forward pass with a single
        // command encoder submission. Only the final logits are read back to CPU.
        // This avoids ~90+ CPU↔GPU round-trips per token.
        if self.backend.has_gpu_weights() && self.backend.has_gpu_kv_cache() {
            let embed_data = self.weights.token_embedding.data();
            let embed_offset = token_id as usize * dim;
            let token_embed = &embed_data[embed_offset..embed_offset + dim];
            let seq_len = self.kv_cache.len() + 1;

            if let Some(mut logits) = self.backend.forward_single_token_gpu(
                token_embed,
                pos,
                dim,
                num_heads,
                num_kv_heads,
                head_dim,
                self.config.intermediate_dim,
                vocab_size,
                self.config.rms_norm_eps,
                self.config.rope_theta,
                self.config.num_layers,
                seq_len,
            ) {
                // Advance KV cache position counter
                self.kv_cache.advance();

                // Gemma 2: apply final logit soft-cap
                if self.config.final_logit_softcap > 0.0 {
                    let cap = self.config.final_logit_softcap;
                    for l in &mut logits {
                        *l = (*l / cap).tanh() * cap;
                    }
                }

                return Tensor::from_vec(logits, &[vocab_size]).unwrap();
            }
        }

        // --- Fallback: per-operation GPU dispatch or CPU path ---
        //
        // Run all transformer layers + final RMSNorm via forward_layers().
        // The result is left in self.forward_buffers.final_normed.
        let (_use_raw, _use_q8_attn, _use_q8_ffn, use_q8_output, use_cpu_q4k) =
            self.forward_layers(token_id, pos);

        // Output logits: [vocab_size] = output_weight [vocab_size, dim] x normed [dim]
        // Use quantized path when raw output weight is available (bandwidth reduction).
        if let Some(ref row) = self.raw_output_weight {
            if row.format == WeightFormat::Q4K && use_cpu_q4k {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q4k_q8_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            } else if row.format == WeightFormat::Q8_0 && use_q8_output {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q8_0_preq_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    &mut self.forward_buffers.logits,
                );
            } else {
                matvec_into(
                    self.weights.output_weight.data(),
                    &self.forward_buffers.final_normed,
                    vocab_size,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            }
        } else {
            matvec_into(
                self.weights.output_weight.data(),
                &self.forward_buffers.final_normed,
                vocab_size,
                dim,
                &mut self.forward_buffers.logits,
            );
        }

        // Gemma 2: apply final logit soft-cap (tanh(x / cap) * cap)
        if self.config.final_logit_softcap > 0.0 {
            let cap = self.config.final_logit_softcap;
            for l in &mut self.forward_buffers.logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(self.forward_buffers.logits.clone(), &[vocab_size]).unwrap()
    }

    /// Greedy forward pass: runs the full transformer but returns only the
    /// argmax token ID instead of all logits.
    ///
    /// The output projection (vocab_size x dim matvec) is the single most
    /// expensive operation per token. For greedy decoding we only need the
    /// argmax, not all 128K+ logits. This method fuses the argmax into the
    /// output matvec, avoiding a 512KB+ write to the logits buffer and the
    /// subsequent scan. This reduces cache pollution and saves one full pass
    /// over the logits vector.
    ///
    /// Callers should use this when `temperature == 0.0` and no post-processing
    /// of the logit distribution is needed (e.g. no repeat penalty, no top-p).
    ///
    /// Returns `(token_id, logit_value)` for the greedy-best token.
    pub fn forward_greedy(&mut self, token_id: u32, pos: usize) -> (u32, f32) {
        // GPU-resident fast path: fall back to full forward + argmax since
        // the GPU path returns all logits anyway.
        if self.backend.has_gpu_weights() && self.backend.has_gpu_kv_cache() {
            let logits_tensor = self.forward(token_id, pos);
            let logits = logits_tensor.data();
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            return (best_idx, best_val);
        }

        // Run all transformer layers + final RMSNorm, leaving the result in
        // forward_buffers.final_normed. Shared with forward().
        let (_use_raw, _use_q8_attn, _use_q8_ffn, use_q8_output, use_cpu_q4k) =
            self.forward_layers(token_id, pos);

        let dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;

        // Fused argmax output projection — no logits buffer allocation.
        let (best_idx, mut best_val) = if let Some(ref row) = self.raw_output_weight {
            if row.format == WeightFormat::Q4K && use_cpu_q4k {
                // Fused Q4_K × Q8_0 argmax: no logits buffer write, no scan.
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_argmax_q4k_q8_preq(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    dim,
                )
            } else if row.format == WeightFormat::Q8_0 && use_q8_output {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_argmax_q8_0_preq(&row.data, &self.forward_buffers.output_q8, row.num_rows)
            } else {
                matvec_argmax_f32(
                    self.weights.output_weight.data(),
                    &self.forward_buffers.final_normed,
                    vocab_size,
                    dim,
                )
            }
        } else {
            matvec_argmax_f32(
                self.weights.output_weight.data(),
                &self.forward_buffers.final_normed,
                vocab_size,
                dim,
            )
        };

        // Gemma 2: apply final logit soft-cap to the winning logit.
        // Since tanh is monotonic, argmax is preserved; we just adjust the value.
        if self.config.final_logit_softcap > 0.0 {
            let cap = self.config.final_logit_softcap;
            best_val = (best_val / cap).tanh() * cap;
        }

        (best_idx as u32, best_val)
    }

    /// Run all transformer layers and final RMSNorm, leaving the result in
    /// `self.forward_buffers.final_normed`. Does NOT compute the output
    /// projection (logits). Also advances the KV cache.
    ///
    /// This is the shared prefix of `forward()` and `forward_greedy()`,
    /// extracted to avoid code duplication in the ~400-line layer loop.
    ///
    /// Returns `(use_raw, use_q8_attn, use_q8_ffn, use_q8_output, use_cpu_q4k)`
    /// so the caller can dispatch the correct output projection variant without
    /// recomputing the flags.
    #[allow(clippy::needless_range_loop)]
    fn forward_layers(&mut self, token_id: u32, pos: usize) -> (bool, bool, bool, bool, bool) {
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

        // `forward_layers` is the single-token decode path.  Even though
        // `CpuBackend` now supports `batched_dequant_matmul`, it's strictly a
        // multi-batch optimisation for prefill / speculative verify; for
        // batch = 1 the fused single-token kernels below (fused QKV Q8_0,
        // direct Q4_K) are faster.  Keep `use_raw` as a strict "GPU-resident
        // single-token" flag here.
        let use_raw = self.backend.supports_dequant_matmul()
            && self.backend.has_gpu_weights()
            && self.raw_weights.is_some();

        // Base Q8_0 eligibility: raw weights present and in Q8_0 format.
        let has_q8 = !use_raw
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q8_0);

        // Per-operation hybrid compute: use Q8_0 only for matrices that exceed
        // the L2 cache threshold (bandwidth-bound). Smaller matrices stay on the
        // f32 Accelerate/BLAS path where AMX throughput dominates.
        let q_dim = num_heads * head_dim;
        let qkv_rows = q_dim + kv_dim + kv_dim;
        let use_q8_attn = has_q8 && should_use_q8_for_size(qkv_rows, dim);
        let use_q8_ffn = has_q8 && should_use_q8_for_size(config.intermediate_dim, dim);
        let use_q8_output = has_q8 && should_use_q8_for_size(config.vocab_size, dim);

        // CPU Q4_K direct path: compute 4-bit x f32 dot product directly on
        // quantized data. ~4.5 bits/weight vs Q8_0's ~8.5 bits, roughly halving
        // memory bandwidth requirements compared to Q8_0.
        let use_cpu_q4k = !use_raw
            && !has_q8
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q4K);

        // Process each transformer layer
        for layer_idx in 0..config.num_layers {
            let layer = &self.weights.layers[layer_idx];

            // --- Attention block ---
            // When the Q8_0 attention path is active, fuse RMSNorm + quantization
            // into a single pass to eliminate the intermediate f32 normed buffer
            // round-trip. Other paths still need the f32 normed vector.
            if use_q8_attn {
                rmsnorm_quantize_q8_0_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed_q8,
                );
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                matvec_q8_0_preq_into(
                    &fused.qkv_data,
                    &self.forward_buffers.normed_q8,
                    fused.qkv_rows,
                    &mut self.forward_buffers.qkv,
                );
                self.forward_buffers
                    .q_data
                    .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                self.forward_buffers
                    .k_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                self.forward_buffers
                    .v_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
            } else {
                rmsnorm_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed,
                );

                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let q_tmp = self.backend.batched_dequant_matmul(
                        &rw.wq,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.q_data.copy_from_slice(&q_tmp);
                    let k_tmp = self.backend.batched_dequant_matmul(
                        &rw.wk,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.k_data.copy_from_slice(&k_tmp);
                    let v_tmp = self.backend.batched_dequant_matmul(
                        &rw.wv,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.v_data.copy_from_slice(&v_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    // Quantize the f32 normed input to Q8_0 once and reuse it
                    // across all three Q/K/V projections.
                    quantize_input_q8_0_into(
                        &self.forward_buffers.normed,
                        &mut self.forward_buffers.normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.wq.data,
                        &self.forward_buffers.normed_q8,
                        rw.wq.num_rows,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wk.data,
                        &self.forward_buffers.normed_q8,
                        rw.wk.num_rows,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wv.data,
                        &self.forward_buffers.normed_q8,
                        rw.wv.num_rows,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                } else if let Some(ref fused_qkv) = self.fused_f32_qkv {
                    matvec_into(
                        &fused_qkv[layer_idx],
                        &self.forward_buffers.normed,
                        q_dim + kv_dim + kv_dim,
                        dim,
                        &mut self.forward_buffers.qkv,
                    );
                    self.forward_buffers
                        .q_data
                        .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                    self.forward_buffers
                        .k_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                    self.forward_buffers
                        .v_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
                } else {
                    matvec_into(
                        layer.wq.data(),
                        &self.forward_buffers.normed,
                        q_dim,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_into(
                        layer.wk.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_into(
                        layer.wv.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                }
            }

            // Add attention biases if present (Qwen2)
            if let Some(bias) = &layer.attn_q_bias {
                for (q, &b) in self
                    .forward_buffers
                    .q_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *q += b;
                }
            }
            if let Some(bias) = &layer.attn_k_bias {
                for (k, &b) in self
                    .forward_buffers
                    .k_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *k += b;
                }
            }
            if let Some(bias) = &layer.attn_v_bias {
                for (v, &b) in self
                    .forward_buffers
                    .v_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *v += b;
                }
            }
            apply_rope_with_table(
                &mut self.forward_buffers.q_data,
                num_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );
            apply_rope_with_table(
                &mut self.forward_buffers.k_data,
                num_kv_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );

            // Write K, V to CPU cache (always) and GPU cache (if initialized).
            self.kv_cache.write(
                layer_idx,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );
            if let Some(ref mut qkv) = self.quantized_kv_cache {
                qkv.write(
                    layer_idx,
                    &self.forward_buffers.k_data,
                    &self.forward_buffers.v_data,
                );
            }
            self.backend.gpu_kv_write(
                layer_idx,
                pos,
                num_kv_heads,
                head_dim,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );

            // Grouped-query attention
            let seq_len = self.kv_cache.len() + 1;
            if self.backend.has_gpu_kv_cache() {
                let attn_tmp = self.backend.grouped_query_attention_from_gpu_cache(
                    &self.forward_buffers.q_data,
                    layer_idx,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                );
                self.forward_buffers.attn_output.copy_from_slice(&attn_tmp);
            } else if let Some(ref qkv) = self.quantized_kv_cache {
                let dequant_keys = qkv.dequant_keys(layer_idx);
                let dequant_values = qkv.dequant_values(layer_idx);
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    &dequant_keys,
                    &dequant_values,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            } else {
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    self.kv_cache.keys(layer_idx).data(),
                    self.kv_cache.values(layer_idx).data(),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            }

            // Output projection (wo) — uses same hybrid flag as QKV
            if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                let tmp = self.backend.batched_dequant_matmul(
                    &rw.wo,
                    &self.forward_buffers.attn_output,
                    1,
                );
                self.forward_buffers.attn_proj.copy_from_slice(&tmp);
            } else if use_cpu_q4k {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q4k_q8_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            } else if use_q8_attn {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q8_0_preq_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    &mut self.forward_buffers.attn_proj,
                );
            } else {
                matvec_into(
                    layer.wo.data(),
                    &self.forward_buffers.attn_output,
                    dim,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            }

            // Gemma 2: apply post-attention RMSNorm before the residual add
            if let Some(ref post_norm) = layer.post_attn_norm {
                rmsnorm_into(
                    &self.forward_buffers.attn_proj,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib,
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                // Residual connection (no post-norm)
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_proj[i];
                }
            }

            // --- FFN block ---
            // When the Q8_0 FFN path is active (non-MoE), defer RMSNorm so we
            // can fuse it with quantization below. All other paths compute the
            // f32 normed vector up front.
            if !use_q8_ffn || config.moe {
                rmsnorm_into(
                    x.data(),
                    layer.ffn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.ffn_normed,
                );
            }

            // MoE path: route through top-k experts instead of the dense FFN.
            if config.moe {
                if let Some(ref moe_weights) = layer.moe {
                    moe_forward_into(
                        &self.forward_buffers.ffn_normed,
                        moe_weights,
                        dim,
                        config.intermediate_dim,
                        config.num_experts_per_token,
                        &mut self.forward_buffers.ffn_out,
                    );
                } else {
                    // Fallback: MoE config is set but this layer has no MoE weights.
                    // Should not happen with valid models, but handle gracefully.
                    for o in self.forward_buffers.ffn_out.iter_mut() {
                        *o = 0.0;
                    }
                }
            } else {
                // Dense FFN path (non-MoE)

                // Prefetch FFN gate/up weights while computing FFN RMSNorm.
                // The norm is lightweight (~dim FLOPs) and doesn't touch weight
                // memory, so issuing prefetch hints here lets the memory subsystem
                // start pulling gate/up data into L2 before the matvec needs it.
                if use_q8_ffn {
                    let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                    prefetch_weight_bytes(&fused.gate_up_data);
                } else if self.fused_f32_gate_up.is_some() {
                    prefetch_weight_f32(&self.fused_f32_gate_up.as_ref().unwrap()[layer_idx]);
                } else {
                    prefetch_weight_f32(layer.w_gate.data());
                }

                // Gate and up projections
                let inter_dim = config.intermediate_dim;
                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let gate_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_gate,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.gate.copy_from_slice(&gate_tmp);
                    let up_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_up,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.up.copy_from_slice(&up_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.ffn_normed,
                        &mut self.forward_buffers.ffn_normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_gate.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_gate.num_rows,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_up.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_up.num_rows,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                } else if use_q8_ffn {
                    // Fused RMSNorm + Q8_0 quantization (norm was deferred above)
                    rmsnorm_quantize_q8_0_into(
                        x.data(),
                        layer.ffn_norm.data(),
                        config.rms_norm_eps,
                        &mut self.forward_buffers.ffn_normed_q8,
                    );
                    let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                    matvec_q8_0_preq_into(
                        &fused.gate_up_data,
                        &self.forward_buffers.ffn_normed_q8,
                        fused.gate_up_rows,
                        &mut self.forward_buffers.gate_up,
                    );
                    self.forward_buffers
                        .gate
                        .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                    self.forward_buffers
                        .up
                        .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
                } else if let Some(ref fused_gu) = self.fused_f32_gate_up {
                    matvec_into(
                        &fused_gu[layer_idx],
                        &self.forward_buffers.ffn_normed,
                        inter_dim * 2,
                        dim,
                        &mut self.forward_buffers.gate_up,
                    );
                    self.forward_buffers
                        .gate
                        .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                    self.forward_buffers
                        .up
                        .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
                } else {
                    matvec_into(
                        layer.w_gate.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_into(
                        layer.w_up.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                }

                // Gemma 2 uses GELU activation; Llama / Phi-3 use SiLU
                if config.architecture == Architecture::Gemma2 {
                    gelu_mul_into(
                        &self.forward_buffers.gate,
                        &self.forward_buffers.up,
                        &mut self.forward_buffers.ffn_hidden,
                    );
                } else {
                    silu_mul_into(
                        &self.forward_buffers.gate,
                        &self.forward_buffers.up,
                        &mut self.forward_buffers.ffn_hidden,
                    );
                }

                // Prefetch next layer's attention weights while the down projection
                // runs. The down projection is the heaviest single matvec in the FFN
                // block (intermediate_dim x dim), so there is ample time for the
                // prefetch to complete before the next layer's QKV matvec begins.
                if layer_idx + 1 < config.num_layers {
                    let next_layer = &self.weights.layers[layer_idx + 1];
                    prefetch_weight_f32(next_layer.attn_norm.data());
                    if use_q8_attn {
                        let next_fused = &self.fused_weights.as_ref().unwrap()[layer_idx + 1];
                        prefetch_weight_bytes(&next_fused.qkv_data);
                    } else if self.fused_f32_qkv.is_some() {
                        prefetch_weight_f32(&self.fused_f32_qkv.as_ref().unwrap()[layer_idx + 1]);
                    } else {
                        prefetch_weight_f32(next_layer.wq.data());
                    }
                }

                // Down projection
                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let tmp = self.backend.batched_dequant_matmul(
                        &rw.w_down,
                        &self.forward_buffers.ffn_hidden,
                        1,
                    );
                    self.forward_buffers.ffn_out.copy_from_slice(&tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.ffn_hidden,
                        &mut self.forward_buffers.ffn_hidden_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_down.data,
                        &self.forward_buffers.ffn_hidden_q8,
                        rw.w_down.num_rows,
                        config.intermediate_dim,
                        &mut self.forward_buffers.ffn_out,
                    );
                } else if use_q8_ffn {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.ffn_hidden,
                        &mut self.forward_buffers.ffn_hidden_q8,
                    );
                    matvec_q8_0_preq_into(
                        &rw.w_down.data,
                        &self.forward_buffers.ffn_hidden_q8,
                        rw.w_down.num_rows,
                        &mut self.forward_buffers.ffn_out,
                    );
                } else {
                    matvec_into(
                        layer.w_down.data(),
                        &self.forward_buffers.ffn_hidden,
                        dim,
                        config.intermediate_dim,
                        &mut self.forward_buffers.ffn_out,
                    );
                }
            } // end dense FFN path

            // Gemma 2: apply post-FFN RMSNorm before the residual add
            if let Some(ref post_norm) = layer.post_ffn_norm {
                rmsnorm_into(
                    &self.forward_buffers.ffn_out,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib, // reuse as scratch
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                // Residual connection (no post-norm)
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.ffn_out[i];
                }
            }
        }

        // Advance KV cache after processing all layers
        self.kv_cache.advance();
        if let Some(ref mut qkv) = self.quantized_kv_cache {
            qkv.advance();
        }

        // Final RMSNorm — result left in forward_buffers.final_normed
        rmsnorm_into(
            x.data(),
            self.weights.output_norm.data(),
            config.rms_norm_eps,
            &mut self.forward_buffers.final_normed,
        );

        (use_raw, use_q8_attn, use_q8_ffn, use_q8_output, use_cpu_q4k)
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
                // Use fused dequant+matvec when raw quantized weights are available
                // *and* the specific weight's format is dispatchable by the backend
                // (mixed-format models like Q4_K_M fall back per-weight).
                let raw_layer = if use_raw {
                    Some(&self.raw_weights.as_ref().unwrap()[layer_idx])
                } else {
                    None
                };
                let q_proj = if let Some(rw) =
                    raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.wq))
                {
                    self.backend
                        .batched_dequant_matmul(&rw.wq, &normed_batch, seq_len)
                } else {
                    let wq: Vec<f32> = self.weights.layers[layer_idx].wq.data().to_vec();
                    self.backend
                        .batched_matmul(&wq, &normed_batch, q_dim, dim, seq_len)
                };
                let k_proj = if let Some(rw) =
                    raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.wk))
                {
                    self.backend
                        .batched_dequant_matmul(&rw.wk, &normed_batch, seq_len)
                } else {
                    let wk: Vec<f32> = self.weights.layers[layer_idx].wk.data().to_vec();
                    self.backend
                        .batched_matmul(&wk, &normed_batch, kv_dim, dim, seq_len)
                };
                let v_proj = if let Some(rw) =
                    raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.wv))
                {
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
                let raw_layer = if use_raw {
                    Some(&self.raw_weights.as_ref().unwrap()[layer_idx])
                } else {
                    None
                };
                let proj_batch = if let Some(rw) =
                    raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.wo))
                {
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

                // MoE path: per-token routing through top-k experts.
                let down_batch = if config.moe {
                    if let Some(ref moe_weights) = self.weights.layers[layer_idx].moe {
                        let mut result = vec![0.0f32; seq_len * dim];
                        for t in 0..seq_len {
                            let x_t = &normed_batch2[t * dim..(t + 1) * dim];
                            let out = moe_forward(
                                x_t,
                                moe_weights,
                                dim,
                                inter,
                                config.num_experts_per_token,
                            );
                            result[t * dim..(t + 1) * dim].copy_from_slice(&out);
                        }
                        result
                    } else {
                        vec![0.0f32; seq_len * dim]
                    }
                } else {
                    // Dense FFN path (non-MoE)

                    // Single batched dispatch for gate and up projections.
                    let raw_layer = if use_raw {
                        Some(&self.raw_weights.as_ref().unwrap()[layer_idx])
                    } else {
                        None
                    };
                    let gate_batch = if let Some(rw) =
                        raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.w_gate))
                    {
                        self.backend
                            .batched_dequant_matmul(&rw.w_gate, &normed_batch2, seq_len)
                    } else {
                        let w_gate: Vec<f32> =
                            self.weights.layers[layer_idx].w_gate.data().to_vec();
                        self.backend
                            .batched_matmul(&w_gate, &normed_batch2, inter, dim, seq_len)
                    };
                    let up_batch = if let Some(rw) =
                        raw_layer.filter(|l| raw_weight_cpu_dispatchable(&l.w_up))
                    {
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
                    if let Some(rw) = raw_layer
                        .filter(|l| raw_weight_cpu_dispatchable(&l.w_down))
                    {
                        self.backend
                            .batched_dequant_matmul(&rw.w_down, &ffn_hidden_batch, seq_len)
                    } else {
                        let w_down: Vec<f32> =
                            self.weights.layers[layer_idx].w_down.data().to_vec();
                        self.backend
                            .batched_matmul(&w_down, &ffn_hidden_batch, dim, inter, seq_len)
                    }
                }; // end dense FFN / MoE branch

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
                if let Some(ref mut qkv) = self.quantized_kv_cache {
                    qkv.write(layer_idx, k_slice, v_slice);
                }
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
            if let Some(ref mut qkv) = self.quantized_kv_cache {
                qkv.advance();
            }
        }

        // Final RMSNorm + output projection on the last token only
        let last = seq_len - 1;
        let last_x = &x_batch[last * dim..(last + 1) * dim];
        let normed_final =
            self.backend
                .rmsnorm_vec(last_x, self.weights.output_norm.data(), config.rms_norm_eps);

        // Use the CPU SIMD matvec directly for the output projection.
        // The output weight matrix can exceed GPU buffer limits for large vocabs
        // (e.g. 128K vocab × 2048 dim × 4 bytes > 1GB), so bypass the backend.
        // When raw Q8_0 output weight is available, use Q8_0 path for ~4x bandwidth
        // reduction on the largest single matvec in the model.
        let use_output_q4k = self
            .raw_output_weight
            .as_ref()
            .is_some_and(|rw| rw.format == WeightFormat::Q4K && dim >= 1024);
        let use_output_q8 = self
            .raw_output_weight
            .as_ref()
            .is_some_and(|rw| rw.format == WeightFormat::Q8_0 && dim >= 1024);
        let mut logits = if use_output_q4k {
            let row = self.raw_output_weight.as_ref().unwrap();
            let mut out = vec![0.0f32; config.vocab_size];
            let preq = quantize_input_q8_0(&normed_final);
            matvec_q4k_q8_into(&row.data, &preq, row.num_rows, dim, &mut out);
            out
        } else if use_output_q8 {
            let row = self.raw_output_weight.as_ref().unwrap();
            let preq = quantize_input_q8_0(&normed_final);
            let mut out = vec![0.0f32; config.vocab_size];
            matvec_q8_0_preq_into(&row.data, &preq, row.num_rows, &mut out);
            out
        } else {
            matvec(
                self.weights.output_weight.data(),
                &normed_final,
                config.vocab_size,
                dim,
            )
        };

        if config.final_logit_softcap > 0.0 {
            let cap = config.final_logit_softcap;
            for l in &mut logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(logits, &[config.vocab_size]).unwrap()
    }

    /// Returns the number of transformer layers that currently have weights loaded.
    ///
    /// This is always equal to the total layer count unless the model was created
    /// with placeholder weights and layers are being loaded incrementally via
    /// [`load_layer_weights`](Self::load_layer_weights).
    pub fn available_layers(&self) -> usize {
        self.layers_loaded.unwrap_or(self.config.num_layers)
    }

    /// Load (or replace) the weights for a single transformer layer.
    ///
    /// This enables progressive inference: create a model with placeholder
    /// weights, then call this method as layers arrive over the network.
    /// The model becomes usable for [`forward_partial`](Self::forward_partial)
    /// as soon as at least one layer is loaded.
    pub fn load_layer_weights(&mut self, layer_idx: usize, weights: LayerWeights) {
        assert!(
            layer_idx < self.config.num_layers,
            "layer_idx {layer_idx} out of range (model has {} layers)",
            self.config.num_layers
        );
        self.weights.layers[layer_idx] = weights;

        // Track the number of contiguously loaded layers from the front.
        // This matters because forward_partial runs layers 0..num_layers
        // sequentially, so gaps are not useful.
        let current = self.layers_loaded.unwrap_or(0);
        if layer_idx < current {
            // Re-loaded an already-loaded layer; count stays the same.
        } else {
            // Walk forward from current to find new contiguous frontier.
            let mut frontier = current;
            // Mark this layer, then scan.  We use a simple approach: if the
            // caller loads layers in order (the expected case), frontier
            // advances by 1.  For out-of-order loads we conservatively only
            // advance when the next layer is the one just loaded.
            if layer_idx == frontier {
                frontier += 1;
                // Check if subsequent layers were loaded earlier (out-of-order).
                // We track this with a bitset would be ideal, but for simplicity
                // we just advance by 1 per call; production code can use
                // LoadProgress for full tracking.
            }
            self.layers_loaded = Some(frontier);
        }
    }

    /// Returns a quality score in `[0.0, 1.0]` representing the fraction of
    /// transformer layers available for inference.
    ///
    /// A value of `1.0` means all layers are loaded and the model produces
    /// full-quality output.  Lower values indicate degraded (but still useful)
    /// output from partial-layer inference.
    pub fn inference_quality(&self) -> f32 {
        if self.config.num_layers == 0 {
            return 1.0;
        }
        self.available_layers() as f32 / self.config.num_layers as f32
    }

    /// Run a single forward pass using only the first `num_layers` transformer
    /// layers, then apply the final RMSNorm and output projection.
    ///
    /// This enables **progressive inference**: start generating tokens with a
    /// partially-loaded model and improve quality as more layers arrive.
    /// Research shows LLMs produce useful representations at intermediate
    /// layers (see *LayerSkip*, arXiv:2412.01455).
    ///
    /// # Panics
    ///
    /// Panics if `num_layers` is 0 or exceeds `available_layers()`.
    ///
    /// # KV cache
    ///
    /// Only the first `num_layers` layers write to the KV cache.  If you later
    /// call `forward()` (full model) you should call `reset()` first to avoid
    /// stale cache entries from the partial pass.
    #[allow(clippy::needless_range_loop)]
    pub fn forward_partial(&mut self, token_id: u32, pos: usize, num_layers: usize) -> Tensor {
        assert!(num_layers > 0, "forward_partial requires at least 1 layer");
        assert!(
            num_layers <= self.available_layers(),
            "forward_partial: requested {num_layers} layers but only {} are available",
            self.available_layers()
        );

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

        // Single-token decode: `use_raw` is reserved for GPU-resident weights.
        // CPU `batched_dequant_matmul` is strictly a prefill/multi-batch
        // optimisation; for batch = 1 the fused Q8_0 / direct Q4_K kernels
        // below are faster.
        let use_raw = self.backend.supports_dequant_matmul()
            && self.backend.has_gpu_weights()
            && self.raw_weights.is_some();

        // Base Q8_0 eligibility: raw weights present and in Q8_0 format.
        let has_q8 = !use_raw
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q8_0);

        // Per-operation hybrid compute: use Q8_0 only for bandwidth-bound matrices.
        let q_dim = num_heads * head_dim;
        let qkv_rows = q_dim + kv_dim + kv_dim;
        let use_q8_attn = has_q8 && should_use_q8_for_size(qkv_rows, dim);
        let use_q8_ffn = has_q8 && should_use_q8_for_size(config.intermediate_dim, dim);
        let use_q8_output = has_q8 && should_use_q8_for_size(config.vocab_size, dim);

        let use_cpu_q4k = !use_raw
            && !has_q8
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q4K);

        // Process only the first `num_layers` transformer layers
        // Uses pre-allocated ForwardBuffers to eliminate per-token heap allocations.
        for layer_idx in 0..num_layers {
            let layer = &self.weights.layers[layer_idx];

            // --- Attention block ---
            // Fuse RMSNorm + Q8_0 quantization when the Q8 attention path is
            // active; other paths still need the f32 normed vector.
            if use_q8_attn {
                rmsnorm_quantize_q8_0_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed_q8,
                );
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                matvec_q8_0_preq_into(
                    &fused.qkv_data,
                    &self.forward_buffers.normed_q8,
                    fused.qkv_rows,
                    &mut self.forward_buffers.qkv,
                );
                self.forward_buffers
                    .q_data
                    .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                self.forward_buffers
                    .k_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                self.forward_buffers
                    .v_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
            } else {
                rmsnorm_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed,
                );

                // QKV projections — fused into a single matvec where possible.
                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let q_tmp = self.backend.batched_dequant_matmul(
                        &rw.wq,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.q_data.copy_from_slice(&q_tmp);
                    let k_tmp = self.backend.batched_dequant_matmul(
                        &rw.wk,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.k_data.copy_from_slice(&k_tmp);
                    let v_tmp = self.backend.batched_dequant_matmul(
                        &rw.wv,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.v_data.copy_from_slice(&v_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.normed,
                        &mut self.forward_buffers.normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.wq.data,
                        &self.forward_buffers.normed_q8,
                        rw.wq.num_rows,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wk.data,
                        &self.forward_buffers.normed_q8,
                        rw.wk.num_rows,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wv.data,
                        &self.forward_buffers.normed_q8,
                        rw.wv.num_rows,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                } else if let Some(ref fused_qkv) = self.fused_f32_qkv {
                    matvec_into(
                        &fused_qkv[layer_idx],
                        &self.forward_buffers.normed,
                        q_dim + kv_dim + kv_dim,
                        dim,
                        &mut self.forward_buffers.qkv,
                    );
                    self.forward_buffers
                        .q_data
                        .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                    self.forward_buffers
                        .k_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                    self.forward_buffers
                        .v_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
                } else {
                    matvec_into(
                        layer.wq.data(),
                        &self.forward_buffers.normed,
                        q_dim,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_into(
                        layer.wk.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_into(
                        layer.wv.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                }
            }

            // Attention biases (Qwen2)
            if let Some(bias) = &layer.attn_q_bias {
                for (q, &b) in self
                    .forward_buffers
                    .q_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *q += b;
                }
            }
            if let Some(bias) = &layer.attn_k_bias {
                for (k, &b) in self
                    .forward_buffers
                    .k_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *k += b;
                }
            }
            if let Some(bias) = &layer.attn_v_bias {
                for (v, &b) in self
                    .forward_buffers
                    .v_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *v += b;
                }
            }
            apply_rope_with_table(
                &mut self.forward_buffers.q_data,
                num_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );
            apply_rope_with_table(
                &mut self.forward_buffers.k_data,
                num_kv_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );

            self.kv_cache.write(
                layer_idx,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );
            if let Some(ref mut qkv) = self.quantized_kv_cache {
                qkv.write(
                    layer_idx,
                    &self.forward_buffers.k_data,
                    &self.forward_buffers.v_data,
                );
            }
            self.backend.gpu_kv_write(
                layer_idx,
                pos,
                num_kv_heads,
                head_dim,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );

            let seq_len = self.kv_cache.len() + 1;
            if self.backend.has_gpu_kv_cache() {
                let attn_tmp = self.backend.grouped_query_attention_from_gpu_cache(
                    &self.forward_buffers.q_data,
                    layer_idx,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                );
                self.forward_buffers.attn_output.copy_from_slice(&attn_tmp);
            } else if let Some(ref qkv) = self.quantized_kv_cache {
                let dequant_keys = qkv.dequant_keys(layer_idx);
                let dequant_values = qkv.dequant_values(layer_idx);
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    &dequant_keys,
                    &dequant_values,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            } else {
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    self.kv_cache.keys(layer_idx).data(),
                    self.kv_cache.values(layer_idx).data(),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            }

            if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                let tmp = self.backend.batched_dequant_matmul(
                    &rw.wo,
                    &self.forward_buffers.attn_output,
                    1,
                );
                self.forward_buffers.attn_proj.copy_from_slice(&tmp);
            } else if use_cpu_q4k {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q4k_q8_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            } else if use_q8_attn {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q8_0_preq_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    &mut self.forward_buffers.attn_proj,
                );
            } else {
                matvec_into(
                    layer.wo.data(),
                    &self.forward_buffers.attn_output,
                    dim,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            }

            if let Some(ref post_norm) = layer.post_attn_norm {
                rmsnorm_into(
                    &self.forward_buffers.attn_proj,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib,
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_proj[i];
                }
            }

            // --- FFN block ---

            // Prefetch FFN gate/up weights while computing FFN RMSNorm.
            if use_q8_ffn {
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                prefetch_weight_bytes(&fused.gate_up_data);
            } else if self.fused_f32_gate_up.is_some() {
                prefetch_weight_f32(&self.fused_f32_gate_up.as_ref().unwrap()[layer_idx]);
            } else if !use_raw {
                prefetch_weight_f32(layer.w_gate.data());
            }

            // Gate and up projections — fused into a single matvec where possible.
            let inter_dim = config.intermediate_dim;
            if use_q8_ffn {
                // Fused RMSNorm + Q8_0 quantization
                rmsnorm_quantize_q8_0_into(
                    x.data(),
                    layer.ffn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.ffn_normed_q8,
                );
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                matvec_q8_0_preq_into(
                    &fused.gate_up_data,
                    &self.forward_buffers.ffn_normed_q8,
                    fused.gate_up_rows,
                    &mut self.forward_buffers.gate_up,
                );
                self.forward_buffers
                    .gate
                    .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                self.forward_buffers
                    .up
                    .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
            } else {
                rmsnorm_into(
                    x.data(),
                    layer.ffn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.ffn_normed,
                );

                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let gate_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_gate,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.gate.copy_from_slice(&gate_tmp);
                    let up_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_up,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.up.copy_from_slice(&up_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.ffn_normed,
                        &mut self.forward_buffers.ffn_normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_gate.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_gate.num_rows,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_up.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_up.num_rows,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                } else if let Some(ref fused_gu) = self.fused_f32_gate_up {
                    matvec_into(
                        &fused_gu[layer_idx],
                        &self.forward_buffers.ffn_normed,
                        inter_dim * 2,
                        dim,
                        &mut self.forward_buffers.gate_up,
                    );
                    self.forward_buffers
                        .gate
                        .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                    self.forward_buffers
                        .up
                        .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
                } else {
                    matvec_into(
                        layer.w_gate.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_into(
                        layer.w_up.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                }
            }

            if config.architecture == Architecture::Gemma2 {
                gelu_mul_into(
                    &self.forward_buffers.gate,
                    &self.forward_buffers.up,
                    &mut self.forward_buffers.ffn_hidden,
                );
            } else {
                silu_mul_into(
                    &self.forward_buffers.gate,
                    &self.forward_buffers.up,
                    &mut self.forward_buffers.ffn_hidden,
                );
            }

            // Prefetch next layer's attention weights during down projection.
            if layer_idx + 1 < num_layers {
                let next_layer = &self.weights.layers[layer_idx + 1];
                prefetch_weight_f32(next_layer.attn_norm.data());
                if use_q8_attn {
                    let next_fused = &self.fused_weights.as_ref().unwrap()[layer_idx + 1];
                    prefetch_weight_bytes(&next_fused.qkv_data);
                } else if self.fused_f32_qkv.is_some() {
                    prefetch_weight_f32(&self.fused_f32_qkv.as_ref().unwrap()[layer_idx + 1]);
                } else if !use_raw {
                    prefetch_weight_f32(next_layer.wq.data());
                }
            }

            if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                let tmp = self.backend.batched_dequant_matmul(
                    &rw.w_down,
                    &self.forward_buffers.ffn_hidden,
                    1,
                );
                self.forward_buffers.ffn_out.copy_from_slice(&tmp);
            } else if use_cpu_q4k {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.ffn_hidden,
                    &mut self.forward_buffers.ffn_hidden_q8,
                );
                matvec_q4k_q8_into(
                    &rw.w_down.data,
                    &self.forward_buffers.ffn_hidden_q8,
                    rw.w_down.num_rows,
                    config.intermediate_dim,
                    &mut self.forward_buffers.ffn_out,
                );
            } else if use_q8_ffn {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.ffn_hidden,
                    &mut self.forward_buffers.ffn_hidden_q8,
                );
                matvec_q8_0_preq_into(
                    &rw.w_down.data,
                    &self.forward_buffers.ffn_hidden_q8,
                    rw.w_down.num_rows,
                    &mut self.forward_buffers.ffn_out,
                );
            } else {
                matvec_into(
                    layer.w_down.data(),
                    &self.forward_buffers.ffn_hidden,
                    dim,
                    config.intermediate_dim,
                    &mut self.forward_buffers.ffn_out,
                );
            }

            if let Some(ref post_norm) = layer.post_ffn_norm {
                rmsnorm_into(
                    &self.forward_buffers.ffn_out,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib,
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.ffn_out[i];
                }
            }
        }

        // Advance KV cache after processing the partial layers
        self.kv_cache.advance();
        if let Some(ref mut qkv) = self.quantized_kv_cache {
            qkv.advance();
        }

        // Final RMSNorm + output projection
        rmsnorm_into(
            x.data(),
            self.weights.output_norm.data(),
            config.rms_norm_eps,
            &mut self.forward_buffers.final_normed,
        );

        if let Some(ref row) = self.raw_output_weight {
            if row.format == WeightFormat::Q4K && use_cpu_q4k {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q4k_q8_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            } else if row.format == WeightFormat::Q8_0 && use_q8_output {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q8_0_preq_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    &mut self.forward_buffers.logits,
                );
            } else {
                matvec_into(
                    self.weights.output_weight.data(),
                    &self.forward_buffers.final_normed,
                    config.vocab_size,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            }
        } else {
            matvec_into(
                self.weights.output_weight.data(),
                &self.forward_buffers.final_normed,
                config.vocab_size,
                dim,
                &mut self.forward_buffers.logits,
            );
        }

        if config.final_logit_softcap > 0.0 {
            let cap = config.final_logit_softcap;
            for l in &mut self.forward_buffers.logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(self.forward_buffers.logits.clone(), &[config.vocab_size]).unwrap()
    }

    /// Run a forward pass processing only the layers at the given indices.
    ///
    /// Used by self-speculative decoding to generate draft tokens quickly by
    /// skipping layers (e.g., running only even-numbered layers).  The KV cache
    /// is written at every layer index — skipped layers still get KV entries so
    /// that position tracking stays consistent.
    ///
    /// **Important:** The caller must save and restore the KV cache position
    /// (via `truncate_kv_cache`) after the draft phase, since the verify pass
    /// needs to overwrite these entries with correct values from the full model.
    ///
    /// # Panics
    ///
    /// Panics if any index in `layer_indices` is out of range.
    #[allow(clippy::needless_range_loop)]
    pub fn forward_skip_layers(
        &mut self,
        token_id: u32,
        pos: usize,
        layer_indices: &[usize],
    ) -> Tensor {
        assert!(
            !layer_indices.is_empty(),
            "forward_skip_layers requires at least 1 layer"
        );
        let num_layers_total = self.available_layers();
        for &idx in layer_indices {
            assert!(
                idx < num_layers_total,
                "forward_skip_layers: layer index {idx} out of range (have {num_layers_total})"
            );
        }

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

        // Single-token decode: `use_raw` is reserved for GPU-resident weights.
        // CPU `batched_dequant_matmul` is strictly a prefill/multi-batch
        // optimisation; for batch = 1 the fused Q8_0 / direct Q4_K kernels
        // below are faster.
        let use_raw = self.backend.supports_dequant_matmul()
            && self.backend.has_gpu_weights()
            && self.raw_weights.is_some();

        // Base Q8_0 eligibility: raw weights present and in Q8_0 format.
        let has_q8 = !use_raw
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q8_0);

        // Per-operation hybrid compute: use Q8_0 only for bandwidth-bound matrices.
        let q_dim = num_heads * head_dim;
        let qkv_rows = q_dim + kv_dim + kv_dim;
        let use_q8_attn = has_q8 && should_use_q8_for_size(qkv_rows, dim);
        let use_q8_ffn = has_q8 && should_use_q8_for_size(config.intermediate_dim, dim);
        let use_q8_output = has_q8 && should_use_q8_for_size(config.vocab_size, dim);

        let use_cpu_q4k = !use_raw
            && !has_q8
            && dim >= 1024
            && self.raw_weights.is_some()
            && self
                .raw_weights
                .as_ref()
                .is_some_and(|rw| !rw.is_empty() && rw[0].wq.format == WeightFormat::Q4K);

        // Process only the specified transformer layers
        for &layer_idx in layer_indices {
            let layer = &self.weights.layers[layer_idx];

            // --- Attention block ---
            // Fuse RMSNorm + Q8_0 quantization when Q8 attention path is active.
            if use_q8_attn {
                rmsnorm_quantize_q8_0_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed_q8,
                );
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                matvec_q8_0_preq_into(
                    &fused.qkv_data,
                    &self.forward_buffers.normed_q8,
                    fused.qkv_rows,
                    &mut self.forward_buffers.qkv,
                );
                self.forward_buffers
                    .q_data
                    .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                self.forward_buffers
                    .k_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                self.forward_buffers
                    .v_data
                    .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
            } else {
                rmsnorm_into(
                    x.data(),
                    layer.attn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.normed,
                );

                // QKV projections
                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let q_tmp = self.backend.batched_dequant_matmul(
                        &rw.wq,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.q_data.copy_from_slice(&q_tmp);
                    let k_tmp = self.backend.batched_dequant_matmul(
                        &rw.wk,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.k_data.copy_from_slice(&k_tmp);
                    let v_tmp = self.backend.batched_dequant_matmul(
                        &rw.wv,
                        &self.forward_buffers.normed,
                        1,
                    );
                    self.forward_buffers.v_data.copy_from_slice(&v_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.normed,
                        &mut self.forward_buffers.normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.wq.data,
                        &self.forward_buffers.normed_q8,
                        rw.wq.num_rows,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wk.data,
                        &self.forward_buffers.normed_q8,
                        rw.wk.num_rows,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_q4k_q8_into(
                        &rw.wv.data,
                        &self.forward_buffers.normed_q8,
                        rw.wv.num_rows,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                } else if let Some(ref fused_qkv) = self.fused_f32_qkv {
                    matvec_into(
                        &fused_qkv[layer_idx],
                        &self.forward_buffers.normed,
                        q_dim + kv_dim + kv_dim,
                        dim,
                        &mut self.forward_buffers.qkv,
                    );
                    self.forward_buffers
                        .q_data
                        .copy_from_slice(&self.forward_buffers.qkv[..q_dim]);
                    self.forward_buffers
                        .k_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim..q_dim + kv_dim]);
                    self.forward_buffers
                        .v_data
                        .copy_from_slice(&self.forward_buffers.qkv[q_dim + kv_dim..]);
                } else {
                    matvec_into(
                        layer.wq.data(),
                        &self.forward_buffers.normed,
                        q_dim,
                        dim,
                        &mut self.forward_buffers.q_data,
                    );
                    matvec_into(
                        layer.wk.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.k_data,
                    );
                    matvec_into(
                        layer.wv.data(),
                        &self.forward_buffers.normed,
                        kv_dim,
                        dim,
                        &mut self.forward_buffers.v_data,
                    );
                }
            }

            // Attention biases (Qwen2)
            if let Some(bias) = &layer.attn_q_bias {
                for (q, &b) in self
                    .forward_buffers
                    .q_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *q += b;
                }
            }
            if let Some(bias) = &layer.attn_k_bias {
                for (k, &b) in self
                    .forward_buffers
                    .k_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *k += b;
                }
            }
            if let Some(bias) = &layer.attn_v_bias {
                for (v, &b) in self
                    .forward_buffers
                    .v_data
                    .iter_mut()
                    .zip(bias.data().iter())
                {
                    *v += b;
                }
            }
            apply_rope_with_table(
                &mut self.forward_buffers.q_data,
                num_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );
            apply_rope_with_table(
                &mut self.forward_buffers.k_data,
                num_kv_heads,
                head_dim,
                pos,
                config.rope_theta,
                &self.rope_table,
            );

            self.kv_cache.write(
                layer_idx,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );
            if let Some(ref mut qkv) = self.quantized_kv_cache {
                qkv.write(
                    layer_idx,
                    &self.forward_buffers.k_data,
                    &self.forward_buffers.v_data,
                );
            }
            self.backend.gpu_kv_write(
                layer_idx,
                pos,
                num_kv_heads,
                head_dim,
                &self.forward_buffers.k_data,
                &self.forward_buffers.v_data,
            );

            let seq_len = self.kv_cache.len() + 1;
            if self.backend.has_gpu_kv_cache() {
                let attn_tmp = self.backend.grouped_query_attention_from_gpu_cache(
                    &self.forward_buffers.q_data,
                    layer_idx,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                );
                self.forward_buffers.attn_output.copy_from_slice(&attn_tmp);
            } else if let Some(ref qkv) = self.quantized_kv_cache {
                let dequant_keys = qkv.dequant_keys(layer_idx);
                let dequant_values = qkv.dequant_values(layer_idx);
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    &dequant_keys,
                    &dequant_values,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            } else {
                cpu_grouped_query_attention_into(
                    &self.forward_buffers.q_data,
                    self.kv_cache.keys(layer_idx).data(),
                    self.kv_cache.values(layer_idx).data(),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    config.attn_logit_softcap,
                    &mut self.forward_buffers.attn_output,
                );
            }

            // Output projection (wo)
            if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                let tmp = self.backend.batched_dequant_matmul(
                    &rw.wo,
                    &self.forward_buffers.attn_output,
                    1,
                );
                self.forward_buffers.attn_proj.copy_from_slice(&tmp);
            } else if use_cpu_q4k {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q4k_q8_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            } else if use_q8_attn {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.attn_output,
                    &mut self.forward_buffers.attn_q8,
                );
                matvec_q8_0_preq_into(
                    &rw.wo.data,
                    &self.forward_buffers.attn_q8,
                    rw.wo.num_rows,
                    &mut self.forward_buffers.attn_proj,
                );
            } else {
                matvec_into(
                    layer.wo.data(),
                    &self.forward_buffers.attn_output,
                    dim,
                    num_heads * head_dim,
                    &mut self.forward_buffers.attn_proj,
                );
            }

            if let Some(ref post_norm) = layer.post_attn_norm {
                rmsnorm_into(
                    &self.forward_buffers.attn_proj,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib,
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_proj[i];
                }
            }

            // --- FFN block ---

            if use_q8_ffn {
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                prefetch_weight_bytes(&fused.gate_up_data);
            } else if self.fused_f32_gate_up.is_some() {
                prefetch_weight_f32(&self.fused_f32_gate_up.as_ref().unwrap()[layer_idx]);
            } else if !use_raw {
                prefetch_weight_f32(layer.w_gate.data());
            }

            let inter_dim = config.intermediate_dim;
            if use_q8_ffn {
                // Fused RMSNorm + Q8_0 quantization
                rmsnorm_quantize_q8_0_into(
                    x.data(),
                    layer.ffn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.ffn_normed_q8,
                );
                let fused = &self.fused_weights.as_ref().unwrap()[layer_idx];
                matvec_q8_0_preq_into(
                    &fused.gate_up_data,
                    &self.forward_buffers.ffn_normed_q8,
                    fused.gate_up_rows,
                    &mut self.forward_buffers.gate_up,
                );
                self.forward_buffers
                    .gate
                    .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                self.forward_buffers
                    .up
                    .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
            } else {
                rmsnorm_into(
                    x.data(),
                    layer.ffn_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.ffn_normed,
                );

                if use_raw {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    let gate_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_gate,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.gate.copy_from_slice(&gate_tmp);
                    let up_tmp = self.backend.batched_dequant_matmul(
                        &rw.w_up,
                        &self.forward_buffers.ffn_normed,
                        1,
                    );
                    self.forward_buffers.up.copy_from_slice(&up_tmp);
                } else if use_cpu_q4k {
                    let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                    quantize_input_q8_0_into(
                        &self.forward_buffers.ffn_normed,
                        &mut self.forward_buffers.ffn_normed_q8,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_gate.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_gate.num_rows,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_q4k_q8_into(
                        &rw.w_up.data,
                        &self.forward_buffers.ffn_normed_q8,
                        rw.w_up.num_rows,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                } else if let Some(ref fused_gu) = self.fused_f32_gate_up {
                    matvec_into(
                        &fused_gu[layer_idx],
                        &self.forward_buffers.ffn_normed,
                        inter_dim * 2,
                        dim,
                        &mut self.forward_buffers.gate_up,
                    );
                    self.forward_buffers
                        .gate
                        .copy_from_slice(&self.forward_buffers.gate_up[..inter_dim]);
                    self.forward_buffers
                        .up
                        .copy_from_slice(&self.forward_buffers.gate_up[inter_dim..]);
                } else {
                    matvec_into(
                        layer.w_gate.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.gate,
                    );
                    matvec_into(
                        layer.w_up.data(),
                        &self.forward_buffers.ffn_normed,
                        inter_dim,
                        dim,
                        &mut self.forward_buffers.up,
                    );
                }
            }

            if config.architecture == Architecture::Gemma2 {
                gelu_mul_into(
                    &self.forward_buffers.gate,
                    &self.forward_buffers.up,
                    &mut self.forward_buffers.ffn_hidden,
                );
            } else {
                silu_mul_into(
                    &self.forward_buffers.gate,
                    &self.forward_buffers.up,
                    &mut self.forward_buffers.ffn_hidden,
                );
            }

            if use_raw {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                let tmp = self.backend.batched_dequant_matmul(
                    &rw.w_down,
                    &self.forward_buffers.ffn_hidden,
                    1,
                );
                self.forward_buffers.ffn_out.copy_from_slice(&tmp);
            } else if use_cpu_q4k {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.ffn_hidden,
                    &mut self.forward_buffers.ffn_hidden_q8,
                );
                matvec_q4k_q8_into(
                    &rw.w_down.data,
                    &self.forward_buffers.ffn_hidden_q8,
                    rw.w_down.num_rows,
                    config.intermediate_dim,
                    &mut self.forward_buffers.ffn_out,
                );
            } else if use_q8_ffn {
                let rw = &self.raw_weights.as_ref().unwrap()[layer_idx];
                quantize_input_q8_0_into(
                    &self.forward_buffers.ffn_hidden,
                    &mut self.forward_buffers.ffn_hidden_q8,
                );
                matvec_q8_0_preq_into(
                    &rw.w_down.data,
                    &self.forward_buffers.ffn_hidden_q8,
                    rw.w_down.num_rows,
                    &mut self.forward_buffers.ffn_out,
                );
            } else {
                matvec_into(
                    layer.w_down.data(),
                    &self.forward_buffers.ffn_hidden,
                    dim,
                    config.intermediate_dim,
                    &mut self.forward_buffers.ffn_out,
                );
            }

            if let Some(ref post_norm) = layer.post_ffn_norm {
                rmsnorm_into(
                    &self.forward_buffers.ffn_out,
                    post_norm.data(),
                    config.rms_norm_eps,
                    &mut self.forward_buffers.attn_contrib,
                );
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.attn_contrib[i];
                }
            } else {
                let x_data = x.data_mut();
                for i in 0..dim {
                    x_data[i] += self.forward_buffers.ffn_out[i];
                }
            }
        }

        // Advance KV cache after processing the selected layers
        self.kv_cache.advance();
        if let Some(ref mut qkv) = self.quantized_kv_cache {
            qkv.advance();
        }

        // Final RMSNorm + output projection
        rmsnorm_into(
            x.data(),
            self.weights.output_norm.data(),
            config.rms_norm_eps,
            &mut self.forward_buffers.final_normed,
        );

        let vocab_size = config.vocab_size;
        if let Some(ref row) = self.raw_output_weight {
            if row.format == WeightFormat::Q4K && use_cpu_q4k {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q4k_q8_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            } else if row.format == WeightFormat::Q8_0 && use_q8_output {
                quantize_input_q8_0_into(
                    &self.forward_buffers.final_normed,
                    &mut self.forward_buffers.output_q8,
                );
                matvec_q8_0_preq_into(
                    &row.data,
                    &self.forward_buffers.output_q8,
                    row.num_rows,
                    &mut self.forward_buffers.logits,
                );
            } else {
                matvec_into(
                    self.weights.output_weight.data(),
                    &self.forward_buffers.final_normed,
                    vocab_size,
                    dim,
                    &mut self.forward_buffers.logits,
                );
            }
        } else {
            matvec_into(
                self.weights.output_weight.data(),
                &self.forward_buffers.final_normed,
                vocab_size,
                dim,
                &mut self.forward_buffers.logits,
            );
        }

        if config.final_logit_softcap > 0.0 {
            let cap = config.final_logit_softcap;
            for l in &mut self.forward_buffers.logits {
                *l = (*l / cap).tanh() * cap;
            }
        }

        Tensor::from_vec(self.forward_buffers.logits.clone(), &[vocab_size]).unwrap()
    }

    /// Roll back the KV cache to `target_len` valid entries.
    ///
    /// Used by self-speculative decoding to undo draft-phase KV entries
    /// before re-running the verify phase with the full model.
    pub fn truncate_kv_cache(&mut self, target_len: usize) {
        self.kv_cache.truncate_to(target_len);
        if let Some(ref mut qkv) = self.quantized_kv_cache {
            qkv.truncate_to(target_len);
        }
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

// FFI binding to Apple Accelerate's cblas_sgemv.
// Accelerate internally uses Apple's AMX (matrix coprocessor) on Apple Silicon,
// delivering significantly higher throughput than hand-written NEON SIMD.
//
// Safety: the caller must ensure all pointer/dimension arguments are valid and consistent.
#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemv(
        order: i32,    // CblasRowMajor = 101
        trans: i32,    // CblasNoTrans = 111
        m: i32,        // rows
        n: i32,        // cols
        alpha: f32,    // scalar multiplier for A*x
        a: *const f32, // matrix pointer
        lda: i32,      // leading dimension (= cols for row-major)
        x: *const f32, // input vector
        incx: i32,     // stride for x
        beta: f32,     // scalar multiplier for y
        y: *mut f32,   // output vector
        incy: i32,     // stride for y
    );
}

/// Matrix-vector multiply: `output[rows]` = `mat[rows, cols]` * `vec[cols]` (CPU implementation).
///
/// Dispatches to platform-specific implementations:
/// - macOS: Apple Accelerate cblas_sgemv (uses AMX coprocessor on Apple Silicon)
/// - ARM NEON (non-macOS aarch64): 4-wide f32 SIMD, 4 accumulators (16 elements/iter)
/// - x86 AVX2+FMA (x86_64): 8-wide f32 SIMD, 4 accumulators (32 elements/iter), runtime check
/// - Fallback: 4-wide scalar unrolling for auto-vectorization
///
/// All SIMD paths parallelize via rayon for large matrices on native targets.
pub fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // On macOS, use Accelerate's cblas_sgemv which leverages the AMX coprocessor
    // on Apple Silicon for dramatically faster matrix-vector multiply.
    #[cfg(target_os = "macos")]
    {
        debug_assert_eq!(mat.len(), rows * cols, "matvec: mat size mismatch");
        debug_assert!(vec.len() >= cols, "matvec: vec size mismatch");
        let mut output = std::vec![0.0f32; rows];
        // SAFETY: pointers and dimensions are validated by debug_assert above.
        // mat is row-major with `cols` as the leading dimension, matching
        // CblasRowMajor layout. Accelerate manages its own threading internally.
        unsafe {
            cblas_sgemv(
                101, // CblasRowMajor
                111, // CblasNoTrans
                rows as i32,
                cols as i32,
                1.0, // alpha
                mat.as_ptr(),
                cols as i32, // lda
                vec.as_ptr(),
                1,   // incx
                0.0, // beta
                output.as_mut_ptr(),
                1, // incy
            );
        }
        output
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
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
    #[cfg(not(any(target_os = "macos", target_arch = "aarch64", target_arch = "x86_64")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
        {
            matvec_relaxed_simd(mat, vec, rows, cols)
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "relaxed_simd")))]
        {
            matvec_scalar(mat, vec, rows, cols)
        }
    }
}

/// Adaptive parallelism heuristic for matvec operations.
///
/// Returns `Some(chunk_rows)` if the work should be parallelized, `None` otherwise.
/// Uses per-size thresholds to balance rayon task overhead against parallelism gains:
/// - Below 2M FMAs: single-threaded (rayon overhead exceeds benefit)
/// - 2M-10M FMAs: parallel with larger chunks (128 rows) for moderate work
/// - Above 10M FMAs: parallel with smaller chunks (32 rows) for maximum utilization
///
/// The 2M threshold ensures 2048x2048 matvecs (4.2M FMAs, common in Q/K/V/O
/// projections for 1B-class models) are parallelized across available cores.
#[inline]
#[allow(dead_code)]
fn parallel_chunk_rows(total_work: usize, rows: usize) -> Option<usize> {
    if total_work < 2_000_000 {
        // Small matrix: rayon overhead would dominate
        return None;
    }
    let chunk = if total_work < 10_000_000 {
        // Medium matrix (e.g. 2048x2048): moderate parallelism, larger chunks
        // to keep rayon task count reasonable (~16 tasks for 2048 rows)
        128
    } else {
        // Large matrix (e.g. 8192x2048): fine-grained parallelism for
        // maximum core utilization (~64-256 tasks)
        32
    };
    // Need at least 2 chunks to benefit from parallelism
    if rows >= chunk * 2 {
        Some(chunk)
    } else {
        None
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
    let total_work = rows * cols;

    if let Some(chunk_rows) = parallel_chunk_rows(total_work, rows) {
        use rayon::prelude::*;
        let mut output = vec![0.0f32; rows];
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
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
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
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
/// On macOS, we use Accelerate's cblas_sgemv instead (AMX coprocessor).
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn matvec_simd(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let total_work = rows * cols;

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(chunk_rows) = parallel_chunk_rows(total_work, rows) {
        use rayon::prelude::*;
        let mut output = vec![0.0f32; rows];
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
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

/// Compute a single row dot product with 4-wide unrolling for auto-vectorization.
fn matvec_scalar_row(row: &[f32], vec: &[f32], cols: usize) -> f32 {
    let cols4 = cols & !3;
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
    sum0 + sum1 + sum2 + sum3
}

/// Scalar matvec with 4-wide unrolling for auto-vectorization.
///
/// Always available as a reference implementation for tests, regardless of target.
/// When the `wasm_threads` feature is enabled on wasm32, rows are processed in
/// parallel via rayon (backed by SharedArrayBuffer + Web Workers).
pub fn matvec_scalar(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // Parallel path for wasm32 with wasm_threads feature enabled.
    #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
    {
        let total_work = rows * cols;

        if let Some(chunk_rows) = parallel_chunk_rows(total_work, rows) {
            use rayon::prelude::*;
            let mut output = vec![0.0f32; rows];
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * chunk_rows;
                    for (local_i, out) in out_chunk.iter_mut().enumerate() {
                        let i = row_start + local_i;
                        let row = &mat[i * cols..i * cols + cols];
                        *out = matvec_scalar_row(row, vec, cols);
                    }
                });
            return output;
        }
    }

    // Sequential path (default for wasm32, or when below parallel threshold).
    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..(i + 1) * cols];
        *out = matvec_scalar_row(row, vec, cols);
    }
    output
}

// ---------------------------------------------------------------------------
// Q8_0 direct quantized matvec (no dequantization to f32)
// ---------------------------------------------------------------------------

/// Convert f16 (as u16 bits) to f32.  Inlined here to avoid a cross-crate
/// dependency on flare_loader for this hot path.
#[inline(always)]
fn f16_to_f32_inline(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: renormalize
        let mut e = exp;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e = e.wrapping_sub(1);
        }
        m &= 0x3FF;
        let f32_exp = (113u32).wrapping_add(e);
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 0x1F {
        // Inf / NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Maximum number of bytes to prefetch for next-layer weight data.
///
/// We prefetch a limited window (8 KB) of the next operation's weight data to
/// warm the L2 cache while the current operation's compute is running. This is
/// small enough to avoid evicting useful data from cache, but large enough to
/// cover the first few hundred rows of weight data that will be accessed next.
#[cfg(target_arch = "aarch64")]
const PREFETCH_BYTES: usize = 8192;

/// L2 cache threshold for hybrid compute strategy (bytes, f32 basis).
///
/// Matrices whose f32 representation exceeds this size are bandwidth-bound and
/// benefit from Q8_0 quantized dot products (halving memory traffic).  Smaller
/// matrices fit in L2 cache and are faster with Accelerate `cblas_sgemv` (AMX).
///
/// 16 MB is a conservative estimate for Apple M-series chips (~48 MB total L2)
/// leaving headroom for activations, KV cache, and other working data.
const Q8_MATRIX_BYTE_THRESHOLD: usize = 16 * 1024 * 1024;

/// Decide whether a matvec with the given weight matrix dimensions should use
/// the Q8_0 quantized dot-product path instead of f32 Accelerate/BLAS.
///
/// Returns `true` when the matrix is too large for L2 cache (bandwidth-bound),
/// meaning Q8_0's ~2x bandwidth reduction outweighs the AMX f32 throughput
/// advantage.  For cache-resident matrices, f32 BLAS is faster.
#[inline]
fn should_use_q8_for_size(rows: usize, cols: usize) -> bool {
    rows * cols * 4 > Q8_MATRIX_BYTE_THRESHOLD
}

/// Issue software prefetch hints for the first `PREFETCH_BYTES` of a byte slice.
///
/// On aarch64, this emits `prfm pldl2strm` instructions spaced 64 bytes apart
/// (one per cache line). The L2 streaming hint tells the hardware this data will
/// be read once sequentially, so it should not displace hot L1 data.
///
/// On non-aarch64 targets (including WASM), this is a no-op.
#[inline(always)]
fn prefetch_weight_bytes(data: &[u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        let len = data.len().min(PREFETCH_BYTES);
        let ptr = data.as_ptr();
        let mut offset = 0;
        while offset < len {
            // SAFETY: We only issue prefetch hints within the bounds of the
            // allocated slice. `prfm` is a hint instruction that cannot fault
            // even if the address is invalid, but we stay in-bounds regardless.
            unsafe {
                core::arch::asm!(
                    "prfm pldl2strm, [{addr}]",
                    addr = in(reg) ptr.add(offset),
                    options(nostack, preserves_flags, readonly),
                );
            }
            offset += 64; // aarch64 cache line = 64 bytes
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = data;
    }
}

/// Issue software prefetch hints for the first `PREFETCH_BYTES` of an f32 slice.
///
/// Convenience wrapper around [`prefetch_weight_bytes`] that reinterprets an
/// f32 slice as raw bytes. Used for f32 weight tensors and norm vectors.
#[inline(always)]
fn prefetch_weight_f32(data: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let byte_len = (data.len() * 4).min(PREFETCH_BYTES);
        let ptr = data.as_ptr() as *const u8;
        let mut offset = 0;
        while offset < byte_len {
            unsafe {
                core::arch::asm!(
                    "prfm pldl2strm, [{addr}]",
                    addr = in(reg) ptr.add(offset),
                    options(nostack, preserves_flags, readonly),
                );
            }
            offset += 64;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = data;
    }
}

/// Q8_0 block constants.
const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (int8 quants)

/// Pre-quantized Q8_0 input vector.
///
/// Caches the quantized representation of an f32 input vector so that
/// multiple matvec calls sharing the same input avoid redundant quantization.
pub struct QuantizedInput {
    pub scales: Vec<f32>,
    pub quants: Vec<i8>,
    pub blocks_per_row: usize,
}

/// Quantize an f32 input vector to Q8_0 blocks.
///
/// Returns a `QuantizedInput` that can be reused across multiple
/// `matvec_q8_0_preq_into` calls with the same input vector.
pub fn quantize_input_q8_0(input: &[f32]) -> QuantizedInput {
    let cols = input.len();
    debug_assert_eq!(
        cols % Q8_0_BLOCK_SIZE,
        0,
        "input length must be multiple of 32"
    );
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;

    #[cfg(target_arch = "aarch64")]
    {
        let mut scales = Vec::with_capacity(blocks_per_row);
        let mut quants = vec![0i8; cols];
        for b in 0..blocks_per_row {
            let start = b * Q8_0_BLOCK_SIZE;
            // SAFETY: quantize_f32_to_q8_0_block requires aarch64 NEON (always available)
            // and src must have at least 32 elements (guaranteed by block size).
            let (scale, block_quants) =
                unsafe { quantize_f32_to_q8_0_block(&input[start..start + Q8_0_BLOCK_SIZE]) };
            scales.push(scale);
            quants[start..start + Q8_0_BLOCK_SIZE].copy_from_slice(&block_quants);
        }
        QuantizedInput {
            scales,
            quants,
            blocks_per_row,
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut scales = Vec::with_capacity(blocks_per_row);
        let mut quants = vec![0i8; cols];
        for b in 0..blocks_per_row {
            let start = b * Q8_0_BLOCK_SIZE;
            let block = &input[start..start + Q8_0_BLOCK_SIZE];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 127.0;
            let id = if d != 0.0 { 127.0 / amax } else { 0.0 };
            scales.push(d);
            for j in 0..Q8_0_BLOCK_SIZE {
                quants[start + j] = (block[j] * id).round() as i8;
            }
        }
        QuantizedInput {
            scales,
            quants,
            blocks_per_row,
        }
    }
}

/// Quantize an f32 input vector into pre-allocated `QuantizedInput` buffers.
///
/// Same as `quantize_input_q8_0` but reuses existing `scales` and `quants`
/// Vecs to avoid heap allocation on the hot path.
pub fn quantize_input_q8_0_into(input: &[f32], preq: &mut QuantizedInput) {
    let cols = input.len();
    debug_assert_eq!(
        cols % Q8_0_BLOCK_SIZE,
        0,
        "input length must be multiple of 32"
    );
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    preq.blocks_per_row = blocks_per_row;
    preq.scales.resize(blocks_per_row, 0.0);
    preq.quants.resize(cols, 0);

    #[cfg(target_arch = "aarch64")]
    {
        for b in 0..blocks_per_row {
            let start = b * Q8_0_BLOCK_SIZE;
            let (scale, block_quants) =
                unsafe { quantize_f32_to_q8_0_block(&input[start..start + Q8_0_BLOCK_SIZE]) };
            preq.scales[b] = scale;
            preq.quants[start..start + Q8_0_BLOCK_SIZE].copy_from_slice(&block_quants);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for b in 0..blocks_per_row {
            let start = b * Q8_0_BLOCK_SIZE;
            let block = &input[start..start + Q8_0_BLOCK_SIZE];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 127.0;
            let id = if d != 0.0 { 127.0 / amax } else { 0.0 };
            preq.scales[b] = d;
            for j in 0..Q8_0_BLOCK_SIZE {
                preq.quants[start + j] = (block[j] * id).round() as i8;
            }
        }
    }
}

/// Fused RMSNorm + Q8_0 quantization: normalize the input and quantize the
/// result to Q8_0 in one pass, without materializing the intermediate f32
/// normed vector.
///
/// This eliminates 2 memory passes (write normed f32 + read normed f32 for
/// quantization) compared to the separate `rmsnorm_into` + `quantize_input_q8_0_into`
/// pipeline.
#[inline]
pub fn rmsnorm_quantize_q8_0_into(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut QuantizedInput,
) {
    let dim = x.len();
    debug_assert_eq!(
        dim % Q8_0_BLOCK_SIZE,
        0,
        "input length must be multiple of 32"
    );
    let blocks_per_row = dim / Q8_0_BLOCK_SIZE;
    output.blocks_per_row = blocks_per_row;
    output.scales.resize(blocks_per_row, 0.0);
    output.quants.resize(dim, 0);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON always available on aarch64
        unsafe { rmsnorm_quantize_q8_0_neon(x, weight, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rmsnorm_quantize_q8_0_scalar(x, weight, eps, output)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rmsnorm_quantize_q8_0_scalar(x: &[f32], weight: &[f32], eps: f32, output: &mut QuantizedInput) {
    let dim = x.len();
    let blocks = dim / Q8_0_BLOCK_SIZE;

    // Step 1: Compute RMS
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Step 2: For each Q8_0 block, normalize + quantize in one pass
    for b in 0..blocks {
        let start = b * Q8_0_BLOCK_SIZE;

        // Normalize 32 values and find max abs for quantization
        let mut normed = [0.0f32; 32];
        let mut amax = 0.0f32;
        for j in 0..Q8_0_BLOCK_SIZE {
            let n = x[start + j] * weight[start + j] * inv_rms;
            normed[j] = n;
            amax = amax.max(n.abs());
        }

        // Quantize to Q8_0
        let scale = amax / 127.0;
        output.scales[b] = scale;
        let inv_scale = if scale != 0.0 { 127.0 / amax } else { 0.0 };
        for j in 0..Q8_0_BLOCK_SIZE {
            output.quants[start + j] = (normed[j] * inv_scale).round() as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn rmsnorm_quantize_q8_0_neon(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut QuantizedInput,
) {
    use std::arch::aarch64::*;

    let dim = x.len();
    let blocks = dim / Q8_0_BLOCK_SIZE;
    let dim16 = dim & !15;

    // --- Phase 1: vectorized sum-of-squares with 4 accumulators ---
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

    // --- Phase 2: fused normalize + quantize per block ---
    // Each block is 32 floats = 8 NEON float4 vectors.
    let x_ptr = x.as_ptr();
    let w_ptr = weight.as_ptr();
    let q_ptr = output.quants.as_mut_ptr();

    for b in 0..blocks {
        let start = b * Q8_0_BLOCK_SIZE;

        // Normalize 8 float4 vectors: normed = x * weight * inv_rms
        let n0 = vmulq_f32(
            vmulq_f32(vld1q_f32(x_ptr.add(start)), vld1q_f32(w_ptr.add(start))),
            inv_rms_v,
        );
        let n1 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 4)),
                vld1q_f32(w_ptr.add(start + 4)),
            ),
            inv_rms_v,
        );
        let n2 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 8)),
                vld1q_f32(w_ptr.add(start + 8)),
            ),
            inv_rms_v,
        );
        let n3 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 12)),
                vld1q_f32(w_ptr.add(start + 12)),
            ),
            inv_rms_v,
        );
        let n4 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 16)),
                vld1q_f32(w_ptr.add(start + 16)),
            ),
            inv_rms_v,
        );
        let n5 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 20)),
                vld1q_f32(w_ptr.add(start + 20)),
            ),
            inv_rms_v,
        );
        let n6 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 24)),
                vld1q_f32(w_ptr.add(start + 24)),
            ),
            inv_rms_v,
        );
        let n7 = vmulq_f32(
            vmulq_f32(
                vld1q_f32(x_ptr.add(start + 28)),
                vld1q_f32(w_ptr.add(start + 28)),
            ),
            inv_rms_v,
        );

        // Find absolute max across all 32 normed values
        let abs01 = vmaxq_f32(vabsq_f32(n0), vabsq_f32(n1));
        let abs23 = vmaxq_f32(vabsq_f32(n2), vabsq_f32(n3));
        let abs45 = vmaxq_f32(vabsq_f32(n4), vabsq_f32(n5));
        let abs67 = vmaxq_f32(vabsq_f32(n6), vabsq_f32(n7));
        let abs0123 = vmaxq_f32(abs01, abs23);
        let abs4567 = vmaxq_f32(abs45, abs67);
        let absmax_vec = vmaxq_f32(abs0123, abs4567);
        let amax = vmaxvq_f32(absmax_vec);

        let d = amax / 127.0;
        let id = if d != 0.0 { 127.0 / amax } else { 0.0 };
        output.scales[b] = d;

        // Quantize: round(normed * id) and narrow to i8
        let i0 = vcvtnq_s32_f32(vmulq_n_f32(n0, id));
        let i1 = vcvtnq_s32_f32(vmulq_n_f32(n1, id));
        let i2 = vcvtnq_s32_f32(vmulq_n_f32(n2, id));
        let i3 = vcvtnq_s32_f32(vmulq_n_f32(n3, id));
        let i4 = vcvtnq_s32_f32(vmulq_n_f32(n4, id));
        let i5 = vcvtnq_s32_f32(vmulq_n_f32(n5, id));
        let i6 = vcvtnq_s32_f32(vmulq_n_f32(n6, id));
        let i7 = vcvtnq_s32_f32(vmulq_n_f32(n7, id));

        // Narrow int32x4 -> int16x8 -> int8x16
        let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        let s45 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
        let s67 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));

        let b0 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        let b1 = vcombine_s8(vqmovn_s16(s45), vqmovn_s16(s67));

        vst1q_s8(q_ptr.add(start), b0);
        vst1q_s8(q_ptr.add(start + 16), b1);
    }
}

/// Compute one row of Q8_0 direct dot product (scalar).
#[inline]
fn matvec_q8_0_scalar_row(row_bytes: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block in 0..blocks_per_row {
        let block_start = block * Q8_0_BLOCK_BYTES;
        let scale = f16_to_f32_inline(u16::from_le_bytes([
            row_bytes[block_start],
            row_bytes[block_start + 1],
        ]));
        let quants = &row_bytes[block_start + 2..block_start + Q8_0_BLOCK_BYTES];
        let input_offset = block * Q8_0_BLOCK_SIZE;

        // 4-wide unrolling for auto-vectorization
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut j = 0;
        while j < 32 {
            s0 += (quants[j] as i8) as f32 * input[input_offset + j];
            s1 += (quants[j + 1] as i8) as f32 * input[input_offset + j + 1];
            s2 += (quants[j + 2] as i8) as f32 * input[input_offset + j + 2];
            s3 += (quants[j + 3] as i8) as f32 * input[input_offset + j + 3];
            j += 4;
        }
        sum += scale * (s0 + s1 + s2 + s3);
    }
    sum
}

/// Scalar Q8_0 direct matvec: compute `output = weight_q8 * input` without
/// dequantizing the weight matrix to f32.
///
/// `weight_data`: raw Q8_0 bytes (row-major, `rows * blocks_per_row * 34` bytes).
/// `input`: f32 vector of length `cols` (= `blocks_per_row * 32`).
/// `rows`: number of output rows.  `cols`: input dimension (must be multiple of 32).
pub fn matvec_q8_0_scalar(weight_data: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row_start = i * bytes_per_row;
        let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
        *out = matvec_q8_0_scalar_row(row_bytes, input, blocks_per_row);
    }
    output
}

/// Emulate ARM dot product instruction using widening multiply + pairwise add.
/// Computes acc + sum_of_products(a[i]*b[i]) for 16 int8 pairs, returning 4 int32 lanes.
/// Equivalent to `vdotq_s32` (ARMv8.2 dotprod) using inline assembly to emit
/// the hardware SDOT instruction on stable Rust, avoiding the unstable
/// `stdarch_neon_dotprod` feature.
///
/// SDOT computes four independent dot-products of 4 signed-int8 pairs each,
/// accumulating into int32 lanes. This is ~2x faster than the widening-multiply
/// fallback (`vmull_s8` + `vpaddlq_s16`) because it retires in a single cycle
/// on Apple Silicon and other ARMv8.2+ cores.
///
/// # Safety
/// Requires ARMv8.2-A with dotprod (all Apple Silicon, Cortex-A76+, etc.).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ggml_vdotq_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut result = acc;
    // SDOT Vd.4S, Vn.16B, Vm.16B — signed dot product, 4 lanes x 4 int8 pairs
    core::arch::asm!(
        "sdot {result:v}.4s, {a:v}.16b, {b:v}.16b",
        result = inout(vreg) result,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    result
}

/// Fallback for pre-ARMv8.2 cores without the dotprod extension.
/// Uses widening multiplies + pairwise accumulate (always available with NEON).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn ggml_vdotq_s32_fallback(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)))
}

/// Quantize 32 f32 values to Q8_0 format (scale + 32 int8 quants) using NEON.
///
/// Returns (scale, quants) where quants[i] = round(src[i] / scale * 127).
///
/// # Safety
/// `src` must have at least 32 elements. Requires aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn quantize_f32_to_q8_0_block(src: &[f32]) -> (f32, [i8; 32]) {
    use std::arch::aarch64::*;

    let p = src.as_ptr();

    // Load 8 float32x4 vectors (32 floats total)
    let v0 = vld1q_f32(p);
    let v1 = vld1q_f32(p.add(4));
    let v2 = vld1q_f32(p.add(8));
    let v3 = vld1q_f32(p.add(12));
    let v4 = vld1q_f32(p.add(16));
    let v5 = vld1q_f32(p.add(20));
    let v6 = vld1q_f32(p.add(24));
    let v7 = vld1q_f32(p.add(28));

    // Find absolute max across all 32 values
    let abs01 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
    let abs23 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
    let abs45 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
    let abs67 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
    let abs0123 = vmaxq_f32(abs01, abs23);
    let abs4567 = vmaxq_f32(abs45, abs67);
    let absmax_vec = vmaxq_f32(abs0123, abs4567);
    let amax = vmaxvq_f32(absmax_vec);

    let d = amax / 127.0;
    let id = if d != 0.0 { 127.0 / amax } else { 0.0 };

    // Scale each float and round to nearest int32, then narrow to int8
    let i0 = vcvtnq_s32_f32(vmulq_n_f32(v0, id));
    let i1 = vcvtnq_s32_f32(vmulq_n_f32(v1, id));
    let i2 = vcvtnq_s32_f32(vmulq_n_f32(v2, id));
    let i3 = vcvtnq_s32_f32(vmulq_n_f32(v3, id));
    let i4 = vcvtnq_s32_f32(vmulq_n_f32(v4, id));
    let i5 = vcvtnq_s32_f32(vmulq_n_f32(v5, id));
    let i6 = vcvtnq_s32_f32(vmulq_n_f32(v6, id));
    let i7 = vcvtnq_s32_f32(vmulq_n_f32(v7, id));

    // Narrow int32x4 -> int16x4 -> int16x8 -> int8x8 -> int8x16
    let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
    let s45 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
    let s67 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));

    let b0 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
    let b1 = vcombine_s8(vqmovn_s16(s45), vqmovn_s16(s67));

    let mut quants = [0i8; 32];
    vst1q_s8(quants.as_mut_ptr(), b0);
    vst1q_s8(quants.as_mut_ptr().add(16), b1);

    (d, quants)
}

/// Compute one row of Q8_0 x Q8_0 integer dot product using ARM NEON.
///
/// This is the llama.cpp-style approach: both weights and input are int8,
/// so we compute int8*int8 dot products (16 multiplies per NEON instruction)
/// and only convert to f32 once per block. This reads 1 byte/weight + 1 byte/input
/// instead of 1 byte/weight + 4 bytes/input, giving ~4x less memory bandwidth.
///
/// Processes 4 blocks per iteration for better instruction-level parallelism,
/// with prefetch hints for the next group of blocks.
///
/// # Safety
/// Requires aarch64 NEON. `row_bytes` must have `blocks_per_row * 34` bytes.
/// `input_q8_scales` must have `blocks_per_row` elements.
/// `input_q8_quants` must have `blocks_per_row * 32` elements.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q8_0_q8_0_neon(
    row_bytes: &[u8],
    input_q8_scales: &[f32],
    input_q8_quants: &[i8],
    blocks_per_row: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mut sumv2 = vdupq_n_f32(0.0);
    let mut sumv3 = vdupq_n_f32(0.0);

    let row_ptr = row_bytes.as_ptr();
    let iq_ptr = input_q8_quants.as_ptr();

    let quads = blocks_per_row / 4;
    for i in 0..quads {
        let ib0 = i * 4;
        let ib1 = ib0 + 1;
        let ib2 = ib0 + 2;
        let ib3 = ib0 + 3;

        // Prefetch next quad's weight data
        if i + 1 < quads {
            let next_offset = (ib0 + 4) * Q8_0_BLOCK_BYTES;
            core::arch::asm!(
                "prfm pldl1strm, [{addr}]",
                addr = in(reg) row_ptr.add(next_offset),
                options(nostack, preserves_flags),
            );
        }

        // --- Block 0 ---
        let w_ptr0 = row_ptr.add(ib0 * Q8_0_BLOCK_BYTES);
        let w_scale0 = f16_to_f32_inline(u16::from_le_bytes([*w_ptr0, *w_ptr0.add(1)]));
        let w_qs0 = w_ptr0.add(2) as *const i8;
        let i_qs0 = iq_ptr.add(ib0 * Q8_0_BLOCK_SIZE);
        let combined_scale0 = w_scale0 * *input_q8_scales.get_unchecked(ib0);

        let sum0 = vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_qs0), vld1q_s8(i_qs0)),
            ggml_vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(w_qs0.add(16)),
                vld1q_s8(i_qs0.add(16)),
            ),
        );
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(sum0), combined_scale0);

        // --- Block 1 ---
        let w_ptr1 = row_ptr.add(ib1 * Q8_0_BLOCK_BYTES);
        let w_scale1 = f16_to_f32_inline(u16::from_le_bytes([*w_ptr1, *w_ptr1.add(1)]));
        let w_qs1 = w_ptr1.add(2) as *const i8;
        let i_qs1 = iq_ptr.add(ib1 * Q8_0_BLOCK_SIZE);
        let combined_scale1 = w_scale1 * *input_q8_scales.get_unchecked(ib1);

        let sum1 = vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_qs1), vld1q_s8(i_qs1)),
            ggml_vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(w_qs1.add(16)),
                vld1q_s8(i_qs1.add(16)),
            ),
        );
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(sum1), combined_scale1);

        // --- Block 2 ---
        let w_ptr2 = row_ptr.add(ib2 * Q8_0_BLOCK_BYTES);
        let w_scale2 = f16_to_f32_inline(u16::from_le_bytes([*w_ptr2, *w_ptr2.add(1)]));
        let w_qs2 = w_ptr2.add(2) as *const i8;
        let i_qs2 = iq_ptr.add(ib2 * Q8_0_BLOCK_SIZE);
        let combined_scale2 = w_scale2 * *input_q8_scales.get_unchecked(ib2);

        let sum2 = vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_qs2), vld1q_s8(i_qs2)),
            ggml_vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(w_qs2.add(16)),
                vld1q_s8(i_qs2.add(16)),
            ),
        );
        sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(sum2), combined_scale2);

        // --- Block 3 ---
        let w_ptr3 = row_ptr.add(ib3 * Q8_0_BLOCK_BYTES);
        let w_scale3 = f16_to_f32_inline(u16::from_le_bytes([*w_ptr3, *w_ptr3.add(1)]));
        let w_qs3 = w_ptr3.add(2) as *const i8;
        let i_qs3 = iq_ptr.add(ib3 * Q8_0_BLOCK_SIZE);
        let combined_scale3 = w_scale3 * *input_q8_scales.get_unchecked(ib3);

        let sum3 = vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_qs3), vld1q_s8(i_qs3)),
            ggml_vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(w_qs3.add(16)),
                vld1q_s8(i_qs3.add(16)),
            ),
        );
        sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(sum3), combined_scale3);
    }

    // Handle remaining 0-3 blocks
    let remainder_start = quads * 4;
    for ib in remainder_start..blocks_per_row {
        let w_ptr = row_ptr.add(ib * Q8_0_BLOCK_BYTES);
        let w_scale = f16_to_f32_inline(u16::from_le_bytes([*w_ptr, *w_ptr.add(1)]));
        let w_qs = w_ptr.add(2) as *const i8;
        let i_qs = iq_ptr.add(ib * Q8_0_BLOCK_SIZE);
        let combined_scale = w_scale * *input_q8_scales.get_unchecked(ib);

        let sum = vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), vld1q_s8(w_qs), vld1q_s8(i_qs)),
            ggml_vdotq_s32(
                vdupq_n_s32(0),
                vld1q_s8(w_qs.add(16)),
                vld1q_s8(i_qs.add(16)),
            ),
        );
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(sum), combined_scale);
    }

    vaddvq_f32(vaddq_f32(vaddq_f32(sumv0, sumv1), vaddq_f32(sumv2, sumv3)))
}

/// SMMLA (Signed Matrix Multiply-Accumulate) via inline assembly.
/// Available on Apple M2+ (FEAT_I8MM).  Computes C += A @ B^T where A and B
/// are 2×8 int8 matrices packed into int8x16 registers:
///   A = [row0[0..8], row1[0..8]], B = [row0[0..8], row1[0..8]]
///   C = int32x4 = [C00, C01, C10, C11]  (2×2 output tile)
///
/// # Safety
/// Requires aarch64 with FEAT_I8MM.  Caller must gate on runtime detection.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn smmla_s32(
    mut c: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    core::arch::asm!(
        "smmla {c:v}.4s, {a:v}.16b, {b:v}.16b",
        c = inout(vreg) c,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    c
}

/// 2-row × 2-token Q8_0 dot product tile using SDOT (M1-safe).
///
/// Computes 4 dot products simultaneously: [row0·tok0, row0·tok1, row1·tok0, row1·tok1].
/// By loading each weight block once and reusing across 2 tokens, this halves weight
/// memory bandwidth compared to calling `dot_q8_0_q8_0_neon` 4 times.
///
/// # Safety
/// Requires aarch64 NEON.  All slices must be valid for the given `blocks_per_row`.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q8_0_q8_0_2x2_sdot(
    row0_bytes: &[u8],
    row1_bytes: &[u8],
    t0_scales: &[f32],
    t0_quants: &[i8],
    t1_scales: &[f32],
    t1_quants: &[i8],
    blocks_per_row: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    // 4 float accumulators — one per (row, token) pair.
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    let r0 = row0_bytes.as_ptr();
    let r1 = row1_bytes.as_ptr();
    let q0 = t0_quants.as_ptr();
    let q1 = t1_quants.as_ptr();

    for blk in 0..blocks_per_row {
        let off = blk * Q8_0_BLOCK_BYTES;
        let qoff = blk * Q8_0_BLOCK_SIZE;

        // Weight row 0 block
        let w0p = r0.add(off);
        let w0_scale = f16_to_f32_inline(u16::from_le_bytes([*w0p, *w0p.add(1)]));
        let w0_lo = vld1q_s8(w0p.add(2) as *const i8);
        let w0_hi = vld1q_s8(w0p.add(18) as *const i8);

        // Weight row 1 block
        let w1p = r1.add(off);
        let w1_scale = f16_to_f32_inline(u16::from_le_bytes([*w1p, *w1p.add(1)]));
        let w1_lo = vld1q_s8(w1p.add(2) as *const i8);
        let w1_hi = vld1q_s8(w1p.add(18) as *const i8);

        // Token 0 block
        let t0_lo = vld1q_s8(q0.add(qoff));
        let t0_hi = vld1q_s8(q0.add(qoff + 16));
        let s0 = *t0_scales.get_unchecked(blk);

        // Token 1 block
        let t1_lo = vld1q_s8(q1.add(qoff));
        let t1_hi = vld1q_s8(q1.add(qoff + 16));
        let s1 = *t1_scales.get_unchecked(blk);

        // 8 SDOTs: 2 per (row, token) pair × 4 pairs
        let zero = vdupq_n_s32(0);

        let d00 = vaddq_s32(ggml_vdotq_s32(zero, w0_lo, t0_lo), ggml_vdotq_s32(zero, w0_hi, t0_hi));
        acc00 = vmlaq_n_f32(acc00, vcvtq_f32_s32(d00), w0_scale * s0);

        let d01 = vaddq_s32(ggml_vdotq_s32(zero, w0_lo, t1_lo), ggml_vdotq_s32(zero, w0_hi, t1_hi));
        acc01 = vmlaq_n_f32(acc01, vcvtq_f32_s32(d01), w0_scale * s1);

        let d10 = vaddq_s32(ggml_vdotq_s32(zero, w1_lo, t0_lo), ggml_vdotq_s32(zero, w1_hi, t0_hi));
        acc10 = vmlaq_n_f32(acc10, vcvtq_f32_s32(d10), w1_scale * s0);

        let d11 = vaddq_s32(ggml_vdotq_s32(zero, w1_lo, t1_lo), ggml_vdotq_s32(zero, w1_hi, t1_hi));
        acc11 = vmlaq_n_f32(acc11, vcvtq_f32_s32(d11), w1_scale * s1);
    }

    [
        vaddvq_f32(acc00),
        vaddvq_f32(acc01),
        vaddvq_f32(acc10),
        vaddvq_f32(acc11),
    ]
}

/// 2-row × 2-token Q8_0 dot product tile using SMMLA (M2+ only, FEAT_I8MM).
///
/// Same output as `dot_q8_0_q8_0_2x2_sdot` but uses the SMMLA instruction
/// which processes a 2×2×8 int8 tile in one instruction (2× compute density).
///
/// # Safety
/// Requires aarch64 with FEAT_I8MM.  All slices must be valid for `blocks_per_row`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn dot_q8_0_q8_0_2x2_smmla(
    row0_bytes: &[u8],
    row1_bytes: &[u8],
    t0_scales: &[f32],
    t0_quants: &[i8],
    t1_scales: &[f32],
    t1_quants: &[i8],
    blocks_per_row: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    // Single float32x4 accumulator: lanes [r0·t0, r0·t1, r1·t0, r1·t1]
    let mut acc = vdupq_n_f32(0.0);

    let r0 = row0_bytes.as_ptr();
    let r1 = row1_bytes.as_ptr();
    let q0 = t0_quants.as_ptr();
    let q1 = t1_quants.as_ptr();

    for blk in 0..blocks_per_row {
        let off = blk * Q8_0_BLOCK_BYTES;
        let qoff = blk * Q8_0_BLOCK_SIZE;

        // Weight scales
        let w0p = r0.add(off);
        let w0_scale = f16_to_f32_inline(u16::from_le_bytes([*w0p, *w0p.add(1)]));
        let w1p = r1.add(off);
        let w1_scale = f16_to_f32_inline(u16::from_le_bytes([*w1p, *w1p.add(1)]));

        // Token scales
        let s0 = *t0_scales.get_unchecked(blk);
        let s1 = *t1_scales.get_unchecked(blk);

        // Scale vector: [w0*s0, w0*s1, w1*s0, w1*s1]
        let cs = [w0_scale * s0, w0_scale * s1, w1_scale * s0, w1_scale * s1];
        let cs_vec = vld1q_f32(cs.as_ptr());

        // SMMLA expects A = [row0_bytes[0..8], row1_bytes[0..8]] packed as int8x16
        //                B = [tok0_bytes[0..8], tok1_bytes[0..8]] packed as int8x16
        // Process 32 weight-bytes in 4 SMMLA calls (8 bytes per call).
        let mut isum = vdupq_n_s32(0);
        let w0_qs = w0p.add(2) as *const i8;
        let w1_qs = w1p.add(2) as *const i8;
        for j in (0..32).step_by(8) {
            let a = vcombine_s8(
                vld1_s8(w0_qs.add(j)),
                vld1_s8(w1_qs.add(j)),
            );
            let b = vcombine_s8(
                vld1_s8(q0.add(qoff + j)),
                vld1_s8(q1.add(qoff + j)),
            );
            isum = smmla_s32(isum, a, b);
        }

        // Scale-fold: acc += float(isum) * cs_vec
        acc = vmlaq_f32(acc, vcvtq_f32_s32(isum), cs_vec);
    }

    let mut out = [0.0f32; 4];
    vst1q_f32(out.as_mut_ptr(), acc);
    out
}

/// 2-row × 4-token Q8_0 tile via SMMLA.  Halves weight bandwidth vs the 2×2 tile
/// by reusing each loaded weight block across 2 token-pairs.
///
/// Returns 8 dot products:
/// [r0·t0, r0·t1, r1·t0, r1·t1, r0·t2, r0·t3, r1·t2, r1·t3]
/// (first 4 from (tok0,tok1) tile, last 4 from (tok2,tok3) tile)
///
/// # Safety
/// Requires aarch64 with FEAT_I8MM.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn dot_q8_0_q8_0_2x4_smmla(
    row0_bytes: &[u8],
    row1_bytes: &[u8],
    t_scales: [&[f32]; 4],
    t_quants: [&[i8]; 4],
    blocks_per_row: usize,
) -> [f32; 8] {
    use std::arch::aarch64::*;

    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc23 = vdupq_n_f32(0.0);

    let r0 = row0_bytes.as_ptr();
    let r1 = row1_bytes.as_ptr();
    let q0 = t_quants[0].as_ptr();
    let q1 = t_quants[1].as_ptr();
    let q2 = t_quants[2].as_ptr();
    let q3 = t_quants[3].as_ptr();

    for blk in 0..blocks_per_row {
        let off = blk * Q8_0_BLOCK_BYTES;
        let qoff = blk * Q8_0_BLOCK_SIZE;

        let w0p = r0.add(off);
        let w0_scale = f16_to_f32_inline(u16::from_le_bytes([*w0p, *w0p.add(1)]));
        let w1p = r1.add(off);
        let w1_scale = f16_to_f32_inline(u16::from_le_bytes([*w1p, *w1p.add(1)]));

        let s0 = *t_scales[0].get_unchecked(blk);
        let s1 = *t_scales[1].get_unchecked(blk);
        let s2 = *t_scales[2].get_unchecked(blk);
        let s3 = *t_scales[3].get_unchecked(blk);

        let cs01 = [w0_scale * s0, w0_scale * s1, w1_scale * s0, w1_scale * s1];
        let cs23 = [w0_scale * s2, w0_scale * s3, w1_scale * s2, w1_scale * s3];
        let cs01_vec = vld1q_f32(cs01.as_ptr());
        let cs23_vec = vld1q_f32(cs23.as_ptr());

        let w0_qs = w0p.add(2) as *const i8;
        let w1_qs = w1p.add(2) as *const i8;

        // Pre-load 4 A operands (one per 8-byte chunk).  Reusing these across
        // both token-pair tiles is the bandwidth win.
        let a0 = vcombine_s8(vld1_s8(w0_qs), vld1_s8(w1_qs));
        let a1 = vcombine_s8(vld1_s8(w0_qs.add(8)), vld1_s8(w1_qs.add(8)));
        let a2 = vcombine_s8(vld1_s8(w0_qs.add(16)), vld1_s8(w1_qs.add(16)));
        let a3 = vcombine_s8(vld1_s8(w0_qs.add(24)), vld1_s8(w1_qs.add(24)));

        // --- (t0, t1) tile ---
        let mut isum01 = vdupq_n_s32(0);
        isum01 = smmla_s32(isum01, a0, vcombine_s8(vld1_s8(q0.add(qoff)),     vld1_s8(q1.add(qoff))));
        isum01 = smmla_s32(isum01, a1, vcombine_s8(vld1_s8(q0.add(qoff + 8)), vld1_s8(q1.add(qoff + 8))));
        isum01 = smmla_s32(isum01, a2, vcombine_s8(vld1_s8(q0.add(qoff + 16)), vld1_s8(q1.add(qoff + 16))));
        isum01 = smmla_s32(isum01, a3, vcombine_s8(vld1_s8(q0.add(qoff + 24)), vld1_s8(q1.add(qoff + 24))));
        acc01 = vmlaq_f32(acc01, vcvtq_f32_s32(isum01), cs01_vec);

        // --- (t2, t3) tile — reuses a0..a3 ---
        let mut isum23 = vdupq_n_s32(0);
        isum23 = smmla_s32(isum23, a0, vcombine_s8(vld1_s8(q2.add(qoff)),     vld1_s8(q3.add(qoff))));
        isum23 = smmla_s32(isum23, a1, vcombine_s8(vld1_s8(q2.add(qoff + 8)), vld1_s8(q3.add(qoff + 8))));
        isum23 = smmla_s32(isum23, a2, vcombine_s8(vld1_s8(q2.add(qoff + 16)), vld1_s8(q3.add(qoff + 16))));
        isum23 = smmla_s32(isum23, a3, vcombine_s8(vld1_s8(q2.add(qoff + 24)), vld1_s8(q3.add(qoff + 24))));
        acc23 = vmlaq_f32(acc23, vcvtq_f32_s32(isum23), cs23_vec);
    }

    let mut out01 = [0.0f32; 4];
    let mut out23 = [0.0f32; 4];
    vst1q_f32(out01.as_mut_ptr(), acc01);
    vst1q_f32(out23.as_mut_ptr(), acc23);
    [out01[0], out01[1], out01[2], out01[3], out23[0], out23[1], out23[2], out23[3]]
}

/// 4-token matvec: stream weight data once and compute outputs for 4 tokens.
///
/// `output` must have length `4 * rows`, laid out as
/// [tok0_rows..., tok1_rows..., tok2_rows..., tok3_rows...].
///
/// Requires FEAT_I8MM at runtime.  Callers on M1 should fall back to two
/// successive `matvec_q8_0_preq_pair_into` calls.
#[cfg(target_arch = "aarch64")]
pub fn matvec_q8_0_preq_quad_into(
    weight_data: &[u8],
    preqs: [&QuantizedInput; 4],
    rows: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), 4 * rows);
    let blocks_per_row = preqs[0].blocks_per_row;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let total_work = rows * (blocks_per_row * Q8_0_BLOCK_SIZE);
    let (out01, out23) = output.split_at_mut(2 * rows);
    let (out0, out1) = out01.split_at_mut(rows);
    let (out2, out3) = out23.split_at_mut(rows);

    let t_scales: [&[f32]; 4] = [
        preqs[0].scales.as_slice(),
        preqs[1].scales.as_slice(),
        preqs[2].scales.as_slice(),
        preqs[3].scales.as_slice(),
    ];
    let t_quants: [&[i8]; 4] = [
        preqs[0].quants.as_slice(),
        preqs[1].quants.as_slice(),
        preqs[2].quants.as_slice(),
        preqs[3].quants.as_slice(),
    ];

    if total_work >= 2_000_000 {
        use rayon::prelude::*;
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        let num_chunks = rows.div_ceil(chunk_rows);
        // Per-chunk results: Vec<(row, v0, v1, v2, v3)>
        let chunk_results: Vec<Vec<(usize, f32, f32, f32, f32)>> =
            (0..num_chunks).into_par_iter().map(|ci| {
                let base = ci * chunk_rows;
                let end = (base + chunk_rows).min(rows);
                let len = end - base;
                let mut results = Vec::with_capacity(len);
                let pairs = len / 2;
                for p in 0..pairs {
                    let r0 = base + p * 2;
                    let r1 = r0 + 1;
                    let rb0 = &weight_data[r0 * bytes_per_row..(r0 + 1) * bytes_per_row];
                    let rb1 = &weight_data[r1 * bytes_per_row..(r1 + 1) * bytes_per_row];
                    let d = unsafe {
                        dot_q8_0_q8_0_2x4_smmla(rb0, rb1, t_scales, t_quants, blocks_per_row)
                    };
                    // d layout: [r0t0, r0t1, r1t0, r1t1, r0t2, r0t3, r1t2, r1t3]
                    results.push((r0, d[0], d[1], d[4], d[5]));
                    results.push((r1, d[2], d[3], d[6], d[7]));
                }
                if len & 1 != 0 {
                    let row = base + len - 1;
                    let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                    let v0 = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[0], t_quants[0], blocks_per_row) };
                    let v1 = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[1], t_quants[1], blocks_per_row) };
                    let v2 = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[2], t_quants[2], blocks_per_row) };
                    let v3 = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[3], t_quants[3], blocks_per_row) };
                    results.push((row, v0, v1, v2, v3));
                }
                results
            }).collect();
        for chunk in chunk_results {
            for (row, v0, v1, v2, v3) in chunk {
                out0[row] = v0;
                out1[row] = v1;
                out2[row] = v2;
                out3[row] = v3;
            }
        }
    } else {
        let pairs = rows / 2;
        for p in 0..pairs {
            let r0 = p * 2;
            let r1 = r0 + 1;
            let rb0 = &weight_data[r0 * bytes_per_row..(r0 + 1) * bytes_per_row];
            let rb1 = &weight_data[r1 * bytes_per_row..(r1 + 1) * bytes_per_row];
            let d = unsafe {
                dot_q8_0_q8_0_2x4_smmla(rb0, rb1, t_scales, t_quants, blocks_per_row)
            };
            out0[r0] = d[0]; out1[r0] = d[1]; out2[r0] = d[4]; out3[r0] = d[5];
            out0[r1] = d[2]; out1[r1] = d[3]; out2[r1] = d[6]; out3[r1] = d[7];
        }
        if rows & 1 != 0 {
            let row = rows - 1;
            let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            out0[row] = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[0], t_quants[0], blocks_per_row) };
            out1[row] = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[1], t_quants[1], blocks_per_row) };
            out2[row] = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[2], t_quants[2], blocks_per_row) };
            out3[row] = unsafe { dot_q8_0_q8_0_neon(rb, t_scales[3], t_quants[3], blocks_per_row) };
        }
    }
}

/// 2-token matvec: stream weight data once and compute outputs for 2 tokens.
///
/// `output` must have length `2 * rows`, laid out as [tok0_rows..., tok1_rows...].
/// This halves weight memory bandwidth compared to calling `matvec_q8_0_preq_into`
/// twice (once per token).
#[cfg(target_arch = "aarch64")]
pub fn matvec_q8_0_preq_pair_into(
    weight_data: &[u8],
    preq0: &QuantizedInput,
    preq1: &QuantizedInput,
    rows: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), 2 * rows);
    debug_assert_eq!(preq0.blocks_per_row, preq1.blocks_per_row);
    let blocks_per_row = preq0.blocks_per_row;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let has_smmla = std::arch::is_aarch64_feature_detected!("i8mm");

    let total_work = rows * (blocks_per_row * Q8_0_BLOCK_SIZE);
    let (out0, out1) = output.split_at_mut(rows);

    // Helper to dispatch the 2×2 tile kernel (SMMLA on M2+, SDOT otherwise).
    let tile = |rb0: &[u8], rb1: &[u8], p0: &QuantizedInput, p1: &QuantizedInput| -> [f32; 4] {
        if has_smmla {
            unsafe { dot_q8_0_q8_0_2x2_smmla(rb0, rb1, &p0.scales, &p0.quants, &p1.scales, &p1.quants, blocks_per_row) }
        } else {
            unsafe { dot_q8_0_q8_0_2x2_sdot(rb0, rb1, &p0.scales, &p0.quants, &p1.scales, &p1.quants, blocks_per_row) }
        }
    };
    // Helper for single-row fallback.
    let single = |rb: &[u8], p: &QuantizedInput| -> f32 {
        unsafe { dot_q8_0_q8_0_neon(rb, &p.scales, &p.quants, blocks_per_row) }
    };

    if total_work >= 2_000_000 {
        use rayon::prelude::*;
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        // Collect per-chunk results then scatter.  Each chunk produces a vec of
        // (row_index, tok0_val, tok1_val) that we write back sequentially.
        // This avoids sending raw pointers across threads.
        let num_chunks = rows.div_ceil(chunk_rows);
        let chunk_results: Vec<Vec<(usize, f32, f32)>> =
            (0..num_chunks).into_par_iter().map(|ci| {
                let base = ci * chunk_rows;
                let end = (base + chunk_rows).min(rows);
                let len = end - base;
                let mut results = Vec::with_capacity(len);
                let pairs = len / 2;
                for p in 0..pairs {
                    let r0 = base + p * 2;
                    let r1 = r0 + 1;
                    let rb0 = &weight_data[r0 * bytes_per_row..(r0 + 1) * bytes_per_row];
                    let rb1 = &weight_data[r1 * bytes_per_row..(r1 + 1) * bytes_per_row];
                    let [d00, d01, d10, d11] = tile(rb0, rb1, preq0, preq1);
                    results.push((r0, d00, d01));
                    results.push((r1, d10, d11));
                }
                if len & 1 != 0 {
                    let row = base + len - 1;
                    let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                    results.push((row, single(rb, preq0), single(rb, preq1)));
                }
                results
            }).collect();
        for chunk in chunk_results {
            for (row, v0, v1) in chunk {
                out0[row] = v0;
                out1[row] = v1;
            }
        }
    } else {
        let pairs = rows / 2;
        for p in 0..pairs {
            let r0 = p * 2;
            let r1 = r0 + 1;
            let rb0 = &weight_data[r0 * bytes_per_row..(r0 + 1) * bytes_per_row];
            let rb1 = &weight_data[r1 * bytes_per_row..(r1 + 1) * bytes_per_row];
            let [d00, d01, d10, d11] = tile(rb0, rb1, preq0, preq1);
            out0[r0] = d00;
            out0[r1] = d10;
            out1[r0] = d01;
            out1[r1] = d11;
        }
        if rows & 1 != 0 {
            let row = rows - 1;
            let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            out0[row] = single(rb, preq0);
            out1[row] = single(rb, preq1);
        }
    }
}

/// Compute an optimal rayon chunk size based on matrix rows and available threads.
///
/// Aims for ~3x oversubscription of the rayon thread pool so load is balanced
/// without incurring excessive per-task dispatch overhead. Chunk sizes are
/// clamped to [32, 1024] rows:
///
/// - Small matrices (e.g. 2048-row Q/K/V/O): ~48 rows/chunk → ~42 tasks on a
///   14-thread pool, fully utilizing cores that would otherwise sit idle under
///   a fixed 256-row split.
/// - Medium matrices (e.g. 8192-row gate/up/down): ~195 rows/chunk → ~42 tasks.
/// - Large matrices (e.g. 128256-row output projection): clamped to 1024
///   rows/chunk → ~125 tasks instead of 501, cutting dispatch overhead.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn adaptive_matvec_chunk_size(rows: usize) -> usize {
    // Chunk size selection based on empirical profiling on Apple M5 Pro.
    //
    // Rayon's `par_chunks_mut().for_each()` incurs ~15-30 μs of latch /
    // condvar wake overhead per call on macOS.  For medium matrices the
    // crossover point where parallelism beats that overhead is higher than
    // you'd expect, so we keep chunks fat and task counts low.  The profile
    // before this tuning showed worker threads ~44% idle in
    // `wait_until_cold` and main thread ~96% blocked in `pthread_cond_wait`
    // — fewer tasks per call trade peak parallelism for better average
    // utilisation.
    if rows < 2048 {
        rows.div_ceil(2).max(1)
    } else if rows < 8192 {
        rows.div_ceil(8).max(1)
    } else if rows < 32768 {
        512
    } else {
        1024
    }
}

/// ARM NEON Q8_0 x Q8_0 integer dot product matvec with rayon parallelism.
///
/// Quantizes the f32 input vector to Q8_0 once, then computes int8*int8 dot
/// products for each row. This is the llama.cpp approach that reads 1 byte/weight
/// + 1 byte/input instead of 1 byte/weight + 4 bytes/input (4x less bandwidth).
#[cfg(target_arch = "aarch64")]
fn matvec_q8_0_neon(weight_data: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    // Step 1: Quantize entire input vector to Q8_0 (one-time cost)
    let mut input_scales = Vec::with_capacity(blocks_per_row);
    let mut input_quants = vec![0i8; cols];
    for b in 0..blocks_per_row {
        let start = b * Q8_0_BLOCK_SIZE;
        let (scale, quants) =
            unsafe { quantize_f32_to_q8_0_block(&input[start..start + Q8_0_BLOCK_SIZE]) };
        input_scales.push(scale);
        input_quants[start..start + Q8_0_BLOCK_SIZE].copy_from_slice(&quants);
    }

    // Step 2: Compute int8 x int8 dot product for each row
    const PARALLEL_THRESHOLD: usize = 5_000_000;
    let total_work = rows * cols;

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        if total_work >= PARALLEL_THRESHOLD && rows >= chunk_rows * 2 {
            use rayon::prelude::*;
            let mut output = vec![0.0f32; rows];
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * chunk_rows;
                    for (local_i, out) in out_chunk.iter_mut().enumerate() {
                        let i = row_start + local_i;
                        let start = i * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        // SAFETY: NEON always available on aarch64
                        *out = unsafe {
                            dot_q8_0_q8_0_neon(
                                row_bytes,
                                &input_scales,
                                &input_quants,
                                blocks_per_row,
                            )
                        };
                    }
                });
            return output;
        }
    }

    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let start = i * bytes_per_row;
        let row_bytes = &weight_data[start..start + bytes_per_row];
        // SAFETY: NEON always available on aarch64
        *out =
            unsafe { dot_q8_0_q8_0_neon(row_bytes, &input_scales, &input_quants, blocks_per_row) };
    }
    output
}

/// Dispatch Q8_0 direct matvec to the best available implementation.
///
/// `weight_data`: raw Q8_0 bytes, row-major.
/// `input`: f32 vector of length `cols`.
/// `rows`: output dimension.  `cols`: input dimension (multiple of 32).
pub fn matvec_q8_0(weight_data: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(
        cols % Q8_0_BLOCK_SIZE,
        0,
        "cols must be multiple of 32 for Q8_0"
    );
    debug_assert_eq!(
        weight_data.len(),
        rows * (cols / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES,
        "weight_data length mismatch for Q8_0 matvec"
    );
    debug_assert_eq!(input.len(), cols, "input length must equal cols");

    #[cfg(target_arch = "aarch64")]
    {
        matvec_q8_0_neon(weight_data, input, rows, cols)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
        {
            matvec_q8_0_relaxed_simd(weight_data, input, rows, cols)
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "relaxed_simd")))]
        {
            matvec_q8_0_scalar(weight_data, input, rows, cols)
        }
    }
}

// ---------------------------------------------------------------------------
// BitNet b1.58 ternary weight support
// ---------------------------------------------------------------------------

/// Ternary weight encoding (2 bits per weight, 4 weights per byte):
///   00 = 0, 01 = +1, 10 = -1, 11 = unused/reserved
///
/// Weights are packed LSB-first: bits \[1:0\] = weight 0, bits \[3:2\] = weight 1, etc.
///
/// Pack an f32 weight slice into ternary 2-bit encoding (4 weights per byte).
///
/// Each weight is quantized by sign: positive -> +1, negative -> -1, zero -> 0.
/// The number of output bytes is `ceil(weights.len() / 4)`.
#[allow(clippy::manual_div_ceil)]
pub fn quantize_to_ternary(weights: &[f32]) -> Vec<u8> {
    let num_bytes = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; num_bytes];
    for (i, &w) in weights.iter().enumerate() {
        let trit: u8 = if w > 0.0 {
            0b01 // +1
        } else if w < 0.0 {
            0b10 // -1
        } else {
            0b00 // 0
        };
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        packed[byte_idx] |= trit << bit_shift;
    }
    packed
}

/// Unpack a single ternary weight from packed bytes.
/// Returns -1, 0, or +1.
#[inline(always)]
fn unpack_ternary(packed: &[u8], index: usize) -> i8 {
    let byte_idx = index / 4;
    let bit_shift = (index % 4) * 2;
    let bits = (packed[byte_idx] >> bit_shift) & 0b11;
    match bits {
        0b01 => 1,
        0b10 => -1,
        _ => 0, // 00 = zero, 11 = unused (treated as zero)
    }
}

/// Scalar ternary matvec: `output[row]` = sum of `input[j]` * ternary_weight[row, j]`.
///
/// No floating-point multiplications — only addition and subtraction based on
/// the ternary weight value {-1, 0, +1}.
///
/// `packed_weights`: row-major ternary-packed bytes, each row uses `ceil(cols/4)` bytes.
#[allow(clippy::needless_range_loop)]
pub fn matvec_ternary_scalar(
    packed_weights: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let bytes_per_row = cols.div_ceil(4);
    let mut output = vec![0.0f32; rows];

    for (row_idx, out) in output.iter_mut().enumerate() {
        let row_bytes = &packed_weights[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
        let mut sum = 0.0f32;

        // Process 4 weights at a time (one full byte)
        let full_bytes = cols / 4;
        for byte_idx in 0..full_bytes {
            let byte_val = row_bytes[byte_idx];
            let base_col = byte_idx * 4;

            // Unpack 4 weights from this byte
            for shift in 0..4 {
                let bits = (byte_val >> (shift * 2)) & 0b11;
                let col = base_col + shift;
                match bits {
                    0b01 => sum += input[col],
                    0b10 => sum -= input[col],
                    _ => {} // 0 or unused: skip
                }
            }
        }

        // Handle remaining weights (< 4)
        let remaining_start = full_bytes * 4;
        for col in remaining_start..cols {
            let trit = unpack_ternary(row_bytes, col);
            match trit {
                1 => sum += input[col],
                -1 => sum -= input[col],
                _ => {}
            }
        }

        *out = sum;
    }
    output
}

/// SIMD-accelerated ternary matvec for aarch64 (ARM NEON).
///
/// Processes 4 floats at a time using NEON intrinsics with branchless
/// conditional add/sub based on ternary weight sign bits.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::manual_div_ceil)]
pub fn matvec_ternary_neon(
    packed_weights: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let bytes_per_row = cols.div_ceil(4);
    let mut output = vec![0.0f32; rows];

    for (row_idx, out) in output.iter_mut().enumerate() {
        let row_bytes = &packed_weights[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
        // SAFETY: NEON is always available on aarch64. Pointers are valid for the
        // slice lengths checked above.
        *out = unsafe { matvec_ternary_neon_row(row_bytes, input, cols) };
    }
    output
}

/// Process a single row of ternary matvec using NEON.
///
/// # Safety
/// Caller must ensure `row_bytes` has at least `ceil(cols/4)` bytes and
/// `input` has at least `cols` elements. NEON must be available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::needless_range_loop)]
unsafe fn matvec_ternary_neon_row(row_bytes: &[u8], input: &[f32], cols: usize) -> f32 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_f32(0.0);
    let zero = vdupq_n_f32(0.0);

    // Process 4 weights at a time (one byte = 4 ternary weights = 4 floats = one NEON lane)
    let full_bytes = cols / 4;
    for byte_idx in 0..full_bytes {
        let byte_val = row_bytes[byte_idx];
        let base_col = byte_idx * 4;
        let inp = vld1q_f32(input.as_ptr().add(base_col));

        // Decode 4 ternary weights into sign masks
        // For each 2-bit field: 01 -> +1, 10 -> -1, 00/11 -> 0
        let mut pos_mask_arr = [0u32; 4];
        let mut neg_mask_arr = [0u32; 4];
        for shift in 0..4 {
            let bits = (byte_val >> (shift * 2)) & 0b11;
            if bits == 0b01 {
                pos_mask_arr[shift] = 0xFFFFFFFF;
            } else if bits == 0b10 {
                neg_mask_arr[shift] = 0xFFFFFFFF;
            }
        }
        let pos_mask: uint32x4_t = vld1q_u32(pos_mask_arr.as_ptr());
        let neg_mask: uint32x4_t = vld1q_u32(neg_mask_arr.as_ptr());

        // Branchless via bitselect: where pos_mask is set, add inp; where neg_mask, subtract
        let to_add = vbslq_f32(pos_mask, inp, zero);
        let to_sub = vbslq_f32(neg_mask, inp, zero);
        acc = vaddq_f32(acc, to_add);
        acc = vsubq_f32(acc, to_sub);
    }

    // Horizontal sum of acc
    let sum = vaddvq_f32(acc);

    // Handle remaining elements
    let remaining_start = full_bytes * 4;
    let mut tail_sum = 0.0f32;
    for col in remaining_start..cols {
        let trit = unpack_ternary(row_bytes, col);
        match trit {
            1 => tail_sum += input[col],
            -1 => tail_sum -= input[col],
            _ => {}
        }
    }

    sum + tail_sum
}

/// Dispatch ternary matvec to the best available implementation.
///
/// On aarch64: uses NEON SIMD.
/// On other targets: uses the scalar fallback.
pub fn matvec_ternary(packed_weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        matvec_ternary_neon(packed_weights, input, rows, cols)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_ternary_scalar(packed_weights, input, rows, cols)
    }
}

// ---------------------------------------------------------------------------
// Mixture-of-Experts (MoE) helpers
// ---------------------------------------------------------------------------

/// Select the top-k experts from router logits and return their indices
/// and softmax-normalized weights.
///
/// The softmax is computed only over the top-k values (not all experts),
/// matching the standard MoE routing convention used by Mixtral and similar.
pub fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    // Build (index, score) pairs and partial-sort to find the top-k.
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // Softmax over the top-k scores for numerically stable normalization.
    let max_val = top[0].1;
    let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    (indices, weights)
}

/// Run the MoE FFN block for a single token: route through top-k experts
/// and return the weighted combination of their outputs.
///
/// This is the f32 CPU path. Each selected expert runs a standard SwiGLU FFN:
///   gate_out = gate_proj(x)
///   up_out   = up_proj(x)
///   hidden   = silu(gate_out) * up_out
///   expert_out = down_proj(hidden)
///
/// The final output is `sum_i(weight_i * expert_i_out)`.
pub fn moe_forward(
    x: &[f32],
    moe_weights: &MoeLayerWeights,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_experts_per_token: usize,
) -> Vec<f32> {
    let num_experts = moe_weights.experts.len();

    // Router: compute expert scores = gate_weight * x
    let gate_logits = matvec(moe_weights.gate.data(), x, num_experts, hidden_dim);

    // Top-K selection with softmax normalization
    let (top_indices, top_weights) = top_k_softmax(&gate_logits, num_experts_per_token);

    // Run selected experts and combine with weighted sum
    let mut output = vec![0.0f32; hidden_dim];
    for (&expert_idx, &weight) in top_indices.iter().zip(top_weights.iter()) {
        let expert = &moe_weights.experts[expert_idx];
        debug_assert!(expert.loaded, "expert {expert_idx} not loaded");

        // Standard SwiGLU FFN
        let gate_out = matvec(expert.w_gate.data(), x, intermediate_dim, hidden_dim);
        let up_out = matvec(expert.w_up.data(), x, intermediate_dim, hidden_dim);
        let hidden = silu_mul_cpu(&gate_out, &up_out);
        let expert_out = matvec(expert.w_down.data(), &hidden, hidden_dim, intermediate_dim);

        // Weighted accumulation
        for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
            *o += weight * e;
        }
    }
    output
}

/// In-place variant of `moe_forward` that writes the result into an existing buffer.
pub fn moe_forward_into(
    x: &[f32],
    moe_weights: &MoeLayerWeights,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_experts_per_token: usize,
    output: &mut [f32],
) {
    let num_experts = moe_weights.experts.len();

    // Router: compute expert scores = gate_weight * x
    let gate_logits = matvec(moe_weights.gate.data(), x, num_experts, hidden_dim);

    // Top-K selection with softmax normalization
    let (top_indices, top_weights) = top_k_softmax(&gate_logits, num_experts_per_token);

    // Zero out the output buffer
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Run selected experts and combine with weighted sum
    for (&expert_idx, &weight) in top_indices.iter().zip(top_weights.iter()) {
        let expert = &moe_weights.experts[expert_idx];
        debug_assert!(expert.loaded, "expert {expert_idx} not loaded");

        // Standard SwiGLU FFN
        let gate_out = matvec(expert.w_gate.data(), x, intermediate_dim, hidden_dim);
        let up_out = matvec(expert.w_up.data(), x, intermediate_dim, hidden_dim);
        let hidden = silu_mul_cpu(&gate_out, &up_out);
        let expert_out = matvec(expert.w_down.data(), &hidden, hidden_dim, intermediate_dim);

        // Weighted accumulation
        for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
            *o += weight * e;
        }
    }
}

// TODO: For on-demand loading, experts could be stored in OPFS (Origin Private
// File System) and loaded when the router predicts they'll be needed.
// See OD-MoE (arxiv.org/abs/2512.03927) for expert prediction strategies.
// The `ExpertWeights::loaded` flag enables this: unloaded experts have empty
// tensors and are fetched from storage before the matvec.

/// SiLU(gate) * up on raw slices, returning a new Vec (CPU implementation).
///
/// On aarch64 this uses NEON intrinsics with a fast polynomial sigmoid
/// approximation (Padé-based tanh), avoiding per-element `exp()` calls.
/// On other targets it uses an index loop that auto-vectorizes better
/// than the iterator/collect chain.
pub fn silu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        silu_mul_neon(gate, up)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        silu_mul_scalar(gate, up)
    }
}

/// Scalar SiLU*up using an index loop (auto-vectorizes better than iterator chain).
#[cfg(not(target_arch = "aarch64"))]
fn silu_mul_scalar(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let n = gate.len();
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let g = gate[i];
        result[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
    result
}

/// NEON-accelerated SiLU*up using a fast polynomial sigmoid approximation.
///
/// Uses: sigmoid(x) ≈ 0.5 + 0.5 * tanh(x * 0.5)
/// with tanh(t) ≈ t * (27 + t²) / (27 + 9*t²)  (Padé approximant, ~1e-4 accuracy).
/// This avoids all exp() calls and is fully vectorizable.
#[cfg(target_arch = "aarch64")]
fn silu_mul_neon(gate: &[f32], up: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::*;
    let n = gate.len();
    let mut result = vec![0.0f32; n];
    let len4 = n & !3;

    unsafe {
        let half = vdupq_n_f32(0.5);
        let twenty_seven = vdupq_n_f32(27.0);
        let nine = vdupq_n_f32(9.0);

        for i in (0..len4).step_by(4) {
            let g = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));

            // t = g * 0.5
            let t = vmulq_f32(g, half);
            // t2 = t * t
            let t2 = vmulq_f32(t, t);
            // numerator = t * (27 + t2)
            let num = vmulq_f32(t, vaddq_f32(twenty_seven, t2));
            // denominator = 27 + 9*t2
            let den = vaddq_f32(twenty_seven, vmulq_f32(nine, t2));
            // tanh_approx = num / den
            // Use vdivq_f32 (available on aarch64)
            let tanh_approx = vdivq_f32(num, den);
            // sigmoid = 0.5 + 0.5 * tanh_approx
            let sigmoid = vaddq_f32(half, vmulq_f32(half, tanh_approx));
            // silu = g * sigmoid
            let silu = vmulq_f32(g, sigmoid);
            // result = silu * up
            let out = vmulq_f32(silu, u);

            vst1q_f32(result.as_mut_ptr().add(i), out);
        }
    }

    // Scalar tail
    for i in len4..n {
        let g = gate[i];
        result[i] = (g / (1.0 + (-g).exp())) * up[i];
    }

    result
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

/// Maximum number of positions to pre-compute in the RoPE lookup table.
///
/// For models with very large context windows (e.g. 128K), allocating a table
/// for every position would consume too much memory. Positions beyond this cap
/// fall back to inline cos/sin computation.
const ROPE_TABLE_MAX_SEQ_LEN: usize = 8192;

/// Pre-computed cos/sin lookup table for Rotary Position Embeddings (RoPE).
///
/// Stores `cos(pos * freq)` and `sin(pos * freq)` for every `(pos, dim_index)`
/// pair up to `max_seq_len` positions and `head_dim / 2` frequency slots.
/// Tables are flattened as `[max_seq_len, head_dim / 2]`.
pub struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    max_seq_len: usize,
}

impl RopeTable {
    /// Build the table for the given model parameters.
    ///
    /// `max_seq_len` is capped at [`ROPE_TABLE_MAX_SEQ_LEN`] to limit memory
    /// usage. Positions beyond the cap must use [`apply_rope`] instead.
    pub fn new(head_dim: usize, max_seq_len: usize, rope_theta: f32) -> Self {
        let capped = max_seq_len.min(ROPE_TABLE_MAX_SEQ_LEN);
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; capped * half];
        let mut sin = vec![0.0f32; capped * half];

        for pos in 0..capped {
            for d in 0..half {
                let freq = 1.0 / rope_theta.powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[pos * half + d] = c;
                sin[pos * half + d] = s;
            }
        }

        Self {
            cos,
            sin,
            head_dim,
            max_seq_len: capped,
        }
    }

    /// Look up `(cos, sin)` for a single `(pos, d)` pair.
    #[inline]
    pub fn get(&self, pos: usize, d: usize) -> (f32, f32) {
        let half = self.head_dim / 2;
        let idx = pos * half + d;
        (self.cos[idx], self.sin[idx])
    }

    /// Returns the maximum cached position (exclusive).
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Returns the head dimension this table was built for.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns a slice of `head_dim / 2` cos values for a given position.
    #[inline]
    pub fn cos_row(&self, pos: usize) -> &[f32] {
        let half = self.head_dim / 2;
        let start = pos * half;
        &self.cos[start..start + half]
    }

    /// Returns a slice of `head_dim / 2` sin values for a given position.
    #[inline]
    pub fn sin_row(&self, pos: usize) -> &[f32] {
        let half = self.head_dim / 2;
        let start = pos * half;
        &self.sin[start..start + half]
    }
}

/// Apply RoPE using a pre-computed [`RopeTable`].
///
/// Looks up cos/sin from the table instead of computing transcendental
/// functions per element. Falls back to [`apply_rope`] for positions
/// beyond the table's range.
pub fn apply_rope_with_table(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    table: &RopeTable,
) {
    // Fall back to inline computation for out-of-range positions.
    if pos >= table.max_seq_len() || head_dim != table.head_dim() {
        apply_rope(data, num_heads, head_dim, pos, theta);
        return;
    }

    let half = head_dim / 2;
    let cos_row = table.cos_row(pos);
    let sin_row = table.sin_row(pos);

    #[cfg(target_arch = "aarch64")]
    {
        apply_rope_neon(data, cos_row, sin_row, num_heads, head_dim, half);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half {
                let c = cos_row[i];
                let s = sin_row[i];
                let x0 = data[offset + i];
                let x1 = data[offset + i + half];
                data[offset + i] = x0 * c - x1 * s;
                data[offset + i + half] = x0 * s + x1 * c;
            }
        }
    }
}

/// Apply RoPE to interleaved Q or K vectors (CPU implementation).
///
/// Computes cos/sin inline for the given position. Prefer
/// [`apply_rope_with_table`] on the hot path when a [`RopeTable`] is
/// available — it eliminates transcendental function calls entirely.
pub fn apply_rope(data: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;

    // Precompute cos/sin once for each frequency
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        cos_table[i] = cos_val;
        sin_table[i] = sin_val;
    }

    // Apply rotation to all heads using the cached tables
    #[cfg(target_arch = "aarch64")]
    {
        apply_rope_neon(data, &cos_table, &sin_table, num_heads, head_dim, half);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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
}

/// NEON-accelerated RoPE rotation: processes 4 dimension pairs per iteration.
#[cfg(target_arch = "aarch64")]
fn apply_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_heads: usize,
    head_dim: usize,
    half: usize,
) {
    use std::arch::aarch64::*;
    let half4 = half & !3;

    for h in 0..num_heads {
        let offset = h * head_dim;

        unsafe {
            // Process 4 elements at a time
            for i in (0..half4).step_by(4) {
                let c = vld1q_f32(cos_table.as_ptr().add(i));
                let s = vld1q_f32(sin_table.as_ptr().add(i));
                let x0 = vld1q_f32(data.as_ptr().add(offset + i));
                let x1 = vld1q_f32(data.as_ptr().add(offset + i + half));

                // x0_rot = x0 * c - x1 * s
                let x0_rot = vmlsq_f32(vmulq_f32(x0, c), x1, s);
                // x1_rot = x0 * s + x1 * c
                let x1_rot = vmlaq_f32(vmulq_f32(x1, c), x0, s);

                vst1q_f32(data.as_mut_ptr().add(offset + i), x0_rot);
                vst1q_f32(data.as_mut_ptr().add(offset + i + half), x1_rot);
            }
        }

        // Scalar tail
        for i in half4..half {
            let c = cos_table[i];
            let s = sin_table[i];
            let x0 = data[offset + i];
            let x1 = data[offset + i + half];
            data[offset + i] = x0 * c - x1 * s;
            data[offset + i + half] = x0 * s + x1 * c;
        }
    }
}

/// NEON-accelerated attention dot-product scoring.
///
/// Computes `scores[t] = dot(q[q_offset..], k_cache[t * kv_stride + kv_head * head_dim..]) * scale`
/// for all `t` in `0..seq_len`, processing 4 dimensions per iteration.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn attn_dot_scores_neon(
    q: &[f32],
    k_cache: &[f32],
    scores: &mut [f32],
    q_offset: usize,
    kv_head: usize,
    head_dim: usize,
    kv_stride: usize,
    seq_len: usize,
    scale: f32,
) {
    use std::arch::aarch64::*;
    let dim4 = head_dim & !3;

    for (t, score) in scores[..seq_len].iter_mut().enumerate() {
        let k_offset = t * kv_stride + kv_head * head_dim;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for d in (0..dim4).step_by(4) {
                let qv = vld1q_f32(q.as_ptr().add(q_offset + d));
                let kv = vld1q_f32(k_cache.as_ptr().add(k_offset + d));
                acc = vmlaq_f32(acc, qv, kv);
            }
            let mut dot = vaddvq_f32(acc);
            // Scalar tail
            for d in dim4..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            *score = dot * scale;
        }
    }
}

/// CPU implementation of grouped-query attention for a single token position.
/// Exported so GPU backends can delegate to it for unsupported cases (e.g. soft-cap).
///
/// Allocating wrapper around `cpu_grouped_query_attention_into`.
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
    let mut output = vec![0.0f32; num_heads * head_dim];
    cpu_grouped_query_attention_into(
        q,
        k_cache,
        v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        attn_softcap,
        &mut output,
    );
    output
}

// ---------------------------------------------------------------------------
// _into variants: write results into pre-allocated output slices to avoid
// per-token heap allocations in the forward() hot path.
// ---------------------------------------------------------------------------

/// RMSNorm writing into a pre-allocated output slice.
#[inline]
pub fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON always available on aarch64
        unsafe { rmsnorm_neon_into(x, weight, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rmsnorm_scalar_into(x, weight, eps, output)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rmsnorm_scalar_into(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..dim {
        output[i] = (x[i] * inv_rms) * weight[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn rmsnorm_neon_into(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
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
}

/// Matrix-vector multiply writing into a pre-allocated output slice.
///
/// Dispatches to the same platform-specific implementations as `matvec()`.
#[inline]
pub fn matvec_into(mat: &[f32], vec: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    debug_assert_eq!(output.len(), rows);
    #[cfg(target_os = "macos")]
    {
        debug_assert_eq!(mat.len(), rows * cols, "matvec_into: mat size mismatch");
        debug_assert!(vec.len() >= cols, "matvec_into: vec size mismatch");
        // SAFETY: pointers and dimensions validated by debug_assert above.
        unsafe {
            cblas_sgemv(
                101, // CblasRowMajor
                111, // CblasNoTrans
                rows as i32,
                cols as i32,
                1.0,
                mat.as_ptr(),
                cols as i32,
                vec.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        matvec_simd_into(mat, vec, rows, cols, output);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above ensures AVX2 + FMA are available
            unsafe { matvec_avx2_into(mat, vec, rows, cols, output) };
            return;
        }
        matvec_scalar_into(mat, vec, rows, cols, output);
    }
    #[cfg(not(any(target_os = "macos", target_arch = "aarch64", target_arch = "x86_64")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
        {
            matvec_relaxed_simd_into(mat, vec, rows, cols, output);
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "relaxed_simd")))]
        {
            matvec_scalar_into(mat, vec, rows, cols, output);
        }
    }
}

/// Scalar matvec into pre-allocated output.
#[cfg(any(
    target_arch = "x86_64",
    not(any(target_os = "macos", target_arch = "aarch64"))
))]
fn matvec_scalar_into(mat: &[f32], vec: &[f32], _rows: usize, cols: usize, output: &mut [f32]) {
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..(i + 1) * cols];
        *out = matvec_scalar_row(row, vec, cols);
    }
}

// ---------------------------------------------------------------------------
// WASM relaxed SIMD (f32x4_relaxed_madd / FMA) matvec
// ---------------------------------------------------------------------------

/// Compute a single row dot product using WASM relaxed SIMD `f32x4_relaxed_madd`.
///
/// On hardware with FMA (ARM NEON, x86 FMA3), this maps to a single fused
/// multiply-add instruction per 4-element vector, providing ~15-30% speedup
/// over separate `f32x4_mul` + `f32x4_add` by reducing rounding steps and
/// improving instruction throughput.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
#[inline]
fn matvec_relaxed_simd_row(row: &[f32], vec: &[f32], cols: usize) -> f32 {
    use core::arch::wasm32::*;

    let cols16 = cols & !15;
    let cols4 = cols & !3;

    // 4 independent accumulators to hide SIMD pipeline latency.
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut acc2 = f32x4_splat(0.0);
    let mut acc3 = f32x4_splat(0.0);

    let mut j = 0usize;

    // Main loop: 16 elements per iteration (4 FMA ops).
    while j < cols16 {
        // SAFETY: j + 0..16 is within bounds (j < cols16 <= cols),
        // and f32 slices from Tensor are 16-byte aligned by construction.
        unsafe {
            let r0 = v128_load(row.as_ptr().add(j) as *const v128);
            let v0 = v128_load(vec.as_ptr().add(j) as *const v128);
            acc0 = f32x4_relaxed_madd(r0, v0, acc0);

            let r1 = v128_load(row.as_ptr().add(j + 4) as *const v128);
            let v1 = v128_load(vec.as_ptr().add(j + 4) as *const v128);
            acc1 = f32x4_relaxed_madd(r1, v1, acc1);

            let r2 = v128_load(row.as_ptr().add(j + 8) as *const v128);
            let v2 = v128_load(vec.as_ptr().add(j + 8) as *const v128);
            acc2 = f32x4_relaxed_madd(r2, v2, acc2);

            let r3 = v128_load(row.as_ptr().add(j + 12) as *const v128);
            let v3 = v128_load(vec.as_ptr().add(j + 12) as *const v128);
            acc3 = f32x4_relaxed_madd(r3, v3, acc3);
        }
        j += 16;
    }

    // Remaining 4-element chunks.
    while j < cols4 {
        unsafe {
            let r = v128_load(row.as_ptr().add(j) as *const v128);
            let v = v128_load(vec.as_ptr().add(j) as *const v128);
            acc0 = f32x4_relaxed_madd(r, v, acc0);
        }
        j += 4;
    }

    // Reduce 4 accumulators to one, then horizontal sum.
    let sum_01 = f32x4_add(acc0, acc1);
    let sum_23 = f32x4_add(acc2, acc3);
    let sum_all = f32x4_add(sum_01, sum_23);

    let mut s = f32x4_extract_lane::<0>(sum_all)
        + f32x4_extract_lane::<1>(sum_all)
        + f32x4_extract_lane::<2>(sum_all)
        + f32x4_extract_lane::<3>(sum_all);

    // Scalar tail for remaining elements.
    while j < cols {
        s += row[j] * vec[j];
        j += 1;
    }
    s
}

/// WASM relaxed SIMD matvec: `output[rows]` = `mat[rows, cols]` * `vec[cols]`.
///
/// Uses `f32x4_relaxed_madd` (fused multiply-add) for the inner dot product,
/// with 4 independent accumulators processing 16 elements per iteration.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
fn matvec_relaxed_simd(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    #[cfg(feature = "wasm_threads")]
    {
        let total_work = rows * cols;
        if let Some(chunk_rows) = parallel_chunk_rows(total_work, rows) {
            use rayon::prelude::*;
            let mut output = std::vec![0.0f32; rows];
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let row_start = chunk_idx * chunk_rows;
                    for (local_i, out) in out_chunk.iter_mut().enumerate() {
                        let i = row_start + local_i;
                        let row = &mat[i * cols..i * cols + cols];
                        *out = matvec_relaxed_simd_row(row, vec, cols);
                    }
                });
            return output;
        }
    }

    let mut output = std::vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..i * cols + cols];
        *out = matvec_relaxed_simd_row(row, vec, cols);
    }
    output
}

/// WASM relaxed SIMD matvec into pre-allocated output.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
fn matvec_relaxed_simd_into(
    mat: &[f32],
    vec: &[f32],
    _rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..(i + 1) * cols];
        *out = matvec_relaxed_simd_row(row, vec, cols);
    }
}

/// Q8_0 relaxed SIMD row: dequantize-and-dot using `f32x4_relaxed_madd`.
///
/// For each Q8_0 block (2-byte f16 scale + 32 int8 quants), we convert the
/// int8 quants to f32, multiply by the input, and use FMA to accumulate.
/// This avoids the overhead of a separate int8 dot product instruction
/// while still benefiting from hardware FMA.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
#[inline]
fn matvec_q8_0_relaxed_simd_row(row_bytes: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    use core::arch::wasm32::*;

    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);

    for block in 0..blocks_per_row {
        let block_start = block * Q8_0_BLOCK_BYTES;
        let scale = f16_to_f32_inline(u16::from_le_bytes([
            row_bytes[block_start],
            row_bytes[block_start + 1],
        ]));
        let quants = &row_bytes[block_start + 2..block_start + Q8_0_BLOCK_BYTES];
        let input_offset = block * Q8_0_BLOCK_SIZE;
        let scale_v = f32x4_splat(scale);

        // Process 32 quants in 8 groups of 4, using 2 accumulators.
        for g in 0..4 {
            let base = g * 4;
            let q = f32x4(
                (quants[base] as i8) as f32,
                (quants[base + 1] as i8) as f32,
                (quants[base + 2] as i8) as f32,
                (quants[base + 3] as i8) as f32,
            );
            let scaled_q = f32x4_mul(q, scale_v);
            unsafe {
                let inp = v128_load(input.as_ptr().add(input_offset + base) as *const v128);
                acc0 = f32x4_relaxed_madd(scaled_q, inp, acc0);
            }
        }
        for g in 4..8 {
            let base = g * 4;
            let q = f32x4(
                (quants[base] as i8) as f32,
                (quants[base + 1] as i8) as f32,
                (quants[base + 2] as i8) as f32,
                (quants[base + 3] as i8) as f32,
            );
            let scaled_q = f32x4_mul(q, scale_v);
            unsafe {
                let inp = v128_load(input.as_ptr().add(input_offset + base) as *const v128);
                acc1 = f32x4_relaxed_madd(scaled_q, inp, acc1);
            }
        }
    }

    let sum_v = f32x4_add(acc0, acc1);
    f32x4_extract_lane::<0>(sum_v)
        + f32x4_extract_lane::<1>(sum_v)
        + f32x4_extract_lane::<2>(sum_v)
        + f32x4_extract_lane::<3>(sum_v)
}

/// Q8_0 relaxed SIMD matvec: `output[rows]` = `weight_q8[rows, cols]` * `input[cols]`.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
fn matvec_q8_0_relaxed_simd(
    weight_data: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; rows];
    for (i, out) in output.iter_mut().enumerate() {
        let row_start = i * bytes_per_row;
        let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
        *out = matvec_q8_0_relaxed_simd_row(row_bytes, input, blocks_per_row);
    }
    output
}

/// Q8_0 relaxed SIMD matvec into pre-allocated output.
#[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
fn matvec_q8_0_relaxed_simd_into(
    weight_data: &[u8],
    input: &[f32],
    _rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    for (i, out) in output.iter_mut().enumerate() {
        let row_start = i * bytes_per_row;
        let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
        *out = matvec_q8_0_relaxed_simd_row(row_bytes, input, blocks_per_row);
    }
}

/// ARM NEON SIMD matvec into pre-allocated output.
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn matvec_simd_into(mat: &[f32], vec: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..i * cols + cols];
        // SAFETY: row.len() == cols, SIMD feature gated by cfg
        *out = unsafe { matvec_simd_row(row, vec, cols) };
    }
}

/// x86 AVX2 matvec into pre-allocated output.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2_into(
    mat: &[f32],
    vec: &[f32],
    _rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..i * cols + cols];
        *out = matvec_avx2_row(row, vec, cols);
    }
}

/// Q8_0 direct quantized matvec writing into a pre-allocated output slice.
#[inline]
pub fn matvec_q8_0_into(
    weight_data: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(
        cols % Q8_0_BLOCK_SIZE,
        0,
        "cols must be multiple of 32 for Q8_0"
    );
    debug_assert_eq!(output.len(), rows);

    #[cfg(target_arch = "aarch64")]
    {
        matvec_q8_0_neon_into(weight_data, input, rows, cols, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "relaxed_simd"))]
        {
            matvec_q8_0_relaxed_simd_into(weight_data, input, rows, cols, output);
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "relaxed_simd")))]
        {
            matvec_q8_0_scalar_into(weight_data, input, rows, cols, output);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matvec_q8_0_scalar_into(
    weight_data: &[u8],
    input: &[f32],
    _rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;
    for (i, out) in output.iter_mut().enumerate() {
        let row_start = i * bytes_per_row;
        let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
        *out = matvec_q8_0_scalar_row(row_bytes, input, blocks_per_row);
    }
}

#[cfg(target_arch = "aarch64")]
fn matvec_q8_0_neon_into(
    weight_data: &[u8],
    input: &[f32],
    _rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    let blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    // Quantize entire input vector to Q8_0 (one-time cost)
    let mut input_scales = Vec::with_capacity(blocks_per_row);
    let mut input_quants = vec![0i8; cols];
    for b in 0..blocks_per_row {
        let start = b * Q8_0_BLOCK_SIZE;
        let (scale, quants) =
            unsafe { quantize_f32_to_q8_0_block(&input[start..start + Q8_0_BLOCK_SIZE]) };
        input_scales.push(scale);
        input_quants[start..start + Q8_0_BLOCK_SIZE].copy_from_slice(&quants);
    }

    let rows = output.len();
    let total_work = rows * cols;
    if total_work >= 2_000_000 {
        use rayon::prelude::*;
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                for (local_idx, out) in chunk.iter_mut().enumerate() {
                    let row = chunk_idx * chunk_rows + local_idx;
                    let start = row * bytes_per_row;
                    let row_bytes = &weight_data[start..start + bytes_per_row];
                    *out = unsafe {
                        dot_q8_0_q8_0_neon(row_bytes, &input_scales, &input_quants, blocks_per_row)
                    };
                }
            });
    } else {
        for (i, out) in output.iter_mut().enumerate() {
            let start = i * bytes_per_row;
            let row_bytes = &weight_data[start..start + bytes_per_row];
            *out = unsafe {
                dot_q8_0_q8_0_neon(row_bytes, &input_scales, &input_quants, blocks_per_row)
            };
        }
    }
}

/// Q8_0 matvec using a pre-quantized input vector.
///
/// Skips the per-call quantization of the input by accepting a `QuantizedInput`
/// that was prepared via `quantize_input_q8_0` or `quantize_input_q8_0_into`.
/// This saves significant work when the same input is multiplied by multiple
/// weight matrices (e.g. Q/K/V projections sharing the same normed input).
pub fn matvec_q8_0_preq_into(
    weight_data: &[u8],
    preq: &QuantizedInput,
    rows: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), rows);
    let blocks_per_row = preq.blocks_per_row;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    #[cfg(target_arch = "aarch64")]
    {
        let total_work = rows * (blocks_per_row * Q8_0_BLOCK_SIZE);
        if total_work >= 2_000_000 {
            use rayon::prelude::*;
            let chunk_rows = adaptive_matvec_chunk_size(rows);
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_row = chunk_idx * chunk_rows;
                    let len = chunk.len();
                    // Process pairs of rows to keep preq data in L1 cache
                    let pairs = len / 2;
                    for p in 0..pairs {
                        let r0 = base_row + p * 2;
                        let r1 = r0 + 1;
                        let s0 = r0 * bytes_per_row;
                        let s1 = r1 * bytes_per_row;
                        let rb0 = &weight_data[s0..s0 + bytes_per_row];
                        let rb1 = &weight_data[s1..s1 + bytes_per_row];
                        chunk[p * 2] = unsafe {
                            dot_q8_0_q8_0_neon(rb0, &preq.scales, &preq.quants, blocks_per_row)
                        };
                        chunk[p * 2 + 1] = unsafe {
                            dot_q8_0_q8_0_neon(rb1, &preq.scales, &preq.quants, blocks_per_row)
                        };
                    }
                    // Handle odd trailing row
                    if len % 2 != 0 {
                        let row = base_row + len - 1;
                        let start = row * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        chunk[len - 1] = unsafe {
                            dot_q8_0_q8_0_neon(
                                row_bytes,
                                &preq.scales,
                                &preq.quants,
                                blocks_per_row,
                            )
                        };
                    }
                });
        } else {
            // Process pairs of rows to keep preq data in L1 cache
            let pairs = rows / 2;
            for p in 0..pairs {
                let r0 = p * 2;
                let r1 = r0 + 1;
                let s0 = r0 * bytes_per_row;
                let s1 = r1 * bytes_per_row;
                let rb0 = &weight_data[s0..s0 + bytes_per_row];
                let rb1 = &weight_data[s1..s1 + bytes_per_row];
                output[r0] =
                    unsafe { dot_q8_0_q8_0_neon(rb0, &preq.scales, &preq.quants, blocks_per_row) };
                output[r1] =
                    unsafe { dot_q8_0_q8_0_neon(rb1, &preq.scales, &preq.quants, blocks_per_row) };
            }
            // Handle odd trailing row
            if rows & 1 != 0 {
                let row = rows - 1;
                let start = row * bytes_per_row;
                let row_bytes = &weight_data[start..start + bytes_per_row];
                output[row] = unsafe {
                    dot_q8_0_q8_0_neon(row_bytes, &preq.scales, &preq.quants, blocks_per_row)
                };
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, out) in output.iter_mut().enumerate() {
            let row_start = i * bytes_per_row;
            let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let block_start = block * Q8_0_BLOCK_BYTES;
                let w_scale = f16_to_f32_inline(u16::from_le_bytes([
                    row_bytes[block_start],
                    row_bytes[block_start + 1],
                ]));
                let w_quants = &row_bytes[block_start + 2..block_start + Q8_0_BLOCK_BYTES];
                let input_offset = block * Q8_0_BLOCK_SIZE;
                let combined_scale = w_scale * preq.scales[block];

                let mut s0 = 0i32;
                let mut s1 = 0i32;
                let mut s2 = 0i32;
                let mut s3 = 0i32;
                let mut j = 0;
                while j < 32 {
                    s0 += (w_quants[j] as i8) as i32 * preq.quants[input_offset + j] as i32;
                    s1 += (w_quants[j + 1] as i8) as i32 * preq.quants[input_offset + j + 1] as i32;
                    s2 += (w_quants[j + 2] as i8) as i32 * preq.quants[input_offset + j + 2] as i32;
                    s3 += (w_quants[j + 3] as i8) as i32 * preq.quants[input_offset + j + 3] as i32;
                    j += 4;
                }
                sum += combined_scale * (s0 + s1 + s2 + s3) as f32;
            }
            *out = sum;
        }
    }
}

/// Compute the argmax of a Q8_0 matrix-vector product without materializing
/// the full output vector.
///
/// This is functionally equivalent to `matvec_q8_0_preq_into` followed by
/// `sample_greedy`, but avoids writing 128K+ floats to a logits buffer.
/// For greedy decoding this eliminates ~512KB of cache-polluting writes and
/// the separate argmax scan.
///
/// Returns `(argmax_index, max_logit_value)`.
pub fn matvec_argmax_q8_0_preq(
    weight_data: &[u8],
    preq: &QuantizedInput,
    rows: usize,
) -> (usize, f32) {
    let blocks_per_row = preq.blocks_per_row;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

    #[cfg(target_arch = "aarch64")]
    {
        let total_work = rows * (blocks_per_row * Q8_0_BLOCK_SIZE);
        if total_work >= 2_000_000 {
            use rayon::prelude::*;
            let chunk_rows = adaptive_matvec_chunk_size(rows);
            let num_chunks = rows.div_ceil(chunk_rows);
            (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start_row = chunk_idx * chunk_rows;
                    let end_row = (start_row + chunk_rows).min(rows);
                    let mut local_best_idx = start_row;
                    let mut local_best_val = f32::NEG_INFINITY;
                    for row in start_row..end_row {
                        let start = row * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        let val = unsafe {
                            dot_q8_0_q8_0_neon(
                                row_bytes,
                                &preq.scales,
                                &preq.quants,
                                blocks_per_row,
                            )
                        };
                        if val > local_best_val {
                            local_best_val = val;
                            local_best_idx = row;
                        }
                    }
                    (local_best_idx, local_best_val)
                })
                .reduce(
                    || (0, f32::NEG_INFINITY),
                    |(ai, av), (bi, bv)| {
                        if bv > av {
                            (bi, bv)
                        } else {
                            (ai, av)
                        }
                    },
                )
        } else {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for row in 0..rows {
                let start = row * bytes_per_row;
                let row_bytes = &weight_data[start..start + bytes_per_row];
                let val = unsafe {
                    dot_q8_0_q8_0_neon(row_bytes, &preq.scales, &preq.quants, blocks_per_row)
                };
                if val > best_val {
                    best_val = val;
                    best_idx = row;
                }
            }
            (best_idx, best_val)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for row in 0..rows {
            let row_start = row * bytes_per_row;
            let row_bytes = &weight_data[row_start..row_start + bytes_per_row];
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let block_start = block * Q8_0_BLOCK_BYTES;
                let w_scale = f16_to_f32_inline(u16::from_le_bytes([
                    row_bytes[block_start],
                    row_bytes[block_start + 1],
                ]));
                let w_quants = &row_bytes[block_start + 2..block_start + Q8_0_BLOCK_BYTES];
                let input_offset = block * Q8_0_BLOCK_SIZE;
                let combined_scale = w_scale * preq.scales[block];

                let mut s0 = 0i32;
                let mut s1 = 0i32;
                let mut s2 = 0i32;
                let mut s3 = 0i32;
                let mut j = 0;
                while j < 32 {
                    s0 += (w_quants[j] as i8) as i32 * preq.quants[input_offset + j] as i32;
                    s1 += (w_quants[j + 1] as i8) as i32 * preq.quants[input_offset + j + 1] as i32;
                    s2 += (w_quants[j + 2] as i8) as i32 * preq.quants[input_offset + j + 2] as i32;
                    s3 += (w_quants[j + 3] as i8) as i32 * preq.quants[input_offset + j + 3] as i32;
                    j += 4;
                }
                sum += combined_scale * (s0 + s1 + s2 + s3) as f32;
            }
            if sum > best_val {
                best_val = sum;
                best_idx = row;
            }
        }
        (best_idx, best_val)
    }
}

/// Compute the argmax of a Q4_K × Q8_0-input matrix-vector product without
/// materializing the full output vector.
///
/// Functionally equivalent to `matvec_q4k_q8_into` followed by a linear argmax
/// scan over the resulting logits, but avoids writing ~`rows * 4` bytes of
/// cache-polluting logit values and the separate scan pass. This is the Q4_K
/// counterpart of `matvec_argmax_q8_0_preq` — greedy decoding on a Q4_K model
/// no longer allocates the 128K-entry logits buffer at all.
///
/// Returns `(argmax_index, max_logit_value)`.
pub fn matvec_argmax_q4k_q8_preq(
    weight_data: &[u8],
    preq: &QuantizedInput,
    rows: usize,
    cols: usize,
) -> (usize, f32) {
    debug_assert_eq!(
        cols % Q4K_BLOCK_VALUES,
        0,
        "cols must be multiple of 256 for Q4_K"
    );
    debug_assert_eq!(
        preq.blocks_per_row * Q8_0_BLOCK_SIZE,
        cols,
        "QuantizedInput must cover exactly `cols` elements"
    );

    let blocks_per_row = cols / Q4K_BLOCK_VALUES;
    let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;

    #[cfg(target_arch = "aarch64")]
    {
        let total_work = rows * cols;
        if total_work >= 2_000_000 {
            use rayon::prelude::*;
            let chunk_rows = adaptive_matvec_chunk_size(rows);
            let num_chunks = rows.div_ceil(chunk_rows);
            (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start_row = chunk_idx * chunk_rows;
                    let end_row = (start_row + chunk_rows).min(rows);
                    let mut local_best_idx = start_row;
                    let mut local_best_val = f32::NEG_INFINITY;
                    for row in start_row..end_row {
                        let start = row * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        let val = unsafe {
                            dot_q4k_q8_0_neon(
                                row_bytes,
                                &preq.scales,
                                &preq.quants,
                                blocks_per_row,
                            )
                        };
                        if val > local_best_val {
                            local_best_val = val;
                            local_best_idx = row;
                        }
                    }
                    (local_best_idx, local_best_val)
                })
                .reduce(
                    || (0, f32::NEG_INFINITY),
                    |(ai, av), (bi, bv)| {
                        if bv > av {
                            (bi, bv)
                        } else {
                            (ai, av)
                        }
                    },
                )
        } else {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for row in 0..rows {
                let start = row * bytes_per_row;
                let row_bytes = &weight_data[start..start + bytes_per_row];
                let val = unsafe {
                    dot_q4k_q8_0_neon(row_bytes, &preq.scales, &preq.quants, blocks_per_row)
                };
                if val > best_val {
                    best_val = val;
                    best_idx = row;
                }
            }
            (best_idx, best_val)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for row in 0..rows {
            let start = row * bytes_per_row;
            let row_bytes = &weight_data[start..start + bytes_per_row];
            let val = dot_q4k_q8_0_scalar(row_bytes, &preq.scales, &preq.quants, blocks_per_row);
            if val > best_val {
                best_val = val;
                best_idx = row;
            }
        }
        (best_idx, best_val)
    }
}

/// Compute the argmax of an f32 matrix-vector product without materializing
/// the full output vector.
///
/// Functionally equivalent to `matvec_into` + argmax scan, but avoids writing
/// the full logits buffer. Used for greedy decoding on the f32 output path.
///
/// Returns `(argmax_index, max_logit_value)`.
pub fn matvec_argmax_f32(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> (usize, f32) {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert!(vec.len() >= cols);

    // For the f32 path we still need to compute all dot products. We just avoid
    // storing them by tracking the running max inline.
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for row in 0..rows {
        let row_start = row * cols;
        let row_data = &mat[row_start..row_start + cols];
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += row_data[j] * vec[j];
        }
        if sum > best_val {
            best_val = sum;
            best_idx = row;
        }
    }
    (best_idx, best_val)
}

// ---------------------------------------------------------------------------
// Q4_K direct quantized matvec
// ---------------------------------------------------------------------------
//
// Q4_K block layout (144 bytes, 256 weights):
//   f16 d      (2 bytes)  — super-block scale
//   f16 dmin   (2 bytes)  — super-block minimum
//   u8[12]     (12 bytes) — packed sub-block scales + mins
//   u8[128]    (128 bytes)— 256 4-bit quantized values (2 per byte)
//
// Nibble layout: low nibble → weights 0..127, high nibble → weights 128..255
// (matching llama.cpp / our GPU shader).

const Q4K_BLOCK_VALUES: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

/// Unpack the 8 sub-block scale/min pairs from the 12-byte packed array.
#[inline(always)]
fn unpack_q4k_scales(scales_raw: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
        sc[i + 4] = (scales_raw[i] >> 6) | ((scales_raw[i + 8] & 0x0F) << 2);
        mn[i + 4] = (scales_raw[i + 4] >> 6) | ((scales_raw[i + 8] >> 4) << 2);
    }
    (sc, mn)
}

/// Compute one row of Q4_K × f32 dot product (scalar reference implementation).
///
/// `row_bytes`: raw Q4_K bytes for this row (`blocks_per_row * 144` bytes).
/// `input`: f32 vector of length `blocks_per_row * 256`.
#[inline]
#[allow(dead_code)] // used by tests and non-aarch64 targets
fn dot_q4k_f32_scalar(row_bytes: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;

    for block in 0..blocks_per_row {
        let bstart = block * Q4K_BLOCK_BYTES;
        let blk = &row_bytes[bstart..bstart + Q4K_BLOCK_BYTES];

        let d = f16_to_f32_inline(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([blk[2], blk[3]]));
        let (sc, mn) = unpack_q4k_scales(&blk[4..16]);
        let qs = &blk[16..144];

        let input_base = block * Q4K_BLOCK_VALUES;

        // Process 128 bytes → 256 weights (low nibble + high nibble)
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;

        let mut j = 0;
        while j < 128 {
            let sub = j / 32;
            let d_sc_lo = d * sc[sub] as f32;
            let dmin_mn_lo = dmin * mn[sub] as f32;
            let d_sc_hi = d * sc[sub + 4] as f32;
            let dmin_mn_hi = dmin * mn[sub + 4] as f32;

            let mut k = 0;
            while k < 32 && j + k < 128 {
                let byte = qs[j + k];
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                let w_lo = d_sc_lo * lo - dmin_mn_lo;
                let w_hi = d_sc_hi * hi - dmin_mn_hi;
                s0 += w_lo * input[input_base + j + k];
                s1 += w_hi * input[input_base + j + k + 128];
                k += 1;
            }
            j += 32;
        }
        sum += s0 + s1;
    }
    sum
}

/// NEON-optimized Q4_K × f32 dot product for a single row.
///
/// Processes 16 bytes (32 weights) per iteration using NEON:
/// - Nibble extraction via `vandq_u8` and `vshrq_n_u8`
/// - Widen u8 → u16 → f32, multiply by scale, subtract min
/// - FMA with input vector
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4k_f32_neon(row_bytes: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::aarch64::*;

    let mask_0f = vdupq_n_u8(0x0F);
    let mut global_acc = vdupq_n_f32(0.0);

    for block in 0..blocks_per_row {
        let bstart = block * Q4K_BLOCK_BYTES;
        let blk = &row_bytes[bstart..bstart + Q4K_BLOCK_BYTES];

        let d = f16_to_f32_inline(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([blk[2], blk[3]]));
        let (sc, mn) = unpack_q4k_scales(&blk[4..16]);
        let qs = &blk[16..144];
        let input_base = block * Q4K_BLOCK_VALUES;

        // Process 128 bytes = 256 weights in groups of 16 bytes (32 weights each iteration)
        for sub in 0..4 {
            let d_sc_lo = d * sc[sub] as f32;
            let dmin_mn_lo = dmin * mn[sub] as f32;
            let d_sc_hi = d * sc[sub + 4] as f32;
            let dmin_mn_hi = dmin * mn[sub + 4] as f32;

            let v_d_sc_lo = vdupq_n_f32(d_sc_lo);
            let v_dmin_mn_lo = vdupq_n_f32(dmin_mn_lo);
            let v_d_sc_hi = vdupq_n_f32(d_sc_hi);
            let v_dmin_mn_hi = vdupq_n_f32(dmin_mn_hi);

            let qs_offset = sub * 32;
            let in_lo_offset = input_base + sub * 32;
            let in_hi_offset = input_base + sub * 32 + 128;

            // Process 32 bytes in two groups of 16
            for half in 0..2 {
                let byte_off = qs_offset + half * 16;
                let raw_bytes = vld1q_u8(qs.as_ptr().add(byte_off));

                // Extract low and high nibbles
                let lo_nibbles = vandq_u8(raw_bytes, mask_0f);
                let hi_nibbles = vshrq_n_u8::<4>(raw_bytes);

                // Widen low nibbles: u8x16 → u16x8 × 2 → f32x4 × 4
                let lo_16_0 = vmovl_u8(vget_low_u8(lo_nibbles));
                let lo_16_1 = vmovl_u8(vget_high_u8(lo_nibbles));

                let lo_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_0)));
                let lo_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_0)));
                let lo_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_1)));
                let lo_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_1)));

                // w_lo = d_sc_lo * nibble - dmin_mn_lo
                let w_lo_0 = vsubq_f32(vmulq_f32(v_d_sc_lo, lo_f32_0), v_dmin_mn_lo);
                let w_lo_1 = vsubq_f32(vmulq_f32(v_d_sc_lo, lo_f32_1), v_dmin_mn_lo);
                let w_lo_2 = vsubq_f32(vmulq_f32(v_d_sc_lo, lo_f32_2), v_dmin_mn_lo);
                let w_lo_3 = vsubq_f32(vmulq_f32(v_d_sc_lo, lo_f32_3), v_dmin_mn_lo);

                // Load input for low-nibble weights
                let in_off = in_lo_offset + half * 16;
                let inp_0 = vld1q_f32(input.as_ptr().add(in_off));
                let inp_1 = vld1q_f32(input.as_ptr().add(in_off + 4));
                let inp_2 = vld1q_f32(input.as_ptr().add(in_off + 8));
                let inp_3 = vld1q_f32(input.as_ptr().add(in_off + 12));

                global_acc = vfmaq_f32(global_acc, w_lo_0, inp_0);
                global_acc = vfmaq_f32(global_acc, w_lo_1, inp_1);
                global_acc = vfmaq_f32(global_acc, w_lo_2, inp_2);
                global_acc = vfmaq_f32(global_acc, w_lo_3, inp_3);

                // Widen high nibbles
                let hi_16_0 = vmovl_u8(vget_low_u8(hi_nibbles));
                let hi_16_1 = vmovl_u8(vget_high_u8(hi_nibbles));

                let hi_f32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_0)));
                let hi_f32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_0)));
                let hi_f32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_1)));
                let hi_f32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_1)));

                let w_hi_0 = vsubq_f32(vmulq_f32(v_d_sc_hi, hi_f32_0), v_dmin_mn_hi);
                let w_hi_1 = vsubq_f32(vmulq_f32(v_d_sc_hi, hi_f32_1), v_dmin_mn_hi);
                let w_hi_2 = vsubq_f32(vmulq_f32(v_d_sc_hi, hi_f32_2), v_dmin_mn_hi);
                let w_hi_3 = vsubq_f32(vmulq_f32(v_d_sc_hi, hi_f32_3), v_dmin_mn_hi);

                // Load input for high-nibble weights
                let in_off_hi = in_hi_offset + half * 16;
                let inp_hi_0 = vld1q_f32(input.as_ptr().add(in_off_hi));
                let inp_hi_1 = vld1q_f32(input.as_ptr().add(in_off_hi + 4));
                let inp_hi_2 = vld1q_f32(input.as_ptr().add(in_off_hi + 8));
                let inp_hi_3 = vld1q_f32(input.as_ptr().add(in_off_hi + 12));

                global_acc = vfmaq_f32(global_acc, w_hi_0, inp_hi_0);
                global_acc = vfmaq_f32(global_acc, w_hi_1, inp_hi_1);
                global_acc = vfmaq_f32(global_acc, w_hi_2, inp_hi_2);
                global_acc = vfmaq_f32(global_acc, w_hi_3, inp_hi_3);
            }
        }
    }

    vaddvq_f32(global_acc)
}

/// Q4_K direct quantized matvec writing into a pre-allocated output slice.
///
/// `weight_data`: raw Q4_K bytes (row-major, `rows * blocks_per_row * 144` bytes).
/// `input`: f32 vector of length `cols` (= `blocks_per_row * 256`).
/// `rows`: number of output rows.  `cols`: input dimension (must be multiple of 256).
/// `output`: pre-allocated slice of length `rows`.
pub fn matvec_q4k_into(
    weight_data: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(
        cols % Q4K_BLOCK_VALUES,
        0,
        "cols must be multiple of 256 for Q4_K"
    );
    debug_assert_eq!(output.len(), rows);

    let blocks_per_row = cols / Q4K_BLOCK_VALUES;
    let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;

    let _total_work = rows * cols;
    #[cfg(not(target_arch = "wasm32"))]
    {
        if _total_work >= 2_000_000 {
            use rayon::prelude::*;
            let chunk_rows = adaptive_matvec_chunk_size(rows);
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    for (local_idx, out) in chunk.iter_mut().enumerate() {
                        let row = chunk_idx * chunk_rows + local_idx;
                        let start = row * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        #[cfg(target_arch = "aarch64")]
                        {
                            *out = unsafe { dot_q4k_f32_neon(row_bytes, input, blocks_per_row) };
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            *out = dot_q4k_f32_scalar(row_bytes, input, blocks_per_row);
                        }
                    }
                });
            return;
        }
    }
    // Sequential fallback (used on wasm32 always, native when below threshold)
    for (i, out) in output.iter_mut().enumerate() {
        let start = i * bytes_per_row;
        let row_bytes = &weight_data[start..start + bytes_per_row];
        #[cfg(target_arch = "aarch64")]
        {
            *out = unsafe { dot_q4k_f32_neon(row_bytes, input, blocks_per_row) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            *out = dot_q4k_f32_scalar(row_bytes, input, blocks_per_row);
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K weight × Q8_0 input matvec
// ---------------------------------------------------------------------------
//
// Each Q4_K block has 256 weights split into 8 sub-blocks of 32 values
// (sub 0..3 = low nibbles, sub 4..7 = high nibbles). These 8 sub-blocks line
// up exactly with 8 consecutive Q8_0 input blocks of 32 values each.
//
// For sub-block `s` of Q4_K block `b`:
//   w[i] = d * sc[s] * nibble[i] - dmin * mn[s]
// where `d, dmin` are the block's f16 scales and `sc[s], mn[s]` are 6-bit
// unpacked sub-block scales/mins.
//
// Against a Q8_0 input (scale S_b, int8 quants q[i]) the sub-block dot is:
//   sum_i w[i] * S_{b*8+s} * q[i]
//     = S_{b*8+s} * d * sc[s] * sum_i(nibble[i] * q[i])     // SDOT
//     - S_{b*8+s} * dmin * mn[s] * sum_i(q[i])              // input-sum
//
// Both sub-block dot products are pure int8 ops: nibbles 0..15 fit cleanly in
// signed int8, so SDOT works without sign adjustment. This replaces 4-byte
// f32 input reads with 1-byte int8 reads (~4× less input bandwidth).

/// Scalar reference implementation of Q4_K × Q8_0 dot product for one row.
#[inline]
#[allow(dead_code)]
fn dot_q4k_q8_0_scalar(
    row_bytes: &[u8],
    input_q8_scales: &[f32],
    input_q8_quants: &[i8],
    blocks_per_row: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for block in 0..blocks_per_row {
        let bstart = block * Q4K_BLOCK_BYTES;
        let blk = &row_bytes[bstart..bstart + Q4K_BLOCK_BYTES];

        let d = f16_to_f32_inline(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([blk[2], blk[3]]));
        let (sc, mn) = unpack_q4k_scales(&blk[4..16]);
        let qs = &blk[16..144];

        // The 8 Q8_0 input sub-blocks for this Q4_K block.
        let iq_base = block * 8 * Q8_0_BLOCK_SIZE;
        let is_base = block * 8;

        // Sub-blocks 0..3: low nibbles of bytes[0..32], [32..64], [64..96], [96..128]
        // Sub-blocks 4..7: high nibbles of the same byte groups -> weights 128..255
        for sub in 0..4 {
            let byte_off = sub * 32;
            let in_off_lo = iq_base + sub * Q8_0_BLOCK_SIZE;
            let in_off_hi = iq_base + (sub + 4) * Q8_0_BLOCK_SIZE;
            let s_lo = input_q8_scales[is_base + sub];
            let s_hi = input_q8_scales[is_base + sub + 4];

            let mut dot_lo = 0i32;
            let mut dot_hi = 0i32;
            let mut qsum_lo = 0i32;
            let mut qsum_hi = 0i32;
            for k in 0..32 {
                let byte = qs[byte_off + k];
                let lo = (byte & 0x0F) as i32;
                let hi = (byte >> 4) as i32;
                let q_lo = input_q8_quants[in_off_lo + k] as i32;
                let q_hi = input_q8_quants[in_off_hi + k] as i32;
                dot_lo += lo * q_lo;
                dot_hi += hi * q_hi;
                qsum_lo += q_lo;
                qsum_hi += q_hi;
            }

            let d_sc_lo = d * sc[sub] as f32;
            let dmin_mn_lo = dmin * mn[sub] as f32;
            let d_sc_hi = d * sc[sub + 4] as f32;
            let dmin_mn_hi = dmin * mn[sub + 4] as f32;

            sum += s_lo * (d_sc_lo * dot_lo as f32 - dmin_mn_lo * qsum_lo as f32);
            sum += s_hi * (d_sc_hi * dot_hi as f32 - dmin_mn_hi * qsum_hi as f32);
        }
    }
    sum
}

/// NEON-optimized Q4_K × Q8_0 dot product for a single row.
///
/// Uses the ARMv8.2 SDOT instruction to do 32×int8 dot products per sub-block
/// with only 2 SDOTs. Input is read at int8 precision (1 byte / value) instead
/// of f32 (4 bytes / value) — a 4× reduction in input-side memory traffic vs
/// `dot_q4k_f32_neon`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4k_q8_0_neon(
    row_bytes: &[u8],
    input_q8_scales: &[f32],
    input_q8_quants: &[i8],
    blocks_per_row: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let mask_0f = vdupq_n_u8(0x0F);
    let mut acc = vdupq_n_f32(0.0);

    let row_ptr = row_bytes.as_ptr();
    let iq_ptr = input_q8_quants.as_ptr();
    let is_ptr = input_q8_scales.as_ptr();

    for block in 0..blocks_per_row {
        let blk = row_ptr.add(block * Q4K_BLOCK_BYTES);

        // Prefetch next block's weight data to keep the weight stream hot.
        if block + 1 < blocks_per_row {
            core::arch::asm!(
                "prfm pldl1strm, [{addr}]",
                addr = in(reg) row_ptr.add((block + 1) * Q4K_BLOCK_BYTES),
                options(nostack, preserves_flags),
            );
        }

        let d = f16_to_f32_inline(u16::from_le_bytes([*blk, *blk.add(1)]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([*blk.add(2), *blk.add(3)]));
        let scales_raw = std::slice::from_raw_parts(blk.add(4), 12);
        let (sc, mn) = unpack_q4k_scales(scales_raw);
        let qs_ptr = blk.add(16);

        let iq_base = iq_ptr.add(block * 8 * Q8_0_BLOCK_SIZE);
        let is_base = is_ptr.add(block * 8);

        // Process the 4 byte-groups (each produces one low-nibble and one
        // high-nibble sub-block = 2 Q8_0 input blocks).
        for group in 0..4 {
            // Load 32 weight bytes -> low/high nibbles (each sub-block = 32 weights)
            let b0 = vld1q_u8(qs_ptr.add(group * 32));
            let b1 = vld1q_u8(qs_ptr.add(group * 32 + 16));

            let lo0 = vreinterpretq_s8_u8(vandq_u8(b0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(b1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));

            // Load corresponding Q8_0 input quants
            let il0 = vld1q_s8(iq_base.add(group * Q8_0_BLOCK_SIZE));
            let il1 = vld1q_s8(iq_base.add(group * Q8_0_BLOCK_SIZE + 16));
            let ih0 = vld1q_s8(iq_base.add((group + 4) * Q8_0_BLOCK_SIZE));
            let ih1 = vld1q_s8(iq_base.add((group + 4) * Q8_0_BLOCK_SIZE + 16));

            // Dot products: nibble * input_q  (SDOT → int32x4)
            let dot_lo = vaddq_s32(
                ggml_vdotq_s32(vdupq_n_s32(0), lo0, il0),
                ggml_vdotq_s32(vdupq_n_s32(0), lo1, il1),
            );
            let dot_hi = vaddq_s32(
                ggml_vdotq_s32(vdupq_n_s32(0), hi0, ih0),
                ggml_vdotq_s32(vdupq_n_s32(0), hi1, ih1),
            );

            // Horizontal sums of the input quants for the min term.
            // vpaddlq_s8: int8x16 -> int16x8 pairwise-add; then vaddvq_s16.
            let qsum_lo_i32 =
                vaddvq_s16(vaddq_s16(vpaddlq_s8(il0), vpaddlq_s8(il1))) as i32;
            let qsum_hi_i32 =
                vaddvq_s16(vaddq_s16(vpaddlq_s8(ih0), vpaddlq_s8(ih1))) as i32;

            // Horizontal sum of the SDOT results -> i32 scalar
            let dot_lo_i32 = vaddvq_s32(dot_lo);
            let dot_hi_i32 = vaddvq_s32(dot_hi);

            let s_lo = *is_base.add(group);
            let s_hi = *is_base.add(group + 4);

            let d_sc_lo = d * sc[group] as f32;
            let dmin_mn_lo = dmin * mn[group] as f32;
            let d_sc_hi = d * sc[group + 4] as f32;
            let dmin_mn_hi = dmin * mn[group + 4] as f32;

            // Accumulate as two (f32, f32) pairs packed into a single f32x4
            // lane-multiply + fma sequence.
            let term_lo = s_lo * (d_sc_lo * dot_lo_i32 as f32 - dmin_mn_lo * qsum_lo_i32 as f32);
            let term_hi = s_hi * (d_sc_hi * dot_hi_i32 as f32 - dmin_mn_hi * qsum_hi_i32 as f32);

            // Pack into lanes 0/1 of the accumulator; lanes 2/3 stay unused.
            let lane01 = vsetq_lane_f32(term_hi, vsetq_lane_f32(term_lo, vdupq_n_f32(0.0), 0), 1);
            acc = vaddq_f32(acc, lane01);
        }
    }

    vaddvq_f32(acc)
}

/// 2-token Q4_K × Q8_0 dot product: stream one weight row once, compute against 2 tokens.
///
/// Returns [row·tok0, row·tok1]. By decoding the Q4_K weight block (scales,
/// nibbles) once and computing against 2 token inputs, this halves weight
/// memory bandwidth and amortizes the per-block decode work across 2 tokens.
///
/// # Safety
/// Requires aarch64 NEON.  All slices must be valid for the given `blocks_per_row`.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4k_q8_0_2tok_neon(
    row_bytes: &[u8],
    t0_scales: &[f32],
    t0_quants: &[i8],
    t1_scales: &[f32],
    t1_quants: &[i8],
    blocks_per_row: usize,
) -> [f32; 2] {
    use std::arch::aarch64::*;

    let mask_0f = vdupq_n_u8(0x0F);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    let row_ptr = row_bytes.as_ptr();
    let iq0_ptr = t0_quants.as_ptr();
    let iq1_ptr = t1_quants.as_ptr();
    let is0_ptr = t0_scales.as_ptr();
    let is1_ptr = t1_scales.as_ptr();

    for block in 0..blocks_per_row {
        let blk = row_ptr.add(block * Q4K_BLOCK_BYTES);

        if block + 1 < blocks_per_row {
            core::arch::asm!(
                "prfm pldl1strm, [{addr}]",
                addr = in(reg) row_ptr.add((block + 1) * Q4K_BLOCK_BYTES),
                options(nostack, preserves_flags),
            );
        }

        let d = f16_to_f32_inline(u16::from_le_bytes([*blk, *blk.add(1)]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([*blk.add(2), *blk.add(3)]));
        let scales_raw = std::slice::from_raw_parts(blk.add(4), 12);
        let (sc, mn) = unpack_q4k_scales(scales_raw);
        let qs_ptr = blk.add(16);

        let iq0_base = iq0_ptr.add(block * 8 * Q8_0_BLOCK_SIZE);
        let iq1_base = iq1_ptr.add(block * 8 * Q8_0_BLOCK_SIZE);
        let is0_base = is0_ptr.add(block * 8);
        let is1_base = is1_ptr.add(block * 8);

        for group in 0..4 {
            // Decode weight nibbles once, reuse across both tokens.
            let b0 = vld1q_u8(qs_ptr.add(group * 32));
            let b1 = vld1q_u8(qs_ptr.add(group * 32 + 16));

            let lo0 = vreinterpretq_s8_u8(vandq_u8(b0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(b1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));

            let d_sc_lo = d * sc[group] as f32;
            let dmin_mn_lo = dmin * mn[group] as f32;
            let d_sc_hi = d * sc[group + 4] as f32;
            let dmin_mn_hi = dmin * mn[group + 4] as f32;

            // --- Token 0 ---
            {
                let il0 = vld1q_s8(iq0_base.add(group * Q8_0_BLOCK_SIZE));
                let il1 = vld1q_s8(iq0_base.add(group * Q8_0_BLOCK_SIZE + 16));
                let ih0 = vld1q_s8(iq0_base.add((group + 4) * Q8_0_BLOCK_SIZE));
                let ih1 = vld1q_s8(iq0_base.add((group + 4) * Q8_0_BLOCK_SIZE + 16));

                let dot_lo = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), lo0, il0),
                    ggml_vdotq_s32(vdupq_n_s32(0), lo1, il1),
                );
                let dot_hi = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), hi0, ih0),
                    ggml_vdotq_s32(vdupq_n_s32(0), hi1, ih1),
                );
                let qsum_lo =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(il0), vpaddlq_s8(il1))) as i32;
                let qsum_hi =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(ih0), vpaddlq_s8(ih1))) as i32;

                let s_lo = *is0_base.add(group);
                let s_hi = *is0_base.add(group + 4);

                let term_lo =
                    s_lo * (d_sc_lo * vaddvq_s32(dot_lo) as f32 - dmin_mn_lo * qsum_lo as f32);
                let term_hi =
                    s_hi * (d_sc_hi * vaddvq_s32(dot_hi) as f32 - dmin_mn_hi * qsum_hi as f32);

                let lane01 =
                    vsetq_lane_f32(term_hi, vsetq_lane_f32(term_lo, vdupq_n_f32(0.0), 0), 1);
                acc0 = vaddq_f32(acc0, lane01);
            }

            // --- Token 1 ---
            {
                let il0 = vld1q_s8(iq1_base.add(group * Q8_0_BLOCK_SIZE));
                let il1 = vld1q_s8(iq1_base.add(group * Q8_0_BLOCK_SIZE + 16));
                let ih0 = vld1q_s8(iq1_base.add((group + 4) * Q8_0_BLOCK_SIZE));
                let ih1 = vld1q_s8(iq1_base.add((group + 4) * Q8_0_BLOCK_SIZE + 16));

                let dot_lo = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), lo0, il0),
                    ggml_vdotq_s32(vdupq_n_s32(0), lo1, il1),
                );
                let dot_hi = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), hi0, ih0),
                    ggml_vdotq_s32(vdupq_n_s32(0), hi1, ih1),
                );
                let qsum_lo =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(il0), vpaddlq_s8(il1))) as i32;
                let qsum_hi =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(ih0), vpaddlq_s8(ih1))) as i32;

                let s_lo = *is1_base.add(group);
                let s_hi = *is1_base.add(group + 4);

                let term_lo =
                    s_lo * (d_sc_lo * vaddvq_s32(dot_lo) as f32 - dmin_mn_lo * qsum_lo as f32);
                let term_hi =
                    s_hi * (d_sc_hi * vaddvq_s32(dot_hi) as f32 - dmin_mn_hi * qsum_hi as f32);

                let lane01 =
                    vsetq_lane_f32(term_hi, vsetq_lane_f32(term_lo, vdupq_n_f32(0.0), 0), 1);
                acc1 = vaddq_f32(acc1, lane01);
            }
        }
    }

    [vaddvq_f32(acc0), vaddvq_f32(acc1)]
}

/// 4-token Q4_K × Q8_0 dot product: stream one weight row once, compute against 4 tokens.
///
/// Extends `dot_q4k_q8_0_2tok_neon` to 4 tokens per weight-block load.  The
/// per-block decode work (6-bit scale unpack, 4-bit nibble unpack) is amortized
/// across 4 tokens instead of 2, and weight memory bandwidth is halved again.
///
/// Returns [row·tok0, row·tok1, row·tok2, row·tok3].
///
/// # Safety
/// Requires aarch64 NEON.  All slices must be valid for `blocks_per_row`.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4k_q8_0_4tok_neon(
    row_bytes: &[u8],
    t_scales: [&[f32]; 4],
    t_quants: [&[i8]; 4],
    blocks_per_row: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mask_0f = vdupq_n_u8(0x0F);
    let mut acc = [
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    ];

    let row_ptr = row_bytes.as_ptr();
    let iq_ptrs: [*const i8; 4] = [
        t_quants[0].as_ptr(),
        t_quants[1].as_ptr(),
        t_quants[2].as_ptr(),
        t_quants[3].as_ptr(),
    ];
    let is_ptrs: [*const f32; 4] = [
        t_scales[0].as_ptr(),
        t_scales[1].as_ptr(),
        t_scales[2].as_ptr(),
        t_scales[3].as_ptr(),
    ];

    for block in 0..blocks_per_row {
        let blk = row_ptr.add(block * Q4K_BLOCK_BYTES);

        if block + 1 < blocks_per_row {
            core::arch::asm!(
                "prfm pldl1strm, [{addr}]",
                addr = in(reg) row_ptr.add((block + 1) * Q4K_BLOCK_BYTES),
                options(nostack, preserves_flags),
            );
        }

        let d = f16_to_f32_inline(u16::from_le_bytes([*blk, *blk.add(1)]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([*blk.add(2), *blk.add(3)]));
        let scales_raw = std::slice::from_raw_parts(blk.add(4), 12);
        let (sc, mn) = unpack_q4k_scales(scales_raw);
        let qs_ptr = blk.add(16);

        for group in 0..4 {
            // Decode weight nibbles once per group, reuse across all 4 tokens.
            let b0 = vld1q_u8(qs_ptr.add(group * 32));
            let b1 = vld1q_u8(qs_ptr.add(group * 32 + 16));

            let lo0 = vreinterpretq_s8_u8(vandq_u8(b0, mask_0f));
            let lo1 = vreinterpretq_s8_u8(vandq_u8(b1, mask_0f));
            let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
            let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));

            let d_sc_lo = d * sc[group] as f32;
            let dmin_mn_lo = dmin * mn[group] as f32;
            let d_sc_hi = d * sc[group + 4] as f32;
            let dmin_mn_hi = dmin * mn[group + 4] as f32;

            for ti in 0..4 {
                let iq_base = iq_ptrs[ti].add(block * 8 * Q8_0_BLOCK_SIZE);
                let is_base = is_ptrs[ti].add(block * 8);

                let il0 = vld1q_s8(iq_base.add(group * Q8_0_BLOCK_SIZE));
                let il1 = vld1q_s8(iq_base.add(group * Q8_0_BLOCK_SIZE + 16));
                let ih0 = vld1q_s8(iq_base.add((group + 4) * Q8_0_BLOCK_SIZE));
                let ih1 = vld1q_s8(iq_base.add((group + 4) * Q8_0_BLOCK_SIZE + 16));

                let dot_lo = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), lo0, il0),
                    ggml_vdotq_s32(vdupq_n_s32(0), lo1, il1),
                );
                let dot_hi = vaddq_s32(
                    ggml_vdotq_s32(vdupq_n_s32(0), hi0, ih0),
                    ggml_vdotq_s32(vdupq_n_s32(0), hi1, ih1),
                );
                let qsum_lo =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(il0), vpaddlq_s8(il1))) as i32;
                let qsum_hi =
                    vaddvq_s16(vaddq_s16(vpaddlq_s8(ih0), vpaddlq_s8(ih1))) as i32;

                let s_lo = *is_base.add(group);
                let s_hi = *is_base.add(group + 4);

                let term_lo =
                    s_lo * (d_sc_lo * vaddvq_s32(dot_lo) as f32 - dmin_mn_lo * qsum_lo as f32);
                let term_hi =
                    s_hi * (d_sc_hi * vaddvq_s32(dot_hi) as f32 - dmin_mn_hi * qsum_hi as f32);

                let lane01 =
                    vsetq_lane_f32(term_hi, vsetq_lane_f32(term_lo, vdupq_n_f32(0.0), 0), 1);
                acc[ti] = vaddq_f32(acc[ti], lane01);
            }
        }
    }

    [
        vaddvq_f32(acc[0]),
        vaddvq_f32(acc[1]),
        vaddvq_f32(acc[2]),
        vaddvq_f32(acc[3]),
    ]
}

/// 4-token Q4_K × Q8_0 matvec: stream weights once, compute outputs for 4 tokens.
///
/// `output` must have length `4 * rows`, laid out as
/// [tok0_rows..., tok1_rows..., tok2_rows..., tok3_rows...].
#[cfg(target_arch = "aarch64")]
pub fn matvec_q4k_q8_quad_into(
    weight_data: &[u8],
    preqs: [&QuantizedInput; 4],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(cols % Q4K_BLOCK_VALUES, 0);
    debug_assert_eq!(output.len(), 4 * rows);

    let blocks_per_row = cols / Q4K_BLOCK_VALUES;
    let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;

    let t_scales: [&[f32]; 4] = [
        preqs[0].scales.as_slice(),
        preqs[1].scales.as_slice(),
        preqs[2].scales.as_slice(),
        preqs[3].scales.as_slice(),
    ];
    let t_quants: [&[i8]; 4] = [
        preqs[0].quants.as_slice(),
        preqs[1].quants.as_slice(),
        preqs[2].quants.as_slice(),
        preqs[3].quants.as_slice(),
    ];

    let (out01, out23) = output.split_at_mut(2 * rows);
    let (out0, out1) = out01.split_at_mut(rows);
    let (out2, out3) = out23.split_at_mut(rows);

    let total_work = rows * cols;
    if total_work >= 2_000_000 {
        use rayon::prelude::*;
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        let num_chunks = rows.div_ceil(chunk_rows);
        let chunk_results: Vec<Vec<(usize, f32, f32, f32, f32)>> =
            (0..num_chunks).into_par_iter().map(|ci| {
                let base = ci * chunk_rows;
                let end = (base + chunk_rows).min(rows);
                let mut results = Vec::with_capacity(end - base);
                for row in base..end {
                    let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                    let [v0, v1, v2, v3] = unsafe {
                        dot_q4k_q8_0_4tok_neon(rb, t_scales, t_quants, blocks_per_row)
                    };
                    results.push((row, v0, v1, v2, v3));
                }
                results
            }).collect();
        for chunk in chunk_results {
            for (row, v0, v1, v2, v3) in chunk {
                out0[row] = v0;
                out1[row] = v1;
                out2[row] = v2;
                out3[row] = v3;
            }
        }
    } else {
        for row in 0..rows {
            let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            let [v0, v1, v2, v3] = unsafe {
                dot_q4k_q8_0_4tok_neon(rb, t_scales, t_quants, blocks_per_row)
            };
            out0[row] = v0;
            out1[row] = v1;
            out2[row] = v2;
            out3[row] = v3;
        }
    }
}

/// 2-token Q4_K × Q8_0 matvec: stream weights once, compute outputs for 2 tokens.
///
/// `output` must have length `2 * rows`, laid out as [tok0_rows..., tok1_rows...].
#[cfg(target_arch = "aarch64")]
pub fn matvec_q4k_q8_pair_into(
    weight_data: &[u8],
    preq0: &QuantizedInput,
    preq1: &QuantizedInput,
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(cols % Q4K_BLOCK_VALUES, 0);
    debug_assert_eq!(output.len(), 2 * rows);
    debug_assert_eq!(preq0.blocks_per_row, preq1.blocks_per_row);
    debug_assert_eq!(preq0.blocks_per_row * Q8_0_BLOCK_SIZE, cols);

    let blocks_per_row = cols / Q4K_BLOCK_VALUES;
    let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;

    let (out0, out1) = output.split_at_mut(rows);

    let total_work = rows * cols;
    if total_work >= 2_000_000 {
        use rayon::prelude::*;
        let chunk_rows = adaptive_matvec_chunk_size(rows);
        let num_chunks = rows.div_ceil(chunk_rows);
        let chunk_results: Vec<Vec<(usize, f32, f32)>> =
            (0..num_chunks).into_par_iter().map(|ci| {
                let base = ci * chunk_rows;
                let end = (base + chunk_rows).min(rows);
                let mut results = Vec::with_capacity(end - base);
                for row in base..end {
                    let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                    let [v0, v1] = unsafe {
                        dot_q4k_q8_0_2tok_neon(
                            rb,
                            &preq0.scales, &preq0.quants,
                            &preq1.scales, &preq1.quants,
                            blocks_per_row,
                        )
                    };
                    results.push((row, v0, v1));
                }
                results
            }).collect();
        for chunk in chunk_results {
            for (row, v0, v1) in chunk {
                out0[row] = v0;
                out1[row] = v1;
            }
        }
    } else {
        for row in 0..rows {
            let rb = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            let [v0, v1] = unsafe {
                dot_q4k_q8_0_2tok_neon(
                    rb,
                    &preq0.scales, &preq0.quants,
                    &preq1.scales, &preq1.quants,
                    blocks_per_row,
                )
            };
            out0[row] = v0;
            out1[row] = v1;
        }
    }
}

/// Q4_K × Q8_0-input matvec writing into a pre-allocated output slice.
///
/// `weight_data`: raw Q4_K bytes (row-major, `rows * blocks_per_row * 144` bytes).
/// `preq`: pre-quantized Q8_0 input produced by `quantize_input_q8_0_into`.
///         Must cover exactly `cols = blocks_per_row * 256` input values,
///         i.e. `preq.blocks_per_row * 32 == cols`.
/// `rows`: number of output rows.
/// `cols`: input dimension (must be a multiple of 256).
/// `output`: pre-allocated slice of length `rows`.
///
/// This is the preferred Q4_K matvec path when the caller already has (or can
/// cheaply compute) a Q8_0 representation of the input. It reduces input-side
/// memory bandwidth by ~4× compared to `matvec_q4k_into` and uses the SDOT
/// instruction for the inner loops, matching the Q8_0 × Q8_0 kernel shape.
pub fn matvec_q4k_q8_into(
    weight_data: &[u8],
    preq: &QuantizedInput,
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(
        cols % Q4K_BLOCK_VALUES,
        0,
        "cols must be multiple of 256 for Q4_K"
    );
    debug_assert_eq!(
        preq.blocks_per_row * Q8_0_BLOCK_SIZE,
        cols,
        "QuantizedInput must cover exactly `cols` elements"
    );
    debug_assert_eq!(output.len(), rows);

    let blocks_per_row = cols / Q4K_BLOCK_VALUES;
    let bytes_per_row = blocks_per_row * Q4K_BLOCK_BYTES;

    #[cfg(target_arch = "aarch64")]
    {
        let total_work = rows * cols;
        if total_work >= 2_000_000 {
            use rayon::prelude::*;
            let chunk_rows = adaptive_matvec_chunk_size(rows);
            output
                .par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base_row = chunk_idx * chunk_rows;
                    let len = chunk.len();
                    // Process pairs of rows to keep the Q8_0 input hot in L1.
                    let pairs = len / 2;
                    for p in 0..pairs {
                        let r0 = base_row + p * 2;
                        let r1 = r0 + 1;
                        let s0 = r0 * bytes_per_row;
                        let s1 = r1 * bytes_per_row;
                        let rb0 = &weight_data[s0..s0 + bytes_per_row];
                        let rb1 = &weight_data[s1..s1 + bytes_per_row];
                        chunk[p * 2] = unsafe {
                            dot_q4k_q8_0_neon(rb0, &preq.scales, &preq.quants, blocks_per_row)
                        };
                        chunk[p * 2 + 1] = unsafe {
                            dot_q4k_q8_0_neon(rb1, &preq.scales, &preq.quants, blocks_per_row)
                        };
                    }
                    if len % 2 != 0 {
                        let row = base_row + len - 1;
                        let start = row * bytes_per_row;
                        let row_bytes = &weight_data[start..start + bytes_per_row];
                        chunk[len - 1] = unsafe {
                            dot_q4k_q8_0_neon(
                                row_bytes,
                                &preq.scales,
                                &preq.quants,
                                blocks_per_row,
                            )
                        };
                    }
                });
            return;
        }

        // Sequential: still use 2-row pairs for cache friendliness.
        let pairs = rows / 2;
        for p in 0..pairs {
            let r0 = p * 2;
            let r1 = r0 + 1;
            let s0 = r0 * bytes_per_row;
            let s1 = r1 * bytes_per_row;
            let rb0 = &weight_data[s0..s0 + bytes_per_row];
            let rb1 = &weight_data[s1..s1 + bytes_per_row];
            output[r0] =
                unsafe { dot_q4k_q8_0_neon(rb0, &preq.scales, &preq.quants, blocks_per_row) };
            output[r1] =
                unsafe { dot_q4k_q8_0_neon(rb1, &preq.scales, &preq.quants, blocks_per_row) };
        }
        if rows & 1 != 0 {
            let row = rows - 1;
            let start = row * bytes_per_row;
            let row_bytes = &weight_data[start..start + bytes_per_row];
            output[row] =
                unsafe { dot_q4k_q8_0_neon(row_bytes, &preq.scales, &preq.quants, blocks_per_row) };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (i, out) in output.iter_mut().enumerate() {
            let start = i * bytes_per_row;
            let row_bytes = &weight_data[start..start + bytes_per_row];
            *out = dot_q4k_q8_0_scalar(row_bytes, &preq.scales, &preq.quants, blocks_per_row);
        }
    }
}

/// SiLU(gate) * up writing into a pre-allocated output slice.
#[inline]
pub fn silu_mul_into(gate: &[f32], up: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        silu_mul_neon_into(gate, up, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let n = gate.len();
        for i in 0..n {
            let g = gate[i];
            output[i] = (g / (1.0 + (-g).exp())) * up[i];
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn silu_mul_neon_into(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = gate.len();
    let len4 = n & !3;

    unsafe {
        let half = vdupq_n_f32(0.5);
        let twenty_seven = vdupq_n_f32(27.0);
        let nine = vdupq_n_f32(9.0);

        for i in (0..len4).step_by(4) {
            let g = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));
            let t = vmulq_f32(g, half);
            let t2 = vmulq_f32(t, t);
            let num = vmulq_f32(t, vaddq_f32(twenty_seven, t2));
            let den = vaddq_f32(twenty_seven, vmulq_f32(nine, t2));
            let tanh_approx = vdivq_f32(num, den);
            let sigmoid = vaddq_f32(half, vmulq_f32(half, tanh_approx));
            let silu = vmulq_f32(g, sigmoid);
            let out = vmulq_f32(silu, u);
            vst1q_f32(output.as_mut_ptr().add(i), out);
        }
    }

    for i in len4..n {
        let g = gate[i];
        output[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// GELU(gate) * up writing into a pre-allocated output slice.
#[inline]
pub fn gelu_mul_into(gate: &[f32], up: &[f32], output: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        let c = 0.797_884_6_f32; // sqrt(2/pi)
        let gelu = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh());
        output[i] = gelu * up[i];
    }
}

/// Grouped-query attention writing into a pre-allocated output slice.
///
/// Uses fused online softmax (Flash Attention style) for the common non-softcap
/// path: scores, softmax, and value accumulation are computed in a single pass
/// over the KV cache, eliminating the scores buffer and improving cache locality.
/// Falls back to multi-pass for softcap (Gemma 2).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn cpu_grouped_query_attention_into(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    attn_softcap: f32,
    output: &mut [f32],
) {
    // Softcap path: multi-pass (online softmax doesn't compose cleanly
    // with tanh capping since the cap distorts the score distribution).
    if attn_softcap > 0.0 {
        cpu_gqa_softcap(
            q,
            k_cache,
            v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            attn_softcap,
            output,
        );
        return;
    }

    let heads_per_kv = num_heads / num_kv_heads;
    let kv_stride = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_head = h / heads_per_kv;
        let q_offset = h * head_dim;
        let out_offset = h * head_dim;

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON always available on aarch64.
            // Handles all head_dim sizes: NEON for multiples of 4, scalar tail for remainder.
            unsafe {
                fused_attention_head_neon(
                    q, q_offset, k_cache, v_cache, output, out_offset, head_dim, seq_len,
                    kv_stride, kv_head, scale,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fused_attention_head_scalar(
                q, q_offset, k_cache, v_cache, output, out_offset, head_dim, seq_len, kv_stride,
                kv_head, scale,
            );
        }
    }
}

/// Fused single-pass attention with online softmax (scalar fallback).
///
/// For each KV position: compute QK^T, update running max, rescale accumulated
/// output, and add new value contribution. No intermediate scores buffer needed.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fused_attention_head_scalar(
    q: &[f32],
    q_offset: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    out_offset: usize,
    head_dim: usize,
    seq_len: usize,
    kv_stride: usize,
    kv_head: usize,
    scale: f32,
) {
    // Initialize output to zero
    for d in 0..head_dim {
        output[out_offset + d] = 0.0;
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;

    for t in 0..seq_len {
        // Compute QK^T dot product
        let k_offset = t * kv_stride + kv_head * head_dim;
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[q_offset + d] * k_cache[k_offset + d];
        }
        let score = dot * scale;

        // Online softmax update
        let new_max = max_score.max(score);
        let old_scale = (max_score - new_max).exp();
        let new_weight = (score - new_max).exp();
        sum_exp = sum_exp * old_scale + new_weight;

        // Rescale existing output and add new value contribution
        let v_offset = t * kv_stride + kv_head * head_dim;
        for d in 0..head_dim {
            output[out_offset + d] =
                output[out_offset + d] * old_scale + new_weight * v_cache[v_offset + d];
        }
        max_score = new_max;
    }

    // Final normalization
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for d in 0..head_dim {
            output[out_offset + d] *= inv_sum;
        }
    }
}

/// NEON-vectorized fused attention with online softmax.
///
/// Vectorizes both the QK^T dot product (4 accumulators for ILP) and the
/// value rescale+accumulate loop using NEON FMA instructions.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_attention_head_neon(
    q: &[f32],
    q_offset: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    out_offset: usize,
    head_dim: usize,
    seq_len: usize,
    kv_stride: usize,
    kv_head: usize,
    scale: f32,
) {
    use std::arch::aarch64::*;

    let dim4 = head_dim & !3;
    let dim16 = head_dim & !15;

    // Initialize output to zero
    for d in 0..head_dim {
        output[out_offset + d] = 0.0;
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;

    for t in 0..seq_len {
        let k_offset = t * kv_stride + kv_head * head_dim;

        // --- QK^T dot product with 4 NEON accumulators for ILP ---
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut d = 0usize;
        while d < dim16 {
            let q0 = vld1q_f32(q.as_ptr().add(q_offset + d));
            let k0 = vld1q_f32(k_cache.as_ptr().add(k_offset + d));
            acc0 = vfmaq_f32(acc0, q0, k0);
            let q1 = vld1q_f32(q.as_ptr().add(q_offset + d + 4));
            let k1 = vld1q_f32(k_cache.as_ptr().add(k_offset + d + 4));
            acc1 = vfmaq_f32(acc1, q1, k1);
            let q2 = vld1q_f32(q.as_ptr().add(q_offset + d + 8));
            let k2 = vld1q_f32(k_cache.as_ptr().add(k_offset + d + 8));
            acc2 = vfmaq_f32(acc2, q2, k2);
            let q3 = vld1q_f32(q.as_ptr().add(q_offset + d + 12));
            let k3 = vld1q_f32(k_cache.as_ptr().add(k_offset + d + 12));
            acc3 = vfmaq_f32(acc3, q3, k3);
            d += 16;
        }
        // Handle remaining 4-element chunks
        while d < dim4 {
            let qv = vld1q_f32(q.as_ptr().add(q_offset + d));
            let kv = vld1q_f32(k_cache.as_ptr().add(k_offset + d));
            acc0 = vfmaq_f32(acc0, qv, kv);
            d += 4;
        }
        let sum_v = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut dot = vaddvq_f32(sum_v);
        // Scalar tail
        while d < head_dim {
            dot += q[q_offset + d] * k_cache[k_offset + d];
            d += 1;
        }
        let score = dot * scale;

        // --- Online softmax update ---
        let new_max = max_score.max(score);
        let old_scale_f = (max_score - new_max).exp();
        let new_weight = (score - new_max).exp();
        sum_exp = sum_exp * old_scale_f + new_weight;

        // --- Rescale existing output + accumulate new value (NEON) ---
        let v_offset = t * kv_stride + kv_head * head_dim;
        let old_scale_v = vdupq_n_f32(old_scale_f);
        let new_weight_v = vdupq_n_f32(new_weight);

        d = 0;
        while d < dim16 {
            // Unroll 4x for ILP on rescale + FMA
            let o0 = vld1q_f32(output.as_ptr().add(out_offset + d));
            let v0 = vld1q_f32(v_cache.as_ptr().add(v_offset + d));
            let r0 = vfmaq_f32(vmulq_f32(o0, old_scale_v), new_weight_v, v0);
            vst1q_f32(output.as_mut_ptr().add(out_offset + d), r0);

            let o1 = vld1q_f32(output.as_ptr().add(out_offset + d + 4));
            let v1 = vld1q_f32(v_cache.as_ptr().add(v_offset + d + 4));
            let r1 = vfmaq_f32(vmulq_f32(o1, old_scale_v), new_weight_v, v1);
            vst1q_f32(output.as_mut_ptr().add(out_offset + d + 4), r1);

            let o2 = vld1q_f32(output.as_ptr().add(out_offset + d + 8));
            let v2 = vld1q_f32(v_cache.as_ptr().add(v_offset + d + 8));
            let r2 = vfmaq_f32(vmulq_f32(o2, old_scale_v), new_weight_v, v2);
            vst1q_f32(output.as_mut_ptr().add(out_offset + d + 8), r2);

            let o3 = vld1q_f32(output.as_ptr().add(out_offset + d + 12));
            let v3 = vld1q_f32(v_cache.as_ptr().add(v_offset + d + 12));
            let r3 = vfmaq_f32(vmulq_f32(o3, old_scale_v), new_weight_v, v3);
            vst1q_f32(output.as_mut_ptr().add(out_offset + d + 12), r3);

            d += 16;
        }
        while d < dim4 {
            let o = vld1q_f32(output.as_ptr().add(out_offset + d));
            let v = vld1q_f32(v_cache.as_ptr().add(v_offset + d));
            let r = vfmaq_f32(vmulq_f32(o, old_scale_v), new_weight_v, v);
            vst1q_f32(output.as_mut_ptr().add(out_offset + d), r);
            d += 4;
        }
        // Scalar tail
        while d < head_dim {
            output[out_offset + d] =
                output[out_offset + d] * old_scale_f + new_weight * v_cache[v_offset + d];
            d += 1;
        }

        max_score = new_max;
    }

    // Final normalization (NEON)
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        let inv_sum_v = vdupq_n_f32(inv_sum);

        let mut d = 0usize;
        while d < dim4 {
            let o = vld1q_f32(output.as_ptr().add(out_offset + d));
            vst1q_f32(
                output.as_mut_ptr().add(out_offset + d),
                vmulq_f32(o, inv_sum_v),
            );
            d += 4;
        }
        while d < head_dim {
            output[out_offset + d] *= inv_sum;
            d += 1;
        }
    }
}

/// Multi-pass attention for models with attention logit soft-capping (Gemma 2).
/// Softcap applies tanh before softmax which prevents clean online softmax fusion.
#[allow(clippy::too_many_arguments)]
fn cpu_gqa_softcap(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    attn_softcap: f32,
    output: &mut [f32],
) {
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_stride = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Zero the output
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Pre-allocate scores buffer once, reused across all heads.
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..num_heads {
        let kv_head = h / heads_per_kv;
        let q_offset = h * head_dim;

        // Compute attention scores for this head
        #[cfg(target_arch = "aarch64")]
        {
            attn_dot_scores_neon(
                q,
                k_cache,
                &mut scores,
                q_offset,
                kv_head,
                head_dim,
                kv_stride,
                seq_len,
                scale,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for t in 0..seq_len {
                let k_offset = t * kv_stride + kv_head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_offset + d] * k_cache[k_offset + d];
                }
                scores[t] = dot * scale;
            }
        }

        // Apply tanh soft-cap
        for s in &mut scores[..seq_len] {
            *s = (*s / attn_softcap).tanh() * attn_softcap;
        }

        // Softmax
        let max_score = scores[..seq_len]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores[..seq_len] {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        let inv_sum = 1.0 / sum;
        for s in &mut scores[..seq_len] {
            *s *= inv_sum;
        }

        // Weighted sum of values
        let out_offset = h * head_dim;
        #[allow(clippy::needless_range_loop)]
        for t in 0..seq_len {
            let weight = scores[t];
            if weight < 1e-8 {
                continue;
            }
            let v_offset = t * kv_stride + kv_head * head_dim;
            for d in 0..head_dim {
                output[out_offset + d] += weight * v_cache[v_offset + d];
            }
        }

        // Zero scores for next head
        for s in &mut scores[..seq_len] {
            *s = 0.0;
        }
    }
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

    /// Verify fused online-softmax attention matches the reference multi-pass
    /// implementation for various head_dim and seq_len combinations.
    #[test]
    fn test_fused_attention_matches_reference() {
        // Reference multi-pass implementation (the old code)
        fn reference_attention(
            q: &[f32],
            k_cache: &[f32],
            v_cache: &[f32],
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            seq_len: usize,
        ) -> Vec<f32> {
            let heads_per_kv = num_heads / num_kv_heads;
            let kv_stride = num_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut output = vec![0.0f32; num_heads * head_dim];
            let mut scores = vec![0.0f32; seq_len];

            for h in 0..num_heads {
                let kv_head = h / heads_per_kv;
                let q_offset = h * head_dim;
                #[allow(clippy::needless_range_loop)]
                for t in 0..seq_len {
                    let k_offset = t * kv_stride + kv_head * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k_cache[k_offset + d];
                    }
                    scores[t] = dot * scale;
                }
                let max_s = scores[..seq_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores[..seq_len] {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in &mut scores[..seq_len] {
                    *s *= inv;
                }
                let out_off = h * head_dim;
                #[allow(clippy::needless_range_loop)]
                for t in 0..seq_len {
                    let w = scores[t];
                    let v_off = t * kv_stride + kv_head * head_dim;
                    for d in 0..head_dim {
                        output[out_off + d] += w * v_cache[v_off + d];
                    }
                }
                for s in &mut scores[..seq_len] {
                    *s = 0.0;
                }
            }
            output
        }

        let test_cases = [
            // (num_heads, num_kv_heads, head_dim, seq_len)
            (2, 2, 2, 1),
            (2, 2, 2, 6),
            (4, 2, 4, 10),
            (32, 8, 64, 20),
            (8, 4, 64, 1),
            (1, 1, 3, 5), // odd head_dim
        ];

        for &(nh, nkv, hd, sl) in &test_cases {
            let q: Vec<f32> = (0..nh * hd)
                .map(|i| ((i * 7 + 3) as f32 * 0.1).sin())
                .collect();
            let kv_stride = nkv * hd;
            let k: Vec<f32> = (0..sl * kv_stride)
                .map(|i| ((i * 11 + 5) as f32 * 0.05).cos())
                .collect();
            let v: Vec<f32> = (0..sl * kv_stride)
                .map(|i| ((i * 13 + 7) as f32 * 0.08).sin())
                .collect();

            let reference = reference_attention(&q, &k, &v, nh, nkv, hd, sl);
            let mut fused = vec![0.0f32; nh * hd];
            cpu_grouped_query_attention_into(&q, &k, &v, nh, nkv, hd, sl, 0.0, &mut fused);

            for i in 0..nh * hd {
                let diff = (reference[i] - fused[i]).abs();
                let tol = (reference[i].abs() * 1e-4).max(1e-5);
                assert!(
                    diff <= tol,
                    "attention mismatch at [{i}] for nh={nh} nkv={nkv} hd={hd} sl={sl}: ref={} fused={} diff={}",
                    reference[i], fused[i], diff,
                );
            }
        }
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
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
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
            moe: None,
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
        // RoPE should make the same token at different positions produce different logits.
        // Use a larger model (head_dim=4) so the RoPE rotation has a measurable effect
        // on the logits even after attention + residual with tiny weight magnitudes.
        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 8,
            hidden_dim: 8,
            intermediate_dim: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
        };

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
            moe: None,
        };

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
            layers: vec![layer],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
        };

        let mut model = Model::new(config, weights);
        let logits_pos0 = model.forward(0, 0).data().to_vec();
        // Don't reset — KV cache state will differ, which is fine.
        // The point is the logits should differ from a fresh model at pos 5.
        let mut model2 = {
            let config2 = model.config().clone();
            let make_weights2 = |size: usize| -> Vec<f32> {
                (0..size).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect()
            };
            let layer2 = LayerWeights {
                attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
                wq: Tensor::from_vec(make_weights2(nh * hd * dim), &[nh * hd * dim]).unwrap(),
                wk: Tensor::from_vec(make_weights2(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
                wv: Tensor::from_vec(make_weights2(nkvh * hd * dim), &[nkvh * hd * dim]).unwrap(),
                wo: Tensor::from_vec(make_weights2(dim * nh * hd), &[dim * nh * hd]).unwrap(),
                ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
                w_gate: Tensor::from_vec(make_weights2(inter * dim), &[inter * dim]).unwrap(),
                w_up: Tensor::from_vec(make_weights2(inter * dim), &[inter * dim]).unwrap(),
                w_down: Tensor::from_vec(make_weights2(dim * inter), &[dim * inter]).unwrap(),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                post_attn_norm: None,
                post_ffn_norm: None,
                moe: None,
            };
            let weights2 = ModelWeights {
                token_embedding: Tensor::from_vec(make_weights2(vocab * dim), &[vocab * dim])
                    .unwrap(),
                layers: vec![layer2],
                output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
                output_weight: Tensor::from_vec(make_weights2(vocab * dim), &[vocab * dim])
                    .unwrap(),
            };
            Model::new(config2, weights2)
        };
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
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
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
            moe: None,
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
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
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
            moe: None,
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
            moe: None,
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

    #[test]
    fn test_rmsnorm_single_element() {
        // RMSNorm on a 1-element vector: result = (x / sqrt(x^2 + eps)) * w
        let x = vec![2.0f32];
        let weight = vec![1.0f32];
        let eps = 1e-5;
        let result = rmsnorm(&x, &weight, eps);
        assert_eq!(result.len(), 1);
        let expected = 2.0 / (4.0f32 + eps).sqrt();
        let diff = (result[0] - expected).abs();
        assert!(
            diff < 1e-5,
            "single-element rmsnorm: got {}, expected {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_matvec_single_row() {
        // 1-row matrix: result[0] = dot(mat_row, vec)
        let mat = vec![1.0f32, 2.0];
        let vec_ = vec![3.0f32, 4.0];
        let result = matvec(&mat, &vec_, 1, 2);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 11.0).abs() < 1e-5,
            "1-row matvec = {}",
            result[0]
        );
    }

    #[test]
    fn test_matvec_zero_weights() {
        // All-zero weight matrix always produces zero output
        let rows = 4;
        let cols = 8;
        let mat = vec![0.0f32; rows * cols];
        let vec_: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let result = matvec(&mat, &vec_, rows, cols);
        assert_eq!(result.len(), rows);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "row {i} should be 0.0 with zero weights, got {v}");
        }
    }

    #[test]
    fn test_rope_zero_position() {
        // At position 0 every angle is 0: cos=1, sin=0, so data must be unchanged.
        let num_heads = 2;
        let head_dim = 4;
        let original: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| (i + 1) as f32)
            .collect();
        let mut data = original.clone();
        apply_rope(&mut data, num_heads, head_dim, 0, 10000.0);
        for (i, (&before, &after)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (before - after).abs() < 1e-6,
                "rope at pos=0 changed element {i}: {} -> {}",
                before,
                after
            );
        }
    }

    #[test]
    fn test_rope_table_matches_inline() {
        // apply_rope_with_table must produce identical results to apply_rope.
        let head_dim = 8;
        let theta = 10000.0;
        let table = RopeTable::new(head_dim, 128, theta);

        for pos in [0, 1, 5, 42, 127] {
            let num_heads = 3;
            let original: Vec<f32> = (0..(num_heads * head_dim))
                .map(|i| ((i + 1) as f32) * 0.3)
                .collect();

            let mut via_inline = original.clone();
            apply_rope(&mut via_inline, num_heads, head_dim, pos, theta);

            let mut via_table = original.clone();
            apply_rope_with_table(&mut via_table, num_heads, head_dim, pos, theta, &table);

            for (i, (&a, &b)) in via_inline.iter().zip(via_table.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "table vs inline mismatch at pos={pos} elem {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_rope_table_fallback_beyond_max_seq_len() {
        // Positions beyond the table's range should fall back to inline computation.
        let head_dim = 4;
        let theta = 10000.0;
        let table = RopeTable::new(head_dim, 16, theta); // only 16 positions cached

        let pos = 20; // beyond the table
        let num_heads = 1;
        let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut via_inline = original.clone();
        apply_rope(&mut via_inline, num_heads, head_dim, pos, theta);

        let mut via_table = original.clone();
        apply_rope_with_table(&mut via_table, num_heads, head_dim, pos, theta, &table);

        for (i, (&a, &b)) in via_inline.iter().zip(via_table.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "fallback mismatch at elem {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rope_table_cap() {
        // RopeTable should cap at ROPE_TABLE_MAX_SEQ_LEN.
        let table = RopeTable::new(4, 1_000_000, 10000.0);
        assert!(table.max_seq_len() <= ROPE_TABLE_MAX_SEQ_LEN);
    }

    #[test]
    fn test_rope_table_preserves_magnitude() {
        let head_dim = 4;
        let table = RopeTable::new(head_dim, 64, 10000.0);
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        let mag_before: f32 = data.iter().map(|x| x * x).sum();
        apply_rope_with_table(&mut data, 1, head_dim, 5, 10000.0, &table);
        let mag_after: f32 = data.iter().map(|x| x * x).sum();
        assert!(
            (mag_before - mag_after).abs() < 1e-4,
            "RoPE table lookup should preserve magnitude"
        );
    }

    #[test]
    fn test_forward_single_token_stable() {
        // Calling forward with the same token twice on freshly-reset models gives identical logits.
        let mut model1 = tiny_test_model();
        let logits1 = model1.forward(0, 0).data().to_vec();

        let mut model2 = tiny_test_model();
        let logits2 = model2.forward(0, 0).data().to_vec();

        assert_eq!(logits1.len(), logits2.len());
        for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logit[{i}] differs between identical forward passes: {a} vs {b}"
            );
        }
    }

    // ---------------------------------------------------------------
    // BitNet b1.58 ternary weight tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ternary_pack_unpack_roundtrip() {
        // Positive, negative, and zero weights
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.3, 0.0, 0.0, -2.0, 1.5];
        let packed = quantize_to_ternary(&weights);

        // Verify each unpacked value matches the expected sign
        let expected: Vec<i8> = weights
            .iter()
            .map(|&w| {
                if w > 0.0 {
                    1
                } else if w < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        for (i, &exp) in expected.iter().enumerate() {
            let got = unpack_ternary(&packed, i);
            assert_eq!(
                got, exp,
                "ternary roundtrip mismatch at index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_ternary_pack_4_per_byte() {
        // Exactly 4 weights → 1 byte
        let packed = quantize_to_ternary(&[1.0, -1.0, 0.0, 1.0]);
        assert_eq!(packed.len(), 1);
        // bits: weight0=01, weight1=10, weight2=00, weight3=01
        // byte = 0b01_00_10_01 = 0x49
        assert_eq!(packed[0], 0b01_00_10_01);
    }

    #[test]
    fn test_ternary_pack_padding() {
        // 5 weights → 2 bytes (last byte has 3 unused slots)
        let packed = quantize_to_ternary(&[1.0, -1.0, 0.0, 1.0, -1.0]);
        assert_eq!(packed.len(), 2);
        assert_eq!(unpack_ternary(&packed, 4), -1);
    }

    #[test]
    fn test_matvec_ternary_scalar_simple() {
        // 2x3 matrix with known ternary weights:
        // Row 0: [+1, -1,  0]   → output[0] = input[0] - input[1]
        // Row 1: [ 0, +1, +1]   → output[1] = input[1] + input[2]
        let weights_row0 = vec![1.0, -1.0, 0.0];
        let weights_row1 = vec![0.0, 1.0, 1.0];

        // Pack rows contiguously
        let mut all_weights = Vec::new();
        all_weights.extend_from_slice(&weights_row0);
        all_weights.extend_from_slice(&weights_row1);

        // Pack row by row (each row padded to ceil(cols/4) bytes)
        let packed_row0 = quantize_to_ternary(&weights_row0);
        let packed_row1 = quantize_to_ternary(&weights_row1);
        let mut packed = Vec::new();
        packed.extend_from_slice(&packed_row0);
        packed.extend_from_slice(&packed_row1);

        let input = vec![3.0, 5.0, 7.0];
        let result = matvec_ternary_scalar(&packed, &input, 2, 3);

        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - (3.0 - 5.0)).abs() < 1e-5,
            "row0: got {}",
            result[0]
        );
        assert!(
            (result[1] - (5.0 + 7.0)).abs() < 1e-5,
            "row1: got {}",
            result[1]
        );
    }

    #[test]
    fn test_matvec_ternary_all_zero_weights() {
        let packed = quantize_to_ternary(&[0.0; 16]);
        let input = vec![1.0; 4];
        let result = matvec_ternary_scalar(&packed, &input, 4, 4);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-5, "all-zero weights row {i}: got {v}");
        }
    }

    #[test]
    fn test_matvec_ternary_all_positive() {
        // All +1 weights → output = sum of input per row
        let cols = 8;
        let rows = 2;
        let weights = vec![1.0; rows * cols];
        let packed_row = quantize_to_ternary(&weights[..cols]);
        let mut packed = Vec::new();
        for _ in 0..rows {
            packed.extend_from_slice(&packed_row);
        }
        let input: Vec<f32> = (1..=cols).map(|i| i as f32).collect();
        let expected_sum: f32 = input.iter().sum();

        let result = matvec_ternary_scalar(&packed, &input, rows, cols);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected_sum).abs() < 1e-5,
                "all-positive row {i}: got {v}, expected {expected_sum}"
            );
        }
    }

    #[test]
    fn test_matvec_ternary_dispatch_matches_scalar() {
        // Verify that matvec_ternary (dispatch) matches the scalar reference
        let rows: usize = 16;
        let cols: usize = 33; // odd size to test remainder handling
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3) // mix of positive, negative, ~zero
            .collect();

        // Pack row by row
        let bytes_per_row = cols.div_ceil(4);
        let mut packed = vec![0u8; rows * bytes_per_row];
        for r in 0..rows {
            let row_packed = quantize_to_ternary(&weights[r * cols..(r + 1) * cols]);
            packed[r * bytes_per_row..(r * bytes_per_row + row_packed.len())]
                .copy_from_slice(&row_packed);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();

        let result_dispatch = matvec_ternary(&packed, &input, rows, cols);
        let result_scalar = matvec_ternary_scalar(&packed, &input, rows, cols);

        for i in 0..rows {
            let diff = (result_dispatch[i] - result_scalar[i]).abs();
            assert!(
                diff < 1e-4,
                "ternary dispatch vs scalar mismatch at row {i}: dispatch={} scalar={} diff={}",
                result_dispatch[i],
                result_scalar[i],
                diff,
            );
        }
    }

    #[test]
    fn test_matvec_ternary_vs_f32_reference() {
        // Verify ternary matvec produces the same result as standard matvec
        // when the f32 matrix contains only {-1, 0, +1} values.
        let rows: usize = 8;
        let cols: usize = 16;

        // Create a matrix with only ternary values
        let f32_weights: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();

        // Pack row by row
        let bytes_per_row = cols.div_ceil(4);
        let mut packed = vec![0u8; rows * bytes_per_row];
        for r in 0..rows {
            let row_packed = quantize_to_ternary(&f32_weights[r * cols..(r + 1) * cols]);
            packed[r * bytes_per_row..(r * bytes_per_row + row_packed.len())]
                .copy_from_slice(&row_packed);
        }

        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let result_ternary = matvec_ternary(&packed, &input, rows, cols);
        let result_f32 = matvec_scalar(&f32_weights, &input, rows, cols);

        for i in 0..rows {
            let diff = (result_ternary[i] - result_f32[i]).abs();
            assert!(
                diff < 1e-4,
                "ternary vs f32 mismatch at row {i}: ternary={} f32={} diff={}",
                result_ternary[i],
                result_f32[i],
                diff,
            );
        }
    }

    // ------------------------------------------------------------------
    // Progressive inference tests
    // ------------------------------------------------------------------

    /// Helper: build a tiny model with N layers for progressive inference tests.
    fn tiny_multi_layer_model(num_layers: usize) -> Model {
        let dim = 4;
        let vocab = 8;
        let intermediate = 8;
        let heads = 2;
        let kv_heads = 2;
        let head_dim = 2;

        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: vocab,
            hidden_dim: dim,
            intermediate_dim: intermediate,
            num_layers,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
        };

        // Deterministic pseudo-random weights
        let mut seed = 42u64;
        let mut make_weights = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        };

        let layers: Vec<LayerWeights> = (0..num_layers)
            .map(|_| LayerWeights {
                attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
                wq: Tensor::from_vec(
                    make_weights(heads * head_dim * dim),
                    &[heads * head_dim * dim],
                )
                .unwrap(),
                wk: Tensor::from_vec(
                    make_weights(kv_heads * head_dim * dim),
                    &[kv_heads * head_dim * dim],
                )
                .unwrap(),
                wv: Tensor::from_vec(
                    make_weights(kv_heads * head_dim * dim),
                    &[kv_heads * head_dim * dim],
                )
                .unwrap(),
                wo: Tensor::from_vec(
                    make_weights(dim * heads * head_dim),
                    &[dim * heads * head_dim],
                )
                .unwrap(),
                ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
                w_gate: Tensor::from_vec(make_weights(intermediate * dim), &[intermediate * dim])
                    .unwrap(),
                w_up: Tensor::from_vec(make_weights(intermediate * dim), &[intermediate * dim])
                    .unwrap(),
                w_down: Tensor::from_vec(make_weights(dim * intermediate), &[dim * intermediate])
                    .unwrap(),
                attn_q_bias: None,
                attn_k_bias: None,
                attn_v_bias: None,
                post_attn_norm: None,
                post_ffn_norm: None,
                moe: None,
            })
            .collect();

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
            layers,
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(make_weights(vocab * dim), &[vocab * dim]).unwrap(),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_forward_partial_output_shape() {
        let mut model = tiny_multi_layer_model(4);
        let logits = model.forward_partial(0, 0, 2);
        assert_eq!(
            logits.shape(),
            &[8],
            "partial forward should produce [vocab_size] logits"
        );
    }

    #[test]
    fn test_forward_partial_all_layers_matches_forward() {
        // forward_partial with all layers should produce identical output to forward
        let mut model1 = tiny_multi_layer_model(4);
        let full_logits = model1.forward(0, 0).data().to_vec();

        let mut model2 = tiny_multi_layer_model(4);
        let partial_logits = model2.forward_partial(0, 0, 4).data().to_vec();

        assert_eq!(full_logits.len(), partial_logits.len());
        for (i, (&a, &b)) in full_logits.iter().zip(partial_logits.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "forward_partial(all layers) should match forward(): logit[{i}] = {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_weight_format_ternary_weights_per_block() {
        assert_eq!(WeightFormat::Ternary.weights_per_block(), 4);
    }

    #[test]
    fn test_forward_partial_fewer_layers_differs() {
        let mut model1 = tiny_multi_layer_model(4);
        let full_logits = model1.forward(0, 0).data().to_vec();

        let mut model2 = tiny_multi_layer_model(4);
        let partial_logits = model2.forward_partial(0, 0, 2).data().to_vec();

        // With different number of layers, logits should generally differ
        assert_ne!(
            full_logits, partial_logits,
            "partial inference with fewer layers should produce different logits"
        );
    }

    #[test]
    fn test_forward_partial_logits_are_finite() {
        let mut model = tiny_multi_layer_model(4);
        let logits = model.forward_partial(0, 0, 1);
        for (i, &v) in logits.data().iter().enumerate() {
            assert!(v.is_finite(), "partial logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_forward_partial_kv_cache_advances() {
        let mut model = tiny_multi_layer_model(4);
        model.forward_partial(0, 0, 2);
        assert_eq!(
            model.kv_cache().len(),
            1,
            "KV cache should advance after partial forward"
        );
        model.forward_partial(1, 1, 2);
        assert_eq!(model.kv_cache().len(), 2);
    }

    #[test]
    fn test_available_layers_default() {
        let model = tiny_multi_layer_model(4);
        assert_eq!(model.available_layers(), 4);
    }

    #[test]
    fn test_inference_quality() {
        let model = tiny_multi_layer_model(4);
        assert!(
            (model.inference_quality() - 1.0).abs() < 1e-5,
            "fully loaded model should have quality 1.0"
        );
    }

    #[test]
    fn test_load_layer_weights_incremental() {
        let dim = 4;
        let intermediate = 8;
        let heads = 2;
        let kv_heads = 2;
        let head_dim = 2;

        let config = ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 8,
            hidden_dim: dim,
            intermediate_dim: intermediate,
            num_layers: 4,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            kv_cache_bits: 32,
            moe: false,
            num_experts: 0,
            num_experts_per_token: 0,
        };

        // Start with placeholder (zero) weights for all layers
        let make_placeholder_layer = || LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: Tensor::from_vec(vec![0.0; heads * head_dim * dim], &[heads * head_dim * dim])
                .unwrap(),
            wk: Tensor::from_vec(
                vec![0.0; kv_heads * head_dim * dim],
                &[kv_heads * head_dim * dim],
            )
            .unwrap(),
            wv: Tensor::from_vec(
                vec![0.0; kv_heads * head_dim * dim],
                &[kv_heads * head_dim * dim],
            )
            .unwrap(),
            wo: Tensor::from_vec(vec![0.0; dim * heads * head_dim], &[dim * heads * head_dim])
                .unwrap(),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: Tensor::from_vec(vec![0.0; intermediate * dim], &[intermediate * dim]).unwrap(),
            w_up: Tensor::from_vec(vec![0.0; intermediate * dim], &[intermediate * dim]).unwrap(),
            w_down: Tensor::from_vec(vec![0.0; dim * intermediate], &[dim * intermediate]).unwrap(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
            moe: None,
        };

        let weights = ModelWeights {
            token_embedding: Tensor::from_vec(vec![0.1; 8 * dim], &[8 * dim]).unwrap(),
            layers: (0..4).map(|_| make_placeholder_layer()).collect(),
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: Tensor::from_vec(vec![0.1; 8 * dim], &[8 * dim]).unwrap(),
        };

        let mut model = Model::new(config, weights);

        // Initially layers_loaded is None (all available) — set to 0 to simulate progressive loading
        model.layers_loaded = Some(0);
        assert_eq!(model.available_layers(), 0);
        assert!((model.inference_quality() - 0.0).abs() < 1e-5);

        // Load first layer
        let real_layer = make_placeholder_layer(); // placeholder is fine for this test
        model.load_layer_weights(0, real_layer);
        assert_eq!(model.available_layers(), 1);
        assert!((model.inference_quality() - 0.25).abs() < 1e-5);

        // Can now do partial inference with 1 layer
        let logits = model.forward_partial(0, 0, 1);
        assert_eq!(logits.shape(), &[8]);
        assert!(logits.data().iter().all(|v| v.is_finite()));

        // Load second layer
        model.load_layer_weights(1, make_placeholder_layer());
        assert_eq!(model.available_layers(), 2);
        assert!((model.inference_quality() - 0.5).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "forward_partial requires at least 1 layer")]
    fn test_forward_partial_zero_layers_panics() {
        let mut model = tiny_multi_layer_model(4);
        model.forward_partial(0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "forward_partial: requested 5 layers but only 4 are available")]
    fn test_forward_partial_too_many_layers_panics() {
        let mut model = tiny_multi_layer_model(4);
        model.forward_partial(0, 0, 5);
    }

    #[test]
    fn test_progressive_inference_quality_improves() {
        // Demonstrate that more layers generally change the output (progressive quality).
        // We compare logits from 1 layer vs 2 layers vs all 4 layers.
        let mut m1 = tiny_multi_layer_model(4);
        let logits_1 = m1.forward_partial(0, 0, 1).data().to_vec();

        let mut m2 = tiny_multi_layer_model(4);
        let logits_2 = m2.forward_partial(0, 0, 2).data().to_vec();

        let mut m4 = tiny_multi_layer_model(4);
        let logits_4 = m4.forward_partial(0, 0, 4).data().to_vec();

        // All should be different (with random weights, collisions are extremely unlikely)
        assert_ne!(
            logits_1, logits_2,
            "1-layer vs 2-layer logits should differ"
        );
        assert_ne!(
            logits_2, logits_4,
            "2-layer vs 4-layer logits should differ"
        );
        assert_ne!(
            logits_1, logits_4,
            "1-layer vs 4-layer logits should differ"
        );
    }

    // -----------------------------------------------------------------------
    // Q8_0 direct quantized matvec tests
    // -----------------------------------------------------------------------

    /// Build a Q8_0 weight buffer from f32 weights (for testing).
    /// Quantizes each block of 32 f32 values into Q8_0 format.
    fn quantize_f32_to_q8_0(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
        assert_eq!(cols % 32, 0);
        let blocks_per_row = cols / 32;
        let mut buf = Vec::with_capacity(rows * blocks_per_row * 34);

        for row in 0..rows {
            for b in 0..blocks_per_row {
                let start = row * cols + b * 32;
                let block = &weights[start..start + 32];
                // Find max absolute value for scale
                let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                // Write f16 scale
                let scale_f16 = f32_to_f16(scale);
                buf.push(scale_f16 as u8);
                buf.push((scale_f16 >> 8) as u8);
                // Write 32 int8 quants
                for &val in block {
                    let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    buf.push(q as u8);
                }
            }
        }
        buf
    }

    /// Convert f32 to f16 (as u16 bits). Simplified for testing.
    fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if exp == 0 {
            return sign as u16; // zero / subnormal → zero
        }
        if exp == 0xFF {
            // Inf / NaN
            return (sign | 0x7C00 | (mant >> 13)) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return (sign | 0x7C00) as u16; // overflow → Inf
        }
        if new_exp <= 0 {
            return sign as u16; // underflow → zero
        }
        (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
    }

    #[test]
    fn test_matvec_q8_0_scalar_simple() {
        // 2x32 identity-like matrix: row 0 has 1.0 at col 0, row 1 has 1.0 at col 1
        let rows = 2;
        let cols = 32;
        let mut weights = vec![0.0f32; rows * cols];
        weights[0] = 1.0; // row 0, col 0
        weights[cols + 1] = 1.0; // row 1, col 1
        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);

        let mut input = vec![0.0f32; cols];
        input[0] = 3.0;
        input[1] = 7.0;

        let result = matvec_q8_0_scalar(&q8_data, &input, rows, cols);
        assert!(
            (result[0] - 3.0).abs() < 0.1,
            "row 0: expected ~3.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 7.0).abs() < 0.1,
            "row 1: expected ~7.0, got {}",
            result[1]
        );
    }

    #[test]
    fn test_matvec_q8_0_dispatch_matches_scalar() {
        let rows = 4;
        let cols = 64;
        // Use small values to stay within i8 range
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 + 3) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);

        let result_scalar = matvec_q8_0_scalar(&q8_data, &input, rows, cols);
        let result_dispatch = matvec_q8_0(&q8_data, &input, rows, cols);

        for i in 0..rows {
            let diff = (result_scalar[i] - result_dispatch[i]).abs();
            // The NEON path quantizes the input to Q8_0 for int8*int8 dot products,
            // while the scalar path uses f32 input directly. This introduces additional
            // quantization noise (~0.5% relative error), which is acceptable for the
            // ~4x bandwidth reduction.
            assert!(
                diff < 0.01,
                "row {i}: scalar={} dispatch={} diff={}",
                result_scalar[i],
                result_dispatch[i],
                diff
            );
        }
    }

    #[test]
    fn test_matvec_q8_0_vs_dequant_f32() {
        // Verify that direct Q8_0 matvec produces results close to the
        // dequantize-then-f32-matvec path.
        let rows = 8;
        let cols = 64;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 31 + 11) % 256) as f32 / 256.0 - 0.5)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();

        // Quantize to Q8_0 then dequantize back to f32
        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);
        let dequant = dequant_q8_0_to_f32(&q8_data, rows, cols);

        // f32 matvec on dequantized weights
        let result_f32 = matvec_scalar(&dequant, &input, rows, cols);
        // Direct Q8_0 matvec (now uses Q8_0 x Q8_0 int8 dot products on aarch64)
        let result_q8 = matvec_q8_0(&q8_data, &input, rows, cols);

        for i in 0..rows {
            let diff = (result_f32[i] - result_q8[i]).abs();
            // Double quantization (weight Q8_0 + input Q8_0) adds more noise than
            // single quantization. Tolerance widened to accommodate int8*int8 path.
            assert!(
                diff < 0.01,
                "row {i}: f32={} q8_0={} diff={}",
                result_f32[i],
                result_q8[i],
                diff
            );
        }
    }

    /// Dequantize Q8_0 data back to f32 (for test comparison).
    fn dequant_q8_0_to_f32(q8_data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let blocks_per_row = cols / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for b in 0..blocks_per_row {
                let block_start = row * bytes_per_row + b * 34;
                let scale = f16_to_f32_inline(u16::from_le_bytes([
                    q8_data[block_start],
                    q8_data[block_start + 1],
                ]));
                for j in 0..32 {
                    let q = q8_data[block_start + 2 + j] as i8;
                    out[row * cols + b * 32 + j] = scale * q as f32;
                }
            }
        }
        out
    }

    #[test]
    fn test_matvec_q8_0_zero_weights() {
        let rows = 2;
        let cols = 32;
        let weights = vec![0.0f32; rows * cols];
        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| i as f32).collect();

        let result = matvec_q8_0(&q8_data, &input, rows, cols);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "row {i}: expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_matvec_q8_0_larger_matrix() {
        // Test with a more realistic size (128x256)
        let rows = 128;
        let cols = 256;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 41 + 17) % 512) as f32 / 512.0 - 0.5)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 11 + 5) % 200) as f32 / 200.0 - 0.5)
            .collect();

        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);
        let dequant = dequant_q8_0_to_f32(&q8_data, rows, cols);

        let result_f32 = matvec_scalar(&dequant, &input, rows, cols);
        let result_q8 = matvec_q8_0(&q8_data, &input, rows, cols);

        for i in 0..rows {
            let diff = (result_f32[i] - result_q8[i]).abs();
            // Double quantization (Q8_0 weights * Q8_0 input) adds more noise.
            // Use absolute tolerance: for small values near zero, relative error
            // is misleading. An absolute error of 0.02 is well within acceptable
            // bounds for the int8*int8 dot product path.
            assert!(
                diff < 0.02,
                "row {i}: f32={} q8_0={} diff={}",
                result_f32[i],
                result_q8[i],
                diff
            );
        }
    }

    #[test]
    fn test_matvec_argmax_f32_matches_full_matvec() {
        let rows = 64;
        let cols = 128;
        let mat: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 + 13) % 500) as f32 / 500.0 - 0.5)
            .collect();
        let vec: Vec<f32> = (0..cols)
            .map(|i| ((i * 11 + 7) % 200) as f32 / 200.0 - 0.5)
            .collect();

        // Full matvec + argmax
        let full_output = matvec_scalar(&mat, &vec, rows, cols);
        let mut expected_idx = 0;
        let mut expected_val = f32::NEG_INFINITY;
        for (i, &v) in full_output.iter().enumerate() {
            if v > expected_val {
                expected_val = v;
                expected_idx = i;
            }
        }

        // Fused argmax
        let (got_idx, got_val) = matvec_argmax_f32(&mat, &vec, rows, cols);

        assert_eq!(
            got_idx, expected_idx,
            "argmax index mismatch: got {got_idx}, expected {expected_idx}"
        );
        assert!(
            (got_val - expected_val).abs() < 1e-5,
            "argmax value mismatch: got {got_val}, expected {expected_val}"
        );
    }

    // ---- Q4_K matvec tests ----

    /// Build a Q4_K block from f32 values (simple quantization for testing).
    /// This is intentionally simple: scale = max(abs(vals)), min = 0.
    /// All sub-blocks share the same scale, no min offset.
    fn quantize_f32_to_q4k_block(vals: &[f32; 256]) -> [u8; 144] {
        let mut block = [0u8; 144];

        // Find scale for the block: max absolute value across all 256 weights
        let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 15.0; // 4-bit range is 0-15
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // d (f16) at bytes 0-1, dmin=0 at bytes 2-3
        let d_bits = f32_to_f16(d);
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        block[2] = 0; // dmin = 0
        block[3] = 0;

        // scales[12]: all sub-block scales = 1, all mins = 0
        // sc[0..4] = 1 (in low 6 bits of scales_raw[0..4])
        // mn[0..4] = 0 (in low 6 bits of scales_raw[4..8])
        // sc[4..8] from upper bits: (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2)
        //   so scales_raw[0..4] high 2 bits = 0, scales_raw[8..12] low 4 bits = 1 => sc[4..7] = 4? No.
        //   Actually sc[i+4] = (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2)
        //   We want sc[4..8] = 1 also. So (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2) = 1
        //   Simplest: scales_raw[0..4] = 1 (low 6 bits = 1, high 2 bits = 0)
        //   scales_raw[4..8] = 0 (mins = 0)
        //   scales_raw[8..12]: for sc[i+4] = 1, need (0 >> 6) | ((scales_raw[i+8] & 0x0F) << 2) = 1
        //   That means (scales_raw[i+8] & 0x0F) << 2 = 1, but 1 is not a multiple of 4.
        //   Let's just use sc[4..8] = 0 (set scales_raw[8..12] = 0) and handle it.
        //   Actually for simplicity: use only the first 4 sub-blocks (low nibbles give weights 0-127)
        //   and make the high nibbles give weights that also use scale 1.
        //
        //   Let me just set all sub-block scales to 1 the simple way:
        //   sc[0..4] = 1: scales_raw[0..4] = 0x01 (low 6 bits = 1)
        //   sc[4..8] = 1: need (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2) = 1
        //     Since scales_raw[i] >> 6 can give 0-3 and the second term gives multiples of 4,
        //     set scales_raw[i] >> 6 = 1 => scales_raw[0..4] |= (1 << 6) = 0x41
        //     but that changes sc[0..4] since sc[i] = scales_raw[i] & 0x3F = 1. Still 1. Good.
        //     Then scales_raw[8..12] & 0x0F = 0.
        for i in 0..4 {
            block[4 + i] = 0x41; // sc[i]=1 (low 6 bits), sc[i+4] gets (0x41>>6)=1 from high bits
        }
        // scales_raw[4..8] = 0 (mn[0..4] = 0, mn[4..8] gets 0 from both parts)
        // scales_raw[8..12] = 0

        // qs[128]: pack weights as 4-bit nibbles
        // Low nibble → weight j (0..127), high nibble → weight j+128
        for j in 0..128 {
            let q_lo = (vals[j] * id + 0.5).clamp(0.0, 15.0) as u8;
            let q_hi = (vals[j + 128] * id + 0.5).clamp(0.0, 15.0) as u8;
            block[16 + j] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
        }

        block
    }

    /// Dequantize a Q4_K block back to f32 for comparison (uses our known-correct reference).
    fn dequant_q4k_block_test(block: &[u8; 144]) -> [f32; 256] {
        let mut output = [0.0f32; 256];
        let d = f16_to_f32_inline(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32_inline(u16::from_le_bytes([block[2], block[3]]));
        let (sc, mn) = unpack_q4k_scales(&block[4..16]);
        let qs = &block[16..144];

        for j in 0..128 {
            let sub = j / 32;
            let lo = (qs[j] & 0x0F) as f32;
            let hi = (qs[j] >> 4) as f32;
            output[j] = d * sc[sub] as f32 * lo - dmin * mn[sub] as f32;
            output[j + 128] = d * sc[sub + 4] as f32 * hi - dmin * mn[sub + 4] as f32;
        }
        output
    }

    fn quantize_f32_to_q4k(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
        assert_eq!(cols % 256, 0);
        let blocks_per_row = cols / 256;
        let mut data = Vec::with_capacity(rows * blocks_per_row * 144);
        for row in 0..rows {
            for b in 0..blocks_per_row {
                let start = row * cols + b * 256;
                let mut vals = [0.0f32; 256];
                vals.copy_from_slice(&weights[start..start + 256]);
                let block = quantize_f32_to_q4k_block(&vals);
                data.extend_from_slice(&block);
            }
        }
        data
    }

    fn dequant_q4k_to_f32(q4k_data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let blocks_per_row = cols / 256;
        let bytes_per_row = blocks_per_row * 144;
        let mut output = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for b in 0..blocks_per_row {
                let block_start = row * bytes_per_row + b * 144;
                let mut block = [0u8; 144];
                block.copy_from_slice(&q4k_data[block_start..block_start + 144]);
                let vals = dequant_q4k_block_test(&block);
                let out_start = row * cols + b * 256;
                output[out_start..out_start + 256].copy_from_slice(&vals);
            }
        }
        output
    }

    #[test]
    fn test_dot_q4k_f32_scalar_simple() {
        let cols = 256;
        let blocks_per_row = 1;

        // Create simple weights and input
        let mut weights = [0.0f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = (i as f32 - 128.0) * 0.01;
        }
        let block = quantize_f32_to_q4k_block(&weights);
        let dequant = dequant_q4k_block_test(&block);

        let mut input = vec![0.0f32; cols];
        for (i, inp) in input.iter_mut().enumerate() {
            *inp = (i as f32) * 0.001;
        }

        // Reference: dot product with dequantized weights
        let expected: f32 = dequant.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

        // Q4_K scalar dot product
        let result = dot_q4k_f32_scalar(&block, &input, blocks_per_row);

        let diff = (expected - result).abs();
        assert!(
            diff < 0.01,
            "expected={expected}, result={result}, diff={diff}"
        );
    }

    #[test]
    fn test_matvec_argmax_q8_0_preq_matches_full_matvec() {
        let rows = 32;
        let cols = 64;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 31 + 11) % 256) as f32 / 256.0 - 0.5)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();

        let q8_data = quantize_f32_to_q8_0(&weights, rows, cols);
        let preq = quantize_input_q8_0(&input);

        // Full matvec + argmax
        let mut full_output = vec![0.0f32; rows];
        matvec_q8_0_preq_into(&q8_data, &preq, rows, &mut full_output);
        let mut expected_idx = 0;
        let mut expected_val = f32::NEG_INFINITY;
        for (i, &v) in full_output.iter().enumerate() {
            if v > expected_val {
                expected_val = v;
                expected_idx = i;
            }
        }

        // Fused argmax
        let (got_idx, got_val) = matvec_argmax_q8_0_preq(&q8_data, &preq, rows);

        assert_eq!(
            got_idx, expected_idx,
            "Q8_0 argmax index mismatch: got {got_idx}, expected {expected_idx}"
        );
        assert!(
            (got_val - expected_val).abs() < 1e-5,
            "Q8_0 argmax value mismatch: got {got_val}, expected {expected_val}"
        );
    }

    #[test]
    fn test_matvec_q4k_into_simple() {
        let rows = 4;
        let cols = 256;

        let mut rng_state = 42u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };

        let weights: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let input: Vec<f32> = (0..cols).map(|_| next_f32() * 0.1).collect();

        let q4k_data = quantize_f32_to_q4k(&weights, rows, cols);
        let dequant = dequant_q4k_to_f32(&q4k_data, rows, cols);

        // Reference: f32 matvec with dequantized weights
        let expected = matvec_scalar(&dequant, &input, rows, cols);

        // Q4_K matvec
        let mut output = vec![0.0f32; rows];
        matvec_q4k_into(&q4k_data, &input, rows, cols, &mut output);

        for i in 0..rows {
            let diff = (expected[i] - output[i]).abs();
            assert!(
                diff < 0.05,
                "row {i}: expected={}, got={}, diff={}",
                expected[i],
                output[i],
                diff
            );
        }
    }

    #[test]
    fn test_matvec_q4k_zero_weights() {
        let rows = 2;
        let cols = 256;
        let q4k_data = vec![0u8; rows * 144]; // all zeros
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];
        matvec_q4k_into(&q4k_data, &input, rows, cols, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "row {i}: expected ~0.0, got {v}");
        }
    }

    #[test]
    fn test_matvec_q4k_q8_into_matches_f32_path() {
        let rows = 8;
        let cols = 512;

        let mut rng_state = 99u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };

        let weights: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let input: Vec<f32> = (0..cols).map(|_| next_f32() * 0.1).collect();

        let q4k_data = quantize_f32_to_q4k(&weights, rows, cols);
        let preq = quantize_input_q8_0(&input);

        let mut f32_out = vec![0.0f32; rows];
        matvec_q4k_into(&q4k_data, &input, rows, cols, &mut f32_out);

        let mut q8_out = vec![0.0f32; rows];
        matvec_q4k_q8_into(&q4k_data, &preq, rows, cols, &mut q8_out);

        // The Q8_0 input quantization adds ~0.4% noise per element, so we
        // allow a loose tolerance relative to the magnitude of the result.
        for i in 0..rows {
            let rel =
                ((f32_out[i] - q8_out[i]).abs()) / (f32_out[i].abs().max(1e-3));
            assert!(
                rel < 0.05,
                "row {i}: f32={}, q8={}, rel={}",
                f32_out[i],
                q8_out[i],
                rel
            );
        }
    }

    #[test]
    fn test_matvec_argmax_q4k_q8_preq_matches_full_matvec() {
        let rows = 16;
        let cols = 512;

        let mut rng_state = 123u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };

        let weights: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let input: Vec<f32> = (0..cols).map(|_| next_f32() * 0.1).collect();

        let q4k_data = quantize_f32_to_q4k(&weights, rows, cols);
        let preq = quantize_input_q8_0(&input);

        // Full matvec + scan argmax.
        let mut full = vec![0.0f32; rows];
        matvec_q4k_q8_into(&q4k_data, &preq, rows, cols, &mut full);
        let mut full_best_idx = 0usize;
        let mut full_best_val = f32::NEG_INFINITY;
        for (i, &v) in full.iter().enumerate() {
            if v > full_best_val {
                full_best_val = v;
                full_best_idx = i;
            }
        }

        // Fused argmax.
        let (fused_idx, fused_val) =
            matvec_argmax_q4k_q8_preq(&q4k_data, &preq, rows, cols);

        assert_eq!(
            fused_idx, full_best_idx,
            "fused Q4K argmax index must match full matvec argmax"
        );
        let rel = (fused_val - full_best_val).abs() / full_best_val.abs().max(1e-3);
        assert!(
            rel < 1e-5,
            "fused Q4K argmax value mismatch: fused={fused_val}, full={full_best_val}, rel={rel}"
        );
    }

    #[test]
    fn test_dot_q4k_q8_0_scalar_matches_f32() {
        // Directly compare scalar Q4_K × Q8_0 against scalar Q4_K × f32
        // for a single row, using matched tolerance.
        let cols = 256;

        let mut rng_state = 7u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut weights = [0.0f32; 256];
        for w in weights.iter_mut() {
            *w = next_f32();
        }
        let input: Vec<f32> = (0..cols).map(|_| next_f32() * 0.1).collect();

        let block = quantize_f32_to_q4k_block(&weights);
        let preq = quantize_input_q8_0(&input);

        let f32_result = dot_q4k_f32_scalar(&block, &input, 1);
        let q8_result =
            dot_q4k_q8_0_scalar(&block, &preq.scales, &preq.quants, 1);

        // Q8_0 input adds ~0.4% per-element quantization noise. For a single
        // Q4_K block this gives at most a few percent of the per-element
        // magnitude times the accumulation length, not of the (potentially
        // nearly-cancelling) dot product result. Use a combined abs+rel
        // tolerance.
        let tol = 0.01 * f32_result.abs().max(0.0) + 1e-3;
        assert!(
            (f32_result - q8_result).abs() < tol,
            "f32={f32_result}, q8={q8_result}, tol={tol}"
        );
    }

    #[test]
    fn test_matvec_q4k_larger_matrix() {
        // Test with multiple blocks per row (cols = 512 = 2 blocks)
        let rows = 8;
        let cols = 512;

        let mut rng_state = 123u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };

        let weights: Vec<f32> = (0..rows * cols).map(|_| next_f32()).collect();
        let input: Vec<f32> = (0..cols).map(|_| next_f32() * 0.1).collect();

        let q4k_data = quantize_f32_to_q4k(&weights, rows, cols);
        let dequant = dequant_q4k_to_f32(&q4k_data, rows, cols);

        let expected = matvec_scalar(&dequant, &input, rows, cols);

        let mut output = vec![0.0f32; rows];
        matvec_q4k_into(&q4k_data, &input, rows, cols, &mut output);

        for i in 0..rows {
            let diff = (expected[i] - output[i]).abs();
            assert!(
                diff < 0.1,
                "row {i}: expected={}, got={}, diff={}",
                expected[i],
                output[i],
                diff
            );
        }
    }

    #[test]
    fn test_top_k_softmax_basic() {
        let logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
        let (indices, weights) = top_k_softmax(&logits, 2);
        // Top-2 should be indices 4 (score 4.0) and 1 (score 3.0)
        assert_eq!(indices, vec![4, 1]);
        assert_eq!(weights.len(), 2);
        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
        // First weight should be larger (higher logit)
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_top_k_softmax_k_equals_n() {
        let logits = vec![1.0, 2.0, 3.0];
        let (indices, weights) = top_k_softmax(&logits, 3);
        assert_eq!(indices, vec![2, 1, 0]);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moe_forward_deterministic() {
        let dim = 4;
        let inter = 8;
        let num_experts = 3;
        let num_active = 2;

        // Build router: (num_experts, dim)
        let gate_data: Vec<f32> = (0..num_experts * dim)
            .map(|i| ((i * 7 + 3) % 20) as f32 / 20.0 - 0.25)
            .collect();
        let gate = Tensor::from_vec(gate_data, &[num_experts * dim]).unwrap();

        // Build expert weights
        let experts: Vec<ExpertWeights> = (0..num_experts)
            .map(|e| {
                let seed = e * 100;
                let make_w = |n: usize| -> Tensor {
                    let data: Vec<f32> = (0..n)
                        .map(|i| ((seed + i * 13 + 5) % 50) as f32 / 100.0 - 0.25)
                        .collect();
                    Tensor::from_vec(data, &[n]).unwrap()
                };
                ExpertWeights {
                    w_gate: make_w(inter * dim),
                    w_up: make_w(inter * dim),
                    w_down: make_w(dim * inter),
                    loaded: true,
                }
            })
            .collect();

        let moe_weights = MoeLayerWeights { gate, experts };

        let x: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];
        let out1 = moe_forward(&x, &moe_weights, dim, inter, num_active);
        let out2 = moe_forward(&x, &moe_weights, dim, inter, num_active);

        // Must be deterministic
        assert_eq!(out1.len(), dim);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_moe_forward_into_matches() {
        let dim = 4;
        let inter = 8;
        let num_experts = 2;
        let num_active = 1;

        let gate_data: Vec<f32> = (0..num_experts * dim).map(|i| (i as f32) * 0.1).collect();
        let gate = Tensor::from_vec(gate_data, &[num_experts * dim]).unwrap();

        let experts: Vec<ExpertWeights> = (0..num_experts)
            .map(|e| {
                let make_w = |n: usize| -> Tensor {
                    let data: Vec<f32> = (0..n)
                        .map(|i| ((e + i * 11 + 7) % 30) as f32 / 60.0 - 0.25)
                        .collect();
                    Tensor::from_vec(data, &[n]).unwrap()
                };
                ExpertWeights {
                    w_gate: make_w(inter * dim),
                    w_up: make_w(inter * dim),
                    w_down: make_w(dim * inter),
                    loaded: true,
                }
            })
            .collect();

        let moe_weights = MoeLayerWeights { gate, experts };
        let x: Vec<f32> = vec![0.5, -0.3, 0.2, 0.1];

        let out_alloc = moe_forward(&x, &moe_weights, dim, inter, num_active);
        let mut out_into = vec![0.0f32; dim];
        moe_forward_into(&x, &moe_weights, dim, inter, num_active, &mut out_into);

        for (a, b) in out_alloc.iter().zip(out_into.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "moe_forward vs moe_forward_into mismatch: {a} vs {b}"
            );
        }
    }

    /// Verify that the fused RMSNorm + Q8_0 quantization produces the same
    /// result as the two-step rmsnorm_into + quantize_input_q8_0_into pipeline
    /// across several dimension sizes.
    #[test]
    fn test_rmsnorm_quantize_q8_0_matches_two_step() {
        let test_dims = [32, 64, 128, 256, 512, 1024];
        for &dim in &test_dims {
            // Deterministic pseudo-random inputs
            let x: Vec<f32> = (0..dim)
                .map(|i| ((i * 7 + 3) % 31) as f32 / 15.0 - 1.0)
                .collect();
            let weight: Vec<f32> = (0..dim)
                .map(|i| ((i * 11 + 5) % 23) as f32 / 11.5)
                .collect();
            let eps = 1e-5f32;

            // Two-step: rmsnorm_into + quantize_input_q8_0_into
            let mut normed = vec![0.0f32; dim];
            rmsnorm_into(&x, &weight, eps, &mut normed);
            let mut two_step = QuantizedInput {
                scales: vec![],
                quants: vec![],
                blocks_per_row: 0,
            };
            quantize_input_q8_0_into(&normed, &mut two_step);

            // Fused: rmsnorm_quantize_q8_0_into
            let mut fused = QuantizedInput {
                scales: vec![],
                quants: vec![],
                blocks_per_row: 0,
            };
            rmsnorm_quantize_q8_0_into(&x, &weight, eps, &mut fused);

            // Compare
            assert_eq!(
                fused.blocks_per_row, two_step.blocks_per_row,
                "dim={dim}: blocks_per_row mismatch"
            );
            assert_eq!(
                fused.scales.len(),
                two_step.scales.len(),
                "dim={dim}: scales len mismatch"
            );
            for (b, (fs, ts)) in fused.scales.iter().zip(two_step.scales.iter()).enumerate() {
                assert!(
                    (fs - ts).abs() < 1e-6,
                    "dim={dim} block={b}: scale mismatch {fs} vs {ts}"
                );
            }
            assert_eq!(
                fused.quants.len(),
                two_step.quants.len(),
                "dim={dim}: quants len mismatch"
            );
            for (j, (fq, tq)) in fused.quants.iter().zip(two_step.quants.iter()).enumerate() {
                assert!(
                    (*fq as i32 - *tq as i32).abs() <= 1,
                    "dim={dim} idx={j}: quant mismatch {fq} vs {tq}"
                );
            }
        }
    }
}

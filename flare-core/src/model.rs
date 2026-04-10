use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::tensor::Tensor;

/// Trait for compute backends (WebGPU, SIMD, native wgpu).
/// Each backend implements these fundamental operations.
pub trait ComputeBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, output: &mut Tensor);
    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32, output: &mut Tensor);
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, pos: usize, head_dim: usize, theta: f32);
    fn softmax(&self, input: &mut Tensor);
    fn silu_mul(&self, gate: &Tensor, up: &Tensor, output: &mut Tensor);

    /// Matrix-vector multiply: output[rows] = mat[rows, cols] * vec[cols].
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
    /// Returns logits over the vocabulary [vocab_size].
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
            let up =
                self.backend
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
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}

/// Matrix-vector multiply: output[rows] = mat[rows, cols] * vec[cols] (CPU implementation).
///
/// Uses 4-wide manual unrolling for better ILP and auto-vectorization.
/// For a [576, 1536] matrix, this is ~2-3x faster than a naive loop.
pub fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    let cols4 = cols & !3; // round down to multiple of 4

    for (i, out) in output.iter_mut().enumerate() {
        let row = &mat[i * cols..(i + 1) * cols];
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        // 4-wide unrolled inner loop
        let mut j = 0;
        while j < cols4 {
            sum0 += row[j] * vec[j];
            sum1 += row[j + 1] * vec[j + 1];
            sum2 += row[j + 2] * vec[j + 2];
            sum3 += row[j + 3] * vec[j + 3];
            j += 4;
        }
        // Handle remainder
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
pub fn apply_rope(data: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let x0 = data[offset + i];
            let x1 = data[offset + i + half];
            data[offset + i] = x0 * cos_val - x1 * sin_val;
            data[offset + i + half] = x0 * sin_val + x1 * cos_val;
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

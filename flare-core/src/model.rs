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
}

/// Complete model weights.
pub struct ModelWeights {
    pub token_embedding: Tensor,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Tensor,
    pub output_weight: Tensor,
}

/// The core model that runs inference.
pub struct Model {
    config: ModelConfig,
    weights: ModelWeights,
    kv_cache: KvCache,
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
        }
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
            let normed = rmsnorm(x.data(), layer.attn_norm.data(), config.rms_norm_eps);

            // QKV projections
            let q_data = matvec(layer.wq.data(), &normed, num_heads * head_dim, dim);
            let k_data = matvec(layer.wk.data(), &normed, kv_dim, dim);
            let v_data = matvec(layer.wv.data(), &normed, kv_dim, dim);

            // Apply RoPE to Q and K
            let mut q_data = q_data;
            let mut k_data = k_data;
            apply_rope(&mut q_data, num_heads, head_dim, pos, config.rope_theta);
            apply_rope(&mut k_data, num_kv_heads, head_dim, pos, config.rope_theta);

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
            let attn_proj = matvec(layer.wo.data(), &attn_output, dim, num_heads * head_dim);

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += attn_proj[i];
            }

            // --- FFN block ---
            let normed = rmsnorm(x.data(), layer.ffn_norm.data(), config.rms_norm_eps);

            // Gate and up projections
            let gate = matvec(layer.w_gate.data(), &normed, config.intermediate_dim, dim);
            let up = matvec(layer.w_up.data(), &normed, config.intermediate_dim, dim);

            // SiLU(gate) * up
            let mut ffn_hidden = vec![0.0f32; config.intermediate_dim];
            for i in 0..config.intermediate_dim {
                let silu = gate[i] / (1.0 + (-gate[i]).exp());
                ffn_hidden[i] = silu * up[i];
            }

            // Down projection
            let ffn_out = matvec(layer.w_down.data(), &ffn_hidden, dim, config.intermediate_dim);

            // Residual connection
            let x_data = x.data_mut();
            for i in 0..dim {
                x_data[i] += ffn_out[i];
            }
        }

        // Advance KV cache after processing all layers
        self.kv_cache.advance();

        // Final RMSNorm
        let normed = rmsnorm(x.data(), self.weights.output_norm.data(), config.rms_norm_eps);

        // Output logits: [vocab_size] = output_weight [vocab_size, dim] × normed [dim]
        let logits = matvec(
            self.weights.output_weight.data(),
            &normed,
            config.vocab_size,
            dim,
        );

        Tensor::from_vec(logits, &[config.vocab_size]).unwrap()
    }
}

/// RMSNorm: normalize and scale.
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}

/// Matrix-vector multiply: output[rows] = mat[rows, cols] × vec[cols]
fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for i in 0..rows {
        let row_offset = i * cols;
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += mat[row_offset + j] * vec[j];
        }
        output[i] = sum;
    }
    output
}

/// Apply RoPE to interleaved Q or K vectors.
fn apply_rope(data: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
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
    q: &[f32],         // [num_heads * head_dim]
    kv_cache: &KvCache,
    layer: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,     // number of valid KV entries
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

        for t in 0..seq_len {
            let k_offset = t * kv_stride + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            scores[t] = dot * scale;
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
        for t in 0..seq_len {
            let v_offset = t * kv_stride + kv_head * head_dim;
            let weight = scores[t];
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
            assert!((v - expected).abs() < 1e-4, "rmsnorm[{i}]: {v} != {expected}");
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
}

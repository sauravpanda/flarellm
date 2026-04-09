use flare_core::model::ComputeBackend;
use flare_core::tensor::Tensor;

/// WASM SIMD128 CPU fallback backend.
/// Used when WebGPU is not available.
pub struct SimdBackend;

impl SimdBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimdBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for SimdBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, output: &mut Tensor) {
        crate::matmul::matmul_cpu(a, b, output);
    }

    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32, output: &mut Tensor) {
        let data = input.data();
        let w = weight.data();
        let out = output.data_mut();
        let dim = w.len();

        // Process each row
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
        let half = head_dim / 2;

        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let q_data = q.data_mut();
            let q0 = q_data[i];
            let q1 = q_data[i + half];
            q_data[i] = q0 * cos_val - q1 * sin_val;
            q_data[i + half] = q0 * sin_val + q1 * cos_val;

            let k_data = k.data_mut();
            let k0 = k_data[i];
            let k1 = k_data[i + half];
            k_data[i] = k0 * cos_val - k1 * sin_val;
            k_data[i + half] = k0 * sin_val + k1 * cos_val;
        }
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
        let g = gate.data();
        let u = up.data();
        let o = output.data_mut();
        for i in 0..g.len() {
            let silu = g[i] / (1.0 + (-g[i]).exp());
            o[i] = silu * u[i];
        }
    }
}

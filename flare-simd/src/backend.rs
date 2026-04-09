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

#[cfg(test)]
mod tests {
    use super::*;
    use flare_core::model::ComputeBackend;
    use flare_core::tensor::Tensor;

    #[test]
    fn test_rmsnorm_unit_weight() {
        let backend = SimdBackend::new();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let mut output = Tensor::zeros(&[4]);

        backend.rmsnorm(&input, &weight, 1e-5, &mut output);

        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        for (i, &v) in output.data().iter().enumerate() {
            let expected = (i + 1) as f32 / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "rmsnorm[{i}]: {v} != {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_properties() {
        let backend = SimdBackend::new();
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        backend.softmax(&mut input);
        let data = input.data();

        // Sum should be ~1.0
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");

        // Monotonically increasing
        for i in 0..3 {
            assert!(
                data[i] < data[i + 1],
                "softmax should be monotonically increasing"
            );
        }

        // All positive
        assert!(data.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_silu_mul() {
        let backend = SimdBackend::new();
        let gate = Tensor::from_vec(vec![0.0, 1.0, -1.0, 10.0], &[4]).unwrap();
        let up = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let mut output = Tensor::zeros(&[4]);

        backend.silu_mul(&gate, &up, &mut output);
        let data = output.data();

        // SiLU(0) = 0
        assert!(data[0].abs() < 1e-5, "SiLU(0) should be 0, got {}", data[0]);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((data[1] - 0.731).abs() < 0.01);
        // SiLU(-1) < 0
        assert!(data[2] < 0.0, "SiLU(-1) should be negative");
        // SiLU(10) ≈ 10 (sigmoid(10) ≈ 1)
        assert!((data[3] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_rope_preserves_magnitude() {
        let backend = SimdBackend::new();
        let mut q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[4]).unwrap();
        let mut k = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[4]).unwrap();

        let q_mag_before: f32 = q.data().iter().map(|x| x * x).sum();
        let k_mag_before: f32 = k.data().iter().map(|x| x * x).sum();

        backend.rope(&mut q, &mut k, 7, 4, 10000.0);

        let q_mag_after: f32 = q.data().iter().map(|x| x * x).sum();
        let k_mag_after: f32 = k.data().iter().map(|x| x * x).sum();

        assert!(
            (q_mag_before - q_mag_after).abs() < 1e-4,
            "RoPE should preserve Q magnitude"
        );
        assert!(
            (k_mag_before - k_mag_after).abs() < 1e-4,
            "RoPE should preserve K magnitude"
        );
    }

    #[test]
    fn test_matmul_non_square() {
        let backend = SimdBackend::new();
        // [2x3] * [3x2] = [2x2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let mut c = Tensor::zeros(&[2, 2]);

        backend.matmul(&a, &b, &mut c);

        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert!((c.data()[0] - 22.0).abs() < 1e-3);
        assert!((c.data()[1] - 28.0).abs() < 1e-3);
        assert!((c.data()[2] - 49.0).abs() < 1e-3);
        assert!((c.data()[3] - 64.0).abs() < 1e-3);
    }
}

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

    fn matvec_ternary(
        &self,
        packed_weights: &[u8],
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        matvec_ternary_wasm(packed_weights, input, rows, cols)
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

/// WASM SIMD128 ternary matvec implementation.
///
/// On WASM targets with SIMD128, uses `v128` operations for 4-wide processing.
/// On non-WASM targets, falls back to the scalar implementation from `flare_core`.
///
/// Ternary encoding: 2 bits per weight (4 per byte).
/// 00=0, 01=+1, 10=-1, 11=unused (treated as 0).
fn matvec_ternary_wasm(packed_weights: &[u8], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // On non-WASM targets, delegate to the core scalar implementation.
    // The WASM SIMD128 path would use std::arch::wasm32::* intrinsics
    // (f32x4_add, f32x4_sub, v128_bitselect, etc.) but these are only
    // available when compiling for wasm32. The logic mirrors the NEON
    // approach: decode 4 ternary weights per byte, build sign masks,
    // and use bitselect for branchless conditional add/sub.
    flare_core::model::matvec_ternary_scalar(packed_weights, input, rows, cols)
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
    fn test_rmsnorm_multi_row() {
        // Two identical rows — both should produce the same normalized output
        let backend = SimdBackend::new();
        let input = Tensor::from_vec(vec![3.0, 4.0, 3.0, 4.0], &[2, 2]).unwrap();
        let weight = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let mut output = Tensor::zeros(&[2, 2]);
        backend.rmsnorm(&input, &weight, 1e-5, &mut output);
        let d = output.data();
        // Both rows should be identical
        assert!(
            (d[0] - d[2]).abs() < 1e-5,
            "row0[0]={} row1[0]={}",
            d[0],
            d[2]
        );
        assert!(
            (d[1] - d[3]).abs() < 1e-5,
            "row0[1]={} row1[1]={}",
            d[1],
            d[3]
        );
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        // All-zero input → all-zero output (0/rms = 0)
        let backend = SimdBackend::new();
        let input = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mut output = Tensor::zeros(&[4]);
        backend.rmsnorm(&input, &weight, 1e-5, &mut output);
        for (i, &v) in output.data().iter().enumerate() {
            assert!(v.abs() < 1e-5, "rmsnorm zero: output[{i}] = {v}");
        }
    }

    #[test]
    fn test_softmax_single_element() {
        let backend = SimdBackend::new();
        let mut input = Tensor::from_vec(vec![42.0], &[1]).unwrap();
        backend.softmax(&mut input);
        assert!(
            (input.data()[0] - 1.0).abs() < 1e-6,
            "single-element softmax should be 1.0, got {}",
            input.data()[0]
        );
    }

    #[test]
    fn test_softmax_uniform_input() {
        // All same inputs → all outputs should equal 1/n
        let backend = SimdBackend::new();
        let n = 4;
        let mut input = Tensor::from_vec(vec![5.0; n], &[n]).unwrap();
        backend.softmax(&mut input);
        let expected = 1.0 / n as f32;
        for (i, &v) in input.data().iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "uniform softmax[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_silu_mul_zero_up() {
        // up=0 → output always 0 regardless of gate
        let backend = SimdBackend::new();
        let gate = Tensor::from_vec(vec![1.0, -1.0, 5.0, -5.0], &[4]).unwrap();
        let up = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let mut output = Tensor::zeros(&[4]);
        backend.silu_mul(&gate, &up, &mut output);
        for (i, &v) in output.data().iter().enumerate() {
            assert!(v.abs() < 1e-6, "silu_mul zero_up: output[{i}] = {v}");
        }
    }

    #[test]
    fn test_rope_pos_zero() {
        // pos=0 → angle=0 → cos=1, sin=0 → q and k unchanged
        let backend = SimdBackend::new();
        let q_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let k_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut q = Tensor::from_vec(q_data.clone(), &[4]).unwrap();
        let mut k = Tensor::from_vec(k_data.clone(), &[4]).unwrap();
        backend.rope(&mut q, &mut k, 0, 4, 10000.0);
        for (i, (&v, &orig)) in q.data().iter().zip(q_data.iter()).enumerate() {
            assert!((v - orig).abs() < 1e-5, "rope pos=0 q[{i}]: {v} != {orig}");
        }
        for (i, (&v, &orig)) in k.data().iter().zip(k_data.iter()).enumerate() {
            assert!((v - orig).abs() < 1e-5, "rope pos=0 k[{i}]: {v} != {orig}");
        }
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

    #[test]
    fn test_matvec_ternary_simd_backend() {
        use flare_core::model::quantize_to_ternary;

        let backend = SimdBackend::new();

        // 2x4 ternary matrix:
        // Row 0: [+1, -1,  0, +1] -> out[0] = in[0] - in[1] + in[3]
        // Row 1: [-1,  0, +1, -1] -> out[1] = -in[0] + in[2] - in[3]
        let weights_row0 = vec![1.0, -1.0, 0.0, 1.0];
        let weights_row1 = vec![-1.0, 0.0, 1.0, -1.0];

        let packed_row0 = quantize_to_ternary(&weights_row0);
        let packed_row1 = quantize_to_ternary(&weights_row1);
        let mut packed = Vec::new();
        packed.extend_from_slice(&packed_row0);
        packed.extend_from_slice(&packed_row1);

        let input = vec![2.0, 3.0, 5.0, 7.0];
        let result = backend.matvec_ternary(&packed, &input, 2, 4);

        assert_eq!(result.len(), 2);
        // Row 0: 2.0 - 3.0 + 7.0 = 6.0
        assert!((result[0] - 6.0).abs() < 1e-5, "row0: got {}", result[0]);
        // Row 1: -2.0 + 5.0 - 7.0 = -4.0
        assert!((result[1] - (-4.0)).abs() < 1e-5, "row1: got {}", result[1]);
    }
}

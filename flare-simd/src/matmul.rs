use flare_core::tensor::Tensor;

/// CPU matrix multiply: C = A × B
/// A is [M, K], B is [K, N], C is [M, N]
///
/// Uses a simple tiled approach for cache locality.
/// On WASM targets with SIMD, the compiler auto-vectorizes the inner loop.
pub fn matmul_cpu(a: &Tensor, b: &Tensor, output: &mut Tensor) {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert!(
        a_shape.len() == 2 && b_shape.len() == 2,
        "matmul requires 2D tensors"
    );
    assert_eq!(a_shape[1], b_shape[0], "inner dimensions must match");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let a_data = a.data();
    let b_data = b.data();
    let c_data = output.data_mut();

    // Zero output
    for v in c_data.iter_mut() {
        *v = 0.0;
    }

    // Tiled multiply for cache locality
    const TILE: usize = 32;

    for i0 in (0..m).step_by(TILE) {
        for j0 in (0..n).step_by(TILE) {
            for k0 in (0..k).step_by(TILE) {
                let i_end = (i0 + TILE).min(m);
                let j_end = (j0 + TILE).min(n);
                let k_end = (k0 + TILE).min(k);

                for i in i0..i_end {
                    for kk in k0..k_end {
                        let a_val = a_data[i * k + kk];
                        for j in j0..j_end {
                            c_data[i * n + j] += a_val * b_data[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_core::tensor::Tensor;

    #[test]
    fn test_matmul_identity() {
        // A = [[1,0],[0,1]], B = [[3,4],[5,6]]
        let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]).unwrap();
        let mut c = Tensor::zeros(&[2, 2]);
        matmul_cpu(&a, &b, &mut c);
        assert_eq!(c.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matmul_basic() {
        // [1,2,3] x [4,5,6,7,8,9] (1x3 * 3x2 = 1x2)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2]).unwrap();
        let mut c = Tensor::zeros(&[1, 2]);
        matmul_cpu(&a, &b, &mut c);
        // [1*4+2*6+3*8, 1*5+2*7+3*9] = [40, 46]
        assert_eq!(c.data(), &[40.0, 46.0]);
    }

    #[test]
    fn test_matmul_1x1() {
        // 1x1 * 1x1 = 1x1: scalar multiplication
        let a = Tensor::from_vec(vec![3.0], &[1, 1]).unwrap();
        let b = Tensor::from_vec(vec![7.0], &[1, 1]).unwrap();
        let mut c = Tensor::zeros(&[1, 1]);
        matmul_cpu(&a, &b, &mut c);
        assert!(
            (c.data()[0] - 21.0).abs() < 1e-5,
            "1x1: got {}",
            c.data()[0]
        );
    }

    #[test]
    fn test_matmul_zero_a() {
        // All-zero A → output all zeros regardless of B
        let a = Tensor::from_vec(vec![0.0; 6], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let mut c = Tensor::from_vec(vec![99.0; 4], &[2, 2]).unwrap();
        matmul_cpu(&a, &b, &mut c);
        for (i, &v) in c.data().iter().enumerate() {
            assert!(v.abs() < 1e-5, "zero_a: c[{i}] = {v}");
        }
    }

    #[test]
    fn test_matmul_zero_b() {
        // All-zero B → output all zeros regardless of A
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![0.0; 6], &[3, 2]).unwrap();
        let mut c = Tensor::from_vec(vec![99.0; 4], &[2, 2]).unwrap();
        matmul_cpu(&a, &b, &mut c);
        for (i, &v) in c.data().iter().enumerate() {
            assert!(v.abs() < 1e-5, "zero_b: c[{i}] = {v}");
        }
    }

    #[test]
    fn test_matmul_negative_values() {
        // A=[[1,-1]], B=[[-1],[1]] → C=[[1*(-1)+(-1)*1]] = [[-2]]
        let a = Tensor::from_vec(vec![1.0, -1.0], &[1, 2]).unwrap();
        let b = Tensor::from_vec(vec![-1.0, 1.0], &[2, 1]).unwrap();
        let mut c = Tensor::zeros(&[1, 1]);
        matmul_cpu(&a, &b, &mut c);
        assert!(
            (c.data()[0] - (-2.0)).abs() < 1e-5,
            "neg: got {}",
            c.data()[0]
        );
    }

    #[test]
    fn test_matmul_larger() {
        // 4x3 * 3x4
        let a = Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[4, 3]).unwrap();
        let b = Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4]).unwrap();
        let mut c = Tensor::zeros(&[4, 4]);
        matmul_cpu(&a, &b, &mut c);

        // Verify dimensions
        assert_eq!(c.shape(), &[4, 4]);

        // Spot check: C[0,0] = 0*0 + 1*4 + 2*8 = 20
        assert!((c.data()[0] - 20.0).abs() < 1e-5);
    }
}

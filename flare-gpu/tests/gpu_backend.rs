//! Integration tests for the GPU compute backend.
//!
//! These tests require a GPU adapter. They are ignored by default
//! and run with `cargo test -- --ignored` or on CI with GPU support.

use flare_core::model::ComputeBackend;
use flare_gpu::WebGpuBackend;

fn try_create_gpu() -> Option<WebGpuBackend> {
    pollster::block_on(WebGpuBackend::new()).ok()
}

#[test]
#[ignore] // requires GPU
fn test_gpu_matvec_identity() {
    let gpu = match try_create_gpu() {
        Some(g) => g,
        None => {
            eprintln!("No GPU adapter found, skipping");
            return;
        }
    };

    // Identity-like: [[1,0,0],[0,1,0],[0,0,1]] * [3,5,7] = [3,5,7]
    let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let vec_data = vec![3.0, 5.0, 7.0];
    let result = gpu.matvec(&mat, &vec_data, 3, 3);

    assert_eq!(result.len(), 3);
    assert!((result[0] - 3.0).abs() < 1e-3, "got {}", result[0]);
    assert!((result[1] - 5.0).abs() < 1e-3, "got {}", result[1]);
    assert!((result[2] - 7.0).abs() < 1e-3, "got {}", result[2]);
}

#[test]
#[ignore] // requires GPU
fn test_gpu_matvec_basic() {
    let gpu = match try_create_gpu() {
        Some(g) => g,
        None => return,
    };

    // [[1,2],[3,4]] * [1,1] = [3, 7]
    let mat = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![1.0, 1.0];
    let result = gpu.matvec(&mat, &v, 2, 2);

    assert_eq!(result.len(), 2);
    assert!((result[0] - 3.0).abs() < 1e-3, "got {}", result[0]);
    assert!((result[1] - 7.0).abs() < 1e-3, "got {}", result[1]);
}

#[test]
#[ignore] // requires GPU
fn test_gpu_matvec_large() {
    let gpu = match try_create_gpu() {
        Some(g) => g,
        None => return,
    };

    // 256x512 matrix * 512 vector — realistic size
    let rows = 256;
    let cols = 512;
    let mat: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let v: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();

    let gpu_result = gpu.matvec(&mat, &v, rows, cols);
    let cpu_result = flare_core::model::matvec(&mat, &v, rows, cols);

    assert_eq!(gpu_result.len(), rows);
    for i in 0..rows {
        assert!(
            (gpu_result[i] - cpu_result[i]).abs() < 0.1,
            "row {i}: gpu={} cpu={}",
            gpu_result[i],
            cpu_result[i]
        );
    }
}

#[test]
#[ignore] // requires GPU
fn test_gpu_silu_mul_vec() {
    let gpu = match try_create_gpu() {
        Some(g) => g,
        None => return,
    };

    let gate = vec![0.0, 1.0, -1.0, 10.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let result = gpu.silu_mul_vec(&gate, &up);
    let cpu = flare_core::model::silu_mul_cpu(&gate, &up);

    assert_eq!(result.len(), 4);
    for i in 0..4 {
        assert!(
            (result[i] - cpu[i]).abs() < 1e-3,
            "silu_mul[{i}]: gpu={} cpu={}",
            result[i],
            cpu[i]
        );
    }
}

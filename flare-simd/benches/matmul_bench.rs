//! Benchmarks for matrix multiplication and core operations.
//!
//! Run with: cargo bench -p flare-simd

use std::time::Instant;

use flare_core::tensor::Tensor;
use flare_simd::matmul::matmul_cpu;

fn bench_matmul(m: usize, k: usize, n: usize, iterations: usize) {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect();

    let a = Tensor::from_vec(a_data, &[m, k]).unwrap();
    let b = Tensor::from_vec(b_data, &[k, n]).unwrap();
    let mut c = Tensor::zeros(&[m, n]);

    // Warmup
    matmul_cpu(&a, &b, &mut c);

    let start = Instant::now();
    for _ in 0..iterations {
        matmul_cpu(&a, &b, &mut c);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_iter = elapsed / iterations as f64;

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / per_iter / 1e9;

    println!(
        "  matmul [{m}x{k}] x [{k}x{n}]: {:.3}ms ({:.2} GFLOPS)",
        per_iter * 1000.0,
        gflops
    );
}

fn bench_sampling() {
    use flare_core::sampling::*;

    let vocab_size = 128256;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.01).sin()).collect();

    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = sample_greedy(&logits);
    }
    let per_iter = start.elapsed().as_secs_f64() / iterations as f64;
    println!("  greedy (vocab {vocab_size}): {:.1}us", per_iter * 1e6);

    let iterations = 1000;
    let start = Instant::now();
    for i in 0..iterations {
        let _ = sample_top_p(&logits, 0.9, (i as f32) / iterations as f32);
    }
    let per_iter = start.elapsed().as_secs_f64() / iterations as f64;
    println!("  top_p  (vocab {vocab_size}): {:.1}us", per_iter * 1e6);

    let start = Instant::now();
    for i in 0..iterations {
        let _ = sample_top_k(&logits, 40, (i as f32) / iterations as f32);
    }
    let per_iter = start.elapsed().as_secs_f64() / iterations as f64;
    println!("  top_k  (vocab {vocab_size}): {:.1}us", per_iter * 1e6);
}

fn main() {
    println!("=== Flare CPU Benchmarks ===\n");

    println!("--- MatMul ---");
    bench_matmul(1, 2048, 2048, 100);
    bench_matmul(1, 4096, 4096, 50);
    bench_matmul(1, 2048, 8192, 50);
    bench_matmul(32, 2048, 2048, 10);

    println!("\n--- Sampling ---");
    bench_sampling();

    println!();
}

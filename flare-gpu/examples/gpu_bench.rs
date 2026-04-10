//! Quick benchmark: CPU vs GPU matvec at inference-realistic sizes.
//!
//! Run with: cargo run -p flare-gpu --example gpu_bench --release

use std::time::Instant;

use flare_core::model::{matvec, ComputeBackend};
use flare_gpu::WebGpuBackend;

fn bench_matvec(gpu: &WebGpuBackend, label: &str, rows: usize, cols: usize, iters: usize) {
    let mat_bytes = rows * cols * 4;
    // Skip if matrix exceeds GPU buffer limit (~256MB)
    if mat_bytes > 250_000_000 {
        println!(
            "{label:35} (skipped: {:.0}MB > GPU buffer limit)",
            mat_bytes as f64 / 1e6
        );
        return;
    }

    let mat: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let v: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();

    // CPU
    let start = Instant::now();
    for _ in 0..iters {
        let _ = matvec(&mat, &v, rows, cols);
    }
    let cpu_us = start.elapsed().as_micros() as f64 / iters as f64;

    // GPU (warmup + bench)
    for _ in 0..3 {
        let _ = gpu.matvec(&mat, &v, rows, cols);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = gpu.matvec(&mat, &v, rows, cols);
    }
    let gpu_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = cpu_us / gpu_us;
    println!("{label:35} CPU: {cpu_us:8.0}µs  GPU: {gpu_us:8.0}µs  speedup: {speedup:.2}x");
}

fn main() {
    println!("Flare GPU Benchmark — CPU vs GPU matvec");
    println!("========================================");
    println!("NOTE: GPU is slow here due to per-call buffer allocation.");
    println!("Real gains need persistent buffers (future work).\n");

    let gpu = pollster::block_on(WebGpuBackend::new()).expect("No GPU adapter found");

    // SmolLM2-135M sizes
    bench_matvec(&gpu, "SmolLM: embed→Q (576x576)", 576, 576, 100);
    bench_matvec(&gpu, "SmolLM: FFN gate (1536x576)", 1536, 576, 100);
    bench_matvec(&gpu, "SmolLM: FFN down (576x1536)", 576, 1536, 100);
    bench_matvec(&gpu, "SmolLM: logits (49152x576)", 49152, 576, 10);

    println!();

    // Qwen2.5-0.5B sizes
    bench_matvec(&gpu, "Qwen: embed→Q (896x896)", 896, 896, 100);
    bench_matvec(&gpu, "Qwen: FFN gate (4864x896)", 4864, 896, 50);
    bench_matvec(&gpu, "Qwen: FFN down (896x4864)", 896, 4864, 50);
    bench_matvec(&gpu, "Qwen: logits (151936x896)", 151936, 896, 5);

    println!();

    // Llama-3.2-1B sizes
    bench_matvec(&gpu, "1B: embed→Q (2048x2048)", 2048, 2048, 50);
    bench_matvec(&gpu, "1B: FFN gate (8192x2048)", 8192, 2048, 20);
    bench_matvec(&gpu, "1B: logits (128256x2048)", 128256, 2048, 3);
}

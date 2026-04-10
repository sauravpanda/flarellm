//! Benchmark for matvec at inference-realistic sizes.
//!
//! Run with: cargo run -p flare-core --example matvec_bench --release

use std::time::Instant;

use flare_core::model::matvec;

fn bench(label: &str, rows: usize, cols: usize, iters: usize) {
    let mat: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let v: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();

    // Warmup
    for _ in 0..5 {
        let _ = matvec(&mat, &v, rows, cols);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = matvec(&mat, &v, rows, cols);
    }
    let elapsed = start.elapsed();
    let us = elapsed.as_micros() as f64 / iters as f64;
    let gflops = (2.0 * rows as f64 * cols as f64) / (us * 1000.0);

    println!("{label:35} {us:8.0}µs  ({gflops:.2} GFLOP/s)");
}

fn main() {
    println!("Flare matvec benchmark");
    println!("======================\n");

    #[cfg(target_arch = "aarch64")]
    println!("Backend: ARM NEON SIMD\n");
    #[cfg(not(target_arch = "aarch64"))]
    println!("Backend: scalar (4-wide unrolled)\n");

    // SmolLM2-135M (dim=576, inter=1536, vocab=49152)
    bench("SmolLM: Q proj (576×576)", 576, 576, 500);
    bench("SmolLM: FFN gate (1536×576)", 1536, 576, 500);
    bench("SmolLM: FFN down (576×1536)", 576, 1536, 500);
    bench("SmolLM: output logits (49152×576)", 49152, 576, 20);

    println!();

    // Qwen2.5-0.5B (dim=896, inter=4864, vocab=151936)
    bench("Qwen: Q proj (896×896)", 896, 896, 200);
    bench("Qwen: FFN gate (4864×896)", 4864, 896, 100);
    bench("Qwen: FFN down (896×4864)", 896, 4864, 100);

    println!();

    // Llama-3.2-1B (dim=2048, inter=8192)
    bench("1B: Q proj (2048×2048)", 2048, 2048, 50);
    bench("1B: FFN gate (8192×2048)", 8192, 2048, 20);
}

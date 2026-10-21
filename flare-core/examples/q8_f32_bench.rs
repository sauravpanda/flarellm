//! Run: cargo run --release -p flarellm-core --example q8_f32_bench
//! Same packed bytes and f32 inputs for scalar and platform dispatch. No model download.
use flare_core::model::{matvec_q8_0_f32, matvec_q8_0_scalar};
use std::{hint::black_box, time::Instant};

fn main() {
    println!(
        "Q8 x f32; arch={}; times are medians of 7 samples",
        std::env::consts::ARCH
    );
    for (label, rows, cols) in [
        ("Q", 2048, 1024),
        ("FFN gate/up", 3072, 1024),
        ("FFN down", 1024, 3072),
    ] {
        let mut weights = vec![0; rows * (cols / 32) * 34];
        for (index, block) in weights.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&0x2800u16.to_le_bytes());
            for (j, q) in block[2..].iter_mut().enumerate() {
                *q = (index * 17 + j * 31) as u8;
            }
        }
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 % 257) as f32 - 128.0) / 129.0)
            .collect();
        let scalar = matvec_q8_0_scalar(&weights, &input, rows, cols);
        let optimized = matvec_q8_0_f32(&weights, &input, rows, cols);
        assert_eq!(
            scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            optimized.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let mut samples = [Vec::new(), Vec::new()];
        for sample in 0..8 {
            // Alternate order to reduce systematic warmup/frequency bias.
            for which in [sample % 2, 1 - sample % 2] {
                let kernel = if which == 0 {
                    matvec_q8_0_scalar
                } else {
                    matvec_q8_0_f32
                };
                let start = Instant::now();
                for _ in 0..50 {
                    black_box(kernel(black_box(&weights), black_box(&input), rows, cols));
                }
                if sample != 0 {
                    samples[which].push(start.elapsed().as_secs_f64() * 1e6 / 50.0);
                }
            }
        }
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{label}: {rows}x{cols}; scalar {:.2} us, dispatch {:.2} us, {:.2}x",
            samples[0][3],
            samples[1][3],
            samples[0][3] / samples[1][3]
        );
    }
}

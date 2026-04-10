# Flare Benchmarks

**Date:** 2026-04-10
**Hardware:** Apple M5 Pro (ARM64)
**Rust:** stable, `--release`

## CPU MatVec (ARM NEON SIMD)

The core inference bottleneck. Uses ARM NEON `vfmaq_f32` with 4-way accumulator unrolling (16 f32 elements/iteration).

| Matrix Size | Model Context | Time | Throughput |
|---|---|---|---|
| 576×576 | SmolLM-135M Q proj | 15µs | 44.2 GFLOP/s |
| 1536×576 | SmolLM-135M FFN gate | 42µs | 42.2 GFLOP/s |
| 576×1536 | SmolLM-135M FFN down | 45µs | 39.4 GFLOP/s |
| 49152×576 | SmolLM-135M logits | 1547µs | 36.6 GFLOP/s |
| 896×896 | Qwen2.5-0.5B Q proj | 45µs | 35.7 GFLOP/s |
| 4864×896 | Qwen2.5-0.5B FFN gate | 239µs | 36.4 GFLOP/s |
| 896×4864 | Qwen2.5-0.5B FFN down | 256µs | 34.0 GFLOP/s |
| 2048×2048 | Llama-1B Q proj | 220µs | 38.1 GFLOP/s |
| 8192×2048 | Llama-1B FFN gate | 948µs | 35.4 GFLOP/s |

### SIMD vs Scalar Comparison

| Matrix Size | Scalar (4-wide unroll) | NEON SIMD | Speedup |
|---|---|---|---|
| 576×576 | 66µs (10 GFLOP/s) | 15µs (44 GFLOP/s) | **4.4x** |
| 1536×576 | 173µs (10 GFLOP/s) | 42µs (42 GFLOP/s) | **4.1x** |
| 49152×576 | 5552µs (10 GFLOP/s) | 1547µs (37 GFLOP/s) | **3.6x** |
| 8192×2048 | 3171µs (11 GFLOP/s) | 948µs (35 GFLOP/s) | **3.3x** |

## GPU (Metal via wgpu) vs CPU

GPU backend currently allocates buffers per-call, so dispatch overhead dominates. Listed for reference; persistent buffer pool will close this gap.

| Matrix Size | CPU (NEON) | GPU (Metal) | GPU/CPU |
|---|---|---|---|
| 576×576 | 15µs | 1765µs | 0.01x |
| 1536×576 | 44µs | 2099µs | 0.02x |
| 49152×576 | 1799µs | 26672µs | 0.07x |
| 4864×896 | 234µs | 6224µs | 0.04x |
| 8192×2048 | 959µs | 16745µs | 0.06x |

> **Note:** GPU is slower due to ~2ms fixed overhead per dispatch (buffer create + upload + readback). With persistent buffers and batched dispatch, GPU should be 5-10x faster than CPU for large matrices.

## Full MatMul (flare-simd tiled)

Batch matrix multiply using tiled loop with TILE=32 for cache locality.

| Dimensions | Time | Throughput |
|---|---|---|
| [1×2048] × [2048×2048] | 1.4ms | 5.97 GFLOP/s |
| [1×4096] × [4096×4096] | 8.2ms | 4.10 GFLOP/s |
| [1×2048] × [2048×8192] | 9.5ms | 3.53 GFLOP/s |
| [32×2048] × [2048×2048] | 34.4ms | 7.79 GFLOP/s |

## Sampling

| Operation | Vocab Size | Time |
|---|---|---|
| Greedy (argmax) | 128,256 | 61µs |
| Top-p (nucleus) | 128,256 | 1,739µs |
| Top-k | 128,256 | 1,587µs |

## End-to-End Inference (estimated)

Based on matvec timings and model architecture, estimated single-token generation:

| Model | Quant | Est. tok/s | Notes |
|---|---|---|---|
| SmolLM2-135M | Q8_0 | ~75-90 | ARM NEON SIMD |
| Qwen2.5-0.5B | Q8_0 | ~15-20 | ARM NEON SIMD |
| Llama-3.2-1B | Q8_0 | ~5-8 | ARM NEON SIMD |

Previous measurements (before NEON SIMD):
- SmolLM2-135M Q8_0: 29 tok/s (scalar 4-wide unroll)
- SmolLM2-135M Q8_0: 19 tok/s (naive scalar)

## How to Run

```bash
# CPU matvec benchmark
cargo run -p flare-core --example matvec_bench --release

# GPU vs CPU comparison
cargo run -p flare-gpu --example gpu_bench --release

# Full matmul + sampling benchmarks
cargo bench -p flare-simd
```

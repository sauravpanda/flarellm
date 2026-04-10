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

## End-to-End Inference — SmolLM2-135M Q8_0 (baseline)

Real measurements using `e2e_bench` with SmolLM2-135M-Instruct Q8_0 (138MB, 30 layers, dim=576).

| Metric | Value |
|---|---|
| Model load time | 0.13s |
| Prefill (6 tokens) | 130 tok/s |
| Decode (16 tokens) | 126 tok/s |
| Decode (64 tokens) | 123 tok/s |
| Decode (256 tokens) | 102 tok/s |
| Sustained decode (512 tokens) | **93.7 tok/s** |

### Performance history

| Version | tok/s (decode) | Change |
|---|---|---|
| Naive scalar | 19 | baseline |
| 4-wide unrolled scalar | 29 | +53% |
| ARM NEON SIMD (current) | 126 | **+6.6x** |

### Phase 1 targets vs actual

| Target | Goal | Actual | Status |
|---|---|---|---|
| CPU 135M | 60 tok/s | 126 tok/s | exceeded |
| GPU 135M | 150 tok/s | not yet wired | pending |

## How to Run

### Single-model benchmark

```bash
# Download the baseline model (~138 MB)
./scripts/download_baseline_model.sh

# Run e2e benchmark (human-readable)
cargo run -p flarellm-server --example e2e_bench --release

# Append result to BENCHMARK_HISTORY.md
cargo run -p flarellm-server --example e2e_bench --release -- --log

# Machine-readable JSON output
cargo run -p flarellm-server --example e2e_bench --release -- --json
```

### Multi-model benchmark

Place one or more `.gguf` files under `models/`, then:

```bash
# Benchmark all models (human-readable summary table)
./scripts/bench_multi.sh

# JSON output — one object per model per line
./scripts/bench_multi.sh --json

# JSON output + append each result to BENCHMARK_HISTORY.md
./scripts/bench_multi.sh --json --log
```

### Performance regression check

```bash
# Pass/fail check against a 60 tok/s floor (uses baseline model)
./scripts/check_perf_regression.sh

# Custom threshold
THRESHOLD_TOK_S=40 ./scripts/check_perf_regression.sh

# Custom model
MODEL_PATH=models/my-model.gguf THRESHOLD_TOK_S=20 ./scripts/check_perf_regression.sh
```

### Automated benchmarks (CI)

The `benchmark` GitHub Actions workflow runs nightly and on `workflow_dispatch`.
It downloads the baseline model, runs the regression check, and uploads the JSON
result as a build artefact. To trigger manually:

```
GitHub → Actions → Benchmark → Run workflow
```

### CPU / GPU microbenchmarks (no model needed)

```bash
# CPU matvec benchmark
cargo run -p flarellm-core --example matvec_bench --release

# GPU vs CPU comparison
cargo run -p flarellm-gpu --example gpu_bench --release

# Full matmul + sampling benchmarks
cargo bench -p flarellm-simd
```

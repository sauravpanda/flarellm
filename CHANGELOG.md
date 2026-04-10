# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `@aspect/flare` npm package: automated publishing workflow triggered on GitHub releases; `npm pack --dry-run` validation added to WASM CI job (#52)
- `FlareProgressiveLoader` WASM export: fetch a GGUF model from a URL with streaming byte-level download progress (#50)
- `load_model_weights_with_progress` in `flare-loader` for layer-by-layer parse callbacks (#50)
- Browser demo tabbed UI with URL-based progressive loading and live progress bar (#50)
- Browser quickstart section in README with `@aspect/flare` npm install example (#52)
- ARM NEON SIMD matvec with 4-way accumulator unrolling (#59)
- x86_64 AVX2+FMA SIMD matvec with runtime feature detection (#74)
- Rayon parallelism for matvec on large matrices (5M+ FMAs threshold) (#72)
- GPU pipeline cache to avoid per-call shader recompilation (#67)
- WebGPU compute backend wired into forward pass via `ComputeBackend` trait (#58)
- WASM compilation verified and added to CI (#48, #69)
- API server with real model loading and inference (#53, #61)
- Benchmark history log via `e2e_bench --log` (#65)
- `simple_chat` example for the umbrella crate (#78)
- Re-exports and feature flags for the `flarellm` umbrella crate (#73)
- NEON SIMD RMSNorm (#76)
- RoPE cos/sin table caching (#77)
- Optimized top_k/top_p sampling with partition selection — 5x faster (#75)
- ARM NEON SIMD matvec implementation (#59)
- End-to-end benchmark example (#64)
- Q4_K_M, Q6_K, Q5_K dequantization support (#12, #45, #60)

### Changed
- Renamed workspace crates to `flarellm-*` namespace for crates.io (#66)
- Bumped `flarellm` umbrella crate to 0.1.0
- `Model` now uses `Box<dyn ComputeBackend>` for pluggable CPU/GPU compute (#58)
- `ComputeBackend: Send + Sync` so models can be shared across threads (#61)

### Fixed
- Q4_K_M / Q5_K / Q6_K dequantization: use `div_ceil` for block count, fixing
  zeroed partial blocks that produced incoherent output (#46, #60)
- WASM `usize` overflow in `MemoryBudget::native()` for 32-bit pointer width (#48)
- Qwen2 attention bias support for correct inference (#57)
- GGUF magic number byte order (#34)
- BPE tokenizer GPT-2 byte-to-unicode mapping (#36)
- Chat template auto-detection from GGUF metadata (#43)

## Performance journey

| Commit | Optimization | SmolLM-135M decode |
|---|---|---|
| Initial | Naive scalar matvec | ~19 tok/s |
| `8aff56f` | 4-wide manual unrolling | ~29 tok/s |
| `58eb62b` | ARM NEON SIMD | ~95 tok/s |
| `1339021` | Rayon parallelism | ~100 tok/s |
| `fbc386e` | Sampling optimization | ~100 tok/s (greedy unchanged) |
| Current | All optimizations | ~120 tok/s |

See [`BENCHMARK_HISTORY.md`](BENCHMARK_HISTORY.md) for detailed timing data.

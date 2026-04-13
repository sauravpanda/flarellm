# Benchmark History

Performance log for Flare LLM. Each entry is a snapshot at a specific commit.
Baseline model: SmolLM2-135M-Instruct Q8_0 (138MB, 30 layers, dim=576).

---

### 2026-04-09 (historical) — `b6f341a` Initial working inference

**Hardware:** Apple M5 Pro, ARM64 (scalar)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Notes:** First working end-to-end inference. Naive scalar matvec.

| Metric | tok/s |
|---|---|
| Decode (estimated) | ~19 |
| Sustained (estimated) | ~19 |

### 2026-04-09 (historical) — `8aff56f` Optimize forward pass (4-wide unrolling)

**Hardware:** Apple M5 Pro, ARM64 (scalar, 4-wide unroll)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Notes:** Added 4-wide manual unrolling to matvec. +53% improvement.

| Metric | tok/s |
|---|---|
| Decode (estimated) | ~29 |
| Sustained (estimated) | ~29 |

### 2026-04-10 (historical) — `58eb62b` ARM NEON SIMD matvec

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Notes:** ARM NEON intrinsics with 4-way accumulator. 4.4x over scalar.

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 126 |
| Decode (256 tok) | 102 |
| Sustained (512 tok) | **93.7** |

### 2026-04-09 19:30 — `ac3a913 Add real e2e benchmark with SmolLM2-135M baseline (#64)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.25s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 123.8 |
| Decode (64 tok) | 120.4 |
| Decode (256 tok) | 107.3 |
| Sustained (512 tok) | **93.1** |

### 2026-04-09 20:13 — `7c9cede Add benchmark history log with --log flag (#65)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 123.6 |
| Decode (64 tok) | 120.2 |
| Decode (256 tok) | 106.8 |
| Sustained (512 tok) | **78.9** |

### 2026-04-09 20:39 — `cd1225a Cache GPU compute pipelines to avoid per-call shader comp...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 121.3 |
| Decode (64 tok) | 121.1 |
| Decode (256 tok) | 110.5 |
| Sustained (512 tok) | **93.9** |

### 2026-04-09 20:40 — `cd1225a Cache GPU compute pipelines to avoid per-call shader comp...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.87s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 13.7 |
| Decode (64 tok) | 13.4 |
| Decode (256 tok) | 12.4 |
| Sustained (512 tok) | **12.5** |

### 2026-04-09 20:54 — `f468210 Cache RoPE cos/sin tables per call (#77)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 141.1 |
| Decode (64 tok) | 133.2 |
| Decode (256 tok) | 112.9 |
| Sustained (512 tok) | **93.9** |

### 2026-04-09 20:55 — `f468210 Cache RoPE cos/sin tables per call (#77)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.90s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 18.4 |
| Decode (64 tok) | 22.7 |
| Decode (256 tok) | 23.1 |
| Sustained (512 tok) | **20.2** |

### 2026-04-13 12:34 — `46b0840 Add doc-tests for flare-core public API (closes #362) (#365)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 148.4 |
| Decode (64 tok) | 143.6 |
| Decode (256 tok) | 117.5 |
| Sustained (512 tok) | **111.2** |

### 2026-04-13 12:34 — `46b0840 Add doc-tests for flare-core public API (closes #362) (#365)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 148.3 |
| Decode (64 tok) | 144.6 |
| Decode (256 tok) | 129.8 |
| Sustained (512 tok) | **105.9** |

### 2026-04-13 13:06 — `b22e91f Merge remote-tracking branch 'origin/worktree-feat-top-k-...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.12s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 155.0 |
| Decode (64 tok) | 151.7 |
| Decode (256 tok) | 134.5 |
| Sustained (512 tok) | **116.7** |

### 2026-04-13 15:00 — `3f04928 Merge pull request #374 from sauravpanda/feat-gpu-bench-366`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 1.0 |
| Decode (64 tok) | 1.1 |
| Decode (256 tok) | 1.1 |
| Sustained (512 tok) | **1.2** |

### 2026-04-13 15:12 — `86f6afe Fix WGSL shader builtin errors and GGUF raw weight dimens...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.09s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 1.4 |
| Decode (64 tok) | 1.4 |
| Decode (256 tok) | 1.2 |
| Sustained (512 tok) | **1.2** |

### 2026-04-13 15:21 — `af233e4 Merge pull request #375 from sauravpanda/gpu-resident-for...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 7.8 |
| Decode (64 tok) | 8.1 |
| Decode (256 tok) | 7.6 |
| Sustained (512 tok) | **6.8** |

### 2026-04-13 15:22 — `af233e4 Merge pull request #375 from sauravpanda/gpu-resident-for...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.09s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 139.5 |
| Decode (64 tok) | 137.3 |
| Decode (256 tok) | 122.1 |
| Sustained (512 tok) | **105.0** |

### 2026-04-13 15:23 — `af233e4 Merge pull request #375 from sauravpanda/gpu-resident-for...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.92s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 25.8 |
| Decode (64 tok) | 25.8 |
| Decode (256 tok) | 25.0 |
| Sustained (512 tok) | **24.2** |

### 2026-04-13 15:41 — `ccad363 Merge pull request #377 from sauravpanda/fix-buffer-shard...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.90s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 2.9 |
| Decode (64 tok) | 2.8 |
| Decode (256 tok) | 2.7 |
| Sustained (512 tok) | **2.5** |

### 2026-04-13 15:56 — `c7f6ad1 Merge pull request #378 from sauravpanda/gpu-forward-pass...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 8.8 |
| Decode (64 tok) | 8.6 |
| Decode (256 tok) | 8.1 |
| Sustained (512 tok) | **7.5** |

### 2026-04-13 16:03 — `c7f6ad1 Merge pull request #378 from sauravpanda/gpu-forward-pass...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.87s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 2.9 |
| Decode (64 tok) | 2.9 |
| Decode (256 tok) | 2.8 |
| Sustained (512 tok) | **2.7** |

### 2026-04-13 16:03 — `c7f6ad1 Merge pull request #378 from sauravpanda/gpu-forward-pass...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 144.6 |
| Decode (64 tok) | 135.0 |
| Decode (256 tok) | 123.9 |
| Sustained (512 tok) | **106.7** |


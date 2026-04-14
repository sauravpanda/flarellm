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

### 2026-04-13 16:47 — `43a26bb Merge pull request #400 from sauravpanda/feat/kivi-2bit-k...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.13s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 130.3 |
| Decode (64 tok) | 133.0 |
| Decode (256 tok) | 119.8 |
| Sustained (512 tok) | **103.6** |

### 2026-04-13 16:49 — `43a26bb Merge pull request #400 from sauravpanda/feat/kivi-2bit-k...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 8.7 |
| Decode (64 tok) | 8.8 |
| Decode (256 tok) | 8.3 |
| Sustained (512 tok) | **7.7** |

### 2026-04-13 16:50 — `43a26bb Merge pull request #400 from sauravpanda/feat/kivi-2bit-k...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.94s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 25.8 |
| Decode (64 tok) | 25.7 |
| Decode (256 tok) | 25.0 |
| Sustained (512 tok) | **24.0** |

### 2026-04-13 17:22 — `01448a5 Escape brackets in doc comment to fix rustdoc build`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.09s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 141.4 |
| Decode (64 tok) | 138.2 |
| Decode (256 tok) | 121.1 |
| Sustained (512 tok) | **104.7** |

### 2026-04-13 17:23 — `01448a5 Escape brackets in doc comment to fix rustdoc build`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.96s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 25.9 |
| Decode (64 tok) | 25.8 |
| Decode (256 tok) | 24.8 |
| Sustained (512 tok) | **23.9** |

### 2026-04-13 17:25 — `01448a5 Escape brackets in doc comment to fix rustdoc build`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.09s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 8.9 |
| Decode (64 tok) | 8.8 |
| Decode (256 tok) | 8.3 |
| Sustained (512 tok) | **7.6** |

### 2026-04-13 17:42 — `0870941 Merge pull request #407 from sauravpanda/direct-q8-0-matvec`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 229.7 |
| Decode (64 tok) | 219.6 |
| Decode (256 tok) | 168.3 |
| Sustained (512 tok) | **150.7** |

### 2026-04-13 17:43 — `0870941 Merge pull request #407 from sauravpanda/direct-q8-0-matvec`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.95s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 24.9 |
| Decode (64 tok) | 24.6 |
| Decode (256 tok) | 23.7 |
| Sustained (512 tok) | **22.8** |

### 2026-04-13 17:47 — `0870941 Merge pull request #407 from sauravpanda/direct-q8-0-matvec`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.91s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 25.4 |
| Decode (64 tok) | 25.7 |
| Decode (256 tok) | 24.5 |
| Sustained (512 tok) | **24.1** |

### 2026-04-13 17:49 — `0870941 Merge pull request #407 from sauravpanda/direct-q8-0-matvec`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.91s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 24.3 |
| Decode (64 tok) | 22.7 |
| Decode (256 tok) | 21.4 |
| Sustained (512 tok) | **21.6** |

### 2026-04-13 17:55 — `285d905 Merge pull request #408 from sauravpanda/optimize-q8-neon...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 113.5 |
| Decode (64 tok) | 114.2 |
| Decode (256 tok) | 105.4 |
| Sustained (512 tok) | **93.1** |

### 2026-04-13 17:56 — `285d905 Merge pull request #408 from sauravpanda/optimize-q8-neon...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.89s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 25.5 |
| Decode (64 tok) | 25.6 |
| Decode (256 tok) | 24.4 |
| Sustained (512 tok) | **23.5** |

### 2026-04-13 17:56 — `285d905 Merge pull request #408 from sauravpanda/optimize-q8-neon...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 207.3 |
| Decode (64 tok) | 179.1 |
| Decode (256 tok) | 175.4 |
| Sustained (512 tok) | **147.1** |

### 2026-04-13 18:13 — `3d1c838 Merge pull request #409 from sauravpanda/q8_0-int-dot-pro...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.97s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 30.8 |
| Decode (64 tok) | 30.7 |
| Decode (256 tok) | 28.9 |
| Sustained (512 tok) | **28.5** |

### 2026-04-13 18:13 — `3d1c838 Merge pull request #409 from sauravpanda/q8_0-int-dot-pro...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.12s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 176.2 |
| Decode (64 tok) | 178.3 |
| Decode (256 tok) | 157.4 |
| Sustained (512 tok) | **134.4** |

### 2026-04-13 18:14 — `3d1c838 Merge pull request #409 from sauravpanda/q8_0-int-dot-pro...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 228.6 |
| Decode (64 tok) | 218.1 |
| Decode (256 tok) | 187.5 |
| Sustained (512 tok) | **146.6** |

### 2026-04-13 18:15 — `3d1c838 Merge pull request #409 from sauravpanda/q8_0-int-dot-pro...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.94s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 29.5 |
| Decode (64 tok) | 31.0 |
| Decode (256 tok) | 29.8 |
| Sustained (512 tok) | **28.3** |

### 2026-04-13 18:24 — `77c10a1 Use ARM SDOT inline assembly for int8 dot product (#410)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.96s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 31.8 |
| Decode (64 tok) | 31.3 |
| Decode (256 tok) | 29.7 |
| Sustained (512 tok) | **28.2** |

### 2026-04-13 18:24 — `77c10a1 Use ARM SDOT inline assembly for int8 dot product (#410)`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 229.1 |
| Decode (64 tok) | 220.9 |
| Decode (256 tok) | 185.0 |
| Sustained (512 tok) | **151.1** |

### 2026-04-13 18:29 — `e99a0ff Merge pull request #411 from sauravpanda/optimize-silu-at...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.92s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 30.8 |
| Decode (64 tok) | 25.5 |
| Decode (256 tok) | 30.6 |
| Sustained (512 tok) | **24.5** |

### 2026-04-13 18:29 — `e99a0ff Merge pull request #411 from sauravpanda/optimize-silu-at...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 211.1 |
| Decode (64 tok) | 217.0 |
| Decode (256 tok) | 191.0 |
| Sustained (512 tok) | **169.1** |

### 2026-04-13 18:30 — `e99a0ff Merge pull request #411 from sauravpanda/optimize-silu-at...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.91s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 30.8 |
| Decode (64 tok) | 31.0 |
| Decode (256 tok) | 30.5 |
| Sustained (512 tok) | **29.5** |

### 2026-04-13 18:54 — `e34507b Merge pull request #412 from sauravpanda/eliminate-forwar...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 1.04s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 21.5 |
| Decode (64 tok) | 21.5 |
| Decode (256 tok) | 21.0 |
| Sustained (512 tok) | **20.4** |

### 2026-04-13 18:54 — `e34507b Merge pull request #412 from sauravpanda/eliminate-forwar...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 233.1 |
| Decode (64 tok) | 225.4 |
| Decode (256 tok) | 201.6 |
| Sustained (512 tok) | **172.1** |

### 2026-04-13 18:57 — `e34507b Merge pull request #412 from sauravpanda/eliminate-forwar...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.86s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 30.4 |
| Decode (64 tok) | 30.2 |
| Decode (256 tok) | 28.1 |
| Sustained (512 tok) | **25.6** |

### 2026-04-13 18:57 — `e34507b Merge pull request #412 from sauravpanda/eliminate-forwar...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 205.3 |
| Decode (64 tok) | 195.1 |
| Decode (256 tok) | 159.3 |
| Sustained (512 tok) | **152.3** |

### 2026-04-13 18:58 — `e34507b Merge pull request #412 from sauravpanda/eliminate-forwar...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.10s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 219.8 |
| Decode (64 tok) | 206.5 |
| Decode (256 tok) | 181.5 |
| Sustained (512 tok) | **156.5** |

### 2026-04-13 19:05 — `8a123d5 Merge pull request #413 from sauravpanda/optimize-q8-matv...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.88s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 30.8 |
| Decode (64 tok) | 30.4 |
| Decode (256 tok) | 29.5 |
| Sustained (512 tok) | **28.4** |

### 2026-04-13 19:05 — `8a123d5 Merge pull request #413 from sauravpanda/optimize-q8-matv...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 165.9 |
| Decode (64 tok) | 220.4 |
| Decode (256 tok) | 185.1 |
| Sustained (512 tok) | **156.9** |

### 2026-04-13 19:25 — `8b28359 Use Q8_0 for output projection matvec to reduce memory ba...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~1498M params, 16 layers, dim=2048  
**Load time:** 0.98s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 33.0 |
| Decode (64 tok) | 32.5 |
| Decode (256 tok) | 31.3 |
| Sustained (512 tok) | **28.6** |

### 2026-04-13 19:25 — `8b28359 Use Q8_0 for output projection matvec to reduce memory ba...`

**Hardware:** Apple M5 Pro, ARM64 (NEON SIMD)  
**Model:** Llama, ~162M params, 30 layers, dim=576  
**Load time:** 0.11s

| Metric | tok/s |
|---|---|
| Decode (16 tok) | 214.0 |
| Decode (64 tok) | 208.8 |
| Decode (256 tok) | 186.6 |
| Sustained (512 tok) | **165.5** |


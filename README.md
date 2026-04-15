# Flare LLM

[![CI](https://github.com/sauravpanda/flarellm/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravpanda/flarellm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)

A WASM-first LLM inference engine with WebGPU acceleration, built in pure Rust.

Run large language models directly in the browser with zero server costs. Single codebase compiles to both native and WebAssembly — same WGSL shaders, same quantization kernels, same inference pipeline.

## Performance

Benchmarked on Apple M5 Pro, April 2026:

| Model | Flare (native) | llama.cpp | Gap | Browser (est.) |
|---|---|---|---|---|
| SmolLM2-135M Q8_0 | **224 tok/s** | 390 tok/s | 1.7x | ~80 tok/s |
| Llama 3.2 1B Q8_0 | **38 tok/s** | 124 tok/s | 3.2x | ~15 tok/s |

See [`BENCHMARK_HISTORY.md`](BENCHMARK_HISTORY.md) for the full performance log.

## Why Flare?

|  | Flare | llama.cpp | WebLLM | Transformers.js |
|---|---|---|---|---|
| Browser inference | **Yes** | No | Yes | Yes |
| Native inference | **Yes** | Yes | No | Via Node |
| Single codebase | **Yes** | No | No | No |
| Standard GGUF files | **Yes** | Yes | No (TVM) | No (ONNX) |
| Pure Rust/WASM | **Yes** | C/C++ | C++ (emscripten) | C++ (ONNX RT) |
| Progressive loading | **Yes** | No | No | No |
| Speculative decoding | **Yes** | Yes | No | No |
| BitNet ternary | **Yes** | Partial | No | No |

### The Browser Advantage

- **Privacy**: Data never leaves the user's device — no server to breach
- **Cost**: No GPU bills. Your "inference cluster" is every user's device
- **Latency**: Zero network round trips. Tokens start immediately
- **Offline**: Works without internet after first model download
- **Scale**: Millions of concurrent users at CDN cost (~$500/mo vs $50K-500K/mo GPU)

## Features

### Inference Engine
- 22+ quantization formats: Q4_0–Q8_1, Q2K–Q6K, IQ1S–IQ4XS, BF16, F16, F32, **BitNet ternary**
- Fused QKV and gate/up projections (43% fewer matvec calls per layer)
- Online softmax attention (Flash Attention style, single-pass)
- N-gram speculative decoding (zero memory overhead)
- KIVI 2-bit KV cache quantization (8x memory savings)
- Progressive inference — generate with partial model load
- Greedy fused argmax (skips logits buffer for temperature=0)
- Zero-allocation forward pass with pre-allocated buffers

### Compute Backends
- **Apple Accelerate** (macOS) — AMX hardware via `cblas_sgemv`
- **ARM NEON SIMD** — Q8_0 int8 dot product with inline `sdot` assembly
- **x86 AVX2/FMA** — auto-detected at runtime
- **WebGPU** — 37 WGSL shaders, GPU-resident forward pass, fused kernels
- **WASM SIMD128** — browser CPU fallback
- **Multi-worker WASM** — parallel matmul via SharedArrayBuffer (opt-in)

### Model Support
- GGUF and SafeTensors formats
- Llama, Qwen2, Mistral, Phi-3, Gemma 2 architectures
- Grouped-query attention (GQA)
- Ring-buffer KV cache with Q8 and Q2 quantization options
- BPE tokenizer (HuggingFace `tokenizer.json` compatible)
- 6 chat templates: Llama3, ChatML, Phi3, Gemma, Alpaca, Raw

### Browser / WASM
- Progressive model loading with download progress
- Cache API integration for offline model persistence
- Streaming token generation with callbacks
- WebGPU auto-detection with SIMD fallback
- COOP/COEP headers for SharedArrayBuffer multi-threading
- Full chat demo with system prompts and performance metrics

## Quick Start

### Native

```bash
# Clone and build
git clone https://github.com/sauravpanda/flarellm
cd flarellm
cargo build --release

# Download a small model
bash scripts/download_baseline_model.sh

# Run benchmark
cargo run --release --example e2e_bench

# Run with GPU acceleration
cargo run --release --example e2e_bench -- --gpu

# Run speculative decoding comparison
cargo run --release --example e2e_bench -- --speculative
```

### Rust API

```rust
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;
use std::io::BufReader;
use std::fs::File;

let mut reader = BufReader::new(File::open("model.gguf").unwrap());
let gguf = GgufFile::parse_header(&mut reader).unwrap();
let config = gguf.to_model_config().unwrap();
let weights = load_model_weights(&gguf, &mut reader).unwrap();
let mut model = Model::new(config, weights);

let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

let mut gen = Generator::new(&mut model, params);
let tokens = gen.generate(
    &prompt_tokens, 128, eos_id,
    || rand::random::<f32>(),
    |token, _| { print!("{}", decode(token)); true },
);
```

### Browser

```bash
# Build WASM
wasm-pack build flare-web --target web

# Open the demo
open flare-web/demo/index.html
```

```javascript
import init, { FlareEngine } from '@aspect/flare';

await init();
const buffer = await fetch('model.gguf').then(r => r.arrayBuffer());
const engine = FlareEngine.load(new Uint8Array(buffer));

// Initialize WebGPU (falls back to CPU SIMD if unavailable)
await engine.init_gpu();

// Stream tokens
engine.begin_stream(promptTokens, 128);
while (!engine.stream_done()) {
    const token = engine.next_token();
    document.body.textContent += engine.decode_token(token);
}
```

### P2P collaborative inference primitives

For experiments in splitting inference across multiple peers (e.g. a
WebRTC mesh), `FlareEngine` exposes the "head" and "tail" of a forward
pass. A coordinator peer can embed a token, hand the hidden state off
through a chain of peers running some subset of transformer layers, and
then run the final RMSNorm + output projection locally to recover
logits:

```javascript
// Coordinator peer: embed the next input token.
const hidden = engine.embed_token(tokenId); // Float32Array, hidden_dim

// ... ship `hidden` through the peer mesh; each peer runs some layers
// on its local `FlareEngine` and forwards the updated hidden state ...

// Coordinator peer: project the final hidden state to logits.
const logits = engine.output_projection(finalHidden); // Float32Array, vocab_size
```

The full WebRTC orchestration layer lives in JavaScript; these two
primitives plus the existing `forward*` methods are the minimum Rust
surface needed to start P2P experimentation ([#389](../../issues/389)).

## Architecture

```
flare-core        Core inference (tensor, model, KV cache, sampling, tokenizer)
flare-loader      Model loading (GGUF, SafeTensors, quantization, progressive)
flare-gpu         WebGPU/wgpu compute backend + 37 WGSL shaders
flare-simd        WASM SIMD128 CPU fallback
flare-web         Browser integration (wasm-bindgen, WebGPU, Cache API)
flare-server      Native server with OpenAI-compatible API
```

### Compute Path Selection

| Platform | Condition | Compute Path |
|---|---|---|
| macOS | Always | Apple Accelerate (AMX) + Q8_0 int8 for large models |
| Linux/Windows ARM | Always | ARM NEON SIMD with rayon parallelism |
| Linux/Windows x86 | AVX2+FMA detected | AVX2 SIMD with rayon parallelism |
| Chrome/Edge/Firefox | WebGPU available | WGSL compute shaders via wgpu |
| Any browser | Fallback | WASM SIMD128 (+ multi-worker if SharedArrayBuffer available) |

### Quantization Support

| Format | Bits/weight | GPU shader | CPU direct matvec | Notes |
|---|---|---|---|---|
| Q4_0, Q4_1 | 4.5 | Yes | Yes | Standard 4-bit |
| Q5_0, Q5_1 | 5.5 | Yes | — | Standard 5-bit |
| Q8_0, Q8_1 | 8.5 | Yes | Yes (NEON sdot) | Best quality/speed tradeoff |
| Q2K–Q6K | 2.5–6.5 | Yes | Q4K direct | K-quant family |
| IQ1S–IQ4XS | 1.5–4.5 | Yes | — | Importance-matrix quants |
| BF16, F16 | 16 | Yes | — | Half precision |
| **Ternary** | **1.58** | **Yes** | **Yes (NEON)** | **BitNet b1.58 — add/sub only** |

### Memory Requirements

| Model | Q8_0 | Q4_K | Ternary | KV Cache (2K) | Total (Q8_0) |
|---|---|---|---|---|---|
| 135M | 138MB | ~75MB | ~32MB | ~8MB | ~150MB |
| 1B | 1.2GB | ~600MB | ~250MB | ~32MB | ~1.2GB |
| 3B | 3.5GB | ~1.8GB | ~750MB | ~96MB | ~3.6GB |
| 7B | 7.5GB | ~3.8GB | ~1.6GB | ~256MB | ~7.8GB |

## Roadmap

### Shipped (v1.0–v1.1)
- [x] CPU inference: GGUF parser, Llama forward pass, sampling, tokenizer
- [x] WebGPU acceleration with GPU-resident forward pass
- [x] Browser demo with streaming chat, progressive loading, WebGPU
- [x] Native server with OpenAI-compatible API
- [x] Apple Accelerate AMX integration
- [x] Q8_0 int8 dot product with ARM sdot
- [x] Fused projections, online softmax attention, zero-alloc forward
- [x] N-gram speculative decoding
- [x] BitNet ternary weight support
- [x] Progressive inference (partial model generation)
- [x] KIVI 2-bit KV cache quantization

### In Progress
- [ ] WebGPU subgroups for attention/matmul speedup ([#386](../../issues/386))
- [ ] WebNN backend for NPU acceleration ([#388](../../issues/388))
- [ ] OPFS model caching for instant offline start ([#394](../../issues/394))

### Planned
- [ ] MoE on-demand expert loading from OPFS ([#387](../../issues/387))
- [ ] P2P collaborative inference via WebRTC ([#389](../../issues/389))
- [ ] LoRA adapter hot-swapping ([#390](../../issues/390))
- [ ] Edge runtime for Cloudflare Workers / Fermyon Spin ([#392](../../issues/392))
- [ ] WebTransport for parallel weight streaming ([#397](../../issues/397))
- [ ] AudioWorklet voice pipeline ([#395](../../issues/395))

See all [open issues](../../issues) for the full roadmap.

## Building

```bash
# Native build
cargo build --release

# Run all tests (550+)
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# WASM build
wasm-pack build flare-web --target web

# WASM with multi-threading support
wasm-pack build flare-web --target web --features wasm_threads

# Profile-guided optimization (PGO) build — squeezes ~5-10% on top of LTO.
# Requires `rustup component add llvm-tools-preview`.
./scripts/build_pgo.sh
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT OR Apache-2.0

# Flare LLM

A WASM-first LLM inference engine with WebGPU acceleration, built in pure Rust.

Run large language models directly in the browser with zero server costs. Single codebase compiles to both native and WebAssembly.

## Why Flare?

- **Privacy**: Data never leaves the user's device
- **Cost**: No GPU server bills — the user's hardware does the work
- **Latency**: Zero network round trips, tokens start immediately
- **Offline**: Works without internet connectivity
- **Scale**: Millions of concurrent users at CDN cost

## Features

- Pure Rust — compiles to native binary and WASM from one codebase
- WebGPU acceleration via `wgpu` (Vulkan/Metal/DX12 on native, WebGPU in browser)
- WASM SIMD128 CPU fallback for browsers without WebGPU
- GGUF and SafeTensors model format support
- Q4_0, Q8_0, F16, F32 quantization with on-the-fly dequantization
- Llama/Qwen2/Mistral architecture support
- Ring-buffer KV cache for memory-efficient generation
- Greedy, top-k, top-p sampling with temperature and repeat penalty
- BPE tokenizer compatible with HuggingFace `tokenizer.json`
- Streaming token generation with callback support

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
flare-core = "0.1"
flare-loader = "0.1"
```

### Load and Run a GGUF Model

```rust
use std::fs::File;
use std::io::BufReader;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

fn main() {
    // Parse GGUF header and extract model config
    let mut reader = BufReader::new(File::open("model.gguf").unwrap());
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let config = gguf.to_model_config().unwrap();

    println!("Model: {:?}, {} layers, {} params",
        config.architecture, config.num_layers, config.estimate_param_count());

    // Load weights (dequantizes Q4/Q8 on the fly)
    let weights = load_model_weights(&gguf, &mut reader).unwrap();
    let mut model = Model::new(config, weights);

    // Load tokenizer
    let tokenizer = BpeTokenizer::from_file("tokenizer.json").unwrap();

    // Encode prompt
    let prompt_tokens = tokenizer.encode("Once upon a time").unwrap();

    // Generate
    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    };

    let mut generator = Generator::new(&mut model, params);
    let generated = generator.generate(
        &prompt_tokens,
        128,                        // max tokens
        tokenizer.eos_token_id(),   // stop on EOS
        || rand::random::<f32>(),   // RNG
        |token_id, _step| {
            // Stream tokens as they arrive
            print!("{}", tokenizer.decode(&[token_id]).unwrap_or_default());
            true // continue generating
        },
    );

    println!("\n({} tokens generated)", generated.len());
}
```

### CLI Usage

```bash
# Build the CLI
cargo build --release --bin flare-cli

# Run with a GGUF model
./target/release/flare-cli model.gguf tokenizer.json --prompt "Explain quantum computing"

# Interactive mode (omit --prompt)
./target/release/flare-cli model.gguf tokenizer.json
```

## Architecture

```
flare-core        Core inference engine (tensor, model, KV cache, sampling, tokenizer)
flare-loader      Model loading (GGUF, SafeTensors, quantization, weight mapping)
flare-gpu         WebGPU/wgpu compute backend + WGSL shaders
flare-simd        WASM SIMD128 CPU fallback
flare-web         Browser integration (wasm-bindgen, WebGPU detection)
flare-server      Native server + CLI binary
```

### Compute Path Selection

Flare automatically selects the best available compute path:

| Platform | GPU Available | Compute Path |
|----------|--------------|-------------|
| Chrome/Firefox | WebGPU | WGSL compute shaders via `wgpu` |
| Older browsers | No | WASM SIMD128 CPU fallback |
| macOS/Linux/Windows | Metal/Vulkan/DX12 | Native `wgpu` |

### Supported Model Formats

| Format | Quantizations | Use Case |
|--------|--------------|----------|
| **GGUF** | Q4_0, Q8_0, F16, F32 | Primary format, pre-quantized models from HuggingFace |
| **SafeTensors** | F16, BF16, F32 | Custom fine-tuned models, LoRA adapters |

### Memory Requirements

| Model | Quantization | Weights | KV Cache (2K ctx) | Total |
|-------|-------------|---------|-------------------|-------|
| 0.5B | Q4_0 | ~300MB | ~16MB | ~320MB |
| 1B | Q4_0 | ~500MB | ~32MB | ~540MB |
| 1.5B | Q4_0 | ~800MB | ~48MB | ~850MB |
| 3B | Q4_0 | ~1.5GB | ~96MB | ~1.6GB |

## Crate Documentation

### flare-core

The core inference engine. Contains everything needed to run a model once weights are loaded.

#### Tensor

```rust
use flare_core::tensor::Tensor;

let t = Tensor::zeros(&[2, 3]);           // 2x3 zero tensor
let t = Tensor::from_vec(data, &[4, 4])?; // from existing data
t.shape();     // &[2, 3]
t.data();      // &[f32]
t.data_mut();  // &mut [f32]
t.numel();     // 6
t.reshape(&[3, 2])?;
```

#### Model Configuration

```rust
use flare_core::config::ModelConfig;

let config = ModelConfig::default(); // Llama-3.2-1B dimensions

// Memory estimation
config.estimate_param_count();                  // ~1.2B
config.estimate_weight_memory(4.0);             // ~600MB at Q4
config.estimate_kv_cache_memory(2048, 8.0);     // ~64MB at Q8
```

#### Sampling

```rust
use flare_core::sampling::*;

let mut logits = vec![1.0, 5.0, 2.0, 0.1];

// Apply transforms
apply_temperature(&mut logits, 0.7);
apply_repeat_penalty(&mut logits, &previous_tokens, 1.1);

// Sample
let token = sample_greedy(&logits);
let token = sample_top_p(&logits, 0.9, rng_value);
let token = sample_top_k(&logits, 40, rng_value);
```

#### KV Cache

Ring-buffer design — fixed memory, no allocations during generation:

```rust
use flare_core::kv_cache::KvCache;

let mut cache = KvCache::new(
    num_layers,   // 16
    max_seq_len,  // 2048
    num_kv_heads, // 8
    head_dim,     // 64
);

cache.write(layer_idx, &key_data, &value_data);
cache.advance();  // move to next position
cache.len();      // number of cached tokens
cache.clear();    // reset for new conversation
```

#### Generation

```rust
use flare_core::generate::Generator;
use flare_core::sampling::SamplingParams;

let params = SamplingParams {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
};

let mut gen = Generator::new(&mut model, params);

// Prefill prompt
gen.prefill(&prompt_tokens);

// Generate token by token
let step = gen.step(rng_value);
println!("Token: {}", step.token_id);

// Or generate in a loop with callback
let tokens = gen.generate(&prompt, 256, eos_id, rng_fn, |token, step| {
    print!("{}", decode(token));
    true // return false to stop early
});
```

#### Tokenizer

```rust
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};

// Load from HuggingFace tokenizer.json
let tok = BpeTokenizer::from_file("tokenizer.json")?;

let ids = tok.encode("Hello, world!")?;
let text = tok.decode(&ids)?;

tok.vocab_size();       // 128256
tok.bos_token_id();     // Some(1)
tok.eos_token_id();     // Some(2)
```

### flare-loader

Load models from GGUF and SafeTensors formats.

#### GGUF Loading

```rust
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;
use std::io::BufReader;
use std::fs::File;

let mut reader = BufReader::new(File::open("model.gguf")?);

// Parse header (fast — doesn't read weight data)
let gguf = GgufFile::parse_header(&mut reader)?;

// Inspect metadata
println!("Architecture: {:?}", gguf.architecture());
println!("Tensors: {}", gguf.tensors.len());

// Extract model config
let config = gguf.to_model_config()?;

// Load all weights (dequantizes to f32)
let weights = load_model_weights(&gguf, &mut reader)?;

// Or load a single tensor
let tensor = gguf.read_tensor_data(&mut reader, &gguf.tensors[0])?;
```

#### SafeTensors Loading

```rust
use flare_loader::safetensors::SafeTensorsFile;
use std::io::Cursor;

let data = std::fs::read("model.safetensors")?;
let mut reader = Cursor::new(&data);

let st = SafeTensorsFile::parse_header(&mut reader)?;

// List tensors
for name in st.tensor_names() {
    println!("{}", name);
}

// Read a specific tensor as f32
let tensor = st.read_tensor(&mut reader, "model.embed_tokens.weight")?;
```

### flare-gpu

WebGPU compute backend. Same code runs on native (Vulkan/Metal/DX12) and browser (WebGPU).

```rust
use flare_gpu::WebGpuBackend;

// Async initialization (works on both native and WASM)
let backend = WebGpuBackend::new().await?;

// Access the underlying wgpu device/queue
let device = backend.device();
let queue = backend.queue();
```

WGSL compute shaders are in `flare-gpu/shaders/`:
- `matmul.wgsl` — Tiled 16x16 matrix multiply
- `rmsnorm.wgsl` — Fused RMSNorm + residual add
- `rope.wgsl` — Rotary position embeddings
- `silu_mul.wgsl` — SiLU activation fused with element-wise multiply

### flare-simd

CPU fallback backend. Implements the same `ComputeBackend` trait as flare-gpu.

```rust
use flare_simd::SimdBackend;
use flare_core::model::ComputeBackend;

let backend = SimdBackend::new();
backend.matmul(&a, &b, &mut output);
backend.rmsnorm(&input, &weight, 1e-5, &mut output);
backend.softmax(&mut logits);
```

## Building

```bash
# Native build
cargo build --release

# Run tests
cargo test --workspace

# Build CLI
cargo build --release --bin flare-cli

# Check everything compiles
cargo check --workspace --all-targets

# Lint
cargo clippy --workspace --all-targets -- -D warnings
```

### WASM Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for browser
wasm-pack build flare-web --target web

# Build for Node.js
wasm-pack build flare-web --target nodejs
```

## Project Status

This project is in early development (Phase 1 complete).

- [x] **Phase 1**: CPU inference in WASM — GGUF parser, Llama forward pass, sampling, tokenizer, CLI
- [ ] **Phase 2**: WebGPU acceleration — wire WGSL shaders into the inference pipeline
- [ ] **Phase 3**: Browser polish — progressive loading, Web Workers, Cache API, npm package
- [ ] **Phase 4**: Native server — OpenAI-compatible API, speculative decoding
- [ ] **Phase 5**: Ecosystem — React hooks, model gallery, documentation site

## Supported Architectures

| Architecture | Status | Models |
|-------------|--------|--------|
| Llama | Supported | Llama 3.2 (1B, 3B), TinyLlama |
| Qwen2 | Config parsing | Qwen2.5 (0.5B, 1.5B, 3B) |
| Mistral | Config parsing | Mistral 7B (requires >4GB, native only) |

## License

MIT OR Apache-2.0

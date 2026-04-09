# Flare - WASM-First LLM Inference Engine

## Project Overview
Flare is a pure Rust → WASM LLM inference engine with WebGPU acceleration. Single codebase compiles to both native binary and WASM for browser/edge deployment.

## Key Constraints
- **Public repo** — never commit secrets, API keys, credentials, or internal URLs
- **Never use AI assistant names in commits, PRs, or code comments** — use generic attribution or none
- All code must be memory-safe and avoid `unsafe` blocks unless absolutely necessary with clear justification
- WASM-first: every design decision must consider WASM constraints (no threads by default, no filesystem, limited memory, 4GB max)

## Tech Stack
- **Language**: Rust (edition 2021)
- **GPU**: wgpu crate (compiles to WebGPU in browser, Vulkan/Metal/DX12 native)
- **Shaders**: WGSL compute shaders
- **WASM toolchain**: wasm-pack, wasm-bindgen
- **Browser integration**: web-sys, js-sys
- **Model formats**: GGUF (primary), SafeTensors (secondary)
- **JS/TS**: TypeScript types, npm package via wasm-pack

## Workspace Crates
- `flare-core` — core inference engine (model, tensor, KV cache, sampling, tokenizer)
- `flare-loader` — model weight loading (GGUF, SafeTensors, progressive/streaming)
- `flare-gpu` — WebGPU/wgpu compute backend + WGSL shaders
- `flare-simd` — WASM SIMD128 CPU fallback
- `flare-web` — browser integration (wasm-bindgen, Web Worker, fetch, Cache API)
- `flare-server` — native server with OpenAI-compatible API

## Build Commands
```bash
# Native build
cargo build --release

# WASM build
wasm-pack build flare-web --target web

# Run tests
cargo test --workspace

# Run native CLI example
cargo run --example native-cli

# Check all targets compile
cargo check --workspace --all-targets
```

## Architecture Notes
- Dual compute path: WebGPU (primary) with WASM SIMD128 fallback
- Progressive model loading: stream weights, start inference before full download
- All inference in browser runs in Web Worker (non-blocking UI)
- KV cache uses ring buffer with Q8 quantization to save memory
- WGSL shaders live in `flare-gpu/shaders/`

## Code Style
- Follow standard Rust conventions (rustfmt, clippy)
- No `unwrap()` in library code — use proper error handling with `thiserror`
- Feature flags for conditional compilation: `#[cfg(target_arch = "wasm32")]` for WASM-specific code
- Prefer zero-copy parsing for model formats where possible

## Security
- This is a public repo. Never commit:
  - API keys, tokens, or credentials
  - Internal URLs or infrastructure details
  - User data or PII
- Validate all model weight data (untrusted input)
- No unsafe memory access when processing GGUF/SafeTensors files

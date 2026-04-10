# Contributing to Flare LLM

Thanks for your interest in contributing! This document describes how to set up
the project, run tests, and submit changes.

## Development setup

```bash
git clone https://github.com/sauravpanda/flarellm
cd flarellm

# Build everything
cargo build --release

# Run all tests
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all
```

### WASM build

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the browser bindings
wasm-pack build flare-web --target web
```

### Running the demo

```bash
wasm-pack build flare-web --target web
cd flare-web && python3 -m http.server 8000
# Open http://localhost:8000/demo/
```

## Workspace layout

- `flare-core/` — tensor, model, sampling, generation (`flarellm-core` on crates.io)
- `flare-loader/` — GGUF / SafeTensors loading (`flarellm-loader`)
- `flare-gpu/` — WebGPU compute backend (`flarellm-gpu`)
- `flare-simd/` — WASM SIMD128 backend (`flarellm-simd`)
- `flare-web/` — browser bindings (`flarellm-web`)
- `flare-server/` — native server + CLI (`flarellm-server`)
- `flarellm/` — umbrella crate that re-exports the others

Note: directory names use the legacy `flare-*` prefix; package names use
`flarellm-*` because `flare-core` is taken on crates.io.

## Running benchmarks

```bash
# Microbenchmarks (matmul, sampling)
cargo bench -p flarellm-simd

# CPU matvec benchmark
cargo run -p flarellm-core --example matvec_bench --release

# GPU vs CPU benchmark
cargo run -p flarellm-gpu --example gpu_bench --release

# End-to-end (requires a GGUF file at models/smollm2-135m-instruct-q8_0.gguf)
cargo run -p flarellm-server --example e2e_bench --release

# Append a new entry to BENCHMARK_HISTORY.md
cargo run -p flarellm-server --example e2e_bench --release -- --log
```

## Performance regression policy

If you make a change that affects inference speed, please:
1. Run `cargo run -p flarellm-server --example e2e_bench --release -- --log`
2. Include the new entry in your PR (commit `BENCHMARK_HISTORY.md`)
3. If the change is a regression, explain why it's worth it

The baseline is SmolLM2-135M-Instruct Q8_0 on whatever hardware you have.
See [`BENCHMARK_HISTORY.md`](BENCHMARK_HISTORY.md) for the full performance log.

## Code style

- Standard Rust conventions: `rustfmt` and `clippy` are CI-enforced
- No `unwrap()` in library code — return `Result` and use `?`
- Error types via `thiserror`
- Document public APIs with `///` doc comments
- Mark unsafe blocks with `// SAFETY:` explaining the invariants

## SIMD code

- ARM NEON paths use `cfg(target_arch = "aarch64")` (always available)
- x86 AVX2 paths use runtime detection via `is_x86_feature_detected!`
- Always compare against the scalar reference (`matvec_scalar`) in tests
- The `test_matvec_simd_matches_scalar` test catches regressions across many sizes

## Submitting changes

1. Create a branch (`git checkout -b feat/my-change`)
2. Make your changes, run tests, format
3. Commit with a descriptive message focused on **why**
4. Push and open a PR
5. CI must be green (check, test, fmt, clippy, wasm, doc)

For commit messages, please **don't** include AI assistant names — this is a
public repo and we use generic attribution.

## Reporting issues

Open an issue with:
- What you expected
- What actually happened
- Minimal reproduction (model, prompt, command)
- Hardware (chip, OS, Rust version)

## License

By contributing, you agree that your contributions will be licensed under
MIT OR Apache-2.0.

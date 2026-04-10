# Flare Browser Demo

A minimal HTML page that loads the WASM build and runs inference entirely in the browser.

## Quick start

```bash
# 1. Build the WASM package
wasm-pack build flare-web --target web

# 2. Serve the demo (any HTTP server works)
cd flare-web
python3 -m http.server 8000

# 3. Open http://localhost:8000/demo/
```

## What it does

1. Loads the `flare_web` WASM module
2. Detects WebGPU availability
3. Lets you upload a GGUF file via `<input type="file">`
4. Calls `FlareEngine.load(bytes)` to parse and load the model
5. Calls `engine.generate_tokens(prompt, 30)` to generate from BOS

## Limitations

- The demo shows raw token IDs (no BPE detokenization in JS yet)
- Loads the entire model into memory at once (no progressive loading yet)
- CPU only (WebGPU compute path not yet wired through wasm-bindgen)

## Try it with SmolLM2-135M Q8_0

```bash
# Download from HuggingFace (~138MB)
curl -L "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" \
  -o SmolLM2-135M-Instruct-Q8_0.gguf
```

Then drag the file into the demo page.

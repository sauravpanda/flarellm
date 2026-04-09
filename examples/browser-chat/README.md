# Browser Chat Demo

A minimal single-file chat UI that validates the Flare WASM integration and
WebGPU detection in the browser. Model inference is not yet wired up — the demo
shows that the WASM module loads, `init()` succeeds, and `webgpu_available()`
reports correctly.

## Quick Start

1. **Build the WASM package** (from the repo root):

   ```bash
   wasm-pack build flare-web --target web
   ```

2. **Serve the repo** (a local HTTP server is required for ES module imports):

   ```bash
   python3 -m http.server 8000
   ```

3. **Open the demo**:

   ```
   http://localhost:8000/examples/browser-chat/
   ```

## What It Does

- Loads the Flare WASM module (`flare_web.js`).
- Calls `init()` to initialize the engine.
- Calls `webgpu_available()` and `device_info()` and displays the results in a
  status bar.
- Provides a chat input — messages get a placeholder response until real
  inference is connected.

## Requirements

- A browser that supports ES modules (all modern browsers).
- WebGPU support is detected but not required — the demo still loads without it.
- The WASM package must be built before opening the page.

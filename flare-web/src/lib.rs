//! Browser integration for Flare LLM.
//!
//! Provides WASM-bindgen exports for loading GGUF models from JS, running
//! inference, and detecting WebGPU. Designed to be built with `wasm-pack`.
//!
//! # Quick start (JS)
//!
//! ```javascript
//! import init, { FlareEngine, webgpu_available } from './pkg/flare_web.js';
//!
//! await init();
//! console.log('WebGPU:', webgpu_available());
//!
//! // Load model from a fetched ArrayBuffer
//! const response = await fetch('model.gguf');
//! const bytes = new Uint8Array(await response.arrayBuffer());
//! const engine = FlareEngine.load(bytes);
//!
//! // Generate
//! const tokens = engine.generate_tokens(new Uint32Array([1, 2, 3]), 50);
//! ```

use std::io::Cursor;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;
use wasm_bindgen::prelude::*;

/// Set up better panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Check if WebGPU is available in the current browser.
#[wasm_bindgen]
pub fn webgpu_available() -> bool {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return false,
    };

    let navigator: JsValue = window.navigator().into();
    js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

/// Get basic device info as a JSON string.
#[wasm_bindgen]
pub fn device_info() -> String {
    let ua: String = web_sys::window()
        .map(|w| w.navigator())
        .and_then(|n| n.user_agent().ok())
        .unwrap_or_default();

    format!(
        r#"{{"webgpu": {}, "userAgent": "{}"}}"#,
        webgpu_available(),
        ua.replace('"', r#"\""#)
    )
}

/// Flare LLM inference engine, exported to JS.
///
/// Holds a loaded model and runs greedy/sampled token generation.
#[wasm_bindgen]
pub struct FlareEngine {
    model: Model,
}

#[wasm_bindgen]
impl FlareEngine {
    /// Load a GGUF model from a Uint8Array of bytes (e.g. from `fetch`).
    #[wasm_bindgen]
    pub fn load(gguf_bytes: &[u8]) -> Result<FlareEngine, JsError> {
        let mut reader = Cursor::new(gguf_bytes);
        let gguf = GgufFile::parse_header(&mut reader)
            .map_err(|e| JsError::new(&format!("GGUF parse error: {e}")))?;
        let config = gguf
            .to_model_config()
            .map_err(|e| JsError::new(&format!("Model config error: {e}")))?;
        let weights = load_model_weights(&gguf, &mut reader)
            .map_err(|e| JsError::new(&format!("Weight load error: {e}")))?;

        Ok(FlareEngine {
            model: Model::new(config, weights),
        })
    }

    /// Reset the KV cache (start a new conversation).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Get the vocabulary size of the loaded model.
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> u32 {
        self.model.config().vocab_size as u32
    }

    /// Get the number of layers.
    #[wasm_bindgen(getter)]
    pub fn num_layers(&self) -> u32 {
        self.model.config().num_layers as u32
    }

    /// Get the hidden dimension.
    #[wasm_bindgen(getter)]
    pub fn hidden_dim(&self) -> u32 {
        self.model.config().hidden_dim as u32
    }

    /// Generate `max_tokens` tokens starting from `prompt_tokens` (greedy).
    /// Returns a Uint32Array of generated token IDs.
    #[wasm_bindgen]
    pub fn generate_tokens(&mut self, prompt_tokens: &[u32], max_tokens: u32) -> Vec<u32> {
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut self.model, params);
        gen.generate(
            prompt_tokens,
            max_tokens as usize,
            None,
            || 0.5,
            |_, _| true,
        )
    }

    /// Generate with sampling parameters. Uses a fixed RNG seed for reproducibility
    /// (browser apps should pass their own RNG via JS-side state).
    #[wasm_bindgen]
    pub fn generate_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Vec<u32> {
        let params = SamplingParams {
            temperature,
            top_p,
            top_k: 40,
            repeat_penalty: 1.1,
        };
        let mut gen = Generator::new(&mut self.model, params);
        // Simple LCG for browser-side RNG (deterministic per call)
        let mut state: u32 = 0x12345678;
        let mut rng = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32)
        };
        gen.generate(
            prompt_tokens,
            max_tokens as usize,
            None,
            &mut rng,
            |_, _| true,
        )
    }
}

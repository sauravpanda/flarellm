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
//! // Apply chat template then generate
//! const prompt = engine.apply_chat_template('What is Rust?', '');
//! const ids = tokenizer.encode(prompt);
//! const tokens = engine.generate_tokens(ids, 50);
//! ```

use std::io::Cursor;

use flare_core::chat::{ChatMessage, ChatTemplate, Role};
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::{self, SamplingParams};
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_loader::gguf::GgufFile;
use flare_loader::weights::{load_model_weights, load_model_weights_with_progress};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

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

/// Detect chat template from GGUF metadata.
/// Prefers the `tokenizer.chat_template` Jinja string when present; falls back
/// to architecture-based detection.
fn detect_chat_template(gguf: &GgufFile) -> ChatTemplate {
    let arch = gguf.architecture().unwrap_or("unknown");
    if let Some(tmpl_str) = gguf
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_str())
    {
        ChatTemplate::from_gguf_template(tmpl_str, arch)
    } else {
        ChatTemplate::from_architecture(arch)
    }
}

/// Flare LLM inference engine, exported to JS.
///
/// Holds a loaded model and runs greedy/sampled token generation.
/// The detected chat template is available via `chat_template_name` and
/// `apply_chat_template` so the browser demo can format prompts correctly
/// for instruction-tuned models.
#[wasm_bindgen]
pub struct FlareEngine {
    model: Model,
    chat_template: ChatTemplate,
    /// EOS token ID from GGUF metadata; generation stops when this token is produced.
    eos_token_id: Option<u32>,
    // --- Token-by-token streaming state ---
    /// Last token fed to the model (updated by begin_stream / next_token).
    stream_last_token: u32,
    /// Current sequence position (prompt length + tokens generated so far).
    stream_pos: usize,
    /// Remaining budget of tokens to generate in the current stream.
    stream_remaining: usize,
    /// Whether the current stream has finished (EOS reached, budget exhausted, or stopped).
    stream_done: bool,
}

#[wasm_bindgen]
impl FlareEngine {
    /// Load a GGUF model from a Uint8Array of bytes (e.g. from `fetch`).
    #[wasm_bindgen]
    pub fn load(gguf_bytes: &[u8]) -> Result<FlareEngine, JsError> {
        let mut reader = Cursor::new(gguf_bytes);
        let gguf = GgufFile::parse_header(&mut reader)
            .map_err(|e| JsError::new(&format!("GGUF parse error: {e}")))?;
        let chat_template = detect_chat_template(&gguf);
        let eos_token_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let config = gguf
            .to_model_config()
            .map_err(|e| JsError::new(&format!("Model config error: {e}")))?;
        let weights = load_model_weights(&gguf, &mut reader)
            .map_err(|e| JsError::new(&format!("Weight load error: {e}")))?;

        Ok(FlareEngine {
            model: Model::new(config, weights),
            chat_template,
            eos_token_id,
            stream_last_token: 0,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
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

    /// Name of the auto-detected chat template (e.g. `"ChatML"`, `"Llama3"`,
    /// `"Alpaca"`, `"Raw"`).  Use this to display the template in the UI and
    /// decide whether to call `apply_chat_template` before encoding.
    #[wasm_bindgen(getter)]
    pub fn chat_template_name(&self) -> String {
        match self.chat_template {
            ChatTemplate::Llama3 => "Llama3".to_string(),
            ChatTemplate::ChatML => "ChatML".to_string(),
            ChatTemplate::Alpaca => "Alpaca".to_string(),
            ChatTemplate::Raw => "Raw".to_string(),
        }
    }

    /// Format a user message (and optional system prompt) using the model's
    /// auto-detected chat template.  Returns the formatted prompt string ready
    /// to be passed to `FlareTokenizer.encode()`.
    ///
    /// Pass an empty string for `system_message` to omit the system turn.
    ///
    /// # JS example
    /// ```javascript
    /// const prompt = engine.apply_chat_template(
    ///   'Explain quantum computing in simple terms.',
    ///   'You are a helpful assistant.'
    /// );
    /// const ids = tokenizer.encode(prompt);
    /// const output = engine.generate_tokens(ids, 128);
    /// ```
    #[wasm_bindgen]
    pub fn apply_chat_template(&self, user_message: &str, system_message: &str) -> String {
        let mut messages = Vec::new();
        if !system_message.is_empty() {
            messages.push(ChatMessage {
                role: Role::System,
                content: system_message.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: Role::User,
            content: user_message.to_string(),
        });
        self.chat_template.apply(&messages)
    }

    /// EOS (end of sequence) token ID from the GGUF model metadata, if present.
    /// Generation stops automatically when this token is produced.
    #[wasm_bindgen(getter)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    // -----------------------------------------------------------------------
    // Token-by-token streaming API
    // -----------------------------------------------------------------------

    /// Prepare for token-by-token streaming.
    ///
    /// Runs the prefill pass on `prompt_tokens`, then initialises internal
    /// state so that subsequent calls to `next_token()` each produce one
    /// output token.  Call `engine.reset()` before `begin_stream()` to start
    /// a fresh conversation.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// engine.begin_stream(promptIds, 128);
    /// function tick() {
    ///   const id = engine.next_token();
    ///   if (id === undefined) { /* done */ return; }
    ///   output.textContent += tokenizer.decode_one(id);
    ///   requestAnimationFrame(tick);   // yield to browser, then continue
    /// }
    /// requestAnimationFrame(tick);
    /// ```
    #[wasm_bindgen]
    pub fn begin_stream(&mut self, prompt_tokens: &[u32], max_tokens: u32) {
        let mut pos = 0usize;
        for &tok in prompt_tokens {
            self.model.forward(tok, pos);
            pos += 1;
        }
        self.stream_pos = pos;
        self.stream_last_token = *prompt_tokens.last().unwrap_or(&0);
        self.stream_remaining = max_tokens as usize;
        self.stream_done = false;
    }

    /// Generate and return the next token ID, or `undefined` when the stream
    /// is complete (EOS reached, `max_tokens` exhausted, or `stop_stream()`
    /// was called).
    ///
    /// Uses greedy (temperature=0) sampling.  Call this inside
    /// `requestAnimationFrame` so the browser can update the DOM between
    /// tokens and the page remains responsive.
    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Option<u32> {
        if self.stream_done || self.stream_remaining == 0 {
            self.stream_done = true;
            return None;
        }

        let logits_tensor = self.model.forward(self.stream_last_token, self.stream_pos);
        let token_id = sampling::sample_greedy(logits_tensor.data());

        self.stream_last_token = token_id;
        self.stream_pos += 1;
        self.stream_remaining -= 1;

        if self.eos_token_id == Some(token_id) {
            self.stream_done = true;
            return None;
        }

        Some(token_id)
    }

    /// Signal the current stream to stop after the next `next_token()` call.
    /// The JS Stop button should call this, then wait for `next_token()` to
    /// return `undefined` before updating the UI.
    #[wasm_bindgen]
    pub fn stop_stream(&mut self) {
        self.stream_done = true;
    }

    /// Whether the current stream has finished.
    #[wasm_bindgen(getter)]
    pub fn stream_done(&self) -> bool {
        self.stream_done
    }

    // -----------------------------------------------------------------------
    // Batch generation (returns all tokens at once)
    // -----------------------------------------------------------------------

    /// Generate `max_tokens` tokens starting from `prompt_tokens` (greedy).
    /// Stops early at EOS. Returns a Uint32Array of generated token IDs.
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
            self.eos_token_id,
            || 0.5,
            |_, _| true,
        )
    }

    /// Generate with sampling parameters. Stops early at EOS.
    /// Uses a fixed LCG RNG seed for reproducibility.
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
            self.eos_token_id,
            &mut rng,
            |_, _| true,
        )
    }
}

/// Progressive loader that fetches a GGUF model from a URL with streaming
/// download progress.
///
/// This enables the browser demo to show download progress as the model
/// arrives over the network, then layer-loading progress as the model is
/// parsed. For a 500MB Q4 model the download phase dominates; displaying
/// progress prevents the page from appearing frozen.
///
/// # JS example
///
/// ```javascript
/// const loader = new FlareProgressiveLoader('https://example.com/model.gguf');
/// const engine = await loader.load((loaded, total) => {
///   const pct = total > 0 ? Math.round(loaded / total * 100) : 0;
///   progressBar.value = pct / 100;
///   statusText.textContent = `Downloading… ${pct}%`;
/// });
/// ```
#[wasm_bindgen]
pub struct FlareProgressiveLoader {
    url: String,
}

#[wasm_bindgen]
impl FlareProgressiveLoader {
    /// Create a loader for the given model URL.
    #[wasm_bindgen(constructor)]
    pub fn new(url: &str) -> FlareProgressiveLoader {
        FlareProgressiveLoader {
            url: url.to_string(),
        }
    }

    /// Fetch the model from the URL, calling `on_progress(loaded_bytes, total_bytes)`
    /// as each chunk arrives, then parse and return a `FlareEngine`.
    ///
    /// `total_bytes` is 0 when the server does not send a `Content-Length` header
    /// (e.g. when the response is gzip-compressed or chunked).
    #[wasm_bindgen]
    pub async fn load(&self, on_progress: js_sys::Function) -> Result<FlareEngine, JsError> {
        let window = web_sys::window().ok_or_else(|| JsError::new("no window object"))?;

        // Kick off the fetch
        let resp_promise = window.fetch_with_str(&self.url);
        let resp_value = wasm_bindgen_futures::JsFuture::from(resp_promise)
            .await
            .map_err(|e| JsError::new(&format!("fetch failed: {e:?}")))?;

        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| JsError::new("response cast failed"))?;

        if !resp.ok() {
            return Err(JsError::new(&format!(
                "HTTP error {} for {}",
                resp.status(),
                self.url
            )));
        }

        // Content-Length is optional; 0 means unknown
        let total_bytes: u32 = resp
            .headers()
            .get("content-length")
            .unwrap_or(None)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        // Stream the body in chunks so we can report progress
        let body = resp
            .body()
            .ok_or_else(|| JsError::new("response has no body"))?;

        let reader: web_sys::ReadableStreamDefaultReader = body
            .get_reader()
            .dyn_into()
            .map_err(|_| JsError::new("ReadableStreamDefaultReader cast failed"))?;

        let mut bytes: Vec<u8> = if total_bytes > 0 {
            Vec::with_capacity(total_bytes as usize)
        } else {
            Vec::new()
        };

        loop {
            let chunk = wasm_bindgen_futures::JsFuture::from(reader.read())
                .await
                .map_err(|e| JsError::new(&format!("stream read error: {e:?}")))?;

            let done = js_sys::Reflect::get(&chunk, &JsValue::from_str("done"))
                .map_err(|_| JsError::new("missing 'done' in stream result"))?
                .as_bool()
                .unwrap_or(false);

            if done {
                break;
            }

            let value = js_sys::Reflect::get(&chunk, &JsValue::from_str("value"))
                .map_err(|_| JsError::new("missing 'value' in stream result"))?;

            let typed_array = js_sys::Uint8Array::new(&value);
            bytes.extend_from_slice(&typed_array.to_vec());

            // Fire progress callback: (loaded, total)
            let _ = on_progress.call2(
                &JsValue::NULL,
                &JsValue::from(bytes.len() as u32),
                &JsValue::from(total_bytes),
            );
        }

        // Signal download complete (in case total was unknown)
        let final_len = bytes.len() as u32;
        let _ = on_progress.call2(
            &JsValue::NULL,
            &JsValue::from(final_len),
            &JsValue::from(final_len),
        );

        // Parse GGUF and build the model
        let mut cursor = Cursor::new(bytes);
        let gguf = GgufFile::parse_header(&mut cursor)
            .map_err(|e| JsError::new(&format!("GGUF parse error: {e}")))?;
        let chat_template = detect_chat_template(&gguf);
        let eos_token_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let config = gguf
            .to_model_config()
            .map_err(|e| JsError::new(&format!("model config error: {e}")))?;

        // Use the progress-aware loader so layer parsing is also trackable in logs
        let weights = load_model_weights_with_progress(&gguf, &mut cursor, |current, total| {
            // Log layer loading progress to browser console (non-blocking)
            let msg = format!("Flare: loading layer {current}/{total}");
            let _ = js_sys::eval(&format!("console.debug({msg:?})"));
        })
        .map_err(|e| JsError::new(&format!("weight load error: {e}")))?;

        Ok(FlareEngine {
            model: Model::new(config, weights),
            chat_template,
            eos_token_id,
            stream_last_token: 0,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
        })
    }
}

/// BPE tokenizer exported to JS for encoding prompts and decoding generated tokens.
///
/// Load from a HuggingFace `tokenizer.json` string, then use `encode` / `decode`
/// in coordination with `FlareEngine` to run full text-in / text-out inference.
///
/// # JS example
///
/// ```javascript
/// const resp = await fetch('tokenizer.json');
/// const json = await resp.text();
/// const tok = FlareTokenizer.from_json(json);
///
/// const ids = tok.encode("Hello, world!");
/// const engine = FlareEngine.load(modelBytes);
/// const out = engine.generate_tokens(ids, 64);
/// console.log(tok.decode(out));
/// ```
#[wasm_bindgen]
pub struct FlareTokenizer {
    inner: BpeTokenizer,
}

#[wasm_bindgen]
impl FlareTokenizer {
    /// Load a tokenizer from the text of a HuggingFace `tokenizer.json` file.
    #[wasm_bindgen]
    pub fn from_json(json: &str) -> Result<FlareTokenizer, JsError> {
        let inner = BpeTokenizer::from_json(json)
            .map_err(|e| JsError::new(&format!("tokenizer load error: {e}")))?;
        Ok(FlareTokenizer { inner })
    }

    /// Encode text to a sequence of token IDs.
    #[wasm_bindgen]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsError> {
        self.inner
            .encode(text)
            .map_err(|e| JsError::new(&format!("encode error: {e}")))
    }

    /// Decode a sequence of token IDs to text.
    #[wasm_bindgen]
    pub fn decode(&self, tokens: &[u32]) -> Result<String, JsError> {
        self.inner
            .decode(tokens)
            .map_err(|e| JsError::new(&format!("decode error: {e}")))
    }

    /// Decode a single token ID to text (useful for streaming output).
    #[wasm_bindgen]
    pub fn decode_one(&self, token_id: u32) -> Result<String, JsError> {
        self.inner
            .decode(&[token_id])
            .map_err(|e| JsError::new(&format!("decode error: {e}")))
    }

    /// BOS (beginning of sequence) token ID, if defined.
    #[wasm_bindgen(getter)]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    /// EOS (end of sequence) token ID, if defined.
    #[wasm_bindgen(getter)]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// Vocabulary size.
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.vocab_size() as u32
    }
}

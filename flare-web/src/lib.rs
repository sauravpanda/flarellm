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
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use flare_core::chat::{ChatMessage, ChatTemplate, Role};
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::{self, SamplingParams};
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_gpu::WebGpuBackend;
use flare_loader::gguf::GgufFile;
use flare_loader::tokenizer::GgufVocab;
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

/// Return the current wall-clock time in milliseconds.
///
/// In WASM uses `performance.now()` for sub-millisecond accuracy.
/// In native builds uses `SystemTime` for coarser but portable timing.
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
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
    /// GGUF vocabulary for token counting (optional: absent for non-GGUF models).
    gguf_vocab: Option<GgufVocab>,
    /// EOS token ID from GGUF metadata; generation stops when this token is produced.
    eos_token_id: Option<u32>,
    /// BOS (beginning of sequence) token ID from GGUF metadata.
    bos_token_id: Option<u32>,
    /// Whether to automatically prepend the BOS token before generation.
    /// Sourced from `tokenizer.ggml.add_bos_token` in GGUF metadata.
    add_bos_token: bool,
    /// Raw Jinja2 chat template string from `tokenizer.chat_template` in GGUF metadata.
    /// `None` if the GGUF file did not include a chat template.
    raw_chat_template: Option<String>,
    // --- Token-by-token streaming state ---
    /// Sampling parameters for the current stream (set by begin_stream_with_params).
    stream_params: SamplingParams,
    /// LCG RNG state for streaming sampling; reset on each begin_stream call.
    stream_rng_state: u32,
    /// Last token fed to the model (updated by begin_stream / next_token).
    stream_last_token: u32,
    /// Current sequence position (prompt length + tokens generated so far).
    stream_pos: usize,
    /// Remaining budget of tokens to generate in the current stream.
    stream_remaining: usize,
    /// Whether the current stream has finished (EOS reached, budget exhausted, or stopped).
    stream_done: bool,
    // --- Per-call performance metrics ---
    /// Milliseconds spent in the prefill phase of the most recent generation call.
    /// For batch generation (`generate_tokens`/`generate_text`) this covers the
    /// full call duration since prefill and decode are not separated internally.
    /// For the streaming API it covers only `begin_stream()`.
    last_prefill_ms: f64,
    /// Milliseconds spent in decode steps of the most recent generation call.
    /// For batch generation this is 0 (the cost is in `last_prefill_ms`).
    /// For the streaming API it accumulates across all `next_token()` calls.
    last_decode_ms: f64,
    /// Number of tokens generated by the most recent generation call
    /// (excludes prompt tokens; excludes the EOS token itself).
    last_tokens_generated: u32,
    // --- Streaming timing accumulator ---
    /// Timestamp (ms) of the start of the current streaming decode phase.
    stream_decode_start_ms: f64,
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
        let bos_token_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let add_bos_token = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let raw_chat_template = gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let gguf_vocab = GgufVocab::from_gguf(&gguf).ok();
        let config = gguf
            .to_model_config()
            .map_err(|e| JsError::new(&format!("Model config error: {e}")))?;
        let weights = load_model_weights(&gguf, &mut reader)
            .map_err(|e| JsError::new(&format!("Weight load error: {e}")))?;

        Ok(FlareEngine {
            model: Model::new(config, weights),
            chat_template,
            gguf_vocab,
            eos_token_id,
            bos_token_id,
            add_bos_token,
            raw_chat_template,
            stream_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            stream_rng_state: 0x12345678,
            stream_last_token: 0,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
            last_prefill_ms: 0.0,
            last_decode_ms: 0.0,
            last_tokens_generated: 0,
            stream_decode_start_ms: 0.0,
        })
    }

    /// Try to initialise the WebGPU compute backend.
    ///
    /// Call this after `load()` to enable GPU-accelerated matrix operations
    /// (matvec, matmul, silu_mul). Falls back silently to CPU if WebGPU is
    /// unavailable or adapter request fails.
    ///
    /// Returns `true` if a GPU backend was successfully initialised.
    ///
    /// ```javascript
    /// const engine = FlareEngine.load(bytes);
    /// const gpuEnabled = await engine.init_gpu();
    /// console.log('GPU:', gpuEnabled);
    /// ```
    #[wasm_bindgen]
    pub async fn init_gpu(&mut self) -> bool {
        if !webgpu_available() {
            return false;
        }
        match WebGpuBackend::new().await {
            Ok(gpu) => {
                self.model.set_backend(Box::new(gpu));
                true
            }
            Err(_) => false,
        }
    }

    /// Load raw quantized weights from GGUF bytes so the GPU fused
    /// dequant+matvec kernels can be used during inference.
    ///
    /// Call this **after** `init_gpu()` so the backend is set before the raw
    /// weights are attached.  The method is a no-op (returns `false`) if a
    /// layer's weights are in an unsupported quantization format — the engine
    /// continues to work using the f32 path loaded at `FlareEngine.load()`.
    ///
    /// Returns `true` if all layers were loaded successfully, `false` if any
    /// layer fell back to the f32 path.
    ///
    /// ```javascript
    /// const engine = FlareEngine.load(bytes);
    /// await engine.init_gpu();
    /// const ok = engine.load_raw_weights(bytes);
    /// console.log('Raw weights loaded:', ok);
    /// ```
    #[wasm_bindgen]
    pub fn load_raw_weights(&mut self, gguf_bytes: &[u8]) -> bool {
        let mut reader = Cursor::new(gguf_bytes);
        let gguf = match GgufFile::parse_header(&mut reader) {
            Ok(g) => g,
            Err(_) => return false,
        };
        let num_layers = self.model.config().num_layers;
        let mut raw_layers = Vec::with_capacity(num_layers);
        let mut all_ok = true;

        for layer_idx in 0..num_layers {
            match gguf.load_raw_layer_weights(&mut reader, layer_idx) {
                Ok(Some(rw)) => raw_layers.push(rw),
                _ => {
                    all_ok = false;
                    // Stop trying — partial raw weights are not supported.
                    break;
                }
            }
        }

        if raw_layers.len() == num_layers {
            self.model.set_raw_weights(raw_layers);
        }

        all_ok
    }

    /// Clear any previously loaded raw quantized weights.
    ///
    /// After calling this the engine uses the f32 dequantized path for all
    /// matrix operations until `load_raw_weights` is called again.
    #[wasm_bindgen]
    pub fn clear_raw_weights(&mut self) {
        self.model.clear_raw_weights();
    }

    /// Returns `true` if raw quantized weights are currently loaded.
    #[wasm_bindgen(getter)]
    pub fn has_raw_weights(&self) -> bool {
        self.model.has_raw_weights()
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
            ChatTemplate::Phi3 => "Phi3".to_string(),
            ChatTemplate::Gemma => "Gemma".to_string(),
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

    /// BOS (beginning of sequence) token ID from the GGUF model metadata, if present.
    /// Some models require this to be prepended to the input token sequence.
    #[wasm_bindgen(getter)]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Whether the model requests automatic BOS token prepending.
    ///
    /// Sourced from `tokenizer.ggml.add_bos_token` in the GGUF metadata.
    /// When `true`, all generation methods (`generate_tokens`, `begin_stream`,
    /// `generate_text`, `generate_stream`) automatically prepend the BOS token
    /// to the input token sequence unless it is already the first token.
    #[wasm_bindgen(getter)]
    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    /// Raw Jinja2 chat template string from the GGUF model metadata, if present.
    ///
    /// This is the `tokenizer.chat_template` field embedded by the model author.
    /// Use this with a JavaScript Jinja2 renderer (e.g. `nunjucks`) for accurate
    /// prompt formatting across all model families, rather than relying on the
    /// simplified built-in `apply_chat_template`.
    ///
    /// Returns `undefined` if the GGUF file did not include a chat template.
    #[wasm_bindgen(getter)]
    pub fn raw_chat_template(&self) -> Option<String> {
        self.raw_chat_template.clone()
    }

    /// Maximum sequence length (context window size) of the loaded model.
    ///
    /// Use this to warn users when their prompt is approaching the limit.
    #[wasm_bindgen(getter)]
    pub fn max_seq_len(&self) -> u32 {
        self.model.config().max_seq_len as u32
    }

    /// Count the number of tokens in `text` using the model's embedded GGUF vocabulary.
    ///
    /// Returns 0 if the model was not loaded from a GGUF file (e.g. SafeTensors only).
    ///
    /// # JS example
    /// ```javascript
    /// const n = engine.count_tokens(textarea.value);
    /// counter.textContent = `${n} / ${engine.max_seq_len} tokens`;
    /// ```
    #[wasm_bindgen]
    pub fn count_tokens(&self, text: &str) -> u32 {
        match &self.gguf_vocab {
            Some(vocab) => vocab.encode(text).len() as u32,
            None => 0,
        }
    }

    /// Encode `text` to token IDs using the embedded GGUF vocabulary.
    ///
    /// Returns an empty array if no GGUF vocab is available.
    ///
    /// Prepend the BOS token to `tokens` if `add_bos_token` is set and the
    /// sequence does not already start with the BOS token ID.
    fn with_bos<'a>(&self, tokens: &'a [u32]) -> std::borrow::Cow<'a, [u32]> {
        if let (true, Some(bos)) = (self.add_bos_token, self.bos_token_id) {
            if tokens.first() != Some(&bos) {
                let mut v = Vec::with_capacity(tokens.len() + 1);
                v.push(bos);
                v.extend_from_slice(tokens);
                return std::borrow::Cow::Owned(v);
            }
        }
        std::borrow::Cow::Borrowed(tokens)
    }

    /// # JS example
    /// ```javascript
    /// const ids = engine.encode_text("Hello, world!");
    /// const output = engine.generate_tokens(ids, 64);
    /// ```
    #[wasm_bindgen]
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        match &self.gguf_vocab {
            Some(vocab) => vocab.encode(text),
            None => Vec::new(),
        }
    }

    /// Decode token IDs to text using the embedded GGUF vocabulary.
    ///
    /// Returns an empty string if no GGUF vocab is available.
    ///
    /// # JS example
    /// ```javascript
    /// const text = engine.decode_ids(generatedIds);
    /// ```
    #[wasm_bindgen]
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        match &self.gguf_vocab {
            Some(vocab) => vocab.decode(ids),
            None => String::new(),
        }
    }

    /// Full text-in / text-out generation using the embedded GGUF vocabulary.
    ///
    /// Encodes `prompt` with the embedded vocab, runs greedy generation for up
    /// to `max_tokens` steps, then decodes the output back to text. Stops
    /// automatically at EOS.
    ///
    /// Returns an empty string if no GGUF vocab is available.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// const response = engine.generate_text("What is Rust?", 128);
    /// output.textContent = response;
    /// ```
    #[wasm_bindgen]
    pub fn generate_text(&mut self, prompt: &str, max_tokens: u32) -> String {
        let raw_tokens = match &self.gguf_vocab {
            Some(vocab) => vocab.encode(prompt),
            None => return String::new(),
        };
        let prompt_tokens = self.with_bos(&raw_tokens);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let eos = self.eos_token_id;
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        let generated = gen.generate(
            &prompt_tokens,
            max_tokens as usize,
            eos,
            || 0.5,
            |_, _| true,
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = generated.len() as u32;
        match &self.gguf_vocab {
            Some(vocab) => vocab.decode(&generated),
            None => String::new(),
        }
    }

    /// Streaming text-in / text-out generation with a per-token JS callback.
    ///
    /// Encodes `prompt` with the embedded GGUF vocabulary, generates up to
    /// `max_tokens` tokens, and calls `on_token(token_str)` with the decoded
    /// text for each token as it is produced.  Returns the number of tokens
    /// generated (excluding any EOS token).
    ///
    /// Returns 0 if no GGUF vocab is available.
    ///
    /// # Note on browser streaming
    /// `on_token` is called synchronously inside WASM, so the browser will
    /// not visually update between tokens.  For visible character-by-character
    /// output, use `begin_stream` + `next_token` with `requestAnimationFrame`.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// let out = '';
    /// const count = engine.generate_stream("What is Rust?", 128, (token) => {
    ///   out += token;
    /// });
    /// output.textContent = out;
    /// ```
    #[wasm_bindgen]
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        on_token: &js_sys::Function,
    ) -> u32 {
        // Encode and generate first, then decode per-token for callbacks.
        let raw_tokens = match &self.gguf_vocab {
            Some(vocab) => vocab.encode(prompt),
            None => return 0,
        };
        let prompt_tokens = self.with_bos(&raw_tokens);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let eos = self.eos_token_id;
        // Generate all tokens — vocab and model are different fields so this
        // is safe to split-borrow after the encode above.
        let generated = {
            let mut gen = Generator::new(&mut self.model, params);
            gen.generate(
                &prompt_tokens,
                max_tokens as usize,
                eos,
                || 0.5,
                |_, _| true,
            )
        };
        let mut count = 0u32;
        if let Some(vocab) = &self.gguf_vocab {
            for id in &generated {
                let token_str = vocab.decode(&[*id]);
                let _ = on_token.call1(&JsValue::NULL, &JsValue::from_str(&token_str));
                count += 1;
            }
        }
        count
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
        // Greedy (temperature=0) — reset params so any previous begin_stream_with_params
        // settings don't bleed into this stream.
        self.stream_params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        self.stream_rng_state = 0x12345678;
        self.begin_stream_impl(prompt_tokens, max_tokens);
    }

    /// Like `begin_stream` but with temperature / top-p sampling.
    ///
    /// `temperature`: 0.0 = greedy, 0.7–1.0 = typical creative range.
    /// `top_p`: nucleus sampling threshold (0.0–1.0); 0.9 is a good default.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// engine.begin_stream_with_params(promptIds, 128, 0.8, 0.9);
    /// function tick() {
    ///   const id = engine.next_token();
    ///   if (id === undefined) return;
    ///   output.textContent += tokenizer.decode_one(id);
    ///   requestAnimationFrame(tick);
    /// }
    /// requestAnimationFrame(tick);
    /// ```
    #[wasm_bindgen]
    pub fn begin_stream_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) {
        self.stream_params = SamplingParams {
            temperature,
            top_p,
            top_k: 40,
            repeat_penalty: 1.1,
        };
        self.stream_rng_state = 0x12345678;
        self.begin_stream_impl(prompt_tokens, max_tokens);
    }

    /// Internal prefill + state initialisation shared by both begin_stream variants.
    fn begin_stream_impl(&mut self, prompt_tokens: &[u32], max_tokens: u32) {
        let effective = self.with_bos(prompt_tokens);
        let t0 = now_ms();
        let mut pos = 0usize;
        for &tok in effective.iter() {
            self.model.forward(tok, pos);
            pos += 1;
        }
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = 0;
        self.stream_decode_start_ms = 0.0;
        self.stream_pos = pos;
        self.stream_last_token = *effective.last().unwrap_or(&0);
        self.stream_remaining = max_tokens as usize;
        self.stream_done = false;
    }

    /// Generate and return the next token ID, or `undefined` when the stream
    /// is complete (EOS reached, `max_tokens` exhausted, or `stop_stream()`
    /// was called).
    ///
    /// Sampling parameters are those set by the most recent `begin_stream` or
    /// `begin_stream_with_params` call.  Call this inside
    /// `requestAnimationFrame` so the browser can update the DOM between
    /// tokens and the page remains responsive.
    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Option<u32> {
        if self.stream_done || self.stream_remaining == 0 {
            self.stream_done = true;
            return None;
        }

        // Start the decode timer on the first call after begin_stream.
        if self.stream_decode_start_ms == 0.0 {
            self.stream_decode_start_ms = now_ms();
        }

        let logits_tensor = self.model.forward(self.stream_last_token, self.stream_pos);
        let token_id = if self.stream_params.temperature == 0.0 {
            sampling::sample_greedy(logits_tensor.data())
        } else {
            let mut logits = logits_tensor.data().to_vec();
            sampling::apply_temperature(&mut logits, self.stream_params.temperature);
            // Advance LCG RNG state for this token.
            self.stream_rng_state = self
                .stream_rng_state
                .wrapping_mul(1664525)
                .wrapping_add(1013904223);
            let rng_val = (self.stream_rng_state as f32) / (u32::MAX as f32);
            sampling::sample_top_p(&logits, self.stream_params.top_p, rng_val)
        };

        self.stream_last_token = token_id;
        self.stream_pos += 1;
        self.stream_remaining -= 1;

        if self.eos_token_id == Some(token_id) {
            self.last_decode_ms = now_ms() - self.stream_decode_start_ms;
            self.stream_done = true;
            return None;
        }

        self.last_tokens_generated += 1;
        self.last_decode_ms = now_ms() - self.stream_decode_start_ms;

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
        let effective = self.with_bos(prompt_tokens);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        let result = gen.generate(
            &effective,
            max_tokens as usize,
            self.eos_token_id,
            || 0.5,
            |_, _| true,
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = result.len() as u32;
        result
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
        let effective = self.with_bos(prompt_tokens);
        let params = SamplingParams {
            temperature,
            top_p,
            top_k: 40,
            repeat_penalty: 1.1,
        };
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        // Simple LCG for browser-side RNG (deterministic per call)
        let mut state: u32 = 0x12345678;
        let mut rng = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32)
        };
        let result = gen.generate(
            &effective,
            max_tokens as usize,
            self.eos_token_id,
            &mut rng,
            |_, _| true,
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = result.len() as u32;
        result
    }

    // -----------------------------------------------------------------------
    // Performance metrics API
    // -----------------------------------------------------------------------

    /// Milliseconds spent in the last prefill (prompt processing) phase.
    ///
    /// For `generate_tokens` / `generate_text` / `generate_with_params` this
    /// covers the entire call (prefill + decode are not separated internally).
    /// For the streaming API (`begin_stream` + `next_token`) this covers only
    /// the `begin_stream()` call.
    #[wasm_bindgen(getter)]
    pub fn last_prefill_ms(&self) -> f64 {
        self.last_prefill_ms
    }

    /// Milliseconds spent in decode steps of the last generation call.
    ///
    /// For batch generation (`generate_tokens` etc.) this is always 0 — see
    /// `last_prefill_ms` for the total time.  For the streaming API this
    /// accumulates across all `next_token()` calls since the last
    /// `begin_stream()`.
    #[wasm_bindgen(getter)]
    pub fn last_decode_ms(&self) -> f64 {
        self.last_decode_ms
    }

    /// Number of tokens generated by the last generation call (excludes prompt
    /// tokens and the EOS token itself).
    #[wasm_bindgen(getter)]
    pub fn last_tokens_generated(&self) -> u32 {
        self.last_tokens_generated
    }

    /// Decode throughput in tokens per second for the last generation call.
    ///
    /// For the streaming API this is calculated from `last_decode_ms`.
    /// For batch generation this is calculated from `last_prefill_ms`
    /// (the total call duration).
    ///
    /// Returns 0.0 if no generation has been run or if timing data is
    /// unavailable.
    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        let ms = if self.last_decode_ms > 0.0 {
            self.last_decode_ms
        } else {
            self.last_prefill_ms
        };
        if ms > 0.0 && self.last_tokens_generated > 0 {
            (self.last_tokens_generated as f64) / (ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Return a JSON string summarising the performance metrics from the last
    /// generation call.
    ///
    /// ```javascript
    /// const perf = JSON.parse(engine.performance_summary());
    /// console.log(`TTFT: ${perf.prefill_ms.toFixed(1)} ms`);
    /// console.log(`Decode: ${perf.tokens_per_second.toFixed(1)} tok/s`);
    /// ```
    #[wasm_bindgen]
    pub fn performance_summary(&self) -> String {
        format!(
            r#"{{"prefill_ms":{:.2},"decode_ms":{:.2},"tokens_generated":{},"tokens_per_second":{:.2}}}"#,
            self.last_prefill_ms,
            self.last_decode_ms,
            self.last_tokens_generated,
            self.tokens_per_second(),
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
        let bos_token_id = gguf
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let add_bos_token = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let raw_chat_template = gguf
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let gguf_vocab = GgufVocab::from_gguf(&gguf).ok();
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
            gguf_vocab,
            eos_token_id,
            bos_token_id,
            add_bos_token,
            raw_chat_template,
            stream_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            stream_rng_state: 0x12345678,
            stream_last_token: 0,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
            last_prefill_ms: 0.0,
            last_decode_ms: 0.0,
            last_tokens_generated: 0,
            stream_decode_start_ms: 0.0,
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

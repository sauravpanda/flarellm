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
use flare_loader::gguf::{GgufFile, MetadataValue};
use flare_loader::tokenizer::GgufVocab;
use flare_loader::weights::{load_model_weights_with_progress, load_model_weights_with_raw};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Set up better panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the rayon thread pool for multi-threaded WASM inference.
///
/// Call this once after WASM init with the desired number of threads
/// (typically `navigator.hardwareConcurrency`). Requires the page to be served
/// with COOP/COEP headers for SharedArrayBuffer access.
///
/// Only available when built with `--features wasm_threads`.
#[cfg(feature = "wasm_threads")]
#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    wasm_bindgen_rayon::init_thread_pool(num_threads);
    Ok(())
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

/// Check if WebNN is available in the current browser.
///
/// WebNN (`navigator.ml`) exposes neural-network acceleration through
/// platform NPUs/DSPs. This is a foundation check so JS code can decide
/// whether to build a WebNN graph from exported weights.
#[wasm_bindgen]
pub fn supports_webnn() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w.navigator(), &"ml".into()).ok())
        .is_some_and(|ml| !ml.is_undefined() && !ml.is_null())
}

/// Check if the browser exposes the Web Speech API for speech recognition.
///
/// This probes `window.SpeechRecognition` and the WebKit-prefixed
/// `window.webkitSpeechRecognition`. Returning `true` means the demo voice
/// mode can capture microphone input and produce transcripts through the
/// platform speech engine. This is a foundation for the voice pipeline
/// (issue #395); a fully offline path will eventually run Whisper in WASM.
#[wasm_bindgen]
pub fn supports_speech_recognition() -> bool {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w, &"SpeechRecognition".into())
                .or_else(|_| js_sys::Reflect::get(&w, &"webkitSpeechRecognition".into()))
                .ok()
        })
        .is_some_and(|sr| !sr.is_undefined() && !sr.is_null())
}

/// Check if the browser exposes the Web Speech API for speech synthesis.
///
/// Returns `true` when `window.speechSynthesis` is available, enabling the
/// demo voice mode to speak model responses. A fully offline path will
/// eventually run a neural TTS model in WASM.
#[wasm_bindgen]
pub fn supports_speech_synthesis() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"speechSynthesis".into()).ok())
        .is_some_and(|ss| !ss.is_undefined() && !ss.is_null())
}

/// Check if WebTransport is available in the current browser.
///
/// WebTransport (`window.WebTransport`) is a modern transport API built on
/// HTTP/3 QUIC streams. It allows opening multiple parallel bidirectional
/// streams to the same origin with lower head-of-line blocking than fetch().
/// Useful for progressive model loading where different byte ranges of the
/// GGUF file can be downloaded concurrently.
///
/// Note: actually using WebTransport for parallel range downloads requires
/// server-side support (HTTP/3 endpoint that accepts byte-range requests
/// on streams). This check only reports browser capability — the JS loader
/// will fall back to `fetch()` when the server does not cooperate.
#[wasm_bindgen]
pub fn supports_webtransport() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"WebTransport".into()).ok())
        .is_some_and(|wt| !wt.is_undefined() && !wt.is_null())
}

/// Check if this WASM build was compiled with relaxed SIMD support.
///
/// Relaxed SIMD provides hardware-specific faster operations like fused
/// multiply-add (`f32x4_relaxed_madd`) that map directly to ARM NEON and
/// x86 SSE/AVX FMA instructions. When enabled, matvec operations use FMA
/// for ~15-30% speedup.
///
/// This is a compile-time feature: the WASM binary either includes relaxed
/// SIMD instructions or it does not. The browser validates them at module
/// load time, so if this module loaded successfully and returns `true`,
/// relaxed SIMD is active.
#[wasm_bindgen]
pub fn supports_relaxed_simd() -> bool {
    cfg!(feature = "relaxed_simd")
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

/// Bare `fn` pointer wrapper around [`now_ms`] for `Model::enable_prefill_profiling`,
/// which requires a `fn() -> f64` (not a closure or method).
fn now_ms_f64() -> f64 {
    now_ms()
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

/// Serialise a single GGUF metadata value to a JSON fragment.
///
/// Returns `None` for large arrays (> 64 entries) or arrays whose elements
/// cannot be serialised, so those keys are silently omitted from the output.
fn metadata_value_to_json(val: &MetadataValue) -> Option<String> {
    match val {
        MetadataValue::Uint8(v) => Some(v.to_string()),
        MetadataValue::Int8(v) => Some(v.to_string()),
        MetadataValue::Uint16(v) => Some(v.to_string()),
        MetadataValue::Int16(v) => Some(v.to_string()),
        MetadataValue::Uint32(v) => Some(v.to_string()),
        MetadataValue::Int32(v) => Some(v.to_string()),
        MetadataValue::Uint64(v) => Some(v.to_string()),
        MetadataValue::Int64(v) => Some(v.to_string()),
        MetadataValue::Float32(v) => {
            if v.is_nan() || v.is_infinite() {
                Some("null".to_string())
            } else {
                Some(v.to_string())
            }
        }
        MetadataValue::Float64(v) => {
            if v.is_nan() || v.is_infinite() {
                Some("null".to_string())
            } else {
                Some(v.to_string())
            }
        }
        MetadataValue::Bool(v) => Some(if *v { "true" } else { "false" }.to_string()),
        MetadataValue::String(s) => {
            let escaped = s
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            Some(format!("\"{}\"", escaped))
        }
        MetadataValue::Array(items) => {
            // Skip large arrays to keep the JSON payload small.
            if items.len() > 64 {
                return None;
            }
            let parts: Vec<String> = items.iter().filter_map(metadata_value_to_json).collect();
            if parts.len() == items.len() {
                Some(format!("[{}]", parts.join(",")))
            } else {
                None
            }
        }
    }
}

/// Build a JSON object string from GGUF metadata.
///
/// Keys containing `"tokens"`, `"merges"`, `"scores"`, or `"added_tokens"` are
/// skipped because they hold large vocabulary arrays that would make the payload
/// impractical.  The remaining scalar and small-array values are serialised
/// in sorted key order for stable output.
fn build_metadata_json(gguf: &GgufFile) -> String {
    const SKIP: &[&str] = &["tokens", "merges", "scores", "added_tokens"];

    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();

    let mut pairs: Vec<String> = Vec::new();
    for key in keys {
        if SKIP.iter().any(|p| key.contains(p)) {
            continue;
        }
        let val = &gguf.metadata[key];
        if let Some(json_val) = metadata_value_to_json(val) {
            let escaped_key = key.replace('\\', "\\\\").replace('"', "\\\"");
            pairs.push(format!("\"{}\":{}", escaped_key, json_val));
        }
    }

    format!("{{{}}}", pairs.join(","))
}

/// Compute top-N log-probabilities from a raw logit vector.
///
/// Returns an interleaved `Vec<f32>` with `n * 2` elements laid out as
/// `[token_id_0 as f32, log_prob_0, token_id_1 as f32, log_prob_1, ...]`
/// sorted by descending log-probability.  If `n == 0` or `logits` is empty,
/// returns an empty vector.
fn compute_top_logprobs(logits: &[f32], n: usize) -> Vec<f32> {
    if n == 0 || logits.is_empty() {
        return Vec::new();
    }
    // Numerically-stable softmax.
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    // log(softmax(x_i)) = x_i - max - log(sum(exp(x_j - max)))
    let log_sum = sum.ln();
    let mut candidates: Vec<(u32, f32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (i as u32, e.ln() - log_sum))
        .collect();
    // Sort descending by log-prob.
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(n);
    let mut out = Vec::with_capacity(n * 2);
    for (id, lp) in candidates {
        out.push(id as f32);
        out.push(lp);
    }
    out
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
    /// Architecture name from `general.architecture` in GGUF metadata (e.g. `"llama"`).
    architecture: String,
    /// Model display name from `general.name` in GGUF metadata, or empty string if absent.
    model_name: String,
    /// Number of tokens consumed in the current KV-cache session (prompt + generated).
    /// Updated after every generation call; reset to 0 by `engine.reset()`.
    kv_pos: usize,
    // --- Token-by-token streaming state ---
    /// Sampling parameters for the current stream (set by begin_stream_with_params).
    stream_params: SamplingParams,
    /// LCG RNG state for streaming sampling; reset on each begin_stream call.
    stream_rng_state: u32,
    /// Last token fed to the model (updated by begin_stream / next_token).
    stream_last_token: u32,
    /// Rolling window of recent token IDs for repetition penalty.
    /// Seeded from the tail of the prompt and extended with each generated token.
    stream_recent_tokens: Vec<u32>,
    /// Maximum number of recent tokens tracked for repetition penalty.
    /// 0 disables repetition penalty entirely. Default: 64.
    repeat_last_n: usize,
    /// Current sequence position (prompt length + tokens generated so far).
    stream_pos: usize,
    /// Remaining budget of tokens to generate in the current stream.
    stream_remaining: usize,
    /// Whether the current stream has finished (EOS reached, budget exhausted, or stopped).
    stream_done: bool,
    /// Why the most-recent stream stopped. One of `"eos"`, `"length"`,
    /// `"stop_sequence"`, `"user"`, or `""` (stream still running / not started).
    stream_stop_reason: String,
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
    // --- Stop sequences ---
    /// Strings that halt generation when they appear in the decoded output.
    stop_sequences: Vec<String>,
    /// Decoded text accumulated during the current streaming session, used to
    /// match against `stop_sequences`.  Cleared at each `begin_stream` call.
    stream_text_accum: String,
    // --- RNG seed ---
    /// User-configurable LCG seed applied to the next generation call.
    /// Defaults to `0x12345678`; reset to default by `reset()`.
    rng_seed: u32,
    // --- GGUF metadata ---
    /// Pre-built JSON string of scalar GGUF metadata (large vocabulary arrays excluded).
    /// Empty string `"{}"` for non-GGUF models.
    metadata_json: String,
    // --- Last-step logits ---
    /// Raw pre-temperature logits from the most recent forward pass.
    /// Populated by `next_token()` and the batch `generate_*` methods.
    /// Empty before any inference; cleared by `reset()`.
    last_logits: Vec<f32>,
    // --- Top-N logprobs ---
    /// How many top log-probability entries to capture after each forward pass.
    /// 0 = disabled (default).
    top_logprobs_n: u32,
    /// Interleaved `[token_id_as_f32, log_prob, ...]` for the last token.
    /// Length is `top_logprobs_n * 2`. Empty when disabled or before inference.
    top_logprobs_data: Vec<f32>,
    // --- UTF-8 streaming accumulator ---
    /// Accumulates raw bytes from consecutive `<0xXX>` byte-level tokens so that
    /// multi-byte UTF-8 sequences (e.g. CJK, emoji) are reassembled correctly
    /// before being returned to the caller.  Cleared by `reset()`.
    utf8_byte_buf: Vec<u8>,
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
        let architecture = gguf.architecture().unwrap_or("unknown").to_string();
        let model_name = gguf
            .metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let gguf_vocab = GgufVocab::from_gguf(&gguf).ok();
        let metadata_json = build_metadata_json(&gguf);
        let config = gguf
            .to_model_config()
            .map_err(|e| JsError::new(&format!("Model config error: {e}")))?;
        // Single-pass load: dequantized f32 tensors and raw quantized bytes
        // come back from one read of the tensor data region, so quantized
        // models light up the WASM SIMD128 matvec path without paying for a
        // second full pass over weight bytes.
        let (weights, raw_layers) = load_model_weights_with_raw(&gguf, &mut reader)
            .map_err(|e| JsError::new(&format!("Weight load error: {e}")))?;

        let mut model = Model::new(config.clone(), weights);

        if let Some(raw_layers) = raw_layers {
            model.set_raw_weights(raw_layers);
        } else {
            web_sys::console::log_1(
                &"flare: raw quantized weights not available, using f32 path".into(),
            );
        }

        // Warm-up: run a throwaway forward(0, 0) + KV reset so the decode
        // path's hot kernels are fully tier-up'd and weight pages are resident
        // before the first real inference.
        //
        // I initially removed this in 0.2.6 thinking batched prefill would
        // cover it — it doesn't.  Dropping warmup regressed SmolLM2-135M
        // decode from a stable 72 tok/s (0.2.5) to 40 tok/s averaged with a
        // 20-73 range (0.2.6): the first few decode calls run cold and drag
        // the average down.  Load-time cost is ~500 ms; paid once per
        // session, saves ~30 tok/s on every decoded token — clear net win.
        model.warmup();

        Ok(FlareEngine {
            model,
            chat_template,
            gguf_vocab,
            eos_token_id,
            bos_token_id,
            add_bos_token,
            raw_chat_template,
            architecture,
            model_name,
            kv_pos: 0,
            stream_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            stream_rng_state: 0x12345678,
            stream_last_token: 0,
            stream_recent_tokens: Vec::new(),
            repeat_last_n: 64,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
            stream_stop_reason: String::new(),
            last_prefill_ms: 0.0,
            last_decode_ms: 0.0,
            last_tokens_generated: 0,
            stream_decode_start_ms: 0.0,
            stop_sequences: Vec::new(),
            stream_text_accum: String::new(),
            rng_seed: 0x12345678,
            metadata_json,
            last_logits: Vec::new(),
            top_logprobs_n: 0,
            top_logprobs_data: Vec::new(),
            utf8_byte_buf: Vec::new(),
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
                // Upload the raw quantized weights (populated in `load`) to
                // GPU buffers so forward() takes the fused single-encoder
                // path instead of falling back to CPU SIMD.  Without this,
                // the GPU backend is active but `has_gpu_weights() == false`
                // and every matmul still runs on the CPU — silently wasting
                // the WebGPU init.  No-ops if raw weights aren't present or
                // the backend doesn't support GPU forward.
                self.model.upload_weights_to_gpu();
                true
            }
            Err(_) => false,
        }
    }

    /// Initialise the WebGPU backend using previously serialised pipeline cache
    /// bytes (from `engine.pipeline_cache_data()`).
    ///
    /// On backends that support driver-managed pipeline caches (Vulkan native),
    /// this allows the driver to reuse compiled GPU machine code from a previous
    /// run, eliminating cold-start shader recompilation (typically 100ms–2s).
    ///
    /// On unsupported backends (WebGPU, Metal, DX12) this behaves identically to
    /// `init_gpu()` — the cache bytes are silently ignored.
    ///
    /// ```javascript
    /// const cached = localStorage.getItem('flare-pipeline-cache');
    /// const cacheBytes = cached ? new Uint8Array(JSON.parse(cached)) : new Uint8Array();
    /// await engine.init_gpu_with_cache(cacheBytes);
    /// // After inference, persist the cache:
    /// const data = engine.pipeline_cache_data();
    /// if (data.length > 0) {
    ///   localStorage.setItem('flare-pipeline-cache', JSON.stringify(Array.from(data)));
    /// }
    /// ```
    #[wasm_bindgen]
    pub async fn init_gpu_with_cache(&mut self, cache_data: &[u8]) -> bool {
        if !webgpu_available() {
            return false;
        }
        let result = if cache_data.is_empty() {
            WebGpuBackend::new().await
        } else {
            WebGpuBackend::new_with_cache(cache_data).await
        };
        match result {
            Ok(gpu) => {
                self.model.set_backend(Box::new(gpu));
                // See `init_gpu` — upload raw quantized weights to GPU buffers
                // so forward() takes the fused single-encoder path.
                self.model.upload_weights_to_gpu();
                true
            }
            Err(_) => false,
        }
    }

    /// Serialise the driver-managed GPU pipeline cache to bytes.
    ///
    /// Returns an opaque blob that can be passed to `init_gpu_with_cache()` on
    /// the next startup to skip shader recompilation.  Store it in
    /// `localStorage` or `IndexedDB` between page loads.
    ///
    /// Returns an empty `Uint8Array` if no GPU is active, or if the current
    /// backend does not support pipeline caching (WebGPU, Metal, DX12).
    #[wasm_bindgen(getter)]
    pub fn pipeline_cache_data(&self) -> Vec<u8> {
        self.model.backend().pipeline_cache_data()
    }

    /// Diagnostic snapshot of the current compute backend as a JSON string.
    ///
    /// Returns an object with:
    /// - `backend` — backend identifier (`"cpu"` or `"webgpu"`).
    /// - `has_gpu_weights` — `true` once weights have been uploaded to GPU buffers.
    /// - `has_gpu_kv_cache` — `true` once GPU-resident KV storage is initialised.
    /// - `has_raw_weights` — `true` once raw quantized weights are held on the CPU side.
    ///
    /// Benchmarks can call this after `init_gpu()` to confirm WebGPU is
    /// actually driving inference instead of silently falling back to CPU.
    ///
    /// ```javascript
    /// await engine.init_gpu();
    /// console.log(JSON.parse(engine.backend_info()));
    /// // { backend: "webgpu", has_gpu_weights: true, has_gpu_kv_cache: true, has_raw_weights: true }
    /// ```
    #[wasm_bindgen]
    pub fn backend_info(&self) -> String {
        let backend = self.model.backend();
        format!(
            "{{\"backend\":\"{}\",\"has_gpu_weights\":{},\"has_gpu_kv_cache\":{},\"has_raw_weights\":{}}}",
            backend.name(),
            backend.has_gpu_weights(),
            backend.has_gpu_kv_cache(),
            self.model.has_raw_weights()
        )
    }

    /// Turn on per-phase wall-clock profiling of `forward_prefill`.
    ///
    /// After calling this, the next `begin_stream*` / `forward_prefill` that runs
    /// over more than one token records a breakdown of where time is spent
    /// (embed, attention, FFN, KV writes, LM head, etc.).  Retrieve the JSON
    /// snapshot via [`FlareEngine::prefill_profile_json`].
    ///
    /// Overhead when enabled: one `performance.now()` call per phase boundary
    /// (~15 per layer).  Turn off via [`FlareEngine::disable_prefill_profiling`]
    /// before production inference.
    #[wasm_bindgen]
    pub fn enable_prefill_profiling(&mut self) {
        self.model.enable_prefill_profiling(now_ms_f64);
    }

    /// Turn off prefill profiling.  Subsequent prefill calls run with zero
    /// timing overhead.
    #[wasm_bindgen]
    pub fn disable_prefill_profiling(&mut self) {
        self.model.disable_prefill_profiling();
    }

    /// JSON snapshot of the most recent prefill profile, or `"null"` if
    /// profiling is disabled or no prefill has run since it was enabled.
    ///
    /// All `*_ms` fields are wall-clock milliseconds, summed across all
    /// transformer layers where applicable.  `seq_len` is the number of
    /// prompt tokens processed; `num_layers` is the transformer depth.
    #[wasm_bindgen]
    pub fn prefill_profile_json(&mut self) -> String {
        match self.model.take_prefill_profile() {
            None => "null".to_string(),
            Some(p) => format!(
                "{{\"seq_len\":{},\"num_layers\":{},\
                 \"embed_ms\":{:.3},\"attn_norm_ms\":{:.3},\"qkv_proj_ms\":{:.3},\
                 \"rope_ms\":{:.3},\"attention_ms\":{:.3},\"attn_out_proj_ms\":{:.3},\
                 \"ffn_norm_ms\":{:.3},\"gate_up_ms\":{:.3},\"silu_mul_ms\":{:.3},\
                 \"down_ms\":{:.3},\"residual_ms\":{:.3},\"kv_write_ms\":{:.3},\
                 \"final_norm_ms\":{:.3},\"lm_head_ms\":{:.3},\"total_ms\":{:.3}}}",
                p.seq_len,
                p.num_layers,
                p.embed_ms,
                p.attn_norm_ms,
                p.qkv_proj_ms,
                p.rope_ms,
                p.attention_ms,
                p.attn_out_proj_ms,
                p.ffn_norm_ms,
                p.gate_up_ms,
                p.silu_mul_ms,
                p.down_ms,
                p.residual_ms,
                p.kv_write_ms,
                p.final_norm_ms,
                p.lm_head_ms,
                p.total_ms,
            ),
        }
    }

    /// Run a single dummy forward pass to pre-compile WebGPU shader pipelines.
    ///
    /// WebGPU (and wgpu on native) compiles shader pipelines lazily on the
    /// first dispatch.  This causes a noticeable latency spike — often 100ms
    /// to several seconds — when the user makes their first inference request.
    ///
    /// Call `warmup()` once after `init_gpu()` completes to trigger all shader
    /// compilations in the background so the first real request feels fast.
    /// The KV cache is reset after the warmup so the engine is in a clean state.
    ///
    /// Returns `true` if the warmup forward pass ran without error, `false` if
    /// the model has not been loaded.
    ///
    /// # JS example
    /// ```javascript
    /// const engine = FlareEngine.load(bytes);
    /// await engine.init_gpu();
    /// engine.warmup(); // trigger shader compilation
    /// // First real inference is now fast
    /// engine.begin_stream(promptIds, 128);
    /// ```
    #[wasm_bindgen]
    pub fn warmup(&mut self) -> bool {
        // A single forward pass at position 0 with token 0 is enough to
        // compile all pipelines the model will use during inference.
        let _logits = self.model.forward(0, 0);
        // Restore clean KV state so the engine is ready for real inference.
        self.model.reset();
        self.kv_pos = 0;
        self.stream_done = true;
        self.stream_stop_reason.clear();
        true
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
            // Upload weights to persistent GPU buffers for single-encoder forward
            self.model.upload_weights_to_gpu();
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
    ///
    /// Also clears stop sequences, the internal text accumulator, and
    /// restores the RNG seed to the default `0x12345678`.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.model.reset();
        self.kv_pos = 0;
        self.stream_recent_tokens.clear();
        self.stop_sequences.clear();
        self.stream_text_accum.clear();
        self.rng_seed = 0x12345678;
        self.last_logits.clear();
        self.top_logprobs_data.clear();
        self.stream_stop_reason.clear();
        self.utf8_byte_buf.clear();
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

    /// Get the number of attention heads.
    #[wasm_bindgen(getter)]
    pub fn num_heads(&self) -> u32 {
        self.model.config().num_heads as u32
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

    /// Model architecture name from `general.architecture` in the GGUF metadata.
    ///
    /// Returns a lowercase string such as `"llama"`, `"mistral"`, `"gemma2"`,
    /// `"phi3"`, or `"qwen2"`. Returns `"unknown"` if the field is absent.
    #[wasm_bindgen(getter)]
    pub fn architecture(&self) -> String {
        self.architecture.clone()
    }

    /// Model display name from `general.name` in the GGUF metadata.
    ///
    /// Returns the human-readable name embedded by the model author (e.g.
    /// `"Llama 3.2 1B Instruct"`). Returns an empty string if the field is absent.
    #[wasm_bindgen(getter)]
    pub fn model_name(&self) -> String {
        self.model_name.clone()
    }

    /// All GGUF model metadata as a JSON string.
    ///
    /// Returns a JSON object mapping each metadata key to its value.
    /// Large vocabulary arrays (`tokenizer.ggml.tokens`, `.merges`, `.scores`,
    /// `.added_tokens`) are omitted to keep the payload practical.
    /// Small arrays (≤ 64 entries) are included as JSON arrays.
    ///
    /// Returns `"{}"` if the model was not loaded from a GGUF file.
    ///
    /// ```javascript
    /// const meta = JSON.parse(engine.metadata_json);
    /// console.log(meta["llama.context_length"]); // e.g. 4096
    /// ```
    #[wasm_bindgen(getter)]
    pub fn metadata_json(&self) -> String {
        self.metadata_json.clone()
    }

    /// Maximum sequence length (context window size) of the loaded model.
    ///
    /// Use this to warn users when their prompt is approaching the limit.
    #[wasm_bindgen(getter)]
    pub fn max_seq_len(&self) -> u32 {
        self.model.config().max_seq_len as u32
    }

    /// Number of tokens currently consumed in the KV-cache session (prompt + generated).
    ///
    /// Updated after every generation call; reset to 0 by `engine.reset()`.
    /// Use with `max_seq_len` to build a context-usage progress bar.
    #[wasm_bindgen(getter)]
    pub fn tokens_used(&self) -> u32 {
        self.kv_pos as u32
    }

    /// How many tokens of context space remain before the window is full.
    ///
    /// Equivalent to `max_seq_len - tokens_used`. Returns 0 when the context is
    /// already full or `max_seq_len` is 0.
    ///
    /// # JS example
    /// ```javascript
    /// if (engine.tokens_remaining < 64) {
    ///   console.warn("Context window almost full — consider resetting.");
    /// }
    /// ```
    #[wasm_bindgen(getter)]
    pub fn tokens_remaining(&self) -> u32 {
        let cap = self.model.config().max_seq_len;
        cap.saturating_sub(self.kv_pos) as u32
    }

    /// Fraction of the context window consumed (0.0 = empty, 1.0 = full).
    ///
    /// Equivalent to `tokens_used / max_seq_len`. Returns 0.0 if `max_seq_len` is 0.
    #[wasm_bindgen(getter)]
    pub fn context_window_pct(&self) -> f32 {
        let cap = self.model.config().max_seq_len;
        if cap == 0 {
            return 0.0;
        }
        (self.kv_pos as f32) / (cap as f32)
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

    /// Decode a single token ID to its text piece.
    ///
    /// Convenience wrapper around `decode_ids` for use directly inside a
    /// `next_token()` loop so callers don't need a separate `FlareTokenizer`.
    ///
    /// Returns an empty string if no GGUF vocab is loaded.
    ///
    /// # JS example
    /// ```javascript
    /// engine.begin_stream(promptIds, 128);
    /// requestAnimationFrame(function tick() {
    ///   const id = engine.next_token();
    ///   if (id !== undefined) output.textContent += engine.decode_token(id);
    ///   if (!engine.stream_done) requestAnimationFrame(tick);
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn decode_token(&self, id: u32) -> String {
        match &self.gguf_vocab {
            Some(vocab) => vocab.decode(&[id]),
            None => String::new(),
        }
    }

    /// Decode a single token ID, correctly handling multi-byte UTF-8 sequences.
    ///
    /// SentencePiece tokenizers encode non-ASCII characters as consecutive
    /// byte-level tokens such as `<0xE4>`, `<0xB8>`, `<0xAD>` (the UTF-8
    /// encoding of `中`).  The basic `decode_token` function returns incorrect
    /// Latin-1 characters in these cases because it treats each byte as an
    /// independent Unicode scalar.
    ///
    /// `decode_token_chunk` accumulates bytes in an internal buffer until a
    /// complete, valid UTF-8 sequence is assembled, then returns it as a
    /// `String`.  While the sequence is incomplete it returns an empty string,
    /// and when a regular (non-byte) token is encountered it flushes any
    /// buffered bytes (replacing invalid sequences with U+FFFD) before
    /// returning the decoded text.
    ///
    /// **Use this instead of `decode_token` whenever you are streaming tokens
    /// that may include non-Latin characters.**
    ///
    /// ```javascript
    /// engine.begin_stream(prompt, 256);
    /// function tick() {
    ///   const id = engine.next_token();
    ///   if (id !== undefined) output.textContent += engine.decode_token_chunk(id);
    ///   if (!engine.stream_done) requestAnimationFrame(tick);
    /// }
    /// requestAnimationFrame(tick);
    /// ```
    #[wasm_bindgen]
    pub fn decode_token_chunk(&mut self, id: u32) -> String {
        let vocab = match &self.gguf_vocab {
            Some(v) => v,
            None => return String::new(),
        };

        if let Some(token) = vocab.decode_token(id) {
            // Detect SentencePiece byte-level tokens: exactly "<0xXX>" (6 chars)
            if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                    self.utf8_byte_buf.push(byte);
                    return match std::str::from_utf8(&self.utf8_byte_buf) {
                        Ok(s) => {
                            let out = s.to_owned();
                            self.utf8_byte_buf.clear();
                            out
                        }
                        // Incomplete multi-byte sequence — keep buffering.
                        Err(_) => String::new(),
                    };
                }
            }
        }

        // Regular (non-byte) token: flush any pending bytes first, then decode.
        let prefix = if self.utf8_byte_buf.is_empty() {
            String::new()
        } else {
            let flushed = String::from_utf8_lossy(&self.utf8_byte_buf).into_owned();
            self.utf8_byte_buf.clear();
            flushed
        };
        let decoded = vocab.decode(&[id]);
        if prefix.is_empty() {
            decoded
        } else {
            prefix + &decoded
        }
    }

    /// Truncate `text` so that it fits within `budget` tokens when encoded.
    ///
    /// Encodes `text` with the embedded GGUF vocabulary, keeps the **last**
    /// `budget` tokens (tail of the text is preferred, so recent context is
    /// preserved), and decodes them back to a string.  Returns `text` unchanged
    /// if it already fits or if no vocab is available.
    ///
    /// A typical call reserves space for the system prompt + generated output:
    ///
    /// ```javascript
    /// // Keep only the tail of the conversation that fits in the context
    /// const budget = engine.max_seq_len - 256; // leave 256 tokens for output
    /// const trimmed = engine.truncate_to_context(conversationText, budget);
    /// ```
    #[wasm_bindgen]
    pub fn truncate_to_context(&self, text: &str, budget: u32) -> String {
        let vocab = match &self.gguf_vocab {
            Some(v) => v,
            None => return text.to_string(),
        };
        let tokens = vocab.encode(text);
        let n = budget as usize;
        if tokens.len() <= n {
            return text.to_string();
        }
        // Keep the last `n` tokens (most recent context).
        vocab.decode(&tokens[tokens.len() - n..])
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
        let stop_seqs = &self.stop_sequences;
        let vocab_ref = &self.gguf_vocab;
        let mut text_accum = String::new();
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        let generated = gen.generate(
            &prompt_tokens,
            max_tokens as usize,
            eos,
            || 0.5,
            |token_id, _pos| {
                if !stop_seqs.is_empty() {
                    if let Some(v) = vocab_ref {
                        let piece = v.decode(&[token_id]);
                        text_accum.push_str(&piece);
                        for seq in stop_seqs.iter() {
                            if text_accum.ends_with(seq.as_str()) {
                                return false;
                            }
                        }
                    }
                }
                true
            },
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = generated.len() as u32;
        self.kv_pos = prompt_tokens.len() + generated.len();
        match &self.gguf_vocab {
            Some(vocab) => vocab.decode(&generated),
            None => String::new(),
        }
    }

    /// Full text-in / text-out generation with explicit sampling parameters.
    ///
    /// Like `generate_text` but with the full set of sampling controls:
    ///
    /// - `temperature`: 0 = greedy, higher = more diverse
    /// - `top_p`: nucleus sampling (1.0 = disabled)
    /// - `top_k`: top-k sampling, applied when `top_p` is 1.0 and `min_p` is 0.0 (0 = disabled)
    /// - `repeat_penalty`: repetition penalty (1.0 = disabled)
    /// - `min_p`: min-p threshold (0.0 = disabled)
    ///
    /// Returns the decoded generated text. Returns an empty string if no GGUF vocab is available.
    /// Respects stop sequences registered via `add_stop_sequence`.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// const response = engine.generate_text_with_params(
    ///   "What is Rust?", 128, 0.8, 0.95, 40, 1.1, 0.0
    /// );
    /// output.textContent = response;
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn generate_text_with_params(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        min_p: f32,
    ) -> String {
        let raw_tokens = match &self.gguf_vocab {
            Some(vocab) => vocab.encode(prompt),
            None => return String::new(),
        };
        let token_ids = self.generate_with_params(
            &raw_tokens,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            min_p,
        );
        match &self.gguf_vocab {
            Some(vocab) => vocab.decode(&token_ids),
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

    /// Streaming text-in / text-out with explicit sampling parameters.
    ///
    /// Like `generate_stream` but with the full set of sampling controls:
    ///
    /// - `temperature`: 0 = greedy, higher = more diverse
    /// - `top_p`: nucleus sampling (1.0 = disabled)
    /// - `top_k`: top-k sampling, applied when `top_p` is 1.0 and `min_p` is 0.0 (0 = disabled)
    /// - `repeat_penalty`: repetition penalty (1.0 = disabled, 1.1–1.3 = typical)
    /// - `min_p`: min-p threshold (0.0 = disabled)
    ///
    /// Encodes `prompt` with the embedded GGUF vocabulary, generates up to
    /// `max_tokens` tokens, and calls `on_token(token_str)` with the decoded
    /// text for each token.  Respects stop sequences registered via
    /// `add_stop_sequence`.  Returns the number of tokens generated.
    ///
    /// Returns 0 if no GGUF vocab is available.
    ///
    /// # JS example
    /// ```javascript
    /// engine.add_stop_sequence("<|im_end|>");
    /// engine.reset();
    /// let out = '';
    /// const count = engine.generate_stream_with_params(
    ///   prompt, 200, 0.8, 0.95, 40, 1.1, 0.0,
    ///   (token) => { out += token; }
    /// );
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn generate_stream_with_params(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        min_p: f32,
        on_token: &js_sys::Function,
    ) -> u32 {
        let raw_tokens = match &self.gguf_vocab {
            Some(vocab) => vocab.encode(prompt),
            None => return 0,
        };
        self.begin_stream_with_params(
            &raw_tokens,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            min_p,
        );
        let mut count = 0u32;
        while let Some(token_id) = self.next_token() {
            if let Some(vocab) = &self.gguf_vocab {
                let token_str = vocab.decode(&[token_id]);
                let _ = on_token.call1(&JsValue::NULL, &JsValue::from_str(&token_str));
            }
            count += 1;
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
        self.stream_rng_state = self.rng_seed;
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
    /// Begin a token-by-token stream with sampling parameters including top-k.
    ///
    /// - `temperature`: controls randomness (0 = greedy, higher = more random)
    /// - `top_p`: nucleus sampling — keep the smallest token set whose cumulative
    ///   probability ≥ `top_p` (1.0 = disabled; applied when < 1.0)
    /// - `top_k`: keep only the `top_k` highest-probability tokens before sampling
    ///   (0 = disabled; applied when `top_p` is 1.0 and `top_k` > 0)
    /// - `repeat_penalty`: penalty applied to logits of recently-seen tokens to
    ///   reduce repetition (1.0 = disabled, 1.1–1.3 = typical range)
    ///
    /// ```javascript
    /// engine.begin_stream_with_params(promptIds, 200, 0.8, 0.95, 40, 1.1, 0.0);
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn begin_stream_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        min_p: f32,
    ) {
        self.stream_params = SamplingParams {
            temperature,
            top_p,
            top_k: top_k as usize,
            repeat_penalty,
            min_p,
            ..Default::default()
        };
        self.stream_rng_state = self.rng_seed;
        self.begin_stream_impl(prompt_tokens, max_tokens);
    }

    /// Begin a token-by-token stream, healing the last prompt token.
    ///
    /// Identical to `begin_stream` but avoids double-processing the final prompt
    /// token: the prefill runs only tokens `[0 .. n-2]`, then the first
    /// `next_token()` call processes the last prompt token at its correct
    /// position `n-1` and produces the first output token.  This keeps RoPE
    /// positional embeddings consistent and is recommended when the prompt
    /// ends at a natural token boundary (e.g. when encoding a user turn in a
    /// chat template).
    ///
    /// Falls back to `begin_stream` for prompts shorter than 2 tokens.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// const ids = engine.encode_text(engine.apply_chat_template(userMsg, sysMsg));
    /// engine.begin_stream_healed(ids, 256);
    /// requestAnimationFrame(function tick() {
    ///   const id = engine.next_token();
    ///   if (id !== undefined) output.textContent += tokenizer.decode_one(id);
    ///   if (!engine.stream_done) requestAnimationFrame(tick);
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn begin_stream_healed(&mut self, prompt_tokens: &[u32], max_tokens: u32) {
        self.stream_params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        self.stream_rng_state = self.rng_seed;
        self.begin_stream_healed_impl(prompt_tokens, max_tokens);
    }

    /// Like `begin_stream_healed` but with full sampling parameters.
    ///
    /// Combines position-consistent prefill (see `begin_stream_healed`) with
    /// the same temperature / top-p / top-k / repeat-penalty / min-p controls
    /// available in `begin_stream_with_params`.
    ///
    /// # JS example
    /// ```javascript
    /// engine.reset();
    /// const ids = engine.encode_text(engine.apply_chat_template(userMsg, sysMsg));
    /// engine.begin_stream_healed_with_params(ids, 256, 0.8, 0.95, 40, 1.1, 0.0);
    /// requestAnimationFrame(function tick() {
    ///   const id = engine.next_token();
    ///   if (id !== undefined) output.textContent += tokenizer.decode_one(id);
    ///   if (!engine.stream_done) requestAnimationFrame(tick);
    /// });
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn begin_stream_healed_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        min_p: f32,
    ) {
        self.stream_params = SamplingParams {
            temperature,
            top_p,
            top_k: top_k as usize,
            repeat_penalty,
            min_p,
            ..Default::default()
        };
        self.stream_rng_state = self.rng_seed;
        self.begin_stream_healed_impl(prompt_tokens, max_tokens);
    }

    /// Internal prefill + state initialisation shared by both begin_stream variants.
    fn begin_stream_impl(&mut self, prompt_tokens: &[u32], max_tokens: u32) {
        let effective = self.with_bos(prompt_tokens);
        let t0 = now_ms();
        // Batched prefill: one pass over all prompt tokens using the tuned
        // batched_dequant_matmul path (2×4 tile on aarch64 / 2×2 on wasm+
        // simd128) instead of N sequential single-token forward() calls.
        // Dramatically lower TTFT on multi-token prompts.
        let pos = if effective.is_empty() {
            0
        } else {
            let _ = self.model.forward_prefill(&effective);
            effective.len()
        };
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = 0;
        self.stream_decode_start_ms = 0.0;
        self.stream_pos = pos;
        self.kv_pos = pos;
        self.stream_last_token = *effective.last().unwrap_or(&0);
        self.stream_remaining = max_tokens as usize;
        self.stream_done = false;
        self.stream_stop_reason.clear();
        // Seed the repetition-penalty window with the tail of the prompt.
        let window = self.repeat_last_n;
        let n = if window == 0 {
            0
        } else {
            effective.len().min(window)
        };
        self.stream_recent_tokens.clear();
        if n > 0 {
            self.stream_recent_tokens
                .extend_from_slice(&effective[effective.len() - n..]);
        }
        // Reset stop-sequence accumulator for this stream.
        self.stream_text_accum.clear();
    }

    /// Healed prefill: run tokens `[0 .. n-2]` during prefill, leave the last
    /// prompt token for the first `next_token()` call at its correct position.
    fn begin_stream_healed_impl(&mut self, prompt_tokens: &[u32], max_tokens: u32) {
        let effective = self.with_bos(prompt_tokens);
        // Short prompt: fall back to standard prefill.
        if effective.len() < 2 {
            self.begin_stream_impl(prompt_tokens, max_tokens);
            return;
        }
        let last_idx = effective.len() - 1;
        let t0 = now_ms();
        // Batched prefill for the all-but-last tokens; the last prompt token
        // is held back so next_token() runs it at its correct RoPE position.
        let pos = if last_idx == 0 {
            0
        } else {
            let _ = self.model.forward_prefill(&effective[..last_idx]);
            last_idx
        };
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = 0;
        self.stream_decode_start_ms = 0.0;
        // Leave stream_pos = last_idx so next_token() runs the last prompt
        // token at its correct position (last_idx), keeping RoPE consistent.
        self.stream_pos = pos; // == last_idx
        self.kv_pos = pos;
        self.stream_last_token = effective[last_idx];
        self.stream_remaining = max_tokens as usize;
        self.stream_done = false;
        self.stream_stop_reason.clear();
        let window = self.repeat_last_n;
        let n = if window == 0 {
            0
        } else {
            effective.len().min(window)
        };
        self.stream_recent_tokens.clear();
        if n > 0 {
            self.stream_recent_tokens
                .extend_from_slice(&effective[effective.len() - n..]);
        }
        self.stream_text_accum.clear();
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
            if !self.stream_done {
                self.stream_done = true;
                self.stream_stop_reason = "length".to_string();
            }
            return None;
        }

        // Start the decode timer on the first call after begin_stream.
        if self.stream_decode_start_ms == 0.0 {
            self.stream_decode_start_ms = now_ms();
        }

        let logits_tensor = self.model.forward(self.stream_last_token, self.stream_pos);
        // Capture raw pre-temperature logits for last_logits() getter.
        self.last_logits = logits_tensor.data().to_vec();
        // Compute top-N log-probabilities if requested.
        if self.top_logprobs_n > 0 {
            self.top_logprobs_data =
                compute_top_logprobs(&self.last_logits, self.top_logprobs_n as usize);
        }
        let token_id = if self.stream_params.temperature == 0.0 {
            sampling::sample_greedy(logits_tensor.data())
        } else {
            let mut logits = logits_tensor.data().to_vec();
            // Apply repetition penalty before temperature so penalty operates on
            // raw logits (consistent with Generator::step and the llama.cpp convention).
            sampling::apply_repeat_penalty(
                &mut logits,
                &self.stream_recent_tokens,
                self.stream_params.repeat_penalty,
            );
            sampling::apply_temperature(&mut logits, self.stream_params.temperature);
            // Advance LCG RNG state for this token.
            self.stream_rng_state = self
                .stream_rng_state
                .wrapping_mul(1664525)
                .wrapping_add(1013904223);
            let rng_val = (self.stream_rng_state as f32) / (u32::MAX as f32);
            // Mirror Generator::step() priority: top_p > min_p > top_k > full nucleus
            if self.stream_params.top_p < 1.0 {
                sampling::sample_top_p(&logits, self.stream_params.top_p, rng_val)
            } else if self.stream_params.min_p > 0.0 {
                sampling::sample_min_p(&logits, self.stream_params.min_p, rng_val)
            } else if self.stream_params.top_k > 0 {
                sampling::sample_top_k(&logits, self.stream_params.top_k, rng_val)
            } else {
                sampling::sample_top_p(&logits, 1.0, rng_val)
            }
        };

        self.stream_last_token = token_id;
        self.stream_pos += 1;
        self.kv_pos = self.stream_pos;
        // Update rolling repetition-penalty window.
        if self.repeat_last_n > 0 {
            if self.stream_recent_tokens.len() >= self.repeat_last_n {
                self.stream_recent_tokens.remove(0);
            }
            self.stream_recent_tokens.push(token_id);
        }
        self.stream_remaining -= 1;

        if self.eos_token_id == Some(token_id) {
            self.last_decode_ms = now_ms() - self.stream_decode_start_ms;
            self.stream_done = true;
            self.stream_stop_reason = "eos".to_string();
            return None;
        }

        // Check stop sequences by decoding the current token and appending it to
        // the accumulated text.  When a match is found the token is withheld (not
        // returned) so stop-sequence text never reaches the caller.
        if !self.stop_sequences.is_empty() {
            if let Some(vocab) = &self.gguf_vocab {
                let piece = vocab.decode(&[token_id]);
                self.stream_text_accum.push_str(&piece);
                for seq in &self.stop_sequences {
                    if self.stream_text_accum.ends_with(seq.as_str()) {
                        self.last_decode_ms = now_ms() - self.stream_decode_start_ms;
                        self.stream_done = true;
                        self.stream_stop_reason = "stop_sequence".to_string();
                        return None;
                    }
                }
            }
        }

        if self.stream_remaining == 0 {
            self.stream_done = true;
            self.stream_stop_reason = "length".to_string();
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
        if !self.stream_done {
            self.stream_stop_reason = "user".to_string();
        }
        self.stream_done = true;
    }

    /// Whether the current stream has finished.
    #[wasm_bindgen(getter)]
    pub fn stream_done(&self) -> bool {
        self.stream_done
    }

    /// Why the most-recent stream stopped.
    ///
    /// Returns one of:
    /// - `"eos"` — the model emitted the EOS token
    /// - `"length"` — `max_tokens` budget was exhausted
    /// - `"stop_sequence"` — a registered stop sequence was matched
    /// - `"user"` — `stop_stream()` was called
    /// - `""` (empty) — stream not yet started or still running
    ///
    /// # JS example
    /// ```javascript
    /// while (!engine.stream_done) engine.next_token();
    /// console.log("Stopped because:", engine.stream_stop_reason);
    /// ```
    #[wasm_bindgen(getter)]
    pub fn stream_stop_reason(&self) -> String {
        self.stream_stop_reason.clone()
    }

    /// Register a stop sequence.
    ///
    /// Generation halts (without emitting the matched tokens) as soon as the
    /// decoded output ends with `sequence`.  Call once per stop string before
    /// `begin_stream` or `generate_with_params`.
    ///
    /// Stop sequences are cleared by `reset()` or `clear_stop_sequences()`.
    ///
    /// ```javascript
    /// engine.add_stop_sequence("<|im_end|>");
    /// engine.add_stop_sequence("</s>");
    /// engine.begin_stream_with_params(promptIds, 200, 0.8, 0.95, 40, 1.1);
    /// ```
    #[wasm_bindgen]
    pub fn add_stop_sequence(&mut self, sequence: &str) {
        if !sequence.is_empty() {
            self.stop_sequences.push(sequence.to_string());
        }
    }

    /// Remove all registered stop sequences.
    #[wasm_bindgen]
    pub fn clear_stop_sequences(&mut self) {
        self.stop_sequences.clear();
    }

    /// Set the LCG RNG seed used for the next sampled generation call.
    ///
    /// Controls the random state passed to `begin_stream_with_params` and
    /// `generate_with_params`, enabling reproducible outputs.  The seed is
    /// applied on the next call and then *not* automatically reset, so the
    /// same seed will be reused on subsequent calls unless `set_rng_seed` or
    /// `reset()` is called again.
    ///
    /// `reset()` restores the seed to the default `0x12345678`.
    ///
    /// ```javascript
    /// engine.set_rng_seed(42);
    /// const out1 = engine.generate_text("Hello", 50);
    /// engine.set_rng_seed(42);
    /// const out2 = engine.generate_text("Hello", 50);
    /// // out1 === out2
    /// ```
    #[wasm_bindgen]
    pub fn set_rng_seed(&mut self, seed: u32) {
        self.rng_seed = seed;
    }

    /// Set the repetition-penalty look-back window (number of recent tokens to
    /// penalise).  Use `0` to disable repetition penalty entirely.  Default: 64.
    ///
    /// Takes effect on the next `begin_stream*` call.
    ///
    /// # JS example
    /// ```javascript
    /// engine.set_repeat_last_n(128); // wider window for creative writing
    /// engine.set_repeat_last_n(0);   // disable repeat penalty
    /// ```
    #[wasm_bindgen]
    pub fn set_repeat_last_n(&mut self, n: u32) {
        self.repeat_last_n = n as usize;
    }

    /// Current repetition-penalty window size (0 = disabled).
    #[wasm_bindgen(getter)]
    pub fn repeat_last_n(&self) -> u32 {
        self.repeat_last_n as u32
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
        let stop_seqs = &self.stop_sequences;
        let vocab = &self.gguf_vocab;
        let mut text_accum = String::new();
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        let result = gen.generate(
            &effective,
            max_tokens as usize,
            self.eos_token_id,
            || 0.5,
            |token_id, _pos| {
                if !stop_seqs.is_empty() {
                    if let Some(v) = vocab {
                        let piece = v.decode(&[token_id]);
                        text_accum.push_str(&piece);
                        for seq in stop_seqs.iter() {
                            if text_accum.ends_with(seq.as_str()) {
                                return false;
                            }
                        }
                    }
                }
                true
            },
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = result.len() as u32;
        self.kv_pos = effective.len() + result.len();
        result
    }

    /// Generate a batch of tokens with explicit sampling parameters.
    ///
    /// - `temperature`: 0 = greedy, higher = more diverse
    /// - `top_p`: nucleus sampling (1.0 = disabled)
    /// - `top_k`: top-k sampling, applied when `top_p` is 1.0 and `min_p` is 0.0 (0 = disabled)
    /// - `repeat_penalty`: repetition penalty applied to recently-seen tokens (1.0 = disabled)
    /// - `min_p`: min-p threshold (0.0 = disabled); applied after `top_p`, before `top_k`
    ///
    /// Stops early at EOS. Uses a fixed LCG RNG seed for reproducibility.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn generate_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        min_p: f32,
    ) -> Vec<u32> {
        let effective = self.with_bos(prompt_tokens);
        let params = SamplingParams {
            temperature,
            top_p,
            top_k: top_k as usize,
            repeat_penalty,
            min_p,
            ..Default::default()
        };
        let stop_seqs = &self.stop_sequences;
        let vocab = &self.gguf_vocab;
        let mut text_accum = String::new();
        let t0 = now_ms();
        let mut gen = Generator::new(&mut self.model, params);
        // Simple LCG for browser-side RNG (seeded from self.rng_seed)
        let mut state: u32 = self.rng_seed;
        let mut rng = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32)
        };
        let result = gen.generate(
            &effective,
            max_tokens as usize,
            self.eos_token_id,
            &mut rng,
            |token_id, _pos| {
                if !stop_seqs.is_empty() {
                    if let Some(v) = vocab {
                        let piece = v.decode(&[token_id]);
                        text_accum.push_str(&piece);
                        for seq in stop_seqs.iter() {
                            if text_accum.ends_with(seq.as_str()) {
                                return false;
                            }
                        }
                    }
                }
                true
            },
        );
        self.last_prefill_ms = now_ms() - t0;
        self.last_decode_ms = 0.0;
        self.last_tokens_generated = result.len() as u32;
        self.kv_pos = effective.len() + result.len();
        result
    }

    // -----------------------------------------------------------------------
    // Perplexity API
    // -----------------------------------------------------------------------

    /// Compute the perplexity of `text` under the loaded model.
    ///
    /// Encodes `text` with the embedded GGUF vocabulary, runs one forward pass
    /// per token, and measures the log-probability of each correct next-token
    /// prediction.  Perplexity = exp(−mean(log_probs)).
    ///
    /// The KV cache is reset **before and after** the evaluation so the engine
    /// returns to a clean state.
    ///
    /// Returns `f32::INFINITY` if the text encodes to fewer than 2 tokens or if
    /// no GGUF vocabulary is available.
    ///
    /// # JS example
    /// ```javascript
    /// const ppl = engine.compute_perplexity("The quick brown fox");
    /// console.log("Perplexity:", ppl);
    /// ```
    #[wasm_bindgen]
    pub fn compute_perplexity(&mut self, text: &str) -> f32 {
        let raw_tokens = match &self.gguf_vocab {
            Some(vocab) => vocab.encode(text),
            None => return f32::INFINITY,
        };
        let tokens = self.with_bos(&raw_tokens);
        if tokens.len() < 2 {
            return f32::INFINITY;
        }

        // Reset before evaluation to start from a clean KV state.
        self.model.reset();

        let n = tokens.len();
        let mut total_log_prob: f64 = 0.0;

        for i in 0..(n - 1) {
            let input_token = tokens[i];
            let target_token = tokens[i + 1] as usize;
            let logits_tensor = self.model.forward(input_token, i);
            let logits = logits_tensor.data();
            // Numerically-stable log-softmax at target_token.
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&x| (x - max_l).exp()).sum();
            let log_prob = (logits[target_token] - max_l) - sum_exp.ln();
            total_log_prob += log_prob as f64;
        }

        // Reset after evaluation, restoring clean state for subsequent inference.
        self.model.reset();
        self.kv_pos = 0;
        self.stream_done = true;

        let nll = -total_log_prob / (n - 1) as f64;
        nll.exp() as f32
    }

    // -----------------------------------------------------------------------
    // Last-step logits API
    // -----------------------------------------------------------------------

    /// Raw pre-temperature logits from the most recent forward pass.
    ///
    /// Returns the full vocabulary logit vector as a `Float32Array`.  These
    /// are the raw values *before* temperature scaling, repetition penalty,
    /// or any sampling filter — equivalent to the model's raw next-token
    /// distribution.
    ///
    /// Useful for:
    /// - Scoring candidate continuations (classification, ranking)
    /// - Computing perplexity / cross-entropy
    /// - Inspecting the model's "confidence" about the next token
    ///
    /// Returns an empty array before any inference has been run, and is
    /// cleared by `reset()`.
    ///
    /// ```javascript
    /// engine.begin_stream(promptIds, 1); // one token prefill+decode
    /// engine.next_token();
    /// const logits = engine.last_logits; // Float32Array of vocab_size
    /// const topTokenId = logits.indexOf(Math.max(...logits));
    /// ```
    #[wasm_bindgen(getter)]
    pub fn last_logits(&self) -> Vec<f32> {
        self.last_logits.clone()
    }

    // -----------------------------------------------------------------------
    // Top-N log-probabilities API
    // -----------------------------------------------------------------------

    /// Set how many top log-probability entries to capture after each forward
    /// pass.  Pass `0` (the default) to disable and save the computation.
    ///
    /// When enabled, `top_logprobs` is populated after every `next_token()`
    /// call and after every token in `generate_stream_with_params`.
    ///
    /// # JS example
    /// ```javascript
    /// engine.set_top_logprobs(5);
    /// engine.begin_stream(promptIds, 64);
    /// while (!engine.stream_done) {
    ///   engine.next_token();
    ///   const lp = engine.top_logprobs; // Float32Array [id0, lp0, id1, lp1, ...]
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn set_top_logprobs(&mut self, n: u32) {
        self.top_logprobs_n = n;
        self.top_logprobs_data.clear();
    }

    /// Interleaved top-N log-probabilities from the last forward pass.
    ///
    /// Layout: `[token_id_0 as f32, log_prob_0, token_id_1 as f32, log_prob_1, ...]`
    /// sorted by descending log-probability.  Length is `top_logprobs_n * 2`.
    ///
    /// Returns an empty array if `set_top_logprobs(0)` (default) or before
    /// any inference has been run.
    #[wasm_bindgen(getter)]
    pub fn top_logprobs(&self) -> Vec<f32> {
        self.top_logprobs_data.clone()
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

    /// Merge a LoRA adapter (SafeTensors format) into the model weights.
    ///
    /// Pass the raw bytes of a `.safetensors` file containing LoRA A/B matrices.
    /// After merging, the adapter's effect is permanent for this engine instance;
    /// call `FlareEngine.load()` again to restore the base model.
    ///
    /// ```javascript
    /// const resp = await fetch('lora-adapter.safetensors');
    /// const bytes = new Uint8Array(await resp.arrayBuffer());
    /// engine.merge_lora(bytes);
    /// ```
    #[wasm_bindgen]
    pub fn merge_lora(&mut self, adapter_bytes: &[u8]) -> Result<(), JsError> {
        let adapter = flare_loader::load_lora_from_safetensors(adapter_bytes)
            .map_err(|e| JsError::new(&format!("LoRA load error: {e}")))?;
        self.model
            .merge_lora(&adapter)
            .map_err(|e| JsError::new(&format!("LoRA merge error: {e}")))?;
        Ok(())
    }

    /// Merge a LoRA adapter with a custom alpha scaling factor.
    ///
    /// Same as `merge_lora` but overrides the alpha value embedded in the
    /// adapter file.  The effective scaling is `alpha / rank`.
    #[wasm_bindgen]
    pub fn merge_lora_with_alpha(
        &mut self,
        adapter_bytes: &[u8],
        alpha: f32,
    ) -> Result<(), JsError> {
        let mut adapter = flare_loader::load_lora_from_safetensors(adapter_bytes)
            .map_err(|e| JsError::new(&format!("LoRA load error: {e}")))?;
        adapter.alpha = alpha;
        self.model
            .merge_lora(&adapter)
            .map_err(|e| JsError::new(&format!("LoRA merge error: {e}")))?;
        Ok(())
    }

    // --- P2P / collaborative inference primitives (issue #389) ---
    //
    // These primitives expose the "head" and "tail" of a forward pass so
    // that a JavaScript orchestrator (e.g. a WebRTC mesh) can split
    // inference across multiple peers. A coordinator peer calls
    // `embed_token` to get the initial hidden state, ships it through a
    // chain of peers that each run some subset of transformer layers, and
    // finally calls `output_projection` on the returned hidden state to
    // obtain logits.
    //
    // Note: processing an arbitrary individual transformer layer is not
    // exposed here because the KV cache state for that layer lives inside
    // the owning `Model`. Full P2P layer sharding will require additional
    // KV-cache plumbing; these two primitives are sufficient to begin
    // P2P experimentation on top of the existing forward methods.

    /// Look up the token embedding row for `token_id` as a flat `Float32Array`.
    ///
    /// The length of the returned vector is `hidden_dim`. See also
    /// [`FlareEngine::output_projection`] for the inverse tail step.
    #[wasm_bindgen]
    pub fn embed_token(&self, token_id: u32) -> Vec<f32> {
        self.model.embed_token(token_id)
    }

    /// Apply final RMSNorm + output projection to a hidden state and
    /// return logits over the vocabulary.
    ///
    /// `hidden` must have length `hidden_dim`. The returned vector has
    /// length `vocab_size`.
    #[wasm_bindgen]
    pub fn output_projection(&self, hidden: Vec<f32>) -> Vec<f32> {
        self.model.output_projection(&hidden)
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
        let architecture = gguf.architecture().unwrap_or("unknown").to_string();
        let model_name = gguf
            .metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let gguf_vocab = GgufVocab::from_gguf(&gguf).ok();
        let metadata_json = build_metadata_json(&gguf);
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
            architecture,
            model_name,
            kv_pos: 0,
            stream_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            stream_rng_state: 0x12345678,
            stream_last_token: 0,
            stream_recent_tokens: Vec::new(),
            repeat_last_n: 64,
            stream_pos: 0,
            stream_remaining: 0,
            stream_done: true,
            stream_stop_reason: String::new(),
            last_prefill_ms: 0.0,
            last_decode_ms: 0.0,
            last_tokens_generated: 0,
            stream_decode_start_ms: 0.0,
            stop_sequences: Vec::new(),
            stream_text_accum: String::new(),
            rng_seed: 0x12345678,
            metadata_json,
            last_logits: Vec::new(),
            top_logprobs_n: 0,
            top_logprobs_data: Vec::new(),
            utf8_byte_buf: Vec::new(),
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

// ---------------------------------------------------------------------------
// OPFS (Origin Private File System) model caching
// ---------------------------------------------------------------------------

/// Directory name under the OPFS root where cached models are stored.
const OPFS_MODELS_DIR: &str = "flare-models";

/// Get the OPFS root directory handle, returning `Err` if unavailable.
async fn opfs_root() -> Result<web_sys::FileSystemDirectoryHandle, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let storage = window.navigator().storage();
    let promise = storage.get_directory();
    let root = wasm_bindgen_futures::JsFuture::from(promise).await?;
    Ok(root.unchecked_into())
}

/// Get (or create) the `flare-models` subdirectory inside OPFS.
async fn opfs_models_dir(create: bool) -> Result<web_sys::FileSystemDirectoryHandle, JsValue> {
    let root = opfs_root().await?;
    let opts = web_sys::FileSystemGetDirectoryOptions::new();
    opts.set_create(create);
    let promise = root.get_directory_handle_with_options(OPFS_MODELS_DIR, &opts);
    let dir = wasm_bindgen_futures::JsFuture::from(promise).await?;
    Ok(dir.unchecked_into())
}

/// Check if a model is cached in OPFS by name.
///
/// Returns `false` if OPFS is unavailable or the model is not found.
#[wasm_bindgen]
pub async fn is_model_cached(model_name: &str) -> bool {
    let dir = match opfs_models_dir(false).await {
        Ok(d) => d,
        Err(_) => return false,
    };
    let promise = dir.get_file_handle(model_name);
    wasm_bindgen_futures::JsFuture::from(promise).await.is_ok()
}

/// Save model bytes to OPFS.
///
/// Creates the `flare-models` directory if it does not exist.  Overwrites any
/// existing file with the same name.
#[wasm_bindgen]
pub async fn cache_model(model_name: &str, data: &[u8]) -> Result<(), JsValue> {
    let dir = opfs_models_dir(true).await?;
    let opts = web_sys::FileSystemGetFileOptions::new();
    opts.set_create(true);
    let file_handle: web_sys::FileSystemFileHandle =
        wasm_bindgen_futures::JsFuture::from(dir.get_file_handle_with_options(model_name, &opts))
            .await?
            .unchecked_into();

    let writable: web_sys::FileSystemWritableFileStream =
        wasm_bindgen_futures::JsFuture::from(file_handle.create_writable())
            .await?
            .unchecked_into();

    let write_promise = writable.write_with_u8_array(data)?;
    wasm_bindgen_futures::JsFuture::from(write_promise).await?;
    wasm_bindgen_futures::JsFuture::from(writable.close()).await?;
    Ok(())
}

/// Load model bytes from OPFS.
///
/// Returns `null` (JS) / `None` (Rust) if the model is not cached or OPFS is
/// unavailable.
#[wasm_bindgen]
pub async fn load_cached_model(model_name: &str) -> Result<JsValue, JsValue> {
    let dir = match opfs_models_dir(false).await {
        Ok(d) => d,
        Err(_) => return Ok(JsValue::NULL),
    };
    let file_handle: web_sys::FileSystemFileHandle =
        match wasm_bindgen_futures::JsFuture::from(dir.get_file_handle(model_name)).await {
            Ok(h) => h.unchecked_into(),
            Err(_) => return Ok(JsValue::NULL),
        };

    let file: web_sys::File = wasm_bindgen_futures::JsFuture::from(file_handle.get_file())
        .await?
        .unchecked_into();

    let array_buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    Ok(uint8_array.into())
}

/// Delete a cached model from OPFS.
#[wasm_bindgen]
pub async fn delete_cached_model(model_name: &str) -> Result<(), JsValue> {
    let dir = opfs_models_dir(false).await?;
    wasm_bindgen_futures::JsFuture::from(dir.remove_entry(model_name)).await?;
    Ok(())
}

/// List all cached models with their sizes (in bytes).
///
/// Returns a JSON-serialised array of objects: `[{name: string, size: number}, ...]`.
/// Returns `"[]"` if OPFS is unavailable or the models directory does not exist.
#[wasm_bindgen]
pub async fn list_cached_models() -> Result<JsValue, JsValue> {
    let dir = match opfs_models_dir(false).await {
        Ok(d) => d,
        Err(_) => {
            return Ok(JsValue::from_str("[]"));
        }
    };

    let entries_iter = dir.entries();
    let mut models: Vec<String> = Vec::new();

    loop {
        let next = wasm_bindgen_futures::JsFuture::from(entries_iter.next()?).await?;
        let done = js_sys::Reflect::get(&next, &JsValue::from_str("done"))?;
        if done.as_bool().unwrap_or(true) {
            break;
        }
        let value = js_sys::Reflect::get(&next, &JsValue::from_str("value"))?;
        let pair = js_sys::Array::from(&value);
        let name: String = pair.get(0).as_string().unwrap_or_default();
        let handle: web_sys::FileSystemFileHandle = pair.get(1).unchecked_into();
        let file: web_sys::File = wasm_bindgen_futures::JsFuture::from(handle.get_file())
            .await?
            .unchecked_into();
        let size = file.size();
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        models.push(format!(r#"{{"name":"{}","size":{}}}"#, escaped_name, size));
    }

    Ok(JsValue::from_str(&format!("[{}]", models.join(","))))
}

/// Get storage usage and quota estimate.
///
/// Returns a JSON string: `{usage: number, quota: number}`.
/// Returns `"{}"` if the Storage API is unavailable.
#[wasm_bindgen]
pub async fn storage_estimate() -> Result<JsValue, JsValue> {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return Ok(JsValue::from_str("{}")),
    };
    let storage = window.navigator().storage();
    let estimate: web_sys::StorageEstimate = match storage.estimate() {
        Ok(promise) => wasm_bindgen_futures::JsFuture::from(promise)
            .await?
            .unchecked_into(),
        Err(_) => return Ok(JsValue::from_str("{}")),
    };
    let usage = estimate.get_usage().unwrap_or(0.0);
    let quota = estimate.get_quota().unwrap_or(0.0);
    Ok(JsValue::from_str(&format!(
        r#"{{"usage":{},"quota":{}}}"#,
        usage, quota
    )))
}

//! Serial native CPU bridge for the Apple SDK. No networking or agent state.
//!
//! The caller owns each engine and must serialize all calls on that engine.
//! Every returned JSON string must be released with `flare_native_string_free`.
use flare_core::{
    chat::{ChatMessage, ChatTemplate},
    config::Architecture,
    model::Model,
    sampling::sample_greedy,
    tokenizer::{BpeTokenizer, Tokenizer},
};
use flare_loader::{
    gguf::GgufFile, quantize::QuantFormat, weights::load_model_weights_with_raw_opt,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    ffi::{c_char, CStr, CString},
    fs::File,
    io::BufReader,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    time::Instant,
};

type Result<T> = std::result::Result<T, String>;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    prompt: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    max_tokens: usize,
}

struct Run {
    prompt: Vec<u32>,
    cursor: usize,
    pending: Option<Vec<f32>>,
    output: Vec<u32>,
    bytes: Vec<u8>,
    max_tokens: usize,
    started: Instant,
    first_token_ms: Option<f64>,
    done: bool,
}

/// Opaque native engine. Only the C bridge exposes its address.
pub struct NativeEngine {
    model: Model,
    tokenizer: BpeTokenizer,
    run: Option<Run>,
    poisoned: bool,
}

impl NativeEngine {
    fn load(model_path: &str, tokenizer_path: &str, context: usize) -> Result<Self> {
        if !(32..=512).contains(&context) {
            return Err("context must be between 32 and 512 tokens".into());
        }
        let mut reader = BufReader::new(File::open(model_path).map_err(|e| e.to_string())?);
        let gguf = GgufFile::parse_header(&mut reader).map_err(|e| e.to_string())?;
        let mut config = gguf.to_model_config().map_err(|e| e.to_string())?;
        // This first mobile path deliberately matches the existing tested Qwen3
        // CPU implementation. Larger architectures need their own memory audit.
        if config.architecture != Architecture::Qwen3
            || config.hidden_dim > 1024
            || config.num_layers > 28
            || config.vocab_size > 151936
            || config.moe
            || gguf
                .tensors
                .iter()
                .any(|t| !matches!(t.dtype, QuantFormat::Q8_0 | QuantFormat::F32))
        {
            return Err(
                "native mobile preview supports dense Qwen3 up to 0.6B, Q8_0/F32 only".into(),
            );
        }
        config.max_seq_len = context;
        let tokenizer = BpeTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
        if tokenizer.vocab_size() > config.vocab_size {
            return Err("tokenizer vocabulary exceeds model vocabulary".into());
        }
        let (weights, raw) =
            load_model_weights_with_raw_opt(&gguf, &mut reader, true).map_err(|e| e.to_string())?;
        // Never construct a model with omitted f32 matrices unless all packed
        // matrices are available to the CPU backend.
        let raw = raw.ok_or("model has no complete packed Q8 projection weights")?;
        let mut model = Model::new(config, weights);
        model.set_raw_weights(raw);
        Ok(Self {
            model,
            tokenizer,
            run: None,
            poisoned: false,
        })
    }

    fn begin(&mut self, request: Request) -> Result<Value> {
        if self.run.as_ref().is_some_and(|r| !r.done) {
            return Err("generation is already active; reset it before starting another".into());
        }
        let text = match (request.prompt, request.messages) {
            (Some(prompt), None) => prompt,
            (None, Some(messages)) if !messages.is_empty() => ChatTemplate::Qwen3.apply(&messages),
            _ => return Err("provide exactly one nonempty prompt or messages array".into()),
        };
        let prompt = self.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let context = self.model.config().max_seq_len;
        if prompt.is_empty()
            || request.max_tokens == 0
            || prompt
                .len()
                .checked_add(request.max_tokens)
                .is_none_or(|n| n > context)
        {
            return Err(
                "prompt plus maxTokens must fit the context; input is never truncated".into(),
            );
        }
        if prompt
            .iter()
            .any(|&id| id as usize >= self.model.config().vocab_size)
        {
            return Err("prompt contains a token outside the model vocabulary".into());
        }
        self.model.reset();
        let count = prompt.len();
        self.run = Some(Run {
            prompt,
            cursor: 0,
            pending: None,
            output: Vec::new(),
            bytes: Vec::new(),
            max_tokens: request.max_tokens,
            started: Instant::now(),
            first_token_ms: None,
            done: false,
        });
        Ok(json!({"promptTokens":count}))
    }

    fn prefill(&mut self) -> Result<Value> {
        let run = self.run.as_mut().ok_or("begin must be called first")?;
        if run.done {
            return Err("generation has finished".into());
        }
        if run.cursor < run.prompt.len() {
            // The core batch API assumes a fresh prompt, not an existing KV
            // prefix. Submit it once in full to preserve attention/RoPE state.
            let end = run.prompt.len();
            let logits = self.model.forward_prefill(&run.prompt);
            run.cursor = end;
            if end == run.prompt.len() {
                if !logits.data().iter().all(|x| x.is_finite()) {
                    return Err("nonfinite prefill logits".into());
                }
                run.pending = Some(logits.data().to_vec());
            }
        }
        Ok(json!({"ready":run.cursor == run.prompt.len(),"processedTokens":run.cursor}))
    }

    fn next(&mut self) -> Result<Value> {
        let run = self.run.as_mut().ok_or("begin must be called first")?;
        if run.done || run.cursor != run.prompt.len() {
            return Err(
                "prefill must finish before decode, and decode cannot follow completion".into(),
            );
        }
        let token = if let Some(logits) = run.pending.take() {
            sample_greedy(&logits)
        } else {
            let previous = *run.output.last().ok_or("missing previous token")?;
            let (token, logit) = self
                .model
                .forward_greedy(previous, run.prompt.len() + run.output.len() - 1);
            if !logit.is_finite() {
                return Err("nonfinite decode logit".into());
            }
            token
        };
        let elapsed = run.started.elapsed().as_secs_f64() * 1000.;
        run.first_token_ms.get_or_insert(elapsed);
        let eos =
            Some(token) == self.tokenizer.eos_token_id() || token == 151643 || token == 151645;
        if !eos {
            run.output.push(token);
            run.bytes.extend(
                self.tokenizer
                    .decode_bytes(&[token])
                    .map_err(|e| e.to_string())?,
            );
        }
        let reason = if eos {
            Some("stop")
        } else if run.output.len() == run.max_tokens {
            Some("length")
        } else {
            None
        };
        run.done = reason.is_some();
        let text = take_utf8(&mut run.bytes, run.done);
        Ok(
            json!({"text":text,"tokenId":if eos {None} else {Some(token)},
            "finishReason":reason,"generatedTokens":run.output.len(),"elapsedMs":elapsed,
            "firstTokenMs":run.first_token_ms,"promptTokens":run.prompt.len()}),
        )
    }

    fn reset(&mut self) -> Result<Value> {
        self.run = None;
        self.model.reset();
        Ok(json!({}))
    }
}

// Retain incomplete UTF-8 across token boundaries, replacing only invalid bytes
// or an incomplete final suffix. Stream concatenation equals final decoding.
fn take_utf8(bytes: &mut Vec<u8>, final_chunk: bool) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                out.push_str(text);
                bytes.clear();
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                out.push_str(&String::from_utf8_lossy(&bytes[..valid]));
                bytes.drain(..valid);
                if let Some(length) = error.error_len() {
                    out.push('\u{fffd}');
                    bytes.drain(..length);
                } else {
                    if final_chunk {
                        out.push('\u{fffd}');
                        bytes.clear();
                    }
                    break;
                }
            }
        }
    }
    out
}

fn response(result: Result<Value>) -> *mut c_char {
    let value = match result {
        Ok(value) => json!({"value":value}),
        Err(error) => json!({"error":error}),
    };
    // JSON serialization escapes embedded NULs, so CString creation cannot fail
    // for these values; still avoid unwinding across the ABI if it ever does.
    match CString::new(value.to_string()) {
        Ok(text) => text.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

unsafe fn string<'a>(value: *const c_char) -> Result<&'a str> {
    if value.is_null() {
        return Err("null string argument".into());
    }
    // SAFETY: the ABI requires a valid, terminated string for this call.
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|e| e.to_string())
}

unsafe fn call(
    engine: *mut NativeEngine,
    f: impl FnOnce(&mut NativeEngine) -> Result<Value>,
) -> *mut c_char {
    // SAFETY: callers serialize operations and hold a live engine until close.
    let Some(engine) = (unsafe { engine.as_mut() }) else {
        return response(Err("null engine".into()));
    };
    if engine.poisoned {
        return response(Err("engine failed; close and reload it".into()));
    }
    match catch_unwind(AssertUnwindSafe(|| f(engine))) {
        Ok(result) => response(result),
        Err(_) => {
            engine.poisoned = true;
            response(Err("engine panic; close and reload it".into()))
        }
    }
}

/// Load an engine; the caller owns the returned handle and status JSON.
///
/// # Safety
/// Paths must be valid terminated UTF-8 strings; status must be writable. Calls
/// on a returned handle must be serial, and it must be closed exactly once.
#[no_mangle]
pub unsafe extern "C" fn flare_native_open(
    model: *const c_char,
    tokenizer: *const c_char,
    context: u32,
    status: *mut *mut c_char,
) -> *mut NativeEngine {
    if status.is_null() {
        return ptr::null_mut();
    }
    let start = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: string validates null/UTF-8; other validity is the caller contract.
        NativeEngine::load(
            unsafe { string(model)? },
            unsafe { string(tokenizer)? },
            context as usize,
        )
    }));
    let (handle, result) = match result {
        Ok(Ok(engine)) => (
            Box::into_raw(Box::new(engine)),
            Ok(json!({"loadMs":start.elapsed().as_secs_f64()*1000.})),
        ),
        Ok(Err(error)) => (ptr::null_mut(), Err(error)),
        Err(_) => (ptr::null_mut(), Err("engine panicked while loading".into())),
    };
    // SAFETY: status points to writable caller-owned storage.
    unsafe {
        *status = response(result);
    }
    handle
}

/// Begin an independent greedy request, using raw prompt or Qwen3 chat messages.
/// # Safety
/// Engine must be live/exclusively accessed; request must be terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn flare_native_begin(
    engine: *mut NativeEngine,
    request: *const c_char,
) -> *mut c_char {
    // SAFETY: forwarded ABI contract; panics are contained by call.
    unsafe {
        call(engine, |e| {
            e.begin(serde_json::from_str(string(request)?).map_err(|e| e.to_string())?)
        })
    }
}

/// Process the entire prompt in one batch. Returns ready=true after prefill.
/// # Safety
/// Engine must be live and exclusively accessed for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn flare_native_prefill(engine: *mut NativeEngine) -> *mut c_char {
    unsafe { call(engine, NativeEngine::prefill) }
}

/// Generate one greedy token, with streaming UTF-8 and timing metadata.
/// # Safety
/// Engine must be live and exclusively accessed for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn flare_native_next(engine: *mut NativeEngine) -> *mut c_char {
    unsafe { call(engine, NativeEngine::next) }
}

/// Discard the current generation and reset KV state, retaining loaded weights.
/// # Safety
/// Engine must be live and exclusively accessed for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn flare_native_reset(engine: *mut NativeEngine) -> *mut c_char {
    unsafe { call(engine, NativeEngine::reset) }
}

/// Release an engine after its last call completes. Null is accepted.
/// # Safety
/// A nonnull handle must come from open, with no concurrent/future use or double close.
#[no_mangle]
pub unsafe extern "C" fn flare_native_close(engine: *mut NativeEngine) {
    if !engine.is_null() {
        // SAFETY: ownership is returned exactly once by the caller.
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Release a response returned by this library. Null is accepted.
/// # Safety
/// A nonnull pointer must be a live response from this library, freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn flare_native_string_free(text: *mut c_char) {
    if !text.is_null() {
        // SAFETY: the caller returns a CString allocated by response.
        unsafe {
            drop(CString::from_raw(text));
        }
    }
}

#[cfg(test)]
mod tests;

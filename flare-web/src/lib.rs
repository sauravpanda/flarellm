//! Browser integration for Flare LLM.
//!
//! Provides WASM-bindgen exports for WebGPU detection, device info,
//! and engine initialization. Designed to be built with `wasm-pack`.

use wasm_bindgen::prelude::*;

/// Check if WebGPU is available in the current browser.
#[wasm_bindgen]
pub fn webgpu_available() -> bool {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return false,
    };

    // Check if navigator.gpu exists
    let navigator: JsValue = window.navigator().into();
    js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

/// Get basic device info for capability detection.
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

/// Placeholder: initialize the Flare engine.
/// Will set up WebGPU device, create compute pipelines, etc.
#[wasm_bindgen]
pub async fn init() -> Result<JsValue, JsValue> {
    // Set up panic hook for better error messages in browser console
    // TODO: add console_error_panic_hook feature for better WASM error messages

    Ok(JsValue::from_str("flare initialized"))
}

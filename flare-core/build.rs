fn main() {
    // Link Accelerate framework only when the TARGET is macOS (or iOS/tvOS).
    // Using CARGO_CFG_TARGET_OS ensures we don't try to link frameworks when
    // cross-compiling to wasm32 or Linux from a macOS host.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os.as_str(), "macos" | "ios" | "tvos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

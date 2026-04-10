//! End-to-end inference benchmark using SmolLM2-135M Q8_0.
//!
//! Run with:
//!   cargo run -p flare-server --example e2e_bench --release
//!
//! Set MODEL_PATH env var to use a different model:
//!   MODEL_PATH=path/to/model.gguf cargo run -p flare-server --example e2e_bench --release

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

const DEFAULT_MODEL_DIR: &str = "models";
const DEFAULT_MODEL_NAME: &str = "smollm2-135m-instruct-q8_0.gguf";

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| format!("{DEFAULT_MODEL_DIR}/{DEFAULT_MODEL_NAME}"));

    if !Path::new(&model_path).exists() {
        eprintln!("Model not found at: {model_path}");
        eprintln!();
        eprintln!("Download SmolLM2-135M Q8_0:");
        eprintln!("  mkdir -p models");
        eprintln!("  huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct-GGUF \\");
        eprintln!("    smollm2-135m-instruct-q8_0.gguf --local-dir models");
        eprintln!();
        eprintln!("Or set MODEL_PATH to your GGUF file.");
        std::process::exit(1);
    }

    // --- Load model ---
    eprintln!("Loading model from {model_path}...");
    let load_start = Instant::now();

    let file = File::open(&model_path).expect("Failed to open model file");
    let mut reader = BufReader::new(file);
    let gguf = GgufFile::parse_header(&mut reader).expect("Failed to parse GGUF");
    let config = gguf.to_model_config().expect("Failed to extract config");
    let weights = load_model_weights(&gguf, &mut reader).expect("Failed to load weights");
    let mut model = Model::new(config.clone(), weights);

    let load_time = load_start.elapsed();
    eprintln!("Loaded in {:.2}s", load_time.as_secs_f64());
    eprintln!(
        "  Architecture: {:?}, Params: ~{}M",
        config.architecture,
        config.estimate_param_count() / 1_000_000
    );
    eprintln!(
        "  Layers: {}, Hidden: {}, Vocab: {}",
        config.num_layers, config.hidden_dim, config.vocab_size
    );

    println!();
    println!("Flare E2E Benchmark — {}", model_path);
    println!("========================================");
    println!("Hardware: {}", get_hw_info());
    println!();

    // --- Benchmark at different generation lengths ---
    // Use token IDs that exist in typical vocab (BOS + common tokens)
    let prompt_tokens: Vec<u32> = vec![1, 100, 200, 300, 400, 500];
    let prompt_len = prompt_tokens.len();

    println!("Generation speed (greedy, prompt={prompt_len} tokens):");
    for &gen_count in &[16, 32, 64, 128, 256] {
        model.reset();

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);

        // Prefill
        let prefill_start = Instant::now();
        gen.prefill(&prompt_tokens);
        let prefill_time = prefill_start.elapsed();

        // Generation
        let gen_start = Instant::now();
        for _ in 0..gen_count {
            gen.step(0.5);
        }
        let gen_time = gen_start.elapsed();

        let prefill_tok_s = prompt_len as f64 / prefill_time.as_secs_f64();
        let gen_tok_s = gen_count as f64 / gen_time.as_secs_f64();

        println!(
            "  gen={gen_count:>3}  prefill: {prefill_tok_s:>7.1} tok/s  decode: {gen_tok_s:>6.1} tok/s  ({:.0}ms total)",
            (prefill_time + gen_time).as_secs_f64() * 1000.0,
        );
    }

    // --- Sustained generation ---
    println!();
    println!("Sustained decode (512 tokens):");
    model.reset();
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut gen = Generator::new(&mut model, params);
    gen.prefill(&prompt_tokens);

    let start = Instant::now();
    for _ in 0..512 {
        gen.step(0.5);
    }
    let elapsed = start.elapsed();
    let tok_s = 512.0 / elapsed.as_secs_f64();
    println!(
        "  512 tokens in {:.0}ms = {tok_s:.1} tok/s",
        elapsed.as_secs_f64() * 1000.0
    );

    println!();
    println!("Model load: {:.2}s", load_time.as_secs_f64());
}

fn get_hw_info() -> String {
    #[cfg(target_arch = "aarch64")]
    {
        "ARM64 (NEON SIMD)".to_string()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        "x86_64 (scalar)".to_string()
    }
}

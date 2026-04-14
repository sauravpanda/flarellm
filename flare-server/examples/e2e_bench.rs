//! End-to-end inference benchmark using SmolLM2-135M Q8_0.
//!
//! Run with:
//!   cargo run -p flarellm-server --example e2e_bench --release
//!
//! Append results to the benchmark history log:
//!   cargo run -p flarellm-server --example e2e_bench --release -- --log
//!
//! Output as JSON (machine-readable, one object per line):
//!   cargo run -p flarellm-server --example e2e_bench --release -- --json
//!
//! Run with GPU (Metal/Vulkan/DX12) acceleration:
//!   cargo run -p flarellm-server --example e2e_bench --release -- --gpu
//!
//! Run speculative decoding A/B comparison:
//!   cargo run -p flarellm-server --example e2e_bench --release -- --speculative
//!
//! Use a different model via --model flag or MODEL_PATH env var:
//!   cargo run -p flarellm-server --example e2e_bench --release -- --model path/to/model.gguf
//!   MODEL_PATH=path/to/model.gguf cargo run -p flarellm-server --example e2e_bench --release

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_gpu::WebGpuBackend;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

const DEFAULT_MODEL_DIR: &str = "models";
const DEFAULT_MODEL_NAME: &str = "smollm2-135m-instruct-q8_0.gguf";
const HISTORY_FILE: &str = "BENCHMARK_HISTORY.md";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_mode = args.iter().any(|a| a == "--log");
    let json_mode = args.iter().any(|a| a == "--json");
    let gpu_mode = args.iter().any(|a| a == "--gpu");
    let speculative_mode = args.iter().any(|a| a == "--speculative");

    // Parse --model <path> flag, falling back to MODEL_PATH env var, then default.
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned())
        .or_else(|| std::env::var("MODEL_PATH").ok())
        .unwrap_or_else(|| format!("{DEFAULT_MODEL_DIR}/{DEFAULT_MODEL_NAME}"));

    if !Path::new(&model_path).exists() {
        eprintln!("Model not found at: {model_path}");
        eprintln!();
        eprintln!("Quick setup (downloads ~138MB):");
        eprintln!("  ./scripts/download_baseline_model.sh");
        eprintln!();
        eprintln!("Or manually:");
        eprintln!("  curl -L \"https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf\" \\");
        eprintln!("    -o models/smollm2-135m-instruct-q8_0.gguf");
        eprintln!();
        eprintln!("Or set MODEL_PATH to point to your own GGUF file.");
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

    // --- GPU backend ---
    let backend_label;
    if gpu_mode {
        eprintln!("Initializing GPU backend...");
        match pollster::block_on(WebGpuBackend::new()) {
            Ok(gpu) => {
                backend_label = "GPU (WebGPU/wgpu)".to_string();
                model.set_backend(Box::new(gpu));
                eprintln!("GPU backend initialized");

                // Load raw quantized weights for fused dequant+matvec kernels
                let file2 = File::open(&model_path).expect("Failed to reopen model file");
                let mut reader2 = BufReader::new(file2);
                let gguf2 = GgufFile::parse_header(&mut reader2).expect("Failed to reparse GGUF");
                let num_layers = config.num_layers;
                let mut raw_layers = Vec::with_capacity(num_layers);
                let mut all_ok = true;

                for layer_idx in 0..num_layers {
                    match gguf2.load_raw_layer_weights(&mut reader2, layer_idx) {
                        Ok(Some(rw)) => raw_layers.push(rw),
                        _ => {
                            all_ok = false;
                            break;
                        }
                    }
                }

                if raw_layers.len() == num_layers {
                    model.set_raw_weights(raw_layers);
                    eprintln!("Raw quantized weights loaded for GPU fused kernels");
                    // Upload weights to persistent GPU buffers for single-encoder forward
                    model.upload_weights_to_gpu();
                    if model.backend().has_gpu_weights() {
                        eprintln!(
                            "GPU-resident weights uploaded (single-encoder forward path enabled)"
                        );
                    }
                } else if !all_ok {
                    eprintln!("Warning: could not load raw weights, using f32 path on GPU");
                }
            }
            Err(e) => {
                eprintln!("Failed to initialize GPU backend: {e}");
                eprintln!("Falling back to CPU backend");
                backend_label = "CPU".to_string();
            }
        }
    } else {
        backend_label = "CPU".to_string();
    }

    let model_info = format!(
        "{:?}, ~{}M params, {} layers, dim={}",
        config.architecture,
        config.estimate_param_count() / 1_000_000,
        config.num_layers,
        config.hidden_dim,
    );
    eprintln!("Loaded in {:.2}s ({model_info})", load_time.as_secs_f64());
    eprintln!("Backend: {backend_label}");

    // --- Speculative decoding A/B comparison ---
    if speculative_mode {
        run_speculative_bench(&mut model, &backend_label, json_mode);
        return;
    }

    // --- Run benchmarks ---
    let prompt_tokens: Vec<u32> = vec![1, 100, 200, 300, 400, 500];
    let prompt_len = prompt_tokens.len();

    let mut results: Vec<(usize, f64, f64)> = Vec::new(); // (gen_count, prefill_tok_s, decode_tok_s)

    for &gen_count in &[16, 32, 64, 128, 256] {
        model.reset();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);

        let prefill_start = Instant::now();
        gen.prefill(&prompt_tokens);
        let prefill_time = prefill_start.elapsed();

        let gen_start = Instant::now();
        for _ in 0..gen_count {
            gen.step(0.5);
        }
        let gen_time = gen_start.elapsed();

        let prefill_tok_s = prompt_len as f64 / prefill_time.as_secs_f64();
        let decode_tok_s = gen_count as f64 / gen_time.as_secs_f64();
        results.push((gen_count, prefill_tok_s, decode_tok_s));
    }

    // Sustained decode
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
    let sustained_elapsed = start.elapsed();
    let sustained_tok_s = 512.0 / sustained_elapsed.as_secs_f64();

    // --- Print results ---
    let hw = get_hw_info();
    let git_info = get_git_info();
    let date = get_date();

    if json_mode {
        // Machine-readable JSON output
        print!("{{");
        print!("\"date\":\"{date}\",");
        print!("\"commit\":\"{}\",", json_escape(&git_info));
        print!("\"hardware\":\"{}\",", json_escape(&hw));
        print!("\"backend\":\"{}\",", json_escape(&backend_label));
        print!("\"model\":\"{}\",", json_escape(&model_info));
        print!("\"load_secs\":{:.3},", load_time.as_secs_f64());
        print!("\"prompt_len\":{prompt_len},");
        print!("\"generation\":[");
        for (i, &(gen_count, prefill, decode)) in results.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"gen_count\":{gen_count},\"prefill_tok_s\":{prefill:.2},\"decode_tok_s\":{decode:.2}}}"
            );
        }
        print!("],");
        print!("\"sustained_512_tok_s\":{sustained_tok_s:.2}");
        println!("}}");
    } else {
        println!();
        println!("Flare E2E Benchmark");
        println!("========================================");
        println!("Date:     {date}");
        println!("Commit:   {git_info}");
        println!("Hardware: {hw}");
        println!("Backend:  {backend_label}");
        println!("Model:    {model_info}");
        println!("Load:     {:.2}s", load_time.as_secs_f64());
        println!();
        println!("Generation speed (greedy, prompt={prompt_len} tokens):");
        for &(gen_count, prefill, decode) in &results {
            println!(
                "  gen={gen_count:>3}  prefill: {prefill:>7.1} tok/s  decode: {decode:>6.1} tok/s"
            );
        }
        println!();
        println!("Sustained decode (512 tokens): {sustained_tok_s:.1} tok/s");
    }

    // --- Append to history log ---
    if log_mode {
        let entry = format_log_entry(
            &date,
            &git_info,
            &hw,
            &model_info,
            load_time.as_secs_f64(),
            &results,
            sustained_tok_s,
        );

        append_to_history(&entry);
        eprintln!();
        eprintln!("Results appended to {HISTORY_FILE}");
    } else {
        eprintln!();
        eprintln!("Tip: run with --log to append results to {HISTORY_FILE}");
    }
}

fn run_speculative_bench(model: &mut Model, backend_label: &str, json_mode: bool) {
    // Use a repetitive prompt that encourages pattern-heavy output (code with
    // loops, counters, and repeated structures).  Since we feed raw token IDs,
    // we build a short prompt followed by a repetitive seed to prime the
    // n-gram cache for speculation.
    let prompt_text = "Write a Python function that prints numbers 1 to 100:";

    // Build a prompt token sequence with a repetitive tail so the n-gram
    // cache has patterns to latch onto during generation.
    let base: Vec<u32> = vec![1, 100, 200, 300, 400, 500, 600];
    let repeat_pattern: Vec<u32> = vec![10, 20, 30, 10, 20, 30, 10, 20, 30];
    let prompt_tokens: Vec<u32> = base.into_iter().chain(repeat_pattern).collect();
    let prompt_len = prompt_tokens.len();

    let gen_count: usize = 256;

    let hw = get_hw_info();
    let git_info = get_git_info();
    let date = get_date();

    // --- Run WITHOUT speculation ---
    model.reset();
    let params_off = SamplingParams {
        temperature: 0.0,
        speculative: false,
        ..Default::default()
    };
    let mut gen_off = Generator::new(model, params_off);
    gen_off.prefill(&prompt_tokens);

    let start_off = Instant::now();
    for _ in 0..gen_count {
        gen_off.step(0.5);
    }
    let elapsed_off = start_off.elapsed();
    let tok_s_off = gen_count as f64 / elapsed_off.as_secs_f64();

    // --- Run WITH speculation ---
    model.reset();
    let params_on = SamplingParams {
        temperature: 0.0,
        speculative: true,
        ..Default::default()
    };
    let mut gen_on = Generator::new(model, params_on);
    gen_on.prefill(&prompt_tokens);

    let start_on = Instant::now();
    for _ in 0..gen_count {
        gen_on.step(0.5);
    }
    let elapsed_on = start_on.elapsed();
    let tok_s_on = gen_count as f64 / elapsed_on.as_secs_f64();

    let stats = gen_on.speculative_stats();
    let acceptance_pct = if stats.drafted > 0 {
        stats.accepted as f64 / stats.drafted as f64 * 100.0
    } else {
        0.0
    };
    let speedup = if tok_s_off > 0.0 {
        tok_s_on / tok_s_off
    } else {
        0.0
    };

    if json_mode {
        print!("{{");
        print!("\"date\":\"{date}\",");
        print!("\"commit\":\"{}\",", json_escape(&git_info));
        print!("\"hardware\":\"{}\",", json_escape(&hw));
        print!("\"backend\":\"{}\",", json_escape(backend_label));
        print!("\"benchmark\":\"speculative\",");
        print!("\"prompt\":\"{}\",", json_escape(prompt_text));
        print!("\"prompt_len\":{prompt_len},");
        print!("\"gen_count\":{gen_count},");
        print!("\"without_speculation_tok_s\":{tok_s_off:.2},");
        print!("\"with_speculation_tok_s\":{tok_s_on:.2},");
        print!("\"speedup\":{speedup:.3},");
        print!("\"attempts\":{},", stats.attempts);
        print!("\"drafted\":{},", stats.drafted);
        print!("\"accepted\":{},", stats.accepted);
        print!("\"acceptance_pct\":{acceptance_pct:.1}");
        println!("}}");
    } else {
        println!();
        println!("Speculative Decoding Benchmark");
        println!("================================");
        println!("Date:     {date}");
        println!("Commit:   {git_info}");
        println!("Hardware: {hw}");
        println!("Backend:  {backend_label}");
        println!("Prompt:   \"{prompt_text}\"");
        println!("Generation: {gen_count} tokens");
        println!();
        println!("Without speculation: {tok_s_off:.1} tok/s");
        println!("With speculation:    {tok_s_on:.1} tok/s ({speedup:.2}x speedup)");
        println!(
            "  Attempts: {}, Drafted: {}, Accepted: {} ({acceptance_pct:.1}% acceptance)",
            stats.attempts, stats.drafted, stats.accepted,
        );
    }
}

fn format_log_entry(
    date: &str,
    git_info: &str,
    hw: &str,
    model_info: &str,
    load_secs: f64,
    results: &[(usize, f64, f64)],
    sustained: f64,
) -> String {
    let decode_16 = results
        .iter()
        .find(|r| r.0 == 16)
        .map(|r| r.2)
        .unwrap_or(0.0);
    let decode_64 = results
        .iter()
        .find(|r| r.0 == 64)
        .map(|r| r.2)
        .unwrap_or(0.0);
    let decode_256 = results
        .iter()
        .find(|r| r.0 == 256)
        .map(|r| r.2)
        .unwrap_or(0.0);

    let mut entry = String::new();
    entry.push_str(&format!("### {date} — `{git_info}`\n\n"));
    entry.push_str(&format!("**Hardware:** {hw}  \n"));
    entry.push_str(&format!("**Model:** {model_info}  \n"));
    entry.push_str(&format!("**Load time:** {load_secs:.2}s\n\n"));
    entry.push_str("| Metric | tok/s |\n");
    entry.push_str("|---|---|\n");
    entry.push_str(&format!("| Decode (16 tok) | {decode_16:.1} |\n"));
    entry.push_str(&format!("| Decode (64 tok) | {decode_64:.1} |\n"));
    entry.push_str(&format!("| Decode (256 tok) | {decode_256:.1} |\n"));
    entry.push_str(&format!("| Sustained (512 tok) | **{sustained:.1}** |\n"));
    entry.push('\n');
    entry
}

fn append_to_history(entry: &str) {
    let path = Path::new(HISTORY_FILE);

    if !path.exists() {
        let mut f = File::create(path).expect("Failed to create history file");
        writeln!(f, "# Benchmark History").unwrap();
        writeln!(f).unwrap();
        writeln!(
            f,
            "Performance log for Flare LLM. Each entry is a snapshot at a specific commit."
        )
        .unwrap();
        writeln!(
            f,
            "Baseline model: SmolLM2-135M-Instruct Q8_0 (138MB, 30 layers, dim=576)."
        )
        .unwrap();
        writeln!(f).unwrap();
        writeln!(f, "---").unwrap();
        writeln!(f).unwrap();
    }

    let mut f = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("Failed to open history file");
    write!(f, "{entry}").expect("Failed to write entry");
}

fn get_hw_info() -> String {
    #[cfg(target_arch = "aarch64")]
    let arch = "ARM64 (NEON SIMD)";
    #[cfg(not(target_arch = "aarch64"))]
    let arch = "x86_64 (scalar)";

    // Try to get chip name on macOS
    if let Ok(output) = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if output.status.success() {
            let chip = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return format!("{chip}, {arch}");
        }
    }
    arch.to_string()
}

fn get_git_info() -> String {
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());

    let subject = Command::new("git")
        .args(["log", "-1", "--format=%s"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                // Truncate long commit messages
                if s.len() > 60 {
                    Some(format!("{}...", &s[..57]))
                } else {
                    Some(s)
                }
            } else {
                None
            }
        })
        .unwrap_or_default();

    if subject.is_empty() {
        hash
    } else {
        format!("{hash} {subject}")
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn get_date() -> String {
    Command::new("date")
        .args(["+%Y-%m-%d %H:%M"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}

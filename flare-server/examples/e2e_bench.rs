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
//! Set MODEL_PATH env var to use a different model:
//!   MODEL_PATH=path/to/model.gguf cargo run -p flarellm-server --example e2e_bench --release

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

const DEFAULT_MODEL_DIR: &str = "models";
const DEFAULT_MODEL_NAME: &str = "smollm2-135m-instruct-q8_0.gguf";
const HISTORY_FILE: &str = "BENCHMARK_HISTORY.md";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_mode = args.iter().any(|a| a == "--log");
    let json_mode = args.iter().any(|a| a == "--json");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| format!("{DEFAULT_MODEL_DIR}/{DEFAULT_MODEL_NAME}"));

    if !Path::new(&model_path).exists() {
        eprintln!("Model not found at: {model_path}");
        eprintln!();
        eprintln!("Download SmolLM2-135M Q8_0:");
        eprintln!("  mkdir -p models");
        eprintln!("  huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \\");
        eprintln!("    SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models");
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
    let model_info = format!(
        "{:?}, ~{}M params, {} layers, dim={}",
        config.architecture,
        config.estimate_param_count() / 1_000_000,
        config.num_layers,
        config.hidden_dim,
    );
    eprintln!("Loaded in {:.2}s ({model_info})", load_time.as_secs_f64());

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

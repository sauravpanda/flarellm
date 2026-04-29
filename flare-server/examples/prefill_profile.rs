//! Per-phase prefill profile for a 32-token prompt on SmolLM2-135M Q8_0.
//!
//! Mirrors what the BrowserAI bench prints from JS, so we can compare native vs
//! browser numbers and validate kernel-level optimizations (gate_up fusion,
//! silu_mul_into, etc.) without needing the browser harness.
//!
//!   cargo run -p flarellm-server --example prefill_profile --release

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use flare_core::model::Model;
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

const MODEL: &str = "models/smollm2-135m-instruct-q8_0.gguf";
const PROMPT_LEN: usize = 32;
const ITERATIONS: usize = 5;

fn now_ms() -> f64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    now.as_secs_f64() * 1000.0
}

fn main() {
    if !Path::new(MODEL).exists() {
        eprintln!("missing {MODEL} — run scripts/download_baseline_model.sh first");
        std::process::exit(1);
    }

    eprintln!("loading {MODEL}");
    let file = File::open(MODEL).unwrap();
    let mut reader = BufReader::new(file);
    let gguf = GgufFile::parse_header(&mut reader).unwrap();
    let config = gguf.to_model_config().unwrap();
    let weights = load_model_weights(&gguf, &mut reader).unwrap();
    let mut model = Model::new(config.clone(), weights);

    {
        let file2 = File::open(MODEL).unwrap();
        let mut reader2 = BufReader::new(file2);
        let gguf2 = GgufFile::parse_header(&mut reader2).unwrap();
        let n = config.num_layers;
        let mut raw = Vec::with_capacity(n);
        let mut ok = true;
        for li in 0..n {
            match gguf2.load_raw_layer_weights(&mut reader2, li) {
                Ok(Some(rw)) => raw.push(rw),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && raw.len() == n {
            model.set_raw_weights(raw);
            if let Ok(Some(rw)) = gguf2
                .read_raw_weight(&mut reader2, "output.weight")
                .or_else(|_| gguf2.read_raw_weight(&mut reader2, "lm_head.weight"))
            {
                model.set_raw_output_weight(rw);
            }
        } else {
            eprintln!("warning: raw weight load failed; running with f32 fallback");
        }
    }

    model.build_fused_f32_weights();
    model.warmup();

    let prompt: Vec<u32> = (1..=(PROMPT_LEN as u32)).collect();

    let mut totals: Vec<f32> = Vec::with_capacity(ITERATIONS);
    let mut last_print = String::new();

    for it in 0..ITERATIONS {
        model.reset();
        model.enable_prefill_profiling(now_ms);
        let _ = model.forward_prefill(&prompt);
        let profile = model.take_prefill_profile().unwrap();
        totals.push(profile.total_ms);

        let line = format!(
            "iter {} | total {:.2} ms | embed {:.2} attn_norm {:.2} qkv {:.2} rope {:.2} attn {:.2} attn_out {:.2} ffn_norm {:.2} gate_up {:.2} silu_mul {:.2} down {:.2} resid {:.2} kv_write {:.2} final_norm {:.2} lm_head {:.2}",
            it,
            profile.total_ms,
            profile.embed_ms,
            profile.attn_norm_ms,
            profile.qkv_proj_ms,
            profile.rope_ms,
            profile.attention_ms,
            profile.attn_out_proj_ms,
            profile.ffn_norm_ms,
            profile.gate_up_ms,
            profile.silu_mul_ms,
            profile.down_ms,
            profile.residual_ms,
            profile.kv_write_ms,
            profile.final_norm_ms,
            profile.lm_head_ms,
        );
        println!("{line}");
        last_print = line;
    }

    let _ = last_print;
    let mean = totals.iter().sum::<f32>() / totals.len() as f32;
    let mut sorted = totals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    println!(
        "\nsummary: mean {:.2} ms | median {:.2} ms | min {:.2} ms | max {:.2} ms (over {} iters, {} tokens)",
        mean,
        median,
        sorted[0],
        sorted[sorted.len() - 1],
        ITERATIONS,
        PROMPT_LEN
    );
}

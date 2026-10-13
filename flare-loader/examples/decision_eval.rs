//! Run the held-out bounded-decision suite with one native CPU model load.
use flare_core::{
    decision::{decide, DecisionRequest},
    model::Model,
    tokenizer::BpeTokenizer,
};
use flare_loader::{gguf::GgufFile, weights::load_model_weights_with_raw_opt};
use std::{
    fs::{self, File},
    io::BufReader,
    sync::OnceLock,
    time::Instant,
};
// Monotonic milliseconds for the existing per-phase profiler.
fn now_ms() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let profiling = args.len() == 6 && args[5] == "--profile";
    if args.len() != 5 && !profiling {
        return Err(
            "usage: decision_eval MODEL TOKENIZER REQUESTS.json OUTPUT.json [--profile]".into(),
        );
    }
    let start = Instant::now();
    let mut reader = BufReader::new(File::open(&args[1])?);
    let gguf = GgufFile::parse_header(&mut reader)?;
    let mut config = gguf.to_model_config()?;
    config.max_seq_len = 512;
    let (weights, raw) = load_model_weights_with_raw_opt(&gguf, &mut reader, true)?;
    let mut model = Model::new(config, weights);
    model.set_raw_weights(raw.ok_or("missing raw weights")?);
    let tokenizer = BpeTokenizer::from_json(&fs::read_to_string(&args[2])?)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let reference: serde_json::Value = serde_json::from_str(&fs::read_to_string(&args[3])?)?;
    let mut records = Vec::new();
    for case in reference["records"].as_array().ok_or("no records")? {
        let request: DecisionRequest = serde_json::from_value(case["request"].clone())?;
        if profiling {
            model.enable_prefill_profiling(now_ms);
        }
        let start = Instant::now();
        let result = decide(&mut model, &tokenizer, &request);
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let mut record =
            serde_json::json!({"id":case["id"],"order":case["order"],"elapsedMs":elapsed});
        if let Some(p) = model.take_prefill_profile() {
            record["prefillProfile"] = serde_json::json!({
                "seqLen": p.seq_len, "numLayers": p.num_layers, "totalMs": p.total_ms,
                "embedMs": p.embed_ms, "attnNormMs": p.attn_norm_ms,
                "qkvProjMs": p.qkv_proj_ms, "ropeMs": p.rope_ms,
                "attentionMs": p.attention_ms, "attnOutProjMs": p.attn_out_proj_ms,
                "ffnNormMs": p.ffn_norm_ms, "gateUpMs": p.gate_up_ms,
                "siluMulMs": p.silu_mul_ms, "downMs": p.down_ms,
                "residualMs": p.residual_ms, "kvWriteMs": p.kv_write_ms,
                "finalNormMs": p.final_norm_ms, "lmHeadMs": p.lm_head_ms,
                "outsidePrefillMs": elapsed - f64::from(p.total_ms),
            });
        }
        match result {
            Ok(result) => {
                record["result"] = serde_json::to_value(result)?;
            }
            Err(error) => {
                record["error"] = error.to_string().into();
            }
        }
        eprintln!("{} order {}: {:.1} ms", case["id"], case["order"], elapsed);
        records.push(record);
    }
    fs::write(
        &args[4],
        serde_json::to_vec_pretty(
            &serde_json::json!({"backend":"native CPU","loadMs":load_ms,"records":records}),
        )?,
    )?;
    Ok(())
}

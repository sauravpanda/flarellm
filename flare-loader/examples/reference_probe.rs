//! Emit CPU logits with explicit input IDs for comparison with an external engine.
use flare_core::model::Model;
use flare_loader::{gguf::GgufFile, weights::load_model_weights_with_raw};
use std::{fs::File, io::BufReader};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: reference_probe MODEL.gguf IDS.json OUTPUT.json [raw]".into());
    }
    let mut reader = BufReader::new(File::open(&args[1])?);
    let gguf = GgufFile::parse_header(&mut reader)?;
    let mut config = gguf.to_model_config()?;
    config.max_seq_len = 256;
    let (weights, raw) = load_model_weights_with_raw(&gguf, &mut reader)?;
    let mut model = Model::new(config, weights);
    if args.get(4).is_some_and(|x| x == "raw") {
        model.set_raw_weights(raw.ok_or("missing raw weights")?);
    }
    let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let mut logits = model.forward_prefill(&ids);
    let mut steps = Vec::new();
    for step in 0..16 {
        let token = logits
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("no logits")?
            .0 as u32;
        steps.push(serde_json::json!({"token": token, "logits": logits.data()}));
        if token == 2 {
            break;
        }
        logits = model.forward(token, ids.len() + step);
    }
    let result = serde_json::json!({"promptIds":ids, "steps":steps});
    std::fs::write(&args[3], serde_json::to_vec(&result)?)?;
    println!(
        "Wrote {} steps",
        result["steps"].as_array().ok_or("missing steps")?.len()
    );
    Ok(())
}

//! Diagnostic Qwen3 Q8 scorer for frozen experimental prompt IDs.
//! This is an opt-in example, not an unchecked application API.
use flare_core::{config::Architecture, decision::normalize, model::Model};
use flare_loader::{gguf::GgufFile, weights::load_model_weights_with_raw_opt};
use std::{
    fs::{self, File},
    io::BufReader,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: decision_scores MODEL REQUESTS.json OUTPUT.json".into());
    }
    let requests: serde_json::Value = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    let mut reader = BufReader::new(File::open(&args[1])?);
    let gguf = GgufFile::parse_header(&mut reader)?;
    let mut config = gguf.to_model_config()?;
    if config.architecture != Architecture::Qwen3 {
        return Err("diagnostic requires Qwen3".into());
    }
    config.max_seq_len = 512;
    let vocab = config.vocab_size;
    let (weights, raw) = load_model_weights_with_raw_opt(&gguf, &mut reader, true)?;
    let mut model = Model::new(config, weights);
    model.set_raw_weights(raw.ok_or("missing raw weights")?);
    let mut records = Vec::new();
    for request in requests["records"].as_array().ok_or("missing records")? {
        let ids: Vec<u32> = serde_json::from_value(request["promptIds"].clone())?;
        let labels: Vec<u32> = serde_json::from_value(request["labelIds"].clone())?;
        let choices: Vec<String> = serde_json::from_value(request["request"]["choices"].clone())?;
        if ids.is_empty()
            || ids.len() > 512
            || !(2..=8).contains(&labels.len())
            || labels.len() != choices.len()
            || ids.iter().chain(&labels).any(|&id| id as usize >= vocab)
            || labels
                .iter()
                .enumerate()
                .any(|(i, id)| labels[..i].contains(id))
        {
            return Err("invalid explicit prompt/label IDs".into());
        }
        model.reset();
        let start = Instant::now();
        let output = model.forward_prefill(&ids);
        let logits = output.data();
        if logits.len() != vocab || !logits.iter().all(|x| x.is_finite()) {
            return Err("invalid full-vocabulary logits".into());
        }
        let selected: Vec<f32> = labels.iter().map(|&id| logits[id as usize]).collect();
        let (index, probabilities) = normalize(&selected)?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f64 = logits
            .iter()
            .map(|&z| (f64::from(z) - f64::from(max)).exp())
            .sum();
        let mass: f64 = selected
            .iter()
            .map(|&z| (f64::from(z) - f64::from(max)).exp())
            .sum();
        let scores: Vec<_> = choices.iter().zip(&selected).zip(&probabilities)
            .map(|((choice, logit), score)| serde_json::json!({"choice":choice,"logit":logit,"score":score})).collect();
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        records.push(
            serde_json::json!({"id":request["id"],"order":request["order"],
            "variant":request["variant"],"promptIds":ids,"labelIds":labels,
            "elapsedMs":elapsed,"labelMass":mass/total,
            "result":{"choice":choices[index],"scores":scores}}),
        );
        fs::write(
            &args[3],
            serde_json::to_vec_pretty(&serde_json::json!({
                "backend":"Flare native CPU, Q8 weights/f32 activations and KV", "records":records
            }))?,
        )?;
        eprintln!(
            "{} order {}: {:.1} ms",
            request["id"], request["order"], elapsed
        );
    }
    Ok(())
}

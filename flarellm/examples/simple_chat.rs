//! Minimal generation example using the flarellm umbrella crate.
//!
//! Run with:
//!   cargo run -p flarellm --example simple_chat --release -- path/to/model.gguf
//!
//! This shows the flow: load GGUF → run greedy generation from prompt token IDs.
//! Real chat apps should use a proper BPE/SentencePiece tokenizer to encode
//! the user text. For Llama-style models, see flarellm-server's CLI which uses
//! the GGUF-embedded vocabulary directly.

use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

use flarellm::core::generate::Generator;
use flarellm::core::model::Model;
use flarellm::core::sampling::SamplingParams;
use flarellm::loader::gguf::GgufFile;
use flarellm::loader::tokenizer::GgufVocab;
use flarellm::loader::weights::load_model_weights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            eprintln!("Usage: simple_chat <model.gguf>");
            std::process::exit(1);
        }
    };

    println!("Loading {model_path}...");
    let load_start = Instant::now();
    let mut reader = BufReader::new(File::open(&model_path)?);
    let gguf = GgufFile::parse_header(&mut reader)?;
    let config = gguf.to_model_config()?;
    let weights = load_model_weights(&gguf, &mut reader)?;
    let vocab = GgufVocab::from_gguf(&gguf)?;
    let mut model = Model::new(config.clone(), weights);
    println!(
        "Loaded {:?} {}M params, {} layers in {:.2}s",
        config.architecture,
        config.estimate_param_count() / 1_000_000,
        config.num_layers,
        load_start.elapsed().as_secs_f64(),
    );

    // Use BOS as the prompt — most basic possible. Real apps should encode
    // user text with a proper BPE tokenizer.
    let bos = vocab.bos_id.unwrap_or(1);
    let prompt_tokens = vec![bos];

    println!("\nGenerating 30 tokens from BOS (token {bos})...\n");
    let gen_start = Instant::now();

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut gen = Generator::new(&mut model, params);

    let mut count = 0;
    gen.generate(
        &prompt_tokens,
        30,
        None, // ignore EOS for demo
        || 0.5,
        |token, _step| {
            let s = vocab.decode(&[token]);
            print!("{s}");
            let _ = std::io::stdout().flush();
            count += 1;
            true
        },
    );
    let elapsed = gen_start.elapsed();
    let tok_s = count as f64 / elapsed.as_secs_f64();
    println!();
    println!(
        "\n{count} tokens in {:.0}ms ({tok_s:.1} tok/s)",
        elapsed.as_millis()
    );

    Ok(())
}

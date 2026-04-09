use std::fs::File;
use std::io::{self, BufReader, Write};
use std::time::Instant;

use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_loader::gguf::GgufFile;
use flare_loader::weights::load_model_weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: flare-cli <model.gguf> [tokenizer.json] [--prompt \"...\"]");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  flare-cli model.gguf");
        eprintln!("  flare-cli model.gguf tokenizer.json --prompt \"Hello world\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let tokenizer_path = args.get(2).filter(|s| !s.starts_with("--"));
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Load GGUF
    eprintln!("Loading model from {}...", model_path);
    let start = Instant::now();

    let mut file = BufReader::new(File::open(model_path).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", model_path, e);
        std::process::exit(1);
    }));

    let gguf = GgufFile::parse_header(&mut file).unwrap_or_else(|e| {
        eprintln!("Error parsing GGUF: {}", e);
        std::process::exit(1);
    });

    let config = gguf.to_model_config().unwrap_or_else(|e| {
        eprintln!("Error extracting config: {}", e);
        std::process::exit(1);
    });

    eprintln!("Architecture: {:?}", config.architecture);
    eprintln!(
        "Parameters:   ~{}M",
        config.estimate_param_count() / 1_000_000
    );
    eprintln!("Layers:       {}", config.num_layers);
    eprintln!("Hidden dim:   {}", config.hidden_dim);
    eprintln!(
        "Heads:        {} (KV: {})",
        config.num_heads, config.num_kv_heads
    );
    eprintln!("Vocab:        {}", config.vocab_size);
    eprintln!("Context:      {}", config.max_seq_len);

    eprintln!("Loading weights...");
    let weights = load_model_weights(&gguf, &mut file).unwrap_or_else(|e| {
        eprintln!("Error loading weights: {}", e);
        std::process::exit(1);
    });

    let mut model = Model::new(config.clone(), weights);
    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());

    // Load tokenizer
    let tokenizer: Option<BpeTokenizer> = tokenizer_path.map(|path| {
        eprintln!("Loading tokenizer from {}...", path);
        BpeTokenizer::from_file(path).unwrap_or_else(|e| {
            eprintln!("Error loading tokenizer: {}", e);
            std::process::exit(1);
        })
    });

    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
    };

    // Simple RNG (xorshift32)
    let mut rng_state: u32 = 0xDEADBEEF
        ^ (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos());
    let mut rng = move || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        (rng_state as f32) / (u32::MAX as f32)
    };

    if let Some(prompt_text) = prompt {
        // Single prompt mode
        generate_text(
            &mut model,
            tokenizer.as_ref(),
            prompt_text,
            &params,
            256,
            &mut rng,
        );
    } else {
        // Interactive mode
        eprintln!("\nEntering interactive mode. Type your prompt and press Enter.");
        eprintln!("Type 'quit' or Ctrl+C to exit.\n");

        loop {
            eprint!("> ");
            io::stderr().flush().ok();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() || input.trim() == "quit" {
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            model.reset();
            generate_text(
                &mut model,
                tokenizer.as_ref(),
                input,
                &params,
                256,
                &mut rng,
            );
            println!();
        }
    }
}

fn generate_text(
    model: &mut Model,
    tokenizer: Option<&BpeTokenizer>,
    prompt: &str,
    params: &SamplingParams,
    max_tokens: usize,
    rng: &mut dyn FnMut() -> f32,
) {
    // Encode prompt
    let prompt_tokens = if let Some(tok) = tokenizer {
        match tok.encode(prompt) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("Tokenization error: {}", e);
                return;
            }
        }
    } else {
        // Fallback: use bytes as token IDs (only useful for testing)
        eprintln!("Warning: no tokenizer loaded, using byte-level encoding");
        prompt.bytes().map(|b| b as u32).collect()
    };

    eprintln!("Prompt: {} tokens", prompt_tokens.len(),);

    let eos_token = tokenizer.and_then(|t| t.eos_token_id());

    let start = Instant::now();
    let mut gen = Generator::new(model, params.clone());

    let generated = gen.generate(
        &prompt_tokens,
        max_tokens,
        eos_token,
        rng,
        |token_id, _step| {
            // Stream tokens as they're generated
            if let Some(tok) = tokenizer {
                if let Ok(text) = tok.decode(&[token_id]) {
                    print!("{}", text);
                    io::stdout().flush().ok();
                }
            }
            true
        },
    );

    let elapsed = start.elapsed().as_secs_f32();
    eprintln!(
        "\n[{} tokens in {:.1}s, {:.1} tok/s]",
        generated.len(),
        elapsed,
        generated.len() as f32 / elapsed,
    );
}

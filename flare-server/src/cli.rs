use std::fs::File;
use std::io::{self, BufReader, Write};
use std::time::Instant;

use flare_core::chat::{ChatMessage, ChatTemplate, Role};
use flare_core::generate::Generator;
use flare_core::model::Model;
use flare_core::sampling::SamplingParams;
use flare_core::tokenizer::{BpeTokenizer, Tokenizer};
use flare_loader::gguf::GgufFile;
use flare_loader::tokenizer::GgufVocab;
use flare_loader::weights::load_model_weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: flare-cli <model.gguf> [tokenizer.json] [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --prompt \"...\"    Single prompt mode");
        eprintln!("  --chat            Chat mode with proper templates");
        eprintln!("  --system \"...\"    System prompt for chat mode");
        eprintln!("  --temp <float>    Temperature (default: 0.7)");
        eprintln!("  --max-tokens <n>  Max tokens to generate (default: 256)");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  flare-cli model.gguf tokenizer.json --prompt \"Hello\"");
        eprintln!("  flare-cli model.gguf tokenizer.json --chat");
        eprintln!("  flare-cli model.gguf tokenizer.json --chat --system \"You are a pirate\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let tokenizer_path = args.get(2).filter(|s| !s.starts_with("--"));
    let prompt = get_arg(&args, "--prompt");
    let chat_mode = args.iter().any(|a| a == "--chat");
    let system_prompt = get_arg(&args, "--system");
    let temperature: f32 = get_arg(&args, "--temp")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let max_tokens: usize = get_arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

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

    // Load tokenizer (from file or GGUF metadata)
    let tokenizer: Option<BpeTokenizer> = tokenizer_path.map(|path| {
        eprintln!("Loading tokenizer from {}...", path);
        BpeTokenizer::from_file(path).unwrap_or_else(|e| {
            eprintln!("Error loading tokenizer: {}", e);
            std::process::exit(1);
        })
    });

    // Fall back to GGUF-embedded vocabulary for decoding
    let gguf_vocab = if tokenizer.is_none() {
        match GgufVocab::from_gguf(&gguf) {
            Ok(vocab) => {
                eprintln!(
                    "Using GGUF-embedded vocabulary ({} tokens, bos={:?}, eos={:?})",
                    vocab.vocab_size, vocab.bos_id, vocab.eos_id
                );
                Some(vocab)
            }
            Err(e) => {
                eprintln!("No tokenizer available: {e}");
                None
            }
        }
    } else {
        None
    };

    let params = SamplingParams {
        temperature,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        min_p: 0.0,
    };

    // Detect chat template from GGUF metadata or architecture
    let arch = gguf.architecture().unwrap_or("llama");
    let template = match gguf.metadata.get("tokenizer.chat_template") {
        Some(flare_loader::gguf::MetadataValue::String(tmpl)) => {
            let detected = ChatTemplate::from_gguf_template(tmpl, arch);
            eprintln!("Chat template: {:?} (from GGUF metadata)", detected);
            detected
        }
        _ => {
            let detected = ChatTemplate::from_architecture(arch);
            eprintln!("Chat template: {:?} (from architecture)", detected);
            detected
        }
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

    if let Some(prompt_text) = prompt.as_deref() {
        if chat_mode {
            // Single chat prompt with template formatting
            let mut messages = Vec::new();
            if let Some(sys) = &system_prompt {
                messages.push(ChatMessage {
                    role: Role::System,
                    content: sys.clone(),
                });
            }
            messages.push(ChatMessage {
                role: Role::User,
                content: prompt_text.to_string(),
            });
            let formatted = template.apply(&messages);
            generate_text(
                &mut model,
                tokenizer.as_ref(),
                gguf_vocab.as_ref(),
                &formatted,
                &params,
                max_tokens,
                &mut rng,
            );
        } else {
            generate_text(
                &mut model,
                tokenizer.as_ref(),
                gguf_vocab.as_ref(),
                prompt_text,
                &params,
                max_tokens,
                &mut rng,
            );
        }
    } else if chat_mode {
        // Interactive chat mode with conversation history
        eprintln!(
            "\nEntering chat mode ({:?} template). Type your message and press Enter.",
            template
        );
        eprintln!("Type 'quit' or Ctrl+C to exit. Type '/reset' to clear history.\n");

        let mut history: Vec<ChatMessage> = Vec::new();
        if let Some(sys) = &system_prompt {
            history.push(ChatMessage {
                role: Role::System,
                content: sys.clone(),
            });
        }

        loop {
            eprint!("You> ");
            io::stderr().flush().ok();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() || input.trim() == "quit" {
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            if input == "/reset" {
                history.retain(|m| m.role == Role::System);
                model.reset();
                eprintln!("(conversation reset)");
                continue;
            }

            history.push(ChatMessage {
                role: Role::User,
                content: input.to_string(),
            });
            let formatted = template.apply(&history);

            model.reset();
            eprint!("AI> ");
            io::stderr().flush().ok();
            let response = generate_and_collect(
                &mut model,
                tokenizer.as_ref(),
                gguf_vocab.as_ref(),
                &formatted,
                &params,
                max_tokens,
                &mut rng,
            );
            println!();

            history.push(ChatMessage {
                role: Role::Assistant,
                content: response,
            });
        }
    } else {
        // Interactive raw prompt mode
        eprintln!("\nEntering interactive mode. Type your prompt and press Enter.");
        eprintln!("Type 'quit' or Ctrl+C to exit. Use --chat for conversation mode.\n");

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
                gguf_vocab.as_ref(),
                input,
                &params,
                max_tokens,
                &mut rng,
            );
            println!();
        }
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Encode text to tokens using available tokenizer, or byte-level fallback.
fn encode_prompt(
    text: &str,
    tokenizer: Option<&BpeTokenizer>,
    vocab: Option<&GgufVocab>,
) -> Vec<u32> {
    if let Some(tok) = tokenizer {
        match tok.encode(text) {
            Ok(tokens) => return tokens,
            Err(e) => eprintln!("BPE tokenization error: {e}"),
        }
    }
    // Byte-level fallback: look up each byte in GGUF vocab if available
    if let Some(v) = vocab {
        // Try encoding as byte tokens <0xHH>
        text.bytes()
            .map(|b| {
                let byte_token = format!("<0x{b:02X}>");
                v.encode_token(&byte_token).unwrap_or(b as u32)
            })
            .collect()
    } else {
        text.bytes().map(|b| b as u32).collect()
    }
}

/// Decode a token ID to text using available tokenizer.
fn decode_token(
    token_id: u32,
    tokenizer: Option<&BpeTokenizer>,
    vocab: Option<&GgufVocab>,
) -> Option<String> {
    if let Some(tok) = tokenizer {
        return tok.decode(&[token_id]).ok();
    }
    vocab.map(|v| v.decode(&[token_id]))
}

/// Get EOS token ID from available tokenizer.
fn get_eos(tokenizer: Option<&BpeTokenizer>, vocab: Option<&GgufVocab>) -> Option<u32> {
    if let Some(tok) = tokenizer {
        return tok.eos_token_id();
    }
    vocab.and_then(|v| v.eos_id)
}

fn generate_text(
    model: &mut Model,
    tokenizer: Option<&BpeTokenizer>,
    vocab: Option<&GgufVocab>,
    prompt: &str,
    params: &SamplingParams,
    max_tokens: usize,
    rng: &mut dyn FnMut() -> f32,
) {
    let prompt_tokens = encode_prompt(prompt, tokenizer, vocab);
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let eos_token = get_eos(tokenizer, vocab);

    let start = Instant::now();
    let mut gen = Generator::new(model, params.clone());

    let generated = gen.generate(
        &prompt_tokens,
        max_tokens,
        eos_token,
        rng,
        |token_id, _step| {
            if let Some(text) = decode_token(token_id, tokenizer, vocab) {
                print!("{text}");
                io::stdout().flush().ok();
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

/// Like generate_text but collects the output string for chat history.
fn generate_and_collect(
    model: &mut Model,
    tokenizer: Option<&BpeTokenizer>,
    vocab: Option<&GgufVocab>,
    prompt: &str,
    params: &SamplingParams,
    max_tokens: usize,
    rng: &mut dyn FnMut() -> f32,
) -> String {
    let prompt_tokens = encode_prompt(prompt, tokenizer, vocab);
    let eos_token = get_eos(tokenizer, vocab);
    let mut collected = String::new();

    let start = Instant::now();
    let mut gen = Generator::new(model, params.clone());

    let generated = gen.generate(
        &prompt_tokens,
        max_tokens,
        eos_token,
        rng,
        |token_id, _step| {
            if let Some(text) = decode_token(token_id, tokenizer, vocab) {
                print!("{text}");
                io::stdout().flush().ok();
                collected.push_str(&text);
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

    collected
}

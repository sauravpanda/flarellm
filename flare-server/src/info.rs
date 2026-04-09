use std::fs::File;
use std::io::BufReader;

use flare_loader::gguf::GgufFile;
use flare_loader::QuantFormat;

fn format_bytes(bytes: usize) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    const KB: f64 = 1_024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn quant_label(q: &QuantFormat) -> &'static str {
    match q {
        QuantFormat::F32 => "F32",
        QuantFormat::F16 => "F16",
        QuantFormat::Q4_0 => "Q4_0",
        QuantFormat::Q4_1 => "Q4_1",
        QuantFormat::Q5_0 => "Q5_0",
        QuantFormat::Q5_1 => "Q5_1",
        QuantFormat::Q8_0 => "Q8_0",
        QuantFormat::Q8_1 => "Q8_1",
        QuantFormat::Q2K => "Q2_K",
        QuantFormat::Q3K => "Q3_K",
        QuantFormat::Q4K => "Q4_K",
        QuantFormat::Q5K => "Q5_K",
        QuantFormat::Q6K => "Q6_K",
        QuantFormat::Unknown(id) => {
            // Leak a string for the unknown case; only happens for unusual formats.
            let s = format!("Unknown({id})");
            Box::leak(s.into_boxed_str())
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: flare-info <model.gguf>");
        eprintln!();
        eprintln!("Inspect a GGUF model file: print architecture, memory estimates,");
        eprintln!("and tensor layout without loading weights into memory.");
        std::process::exit(1);
    }

    let model_path = &args[1];

    let mut file = BufReader::new(File::open(model_path).unwrap_or_else(|e| {
        eprintln!("Error opening {model_path}: {e}");
        std::process::exit(1);
    }));

    let gguf = GgufFile::parse_header(&mut file).unwrap_or_else(|e| {
        eprintln!("Error parsing GGUF header: {e}");
        std::process::exit(1);
    });

    // --- Model info -----------------------------------------------------------

    let arch_str = gguf.architecture().unwrap_or("unknown");

    let config = gguf.to_model_config().ok();

    println!("=== Model Info ===");
    println!("GGUF version:   {}", gguf.version);
    println!("Architecture:   {arch_str}");

    if let Some(cfg) = &config {
        let params = cfg.estimate_param_count();
        println!("Parameters:     ~{:.2}M ({params})", params as f64 / 1e6);
        println!("Layers:         {}", cfg.num_layers);
        println!("Hidden dim:     {}", cfg.hidden_dim);
        println!("Heads:          {}", cfg.num_heads);
        println!("KV heads:       {}", cfg.num_kv_heads);
        println!("Vocab size:     {}", cfg.vocab_size);
        println!("Context length: {}", cfg.max_seq_len);
    } else {
        println!("(Could not extract full model config from metadata)");
    }

    // --- Memory estimates -----------------------------------------------------

    if let Some(cfg) = &config {
        println!();
        println!("=== Weight Memory Estimates ===");
        for (label, bpw) in [
            ("Q4 (4.5 bpw)", 4.5_f32),
            ("Q8 (8.5 bpw)", 8.5),
            ("F16", 16.0),
        ] {
            let mem = cfg.estimate_weight_memory(bpw);
            println!("  {label:<16} {}", format_bytes(mem));
        }

        println!();
        println!("=== KV Cache Memory (F16) ===");
        for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
            if ctx > cfg.max_seq_len {
                break;
            }
            let mem = cfg.estimate_kv_cache_memory(ctx, 16.0);
            println!("  ctx {ctx:<8} {}", format_bytes(mem));
        }

        println!();
        println!("=== KV Cache Memory (Q8) ===");
        for ctx in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
            if ctx > cfg.max_seq_len {
                break;
            }
            let mem = cfg.estimate_kv_cache_memory(ctx, 8.0);
            println!("  ctx {ctx:<8} {}", format_bytes(mem));
        }
    }

    // --- Actual file weight size ----------------------------------------------

    let total_weight_bytes: u64 = gguf.tensors.iter().map(|t| t.byte_size()).sum();
    println!();
    println!(
        "=== On-Disk Weight Size ({} tensors) ===",
        gguf.tensors.len()
    );
    println!("  Total: {}", format_bytes(total_weight_bytes as usize));

    // --- Tensor listing -------------------------------------------------------

    println!();
    println!("=== Tensors ===");
    println!(
        "{:<60} {:<14} {:<10} {:>12}",
        "Name", "Shape", "Quant", "Size"
    );
    println!("{}", "-".repeat(98));

    for t in &gguf.tensors {
        let shape_str = t
            .dimensions
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" x ");

        println!(
            "{:<60} {:<14} {:<10} {:>12}",
            t.name,
            shape_str,
            quant_label(&t.dtype),
            format_bytes(t.byte_size() as usize),
        );
    }
}

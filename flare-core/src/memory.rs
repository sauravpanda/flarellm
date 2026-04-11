//! Memory planning for model inference.
//!
//! Estimates memory requirements and checks if a model fits within
//! available memory constraints (browser WASM ~4GB, WebGPU ~2GB).

use crate::config::ModelConfig;

/// Memory budget for different deployment targets.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum total memory in bytes.
    pub total_bytes: usize,
    /// Maximum GPU buffer memory in bytes.
    pub gpu_bytes: usize,
    /// Label for this budget (e.g., "Chrome WASM", "Native").
    pub label: String,
}

impl MemoryBudget {
    /// Conservative browser budget: 2GB WASM + 1.5GB WebGPU.
    pub fn browser() -> Self {
        Self {
            total_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            gpu_bytes: 1536 * 1024 * 1024,       // 1.5GB
            label: "Browser (Chrome)".into(),
        }
    }

    /// Mobile browser budget: tighter limits.
    pub fn mobile() -> Self {
        Self {
            total_bytes: 1024 * 1024 * 1024, // 1GB
            gpu_bytes: 512 * 1024 * 1024,    // 512MB
            label: "Mobile Browser".into(),
        }
    }

    /// Native budget: effectively unlimited for small models.
    pub fn native() -> Self {
        // On wasm32, usize is 32-bit so we cap at safe values; native gets 16GB/8GB.
        #[cfg(target_pointer_width = "64")]
        const TOTAL: usize = 16 * 1024 * 1024 * 1024; // 16GB
        #[cfg(not(target_pointer_width = "64"))]
        const TOTAL: usize = 3500 * 1024 * 1024; // 3.5GB (max safe on 32-bit)

        #[cfg(target_pointer_width = "64")]
        const GPU: usize = 8 * 1024 * 1024 * 1024; // 8GB
        #[cfg(not(target_pointer_width = "64"))]
        const GPU: usize = 2 * 1024 * 1024 * 1024; // 2GB

        Self {
            total_bytes: TOTAL,
            gpu_bytes: GPU,
            label: "Native".into(),
        }
    }
}

/// Detailed memory breakdown for a model configuration.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Weight memory in bytes.
    pub weights_bytes: usize,
    /// KV cache memory in bytes.
    pub kv_cache_bytes: usize,
    /// Activation memory (intermediate buffers) in bytes.
    pub activation_bytes: usize,
    /// Total estimated memory.
    pub total_bytes: usize,
    /// Whether this plan fits in the given budget.
    pub fits: bool,
    /// Recommended context length (may be reduced to fit).
    pub recommended_context: usize,
    /// Recommended quantization bits per weight.
    pub recommended_quant_bits: f32,
}

/// Plan memory usage for a model with given quantization and context length.
pub fn plan_memory(
    config: &ModelConfig,
    quant_bits: f32,
    context_len: usize,
    budget: &MemoryBudget,
) -> MemoryPlan {
    let weights_bytes = config.estimate_weight_memory(quant_bits);
    let kv_cache_bytes = config.estimate_kv_cache_memory(context_len, 16.0); // F16 KV cache

    // Activation memory: roughly 2 * batch_size * hidden_dim * sizeof(f32)
    // For single-token generation, this is small
    let activation_bytes = 2 * config.hidden_dim * 4 // residual + hidden state
        + config.intermediate_dim * 4 * 3 // FFN gate + up + hidden
        + config.num_heads * context_len * 4; // attention scores

    let total_bytes = weights_bytes + kv_cache_bytes + activation_bytes;
    let fits = total_bytes <= budget.total_bytes;

    // Find recommended context if it doesn't fit
    let recommended_context = if fits {
        context_len
    } else {
        find_max_context(config, quant_bits, budget)
    };

    // Find recommended quantization if it doesn't fit
    let recommended_quant_bits = if fits {
        quant_bits
    } else {
        find_min_quant(config, context_len, budget)
    };

    MemoryPlan {
        weights_bytes,
        kv_cache_bytes,
        activation_bytes,
        total_bytes,
        fits,
        recommended_context,
        recommended_quant_bits,
    }
}

/// Find the maximum context length that fits in the budget.
fn find_max_context(config: &ModelConfig, quant_bits: f32, budget: &MemoryBudget) -> usize {
    let weights = config.estimate_weight_memory(quant_bits);
    let activations_base = 2 * config.hidden_dim * 4 + config.intermediate_dim * 4 * 3;
    let remaining = budget
        .total_bytes
        .saturating_sub(weights + activations_base);

    // KV cache per token: 2 * num_layers * num_kv_heads * head_dim * 2 (F16)
    let kv_per_token = 2 * config.num_layers * config.num_kv_heads * config.head_dim * 2;
    let attn_per_token = config.num_heads * 4;
    let per_token = kv_per_token + attn_per_token;

    if per_token == 0 {
        return config.max_seq_len;
    }

    let max_ctx = remaining / per_token;
    max_ctx.min(config.max_seq_len).max(128) // at least 128 tokens
}

/// Find the minimum quantization bits that fits in the budget.
fn find_min_quant(config: &ModelConfig, context_len: usize, budget: &MemoryBudget) -> f32 {
    for &bits in &[2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 16.0, 32.0] {
        let weights = config.estimate_weight_memory(bits);
        let kv = config.estimate_kv_cache_memory(context_len, 16.0);
        let activations = 2 * config.hidden_dim * 4 + config.intermediate_dim * 4 * 3;
        if weights + kv + activations <= budget.total_bytes {
            return bits;
        }
    }
    32.0 // can't fit even at lowest quant
}

/// Quick check: can this model run at all in the given budget?
pub fn can_run(config: &ModelConfig, quant_bits: f32, budget: &MemoryBudget) -> bool {
    let min_memory = config.estimate_weight_memory(quant_bits)
        + config.estimate_kv_cache_memory(128, 16.0) // minimum 128 context
        + 2 * config.hidden_dim * 4;
    min_memory <= budget.total_bytes
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.0}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;

    fn small_model() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 32000,
            hidden_dim: 2048,
            intermediate_dim: 5504,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        }
    }

    #[test]
    fn test_plan_fits_native() {
        let config = small_model();
        let plan = plan_memory(&config, 4.0, 2048, &MemoryBudget::native());
        assert!(plan.fits);
        assert!(plan.total_bytes > 0);
    }

    #[test]
    fn test_plan_fits_browser() {
        let config = small_model();
        let plan = plan_memory(&config, 4.0, 2048, &MemoryBudget::browser());
        // Small model at Q4 should fit in browser
        assert!(plan.fits);
    }

    #[test]
    fn test_plan_too_large_for_mobile() {
        let mut config = small_model();
        config.hidden_dim = 8192;
        config.num_layers = 80;
        config.intermediate_dim = 28672;
        // This is roughly a 70B model - won't fit in mobile
        let plan = plan_memory(&config, 4.0, 2048, &MemoryBudget::mobile());
        assert!(!plan.fits);
    }

    #[test]
    fn test_recommended_context() {
        let mut config = small_model();
        config.hidden_dim = 4096;
        config.num_layers = 32;
        // Force tight budget
        let tight_budget = MemoryBudget {
            total_bytes: 800 * 1024 * 1024, // 800MB
            gpu_bytes: 512 * 1024 * 1024,
            label: "tight".into(),
        };
        let plan = plan_memory(&config, 4.0, 8192, &tight_budget);
        if !plan.fits {
            assert!(plan.recommended_context < 8192);
            assert!(plan.recommended_context >= 128);
        }
    }

    #[test]
    fn test_can_run() {
        let config = small_model();
        assert!(can_run(&config, 4.0, &MemoryBudget::browser()));
        assert!(can_run(&config, 4.0, &MemoryBudget::native()));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1024), "1KB");
        assert_eq!(format_bytes(1536 * 1024), "2MB");
        assert_eq!(format_bytes(500 * 1024 * 1024), "500MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0GB");
    }
}

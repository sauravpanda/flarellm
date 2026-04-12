use serde::{Deserialize, Serialize};

/// Model architecture configuration parsed from GGUF metadata or config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Attention logit soft-cap for Gemma 2 (`tanh(score / cap) * cap`).
    /// Zero means no capping (Llama, Qwen, Mistral, Phi-3).
    #[serde(default)]
    pub attn_logit_softcap: f32,
    /// Final logit soft-cap for Gemma 2.
    /// Zero means no capping.
    #[serde(default)]
    pub final_logit_softcap: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Qwen2,
    Mistral,
    /// Phi-3 / Phi-3.5 mini — forward pass identical to Llama, different chat template.
    Phi3,
    /// Gemma 2 — post-norms, logit soft-capping, GELU activation, alternating attention.
    Gemma2,
}

impl ModelConfig {
    /// Estimate total memory needed for model weights at a given quantization.
    pub fn estimate_weight_memory(&self, bits_per_weight: f32) -> usize {
        let total_params = self.estimate_param_count();
        (total_params as f64 * bits_per_weight as f64 / 8.0) as usize
    }

    /// Estimate total parameter count.
    pub fn estimate_param_count(&self) -> usize {
        let embed = self.vocab_size * self.hidden_dim;
        let per_layer = {
            let attn_qkv =
                self.hidden_dim * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim;
            let attn_out = self.num_heads * self.head_dim * self.hidden_dim;
            let ffn = 3 * self.hidden_dim * self.intermediate_dim; // gate + up + down
            let norms = 2 * self.hidden_dim;
            attn_qkv + attn_out + ffn + norms
        };
        let output_head = self.vocab_size * self.hidden_dim;
        embed + self.num_layers * per_layer + output_head
    }

    /// Estimate KV cache memory for a given sequence length and quantization bits.
    pub fn estimate_kv_cache_memory(&self, seq_len: usize, bits_per_value: f32) -> usize {
        let kv_per_layer = 2 * seq_len * self.num_kv_heads * self.head_dim;
        let total_elements = self.num_layers * kv_per_layer;
        (total_elements as f64 * bits_per_value as f64 / 8.0) as usize
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Roughly Llama-3.2-1B dimensions
        Self {
            architecture: Architecture::Llama,
            vocab_size: 128256,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            num_layers: 16,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        // vocab=100 hidden=4 intermediate=8 layers=1 heads=2 kv_heads=1 head_dim=2
        // embed=400 per_layer=152 output=400 → total=952
        ModelConfig {
            architecture: Architecture::Llama,
            vocab_size: 100,
            hidden_dim: 4,
            intermediate_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        }
    }

    #[test]
    fn test_memory_estimates() {
        let config = ModelConfig::default();
        let weight_mem = config.estimate_weight_memory(4.0);
        // ~1B params at 4 bits = ~500MB
        assert!(weight_mem > 400_000_000);
        assert!(weight_mem < 800_000_000);
    }

    #[test]
    fn test_kv_cache_estimate() {
        let config = ModelConfig::default();
        let kv_mem = config.estimate_kv_cache_memory(2048, 8.0);
        // Should be in the tens of MB range
        assert!(kv_mem > 10_000_000);
        assert!(kv_mem < 200_000_000);
    }

    #[test]
    fn test_estimate_param_count_tiny_model() {
        // Manually computed: embed=400, per_layer=152, output=400 → 952
        // per_layer: attn_qkv=4*(2+2)*2=32, attn_out=2*2*4=16, ffn=3*4*8=96, norms=2*4=8
        let config = tiny_config();
        assert_eq!(config.estimate_param_count(), 952);
    }

    #[test]
    fn test_architecture_equality() {
        assert_eq!(Architecture::Llama, Architecture::Llama);
        assert_eq!(Architecture::Qwen2, Architecture::Qwen2);
        assert_eq!(Architecture::Mistral, Architecture::Mistral);
        assert_eq!(Architecture::Phi3, Architecture::Phi3);
        assert_eq!(Architecture::Gemma2, Architecture::Gemma2);
        assert_ne!(Architecture::Llama, Architecture::Gemma2);
    }

    #[test]
    fn test_architecture_clone() {
        let arch = Architecture::Phi3;
        let cloned = arch;
        assert_eq!(arch, cloned);
    }

    #[test]
    fn test_default_softcap_zero() {
        let config = ModelConfig::default();
        assert_eq!(config.attn_logit_softcap, 0.0);
        assert_eq!(config.final_logit_softcap, 0.0);
    }

    #[test]
    fn test_kv_cache_zero_seq_len() {
        let config = tiny_config();
        assert_eq!(config.estimate_kv_cache_memory(0, 8.0), 0);
    }

    #[test]
    fn test_kv_cache_one_token() {
        // layers=1, kv_heads=1, head_dim=2, seq_len=1, bits=8
        // kv_per_layer=2*1*1*2=4 elements, total=4, bytes=(4*8/8)=4
        let config = tiny_config();
        assert_eq!(config.estimate_kv_cache_memory(1, 8.0), 4);
    }

    #[test]
    fn test_weight_memory_scales_with_bits() {
        let config = tiny_config();
        let mem8 = config.estimate_weight_memory(8.0);
        let mem16 = config.estimate_weight_memory(16.0);
        // 16-bit should be exactly 2× 8-bit
        assert_eq!(mem16, mem8 * 2);
        // 8-bit should equal param count (1 byte per param)
        assert_eq!(mem8, config.estimate_param_count());
    }
}

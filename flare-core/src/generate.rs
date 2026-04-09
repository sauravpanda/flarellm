use crate::model::Model;
use crate::sampling::{self, SamplingParams};

/// Result of generating a single token.
pub struct GenerationStep {
    pub token_id: u32,
    pub logits: Vec<f32>,
}

/// Autoregressive text generation loop.
pub struct Generator<'a> {
    model: &'a mut Model,
    params: SamplingParams,
    tokens: Vec<u32>,
    position: usize,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a mut Model, params: SamplingParams) -> Self {
        Self {
            model,
            params,
            tokens: Vec::new(),
            position: 0,
        }
    }

    /// Prefill: process multiple prompt tokens at once (one-by-one for now).
    /// Returns logits from the last token.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &token in prompt_tokens {
            let output = self.model.forward(token, self.position);
            logits = output.data().to_vec();
            self.tokens.push(token);
            self.position += 1;
        }
        logits
    }

    /// Generate a single next token, returning the token ID.
    pub fn step(&mut self, rng_val: f32) -> GenerationStep {
        // Get logits from the last token
        let last_token = *self.tokens.last().unwrap_or(&0);
        let logits_tensor = self.model.forward(last_token, self.position);
        let mut logits = logits_tensor.data().to_vec();

        // Apply sampling transforms
        sampling::apply_repeat_penalty(&mut logits, &self.tokens, self.params.repeat_penalty);
        sampling::apply_temperature(&mut logits, self.params.temperature);

        // Sample
        let token_id = if self.params.temperature == 0.0 {
            sampling::sample_greedy(&logits)
        } else if self.params.top_p < 1.0 {
            sampling::sample_top_p(&logits, self.params.top_p, rng_val)
        } else {
            sampling::sample_top_k(&logits, self.params.top_k, rng_val)
        };

        self.tokens.push(token_id);
        self.position += 1;

        GenerationStep { token_id, logits }
    }

    /// Generate up to `max_tokens` tokens, calling the callback for each.
    /// Returns the full list of generated token IDs.
    /// The callback receives (token_id, step_number) and returns true to continue.
    pub fn generate<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        eos_token: Option<u32>,
        mut rng: impl FnMut() -> f32,
        mut on_token: F,
    ) -> Vec<u32>
    where
        F: FnMut(u32, usize) -> bool,
    {
        // Prefill
        self.prefill(prompt_tokens);

        let mut generated = Vec::new();

        for step in 0..max_tokens {
            let result = self.step(rng());
            generated.push(result.token_id);

            // Check EOS
            if let Some(eos) = eos_token {
                if result.token_id == eos {
                    break;
                }
            }

            // Callback — return false to stop
            if !on_token(result.token_id, step) {
                break;
            }
        }

        generated
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::model::{LayerWeights, Model, ModelWeights};
    use crate::tensor::Tensor;

    /// Build a tiny model for testing the generation loop.
    fn tiny_model() -> Model {
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            intermediate_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            max_seq_len: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            ..Default::default()
        };

        let dim = config.hidden_dim;
        let vocab = config.vocab_size;
        let inter = config.intermediate_dim;
        let nh = config.num_heads;
        let nkvh = config.num_kv_heads;
        let hd = config.head_dim;

        // Initialize with small random-ish values
        let make_tensor = |size: usize| -> Tensor {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
            Tensor::from_vec(data, &[size]).unwrap()
        };

        let layer = LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: make_tensor(nh * hd * dim),
            wk: make_tensor(nkvh * hd * dim),
            wv: make_tensor(nkvh * hd * dim),
            wo: make_tensor(dim * nh * hd),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: make_tensor(inter * dim),
            w_up: make_tensor(inter * dim),
            w_down: make_tensor(dim * inter),
        };

        let weights = ModelWeights {
            token_embedding: make_tensor(vocab * dim),
            layers: vec![layer],
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: make_tensor(vocab * dim),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_generation_loop() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0, // greedy
            ..Default::default()
        };

        let mut gen = Generator::new(&mut model, params);
        let prompt = vec![1u32, 2];
        let generated = gen.generate(&prompt, 3, None, || 0.5, |_token, _step| true);

        assert_eq!(generated.len(), 3);
        // Total tokens should be prompt + generated
        assert_eq!(gen.tokens().len(), 2 + 3);
    }

    #[test]
    fn test_eos_stops_generation() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };

        let _gen = Generator::new(&mut model, params);
        // Generate with EOS = the token we expect to get (greedy will pick same token)
        let prompt = vec![0u32];

        // First find what greedy gives us
        let logits_tensor = model.forward(0, 0);
        let first_token = sampling::sample_greedy(logits_tensor.data());
        model.reset();

        let mut gen = Generator::new(
            &mut model,
            SamplingParams {
                // used in generate() below
                temperature: 0.0,
                ..Default::default()
            },
        );

        let generated = gen.generate(
            &prompt,
            100, // high max, but EOS should stop us
            Some(first_token),
            || 0.5,
            |_token, _step| true,
        );

        // Should stop after first generated token (which matches EOS)
        assert_eq!(generated.len(), 1);
        assert_eq!(generated[0], first_token);
    }
}

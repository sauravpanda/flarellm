use std::collections::HashMap;

use crate::model::Model;
use crate::sampling::{self, SamplingParams};

/// Maximum number of entries in the n-gram cache to prevent unbounded memory growth.
const NGRAM_CACHE_MAX_ENTRIES: usize = 10_000;

/// Maximum number of draft tokens to propose per speculation attempt.
const MAX_DRAFT_TOKENS: usize = 5;

/// N-gram sizes to record (2, 3, and 4-gram).
const NGRAM_SIZES: &[usize] = &[2, 3, 4];

/// Result of generating a single token.
pub struct GenerationStep {
    pub token_id: u32,
    pub logits: Vec<f32>,
}

/// Statistics for speculative decoding.
#[derive(Debug, Default, Clone)]
pub struct SpeculativeStats {
    /// Total number of speculation attempts.
    pub attempts: usize,
    /// Total number of draft tokens proposed across all attempts.
    pub drafted: usize,
    /// Total number of draft tokens accepted (verified correct).
    pub accepted: usize,
}

/// Cache of n-gram patterns seen during generation.
///
/// Maps an n-gram context (last N tokens) to the list of tokens that followed
/// that context during generation.  Used to propose draft tokens for
/// speculative decoding.
struct NgramCache {
    /// Maps context n-gram → list of next tokens seen after that n-gram.
    table: HashMap<Vec<u32>, Vec<u32>>,
    /// Total number of entries across all keys, used for eviction.
    total_entries: usize,
}

impl NgramCache {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
            total_entries: 0,
        }
    }

    /// Record n-grams of all configured sizes from the token sequence.
    /// Call this after each new token is appended to the context.
    fn record(&mut self, tokens: &[u32]) {
        for &n in NGRAM_SIZES {
            // Need at least n tokens for the context and 1 for the next token.
            if tokens.len() < n + 1 {
                continue;
            }
            let start = tokens.len() - n - 1;
            let context = &tokens[start..start + n];
            let next_token = tokens[tokens.len() - 1];

            let entry = self.table.entry(context.to_vec()).or_default();
            entry.push(next_token);
            self.total_entries += 1;
        }

        // Evict if we exceed the maximum size.
        if self.total_entries > NGRAM_CACHE_MAX_ENTRIES {
            self.evict();
        }
    }

    /// Look up draft tokens for the given context suffix.
    /// Tries the longest n-gram first for better matches.
    fn lookup_drafts(&self, tokens: &[u32]) -> Vec<u32> {
        // Try longest n-gram first (4-gram, then 3-gram, then 2-gram).
        for &n in NGRAM_SIZES.iter().rev() {
            if tokens.len() < n {
                continue;
            }
            let context = &tokens[tokens.len() - n..];
            if let Some(continuations) = self.table.get(context) {
                // Return up to MAX_DRAFT_TOKENS from the most recent continuations.
                let start = continuations.len().saturating_sub(MAX_DRAFT_TOKENS);
                return continuations[start..].to_vec();
            }
        }
        Vec::new()
    }

    /// Evict entries to bring the cache below the maximum size.
    /// Uses a simple strategy: clear the entire table and reset.
    fn evict(&mut self) {
        // Keep only the most recent half by clearing old entries.
        // A simple but effective strategy: clear everything and rebuild
        // from subsequent tokens.  This avoids complex LRU overhead.
        self.table.clear();
        self.total_entries = 0;
    }
}

/// Autoregressive text generation loop.
pub struct Generator<'a> {
    model: &'a mut Model,
    params: SamplingParams,
    tokens: Vec<u32>,
    position: usize,
    ngram_cache: NgramCache,
    speculative_stats: SpeculativeStats,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a mut Model, params: SamplingParams) -> Self {
        Self {
            model,
            params,
            tokens: Vec::new(),
            position: 0,
            ngram_cache: NgramCache::new(),
            speculative_stats: SpeculativeStats::default(),
        }
    }

    /// Returns whether n-gram speculative decoding is active for the current params.
    fn is_speculative(&self) -> bool {
        self.params.speculative && self.params.temperature == 0.0
    }

    /// Returns whether self-speculative (layer-skipping) decoding is active.
    fn is_self_speculative(&self) -> bool {
        self.params.self_speculative && self.params.temperature == 0.0
    }

    /// Prefill: process the prompt as a batch, returning logits for the last token.
    ///
    /// Uses `Model::forward_prefill` which computes all Q/K/V projections
    /// in-order for the full sequence and does causal attention inline,
    /// then writes the entire K/V batch into the KV cache at once.
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Vec<f32> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }
        let output = self.model.forward_prefill(prompt_tokens);
        self.tokens.extend_from_slice(prompt_tokens);
        self.position += prompt_tokens.len();

        // Seed the n-gram cache from the prompt tokens.
        if self.is_speculative() {
            for i in 0..prompt_tokens.len() {
                let end = i + 1;
                if end <= self.tokens.len() {
                    // We record incrementally: for each token position in the
                    // prompt, treat the tokens up to that point as the context.
                    // But we only need the final few tokens for each n-gram size.
                }
            }
            // Record n-grams from the full prompt at once.
            self.record_prompt_ngrams();
        }

        output.data().to_vec()
    }

    /// Record n-grams from the entire current token sequence (used after prefill).
    fn record_prompt_ngrams(&mut self) {
        for &n in NGRAM_SIZES {
            if self.tokens.len() < n + 1 {
                continue;
            }
            for i in 0..=(self.tokens.len() - n - 1) {
                let context = self.tokens[i..i + n].to_vec();
                let next_token = self.tokens[i + n];
                let entry = self.ngram_cache.table.entry(context).or_default();
                entry.push(next_token);
                self.ngram_cache.total_entries += 1;
            }
        }
        if self.ngram_cache.total_entries > NGRAM_CACHE_MAX_ENTRIES {
            self.ngram_cache.evict();
        }
    }

    /// Generate a single next token, returning the token ID.
    pub fn step(&mut self, rng_val: f32) -> GenerationStep {
        let last_token = *self.tokens.last().unwrap_or(&0);

        // Fast path: greedy decoding without repeat penalty can use
        // forward_greedy() which fuses the argmax into the output projection,
        // avoiding a 512KB+ logits buffer write and the subsequent scan.
        if self.params.temperature == 0.0 && self.params.repeat_penalty == 1.0 {
            let (token_id, _logit_val) = self.model.forward_greedy(last_token, self.position);

            self.tokens.push(token_id);
            self.position += 1;

            if self.is_speculative() {
                self.ngram_cache.record(&self.tokens);
            }

            // Return empty logits vec since we skipped full logits computation.
            return GenerationStep {
                token_id,
                logits: Vec::new(),
            };
        }

        // Standard path: compute full logits for sampling or repeat penalty.
        let logits_tensor = self.model.forward(last_token, self.position);
        let mut logits = logits_tensor.data().to_vec();

        // Apply sampling transforms
        sampling::apply_repeat_penalty(&mut logits, &self.tokens, self.params.repeat_penalty);
        sampling::apply_temperature(&mut logits, self.params.temperature);

        // Sample — priority: greedy > top_p > min_p > top_k > full nucleus
        let token_id = if self.params.temperature == 0.0 {
            sampling::sample_greedy(&logits)
        } else if self.params.top_p < 1.0 {
            sampling::sample_top_p(&logits, self.params.top_p, rng_val)
        } else if self.params.min_p > 0.0 {
            sampling::sample_min_p(&logits, self.params.min_p, rng_val)
        } else if self.params.top_k > 0 {
            sampling::sample_top_k(&logits, self.params.top_k, rng_val)
        } else {
            sampling::sample_top_p(&logits, 1.0, rng_val)
        };

        self.tokens.push(token_id);
        self.position += 1;

        // Update n-gram cache after generating a token.
        if self.is_speculative() {
            self.ngram_cache.record(&self.tokens);
        }

        GenerationStep { token_id, logits }
    }

    /// Attempt speculative decoding: look up draft tokens from the n-gram
    /// cache, run forward passes to verify each one, and return all accepted
    /// tokens.  Returns an empty vec if no drafts are available or none match.
    fn speculative_step(&mut self) -> Vec<GenerationStep> {
        let drafts = self.ngram_cache.lookup_drafts(&self.tokens);
        if drafts.is_empty() {
            return Vec::new();
        }

        self.speculative_stats.attempts += 1;
        self.speculative_stats.drafted += drafts.len();

        let mut accepted = Vec::new();

        // Use the fused greedy path when no repeat penalty is active.
        let use_greedy_fused = self.params.repeat_penalty == 1.0;

        for &draft_token in &drafts {
            let last_token = *self.tokens.last().unwrap_or(&0);

            let (verified_token, logits) = if use_greedy_fused {
                // Fast path: fused argmax in the output projection.
                let (tid, _val) = self.model.forward_greedy(last_token, self.position);
                (tid, Vec::new())
            } else {
                // Standard path: full logits needed for repeat penalty.
                let logits_tensor = self.model.forward(last_token, self.position);
                let mut logits = logits_tensor.data().to_vec();
                sampling::apply_repeat_penalty(
                    &mut logits,
                    &self.tokens,
                    self.params.repeat_penalty,
                );
                let tid = sampling::sample_greedy(&logits);
                (tid, logits)
            };

            if verified_token == draft_token {
                // Draft matches — accept it.
                self.tokens.push(verified_token);
                self.position += 1;
                self.ngram_cache.record(&self.tokens);
                self.speculative_stats.accepted += 1;

                accepted.push(GenerationStep {
                    token_id: verified_token,
                    logits,
                });
            } else {
                // Mismatch — accept the verified token (which differs from
                // the draft) and stop speculation.  The verified token is the
                // correct greedy output for this position.
                self.tokens.push(verified_token);
                self.position += 1;
                self.ngram_cache.record(&self.tokens);

                accepted.push(GenerationStep {
                    token_id: verified_token,
                    logits,
                });
                break;
            }
        }

        accepted
    }

    /// Attempt self-speculative decoding: generate draft tokens using a
    /// layer-skipping forward pass, then verify each with the full model.
    ///
    /// Returns all accepted tokens (at least one — the verified token at
    /// the first position).  Returns an empty vec if the model has fewer
    /// than 2 layers (layer skipping would be pointless).
    fn self_speculative_step(&mut self) -> Vec<GenerationStep> {
        let num_layers = self.model.config().num_layers;
        if num_layers < 2 {
            return Vec::new();
        }

        let draft_skip = self.params.draft_skip.max(2);
        let max_drafts = self.params.draft_tokens.max(1);

        // Build the set of layer indices to use for drafting (skip layers).
        let draft_layers: Vec<usize> = (0..num_layers).step_by(draft_skip).collect();

        // Save KV cache state so we can roll back after drafting.
        let saved_kv_len = self.model.kv_cache().len();
        let saved_position = self.position;

        // --- Draft phase: generate tokens with the reduced model ---
        let mut draft_tokens = Vec::with_capacity(max_drafts);
        let mut draft_last_token = *self.tokens.last().unwrap_or(&0);

        for _ in 0..max_drafts {
            let logits_tensor =
                self.model
                    .forward_skip_layers(draft_last_token, self.position, &draft_layers);
            let draft_token = sampling::sample_greedy(logits_tensor.data());
            draft_tokens.push(draft_token);
            draft_last_token = draft_token;
            self.position += 1;
        }

        // --- Roll back KV cache and position to pre-draft state ---
        self.model.truncate_kv_cache(saved_kv_len);
        self.position = saved_position;

        // --- Verify phase: run full forward pass for each draft position ---
        self.speculative_stats.attempts += 1;
        self.speculative_stats.drafted += draft_tokens.len();

        let mut accepted = Vec::new();

        for &draft_token in &draft_tokens {
            let last_token = *self.tokens.last().unwrap_or(&0);

            let (verified_token, logits) = if self.params.repeat_penalty == 1.0 {
                let (tid, _val) = self.model.forward_greedy(last_token, self.position);
                (tid, Vec::new())
            } else {
                let logits_tensor = self.model.forward(last_token, self.position);
                let mut logits = logits_tensor.data().to_vec();
                sampling::apply_repeat_penalty(
                    &mut logits,
                    &self.tokens,
                    self.params.repeat_penalty,
                );
                let tid = sampling::sample_greedy(&logits);
                (tid, logits)
            };

            if verified_token == draft_token {
                // Draft matches — accept it.
                self.tokens.push(verified_token);
                self.position += 1;
                self.speculative_stats.accepted += 1;

                if self.is_speculative() {
                    self.ngram_cache.record(&self.tokens);
                }

                accepted.push(GenerationStep {
                    token_id: verified_token,
                    logits,
                });
            } else {
                // Mismatch — accept the verified token (correct output for
                // this position) and stop.
                self.tokens.push(verified_token);
                self.position += 1;

                if self.is_speculative() {
                    self.ngram_cache.record(&self.tokens);
                }

                accepted.push(GenerationStep {
                    token_id: verified_token,
                    logits,
                });
                break;
            }
        }

        accepted
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
        let mut step = 0;
        let use_speculation = self.is_speculative();
        let use_self_speculation = self.is_self_speculative();

        while step < max_tokens {
            // Self-speculative decoding (layer-skipping draft + full verify)
            if use_self_speculation && step > 0 {
                let spec_results = self.self_speculative_step();
                if !spec_results.is_empty() {
                    let mut should_break = false;
                    for result in spec_results {
                        generated.push(result.token_id);

                        if let Some(eos) = eos_token {
                            if result.token_id == eos {
                                should_break = true;
                                break;
                            }
                        }

                        if !on_token(result.token_id, step) {
                            should_break = true;
                            break;
                        }

                        step += 1;
                        if step >= max_tokens {
                            should_break = true;
                            break;
                        }
                    }
                    if should_break {
                        break;
                    }
                    continue;
                }
            }

            // N-gram speculative decoding
            if use_speculation && step > 0 {
                // Try speculative decoding first.
                let spec_results = self.speculative_step();
                if !spec_results.is_empty() {
                    let mut should_break = false;
                    for result in spec_results {
                        generated.push(result.token_id);

                        // Check EOS
                        if let Some(eos) = eos_token {
                            if result.token_id == eos {
                                should_break = true;
                                break;
                            }
                        }

                        if !on_token(result.token_id, step) {
                            should_break = true;
                            break;
                        }

                        step += 1;
                        if step >= max_tokens {
                            should_break = true;
                            break;
                        }
                    }
                    if should_break {
                        break;
                    }
                    continue;
                }
            }

            // Normal (non-speculative) step.
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

            step += 1;
        }

        generated
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns statistics about speculative decoding performance.
    pub fn speculative_stats(&self) -> &SpeculativeStats {
        &self.speculative_stats
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
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
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

    /// Verify that batched prefill (forward_prefill) produces the same first
    /// generated token as the legacy token-by-token forward loop.
    #[test]
    fn test_batched_prefill_matches_sequential() {
        let prompt = vec![1u32, 2, 3, 4];

        // --- Sequential path: token-by-token ---
        let seq_token = {
            let mut model = tiny_model();
            // Reset and run cleanly using the generator's sequential path
            model.reset();
            let params = crate::sampling::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            // Use old sequential path directly: call forward() for each token
            let mut logits = Vec::new();
            for &tok in &prompt {
                let out = gen.model.forward(tok, gen.position);
                logits = out.data().to_vec();
                gen.tokens.push(tok);
                gen.position += 1;
            }
            crate::sampling::sample_greedy(&logits)
        };

        // --- Batched path: forward_prefill ---
        let batch_token = {
            let mut model = tiny_model();
            let params = crate::sampling::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            // prefill() now calls forward_prefill() internally
            let logits = gen.prefill(&prompt);
            crate::sampling::sample_greedy(&logits)
        };

        assert_eq!(
            seq_token, batch_token,
            "batched prefill must produce the same greedy token as sequential: seq={seq_token}, batch={batch_token}"
        );
    }

    /// Single-token prefill edge case: must behave identically to forward().
    #[test]
    fn test_batched_prefill_single_token() {
        let token = 3u32;

        let seq_logits = {
            let mut model = tiny_model();
            model.forward(token, 0).data().to_vec()
        };

        let batch_logits = {
            let mut model = tiny_model();
            model.forward_prefill(&[token]).data().to_vec()
        };

        for (i, (&s, &b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
            assert!(
                (s - b).abs() < 1e-4,
                "single-token prefill mismatch at logit[{i}]: seq={s}, batch={b}"
            );
        }
    }

    #[test]
    fn test_generate_callback_early_stop() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let prompt = vec![1u32];

        // Callback returns false after the 2nd generated token (step 1)
        let generated = gen.generate(&prompt, 10, None, || 0.5, |_tok, step| step < 1);

        // step 0: callback returns true (0 < 1), step 1: callback returns false
        // So we get 2 tokens generated
        assert_eq!(
            generated.len(),
            2,
            "should stop when callback returns false"
        );
    }

    #[test]
    fn test_prefill_empty_returns_empty() {
        let mut model = tiny_model();
        let params = SamplingParams::default();
        let mut gen = Generator::new(&mut model, params);

        let logits = gen.prefill(&[]);
        assert!(
            logits.is_empty(),
            "empty prefill should return empty logits"
        );
        assert_eq!(gen.position(), 0);
        assert!(gen.tokens().is_empty());
    }

    #[test]
    fn test_position_and_tokens_after_generation() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let prompt = vec![1u32, 2u32];

        let generated = gen.generate(&prompt, 3, None, || 0.5, |_, _| true);

        assert_eq!(gen.position(), prompt.len() + generated.len());
        assert_eq!(gen.tokens().len(), prompt.len() + generated.len());
        // First tokens are the prompt
        assert_eq!(&gen.tokens()[..prompt.len()], &prompt[..]);
    }

    #[test]
    fn test_generate_max_tokens_respected() {
        // Output must never exceed max_tokens regardless of EOS or callback
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let max = 5;
        let generated = gen.generate(&[1u32], max, None, || 0.5, |_, _| true);
        assert!(
            generated.len() <= max,
            "generated {} tokens but max was {}",
            generated.len(),
            max
        );
        assert_eq!(
            generated.len(),
            max,
            "should generate exactly max tokens when no EOS"
        );
    }

    #[test]
    fn test_generate_deterministic_with_greedy() {
        // Greedy (temperature=0) + same prompt must produce identical token sequences
        let prompt = vec![2u32, 3u32];
        let max = 4;

        let run = |prompt: &[u32]| -> Vec<u32> {
            let mut model = tiny_model();
            let params = SamplingParams {
                temperature: 0.0,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(prompt, max, None, || 0.5, |_, _| true)
        };

        let first = run(&prompt);
        let second = run(&prompt);
        assert_eq!(
            first, second,
            "greedy generation must be deterministic: {first:?} vs {second:?}"
        );
    }

    #[test]
    fn test_step_without_prefill_does_not_panic() {
        // Calling step() directly without prefill should not panic (uses token 0 as fallback)
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let result = gen.step(0.5);
        assert!(result.token_id < 8, "token_id must be in vocab range");
    }

    #[test]
    fn test_position_advances_per_step() {
        // Each step() call must increment position by exactly 1
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        gen.tokens.push(1); // seed one token so step() has something to work with
        gen.position = 1;

        for expected_pos in 2..=5 {
            gen.step(0.5);
            assert_eq!(
                gen.position(),
                expected_pos,
                "position should be {expected_pos} after step"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Speculative decoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ngram_cache_record_and_lookup() {
        let mut cache = NgramCache::new();

        // Record tokens [1, 2, 3, 4, 5]
        let tokens: &[u32] = &[1, 2, 3, 4, 5];
        for i in 1..=tokens.len() {
            cache.record(&tokens[..i]);
        }

        // Look up the 2-gram [4, 5] — nothing follows yet.
        // Look up the 2-gram [3, 4] — should find [5].
        let drafts = cache.lookup_drafts(&[3, 4]);
        assert!(
            drafts.contains(&5),
            "n-gram cache should find 5 after [3, 4], got {drafts:?}"
        );
    }

    #[test]
    fn test_ngram_cache_eviction() {
        let mut cache = NgramCache::new();

        // Fill cache past the limit.
        let mut tokens = Vec::new();
        for i in 0..=(NGRAM_CACHE_MAX_ENTRIES as u32 + 100) {
            tokens.push(i);
            cache.record(&tokens);
        }

        // After eviction, total_entries should be manageable.
        assert!(
            cache.total_entries <= NGRAM_CACHE_MAX_ENTRIES,
            "cache should have been evicted: {} entries",
            cache.total_entries
        );
    }

    #[test]
    fn test_speculative_disabled_with_temperature() {
        // Speculative decoding should not activate with temperature > 0.
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.7,
            speculative: true,
            ..Default::default()
        };
        let gen = Generator::new(&mut model, params);
        assert!(
            !gen.is_speculative(),
            "speculation should be disabled with temperature > 0"
        );
    }

    #[test]
    fn test_speculative_disabled_by_flag() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            speculative: false,
            ..Default::default()
        };
        let gen = Generator::new(&mut model, params);
        assert!(
            !gen.is_speculative(),
            "speculation should be disabled when flag is false"
        );
    }

    #[test]
    fn test_speculative_enabled() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            speculative: true,
            ..Default::default()
        };
        let gen = Generator::new(&mut model, params);
        assert!(
            gen.is_speculative(),
            "speculation should be enabled with greedy + flag"
        );
    }

    #[test]
    fn test_speculative_output_matches_greedy() {
        // The most critical test: speculative decoding must produce the
        // exact same output as non-speculative greedy decoding.
        let prompt = vec![1u32, 2, 3];
        let max = 8;

        // Run without speculation.
        let baseline = {
            let mut model = tiny_model();
            let params = SamplingParams {
                temperature: 0.0,
                speculative: false,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        // Run with speculation.
        let speculative = {
            let mut model = tiny_model();
            let params = SamplingParams {
                temperature: 0.0,
                speculative: true,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        assert_eq!(
            baseline, speculative,
            "speculative output must match greedy baseline: baseline={baseline:?} vs speculative={speculative:?}"
        );
    }

    #[test]
    fn test_forward_greedy_matches_forward_argmax() {
        // forward_greedy must produce the same token as forward + argmax.
        // This test uses repeat_penalty=1.0 + temperature=0.0 to exercise
        // the fused greedy code path in Generator::step().
        let prompt = vec![1u32, 2, 3];
        let max = 6;

        // Run with forward_greedy path (repeat_penalty == 1.0).
        let greedy_fused = {
            let mut model = tiny_model();
            let params = SamplingParams {
                temperature: 0.0,
                repeat_penalty: 1.0,
                speculative: false,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        // Run with standard forward path (repeat_penalty == 1.0 but
        // force through the normal code path by using a tiny temperature
        // that behaves like greedy for practical purposes, but != 0.0).
        // Actually, just verify consistency by running forward + argmax
        // directly.
        let standard = {
            let mut model = tiny_model();
            let prompt_tokens = &prompt;
            let mut position = 0;
            // Prefill
            for &t in prompt_tokens {
                model.forward(t, position);
                position += 1;
            }
            let mut tokens = prompt.to_vec();
            let mut result = Vec::new();
            for _ in 0..max {
                let last = *tokens.last().unwrap();
                let logits_tensor = model.forward(last, position);
                let logits = logits_tensor.data();
                let token = crate::sampling::sample_greedy(logits);
                tokens.push(token);
                result.push(token);
                position += 1;
            }
            result
        };

        assert_eq!(
            greedy_fused, standard,
            "forward_greedy must match forward+argmax: fused={greedy_fused:?} vs standard={standard:?}"
        );
    }

    #[test]
    fn test_speculative_stats_populated() {
        let mut model = tiny_model();
        let params = SamplingParams {
            temperature: 0.0,
            speculative: true,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let prompt = vec![1u32, 2];
        let _generated = gen.generate(&prompt, 10, None, || 0.5, |_, _| true);

        // Stats should be accessible (may or may not have attempts depending
        // on whether the tiny model produces repetitive output).
        let stats = gen.speculative_stats();
        assert!(
            stats.accepted <= stats.drafted,
            "accepted ({}) must not exceed drafted ({})",
            stats.accepted,
            stats.drafted
        );
    }

    #[test]
    fn test_ngram_cache_prefers_longer_match() {
        let mut cache = NgramCache::new();

        // Record a 2-gram match: [1, 2] → 99
        cache.table.insert(vec![1, 2], vec![99]);
        cache.total_entries += 1;

        // Record a 4-gram match: [0, 0, 1, 2] → 42
        cache.table.insert(vec![0, 0, 1, 2], vec![42]);
        cache.total_entries += 1;

        // Lookup with context [0, 0, 1, 2] should prefer the 4-gram.
        let drafts = cache.lookup_drafts(&[0, 0, 1, 2]);
        assert_eq!(
            drafts,
            vec![42],
            "should prefer longer n-gram match, got {drafts:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Self-speculative decoding tests
    // -----------------------------------------------------------------------

    /// Build a tiny model with multiple layers for self-speculative tests.
    fn tiny_model_multi_layer() -> Model {
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            intermediate_dim: 8,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            max_seq_len: 32,
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

        let make_tensor = |size: usize| -> Tensor {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
            Tensor::from_vec(data, &[size]).unwrap()
        };

        let make_layer = || LayerWeights {
            attn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            wq: make_tensor(nh * hd * dim),
            wk: make_tensor(nkvh * hd * dim),
            wv: make_tensor(nkvh * hd * dim),
            wo: make_tensor(dim * nh * hd),
            ffn_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            w_gate: make_tensor(inter * dim),
            w_up: make_tensor(inter * dim),
            w_down: make_tensor(dim * inter),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
        };

        let weights = ModelWeights {
            token_embedding: make_tensor(vocab * dim),
            layers: (0..config.num_layers).map(|_| make_layer()).collect(),
            output_norm: Tensor::from_vec(vec![1.0; dim], &[dim]).unwrap(),
            output_weight: make_tensor(vocab * dim),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_self_speculative_output_matches_greedy() {
        // Self-speculative decoding must produce the exact same output
        // as non-speculative greedy decoding.
        let prompt = vec![1u32, 2, 3];
        let max = 8;

        // Run without self-speculation.
        let baseline = {
            let mut model = tiny_model_multi_layer();
            let params = SamplingParams {
                temperature: 0.0,
                speculative: false,
                self_speculative: false,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        // Run with self-speculation.
        let self_spec = {
            let mut model = tiny_model_multi_layer();
            let params = SamplingParams {
                temperature: 0.0,
                speculative: false,
                self_speculative: true,
                draft_skip: 2,
                draft_tokens: 4,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        assert_eq!(
            baseline, self_spec,
            "self-speculative output must match greedy baseline: baseline={baseline:?} vs self_spec={self_spec:?}"
        );
    }

    #[test]
    fn test_self_speculative_disabled_with_temperature() {
        let mut model = tiny_model_multi_layer();
        let params = SamplingParams {
            temperature: 0.7,
            self_speculative: true,
            ..Default::default()
        };
        let gen = Generator::new(&mut model, params);
        assert!(
            !gen.is_self_speculative(),
            "self-speculation should be disabled with temperature > 0"
        );
    }

    #[test]
    fn test_self_speculative_disabled_by_flag() {
        let mut model = tiny_model_multi_layer();
        let params = SamplingParams {
            temperature: 0.0,
            self_speculative: false,
            ..Default::default()
        };
        let gen = Generator::new(&mut model, params);
        assert!(
            !gen.is_self_speculative(),
            "self-speculation should be disabled when flag is false"
        );
    }

    #[test]
    fn test_self_speculative_stats_populated() {
        let mut model = tiny_model_multi_layer();
        let params = SamplingParams {
            temperature: 0.0,
            speculative: false,
            self_speculative: true,
            draft_skip: 2,
            draft_tokens: 3,
            ..Default::default()
        };
        let mut gen = Generator::new(&mut model, params);
        let prompt = vec![1u32, 2];
        let _generated = gen.generate(&prompt, 10, None, || 0.5, |_, _| true);

        let stats = gen.speculative_stats();
        // With 10 max_tokens and step>0 trigger, we should have attempts
        assert!(
            stats.attempts > 0,
            "self-speculation should have at least one attempt"
        );
        assert!(
            stats.drafted > 0,
            "self-speculation should have drafted tokens"
        );
        assert!(
            stats.accepted <= stats.drafted,
            "accepted ({}) must not exceed drafted ({})",
            stats.accepted,
            stats.drafted
        );
    }

    #[test]
    fn test_self_speculative_single_layer_model_noop() {
        // With only 1 layer, self-speculative step should return empty
        // and fall through to normal generation.
        let prompt = vec![1u32, 2];
        let max = 4;

        let baseline = {
            let mut model = tiny_model(); // 1 layer
            let params = SamplingParams {
                temperature: 0.0,
                speculative: false,
                self_speculative: false,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        let self_spec = {
            let mut model = tiny_model(); // 1 layer
            let params = SamplingParams {
                temperature: 0.0,
                speculative: false,
                self_speculative: true,
                ..Default::default()
            };
            let mut gen = Generator::new(&mut model, params);
            gen.generate(&prompt, max, None, || 0.5, |_, _| true)
        };

        assert_eq!(
            baseline, self_spec,
            "single-layer model should produce same output with self-spec enabled"
        );
    }
}

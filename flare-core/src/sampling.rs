/// Sampling strategies for token generation.

/// Parameters controlling text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

/// Apply temperature scaling to logits in-place.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 0.0 || temperature == 1.0 {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}

/// Apply repetition penalty to logits for tokens that have appeared before.
pub fn apply_repeat_penalty(logits: &mut [f32], previous_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token in previous_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Greedy sampling: return the index of the maximum logit.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Compute softmax probabilities from logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Top-p (nucleus) sampling: sample from the smallest set of tokens
/// whose cumulative probability exceeds `top_p`.
pub fn sample_top_p(logits: &[f32], top_p: f32, rng_val: f32) -> u32 {
    let probs = softmax(logits);

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff
    let mut cumulative = 0.0;
    let mut cutoff_idx = indices.len();
    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize and sample
    let candidates = &indices[..cutoff_idx];
    let total: f32 = candidates.iter().map(|&i| probs[i]).sum();
    let threshold = rng_val * total;

    let mut acc = 0.0;
    for &idx in candidates {
        acc += probs[idx];
        if acc >= threshold {
            return idx as u32;
        }
    }

    candidates[candidates.len() - 1] as u32
}

/// Top-k sampling: keep only the top k tokens by probability.
pub fn sample_top_k(logits: &[f32], top_k: usize, rng_val: f32) -> u32 {
    let probs = softmax(logits);

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    indices.truncate(top_k);

    let total: f32 = indices.iter().map(|&i| probs[i]).sum();
    let threshold = rng_val * total;

    let mut acc = 0.0;
    for &idx in &indices {
        acc += probs[idx];
        if acc >= threshold {
            return idx as u32;
        }
    }

    indices[indices.len() - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_temperature_zero_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_temperature(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_deterministic_at_zero() {
        let logits = vec![1.0, 10.0, 0.1];
        // With rng_val near 0, should pick the highest prob token
        let token = sample_top_p(&logits, 0.9, 0.01);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_top_k() {
        let logits = vec![1.0, 10.0, 0.1, 0.01];
        let token = sample_top_k(&logits, 2, 0.01);
        // Should be one of the top 2
        assert!(token == 1 || token == 0);
    }
}

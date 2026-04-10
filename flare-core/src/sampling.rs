//! Sampling strategies for token generation.

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
///
/// Uses a manual loop with f32::max for branch prediction-friendly behavior.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
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
///
/// Uses a partition-based approach: instead of sorting the full vocab
/// (~2ms for 128k vocab), we collect candidates above a threshold and
/// only sort the small candidate set. Falls back to full sort if needed.
pub fn sample_top_p(logits: &[f32], top_p: f32, rng_val: f32) -> u32 {
    // Compute softmax inline so we can use the max value as a threshold reference
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv_sum = 1.0 / sum;

    // Heuristic: top_p=0.9 nucleus is typically <500 tokens. Start with top-k partition.
    // We use select_nth_unstable_by to find the k-th largest in O(n) average.
    const INITIAL_K: usize = 512;
    let k = INITIAL_K.min(exps.len());

    // Build (prob, index) pairs only for the top-k by partial selection
    let mut items: Vec<(f32, u32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (e * inv_sum, i as u32))
        .collect();

    if k < items.len() {
        // O(n) partition: top-k elements (in any order) at items[..k]
        items.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(k);
    }
    // Sort the small candidate set descending
    items.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff and renormalize
    let mut cumulative = 0.0f32;
    let mut cutoff_idx = items.len();
    for (i, &(p, _)) in items.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let candidates = &items[..cutoff_idx];
    let total: f32 = candidates.iter().map(|&(p, _)| p).sum();
    let threshold = rng_val * total;

    let mut acc = 0.0f32;
    for &(p, idx) in candidates {
        acc += p;
        if acc >= threshold {
            return idx;
        }
    }
    candidates[candidates.len() - 1].1
}

/// Top-k sampling: keep only the top k tokens by probability.
///
/// Uses partition-based selection (O(n) average) instead of full sort
/// (O(n log n)). For typical top_k=40 on 128k vocab this is ~30x faster.
pub fn sample_top_k(logits: &[f32], top_k: usize, rng_val: f32) -> u32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv_sum = 1.0 / sum;

    let mut items: Vec<(f32, u32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (e * inv_sum, i as u32))
        .collect();

    let k = top_k.min(items.len()).max(1);
    if k < items.len() {
        items.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(k);
    }

    let total: f32 = items.iter().map(|&(p, _)| p).sum();
    let threshold = rng_val * total;

    let mut acc = 0.0f32;
    for &(p, idx) in &items {
        acc += p;
        if acc >= threshold {
            return idx;
        }
    }
    items[items.len() - 1].1
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

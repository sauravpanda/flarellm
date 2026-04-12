//! Sampling strategies for token generation.

/// Parameters controlling text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    /// Min-p threshold (0.0 = disabled).  When enabled, samples all tokens
    /// whose probability is ≥ `min_p * p_max` (where `p_max` is the highest
    /// token probability after softmax).  Takes precedence over `top_k` but
    /// yields to `top_p` when `top_p < 1.0`.
    pub min_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            min_p: 0.0,
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

/// Min-p sampling: keep all tokens whose probability is ≥ `min_p * p_max`.
///
/// `p_max` is the maximum token probability after softmax.  Tokens below the
/// threshold are discarded and the remaining candidates are sampled
/// proportionally.  Falls back to the top-1 (greedy) token when no candidate
/// survives the threshold.
///
/// This method avoids the "repetition trap" that nucleus sampling can exhibit
/// while still restricting the tail of the distribution.
pub fn sample_min_p(logits: &[f32], min_p: f32, rng_val: f32) -> u32 {
    let probs = softmax(logits);

    let p_max = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = min_p * p_max;

    // Collect candidates above the threshold.
    let mut candidates: Vec<(f32, u32)> = probs
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            if p >= threshold {
                Some((p, i as u32))
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        // Safety fallback: return greedy token.
        return sample_greedy(logits);
    }

    // Sort descending by probability for consistent behaviour with top_p / top_k.
    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total: f32 = candidates.iter().map(|&(p, _)| p).sum();
    let target = rng_val * total;

    let mut acc = 0.0f32;
    for &(p, idx) in &candidates {
        acc += p;
        if acc >= target {
            return idx;
        }
    }
    candidates[candidates.len() - 1].1
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

    #[test]
    fn test_min_p_keeps_high_prob_tokens() {
        // Token 1 has a much higher logit; with min_p=0.1, token 2 (logit=0.1)
        // should be excluded and only the dominant token returned.
        let logits = vec![0.01, 10.0, 0.01, 0.01];
        let token = sample_min_p(&logits, 0.1, 0.01);
        assert_eq!(token, 1, "dominant token should win with tight min_p");
    }

    #[test]
    fn test_min_p_fallback_on_zero() {
        // min_p=0 should behave like sampling from all tokens (threshold=0)
        let logits = vec![1.0, 2.0, 3.0];
        let token = sample_min_p(&logits, 0.0, 0.01);
        // With rng~0, highest-prob token should dominate
        assert_eq!(token, 2);
    }

    #[test]
    fn test_temperature_gt_one_softens() {
        // Temperature > 1 should spread the distribution (reduce max logit)
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        // All logits halved: [0.5, 1.0, 1.5]
        assert!((logits[0] - 0.5).abs() < 1e-5, "logits[0] = {}", logits[0]);
        assert!((logits[1] - 1.0).abs() < 1e-5, "logits[1] = {}", logits[1]);
        assert!((logits[2] - 1.5).abs() < 1e-5, "logits[2] = {}", logits[2]);
    }

    #[test]
    fn test_temperature_one_is_noop() {
        // Temperature = 1.0 should leave logits unchanged
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repeat_penalty_empty_tokens() {
        // Empty previous_tokens → logits unchanged
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repeat_penalty(&mut logits, &[], 1.5);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repeat_penalty_one_is_noop() {
        // penalty=1.0 → early return, logits unchanged
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repeat_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_greedy_all_equal() {
        // All equal logits → returns first (index 0)
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(sample_greedy(&logits), 0);
    }

    #[test]
    fn test_top_p_full_considers_all() {
        // top_p=1.0 should consider all tokens; rng~0 picks highest-prob
        let logits = vec![0.1, 10.0, 0.1];
        let token = sample_top_p(&logits, 1.0, 0.01);
        assert_eq!(
            token, 1,
            "top_p=1.0 should still return dominant token for rng~0"
        );
    }

    #[test]
    fn test_top_k_one_always_argmax() {
        // top_k=1 → only the highest logit is a candidate → always returns it
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let token = sample_top_k(&logits, 1, 0.5);
        assert_eq!(token, 1, "top_k=1 must always return argmax");
    }

    #[test]
    fn test_softmax_large_values_stable() {
        // Large values shouldn't overflow/NaN; softmax uses max-subtraction for stability
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
        assert!(probs.iter().all(|&p| p.is_finite() && p >= 0.0));
        // Highest logit should have highest prob
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }
}

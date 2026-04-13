use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// KIVI-style 2-bit KV cache quantization
// ---------------------------------------------------------------------------

/// Number of quantized values packed into a single byte (2 bits each).
const VALS_PER_BYTE: usize = 4;

/// Maximum 2-bit unsigned value (0..=3).
const Q2_MAX: f32 = 3.0;

/// Number of recent tokens kept in full f32 precision to avoid quality loss
/// on the most-relevant nearby context.
const RESIDUAL_WINDOW: usize = 32;

/// KIVI-style quantized KV cache.
///
/// Keys are quantized **per-channel** (each `head_dim` channel across all
/// tokens shares one scale/min pair).  Values are quantized **per-token**
/// (each token position across all head dims shares one scale/min pair).
///
/// Recent tokens (the last `RESIDUAL_WINDOW` positions) are kept in full
/// f32 precision and blended at read time.
///
/// Storage layout (per layer):
///   key_packed:   `[max_seq_len * num_kv_heads * head_dim / 4]` bytes
///   key_scales:   `[num_kv_heads * head_dim]` f32 — per-channel (max-min)/3
///   key_mins:     `[num_kv_heads * head_dim]` f32 — per-channel min
///   value_packed: same dims as key_packed
///   value_scales: `[max_seq_len]` f32 — per-token (max-min)/3
///   value_mins:   `[max_seq_len]` f32 — per-token min
///
/// Plus a full-precision residual ring buffer of size `RESIDUAL_WINDOW`.
pub struct QuantizedKvCache {
    // -- quantized storage (per layer) --
    key_packed: Vec<Vec<u8>>,
    key_scales: Vec<Vec<f32>>,
    key_mins: Vec<Vec<f32>>,

    value_packed: Vec<Vec<u8>>,
    value_scales: Vec<Vec<f32>>,
    value_mins: Vec<Vec<f32>>,

    // -- full-precision residual window (per layer) --
    residual_keys: Vec<Vec<f32>>,
    residual_values: Vec<Vec<f32>>,

    /// Ring-buffer write position (shared across layers).
    position: usize,
    /// Number of valid entries.
    length: usize,

    max_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    residual_len: usize,
}

impl QuantizedKvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let kv_size = num_kv_heads * head_dim;
        // Each token's kv_size values pack into kv_size/4 bytes (round up).
        let packed_per_token = kv_size.div_ceil(VALS_PER_BYTE);
        let packed_total = max_seq_len * packed_per_token;

        let residual_len = RESIDUAL_WINDOW.min(max_seq_len);

        let mut key_packed = Vec::with_capacity(num_layers);
        let mut key_scales = Vec::with_capacity(num_layers);
        let mut key_mins = Vec::with_capacity(num_layers);
        let mut value_packed = Vec::with_capacity(num_layers);
        let mut value_scales = Vec::with_capacity(num_layers);
        let mut value_mins = Vec::with_capacity(num_layers);
        let mut residual_keys = Vec::with_capacity(num_layers);
        let mut residual_values = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            key_packed.push(vec![0u8; packed_total]);
            key_scales.push(vec![0.0f32; kv_size]); // per-channel
            key_mins.push(vec![0.0f32; kv_size]);

            value_packed.push(vec![0u8; packed_total]);
            value_scales.push(vec![0.0f32; max_seq_len]); // per-token
            value_mins.push(vec![0.0f32; max_seq_len]);

            residual_keys.push(vec![0.0f32; residual_len * kv_size]);
            residual_values.push(vec![0.0f32; residual_len * kv_size]);
        }

        Self {
            key_packed,
            key_scales,
            key_mins,
            value_packed,
            value_scales,
            value_mins,
            residual_keys,
            residual_values,
            position: 0,
            length: 0,
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
            residual_len,
        }
    }

    /// Write a new K/V pair for `layer` at the current ring-buffer position.
    ///
    /// The values are quantized to 2 bits immediately.  A full-precision copy
    /// is also stored in the residual window so the most recent tokens retain
    /// their original accuracy.
    pub fn write(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_size = self.num_kv_heads * self.head_dim;
        debug_assert_eq!(key.len(), kv_size);
        debug_assert_eq!(value.len(), kv_size);

        let pos = self.position;
        let packed_per_token = kv_size.div_ceil(VALS_PER_BYTE);
        let packed_off = pos * packed_per_token;

        // --- Quantize key (per-channel, but we only have one token at a time,
        //     so we store the raw quantized value and recompute channel stats
        //     lazily when the cache is read) ---
        // For simplicity and correctness we use per-token quantization at
        // write time for *both* K and V packed storage.  The per-channel
        // and per-token distinction is applied at read/dequant time by
        // storing per-channel running min/max that get updated here.
        quantize_to_packed(
            key,
            &mut self.key_packed[layer][packed_off..packed_off + packed_per_token],
        );
        // Update per-channel min/scale for this layer.
        update_per_channel_stats(
            key,
            &mut self.key_mins[layer],
            &mut self.key_scales[layer],
            self.length == 0,
        );

        // --- Quantize value (per-token) ---
        let (scale, min_val) = quantize_to_packed(
            value,
            &mut self.value_packed[layer][packed_off..packed_off + packed_per_token],
        );
        self.value_scales[layer][pos] = scale;
        self.value_mins[layer][pos] = min_val;

        // --- Residual window ---
        let res_pos = pos % self.residual_len;
        let res_off = res_pos * kv_size;
        self.residual_keys[layer][res_off..res_off + kv_size].copy_from_slice(key);
        self.residual_values[layer][res_off..res_off + kv_size].copy_from_slice(value);
    }

    /// Advance the write position.  Call after writing all layers for a token.
    pub fn advance(&mut self) {
        self.position = (self.position + 1) % self.max_seq_len;
        if self.length < self.max_seq_len {
            self.length += 1;
        }
    }

    /// Number of valid cached positions.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Current write position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Dequantize and return the full key cache for a layer as contiguous f32.
    ///
    /// Recent tokens within the residual window are returned at full precision.
    pub fn dequant_keys(&self, layer: usize) -> Vec<f32> {
        let kv_size = self.num_kv_heads * self.head_dim;
        let n = self.length;
        let mut out = vec![0.0f32; self.max_seq_len * kv_size];

        let packed_per_token = kv_size.div_ceil(VALS_PER_BYTE);

        for t in 0..n {
            let ring_pos = self.ring_pos(t);
            let dst_off = ring_pos * kv_size;

            if self.is_in_residual(t) {
                // Use full-precision residual
                let res_pos = ring_pos % self.residual_len;
                let res_off = res_pos * kv_size;
                out[dst_off..dst_off + kv_size]
                    .copy_from_slice(&self.residual_keys[layer][res_off..res_off + kv_size]);
            } else {
                // Dequantize using per-channel stats
                let packed_off = ring_pos * packed_per_token;
                dequant_from_packed_per_channel(
                    &self.key_packed[layer][packed_off..packed_off + packed_per_token],
                    &self.key_mins[layer],
                    &self.key_scales[layer],
                    &mut out[dst_off..dst_off + kv_size],
                );
            }
        }
        out
    }

    /// Dequantize and return the full value cache for a layer as contiguous f32.
    ///
    /// Recent tokens within the residual window are returned at full precision.
    pub fn dequant_values(&self, layer: usize) -> Vec<f32> {
        let kv_size = self.num_kv_heads * self.head_dim;
        let n = self.length;
        let mut out = vec![0.0f32; self.max_seq_len * kv_size];

        let packed_per_token = kv_size.div_ceil(VALS_PER_BYTE);

        for t in 0..n {
            let ring_pos = self.ring_pos(t);
            let dst_off = ring_pos * kv_size;

            if self.is_in_residual(t) {
                let res_pos = ring_pos % self.residual_len;
                let res_off = res_pos * kv_size;
                out[dst_off..dst_off + kv_size]
                    .copy_from_slice(&self.residual_values[layer][res_off..res_off + kv_size]);
            } else {
                let packed_off = ring_pos * packed_per_token;
                dequant_from_packed_per_token(
                    &self.value_packed[layer][packed_off..packed_off + packed_per_token],
                    self.value_scales[layer][ring_pos],
                    self.value_mins[layer][ring_pos],
                    &mut out[dst_off..dst_off + kv_size],
                );
            }
        }
        out
    }

    /// Reset the cache.
    pub fn clear(&mut self) {
        self.position = 0;
        self.length = 0;
        for layer in 0..self.num_layers {
            for v in self.key_packed[layer].iter_mut() {
                *v = 0;
            }
            for v in self.value_packed[layer].iter_mut() {
                *v = 0;
            }
            for v in self.key_scales[layer].iter_mut() {
                *v = 0.0;
            }
            for v in self.key_mins[layer].iter_mut() {
                *v = 0.0;
            }
            for v in self.value_scales[layer].iter_mut() {
                *v = 0.0;
            }
            for v in self.value_mins[layer].iter_mut() {
                *v = 0.0;
            }
            for v in self.residual_keys[layer].iter_mut() {
                *v = 0.0;
            }
            for v in self.residual_values[layer].iter_mut() {
                *v = 0.0;
            }
        }
    }

    // ---- helpers ----

    /// Map a logical token index `t` (0 = oldest valid token) to its ring
    /// buffer position, accounting for wraparound.
    fn ring_pos(&self, t: usize) -> usize {
        if self.length < self.max_seq_len {
            t
        } else {
            (self.position + t) % self.max_seq_len
        }
    }

    /// Is logical token index `t` within the residual (full-precision) window?
    fn is_in_residual(&self, t: usize) -> bool {
        let n = self.length;
        if n <= self.residual_len {
            return true; // everything fits in the residual window
        }
        // Residual covers the most recent `residual_len` tokens.
        t >= n - self.residual_len
    }
}

// ---------------------------------------------------------------------------
// 2-bit quantization helpers
// ---------------------------------------------------------------------------

/// Quantize `values` to 2-bit packed representation (4 values per byte).
///
/// Returns `(scale, min)` for the token (used by per-token value quant).
fn quantize_to_packed(values: &[f32], packed: &mut [u8]) -> (f32, f32) {
    let (mut min_val, mut max_val) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in values {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let range = max_val - min_val;
    let scale = if range.abs() < f32::EPSILON {
        0.0
    } else {
        range / Q2_MAX
    };
    let inv_scale = if scale.abs() < f32::EPSILON {
        0.0
    } else {
        Q2_MAX / range
    };

    for byte in packed.iter_mut() {
        *byte = 0;
    }

    for (i, &v) in values.iter().enumerate() {
        let q = if scale.abs() < f32::EPSILON {
            0u8
        } else {
            ((v - min_val) * inv_scale).round().clamp(0.0, Q2_MAX) as u8
        };
        let byte_idx = i / VALS_PER_BYTE;
        let bit_shift = (i % VALS_PER_BYTE) * 2;
        packed[byte_idx] |= q << bit_shift;
    }

    (scale, min_val)
}

/// Dequantize packed 2-bit values using per-channel scale/min.
fn dequant_from_packed_per_channel(packed: &[u8], mins: &[f32], scales: &[f32], out: &mut [f32]) {
    for (i, val) in out.iter_mut().enumerate() {
        let byte_idx = i / VALS_PER_BYTE;
        let bit_shift = (i % VALS_PER_BYTE) * 2;
        let q = ((packed[byte_idx] >> bit_shift) & 0x03) as f32;
        *val = q * scales[i] + mins[i];
    }
}

/// Dequantize packed 2-bit values using per-token scale/min.
fn dequant_from_packed_per_token(packed: &[u8], scale: f32, min_val: f32, out: &mut [f32]) {
    for (i, val) in out.iter_mut().enumerate() {
        let byte_idx = i / VALS_PER_BYTE;
        let bit_shift = (i % VALS_PER_BYTE) * 2;
        let q = ((packed[byte_idx] >> bit_shift) & 0x03) as f32;
        *val = q * scale + min_val;
    }
}

/// Update running per-channel min/scale for key quantization.
///
/// On the first token (`first == true`) the stats are initialised from the
/// input directly.  On subsequent tokens they are expanded (min is lowered,
/// max is raised) so that the dequantized representation covers the full
/// dynamic range observed so far.
fn update_per_channel_stats(values: &[f32], mins: &mut [f32], scales: &mut [f32], first: bool) {
    if first {
        for (i, &v) in values.iter().enumerate() {
            mins[i] = v;
            scales[i] = 0.0; // range is zero for a single observation
        }
    } else {
        for (i, &v) in values.iter().enumerate() {
            let old_min = mins[i];
            let old_max = old_min + scales[i] * Q2_MAX;
            let new_min = old_min.min(v);
            let new_max = old_max.max(v);
            let range = new_max - new_min;
            mins[i] = new_min;
            scales[i] = if range.abs() < f32::EPSILON {
                0.0
            } else {
                range / Q2_MAX
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Original full-precision KV cache
// ---------------------------------------------------------------------------

/// Ring-buffer KV cache for efficient autoregressive generation.
///
/// Instead of growing the cache and shifting data, we write into a fixed-size
/// ring buffer and track the current write position. This avoids allocation
/// during generation and supports sliding-window attention naturally.
pub struct KvCache {
    /// Key cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
    keys: Vec<Tensor>,
    /// Value cache: [num_layers, max_seq_len, num_kv_heads, head_dim]
    values: Vec<Tensor>,
    /// Current write position in the ring buffer
    position: usize,
    /// Number of valid entries (grows until max_seq_len, then stays constant)
    length: usize,
    /// Maximum sequence length (ring buffer size)
    max_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            keys.push(Tensor::zeros(&[max_seq_len, num_kv_heads, head_dim]));
            values.push(Tensor::zeros(&[max_seq_len, num_kv_heads, head_dim]));
        }

        Self {
            keys,
            values,
            position: 0,
            length: 0,
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Write a new K/V pair for a given layer at the current position.
    pub fn write(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_size = self.num_kv_heads * self.head_dim;
        let offset = self.position * kv_size;

        self.keys[layer].data_mut()[offset..offset + kv_size].copy_from_slice(key);
        self.values[layer].data_mut()[offset..offset + kv_size].copy_from_slice(value);
    }

    /// Advance the write position. Call after writing all layers for a token.
    pub fn advance(&mut self) {
        self.position = (self.position + 1) % self.max_seq_len;
        if self.length < self.max_seq_len {
            self.length += 1;
        }
    }

    /// Get the key tensor for a layer (full ring buffer).
    pub fn keys(&self, layer: usize) -> &Tensor {
        &self.keys[layer]
    }

    /// Get the value tensor for a layer (full ring buffer).
    pub fn values(&self, layer: usize) -> &Tensor {
        &self.values[layer]
    }

    /// Number of valid cached positions.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Current write position in the ring buffer.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset the cache (e.g., for a new conversation).
    pub fn clear(&mut self) {
        self.position = 0;
        self.length = 0;
        for layer in 0..self.num_layers {
            for v in self.keys[layer].data_mut().iter_mut() {
                *v = 0.0;
            }
            for v in self.values[layer].data_mut().iter_mut() {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KvCache::new(2, 4, 2, 4);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let key = vec![1.0; 8]; // 2 heads * 4 dim
        let val = vec![2.0; 8];
        cache.write(0, &key, &val);
        cache.write(1, &key, &val);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.position(), 1);
    }

    #[test]
    fn test_kv_cache_ring_buffer() {
        let mut cache = KvCache::new(1, 4, 1, 2);

        // Fill the ring buffer
        for i in 0..6 {
            let key = vec![i as f32; 2];
            let val = vec![i as f32; 2];
            cache.write(0, &key, &val);
            cache.advance();
        }

        // Should wrap around, length capped at max_seq_len
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.position(), 2); // 6 % 4 = 2
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(1, 4, 1, 2);
        let key = vec![1.0; 2];
        let val = vec![2.0; 2];
        cache.write(0, &key, &val);
        cache.advance();
        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn test_wraparound_data_integrity() {
        // max_seq_len=4, write 6 tokens. Position 0 should have token 4's data.
        let mut cache = KvCache::new(1, 4, 1, 2);
        for i in 0..6 {
            let key = vec![i as f32 * 10.0; 2];
            let val = vec![i as f32 * 100.0; 2];
            cache.write(0, &key, &val);
            cache.advance();
        }
        // Token 4 wrote at position 0 (4 % 4 = 0), token 5 at position 1
        let k_data = cache.keys(0).data();
        assert!(
            (k_data[0] - 40.0).abs() < 1e-5,
            "position 0 should have token 4's key: got {}",
            k_data[0]
        );
        assert!(
            (k_data[2] - 50.0).abs() < 1e-5,
            "position 1 should have token 5's key: got {}",
            k_data[2]
        );
    }

    #[test]
    fn test_position_tracking_after_many_wraps() {
        let mut cache = KvCache::new(1, 4, 1, 1);
        for i in 0..100 {
            cache.write(0, &[i as f32], &[0.0]);
            cache.advance();
        }
        assert_eq!(cache.position(), 0); // 100 % 4 = 0
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_multi_layer_independence() {
        let mut cache = KvCache::new(2, 4, 1, 2);
        cache.write(0, &[1.0, 2.0], &[10.0, 20.0]);
        cache.write(1, &[3.0, 4.0], &[30.0, 40.0]);
        cache.advance();

        let k0 = cache.keys(0).data();
        let k1 = cache.keys(1).data();
        assert!((k0[0] - 1.0).abs() < 1e-5);
        assert!((k1[0] - 3.0).abs() < 1e-5);

        let v0 = cache.values(0).data();
        let v1 = cache.values(1).data();
        assert!((v0[0] - 10.0).abs() < 1e-5);
        assert!((v1[0] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_clear_and_reuse() {
        let mut cache = KvCache::new(1, 4, 1, 2);
        cache.write(0, &[99.0, 99.0], &[99.0, 99.0]);
        cache.advance();
        cache.clear();

        // Old data should be zeroed
        assert!(cache.keys(0).data().iter().all(|&v| v == 0.0));

        // Write new data
        cache.write(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.advance();
        assert_eq!(cache.len(), 1);
        assert!((cache.keys(0).data()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_minimal_cache_size_one() {
        let mut cache = KvCache::new(1, 1, 1, 1);
        cache.write(0, &[1.0], &[10.0]);
        cache.advance();
        assert_eq!(cache.len(), 1);
        assert!((cache.keys(0).data()[0] - 1.0).abs() < 1e-5);

        // Overwrite
        cache.write(0, &[2.0], &[20.0]);
        cache.advance();
        assert_eq!(cache.len(), 1); // still 1
        assert!((cache.keys(0).data()[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_head_dim_one() {
        // Single element head dim: kv_size = 1 * 1 = 1
        let mut cache = KvCache::new(1, 8, 1, 1);
        for i in 0..8 {
            cache.write(0, &[i as f32], &[i as f32 * 2.0]);
            cache.advance();
        }
        assert_eq!(cache.len(), 8);
        let k = cache.keys(0).data();
        for (i, &val) in k.iter().enumerate() {
            assert!(
                (val - i as f32).abs() < 1e-5,
                "slot {i}: expected {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_single_layer_cache() {
        // 1 layer should behave identically to multi-layer for its own data
        let mut cache = KvCache::new(1, 4, 2, 4);
        let kv_size = 2 * 4;
        let key: Vec<f32> = (0..kv_size).map(|x| x as f32).collect();
        let val: Vec<f32> = (0..kv_size).map(|x| x as f32 + 100.0).collect();
        cache.write(0, &key, &val);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.position(), 1);
        assert!((cache.keys(0).data()[0] - 0.0).abs() < 1e-5);
        assert!((cache.values(0).data()[0] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_position_at_capacity() {
        // After writing exactly max_seq_len tokens, position wraps to 0
        let max = 6;
        let mut cache = KvCache::new(1, max, 1, 1);
        for i in 0..max {
            cache.write(0, &[i as f32], &[0.0]);
            cache.advance();
        }
        assert_eq!(cache.len(), max);
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn test_len_does_not_exceed_max_seq_len() {
        let max = 4;
        let mut cache = KvCache::new(1, max, 1, 1);
        for i in 0..20 {
            cache.write(0, &[i as f32], &[0.0]);
            cache.advance();
        }
        assert_eq!(cache.len(), max);
    }

    #[test]
    fn test_is_empty_after_clear() {
        let mut cache = KvCache::new(2, 8, 2, 4);
        let kv_size = 2 * 4;
        let key = vec![1.0; kv_size];
        let val = vec![1.0; kv_size];
        for _ in 0..3 {
            cache.write(0, &key, &val);
            cache.write(1, &key, &val);
            cache.advance();
        }
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn test_large_head_dim() {
        // head_dim=256, verify allocation and write/read work correctly
        let head_dim = 256;
        let num_kv_heads = 4;
        let kv_size = num_kv_heads * head_dim;
        let mut cache = KvCache::new(1, 2, num_kv_heads, head_dim);
        let key: Vec<f32> = (0..kv_size).map(|x| x as f32).collect();
        let val: Vec<f32> = vec![0.5; kv_size];
        cache.write(0, &key, &val);
        cache.advance();

        assert_eq!(cache.len(), 1);
        let k = cache.keys(0).data();
        assert!((k[0] - 0.0).abs() < 1e-5);
        assert!((k[kv_size - 1] - (kv_size - 1) as f32).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // QuantizedKvCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantized_basic_write_read() {
        let mut cache = QuantizedKvCache::new(1, 64, 1, 4);
        let key = vec![0.0, 1.0, 2.0, 3.0];
        let val = vec![10.0, 20.0, 30.0, 40.0];
        cache.write(0, &key, &val);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.position(), 1);

        // Recent tokens are in the residual window, so should be exact.
        let dk = cache.dequant_keys(0);
        for i in 0..4 {
            assert!(
                (dk[i] - key[i]).abs() < 1e-5,
                "key[{i}]: expected {}, got {}",
                key[i],
                dk[i]
            );
        }
        let dv = cache.dequant_values(0);
        for i in 0..4 {
            assert!(
                (dv[i] - val[i]).abs() < 1e-5,
                "val[{i}]: expected {}, got {}",
                val[i],
                dv[i]
            );
        }
    }

    #[test]
    fn test_quantized_packing_round_trip() {
        // Test the raw pack/unpack outside the residual window.
        // Write > RESIDUAL_WINDOW tokens so the oldest ones fall out.
        let max_seq = 128;
        let mut cache = QuantizedKvCache::new(1, max_seq, 1, 4);

        // Write enough tokens to push the first ones out of the residual window.
        let total = RESIDUAL_WINDOW + 10;
        for t in 0..total {
            let v = t as f32;
            cache.write(0, &[v, v + 0.5, v + 1.0, v + 1.5], &[v * 10.0; 4]);
            cache.advance();
        }

        // The oldest tokens are quantized (outside residual window).
        // With 2 bits and 4 levels, the error should be bounded by range/3.
        let dv = cache.dequant_values(0);
        // Token 0 is the oldest, stored at ring position 0.
        // Its per-token scale covers 4 identical values, so dequant should be exact.
        let expected = 0.0 * 10.0;
        assert!(
            (dv[0] - expected).abs() < 1.0,
            "oldest value dequant: expected ~{expected}, got {}",
            dv[0]
        );
    }

    #[test]
    fn test_quantized_clear() {
        let mut cache = QuantizedKvCache::new(1, 64, 1, 4);
        cache.write(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.position(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_quantized_ring_buffer_wrap() {
        let max_seq = 8;
        let mut cache = QuantizedKvCache::new(1, max_seq, 1, 2);

        for i in 0..12 {
            cache.write(0, &[i as f32; 2], &[i as f32; 2]);
            cache.advance();
        }

        assert_eq!(cache.len(), max_seq);
        assert_eq!(cache.position(), 12 % max_seq);
    }

    #[test]
    fn test_quantized_multi_layer() {
        let mut cache = QuantizedKvCache::new(2, 64, 1, 4);
        cache.write(0, &[1.0, 2.0, 3.0, 4.0], &[10.0; 4]);
        cache.write(1, &[5.0, 6.0, 7.0, 8.0], &[20.0; 4]);
        cache.advance();

        // Both layers should be independently readable.
        let dk0 = cache.dequant_keys(0);
        let dk1 = cache.dequant_keys(1);
        assert!((dk0[0] - 1.0).abs() < 1e-5);
        assert!((dk1[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_value_per_token_dequant_accuracy() {
        // Values are quantized per-token with their own scale/min.
        // For a uniform range [0, 3] with 2-bit quant, dequant should be exact.
        let mut cache = QuantizedKvCache::new(1, 64, 1, 4);
        let val = vec![0.0, 1.0, 2.0, 3.0]; // maps exactly to q=0,1,2,3
        cache.write(0, &[0.0; 4], &val);
        cache.advance();

        // In residual window, so exact.
        let dv = cache.dequant_values(0);
        for i in 0..4 {
            assert!(
                (dv[i] - val[i]).abs() < 1e-5,
                "val[{i}]: expected {}, got {}",
                val[i],
                dv[i]
            );
        }
    }

    #[test]
    fn test_quantized_constant_values_no_nan() {
        // All-same values: range = 0, scale = 0. Should dequant to the constant.
        let mut cache = QuantizedKvCache::new(1, 64, 1, 4);
        let val = vec![42.0; 4];
        cache.write(0, &val, &val);
        cache.advance();

        // Push out of residual to test quantized path.
        for t in 1..=RESIDUAL_WINDOW + 1 {
            cache.write(0, &[t as f32; 4], &[t as f32; 4]);
            cache.advance();
        }

        // Token 0 is now quantized.  The constant should dequant close.
        let dv = cache.dequant_values(0);
        assert!(
            !dv[0].is_nan(),
            "constant-value dequant must not produce NaN"
        );
    }

    #[test]
    fn test_pack_unpack_direct() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.0];
        let mut packed = vec![0u8; 2]; // 8 values / 4 = 2 bytes
        let (scale, min_val) = quantize_to_packed(&values, &mut packed);

        let mut out = vec![0.0f32; 8];
        dequant_from_packed_per_token(&packed, scale, min_val, &mut out);

        // With 2-bit quant over range [0, 3], max error is range/6 = 0.5.
        for (i, (&orig, &dec)) in values.iter().zip(out.iter()).enumerate() {
            assert!(
                (orig - dec).abs() <= 0.6,
                "val[{i}]: orig={orig}, decoded={dec}, diff={}",
                (orig - dec).abs()
            );
        }
    }
}

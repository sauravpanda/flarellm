use crate::tensor::Tensor;

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
    pub fn write(
        &mut self,
        layer: usize,
        key: &[f32],
        value: &[f32],
    ) {
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
}

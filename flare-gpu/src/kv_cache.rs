use wgpu::util::DeviceExt;

/// GPU-resident ring-buffer KV cache.
///
/// Each transformer layer gets a dedicated pair of `wgpu::Buffer`s for keys
/// and values, pre-allocated at `max_seq_len * num_kv_heads * head_dim * 4`
/// bytes each.  New K/V pairs are written via `queue.write_buffer` at the
/// appropriate ring-buffer offset — no CPU readback is ever needed.
///
/// Buffer layout per layer (same for keys and values):
///   `[max_seq_len, num_kv_heads, head_dim]` row-major f32
///
/// This means the K/V data for sequence position `p` and KV head `h` starts at
/// byte offset `(p * num_kv_heads * head_dim + h * head_dim) * 4`.
pub struct GpuKvCache {
    key_bufs: Vec<wgpu::Buffer>,
    val_bufs: Vec<wgpu::Buffer>,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GpuKvCache {
    /// Allocate GPU buffers for `num_layers` layers, zeroed.
    pub fn new(
        device: &wgpu::Device,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let buf_bytes = max_seq_len * num_kv_heads * head_dim * std::mem::size_of::<f32>();
        let zeros = vec![0u8; buf_bytes];

        let mut key_bufs = Vec::with_capacity(num_layers);
        let mut val_bufs = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            key_bufs.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_kv:key"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            );
            val_bufs.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_kv:val"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            );
        }

        Self {
            key_bufs,
            val_bufs,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Write a K/V pair at ring-buffer position `pos` for the given `layer`.
    ///
    /// `key` and `value` must each have `num_kv_heads * head_dim` elements.
    /// The write is host→device via `queue.write_buffer` — no GPU readback.
    pub fn write(&self, queue: &wgpu::Queue, layer: usize, pos: usize, key: &[f32], value: &[f32]) {
        let kv_elems = self.num_kv_heads * self.head_dim;
        debug_assert_eq!(key.len(), kv_elems);
        debug_assert_eq!(value.len(), kv_elems);

        let ring_pos = pos % self.max_seq_len;
        let byte_offset = (ring_pos * kv_elems * std::mem::size_of::<f32>()) as u64;

        queue.write_buffer(
            &self.key_bufs[layer],
            byte_offset,
            bytemuck::cast_slice(key),
        );
        queue.write_buffer(
            &self.val_bufs[layer],
            byte_offset,
            bytemuck::cast_slice(value),
        );
    }

    /// GPU buffer for layer `layer` keys: `[max_seq_len, num_kv_heads, head_dim]` f32.
    pub fn key_buf(&self, layer: usize) -> &wgpu::Buffer {
        &self.key_bufs[layer]
    }

    /// GPU buffer for layer `layer` values: `[max_seq_len, num_kv_heads, head_dim]` f32.
    pub fn val_buf(&self, layer: usize) -> &wgpu::Buffer {
        &self.val_bufs[layer]
    }
}

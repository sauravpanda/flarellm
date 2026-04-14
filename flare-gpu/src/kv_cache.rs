use half::f16;
use wgpu::util::DeviceExt;

/// GPU-resident ring-buffer KV cache.
///
/// Each transformer layer gets a dedicated pair of `wgpu::Buffer`s for keys
/// and values, pre-allocated at `max_seq_len * num_kv_heads * head_dim * elem_size`
/// bytes each.  New K/V pairs are written via `queue.write_buffer` at the
/// appropriate ring-buffer offset — no CPU readback is ever needed.
///
/// Buffer layout per layer (same for keys and values):
///   `[max_seq_len, num_kv_heads, head_dim]` row-major, either f32 or f16
///
/// This means the K/V data for sequence position `p` and KV head `h` starts at
/// byte offset `(p * num_kv_heads * head_dim + h * head_dim) * elem_size`.
pub struct GpuKvCache {
    key_bufs: Vec<wgpu::Buffer>,
    val_bufs: Vec<wgpu::Buffer>,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Whether buffers store f16 (true) or f32 (false).
    use_f16: bool,
}

impl GpuKvCache {
    /// Allocate GPU buffers for `num_layers` layers, zeroed, using f32 storage.
    pub fn new(
        device: &wgpu::Device,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new_impl(device, num_layers, max_seq_len, num_kv_heads, head_dim, false)
    }

    /// Allocate GPU buffers for `num_layers` layers, zeroed, using f16 storage.
    ///
    /// This halves the KV cache memory on GPU, which is the dominant memory
    /// consumer during long-context generation.  Only use when the device
    /// supports `Features::SHADER_F16` so that the attention shader can read
    /// `array<f16>` directly.
    pub fn new_f16(
        device: &wgpu::Device,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new_impl(device, num_layers, max_seq_len, num_kv_heads, head_dim, true)
    }

    fn new_impl(
        device: &wgpu::Device,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        use_f16: bool,
    ) -> Self {
        let elem_size = if use_f16 { 2 } else { 4 };
        let buf_bytes = max_seq_len * num_kv_heads * head_dim * elem_size;
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
            use_f16,
        }
    }

    /// Whether this cache stores values in f16 format.
    pub fn is_f16(&self) -> bool {
        self.use_f16
    }

    /// Write a K/V pair at ring-buffer position `pos` for the given `layer`.
    ///
    /// `key` and `value` must each have `num_kv_heads * head_dim` elements (f32).
    /// If the cache is in f16 mode, values are converted to f16 before upload.
    /// The write is host→device via `queue.write_buffer` — no GPU readback.
    pub fn write(&self, queue: &wgpu::Queue, layer: usize, pos: usize, key: &[f32], value: &[f32]) {
        let kv_elems = self.num_kv_heads * self.head_dim;
        debug_assert_eq!(key.len(), kv_elems);
        debug_assert_eq!(value.len(), kv_elems);

        let ring_pos = pos % self.max_seq_len;

        if self.use_f16 {
            let byte_offset = (ring_pos * kv_elems * 2) as u64;

            // Convert f32 -> f16 on CPU before upload.
            // Reuse a stack-friendly approach: convert in-place to a small vec.
            let key_f16: Vec<f16> = key.iter().map(|&v| f16::from_f32(v)).collect();
            let val_f16: Vec<f16> = value.iter().map(|&v| f16::from_f32(v)).collect();

            queue.write_buffer(
                &self.key_bufs[layer],
                byte_offset,
                bytemuck::cast_slice(&key_f16),
            );
            queue.write_buffer(
                &self.val_bufs[layer],
                byte_offset,
                bytemuck::cast_slice(&val_f16),
            );
        } else {
            let byte_offset = (ring_pos * kv_elems * 4) as u64;

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
    }

    /// Byte size of one KV element (2 for f16, 4 for f32).
    pub fn elem_size(&self) -> usize {
        if self.use_f16 { 2 } else { 4 }
    }

    /// GPU buffer for layer `layer` keys.
    pub fn key_buf(&self, layer: usize) -> &wgpu::Buffer {
        &self.key_bufs[layer]
    }

    /// GPU buffer for layer `layer` values.
    pub fn val_buf(&self, layer: usize) -> &wgpu::Buffer {
        &self.val_bufs[layer]
    }
}

// On non-WASM targets the pool uses `Mutex<HashMap<...>>` to store idle buffers.
// On WASM the pool is a no-op: `wgpu::Buffer` contains an internal `RefCell` in
// the WebGPU backend which makes it `!Send`, so we cannot store it across "yields"
// in a `Mutex`.  WASM is single-threaded, so the allocation overhead is also much
// lower than on native, and the no-op path keeps the code safe.
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

use wgpu::util::DeviceExt;

/// Maximum number of idle buffers kept per (usage, size) slot.
/// Limits peak GPU memory from accumulating freed buffers.
#[cfg(not(target_arch = "wasm32"))]
const MAX_POOL_DEPTH: usize = 4;

// ---------------------------------------------------------------------------
// Buffer pool — avoids allocating new GPU buffers on every matvec call
// ---------------------------------------------------------------------------

/// A pool of reusable wgpu buffers keyed by their byte size.
///
/// During LLM decode, the same buffer sizes are requested on every token step.
/// Without pooling, each `matvec` call allocates 5 new GPU buffers and
/// destroys them after readback — each allocation takes ~1–2 ms of wgpu
/// overhead.  With pooling the first decode step pays that cost; every
/// subsequent step reuses the pre-allocated buffers from cache.
///
/// **Native targets**: buffers are cached per-size using `Mutex<HashMap>`.
///
/// **WASM target**: the pool is a no-op — `wgpu::Buffer` is `!Send` on the
/// WebGPU backend (it contains an internal `RefCell`), so it cannot be held
/// in a `Mutex` across an async yield point.  WASM is single-threaded so the
/// allocation overhead is lower, making no-op pooling an acceptable trade-off.
///
/// Safety: `Mutex` gives interior mutability so `BufferPool` can be owned
/// by `WebGpuBackend` and used through `&self` (matching the `ComputeBackend`
/// trait signature, which requires `Send + Sync`).  All GPU work is
/// synchronously polled before buffers are returned to the pool, so there
/// are no logical races — each buffer is fully idle before re-entry.
pub struct BufferPool {
    #[cfg(not(target_arch = "wasm32"))]
    storage: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    output: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    staging: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    #[cfg(not(target_arch = "wasm32"))]
    uniform: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            storage: Mutex::new(HashMap::new()),
            #[cfg(not(target_arch = "wasm32"))]
            output: Mutex::new(HashMap::new()),
            #[cfg(not(target_arch = "wasm32"))]
            staging: Mutex::new(HashMap::new()),
            #[cfg(not(target_arch = "wasm32"))]
            uniform: Mutex::new(HashMap::new()),
        }
    }

    // --- Storage buffers (STORAGE | COPY_DST) ---

    /// Get a storage buffer of `data.len()` bytes, uploading `data` to it.
    /// Returns a pooled buffer if one of the right size is available, otherwise
    /// allocates a new one.
    pub fn get_storage(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> wgpu::Buffer {
        let size = data.len() as u64;
        let buf = self.pop_storage(device, size);
        queue.write_buffer(&buf, 0, data);
        buf
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn pop_storage(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        self.storage
            .lock()
            .expect("buffer pool mutex poisoned")
            .entry(size)
            .or_default()
            .pop()
            .unwrap_or_else(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pool:storage"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
    }

    #[cfg(target_arch = "wasm32")]
    fn pop_storage(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool:storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a storage buffer to the pool for reuse.
    pub fn return_storage(&self, buf: wgpu::Buffer) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let size = buf.size();
            let mut pool = self.storage.lock().expect("buffer pool mutex poisoned");
            let slot = pool.entry(size).or_default();
            if slot.len() < MAX_POOL_DEPTH {
                slot.push(buf);
                return;
            }
        }
        drop(buf);
    }

    // --- Output buffers (STORAGE | COPY_SRC) ---

    /// Get an output-only storage buffer of `size` bytes.
    /// No data upload — the GPU shader will write to it.
    pub fn get_output(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        #[cfg(not(target_arch = "wasm32"))]
        {
            return self
                .output
                .lock()
                .expect("buffer pool mutex poisoned")
                .entry(size)
                .or_default()
                .pop()
                .unwrap_or_else(|| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pool:output"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                });
        }
        #[cfg(target_arch = "wasm32")]
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool:output"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Return an output buffer to the pool.
    pub fn return_output(&self, buf: wgpu::Buffer) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let size = buf.size();
            let mut pool = self.output.lock().expect("buffer pool mutex poisoned");
            let slot = pool.entry(size).or_default();
            if slot.len() < MAX_POOL_DEPTH {
                slot.push(buf);
                return;
            }
        }
        drop(buf);
    }

    // --- Staging buffers (MAP_READ | COPY_DST) ---

    /// Get a CPU-readable staging buffer of `size` bytes.
    pub fn get_staging(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        #[cfg(not(target_arch = "wasm32"))]
        {
            return self
                .staging
                .lock()
                .expect("buffer pool mutex poisoned")
                .entry(size)
                .or_default()
                .pop()
                .unwrap_or_else(|| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pool:staging"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });
        }
        #[cfg(target_arch = "wasm32")]
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool:staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a staging buffer to the pool.
    pub fn return_staging(&self, buf: wgpu::Buffer) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let size = buf.size();
            let mut pool = self.staging.lock().expect("buffer pool mutex poisoned");
            let slot = pool.entry(size).or_default();
            if slot.len() < MAX_POOL_DEPTH {
                slot.push(buf);
                return;
            }
        }
        drop(buf);
    }

    // --- Uniform buffers (UNIFORM | COPY_DST) ---

    /// Get a uniform buffer, uploading `data` to it.
    pub fn get_uniform(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> wgpu::Buffer {
        let size = data.len() as u64;
        let buf = self.pop_uniform(device, size);
        queue.write_buffer(&buf, 0, data);
        buf
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn pop_uniform(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        self.uniform
            .lock()
            .expect("buffer pool mutex poisoned")
            .entry(size)
            .or_default()
            .pop()
            .unwrap_or_else(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pool:uniform"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
    }

    #[cfg(target_arch = "wasm32")]
    fn pop_uniform(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool:uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a uniform buffer to the pool.
    pub fn return_uniform(&self, buf: wgpu::Buffer) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let size = buf.size();
            let mut pool = self.uniform.lock().expect("buffer pool mutex poisoned");
            let slot = pool.entry(size).or_default();
            if slot.len() < MAX_POOL_DEPTH {
                slot.push(buf);
                return;
            }
        }
        drop(buf);
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Legacy one-shot helpers (used in tests and benchmarks)
// ---------------------------------------------------------------------------

/// Create a GPU storage buffer from CPU data.
pub fn create_storage_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create an empty GPU storage buffer for output.
pub fn create_output_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a uniform buffer for shader parameters.
pub fn create_uniform_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a staging buffer for reading back GPU results to CPU.
pub fn create_staging_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

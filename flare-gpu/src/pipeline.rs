//! Cached compute pipelines to amortize shader compilation cost.
//!
//! Compiling a WGSL shader and creating a compute pipeline costs ~100µs to
//! several ms on first call. This in-memory cache holds the compiled artifacts
//! so repeated calls (e.g., per-layer matvec in inference) only pay the cost once.
//!
//! Additionally, if `Features::PIPELINE_CACHE` is supported by the GPU backend
//! (currently Vulkan only), a `wgpu::PipelineCache` can be supplied.  The driver
//! then serialises compiled GPU machine code to an opaque blob; restoring the blob
//! on the next run lets the driver skip recompilation entirely.
//!
//! On backends that do not support `PIPELINE_CACHE` (WebGPU, Metal, DX12, GL) the
//! `wgpu_cache` field is `None` and `get_data()` returns an empty `Vec`.

use std::collections::HashMap;
use std::sync::RwLock;

/// A fully built pipeline along with its bind group layout.
pub struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// Lazy in-memory cache for compute pipelines, optionally backed by a
/// `wgpu::PipelineCache` for cross-session persistence.
///
/// Uses `RwLock` for interior mutability so it can be held behind `&self`.
pub struct PipelineCache {
    cache: RwLock<HashMap<&'static str, CachedPipeline>>,
    /// Driver-managed binary cache; `None` on backends that don't support it.
    wgpu_cache: Option<wgpu::PipelineCache>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    /// Create a cache with no driver-level pipeline cache backing.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            wgpu_cache: None,
        }
    }

    /// Create a cache backed by a driver-level `wgpu::PipelineCache`.
    ///
    /// The driver cache accelerates repeated starts on backends that support
    /// `Features::PIPELINE_CACHE` (Vulkan).  On other backends, pass the
    /// result of [`try_create_wgpu_cache`] which returns `None` safely.
    pub fn new_with_wgpu_cache(wgpu_cache: wgpu::PipelineCache) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            wgpu_cache: Some(wgpu_cache),
        }
    }

    /// Serialise the driver-managed binary cache to bytes.
    ///
    /// Returns an empty `Vec` if no driver cache is active or if the backend
    /// does not produce any data (e.g. the driver manages its own cache).
    /// The bytes can be stored and later fed back via
    /// [`WebGpuBackend::new_with_cache`] to skip recompilation on next start.
    pub fn get_data(&self) -> Vec<u8> {
        self.wgpu_cache
            .as_ref()
            .and_then(|c| c.get_data())
            .unwrap_or_default()
    }

    /// Run `f` with the cached pipeline for `name`, building it on first call.
    ///
    /// `bind_layout_entries` defines the bind group layout (typically 4 entries:
    /// 2-3 storage buffers and a uniform). `entry_point` is the WGSL function name.
    pub fn with_pipeline<F, R>(
        &self,
        device: &wgpu::Device,
        name: &'static str,
        shader_source: &'static str,
        entry_point: &'static str,
        bind_layout_entries: &[wgpu::BindGroupLayoutEntry],
        f: F,
    ) -> R
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        // Fast path: read lock if already cached
        {
            let read = self.cache.read().expect("pipeline cache poisoned");
            if let Some(cached) = read.get(name) {
                return f(cached);
            }
        }

        // Slow path: build pipeline under write lock
        let mut write = self.cache.write().expect("pipeline cache poisoned");
        // Re-check in case another thread just inserted it
        if !write.contains_key(name) {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: bind_layout_entries,
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                // Pass the driver cache when available so compiled GPU machine
                // code can be reused across sessions on supported backends.
                cache: self.wgpu_cache.as_ref(),
            });

            write.insert(name, CachedPipeline { pipeline, layout });
        }

        f(write.get(name).expect("just inserted"))
    }
}

/// Attempt to create a `wgpu::PipelineCache` on `device` using `data` as the
/// initial blob (pass `None` for a fresh cache).
///
/// Returns `None` if the `PIPELINE_CACHE` feature is not available on the
/// current backend.
///
/// # Safety
///
/// The caller must ensure `data`, when `Some`, was obtained from a previous call
/// to [`PipelineCache::get_data`] on a *compatible* device (same GPU, same driver
/// version). Passing mismatched data is safe only because `fallback: true` makes
/// the driver silently discard invalid blobs rather than invoking UB.
///
/// The `unsafe` block is required by the wgpu API (`Device::create_pipeline_cache`
/// is marked `unsafe` because the caller is responsible for data validity).
pub fn try_create_wgpu_cache(
    device: &wgpu::Device,
    features: wgpu::Features,
    data: Option<&[u8]>,
) -> Option<wgpu::PipelineCache> {
    if !features.contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    // SAFETY: `fallback: true` means the driver discards invalid data rather than
    // causing undefined behaviour.  When `data` is `Some`, it must come from
    // `PipelineCache::get_data()` on a compatible device; the caller is responsible
    // for that contract, but the fallback flag provides a safety net.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("flare-pipeline-cache"),
            data,
            fallback: true,
        })
    };
    Some(cache)
}

//! Cached compute pipelines to amortize shader compilation cost.
//!
//! Compiling a WGSL shader and creating a compute pipeline costs ~100µs to
//! several ms on first call. This cache holds the compiled artifacts so
//! repeated calls (e.g., per-layer matvec in inference) only pay the cost once.

use std::collections::HashMap;
use std::sync::RwLock;

/// A fully built pipeline along with its bind group layout.
pub struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

/// Lazy cache for compute pipelines, keyed by name.
///
/// Uses RwLock for interior mutability so it can be held behind &self.
pub struct PipelineCache {
    cache: RwLock<HashMap<&'static str, CachedPipeline>>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
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
                cache: None,
            });

            write.insert(name, CachedPipeline { pipeline, layout });
        }

        f(write.get(name).expect("just inserted"))
    }
}

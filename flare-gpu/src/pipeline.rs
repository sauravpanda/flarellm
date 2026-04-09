use std::collections::HashMap;

/// Cache for compiled compute pipelines to avoid recompilation.
pub struct PipelineCache {
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Get or create a compute pipeline from WGSL source.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> &wgpu::ComputePipeline {
        if !self.pipelines.contains_key(name) {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &[],
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

            self.pipelines.insert(name.to_string(), pipeline);
        }

        &self.pipelines[name]
    }
}

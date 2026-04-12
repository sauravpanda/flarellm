//! Progressive model loading for streaming inference.
//!
//! Instead of downloading the entire model before inference starts,
//! load weights layer-by-layer and begin inference as soon as the
//! first layers are available. On a 100Mbps connection, a 500MB
//! Q4 model can produce the first token in ~3 seconds.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Tracks which layers are loaded and ready for inference.
pub struct LoadProgress {
    layers_loaded: AtomicUsize,
    total_layers: usize,
    embedding_ready: AtomicBool,
    output_ready: AtomicBool,
}

impl LoadProgress {
    pub fn new(total_layers: usize) -> Self {
        Self {
            layers_loaded: AtomicUsize::new(0),
            total_layers,
            embedding_ready: AtomicBool::new(false),
            output_ready: AtomicBool::new(false),
        }
    }

    /// Mark the embedding layer as loaded.
    pub fn set_embedding_ready(&self) {
        self.embedding_ready.store(true, Ordering::Release);
    }

    /// Mark the output projection as loaded.
    pub fn set_output_ready(&self) {
        self.output_ready.store(true, Ordering::Release);
    }

    /// Mark a transformer layer as loaded.
    pub fn mark_layer_loaded(&self) {
        self.layers_loaded.fetch_add(1, Ordering::Release);
    }

    /// Check if a specific layer index is available.
    pub fn is_layer_ready(&self, layer_idx: usize) -> bool {
        self.layers_loaded.load(Ordering::Acquire) > layer_idx
    }

    /// Check if the model can start prefill (embedding + at least one layer).
    pub fn can_start_inference(&self) -> bool {
        self.embedding_ready.load(Ordering::Acquire)
            && self.layers_loaded.load(Ordering::Acquire) > 0
    }

    /// Check if all weights are loaded.
    pub fn is_complete(&self) -> bool {
        self.embedding_ready.load(Ordering::Acquire)
            && self.output_ready.load(Ordering::Acquire)
            && self.layers_loaded.load(Ordering::Acquire) >= self.total_layers
    }

    /// Number of layers currently loaded.
    pub fn loaded_count(&self) -> usize {
        self.layers_loaded.load(Ordering::Acquire)
    }

    /// Total number of layers.
    pub fn total_count(&self) -> usize {
        self.total_layers
    }

    /// Progress as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f32 {
        // Embedding and output are ~2 layers worth of weight
        let extra = if self.embedding_ready.load(Ordering::Acquire) {
            1
        } else {
            0
        } + if self.output_ready.load(Ordering::Acquire) {
            1
        } else {
            0
        };
        let loaded = self.layers_loaded.load(Ordering::Acquire) + extra;
        let total = self.total_layers + 2; // +2 for embedding and output
        loaded as f32 / total as f32
    }
}

/// Configuration for progressive loading behavior.
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Minimum number of layers to load before starting inference.
    /// Default: 1 (start as soon as embedding + 1 layer ready).
    pub min_layers_before_start: usize,

    /// Whether to cache downloaded weights to disk/Cache API.
    pub cache_weights: bool,

    /// Chunk size for HTTP range requests (bytes).
    /// Default: 2MB chunks for good throughput without huge memory spikes.
    pub chunk_size: usize,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            min_layers_before_start: 1,
            cache_weights: true,
            chunk_size: 2 * 1024 * 1024, // 2MB
        }
    }
}

/// A plan for loading model layers in priority order.
/// Embedding and first layers are loaded first, output layer last.
pub struct LoadPlan {
    /// Ordered list of (component_name, byte_offset, byte_size) to fetch.
    pub steps: Vec<LoadStep>,
}

/// A single step in the loading plan.
#[derive(Debug, Clone)]
pub struct LoadStep {
    /// Component identifier (e.g., "embedding", "layer.0", "output")
    pub component: String,
    /// Byte offset in the model file.
    pub offset: u64,
    /// Byte size of this component.
    pub size: u64,
}

impl LoadPlan {
    /// Create a loading plan from GGUF tensor info.
    /// Orders: embedding first, then layers 0..N, then output norm + output.
    pub fn from_tensor_info(
        tensors: &[(String, u64, u64)], // (name, offset, size)
        num_layers: usize,
    ) -> Self {
        let mut embedding_steps = Vec::new();
        let mut layer_steps: Vec<Vec<(String, u64, u64)>> = vec![Vec::new(); num_layers];
        let mut output_steps = Vec::new();

        for (name, offset, size) in tensors {
            if name.starts_with("token_embd") || name.starts_with("model.embed_tokens") {
                embedding_steps.push((name.clone(), *offset, *size));
            } else if name.starts_with("output") || name.starts_with("lm_head") {
                output_steps.push((name.clone(), *offset, *size));
            } else {
                // Try to extract layer index
                let layer_idx = extract_layer_index(name);
                if let Some(idx) = layer_idx {
                    if idx < num_layers {
                        layer_steps[idx].push((name.clone(), *offset, *size));
                    }
                }
            }
        }

        let mut steps = Vec::new();

        // Embedding first
        for (_name, offset, size) in &embedding_steps {
            steps.push(LoadStep {
                component: "embedding".into(),
                offset: *offset,
                size: *size,
            });
        }

        // Layers in order
        for (idx, layer_tensors) in layer_steps.iter().enumerate() {
            let total_size: u64 = layer_tensors.iter().map(|(_, _, s)| s).sum();
            let min_offset = layer_tensors.iter().map(|(_, o, _)| *o).min().unwrap_or(0);
            steps.push(LoadStep {
                component: format!("layer.{idx}"),
                offset: min_offset,
                size: total_size,
            });
        }

        // Output last
        for (_name, offset, size) in &output_steps {
            steps.push(LoadStep {
                component: "output".into(),
                offset: *offset,
                size: *size,
            });
        }

        Self { steps }
    }
}

/// Extract layer index from tensor name (e.g., "blk.5.attn_q.weight" -> 5).
fn extract_layer_index(name: &str) -> Option<usize> {
    // Try "blk.N." pattern
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    // Try "model.layers.N." pattern
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_progress_basic() {
        let progress = LoadProgress::new(16);
        assert!(!progress.can_start_inference());
        assert!(!progress.is_complete());
        assert_eq!(progress.loaded_count(), 0);

        progress.set_embedding_ready();
        assert!(!progress.can_start_inference()); // no layers yet

        progress.mark_layer_loaded();
        assert!(progress.can_start_inference());
        assert!(!progress.is_complete());
    }

    #[test]
    fn test_load_progress_complete() {
        let progress = LoadProgress::new(2);
        progress.set_embedding_ready();
        progress.mark_layer_loaded();
        progress.mark_layer_loaded();
        progress.set_output_ready();
        assert!(progress.is_complete());
        assert!((progress.progress() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_ready_check() {
        let progress = LoadProgress::new(4);
        progress.mark_layer_loaded();
        progress.mark_layer_loaded();
        assert!(progress.is_layer_ready(0));
        assert!(progress.is_layer_ready(1));
        assert!(!progress.is_layer_ready(2));
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("blk.5.attn_q.weight"), Some(5));
        assert_eq!(extract_layer_index("blk.0.ffn_gate.weight"), Some(0));
        assert_eq!(
            extract_layer_index("model.layers.12.self_attn.q_proj.weight"),
            Some(12)
        );
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output.weight"), None);
    }

    #[test]
    fn test_load_plan_ordering() {
        let tensors = vec![
            ("blk.1.attn_q.weight".into(), 200, 50),
            ("token_embd.weight".into(), 0, 100),
            ("blk.0.attn_q.weight".into(), 100, 50),
            ("output.weight".into(), 300, 100),
        ];
        let plan = LoadPlan::from_tensor_info(&tensors, 2);

        assert_eq!(plan.steps[0].component, "embedding");
        assert_eq!(plan.steps[1].component, "layer.0");
        assert_eq!(plan.steps[2].component, "layer.1");
        assert_eq!(plan.steps[3].component, "output");
    }

    #[test]
    fn test_progressive_config_defaults() {
        let config = ProgressiveConfig::default();
        assert_eq!(config.min_layers_before_start, 1);
        assert!(config.cache_weights);
        assert_eq!(config.chunk_size, 2 * 1024 * 1024);
    }

    #[test]
    fn test_progress_partial_stages() {
        // total_layers=2 → denominator=4 (embedding + 2 layers + output)
        let progress = LoadProgress::new(2);
        assert!((progress.progress() - 0.0).abs() < 1e-5);
        progress.set_embedding_ready();
        assert!((progress.progress() - 0.25).abs() < 1e-5); // 1/4
        progress.mark_layer_loaded();
        assert!((progress.progress() - 0.5).abs() < 1e-5); // 2/4
        progress.mark_layer_loaded();
        assert!((progress.progress() - 0.75).abs() < 1e-5); // 3/4
        progress.set_output_ready();
        assert!((progress.progress() - 1.0).abs() < 1e-5); // 4/4
    }

    #[test]
    fn test_is_layer_ready_out_of_bounds() {
        let progress = LoadProgress::new(4);
        progress.mark_layer_loaded();
        progress.mark_layer_loaded();
        // 2 layers loaded → indices 0 and 1 ready; large index is not
        assert!(!progress.is_layer_ready(1000));
    }

    #[test]
    fn test_extract_layer_index_multi_digit() {
        assert_eq!(extract_layer_index("blk.22.attn_q.weight"), Some(22));
        assert_eq!(extract_layer_index("blk.100.ffn_down.weight"), Some(100));
    }

    #[test]
    fn test_extract_layer_index_unexpected_prefix() {
        assert_eq!(extract_layer_index("rope.freqs"), None);
        assert_eq!(extract_layer_index("norm.weight"), None);
    }

    #[test]
    fn test_load_plan_no_layer_tensors() {
        // Only embedding and output — layer steps should be present but empty (size=0)
        let tensors = vec![
            ("token_embd.weight".into(), 0u64, 100u64),
            ("output.weight".into(), 100, 100),
        ];
        let plan = LoadPlan::from_tensor_info(&tensors, 2);
        assert_eq!(plan.steps[0].component, "embedding");
        assert_eq!(plan.steps[1].component, "layer.0");
        assert_eq!(plan.steps[1].size, 0);
        assert_eq!(plan.steps[2].component, "layer.1");
        assert_eq!(plan.steps[2].size, 0);
        assert_eq!(plan.steps[3].component, "output");
    }

    #[test]
    fn test_load_plan_layer_size_aggregation() {
        // Two tensors in layer 0 — their sizes should be summed
        let tensors = vec![
            ("token_embd.weight".into(), 0u64, 50u64),
            ("blk.0.attn_q.weight".into(), 50, 100),
            ("blk.0.attn_k.weight".into(), 150, 200),
            ("output.weight".into(), 350, 50),
        ];
        let plan = LoadPlan::from_tensor_info(&tensors, 1);
        let layer_step = plan
            .steps
            .iter()
            .find(|s| s.component == "layer.0")
            .unwrap();
        assert_eq!(layer_step.size, 300); // 100 + 200
        assert_eq!(layer_step.offset, 50); // min of 50 and 150
    }

    #[test]
    fn test_load_progress_zero_layers() {
        // zero total_layers is a valid edge case (e.g. config parsing failed)
        let progress = LoadProgress::new(0);
        assert_eq!(progress.total_count(), 0);
        assert_eq!(progress.loaded_count(), 0);
        // With 0 layers, is_complete requires embedding + output only
        progress.set_embedding_ready();
        progress.set_output_ready();
        assert!(
            progress.is_complete(),
            "0-layer model should be complete after embedding+output"
        );
    }

    #[test]
    fn test_can_start_inference_requires_embedding() {
        // Loading layers without embedding must not allow inference start
        let progress = LoadProgress::new(4);
        progress.mark_layer_loaded();
        progress.mark_layer_loaded();
        assert!(
            !progress.can_start_inference(),
            "inference must not start without embedding, even if layers are loaded"
        );
        progress.set_embedding_ready();
        assert!(
            progress.can_start_inference(),
            "inference can start once embedding + 1 layer ready"
        );
    }

    #[test]
    fn test_is_complete_requires_output() {
        // All transformer layers + embedding loaded, but no output → not complete
        let progress = LoadProgress::new(2);
        progress.set_embedding_ready();
        progress.mark_layer_loaded();
        progress.mark_layer_loaded();
        assert!(
            !progress.is_complete(),
            "must not be complete without output projection"
        );
        progress.set_output_ready();
        assert!(progress.is_complete());
    }

    #[test]
    fn test_load_plan_empty_tensor_list() {
        // Empty tensor list → plan has only the (empty) layer placeholders
        let plan = LoadPlan::from_tensor_info(&[], 3);
        // 3 layer slots, no embedding, no output
        let layer_components: Vec<&str> = plan.steps.iter().map(|s| s.component.as_str()).collect();
        assert_eq!(layer_components, ["layer.0", "layer.1", "layer.2"]);
    }

    #[test]
    fn test_load_plan_model_layers_prefix() {
        // Tensors using "model.layers.N." prefix should be assigned to the correct layer step
        let tensors = vec![
            ("token_embd.weight".into(), 0u64, 64u64),
            ("model.layers.0.self_attn.q_proj.weight".into(), 64, 128),
            ("model.layers.1.self_attn.q_proj.weight".into(), 192, 128),
            ("lm_head.weight".into(), 320, 64),
        ];
        let plan = LoadPlan::from_tensor_info(&tensors, 2);
        let l0 = plan
            .steps
            .iter()
            .find(|s| s.component == "layer.0")
            .unwrap();
        let l1 = plan
            .steps
            .iter()
            .find(|s| s.component == "layer.1")
            .unwrap();
        assert_eq!(l0.size, 128);
        assert_eq!(l1.size, 128);
    }
}

//! LoRA (Low-Rank Adaptation) adapter support.
//!
//! LoRA decomposes weight updates as `W_new = W + (alpha / rank) * B @ A` where
//! A is `(rank, in_dim)` and B is `(out_dim, rank)`.  This module provides the
//! data structures and merge logic for applying LoRA adapters to model weights.
//!
//! For browser deployment, merge-at-load-time is preferred: no inference overhead.

use thiserror::Error;

/// Errors related to LoRA adapter loading and merging.
#[derive(Debug, Error)]
pub enum LoraError {
    #[error("LoRA rank mismatch: tensor has {got} elements, expected multiple of rank {rank}")]
    RankMismatch { rank: usize, got: usize },

    #[error("LoRA dimension mismatch for {tensor}: expected {expected}, got {got}")]
    DimensionMismatch {
        tensor: String,
        expected: usize,
        got: usize,
    },

    #[error("LoRA layer index {index} exceeds model layer count {count}")]
    LayerIndexOutOfRange { index: usize, count: usize },

    #[error("LoRA adapter has no layers")]
    EmptyAdapter,

    #[error("LoRA parse error: {0}")]
    ParseError(String),
}

/// A complete LoRA adapter that can be merged into model weights.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Low-rank dimension (typically 8-64).
    pub rank: usize,
    /// Scaling factor. The effective scale applied is `alpha / rank`.
    pub alpha: f32,
    /// Per-layer LoRA weight pairs. Indexed by transformer layer index.
    pub layers: Vec<LoraLayerWeights>,
}

/// LoRA weight pairs for a single transformer layer.
///
/// Each pair consists of A `(rank, in_dim)` and B `(out_dim, rank)` matrices
/// stored in row-major order.  `None` means that projection is not adapted.
#[derive(Debug, Clone, Default)]
pub struct LoraLayerWeights {
    // --- Attention projections ---
    /// Query projection A matrix: `(rank, dim)`, row-major
    pub wq_a: Option<Vec<f32>>,
    /// Query projection B matrix: `(dim, rank)`, row-major
    pub wq_b: Option<Vec<f32>>,

    /// Key projection A matrix: `(rank, kv_dim)`, row-major
    pub wk_a: Option<Vec<f32>>,
    /// Key projection B matrix: `(kv_dim, rank)`, row-major
    pub wk_b: Option<Vec<f32>>,

    /// Value projection A matrix: `(rank, kv_dim)`, row-major
    pub wv_a: Option<Vec<f32>>,
    /// Value projection B matrix: `(kv_dim, rank)`, row-major
    pub wv_b: Option<Vec<f32>>,

    /// Output projection A matrix: `(rank, dim)`, row-major
    pub wo_a: Option<Vec<f32>>,
    /// Output projection B matrix: `(dim, rank)`, row-major
    pub wo_b: Option<Vec<f32>>,

    // --- FFN projections ---
    /// Gate projection A matrix: `(rank, dim)`, row-major
    pub w_gate_a: Option<Vec<f32>>,
    /// Gate projection B matrix: `(intermediate_dim, rank)`, row-major
    pub w_gate_b: Option<Vec<f32>>,

    /// Up projection A matrix: `(rank, dim)`, row-major
    pub w_up_a: Option<Vec<f32>>,
    /// Up projection B matrix: `(intermediate_dim, rank)`, row-major
    pub w_up_b: Option<Vec<f32>>,

    /// Down projection A matrix: `(rank, intermediate_dim)`, row-major
    pub w_down_a: Option<Vec<f32>>,
    /// Down projection B matrix: `(dim, rank)`, row-major
    pub w_down_b: Option<Vec<f32>>,
}

/// Compute the matrix product `C = B @ A` where:
/// - B is `(out_dim, rank)`, row-major
/// - A is `(rank, in_dim)`, row-major
/// - C is `(out_dim, in_dim)`, row-major
///
/// This is a simple O(out_dim * in_dim * rank) matmul used for LoRA merging.
/// Performance is not critical since merging happens once at load time.
#[allow(dead_code)]
pub(crate) fn matmul_ba(b: &[f32], a: &[f32], out_dim: usize, rank: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(b.len(), out_dim * rank);
    debug_assert_eq!(a.len(), rank * in_dim);

    let mut result = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for k in 0..rank {
            let b_val = b[row * rank + k];
            let a_row = &a[k * in_dim..(k + 1) * in_dim];
            let out_row = &mut result[row * in_dim..(row + 1) * in_dim];
            for col in 0..in_dim {
                out_row[col] += b_val * a_row[col];
            }
        }
    }
    result
}

/// Apply a LoRA delta (`scale * B @ A`) to a weight tensor in-place.
///
/// - `weight_data`: mutable slice of the weight tensor, `(out_dim, in_dim)` row-major
/// - `a`: LoRA A matrix, `(rank, in_dim)` row-major
/// - `b`: LoRA B matrix, `(out_dim, rank)` row-major
/// - `scale`: `alpha / rank`
///
/// Returns an error if dimensions are inconsistent.
pub(crate) fn apply_lora_delta(
    weight_data: &mut [f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    scale: f32,
) -> Result<(), LoraError> {
    let total = weight_data.len();
    let in_dim = a.len() / rank;
    let out_dim = b.len() / rank;

    if a.len() != rank * in_dim {
        return Err(LoraError::RankMismatch {
            rank,
            got: a.len(),
        });
    }
    if b.len() != out_dim * rank {
        return Err(LoraError::RankMismatch {
            rank,
            got: b.len(),
        });
    }
    if total != out_dim * in_dim {
        return Err(LoraError::DimensionMismatch {
            tensor: "weight".into(),
            expected: out_dim * in_dim,
            got: total,
        });
    }

    // Compute delta = B @ A and add scaled delta to weight
    // Fused to avoid allocating the full delta matrix
    for row in 0..out_dim {
        for k in 0..rank {
            let b_val = b[row * rank + k] * scale;
            let a_row = &a[k * in_dim..(k + 1) * in_dim];
            let w_row = &mut weight_data[row * in_dim..(row + 1) * in_dim];
            for col in 0..in_dim {
                w_row[col] += b_val * a_row[col];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_ba_identity() {
        // rank=2, out_dim=2, in_dim=2
        // B = [[1,0],[0,1]], A = [[1,0],[0,1]]
        // Result should be identity
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let result = matmul_ba(&b, &a, 2, 2, 2);
        assert_eq!(result, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_matmul_ba_simple() {
        // B = [[1,2],[3,4]] (2x2), A = [[5,6],[7,8]] (2x2)
        // Result = B @ A = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
        //        = [[19, 22],[43, 50]]
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let a = vec![5.0, 6.0, 7.0, 8.0];
        let result = matmul_ba(&b, &a, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_ba_rectangular() {
        // B = [[1,0]] (1x2), A = [[1,2,3],[0,0,0]] (2x3)
        // Result = [[1,2,3]] (1x3)
        let b = vec![1.0, 0.0];
        let a = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let result = matmul_ba(&b, &a, 1, 2, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_lora_delta_simple() {
        // W = zeros(2,2), rank=1, A = [1,0] (1x2), B = [1,1] (2x1), scale=1.0
        // delta = B @ A = [[1,0],[1,0]]
        // W_new = [[1,0],[1,0]]
        let mut w = vec![0.0; 4];
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        apply_lora_delta(&mut w, &a, &b, 1, 1.0).unwrap();
        assert_eq!(w, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_apply_lora_delta_with_scale() {
        let mut w = vec![10.0; 4];
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        // delta = [[1,1],[1,1]], scale = 0.5
        // W_new = [[10.5, 10.5],[10.5, 10.5]]
        apply_lora_delta(&mut w, &a, &b, 1, 0.5).unwrap();
        for &v in &w {
            assert!((v - 10.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_lora_delta_dimension_mismatch() {
        let mut w = vec![0.0; 6]; // 2x3
        let a = vec![1.0, 0.0]; // rank=1, in_dim=2
        let b = vec![1.0, 1.0]; // out_dim=2, rank=1
        // w is 2x3=6, but B@A produces 2x2=4 -> mismatch
        let result = apply_lora_delta(&mut w, &a, &b, 1, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_default_layer() {
        let layer = LoraLayerWeights::default();
        assert!(layer.wq_a.is_none());
        assert!(layer.wq_b.is_none());
        assert!(layer.wk_a.is_none());
        assert!(layer.w_gate_a.is_none());
    }

    #[test]
    fn test_lora_error_display() {
        let err = LoraError::RankMismatch { rank: 8, got: 10 };
        assert!(err.to_string().contains("8"));
        assert!(err.to_string().contains("10"));

        let err = LoraError::EmptyAdapter;
        assert!(err.to_string().contains("no layers"));
    }

    #[test]
    fn test_apply_lora_delta_rank2() {
        // W = zeros(2,3), rank=2
        // A = [[1,0,0],[0,1,0]] (2x3)
        // B = [[1,0],[0,1]] (2x2)
        // delta = B @ A = [[1,0,0],[0,1,0]]
        // scale = 2.0 -> W_new = [[2,0,0],[0,2,0]]
        let mut w = vec![0.0; 6];
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        apply_lora_delta(&mut w, &a, &b, 2, 2.0).unwrap();
        assert_eq!(w, vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }
}

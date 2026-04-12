use std::fmt;

/// Lightweight tensor type for CPU-side operations.
/// GPU tensors are managed separately in flare-gpu via wgpu buffers.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let strides = compute_strides(shape);
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        let strides = compute_strides(shape);
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape without copying data. Total element count must match.
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<(), TensorError> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: self.data.len(),
                got: new_size,
            });
        }
        self.shape = new_shape.to_vec();
        self.strides = compute_strides(new_shape);
        Ok(())
    }

    /// Element-wise addition in place: self += other
    pub fn add_inplace(&mut self, other: &Tensor) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.numel(),
                got: other.numel(),
            });
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// Multiply all elements by a scalar
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.data {
            *v *= factor;
        }
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("numel", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
    }

    #[test]
    fn test_from_vec_mismatch() {
        let result = Tensor::from_vec(vec![1.0, 2.0], &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape() {
        let mut t = Tensor::zeros(&[2, 3]);
        t.reshape(&[3, 2]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
    }

    #[test]
    fn test_add_inplace() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        a.add_inplace(&b).unwrap();
        assert_eq!(a.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scale() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        t.scale(2.0);
        assert_eq!(t.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_negative() {
        let mut t = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3]).unwrap();
        t.scale(-1.0);
        assert_eq!(t.data(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_scale_zero() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        t.scale(0.0);
        for &v in t.data() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_add_inplace_shape_mismatch() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(a.add_inplace(&b).is_err());
    }

    #[test]
    fn test_reshape_incompatible() {
        let mut t = Tensor::zeros(&[2, 3]);
        assert!(
            t.reshape(&[2, 4]).is_err(),
            "reshape to incompatible numel should fail"
        );
    }

    #[test]
    fn test_reshape_same_numel() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        t.reshape(&[6]).unwrap();
        assert_eq!(t.shape(), &[6]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_numel_shapes() {
        assert_eq!(Tensor::zeros(&[4]).numel(), 4);
        assert_eq!(Tensor::zeros(&[2, 3]).numel(), 6);
        assert_eq!(Tensor::zeros(&[2, 3, 4]).numel(), 24);
    }

    #[test]
    fn test_ndim() {
        assert_eq!(Tensor::zeros(&[4]).ndim(), 1);
        assert_eq!(Tensor::zeros(&[2, 3]).ndim(), 2);
        assert_eq!(Tensor::zeros(&[2, 3, 4]).ndim(), 3);
    }

    #[test]
    fn test_strides_row_major() {
        // [2, 3, 4] → strides = [12, 4, 1]
        let t = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(t.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_zeros_3d() {
        let t = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }
}

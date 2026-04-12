//! Unified error types for the Flare inference engine.

use thiserror::Error;

/// Top-level error type for Flare operations.
#[derive(Debug, Error)]
pub enum FlareError {
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),

    #[error("tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors related to model loading and inference.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("missing model weights: {0}")]
    MissingWeights(String),

    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),

    #[error("context length exceeded: {current} > {max}")]
    ContextLengthExceeded { current: usize, max: usize },

    #[error("out of memory: need {needed} bytes, available {available}")]
    OutOfMemory { needed: usize, available: usize },
}

/// Convenience result type using FlareError.
pub type Result<T> = std::result::Result<T, FlareError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FlareError::Model(ModelError::UnsupportedArchitecture("gpt4".into()));
        assert_eq!(
            err.to_string(),
            "model error: unsupported model architecture: gpt4"
        );
    }

    #[test]
    fn test_context_length_error() {
        let err = ModelError::ContextLengthExceeded {
            current: 4096,
            max: 2048,
        };
        assert!(err.to_string().contains("4096"));
        assert!(err.to_string().contains("2048"));
    }

    #[test]
    fn test_oom_error() {
        let err = ModelError::OutOfMemory {
            needed: 2_000_000_000,
            available: 1_000_000_000,
        };
        assert!(err.to_string().contains("2000000000"));
    }

    #[test]
    fn test_from_tokenizer_error() {
        let tok_err = crate::tokenizer::TokenizerError::LoadError("bad file".into());
        let err: FlareError = tok_err.into();
        assert!(err.to_string().contains("bad file"));
    }

    #[test]
    fn test_from_tensor_error() {
        let t_err = crate::tensor::TensorError::ShapeMismatch {
            expected: 10,
            got: 5,
        };
        let err: FlareError = t_err.into();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn test_missing_weights_display() {
        let err = ModelError::MissingWeights("embedding".into());
        assert_eq!(err.to_string(), "missing model weights: embedding");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = ModelError::InvalidConfig("hidden_dim=0".into());
        assert_eq!(err.to_string(), "invalid model configuration: hidden_dim=0");
    }

    #[test]
    fn test_flare_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: FlareError = io_err.into();
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_model_error_into_flare_error() {
        let model_err = ModelError::MissingWeights("wq".into());
        let flare_err: FlareError = model_err.into();
        assert!(matches!(flare_err, FlareError::Model(_)));
        assert!(flare_err.to_string().contains("wq"));
    }
}

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

    #[test]
    fn test_flare_error_is_send_sync() {
        // Compile-time assertion: FlareError must be Send + Sync for async use
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FlareError>();
        assert_send_sync::<ModelError>();
    }

    #[test]
    fn test_unsupported_architecture_exact_message() {
        let err = ModelError::UnsupportedArchitecture("phi4".into());
        assert_eq!(err.to_string(), "unsupported model architecture: phi4");
    }

    #[test]
    fn test_oom_available_bytes_in_message() {
        // Both 'needed' and 'available' must appear in the OOM error message
        let err = ModelError::OutOfMemory {
            needed: 3_000_000_000,
            available: 500_000_000,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("3000000000"),
            "needed bytes missing from OOM message: {msg}"
        );
        assert!(
            msg.contains("500000000"),
            "available bytes missing from OOM message: {msg}"
        );
    }

    #[test]
    fn test_model_error_debug_non_empty() {
        // Debug output must be non-empty for every ModelError variant
        let variants: Vec<ModelError> = vec![
            ModelError::UnsupportedArchitecture("x".into()),
            ModelError::MissingWeights("y".into()),
            ModelError::InvalidConfig("z".into()),
            ModelError::ContextLengthExceeded { current: 1, max: 2 },
            ModelError::OutOfMemory {
                needed: 1,
                available: 2,
            },
        ];
        for v in &variants {
            let dbg = format!("{v:?}");
            assert!(!dbg.is_empty(), "Debug output empty for {v}");
        }
    }

    #[test]
    fn test_flare_error_tokenizer_variant_matches() {
        // From<TokenizerError> must produce FlareError::Tokenizer, not another variant
        let tok_err = crate::tokenizer::TokenizerError::LoadError("x".into());
        let flare_err: FlareError = tok_err.into();
        assert!(
            matches!(flare_err, FlareError::Tokenizer(_)),
            "expected FlareError::Tokenizer, got {flare_err:?}"
        );
    }
}

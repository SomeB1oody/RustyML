//! Neural-network error type
//!
//! [`NnError`] enumerates the failures specific to the neural-network framework (layer state,
//! weight shapes, model compilation). Callers receive it through the crate-wide
//! [`Error::NeuralNetwork`](crate::error::Error::NeuralNetwork) variant, into which it converts via
//! `?` (a `#[from]` bridge). See [`crate::error`] for the unified
//! [`Error`](crate::error::Error) that aggregates the per-domain error enums

use crate::error::Error;

/// Neural-network-specific errors, surfaced through [`Error::NeuralNetwork`]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NnError {
    /// An output or gradient was requested from a layer before its forward pass had run
    ///
    /// The payload is the layer's name (e.g. `"Dense"`, `"LSTM"`)
    #[error(
        "forward pass has not been run on layer `{0}`; run `forward` before accessing outputs or `backward`"
    )]
    ForwardPassNotRun(&'static str),

    /// A weight array assigned to a layer did not match the shape the layer expects
    #[error("weight shape mismatch for `{name}`: layer expects {expected:?}, got {found:?}")]
    WeightShape {
        /// The parameter being set (e.g. `"weight"`, `"bias"`)
        name: String,
        /// The shape the layer requires
        expected: Vec<usize>,
        /// The shape that was supplied
        found: Vec<usize>,
    },

    /// The model was used for training/inference before a required component was configured
    ///
    /// The payload names the missing component (e.g. `"optimizer"`, `"loss function"`)
    #[error("model has not been compiled: `{0}` is not specified")]
    NotCompiled(&'static str),

    /// An operation was attempted on a model that contains no layers
    #[error("model has no layers")]
    EmptyModel,
}

impl Error {
    /// Builds [`Error::NeuralNetwork`]`(`[`NnError::ForwardPassNotRun`]`)` for the named layer
    #[cold]
    pub fn forward_pass_not_run(layer: &'static str) -> Self {
        Self::NeuralNetwork(NnError::ForwardPassNotRun(layer))
    }
}

/// Unit tests for `NnError` display formatting and the `forward_pass_not_run` constructor
#[cfg(test)]
mod tests {
    use super::NnError;
    use crate::error::Error;

    /// `#[error(transparent)]` on `Error::NeuralNetwork` forwards the inner `NnError`'s
    /// own Display: `EmptyModel` => `#[error("model has no layers")]`
    #[test]
    fn display_neural_network_transparent_forwards_inner() {
        let inner = NnError::EmptyModel;
        assert_eq!(inner.to_string(), "model has no layers");
        // The transparent outer variant must render identically to the inner enum
        let outer: Error = Error::from(NnError::EmptyModel);
        assert_eq!(outer.to_string(), inner.to_string());
    }

    /// Transparent forwarding also holds for a parameterized `NnError` variant:
    /// `NotCompiled("optimizer")` => `"model has not been compiled: `optimizer` is not specified"`
    #[test]
    fn display_neural_network_transparent_forwards_parameterized_inner() {
        let outer: Error = Error::from(NnError::NotCompiled("optimizer"));
        assert_eq!(
            outer.to_string(),
            "model has not been compiled: `optimizer` is not specified"
        );
    }

    /// `Error::forward_pass_not_run` builds the `NeuralNetwork(ForwardPassNotRun)` variant for the
    /// named layer
    #[test]
    fn forward_pass_not_run_constructor_names_layer() {
        match Error::forward_pass_not_run("Dense") {
            Error::NeuralNetwork(NnError::ForwardPassNotRun(layer)) => assert_eq!(layer, "Dense"),
            other => panic!("expected NeuralNetwork(ForwardPassNotRun), got {other:?}"),
        }
    }
}

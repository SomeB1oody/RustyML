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
        /// The parameter being set (e.g. `"kernel"`, `"bias"`)
        name: String,
        /// The shape the layer needs
        expected: Vec<usize>,
        /// The shape that was supplied
        found: Vec<usize>,
    },

    /// The model was used for training/inference before a needed component was configured
    ///
    /// The payload names the missing component (e.g. `"optimizer"`, `"loss function"`)
    #[error("model has not been compiled: `{0}` is not specified")]
    NotCompiled(&'static str),

    /// An operation was tried on a model that contains no layers
    #[error("model has no layers")]
    EmptyModel,

    /// A layer was used before `build` gave it the arrays it holds
    ///
    /// A layer allocates every array it owns in
    /// [`UnaryLayer::build`](crate::neural_network::traits::UnaryLayer::build), from the shape
    /// of the input. Until then it holds no kernel, no bias, and no shape to check an input
    /// against.
    /// [`UnaryLayer::forward_mut`](crate::neural_network::traits::UnaryLayer::forward_mut) builds
    /// the layer from the tensor it receives, so only the paths that take `&self` can report this
    ///
    /// The payload is the layer's name (e.g. `"Dense"`, `"Conv2D"`)
    #[error(
        "layer `{0}` is not built; build it with `UnaryLayer::build`, or add it to a \
         `SequentialBuilder` and build the model"
    )]
    NotBuilt(&'static str),
}

impl Error {
    /// Builds [`Error::NeuralNetwork`]`(`[`NnError::ForwardPassNotRun`]`)` for the named layer
    #[cold]
    pub fn forward_pass_not_run(layer: &'static str) -> Self {
        Self::NeuralNetwork(NnError::ForwardPassNotRun(layer))
    }

    /// Builds [`Error::NeuralNetwork`]`(`[`NnError::NotBuilt`]`)` for the named layer
    #[cold]
    pub fn not_built(layer: &'static str) -> Self {
        Self::NeuralNetwork(NnError::NotBuilt(layer))
    }
}

/// Unit tests for `NnError` display formatting and the `forward_pass_not_run` constructor
#[cfg(test)]
mod tests {
    use super::NnError;
    use crate::error::Error;

    /// `#[error(transparent)]` on `Error::NeuralNetwork` forwards the inner `NnError`'s
    /// own Display: `EmptyModel` renders as `#[error("model has no layers")]`
    #[test]
    fn display_neural_network_transparent_forwards_inner() {
        let inner = NnError::EmptyModel;
        assert_eq!(inner.to_string(), "model has no layers");
        let outer: Error = Error::from(NnError::EmptyModel);
        assert_eq!(outer.to_string(), inner.to_string());
    }

    /// Transparent forwarding also holds for a parameterized `NnError` variant:
    /// `NotCompiled("optimizer")` renders as
    /// `"model has not been compiled: `optimizer` is not specified"`
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

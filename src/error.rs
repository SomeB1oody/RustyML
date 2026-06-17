//! Error types for RustyML
//!
//! Every fallible operation in the crate returns [`RustymlResult<T>`](crate::error::RustymlResult), an alias for
//! `std::result::Result<T, Error>`. [`Error`](crate::error::Error) is the single, unified error type. It groups
//! domain-specific failures into the nested [`NnError`](crate::error::NnError), [`TreeError`](crate::error::TreeError), and [`IoError`](crate::error::IoError)
//! sub-enums so callers can `match` precisely without the shared variants carrying concerns that
//! only apply to one part of the crate
//!
//! # Categories
//!
//! - **Input validation**: [`Error::EmptyInput`](crate::error::Error::EmptyInput), [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch),
//!   [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch), [`Error::NonFinite`](crate::error::Error::NonFinite), [`Error::InvalidParameter`](crate::error::Error::InvalidParameter),
//!   [`Error::InvalidInput`](crate::error::Error::InvalidInput)
//! - **Model state**: [`Error::NotFitted`](crate::error::Error::NotFitted), and the neural-network states in [`NnError`](crate::error::NnError)
//! - **Numerics / computation**: [`Error::NotConverged`](crate::error::Error::NotConverged), [`Error::Computation`](crate::error::Error::Computation)
//! - **Domain-specific**: [`Error::NeuralNetwork`](crate::error::Error::NeuralNetwork), [`Error::Tree`](crate::error::Error::Tree)
//! - **I/O and serialization**: [`Error::Io`](crate::error::Error::Io)
//!
//! # Conventions
//!
//! - A non-finite **hyperparameter** supplied by the user is an [`Error::InvalidParameter`](crate::error::Error::InvalidParameter)
//!   (the reason mentions finiteness); a non-finite value produced by the **data or a computation**
//!   is an [`Error::NonFinite`](crate::error::Error::NonFinite)
//! - [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch) compares scalar counts (e.g. number of features); use
//!   [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch) when whole tensor shapes differ
//!
//! # Constructing errors
//!
//! Prefer the smart constructors ([`Error::dimension_mismatch`](crate::error::Error::dimension_mismatch), [`Error::invalid_parameter`](crate::error::Error::invalid_parameter), and so on)
//! over building variants by hand. They accept flexible argument types and keep the wording
//! consistent across the crate. To attach context to a foreign error while preserving its source
//! chain, use [`Context::context`](crate::error::Context::context) / [`Context::with_context`](crate::error::Context::with_context)
//!
//! [`Error`](crate::error::Error) is `#[non_exhaustive]`; match with a trailing `_` arm to stay forward-compatible

/// The unified error type for all fallible RustyML operations
///
/// See the [module documentation](crate::error) for the category breakdown and conventions
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An input array, vector, or dataset was empty where data was required
    ///
    /// The payload describes what was empty (e.g. `"input data"`, `"target vector"`)
    #[error("input is empty: {0}")]
    EmptyInput(String),

    /// Two scalar counts that had to agree did not (e.g. number of features at predict time
    /// versus at fit time, or the length of `x` versus `y`)
    #[error("dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// The expected count
        expected: usize,
        /// The count that was actually provided
        found: usize,
    },

    /// Two whole tensor shapes that had to agree did not (e.g. a gradient's shape versus the
    /// activation it flows into)
    #[error("shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        /// The expected shape
        expected: Vec<usize>,
        /// The shape that was actually provided
        found: Vec<usize>,
    },

    /// A value in the data, or produced during a computation, was `NaN` or infinite
    ///
    /// The payload names where it occurred. For an invalid non-finite *hyperparameter*, use
    /// [`Error::InvalidParameter`] instead
    #[error("non-finite value (NaN or infinity) encountered in {0}")]
    NonFinite(String),

    /// A user-supplied hyperparameter was out of its valid range
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// Name of the offending parameter
        name: String,
        /// Why it is invalid (range, sign, finiteness, and so on)
        reason: String,
    },

    /// Input that failed validation in a way not captured by a more specific variant
    /// (e.g. an unexpected tensor rank, malformed labels, or a relational constraint between
    /// the data and the configuration)
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A method that requires a trained model was called before the model was fitted
    ///
    /// The payload is the model's name (e.g. `"KMeans"`)
    #[error("model `{0}` has not been fitted; call `fit` before this operation")]
    NotFitted(&'static str),

    /// An iterative algorithm failed to reach its convergence criterion
    #[error("failed to converge: {0}")]
    NotConverged(String),

    /// A computation failed for a reason that is not a validation problem (numerical breakdown,
    /// a violated internal invariant, or a wrapped lower-level error)
    ///
    /// When wrapping a foreign error, prefer [`Context::context`] so the original error is
    /// preserved as the [`source`](std::error::Error::source)
    #[error("computation failed: {context}")]
    Computation {
        /// Human-readable description of what failed
        context: String,
        /// The underlying error, if this wraps one
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// A neural-network-specific error. See [`NnError`]
    #[error(transparent)]
    NeuralNetwork(#[from] NnError),

    /// A decision-tree-specific error. See [`TreeError`]
    #[error(transparent)]
    Tree(#[from] TreeError),

    /// An I/O or serialization error. See [`IoError`]
    #[error(transparent)]
    Io(#[from] IoError),
}

impl Error {
    /// Builds an [`Error::EmptyInput`] describing what was empty
    pub fn empty_input(what: impl Into<String>) -> Self {
        Self::EmptyInput(what.into())
    }

    /// Builds an [`Error::DimensionMismatch`] from the expected and found counts
    pub fn dimension_mismatch(expected: usize, found: usize) -> Self {
        Self::DimensionMismatch { expected, found }
    }

    /// Builds an [`Error::ShapeMismatch`] from the expected and found shapes
    ///
    /// Accepts anything convertible to `Vec<usize>`, including `&[usize]` (e.g. `array.shape()`)
    pub fn shape_mismatch(expected: impl Into<Vec<usize>>, found: impl Into<Vec<usize>>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Builds an [`Error::NonFinite`] naming where the non-finite value occurred
    pub fn non_finite(context: impl Into<String>) -> Self {
        Self::NonFinite(context.into())
    }

    /// Builds an [`Error::InvalidParameter`] from a parameter name and the reason it is invalid
    pub fn invalid_parameter(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Builds an [`Error::InvalidInput`] (the fallback for validation failures without a more
    /// specific variant)
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Builds an [`Error::NotFitted`] for the named model
    pub fn not_fitted(model: &'static str) -> Self {
        Self::NotFitted(model)
    }

    /// Builds an [`Error::NotConverged`] with a description of the failure
    pub fn not_converged(msg: impl Into<String>) -> Self {
        Self::NotConverged(msg.into())
    }

    /// Builds an [`Error::Computation`] with no wrapped source
    ///
    /// To wrap a lower-level error, use [`Context::context`] instead
    pub fn computation(context: impl Into<String>) -> Self {
        Self::Computation {
            context: context.into(),
            source: None,
        }
    }

    /// Builds [`Error::NeuralNetwork`]`(`[`NnError::ForwardPassNotRun`]`)` for the named layer
    pub fn forward_pass_not_run(layer: &'static str) -> Self {
        Self::NeuralNetwork(NnError::ForwardPassNotRun(layer))
    }
}

/// Lets `?` lift a raw [`std::io::Error`] directly into [`Error`] (as [`IoError::Std`])
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(IoError::Std(e))
    }
}

/// Lets `?` lift a raw [`postcard::Error`] directly into [`Error`] (as [`IoError::Serialization`])
impl From<postcard::Error> for Error {
    fn from(e: postcard::Error) -> Self {
        Self::Io(IoError::Serialization(e))
    }
}

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

/// Decision-tree-specific errors, surfaced through [`Error::Tree`]
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TreeError {
    /// A classification-only operation (e.g. `predict_proba`) was called on a regression tree
    #[error("operation requires a classification tree")]
    NotClassificationTree,

    /// The tree's internal structure violated an invariant (a missing child, an absent
    /// categorical fallback, or a leaf without stored probabilities)
    #[error("corrupt tree structure: {0}")]
    CorruptStructure(&'static str),
}

/// I/O and serialization errors, surfaced through [`Error::Io`]
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// A standard I/O error from a filesystem operation
    #[error("I/O error: {0}")]
    Std(#[from] std::io::Error),

    /// A binary serialization or deserialization error (postcard format)
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    /// The model being loaded does not match the saved model: a different number of layers, a
    /// different layer type at some position, or a weight whose shape does not match the target
    /// layer's configured shape
    #[error("model structure mismatch: {0}")]
    ModelStructureMismatch(String),
}

/// An alias for `Result<T, `[`Error`]`>`
///
/// Named distinctly, rather than shadowing [`std::result::Result`], so that a
/// `-> RustymlResult<T>` signature unmistakably means *this* crate's [`Error`]. Only [`Context`]
/// and the public signatures that use it need this alias
pub type RustymlResult<T> = std::result::Result<T, Error>;

/// Extension methods for attaching context to a foreign error while preserving its source chain
///
/// Implemented for every `Result<T, E>` whose error type is a `Send + Sync + 'static`
/// [`std::error::Error`], so it composes with `?`:
///
/// ```ignore
/// use crate::error::Context;
/// let reshaped = array.into_shape(shape).context("reshape for layer normalization")?;
/// ```
pub trait Context<T> {
    /// Wraps the error as the [`source`](std::error::Error::source) of an [`Error::Computation`]
    /// carrying the given context
    ///
    /// The `context` argument is evaluated **eagerly**, before the call, on both the `Ok` and
    /// `Err` paths. Use this when the context is a string literal or a value you already hold
    /// If producing the message does any work (e.g. `format!`), use
    /// [`with_context`](Context::with_context) instead so that work happens only on failure
    fn context(self, context: impl Into<String>) -> RustymlResult<T>;

    /// Like [`context`](Context::context), but the context is produced **lazily** by a closure
    /// that runs only on the `Err` path
    ///
    /// Prefer this whenever building the message allocates or computes, typically anything using
    /// `format!`, so the common `Ok` path pays nothing
    fn with_context<F, S>(self, f: F) -> RustymlResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E> Context<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context(self, context: impl Into<String>) -> RustymlResult<T> {
        self.map_err(|e| Error::Computation {
            context: context.into(),
            source: Some(Box::new(e)),
        })
    }

    fn with_context<F, S>(self, f: F) -> RustymlResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| Error::Computation {
            context: f().into(),
            source: Some(Box::new(e)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::error::Error as StdError;

    /// `.context(msg)` on an `Err` wraps it as `Error::Computation` carrying the message, and the
    /// original error is preserved as the `source` (downcastable to the concrete type), so the
    /// error chain is not lost
    #[test]
    fn context_wraps_err_as_computation_preserving_source() {
        let parsed: Result<i32, _> = "not a number".parse::<i32>();
        match parsed.context("parsing the threshold") {
            Err(Error::Computation { context, source }) => {
                assert_eq!(context, "parsing the threshold");
                let src = source.expect("the original error must be preserved as the source");
                assert!(
                    src.downcast_ref::<std::num::ParseIntError>().is_some(),
                    "source must downcast back to the original ParseIntError"
                );
            }
            other => panic!("expected Error::Computation, got {other:?}"),
        }
    }

    /// The wrapped error is reachable through the standard `std::error::Error::source()` chain
    #[test]
    fn context_exposes_wrapped_error_via_std_source() {
        let err = "x".parse::<i32>().context("ctx").unwrap_err();
        let src = StdError::source(&err).expect("source() must return Some");
        assert!(src.downcast_ref::<std::num::ParseIntError>().is_some());
    }

    /// `.context(_)` is a transparent pass-through on the `Ok` path
    #[test]
    fn context_is_passthrough_on_ok() {
        let ok: Result<i32, std::num::ParseIntError> = Ok(42);
        assert_eq!(ok.context("unused").unwrap(), 42);
    }

    /// `with_context`'s closure is **lazy**: it runs only on the `Err` path, never on `Ok`
    #[test]
    fn with_context_closure_runs_only_on_err() {
        // Ok path: the closure must not run
        let ran_on_ok = Cell::new(false);
        let ok: Result<i32, std::num::ParseIntError> = Ok(7);
        let passed = ok.with_context(|| {
            ran_on_ok.set(true);
            "should never be built"
        });
        assert_eq!(passed.unwrap(), 7);
        assert!(
            !ran_on_ok.get(),
            "with_context closure must not run on the Ok path"
        );

        // Err path: the closure runs and supplies the context
        let ran_on_err = Cell::new(false);
        let wrapped = "nope".parse::<i32>().with_context(|| {
            ran_on_err.set(true);
            format!("lazy context {}", 1)
        });
        assert!(
            ran_on_err.get(),
            "with_context closure must run on the Err path"
        );
        match wrapped {
            Err(Error::Computation { context, source }) => {
                assert_eq!(context, "lazy context 1");
                assert!(source.is_some(), "source must be preserved");
            }
            other => panic!("expected Error::Computation, got {other:?}"),
        }
    }
    // Display / to_string rendering of the #[error("...")] messages. Each expected string is
    // reproduced by hand from the `#[error("...")]` format string; nothing here runs the impl to
    // obtain the text. Debug-formatted `Vec<usize>` (the `{:?}` in ShapeMismatch) renders as
    // `[a, b]` with `, ` separators

    /// `#[error("input is empty: {0}")]` with payload `"target vector"`
    #[test]
    fn display_empty_input() {
        let e = Error::empty_input("target vector");
        assert_eq!(e.to_string(), "input is empty: target vector");
    }

    /// `#[error("dimension mismatch: expected {expected}, found {found}")]`
    #[test]
    fn display_dimension_mismatch() {
        let e = Error::dimension_mismatch(3, 5);
        assert_eq!(e.to_string(), "dimension mismatch: expected 3, found 5");
    }

    /// `#[error("shape mismatch: expected {expected:?}, found {found:?}")]`; the `{:?}`
    /// renders each `Vec<usize>` as `[.., ..]`
    #[test]
    fn display_shape_mismatch() {
        let e = Error::shape_mismatch(vec![2usize, 3], vec![2usize, 4]);
        assert_eq!(
            e.to_string(),
            "shape mismatch: expected [2, 3], found [2, 4]"
        );
    }

    /// `#[error("invalid parameter `{name}`: {reason}")]`; note the literal backticks
    /// around the parameter name
    #[test]
    fn display_invalid_parameter() {
        let e = Error::invalid_parameter("C", "must be > 0");
        assert_eq!(e.to_string(), "invalid parameter `C`: must be > 0");
    }

    /// `#[error("model `{0}` has not been fitted; call `fit` before this operation")]`
    #[test]
    fn display_not_fitted() {
        let e = Error::not_fitted("KMeans");
        assert_eq!(
            e.to_string(),
            "model `KMeans` has not been fitted; call `fit` before this operation"
        );
    }

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

    /// `#[error(transparent)]` on `Error::Tree` forwards the inner `TreeError`'s Display:
    /// `NotClassificationTree` => `#[error("operation requires a classification tree")]`
    #[test]
    fn display_tree_transparent_forwards_inner() {
        let inner = TreeError::NotClassificationTree;
        assert_eq!(
            inner.to_string(),
            "operation requires a classification tree"
        );
        let outer: Error = Error::from(TreeError::NotClassificationTree);
        assert_eq!(outer.to_string(), inner.to_string());
    }
}

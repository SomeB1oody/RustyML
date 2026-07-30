//! Error types for RustyML
//!
//! Every fallible operation in the crate returns [`RustymlResult<T>`](crate::error::RustymlResult),
//! an alias for `std::result::Result<T, Error>`. [`Error`](crate::error::Error) is the single,
//! unified error type.
//!
//! `Error` aggregates 2 domain-specific error enums: `NnError` (from `neural_network`) and
//! `TreeError` (from `machine_learning`). It also wraps the shared
//! [`IoError`](crate::error::IoError). Callers can `match` precisely, since a shared variant never
//! carries a concern specific to one part of the crate
//!
//! # Categories
//!
//! - **Input validation**: [`Error::EmptyInput`](crate::error::Error::EmptyInput),
//!   [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch),
//!   [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch),
//!   [`Error::NonFinite`](crate::error::Error::NonFinite),
//!   [`Error::InvalidParameter`](crate::error::Error::InvalidParameter),
//!   [`Error::InvalidInput`](crate::error::Error::InvalidInput)
//! - **Model state**: [`Error::NotFitted`](crate::error::Error::NotFitted), and the neural-network
//!   states in `NnError`
//! - **Numerics / computation**: [`Error::NotConverged`](crate::error::Error::NotConverged),
//!   [`Error::Computation`](crate::error::Error::Computation)
//! - **Domain-specific**: `Error::NeuralNetwork` (wrapping `neural_network`'s `NnError`) and
//!   `Error::Tree` (wrapping `machine_learning`'s `TreeError`)
//! - **I/O and serialization**: [`Error::Io`](crate::error::Error::Io)
//!
//! # Conventions
//!
//! - A non-finite **hyperparameter** from the user is an
//!   [`Error::InvalidParameter`](crate::error::Error::InvalidParameter) (the reason names
//!   finiteness). A non-finite value from the **data or a computation** is an
//!   [`Error::NonFinite`](crate::error::Error::NonFinite)
//! - [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch) compares scalar counts
//!   (e.g. number of features). Use [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch)
//!   when whole tensor shapes differ
//!
//! # Constructing errors
//!
//! Prefer the smart constructors
//! ([`Error::dimension_mismatch`](crate::error::Error::dimension_mismatch),
//! [`Error::invalid_parameter`](crate::error::Error::invalid_parameter), and so on) over building
//! variants by hand. They accept flexible argument types and keep the wording consistent across the
//! crate. To attach context to a foreign error while preserving its source chain, use
//! [`Context::context`](crate::error::Context::context) /
//! [`Context::with_context`](crate::error::Context::with_context)
//!
//! [`Error`](crate::error::Error) is `#[non_exhaustive]`. Match with a trailing `_` arm to stay
//! forward-compatible

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

    /// 2 scalar counts that had to agree did not. Examples: number of features at predict
    /// time versus at fit time, or the length of `x` versus `y`
    #[error("dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// The expected count
        expected: usize,
        /// The count that was actually provided
        found: usize,
    },

    /// 2 whole tensor shapes that had to agree did not (e.g. a gradient's shape versus the
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

    /// Input that failed validation in a way not captured by a more specific variant. Examples:
    /// an unexpected tensor rank, malformed labels, or a relational constraint between the data
    /// and the configuration
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
    /// When wrapping a foreign error, prefer [`Context::context`], which preserves the original
    /// error as the [`source`](std::error::Error::source)
    #[error("computation failed: {context}")]
    Computation {
        /// Human-readable description of what failed
        context: String,
        /// The underlying error, if this wraps one
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// A neural-network-specific error. See [`NnError`](crate::neural_network::NnError)
    #[cfg(feature = "neural_network")]
    #[error(transparent)]
    NeuralNetwork(#[from] crate::neural_network::NnError),

    /// A decision-tree-specific error. See [`TreeError`](crate::machine_learning::TreeError)
    #[cfg(feature = "machine_learning")]
    #[error(transparent)]
    Tree(#[from] crate::machine_learning::TreeError),

    /// An I/O or serialization error. See [`IoError`]
    #[error(transparent)]
    Io(#[from] IoError),
}

impl Error {
    /// Builds an [`Error::EmptyInput`] describing what was empty
    #[cold]
    pub fn empty_input(what: impl Into<String>) -> Self {
        Self::EmptyInput(what.into())
    }

    /// Builds an [`Error::DimensionMismatch`] from the expected and found counts
    #[cold]
    pub fn dimension_mismatch(expected: usize, found: usize) -> Self {
        Self::DimensionMismatch { expected, found }
    }

    /// Builds an [`Error::ShapeMismatch`] from the expected and found shapes
    ///
    /// Accepts anything convertible to `Vec<usize>`, including `&[usize]` (e.g. `array.shape()`)
    #[cold]
    pub fn shape_mismatch(expected: impl Into<Vec<usize>>, found: impl Into<Vec<usize>>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Builds an [`Error::NonFinite`] naming where the non-finite value occurred
    #[cold]
    pub fn non_finite(context: impl Into<String>) -> Self {
        Self::NonFinite(context.into())
    }

    /// Builds an [`Error::InvalidParameter`] from a parameter name and the reason it is invalid
    #[cold]
    pub fn invalid_parameter(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Builds an [`Error::InvalidInput`] (the fallback for validation failures without a more
    /// specific variant)
    #[cold]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Builds an [`Error::NotFitted`] for the named model
    #[cold]
    pub fn not_fitted(model: &'static str) -> Self {
        Self::NotFitted(model)
    }

    /// Builds an [`Error::NotConverged`] with a description of the failure
    #[cold]
    pub fn not_converged(msg: impl Into<String>) -> Self {
        Self::NotConverged(msg.into())
    }

    /// Builds an [`Error::Computation`] with no wrapped source
    ///
    /// To wrap a lower-level error, use [`Context::context`] instead
    #[cold]
    pub fn computation(context: impl Into<String>) -> Self {
        Self::Computation {
            context: context.into(),
            source: None,
        }
    }
}

/// Lets `?` lift a raw [`std::io::Error`] directly into [`Error`] (as [`IoError::Std`])
impl From<std::io::Error> for Error {
    #[cold]
    fn from(e: std::io::Error) -> Self {
        Self::Io(IoError::Std(e))
    }
}

/// Lets `?` lift a raw [`postcard::Error`] directly into [`Error`] (as [`IoError::Serialization`])
impl From<postcard::Error> for Error {
    #[cold]
    fn from(e: postcard::Error) -> Self {
        Self::Io(IoError::Serialization(e))
    }
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

    /// The loaded model does not match the saved model. Causes include a different number of
    /// layers, a different layer type at some position, or a weight shape that does not match
    /// the target layer
    #[error("model structure mismatch: {0}")]
    ModelStructureMismatch(String),

    /// The file is not a RustyML model file, or its on-disk format version is not the one this
    /// build writes
    ///
    /// Weight containers carry no layout tag of their own. A stale file's shapes can match the
    /// current model's by coincidence. A square convolution kernel is one example. A `Flatten` ->
    /// `Dense` pair is another, since its `Dense` weight shape stays the same under either tensor
    /// layout. The format version is the only reliable guard against loading weights laid out for
    /// a previous release
    #[error("unsupported model format: {0}")]
    UnsupportedModelFormat(String),
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
    /// `Err` paths. Use this when the context is a string literal or a value you already hold.
    /// If producing the message does any work (e.g. `format!`), use
    /// [`with_context`](Context::with_context) instead, so that work happens only on failure
    fn context(self, context: impl Into<String>) -> RustymlResult<T>;

    /// Like [`context`](Context::context), but a closure produces the context **lazily**, running
    /// only on the `Err` path
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
    #[cold]
    fn context(self, context: impl Into<String>) -> RustymlResult<T> {
        self.map_err(|e| Error::Computation {
            context: context.into(),
            source: Some(Box::new(e)),
        })
    }

    #[cold]
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

/// Unit tests for error construction, `Display` formatting, and the `Context` extension trait
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::error::Error as StdError;

    /// `.context(msg)` on an `Err` wraps it as `Error::Computation`, carrying the message. The
    /// original error survives as the `source` (downcastable to the concrete type), so the chain
    /// is not lost
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
    // Each expected string below is reproduced by hand from its `#[error("...")]` format string.
    // Debug-formatted `Vec<usize>` (the `{:?}` in ShapeMismatch) renders as `[a, b]`.

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

    /// `#[error("shape mismatch: expected {expected:?}, found {found:?}")]`. The `{:?}`
    /// renders each `Vec<usize>` as `[.., ..]`
    #[test]
    fn display_shape_mismatch() {
        let e = Error::shape_mismatch(vec![2usize, 3], vec![2usize, 4]);
        assert_eq!(
            e.to_string(),
            "shape mismatch: expected [2, 3], found [2, 4]"
        );
    }

    /// `#[error("invalid parameter `{name}`: {reason}")]`. Note the literal backticks
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

    // Below: smart constructors not covered above (NonFinite, InvalidInput, the source-less
    // Computation constructor), plus the RustymlResult alias.

    /// `Error::non_finite` builds `NonFinite` carrying the supplied context string
    #[test]
    fn non_finite_constructor_carries_context() {
        match Error::non_finite("weights") {
            Error::NonFinite(ref ctx) => assert!(
                ctx.contains("weights"),
                "context should mention 'weights', got: {ctx}"
            ),
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }

    /// `Error::invalid_input` builds `InvalidInput` carrying the supplied message
    #[test]
    fn invalid_input_constructor_carries_message() {
        match Error::invalid_input("unexpected rank") {
            Error::InvalidInput(ref msg) => assert!(
                msg.contains("unexpected rank"),
                "message should mention the supplied text, got: {msg}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// `Error::computation` builds `Computation` with the context set and `source` `None`.
    /// Wrap a lower-level error as the source through `Context::context` instead
    #[test]
    fn computation_constructor_has_no_source() {
        match Error::computation("overflow") {
            Error::Computation {
                ref context,
                ref source,
            } => {
                assert!(
                    context.contains("overflow"),
                    "context should contain 'overflow'"
                );
                assert!(
                    source.is_none(),
                    "source should be None for Error::computation"
                );
            }
            other => panic!("expected Computation, got {other:?}"),
        }
    }

    /// `RustymlResult<T>` is a transparent alias for `Result<T, Error>` and is usable as one
    #[test]
    fn rustyml_result_is_result_alias() {
        let ok: RustymlResult<i32> = Ok(42);
        assert!(matches!(ok, Ok(42)));
        let err: RustymlResult<i32> = Err(Error::empty_input("test"));
        assert!(matches!(err, Err(Error::EmptyInput(_))));
    }
}

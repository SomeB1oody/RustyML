//! Error types for RustyML.
//!
//! Every fallible operation in the crate returns [`RustymlResult<T>`](crate::error::RustymlResult), an alias for
//! `std::result::Result<T, Error>`. [`Error`](crate::error::Error) is the single, unified error type; it groups
//! genuinely domain-specific failures into the nested [`NnError`](crate::error::NnError), [`TreeError`](crate::error::TreeError), and [`IoError`](crate::error::IoError)
//! sub-enums so callers can `match` precisely without the shared variants being polluted by
//! concerns that only apply to one part of the crate.
//!
//! # Categories
//!
//! - **Input validation** — [`Error::EmptyInput`](crate::error::Error::EmptyInput), [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch),
//!   [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch), [`Error::NonFinite`](crate::error::Error::NonFinite), [`Error::InvalidParameter`](crate::error::Error::InvalidParameter),
//!   [`Error::InvalidInput`](crate::error::Error::InvalidInput).
//! - **Model state** — [`Error::NotFitted`](crate::error::Error::NotFitted), and the neural-network states in [`NnError`](crate::error::NnError).
//! - **Numerics / computation** — [`Error::NotConverged`](crate::error::Error::NotConverged), [`Error::Computation`](crate::error::Error::Computation).
//! - **Domain-specific** — [`Error::NeuralNetwork`](crate::error::Error::NeuralNetwork), [`Error::Tree`](crate::error::Error::Tree).
//! - **I/O & serialization** — [`Error::Io`](crate::error::Error::Io).
//!
//! # Conventions
//!
//! - A non-finite **hyperparameter** supplied by the user is an [`Error::InvalidParameter`](crate::error::Error::InvalidParameter)
//!   (the reason mentions finiteness); a non-finite value produced by the **data or a computation**
//!   is an [`Error::NonFinite`](crate::error::Error::NonFinite).
//! - [`Error::DimensionMismatch`](crate::error::Error::DimensionMismatch) compares scalar counts (e.g. number of features); use
//!   [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch) when whole tensor shapes differ.
//!
//! # Constructing errors
//!
//! Prefer the smart constructors ([`Error::dimension_mismatch`](crate::error::Error::dimension_mismatch), [`Error::invalid_parameter`](crate::error::Error::invalid_parameter), …)
//! over building variants by hand — they accept flexible argument types and keep the wording
//! consistent across the crate. To attach context to a foreign error while preserving its source
//! chain, use [`Context::context`](crate::error::Context::context) / [`Context::with_context`](crate::error::Context::with_context).
//!
//! [`Error`](crate::error::Error) is `#[non_exhaustive]`; match with a trailing `_` arm to stay forward-compatible.

use std::fs::File;
use std::io::BufReader;

/// The unified error type for all fallible RustyML operations.
///
/// See the [module documentation](crate::error) for the category breakdown and conventions.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An input array, vector, or dataset was empty where data was required.
    ///
    /// The payload describes what was empty (e.g. `"input data"`, `"target vector"`).
    #[error("input is empty: {0}")]
    EmptyInput(String),

    /// Two scalar counts that had to agree did not (e.g. number of features at predict time
    /// versus at fit time, or the length of `x` versus `y`).
    #[error("dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// The expected count.
        expected: usize,
        /// The count that was actually provided.
        found: usize,
    },

    /// Two whole tensor shapes that had to agree did not (e.g. a gradient's shape versus the
    /// activation it flows into).
    #[error("shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The shape that was actually provided.
        found: Vec<usize>,
    },

    /// A value in the data, or produced during a computation, was `NaN` or infinite.
    ///
    /// The payload names where it occurred. For an invalid non-finite *hyperparameter*, use
    /// [`Error::InvalidParameter`] instead.
    #[error("non-finite value (NaN or infinity) encountered in {0}")]
    NonFinite(String),

    /// A user-supplied hyperparameter was out of its valid range.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// Name of the offending parameter.
        name: String,
        /// Why it is invalid (range, sign, finiteness, …).
        reason: String,
    },

    /// Input that failed validation in a way not captured by a more specific variant
    /// (e.g. an unexpected tensor rank, malformed labels, or a relational constraint between
    /// the data and the configuration).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A method that requires a trained model was called before the model was fitted.
    ///
    /// The payload is the model's name (e.g. `"KMeans"`).
    #[error("model `{0}` has not been fitted; call `fit` before this operation")]
    NotFitted(&'static str),

    /// An iterative algorithm failed to reach its convergence criterion.
    #[error("failed to converge: {0}")]
    NotConverged(String),

    /// A computation failed for a reason that is not a validation problem (numerical breakdown,
    /// a violated internal invariant, or a wrapped lower-level error).
    ///
    /// When wrapping a foreign error, prefer [`Context::context`] so the original error is
    /// preserved as the [`source`](std::error::Error::source).
    #[error("computation failed: {context}")]
    Computation {
        /// Human-readable description of what failed.
        context: String,
        /// The underlying error, if this wraps one.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// A neural-network-specific error. See [`NnError`].
    #[error(transparent)]
    NeuralNetwork(#[from] NnError),

    /// A decision-tree-specific error. See [`TreeError`].
    #[error(transparent)]
    Tree(#[from] TreeError),

    /// An I/O or serialization error. See [`IoError`].
    #[error(transparent)]
    Io(#[from] IoError),
}

impl Error {
    /// Builds an [`Error::EmptyInput`] describing what was empty.
    pub fn empty_input(what: impl Into<String>) -> Self {
        Self::EmptyInput(what.into())
    }

    /// Builds an [`Error::DimensionMismatch`] from the expected and found counts.
    pub fn dimension_mismatch(expected: usize, found: usize) -> Self {
        Self::DimensionMismatch { expected, found }
    }

    /// Builds an [`Error::ShapeMismatch`] from the expected and found shapes.
    ///
    /// Accepts anything convertible to `Vec<usize>`, including `&[usize]` (e.g. `array.shape()`).
    pub fn shape_mismatch(expected: impl Into<Vec<usize>>, found: impl Into<Vec<usize>>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Builds an [`Error::NonFinite`] naming where the non-finite value occurred.
    pub fn non_finite(context: impl Into<String>) -> Self {
        Self::NonFinite(context.into())
    }

    /// Builds an [`Error::InvalidParameter`] from a parameter name and the reason it is invalid.
    pub fn invalid_parameter(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Builds an [`Error::InvalidInput`] (the fallback for validation failures without a more
    /// specific variant).
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Builds an [`Error::NotFitted`] for the named model.
    pub fn not_fitted(model: &'static str) -> Self {
        Self::NotFitted(model)
    }

    /// Builds an [`Error::NotConverged`] with a description of the failure.
    pub fn not_converged(msg: impl Into<String>) -> Self {
        Self::NotConverged(msg.into())
    }

    /// Builds an [`Error::Computation`] with no wrapped source.
    ///
    /// To wrap a lower-level error, use [`Context::context`] instead.
    pub fn computation(context: impl Into<String>) -> Self {
        Self::Computation {
            context: context.into(),
            source: None,
        }
    }

    /// Convenience for the very common neural-network guard: builds
    /// [`Error::NeuralNetwork`]`(`[`NnError::ForwardPassNotRun`]`)` for the named layer.
    pub fn forward_pass_not_run(layer: &'static str) -> Self {
        Self::NeuralNetwork(NnError::ForwardPassNotRun(layer))
    }
}

/// Lets `?` lift a raw [`std::io::Error`] directly into [`Error`] (as [`IoError::Std`]).
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(IoError::Std(e))
    }
}

/// Lets `?` lift a raw [`serde_json::Error`] directly into [`Error`] (as [`IoError::Json`]).
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Io(IoError::Json(e))
    }
}

/// Neural-network-specific errors, surfaced through [`Error::NeuralNetwork`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NnError {
    /// An output or gradient was requested from a layer before its forward pass had run.
    ///
    /// The payload is the layer's name (e.g. `"Dense"`, `"LSTM"`).
    #[error("forward pass has not been run on layer `{0}`; run `forward` before accessing outputs or `backward`")]
    ForwardPassNotRun(&'static str),

    /// A weight array assigned to a layer did not match the shape the layer expects.
    #[error("weight shape mismatch for `{name}`: layer expects {expected:?}, got {found:?}")]
    WeightShape {
        /// The parameter being set (e.g. `"weight"`, `"bias"`).
        name: String,
        /// The shape the layer requires.
        expected: Vec<usize>,
        /// The shape that was supplied.
        found: Vec<usize>,
    },

    /// The model was used for training/inference before a required component was configured.
    ///
    /// The payload names the missing component (e.g. `"optimizer"`, `"loss function"`).
    #[error("model has not been compiled: `{0}` is not specified")]
    NotCompiled(&'static str),

    /// An operation was attempted on a model that contains no layers.
    #[error("model has no layers")]
    EmptyModel,
}

/// Decision-tree-specific errors, surfaced through [`Error::Tree`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TreeError {
    /// A classification-only operation (e.g. `predict_proba`) was called on a regression tree.
    #[error("operation requires a classification tree")]
    NotClassificationTree,

    /// The tree's internal structure violated an invariant (a missing child, an absent
    /// categorical fallback, or a leaf without stored probabilities).
    #[error("corrupt tree structure: {0}")]
    CorruptStructure(&'static str),
}

/// I/O and serialization errors, surfaced through [`Error::Io`].
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// A standard I/O error from a filesystem operation.
    #[error("I/O error: {0}")]
    Std(#[from] std::io::Error),

    /// A JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The model being loaded does not match the saved model: a different number of layers, a
    /// different layer type at some position, or a weight whose shape does not match the target
    /// layer's configured shape.
    #[error("model structure mismatch: {0}")]
    ModelStructureMismatch(String),
}

impl IoError {
    /// Opens `path` and wraps it in a [`BufReader`], returning the raw [`std::io::Error`] on
    /// failure (callers in `Error`-returning functions receive it as [`Error::Io`] via `?`).
    pub fn load_in_buf_reader(path: impl AsRef<std::path::Path>) -> std::io::Result<BufReader<File>> {
        Ok(BufReader::new(File::open(path)?))
    }
}

/// A convenient alias for `Result<T, `[`Error`]`>`. Named distinctly — rather than shadowing
/// [`std::result::Result`] — so that a `-> RustymlResult<T>` signature unmistakably means *this*
/// crate's [`Error`]. Only [`Context`] and the public signatures that use it need this alias.
pub type RustymlResult<T> = std::result::Result<T, Error>;

/// Extension methods for attaching context to a foreign error while preserving its source chain.
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
    /// carrying the given context.
    ///
    /// The `context` argument is evaluated **eagerly** — before the call, on both the `Ok` and
    /// `Err` paths. Use this when the context is a string literal or a value you already hold.
    /// If producing the message does any work (e.g. `format!`), use
    /// [`with_context`](Context::with_context) instead so that work happens only on failure.
    fn context(self, context: impl Into<String>) -> RustymlResult<T>;

    /// Like [`context`](Context::context), but the context is produced **lazily** by a closure
    /// that runs only on the `Err` path.
    ///
    /// Prefer this whenever building the message allocates or computes — typically anything using
    /// `format!` — so the common `Ok` path pays nothing:
    ///
    /// ```ignore
    /// // `format!` runs only if `into_shape` actually fails:
    /// arr.into_shape(shape).with_context(|| format!("reshape to {shape:?}"))?;
    /// ```
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

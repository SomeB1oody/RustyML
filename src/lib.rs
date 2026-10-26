//! Machine learning estimators, neural networks, numeric kernels, and data utilities.
//!
//! Feature flags select each category and its common prelude exports.
//! The crate includes model serialization and thread-local random seed controls.
//! Evaluation metrics return scores directly and panic when input checks fail.
//!
//! ## Quick Start
//!
//! ### Machine Learning Example
//!
//! Add RustyML to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustyml = "*"
//! # The default feature set is `full`. To slim the build, set `default-features = false` and
//! # list the features you need, for example `features = ["machine_learning"]`
//! # Add `"show_progress"` to show progress bars during training
//! ```
//!
//! In your Rust code, write:
//! ```rust,no_run
//! use rustyml::machine_learning::LinearRegression;
//! use rustyml::machine_learning::linear_model::LeastSquaresSolver;
//! use ndarray::{Array1, Array2};
//!
//! // Create a linear regression model.
//! let mut model = LinearRegression::new(true).with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 }).unwrap();
//!
//! // Prepare training data.
//! let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
//! let raw_y = vec![6.0, 9.0, 12.0];
//!
//! // Convert Vec to ndarray types.
//! let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
//! let y = Array1::from_vec(raw_y);
//!
//! // Train the model.
//! model.fit(&x.view(), &y.view()).unwrap();
//!
//! // Make predictions.
//! let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
//! let _predictions = model.predict(&new_data.view());
//!
//! // Save the trained model to a file.
//! model.save_to_path("linear_regression_model.bin").unwrap();
//!
//! // Load the model from the file.
//! let loaded_model = LinearRegression::load_from_path("linear_regression_model.bin").unwrap();
//!
//! // Use the loaded model for predictions.
//! let _loaded_predictions = loaded_model.predict(&new_data.view());
//!
//! // Clone is implemented.
//! let _model_copy = model.clone();
//!
//! // Debug is implemented.
//! println!("{:?}", model);
//! ```
//!
//! ## Neural Network Example
//!
//! Add RustyML to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustyml = "*"
//! # The default feature set is `full`. To slim the build, set `default-features = false` and
//! # list the features you need, for example `features = ["neural_network"]`
//! # Add `"show_progress"` to show progress bars during training
//! ```
//!
//! In your Rust code, write:
//! ```rust,no_run
//! use rustyml::neural_network::{
//!     Shape,
//!     sequential::SequentialBuilder,
//!     layers::{Activation, Dense},
//!     optimizers::Adam,
//!     losses::CategoricalCrossEntropy,
//! };
//! use ndarray::Array;
//!
//! // Create training data.
//! let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features
//! let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
//!
//! // Build a neural network.
//! let mut model = SequentialBuilder::new()
//!     .add(Dense::new(128, Activation::ReLU).unwrap())
//!     .add(Dense::new(64, Activation::ReLU).unwrap())
//!     .add(Dense::new(10, Activation::Softmax { axis: -1 }).unwrap())
//!     .build(&Shape::known(x.shape()))
//!     .unwrap();
//! model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
//!
//! // Display model structure.
//! model.summary();
//!
//! // Train the model.
//! // The returned History holds 1 loss per epoch, measured during that epoch, not after it.
//! let history = model.fit(&x, &y, 10).unwrap();
//! println!("Per-epoch loss: {:?}", history.loss());
//!
//! // Score the model's current weights, an inference-mode pass that changes nothing.
//! println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());
//!
//! // Save model weights to file.
//! model.save_to_path("model.bin").unwrap();
//!
//! // Create a new model with the same architecture.
//! let mut new_model = SequentialBuilder::new()
//!     .add(Dense::new(128, Activation::ReLU).unwrap())
//!     .add(Dense::new(64, Activation::ReLU).unwrap())
//!     .add(Dense::new(10, Activation::Softmax { axis: -1 }).unwrap())
//!     .build(&Shape::known(x.shape()))
//!     .unwrap();
//!
//! // Load weights from file.
//! new_model.load_from_path("model.bin").unwrap();
//!
//! // Compile before using (required for training, optional for prediction).
//! new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
//!
//! // Make predictions with loaded model.
//! let predictions = new_model.predict(&x).unwrap();
//! println!("Predictions shape: {:?}", predictions.shape());
//! ```
//!
//! ## Feature Flags
//!
//! The crate uses feature flags for modular compilation:
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `machine_learning` | Classical ML algorithms |
//! | `neural_network` | Neural network framework |
//! | `utils` | Data preprocessing and dataset splitting |
//! | `metrics` | Evaluation metrics |
//! | `math` | Numerical primitives (distances, matrix products, parallel reductions) |
//! | `full` | Enables all of the above |
//! | `default` | Enables `full` |
//! | `show_progress` | Show progress bars when training |
//!
//! `machine_learning`, `neural_network`, `utils`, and `metrics` each enable `math`.
//!
//! The default enables every category. Features are additive.
//! Set `default-features = false` to select only the features you need.

#[cfg(any(
    feature = "machine_learning",
    feature = "utils",
    feature = "neural_network"
))]
use serde::{Deserialize, Serialize};

#[cfg(feature = "show_progress")]
use indicatif::{ProgressBar, ProgressStyle};

/// Creates a progress bar with the crate's style
///
/// Compiled only when the `show_progress` feature is enabled
///
/// # Parameters
///
/// - `total` - the total number of iterations or items to process
/// - `template` - the template string, using placeholders {elapsed_precise}, {bar:40}, {pos},
///   {len}, {msg}, for example `"[{elapsed_precise}] {bar:40} {pos}/{len} | Cost: {msg}"`
///
/// # Returns
///
/// - `ProgressBar` - a progress bar ready to update
#[cfg(feature = "show_progress")]
#[allow(dead_code)]
fn create_progress_bar(total: u64, template: &str) -> ProgressBar {
    let progress_bar = ProgressBar::new(total);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(template)
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    progress_bar
}

/// Generates a public getter method that returns a field by value
///
/// The generated method includes documentation describing the field being accessed
///
/// # Parameters
///
/// - `$method_name` - the name of the getter method (for example get_fit_intercept)
/// - `$field_name` - the name of the field to access (for example fit_intercept)
/// - `$return_type` - the return type of the getter method
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! get_field {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("- `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field")]
        #[inline]
        pub fn $method_name(&self) -> $return_type {
            self.$field_name
        }
    };
}

/// Generates a public getter method returning an immutable reference to a field
///
/// The generated method includes documentation describing the field being accessed
///
/// # Parameters
///
/// - `$method_name` - the identifier for the generated getter method name
/// - `$field_name` - the identifier of the struct field to access
/// - `$return_type` - the type for the return value (typically a reference type like `&Type`)
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! get_field_as_ref {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("- `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field as a reference")]
        #[inline]
        pub fn $method_name(&self) -> $return_type {
            self.$field_name.as_ref()
        }
    };
}

/// Generates `save_to_path` and `load_from_path` methods for model structs
///
/// - `save_to_path` - saves the model to a binary file at the specified path
/// - `load_from_path` - loads a model from a binary file at the specified path
///
/// # Parameters
///
/// - `$model_type` - the type of the model struct (for example LinearRegression, LogisticRegression)
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! model_save_and_load_methods {
    ($model_type:ty) => {
        /// Saves the trained model to a binary file at the specified path
        ///
        /// Serializes the entire model state, including coefficients, intercept,
        /// hyperparameters, and training metadata, to a compact binary format using postcard
        ///
        /// # Parameters
        ///
        /// - `path` - file path where the model is saved (for example "stored_model.bin")
        ///
        /// # Returns
        ///
        /// - `Result<(), Error>` - success, or an error if saving failed
        ///
        /// # Errors
        ///
        /// - `Error::Io` - file creation or write failed, or serialization failed
        pub fn save_to_path(&self, path: &str) -> Result<(), crate::error::Error> {
            use std::fs::File;
            use std::io::{BufWriter, Write};

            let bytes = postcard::to_allocvec(self)?;

            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            writer.write_all(&bytes)?;

            writer.flush()?;

            Ok(())
        }

        /// Loads a trained model from a binary file at the specified path
        ///
        /// Deserializes a previously saved model from the postcard binary format, restoring all
        /// model parameters, hyperparameters, and training state
        ///
        /// # Parameters
        ///
        /// - `path` - file path from which to load the model (for example "stored_model.bin")
        ///
        /// # Returns
        ///
        /// - `Result<Self, Error>` - the loaded model instance, or an error if loading failed
        ///
        /// # Errors
        ///
        /// - `Error::Io` - file not found or read failed, or deserialization failed (invalid
        ///   format or corrupted data)
        pub fn load_from_path(path: &str) -> Result<Self, crate::error::Error> {
            let bytes = std::fs::read(path)?;

            let model: $model_type = postcard::from_bytes(&bytes)?;

            Ok(model)
        }
    };
}

/// Generates a runtime-tunable `usize` parallelism gate: a private atomic backing store plus a
/// `pub(crate)` getter and setter
///
/// The atomic store starts with the given default and accepts runtime changes.
/// All 3 generated items receive the leading attributes, including feature gates.
/// Reads and writes use relaxed atomic order.
/// The public [`crate::tuning`] module forwards to selected getters and setters.
///
/// # Syntax
///
/// `[attrs..] [vis] STORE => getter / setter = default`. The attributes (docs and any
/// `#[cfg(...)]`) and the visibility are optional. `STORE` is the static's name, and `=>` and
/// `/` are just separators
///
/// # Usage and expansion
///
/// This invocation:
///
/// ```ignore
/// tunable_gate! {
///     /// Docs for the gate
///     #[cfg(feature = "neural_network")]
///     pub(crate) SOME_GATE => some_gate / set_some_gate = 65_536
/// }
/// ```
///
/// expands to:
///
/// ```ignore
/// /// Docs for the gate
/// #[cfg(feature = "neural_network")]
/// static SOME_GATE: AtomicUsize = AtomicUsize::new(65_536);
///
/// /// Docs for the gate
/// #[cfg(feature = "neural_network")]
/// #[doc(hidden)]
/// pub(crate) fn some_gate() -> usize { SOME_GATE.load(Relaxed) }
///
/// /// Docs for the gate
/// #[cfg(feature = "neural_network")]
/// #[doc(hidden)]
/// pub(crate) fn set_some_gate(value: usize) { SOME_GATE.store(value, Relaxed); }
/// ```
///
/// Call sites read the gate through `some_gate()`, never through `SOME_GATE` directly
#[allow(unused_macros)]
macro_rules! tunable_gate {
    (
        $(#[$attr:meta])*
        $vis:vis $store:ident => $get:ident / $set:ident = $default:expr
    ) => {
        $(#[$attr])*
        static $store: ::std::sync::atomic::AtomicUsize =
            ::std::sync::atomic::AtomicUsize::new($default);

        $(#[$attr])*
        #[doc(hidden)]
        #[inline]
        ///
        /// # Returns
        ///
        /// - `usize` - Current gate threshold
        $vis fn $get() -> usize {
            $store.load(::std::sync::atomic::Ordering::Relaxed)
        }

        $(#[$attr])*
        #[doc(hidden)]
        ///
        /// # Parameters
        ///
        /// - `value` - New gate threshold
        $vis fn $set(value: usize) {
            $store.store(value, ::std::sync::atomic::Ordering::Relaxed);
        }
    };
}

/// Runtime overrides for the crate's parallel and serial gate thresholds
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils",
    feature = "metrics",
    feature = "math"
))]
pub mod tuning;

/// The crate's unified error type ([`error::Error`]) and its result alias
/// ([`error::RustymlResult`])
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub mod error;

/// Crate-wide control of pseudo-random number generation for reproducibility
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub mod random;

/// Re-export of the global-seed controls defined in [`random`]
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub use random::{clear_global_seed, set_global_seed};

/// Crate-internal parallel/serial gate thresholds, 1 runtime-tunable gate per calibrated
/// kernel cost class (`f32` classes for the neural-network layers, `f64` classes for ML/utils)
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub(crate) mod parallel_gates;

/// Shared low-level numeric primitives: distance metrics, `gemmkit`-backed matrix products, and
/// deterministic parallel reductions
#[cfg(feature = "math")]
pub mod math;

/// Classical supervised and unsupervised estimators: regression, classification, clustering,
/// dimensionality reduction, and anomaly detection
#[cfg(feature = "machine_learning")]
pub mod machine_learning;

/// Shared fit, predict, and transform traits for estimators and stateful preprocessing tools.
#[cfg(any(feature = "machine_learning", feature = "utils"))]
pub mod traits;

/// Single-import re-export of the crate's most commonly used types, traits, and functions
pub mod prelude;

/// Data preprocessing (normalize, standardize, the stateful scaler family, label encoding)
/// and train/test dataset splitting
#[cfg(feature = "utils")]
pub mod utils;

/// Model-evaluation metrics for regression, classification, and clustering
#[cfg(feature = "metrics")]
pub mod metrics;

/// Neural-network framework: layers, optimizers, loss functions, the sequential model, and the
/// graph model
#[cfg(feature = "neural_network")]
pub mod neural_network;

/// Internal hooks for the `benches/` targets. Not part of the public API, with no stability
/// guarantee
#[cfg(any(feature = "machine_learning", feature = "neural_network"))]
#[doc(hidden)]
pub mod bench_internals;

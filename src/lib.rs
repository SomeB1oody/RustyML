//! # RustyML: Machine Learning and Deep Learning in Pure Rust
//!
//! RustyML is a machine learning and deep learning library written entirely in Rust. It
//! implements classical ML algorithms, neural networks, and data-processing utilities
//!
//! ## Overview
//!
//! The crate covers a full workflow: preprocessing, feature engineering, model training, and
//! evaluation. It validates input and reports errors
//!
//! Estimator defaults, score orientations, and metric conventions follow scikit-learn and are
//! checked numerically against it, so a ported pipeline produces the same numbers. Where the
//! crate departs from scikit-learn, the item's own documentation says so. Known departures:
//!
//! - `metrics` panics instead of returning `Result`
//! - `roc_curve` always returns the full threshold sweep
//! - `MeanShift` has an opt-in Gaussian kernel
//!
//! ## Architecture
//!
//! The crate splits into 5 modules. A feature flag gates each one:
//!
//! ### [`machine_learning`]
//! Classical machine learning algorithms for supervised and unsupervised learning:
//! - **Regression**: Linear Regression with L1/L2 regularization, solved in closed form by default
//!   or by gradient descent
//! - **Classification**: Logistic Regression, KNN, Decision Tree, SVC, Linear SVC, LDA
//! - **Clustering**: KMeans, DBSCAN, MeanShift. All 3 label samples as `Array1<isize>`, with
//!   `-1` for noise or unassigned points
//! - **Dimensionality Reduction**: PCA, Kernel PCA, t-SNE
//! - **Anomaly Detection**: Isolation Forest, scoring in `[-1, 0)` where lower is more anomalous
//!   and predicting `-1` (outlier) / `+1` (inlier)
//!
//! ### [`neural_network`]
//! Neural network framework built around a sequential model. Tensors are channels-last, and
//! kernel shapes match Keras, so a layout carried over from Keras needs no permutation:
//! - **Layers**: Dense, SimpleRNN, LSTM, GRU, Convolution, Pooling, Dropout
//! - **Optimizers**: SGD, Adam, AdamW, RMSprop, AdaGrad
//! - **Loss Functions**: MSE, MAE, Binary/Categorical/Sparse Categorical Cross-Entropy
//! - **Models**: Sequential architecture for feed-forward networks. `fit` and `fit_with_batches`
//!   return a `History` of one loss per epoch. A hand-written loop can call the public
//!   `train_batch` instead, and `evaluate` scores the model without training it
//!
//! ### [`utils`]
//! Data preprocessing and dataset-splitting utilities:
//! - **Preprocessing**: one-shot `standardize` / `normalize`, the stateful scaler family
//!   (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`) for reusing
//!   training statistics on later batches, and label encoding
//! - **Dataset Splitting**: train/test split (optionally stratified)
//!
//! ### [`metrics`]
//! Evaluation metrics for model performance. Unlike the rest of the crate, these functions panic
//! on a precondition violation instead of returning a `Result`, which keeps this leaf module
//! dependency-light:
//! - **Regression**: MSE, RMSE, MAE, R^2 score
//! - **Classification**: Accuracy, Confusion Matrix, AUC-ROC, F1-score
//! - **Clustering**: Adjusted Rand Index, Normalized/Adjusted Mutual Information, Silhouette
//!   Score. Every one of them takes `isize` labels, the type the clustering estimators return
//!
//! ### [`math`]
//! Low-level numeric primitives shared across modules:
//! - **Distance Metrics**: Euclidean, Manhattan, Minkowski, plus the `DistanceCalculationMetric`
//!   dispatcher
//! - **Matrix Products**: `gemmkit`-backed GEMM/GEMV with automatic parallelism
//! - **Reductions**: deterministic blocked parallel reductions
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
//! # list the features you need, e.g. `features = ["machine_learning"]`
//! # Add `"show_progress"` to show progress bars during training
//! ```
//!
//! In your Rust code, write:
//! ```rust,no_run
//! use rustyml::machine_learning::LinearRegression;
//! use rustyml::machine_learning::linear_model::LeastSquaresSolver;
//! use ndarray::{Array1, Array2};
//!
//! // Create a linear regression model
//! let mut model = LinearRegression::new(true).with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 }).unwrap();
//!
//! // Prepare training data
//! let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
//! let raw_y = vec![6.0, 9.0, 12.0];
//!
//! // Convert Vec to ndarray types
//! let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
//! let y = Array1::from_vec(raw_y);
//!
//! // Train the model
//! model.fit(&x.view(), &y.view()).unwrap();
//!
//! // Make predictions
//! let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
//! let _predictions = model.predict(&new_data.view());
//!
//! // Save the trained model to a file
//! model.save_to_path("linear_regression_model.bin").unwrap();
//!
//! // Load the model from the file
//! let loaded_model = LinearRegression::load_from_path("linear_regression_model.bin").unwrap();
//!
//! // Use the loaded model for predictions
//! let _loaded_predictions = loaded_model.predict(&new_data.view());
//!
//! // Clone is implemented
//! let _model_copy = model.clone();
//!
//! // Debug is implemented
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
//! # list the features you need, e.g. `features = ["neural_network"]`
//! # Add `"show_progress"` to show progress bars during training
//! ```
//!
//! In your Rust code, write:
//! ```rust,no_run
//! use rustyml::neural_network::{
//!     sequential::Sequential,
//!     layers::{Activation, Dense},
//!     optimizers::Adam,
//!     losses::CategoricalCrossEntropy,
//! };
//! use ndarray::Array;
//!
//! // Create training data
//! let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features
//! let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
//!
//! // Build a neural network
//! let mut model = Sequential::new();
//! model
//!     .add(Dense::new(784, 128, Activation::ReLU).unwrap())
//!     .add(Dense::new(128, 64, Activation::ReLU).unwrap())
//!     .add(Dense::new(64, 10, Activation::Softmax).unwrap())
//!     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
//!
//! // Display model structure
//! model.summary();
//!
//! // Train the model
//! // The returned History holds one loss per epoch, measured during that epoch, not after it
//! let history = model.fit(&x, &y, 10).unwrap();
//! println!("Per-epoch loss: {:?}", history.loss());
//!
//! // Score the model's current weights, an inference-mode pass that changes nothing
//! println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());
//!
//! // Save model weights to file
//! model.save_to_path("model.bin").unwrap();
//!
//! // Create a new model with the same architecture
//! let mut new_model = Sequential::new();
//! new_model
//!     .add(Dense::new(784, 128, Activation::ReLU).unwrap())
//!     .add(Dense::new(128, 64, Activation::ReLU).unwrap())
//!     .add(Dense::new(64, 10, Activation::Softmax).unwrap());
//!
//! // Load weights from file
//! new_model.load_from_path("model.bin").unwrap();
//!
//! // Compile before using (required for training, optional for prediction)
//! new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), CategoricalCrossEntropy::new(false));
//!
//! // Make predictions with loaded model
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
//! The default enables everything. A scikit-learn workflow reaches across modules
//! (`utils::train_test_split` -> `machine_learning` -> `metrics`), so a fresh `cargo add rustyml`
//! should have all of it. Features are additive. Naming one does not turn the rest off, so to
//! restrict a build, set `default-features = false` and list what you need.

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
/// - `$method_name` - The name of the getter method (e.g. get_fit_intercept)
/// - `$field_name` - The name of the field to access (e.g. fit_intercept)
/// - `$return_type` - The return type of the getter method
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! get_field {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("* `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field")]
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
/// - `$method_name` - The identifier for the generated getter method name
/// - `$field_name` - The identifier of the struct field to access
/// - `$return_type` - The type for the return value (typically a reference type like `&Type`)
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! get_field_as_ref {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("* `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field as a reference")]
        #[inline]
        pub fn $method_name(&self) -> $return_type {
            self.$field_name.as_ref()
        }
    };
}

/// Generates `save_to_path` and `load_from_path` methods for model structs
///
/// - `save_to_path` - Saves the model to a binary file at the specified path
/// - `load_from_path` - Loads a model from a binary file at the specified path
///
/// # Parameters
///
/// - `$model_type` - The type of the model struct (e.g. LinearRegression, LogisticRegression)
#[cfg(any(feature = "machine_learning", feature = "utils"))]
macro_rules! model_save_and_load_methods {
    ($model_type:ty) => {
        /// Saves the trained model to a binary file at the specified path
        ///
        /// Serializes the entire model state including coefficients, intercept,
        /// hyperparameters, and training metadata to a compact binary format using postcard
        ///
        /// # Parameters
        ///
        /// - `path` - File path where the model will be saved (e.g. "stored_model.bin")
        ///
        /// # Returns
        ///
        /// - `Ok(())` - Model successfully saved to file
        /// - `Err(Error::Io)` - File creation/write failed, or serialization failed
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
        /// - `path` - File path from which to load the model (e.g. "stored_model.bin")
        ///
        /// # Returns
        ///
        /// - `Ok(Self)` - Successfully loaded model instance
        /// - `Err(Error::Io)` - File not found/read failed, or deserialization failed
        ///   (invalid format or corrupted data)
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
/// Replaces a `const` gate threshold with an [`AtomicUsize`](std::sync::atomic::AtomicUsize)
/// initialized to the same default, so the value can be overridden at runtime without a
/// recompile. Every leading attribute (`#[cfg(...)]`, doc comments) is applied to all 3
/// generated items, so a gate keeps the exact feature-gating of the constant it replaces. The
/// hot-path read is a single `Relaxed` load. The gate only selects a strategy and never changes a
/// result, so no stronger ordering is needed. The public [`crate::tuning`] facade wraps the
/// generated setters into one discoverable surface
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
/// Call sites read the gate through `some_gate()` instead of the old `SOME_GATE` constant
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
        $vis fn $get() -> usize {
            $store.load(::std::sync::atomic::Ordering::Relaxed)
        }

        $(#[$attr])*
        #[doc(hidden)]
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

/// Crate-internal parallel/serial gate thresholds, one constant per calibrated kernel cost
/// class (`f32` classes for the neural-network layers, `f64` classes for ML/utils)
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

/// Every model and stateful transformer in the crate implements the shared estimator contract
/// (`Fit`, `Predict`, `Transform`, `FitTransform`)
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

/// Neural-network framework: layers, optimizers, loss functions, and the sequential model
#[cfg(feature = "neural_network")]
pub mod neural_network;

/// Internal hooks for the `benches/` targets. Not part of the public API, with no stability
/// guarantee
#[cfg(any(feature = "machine_learning", feature = "neural_network"))]
#[doc(hidden)]
pub mod bench_internals;

//! Utilities for preprocessing and dataset splitting
//!
//! Preprocessing covers normalization, standardization, and label encoding. Dataset splitting
//! covers train-test partitioning.
//!
//! # Preprocessing
//! - **normalize**: scale samples to unit norm along a chosen axis (L1, L2, or max order)
//! - **standardize**: z-score standardization (zero mean, unit variance) for feature scaling
//! - **scalers**: the stateful form of the above. Fit on the training matrix, then reuse those
//!   statistics for every later batch. The set is `StandardScaler` (z-score), `MinMaxScaler`
//!   (bounded range), `MaxAbsScaler` (magnitude, keeping zeros and signs), `RobustScaler` (median
//!   and IQR, resists outliers), and `Normalizer` (per-sample norm)
//! - **label encoding**: convert between dense labels and one-hot or sparse categorical formats
//!
//! # Dataset splitting
//! - **train_test_split**: split into train and test sets with a configurable ratio, optionally
//!   stratified
//!
//! # Key features
//! - **Parallel processing**: uses rayon for parallel computation above a size gate
//! - **Input validation**: reports descriptive errors on malformed input
//!
//! # Examples
//!
//! ```rust
//! use rustyml::utils::standardize::{standardize, StandardizationAxis};
//! use rustyml::utils::StandardScaler;
//! use ndarray::array;
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! // One-shot, stateless: the statistics come from `x` itself
//! let standardized = standardize(&x, StandardizationAxis::Column).unwrap();
//! assert_eq!(standardized.dim(), (3, 2));
//!
//! // Stateful: fit on the training matrix, then scale later batches by those numbers
//! let mut scaler = StandardScaler::new();
//! let x_train_scaled = scaler.fit_transform(&x).unwrap();
//! let x_new_scaled = scaler.transform(&array![[7.0, 8.0]]).unwrap();
//! assert_eq!(x_train_scaled, standardized);
//! assert_eq!(x_new_scaled.dim(), (1, 2));
//! ```

/// Conversion between label formats
pub mod label_encoding;
/// Normalize data along a specified axis using the given norm order
pub mod normalize;
/// Stateful feature scalers that store their training statistics
pub mod scaler;
/// Standardize data to have zero mean and unit variance
pub mod standardize;
/// Split datasets into training and test sets
pub mod train_test_split;

/// The crate-wide estimator traits, re-exported here for convenience. Their canonical
/// home is [`crate::traits`]
pub use crate::traits::{Fit, FitTransform, Predict, Transform};
pub use label_encoding::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};
pub use normalize::{NormalizationAxis, NormalizationOrder, normalize};
pub use scaler::{MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler};
pub use standardize::{StandardizationAxis, standardize};
pub use train_test_split::{TrainTestSplit, train_test_split, train_test_split_stratified};

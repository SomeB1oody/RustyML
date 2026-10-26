//! Data preprocessing and dataset splits.
//!
//! The submodules provide label encoding, array normalization, standardization, and train-test splits.
//! Stateful scalers store training statistics for later data batches.
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
//! // Stateless normalization: the statistics come from `x` itself
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
/// Standardize data to have mean 0 and variance 1
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

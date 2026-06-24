//! Utilities for preprocessing and dataset splitting
//!
//! Preprocessing covers normalization, standardization, and label encoding; dataset
//! splitting covers train/test partitioning
//!
//! # Preprocessing
//! - **normalize**: scale samples to unit norm along a chosen axis (L1 / L2 / max order)
//! - **standardize**: z-score standardization (zero mean, unit variance) for feature scaling
//! - **label encoding**: convert between dense labels and one-hot / sparse categorical formats
//!
//! # Dataset splitting
//! - **train_test_split**: split into train/test sets with a configurable ratio, optionally stratified
//!
//! # Key features
//! - **Parallel processing**: rayon-based parallel computation
//! - **Input validation**: descriptive errors on malformed input
//!
//! # Examples
//!
//! ```rust
//! use rustyml::utils::standardize::{standardize, StandardizationAxis};
//! use ndarray::array;
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let standardized = standardize(&x, StandardizationAxis::Column).unwrap();
//! assert_eq!(standardized.dim(), (3, 2));
//! ```

/// Conversion between label formats
pub mod label_encoding;
/// Normalize data along a specified axis using the given norm order
pub mod normalize;
/// Standardize data to have zero mean and unit variance
pub mod standardize;
/// Split datasets into training and test sets
pub mod train_test_split;

pub use label_encoding::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};
pub use normalize::{NormalizationAxis, NormalizationOrder, normalize};
pub use standardize::{StandardizationAxis, standardize};
pub use train_test_split::{TrainTestSplit, train_test_split, train_test_split_stratified};

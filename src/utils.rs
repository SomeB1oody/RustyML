//! Utilities for preprocessing and dataset splitting
//!
//! Preprocessing covers normalization, standardization, and label encoding; dataset
//! splitting covers train/test partitioning.

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

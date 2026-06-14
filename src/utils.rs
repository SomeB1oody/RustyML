//! Utilities for preprocessing, dimensionality reduction, and dataset splitting
//!
//! Preprocessing covers normalization, standardization, and label encoding;
//! dimensionality reduction covers PCA, kernel PCA, and t-SNE; dataset splitting
//! covers train/test partitioning

// `KernelType` lives in the crate-level `types` module (shared with SVC); re-exported here so it is reachable as `utils::KernelType` alongside the kernel-PCA API
pub use crate::types::KernelType;

/// Kernel Principal Component Analysis
pub mod kernel_pca;
/// Conversion between label formats
pub mod label_encoding;
/// Normalize data along a specified axis using the given norm order
pub mod normalize;
/// Principal Component Analysis (PCA)
pub mod pca;
/// Standardize data to have zero mean and unit variance
pub mod standardize;
/// t-SNE algorithm for dimensionality reduction
pub mod t_sne;
/// Split datasets into training and test sets
pub mod train_test_split;

/// Internal shared linear-algebra helpers (power iteration) used by the transformers
mod linalg;
/// Internal shared input-validation helpers used across the transformers and preprocessors
mod validation;

// Explicit re-exports keep the flat `utils::` API surface stable, matching the `machine_learning` convention: a new `pub` item in a submodule no longer silently changes this namespace
pub use kernel_pca::{EigenSolver, KernelPCA};
pub use label_encoding::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};
pub use normalize::{NormalizationAxis, NormalizationOrder, normalize};
pub use pca::{PCA, SVDSolver};
pub use standardize::{StandardizationAxis, standardize};
pub use t_sne::{Init, TSNE, TSNEMethod};
pub use train_test_split::{TrainTestSplit, train_test_split, train_test_split_stratified};

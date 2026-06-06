// `KernelType` lives in the crate-level `types` module (shared with SVC); re-exported
// here so it is reachable as `utility::KernelType` alongside the kernel-PCA API.
pub use crate::types::KernelType;

/// This module implements Kernel Principal Component Analysis.
pub mod kernel_pca;
/// This module provides functions for converting between different label formats
pub mod label_encoding;
/// This module provides a method to Normalize data along specified axis using the given norm order
pub mod normalize;
/// This module provides an implementation of Principal Component Analysis (PCA)
pub mod principal_component_analysis;
/// This module provides a method to standardizes data to have zero mean and unit variance
pub mod standardize;
/// This module provides an implementation of the t-SNE algorithm for dimensionality reduction.
pub mod t_sne;
/// This module provides functionality for splitting datasets into training and test sets
pub mod train_test_split;

/// Internal shared linear-algebra helpers (power iteration) used by the transformers
mod linalg;
/// Internal shared input-validation helpers used by every transformer/preprocessor
mod validation;

// Explicit re-exports keep the flat `utility::` API surface intentional and stable,
// matching the `machine_learning` module's convention: adding a `pub` item to a
// submodule no longer silently changes this namespace.
pub use kernel_pca::{EigenSolver, KernelPCA};
pub use label_encoding::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};
pub use normalize::{NormalizationAxis, NormalizationOrder, normalize};
pub use principal_component_analysis::{PCA, SVDSolver};
pub use standardize::{StandardizationAxis, standardize};
pub use t_sne::TSNE;
pub use train_test_split::{TrainTestSplit, train_test_split};

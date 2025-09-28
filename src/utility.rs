use crate::ModelError;
use ahash::{AHashMap, AHashSet};
use ndarray::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

/// Kernel function types for Support Vector Machine
///
/// # Variants
/// - `Linear` - Linear kernel: K(x, y) = x·y
/// - `Poly` - Polynomial kernel: K(x, y) = (gamma·x·y + coef0)^degree
/// - `RBF` - Radial Basis Function kernel: K(x, y) = exp(-gamma·|x-y|^2)
/// - `Sigmoid` - Sigmoid kernel: K(x, y) = tanh(gamma·x·y + coef0)
#[derive(Debug, Clone)]
pub enum KernelType {
    Linear,
    Poly { degree: u32, gamma: f64, coef0: f64 },
    RBF { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
}

/// This module provides an implementation of Principal Component Analysis (PCA),
/// a dimensionality reduction technique that transforms high-dimensional data
/// into a lower-dimensional space while preserving maximum variance
pub mod principal_component_analysis;

/// This module provides functionality for splitting datasets into training and test sets,
/// which is a fundamental preprocessing step in machine learning workflows
pub mod train_test_split;

/// This module implements Kernel Principal Component Analysis, a non-linear dimensionality
/// reduction technique that uses kernel methods to extend PCA to non-linear transformations
pub mod kernel_pca;

/// This module provides an implementation of Linear Discriminant Analysis, a supervised
/// dimensionality reduction technique that finds a linear combination of features that
/// characterizes or separates two or more classes
pub mod linear_discriminant_analysis;

/// This module provides an implementation of the t-SNE algorithm for dimensionality reduction.
/// t-SNE is particularly well-suited for visualizing high-dimensional data in 2D or 3D spaces
/// while preserving local structures in the data
pub mod t_sne;

/// This module provides functions for converting between different label formats
/// commonly used in machine learning, particularly for classification tasks.
/// It supports conversions between sparse categorical (integer) labels and
/// categorical (one-hot encoded) formats, with optimized performance using AHashMap.
pub mod label_encoding;

/// This module provides a method to standardizes data to have zero mean and unit variance
pub mod standardize;

pub mod normalize;

pub use kernel_pca::*;
pub use label_encoding::*;
pub use linear_discriminant_analysis::*;
pub use normalize::*;
pub use principal_component_analysis::*;
pub use standardize::*;
pub use t_sne::*;
pub use train_test_split::*;

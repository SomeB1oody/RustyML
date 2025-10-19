use crate::error::ModelError;
use ahash::{AHashMap, AHashSet};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Data;
use ndarray::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Kernel function types for Support Vector Machine
///
/// # Variants
/// - `Linear` - Linear kernel: K(x, y) = x·y
/// - `Poly` - Polynomial kernel: K(x, y) = (gamma·x·y + coef0)^degree
/// - `RBF` - Radial Basis Function kernel: K(x, y) = exp(-gamma·|x-y|^2)
/// - `Sigmoid` - Sigmoid kernel: K(x, y) = tanh(gamma·x·y + coef0)
#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum KernelType {
    Linear,
    Poly { degree: u32, gamma: f64, coef0: f64 },
    RBF { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
}

/// This module provides an implementation of Principal Component Analysis (PCA)
pub mod principal_component_analysis;

/// This module provides functionality for splitting datasets into training and test sets
pub mod train_test_split;

/// This module implements Kernel Principal Component Analysis.
pub mod kernel_pca;

/// This module provides an implementation of Linear Discriminant Analysis.
pub mod linear_discriminant_analysis;

/// This module provides an implementation of the t-SNE algorithm for dimensionality reduction.
pub mod t_sne;

/// This module provides functions for converting between different label formats
pub mod label_encoding;

/// This module provides a method to Normalize data along specified axis using the given norm order
pub mod normalize;

/// This module provides a method to standardizes data to have zero mean and unit variance
pub mod standardize;

pub use kernel_pca::*;
pub use label_encoding::*;
pub use linear_discriminant_analysis::*;
pub use normalize::*;
pub use principal_component_analysis::*;
pub use standardize::*;
pub use t_sne::*;
pub use train_test_split::*;

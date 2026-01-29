use crate::error::ModelError;
use crate::{Deserialize, Serialize};
use ahash::{AHashMap, AHashSet};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Data;
use ndarray::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

/// This module implements Kernel Principal Component Analysis.
pub mod kernel_pca;
/// This module provides functions for converting between different label formats
pub mod label_encoding;
/// This module provides an implementation of Linear Discriminant Analysis.
pub mod linear_discriminant_analysis;
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

pub use kernel_pca::*;
pub use label_encoding::*;
pub use linear_discriminant_analysis::*;
pub use normalize::*;
pub use principal_component_analysis::*;
pub use standardize::*;
pub use t_sne::*;
pub use train_test_split::*;

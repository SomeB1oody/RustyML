use ndarray::{Array1, Array2};

/// Standardizes data to have zero mean and unit variance
///
/// This function transforms input data by subtracting the mean and dividing
/// by the standard deviation for each feature, resulting in standardized data
/// where each feature has a mean of 0 and a standard deviation of 1.
///
/// # Parameters
///
/// * `x` - A 2D array where rows represent samples and columns represent features
///
/// # Returns
///
/// * `Array2<f64>` - A standardized 2D array with the same shape as the input
///
/// # Implementation Details
///
/// - Calculates mean and standard deviation for each feature column
/// - Handles cases where standard deviation is zero (or very small) by setting it to 1.0
/// - Applies the z-score transformation: (x - mean) / std_dev
pub fn standardize(x: &Array2<f64>) -> Array2<f64> {
    use crate::math::standard_deviation;
    use rayon::prelude::*;

    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Use parallel iteration to calculate mean and standard deviation for each column
    let feature_indices: Vec<usize> = (0..n_features).collect();
    let stats: Vec<(f64, f64)> = feature_indices
        .par_iter()
        .map(|&i| {
            let col = x.column(i);
            let mean = col.mean().unwrap_or(0.0);
            let std = standard_deviation(col).unwrap();
            // Handle cases where standard deviation is zero
            let std = if std < 1e-10 { 1.0 } else { std };
            (mean, std)
        })
        .collect();

    // Extract mean and standard deviation arrays
    let means = Array1::from_iter(stats.iter().map(|&(mean, _)| mean));
    let stds = Array1::from_iter(stats.iter().map(|&(_, std)| std));

    // Parallelize data standardization
    let mut x_std = Array2::<f64>::zeros((n_samples, n_features));

    // Method 1: Process rows in parallel
    let row_indices: Vec<usize> = (0..n_samples).collect();
    let results: Vec<_> = row_indices
        .par_iter()
        .map(|&i| {
            let mut row = Vec::with_capacity(n_features);
            for j in 0..n_features {
                row.push((x[[i, j]] - means[j]) / stds[j]);
            }
            (i, row)
        })
        .collect();

    // Collect results
    for (i, row) in results {
        for j in 0..n_features {
            x_std[[i, j]] = row[j];
        }
    }

    x_std
}

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

pub use kernel_pca::*;
pub use linear_discriminant_analysis::*;
pub use principal_component_analysis::*;
pub use t_sne::*;
pub use train_test_split::*;

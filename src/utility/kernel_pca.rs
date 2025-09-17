use crate::ModelError;
pub use crate::machine_learning::KernelType;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// A Kernel Principal Component Analysis implementation.
///
/// KernelPCA performs a non-linear dimensionality reduction by using
/// kernel methods to map the data into a higher dimensional space where
/// linear PCA is performed.
///
/// # Fields
///
/// - `kernel` - The kernel function type used to compute the kernel matrix
/// - `n_components` - The number of components to extract
/// - `eigenvalues` - Eigenvalues in descending order (available after fitting)
/// - `eigenvectors` - Corresponding normalized eigenvectors (available after fitting)
/// - `x_fit` - Training data, used for subsequent transformation of new data
/// - `row_means` - Mean of each row in the training kernel matrix (used for centering)
/// - `total_mean` - Overall mean of the training kernel matrix
///
/// # Example
/// ```rust
/// use ndarray::{Array2, arr2};
/// use rustyml::utility::kernel_pca::{KernelPCA, KernelType};
///
/// // Create some sample data
/// let data = arr2(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0],
/// ]);
///
/// // Create a KernelPCA model with RBF kernel and 2 components
/// let mut kpca = KernelPCA::new(
///     KernelType::RBF { gamma: 0.1 },
///     2
/// );
///
/// // Fit the model and transform the data
/// let transformed = kpca.fit_transform(data.view()).unwrap();
///
/// // The transformed data now has 2 columns instead of 3
/// assert_eq!(transformed.ncols(), 2);
/// assert_eq!(transformed.nrows(), 4);
///
/// // We can also transform new data
/// let new_data = arr2(&[
///     [2.0, 3.0, 4.0],
///     [5.0, 6.0, 7.0],
/// ]);
/// let new_transformed = kpca.transform(new_data.view()).unwrap();
/// assert_eq!(new_transformed.ncols(), 2);
/// assert_eq!(new_transformed.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KernelPCA {
    kernel: KernelType,
    n_components: usize,
    eigenvalues: Option<Array1<f64>>,
    eigenvectors: Option<Array2<f64>>,
    x_fit: Option<Array2<f64>>,
    row_means: Option<Array1<f64>>,
    total_mean: Option<f64>,
}

/// Calculates the kernel value between two samples based on the specified kernel type.
///
/// This function computes the kernel function value between two feature vectors
/// according to the provided kernel type.
///
/// # Parameters
///
/// - `x` - First feature vector
/// - `y` - Second feature vector
/// - `kernel` - Kernel type configuration
///
/// # Returns
///
/// * `f64` - The computed kernel value as a floating-point number
pub fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, kernel: &KernelType) -> f64 {
    match kernel {
        KernelType::Linear => x.dot(y),
        KernelType::Poly {
            degree,
            gamma,
            coef0,
        } => (gamma * x.dot(y) + coef0).powi(*degree as i32),
        KernelType::RBF { gamma } => {
            let diff = x - y;
            let norm_sq = diff.dot(&diff);
            (-gamma * norm_sq).exp()
        }
        KernelType::Sigmoid { gamma, coef0 } => (gamma * x.dot(y) + coef0).tanh(),
    }
}

/// Default implementation for KernelPCA.
impl Default for KernelPCA {
    fn default() -> Self {
        KernelPCA {
            kernel: KernelType::Linear,
            n_components: 2,
            eigenvalues: None,
            eigenvectors: None,
            x_fit: None,
            row_means: None,
            total_mean: None,
        }
    }
}

impl KernelPCA {
    /// Creates a new KernelPCA model with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `kernel` - The kernel function type to use
    /// - `n_components` - The number of components to extract
    ///
    /// # Returns
    ///
    /// * `Self` - A new KernelPCA instance
    pub fn new(kernel: KernelType, n_components: usize) -> Self {
        KernelPCA {
            kernel,
            n_components,
            eigenvalues: None,
            eigenvectors: None,
            x_fit: None,
            row_means: None,
            total_mean: None,
        }
    }

    /// Gets the kernel type used by this KernelPCA instance.
    ///
    /// # Returns
    ///
    /// * `&KernelType` - A reference to the kernel type
    pub fn get_kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Gets the number of components this KernelPCA instance extracts.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of components
    pub fn get_n_components(&self) -> usize {
        self.n_components
    }

    /// Gets the eigenvalues computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - containing a reference to the eigenvalues if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_eigenvalues(&self) -> Result<&Array1<f64>, ModelError> {
        match self.eigenvalues.as_ref() {
            Some(eigenvalues) => Ok(eigenvalues),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the eigenvectors computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - containing a reference to the eigenvectors if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_eigenvectors(&self) -> Result<&Array2<f64>, ModelError> {
        match self.eigenvectors.as_ref() {
            Some(eigenvectors) => Ok(eigenvectors),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the training data saved during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - containing a reference to the training data if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_x_fit(&self) -> Result<&Array2<f64>, ModelError> {
        match self.x_fit.as_ref() {
            Some(x_fit) => Ok(x_fit),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the row means computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - containing a reference to the row means if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_row_means(&self) -> Result<&Array1<f64>, ModelError> {
        match self.row_means.as_ref() {
            Some(row_means) => Ok(row_means),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the total mean computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - containing the total mean if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_total_mean(&self) -> Result<f64, ModelError> {
        match self.total_mean {
            Some(total_mean) => Ok(total_mean),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Fits the KernelPCA model to the input data.
    ///
    /// This method computes the kernel matrix, centers it, and performs eigendecomposition
    /// to extract the principal components.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - containing a mutable reference to the fitted model
    /// - `Err(Box<dyn std::error::Error>)` - if there are validation errors or computation errors
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<&mut Self, Box<dyn std::error::Error>> {
        if x.is_empty() {
            return Err(Box::new(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            )));
        }

        if self.n_components <= 0 {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "n_components={} must be greater than 0",
                self.n_components
            ))));
        }

        if x.nrows() < self.n_components {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "n_components={} must be less than the number of samples={}",
                self.n_components,
                x.nrows()
            ))));
        }

        let n_samples = x.nrows();
        // Save training data
        self.x_fit = Some(x.to_owned());

        // Calculate kernel matrix: k_matrix[i, j] = kernel(x[i], x[j])
        let mut k_matrix = Array2::<f64>::zeros((n_samples, n_samples));
        let kernel_vals: Vec<((usize, usize), f64)> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                for j in i..n_samples {
                    let k_val = compute_kernel(&x.row(i), &x.row(j), &self.kernel);
                    local_results.push(((i, j), k_val));
                }
                local_results
            })
            .collect();

        for ((i, j), k_val) in kernel_vals {
            k_matrix[[i, j]] = k_val;
            k_matrix[[j, i]] = k_val;
        }

        // Calculate mean for each row and the overall mean
        let row_means = k_matrix.mean_axis(Axis(1)).unwrap();
        let total_mean = k_matrix.mean().unwrap();
        self.row_means = Some(row_means.clone());
        self.total_mean = Some(total_mean);

        // Center the kernel matrix: k_centered[i,j] = k_matrix[i,j] - row_means[i] - row_means[j] + total_mean
        let mut k_centered = k_matrix.clone();
        let centered_vals: Vec<((usize, usize), f64)> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                for j in 0..n_samples {
                    let centered_val = k_matrix[[i, j]] - row_means[i] - row_means[j] + total_mean;
                    local_results.push(((i, j), centered_val));
                }
                local_results
            })
            .collect();

        for ((i, j), centered_val) in centered_vals {
            k_centered[[i, j]] = centered_val;
        }

        // Perform eigenvalue decomposition on the centered kernel matrix
        let k_centered_slice = k_centered
            .as_slice()
            .ok_or("Failed to convert k_centered to slice")?;
        let k_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_samples, k_centered_slice);
        let sym_eigen = nalgebra::SymmetricEigen::new(k_mat);

        // Convert eigenvalues and eigenvectors from nalgebra into Vec and create (eigenvalue, eigenvector) pairs
        let mut eig_pairs: Vec<(f64, Vec<f64>)> = (0..n_samples)
            .map(|i| {
                let eigenvalue = sym_eigen.eigenvalues[i];
                let eigenvector = sym_eigen
                    .eigenvectors
                    .column(i)
                    .iter()
                    .cloned()
                    .collect::<Vec<f64>>();
                (eigenvalue, eigenvector)
            })
            .collect();

        // Sort (eigenvalue, eigenvector) pairs by eigenvalue in descending order
        eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select the top n_components eigenvalues and corresponding eigenvectors, and normalize the eigenvectors
        let mut selected_eigenvalues = Array1::<f64>::zeros(self.n_components);
        let mut selected_eigenvectors = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..self.n_components {
            let (val, ref vec) = eig_pairs[i];
            selected_eigenvalues[i] = val;
            // When eigenvalue is large enough, normalize: normalized_vec = eigenvector / sqrt(eigenvalue)
            let norm_factor = if val > 1e-10 { val.sqrt() } else { 1.0 };
            let normalized_vec: Vec<f64> = vec.iter().map(|&v| v / norm_factor).collect();
            for j in 0..n_samples {
                selected_eigenvectors[[j, i]] = normalized_vec[j];
            }
        }

        self.eigenvalues = Some(selected_eigenvalues);
        self.eigenvectors = Some(selected_eigenvectors);

        Ok(self)
    }

    /// Transforms new data using the fitted KernelPCA model.
    ///
    /// Projects new data points into the principal component space.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` containing the transformed data (projection of the input data)
    /// - `Err(Box<dyn std::error::Error>)` if the model hasn't been fitted or computation errors occur
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        // Check if model is fitted
        let (x_fit, eigenvectors, row_means, total_mean) = match (
            &self.x_fit,
            &self.eigenvectors,
            &self.row_means,
            &self.total_mean,
        ) {
            (Some(x_fit), Some(eigenvectors), Some(row_means), Some(total_mean)) => {
                (x_fit, eigenvectors, row_means, *total_mean)
            }
            _ => return Err(Box::new(ModelError::NotFitted)),
        };

        if x.is_empty() {
            return Err(Box::new(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            )));
        }

        if x.ncols() != x_fit.ncols() {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "Number of features in new data ({}) doesn't match training data ({})",
                x.ncols(),
                x_fit.ncols()
            ))));
        }

        let n_train = x_fit.nrows();
        let n_new = x.nrows();

        // Pre-allocate kernel matrix
        let mut k_new = Array2::<f64>::zeros((n_train, n_new));

        // Calculate kernel values in parallel with better memory access pattern
        let kernel_values: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|j| {
                let x_j = x.row(j);
                let mut column_values = Vec::with_capacity(n_train);
                for i in 0..n_train {
                    let k_val = compute_kernel(&x_fit.row(i), &x_j, &self.kernel);
                    column_values.push(k_val);
                }
                column_values
            })
            .collect();

        // Fill the kernel matrix from computed values
        for (j, column_values) in kernel_values.iter().enumerate() {
            for (i, &k_val) in column_values.iter().enumerate() {
                k_new[[i, j]] = k_val;
            }
        }

        // Precompute column means efficiently
        let column_means: Vec<f64> = (0..n_new)
            .into_par_iter()
            .map(|j| k_new.column(j).sum() / n_train as f64)
            .collect();

        // Center the kernel matrix in-place
        k_new.indexed_iter_mut().for_each(|((i, j), val)| {
            *val = *val - row_means[i] - column_means[j] + total_mean;
        });

        // Project using eigenvectors: result = eigenvectors^T * k_new_centered
        let transformed = eigenvectors.t().dot(&k_new);
        Ok(transformed.t().to_owned())
    }

    /// Fits the model to the data and then transforms it.
    ///
    /// This is a convenience method that calls fit() followed by transform()
    /// on the same data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - containing the transformed data (projection of the input data)
    /// - `Err(Box<dyn std::error::Error>)` - if there are validation errors or computation errors
    pub fn fit_transform(
        &mut self,
        x: ArrayView2<f64>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        self.fit(x)?;
        match self.transform(x) {
            Ok(transformed) => Ok(transformed),
            Err(err) => Err(err),
        }
    }
}

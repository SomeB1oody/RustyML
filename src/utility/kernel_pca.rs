pub use super::KernelType;
use super::*;

/// Threshold for switching between sequential and parallel computation.
/// When the number of samples is less than this threshold, sequential computation is used.
/// Otherwise, parallel computation is used to improve performance on large datasets.
const KERNEL_PCA_PARALLEL_THRESHOLD: usize = 100;

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
/// ).unwrap();
///
/// // Fit the model and transform the data
/// let transformed = kpca.fit_transform(&data).unwrap();
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
/// let new_transformed = kpca.transform(&new_data).unwrap();
/// assert_eq!(new_transformed.ncols(), 2);
/// assert_eq!(new_transformed.nrows(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub fn compute_kernel<S>(x: &ArrayBase<S, Ix1>, y: &ArrayBase<S, Ix1>, kernel: &KernelType) -> f64
where
    S: Data<Elem = f64>,
{
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
///
/// Creates a new KernelPCA instance with default values:
/// - `kernel`: Linear kernel (no transformation, equivalent to standard PCA)
/// - `n_components`: 2 (commonly used for visualization purposes)
/// - `eigenvalues`: None (computed during fitting process)
/// - `eigenvectors`: None (computed during fitting process)
/// - `x_fit`: None (stores training data after fitting)
/// - `row_means`: None (mean of each row in kernel matrix, computed during fitting)
/// - `total_mean`: None (overall mean of kernel matrix, computed during fitting)
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
    /// - `Ok(Self)` - A new KernelPCA instance
    /// - `Err(ModelError::InputValidationError)` - If input parameters are invalid
    pub fn new(kernel: KernelType, n_components: usize) -> Result<Self, ModelError> {
        // Validate n_components
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        // Validate kernel parameters
        match &kernel {
            KernelType::Linear => {
                // No parameters to validate for linear kernel
            }
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                if *degree == 0 {
                    return Err(ModelError::InputValidationError(
                        "Polynomial kernel degree must be greater than 0".to_string(),
                    ));
                }
                if !gamma.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Polynomial kernel gamma must be finite, got {}",
                        gamma
                    )));
                }
                if !coef0.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Polynomial kernel coef0 must be finite, got {}",
                        coef0
                    )));
                }
            }
            KernelType::RBF { gamma } => {
                if *gamma <= 0.0 || !gamma.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "RBF kernel gamma must be positive and finite, got {}",
                        gamma
                    )));
                }
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                if !gamma.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Sigmoid kernel gamma must be finite, got {}",
                        gamma
                    )));
                }
                if !coef0.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Sigmoid kernel coef0 must be finite, got {}",
                        coef0
                    )));
                }
            }
        }

        Ok(KernelPCA {
            kernel,
            n_components,
            eigenvalues: None,
            eigenvectors: None,
            x_fit: None,
            row_means: None,
            total_mean: None,
        })
    }

    // Getters
    get_field!(get_kernel, kernel, KernelType);
    get_field!(get_n_components, n_components, usize);
    get_field_as_ref!(get_eigenvalues, eigenvalues, Option<&Array1<f64>>);
    get_field_as_ref!(get_eigenvectors, eigenvectors, Option<&Array2<f64>>);
    get_field_as_ref!(get_x_fit, x_fit, Option<&Array2<f64>>);
    get_field_as_ref!(get_row_means, row_means, Option<&Array1<f64>>);
    get_field!(get_total_mean, total_mean, Option<f64>);

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
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Box<dyn std::error::Error>>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        if x.is_empty() {
            return Err(Box::new(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            )));
        }

        if x.nrows() < self.n_components {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "n_components={} must be less than or equal to the number of samples={}",
                self.n_components,
                x.nrows()
            ))));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(Box::new(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            )));
        }

        let n_samples = x.nrows();
        // Save training data
        self.x_fit = Some(x.to_owned());

        // Calculate kernel matrix: k_matrix[i, j] = kernel(x[i], x[j])
        let mut k_matrix = Array2::<f64>::zeros((n_samples, n_samples));

        // Create progress bar for kernel matrix computation
        let total_computations = (n_samples * (n_samples + 1)) / 2;
        let progress_bar = ProgressBar::new(total_computations as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message("Computing kernel matrix");

        // Closure for kernel computation logic (shared between sequential and parallel)
        let compute_kernel_row = |i: usize| {
            let mut local_results = Vec::new();
            for j in i..n_samples {
                let k_val = compute_kernel(&x.row(i), &x.row(j), &self.kernel);
                local_results.push(((i, j), k_val));
            }
            local_results
        };

        // Use adaptive parallelization based on dataset size
        let kernel_vals: Vec<((usize, usize), f64)> = if n_samples < KERNEL_PCA_PARALLEL_THRESHOLD {
            (0..n_samples).flat_map(compute_kernel_row).collect()
        } else {
            (0..n_samples)
                .into_par_iter()
                .flat_map(compute_kernel_row)
                .collect()
        };

        for ((i, j), k_val) in kernel_vals {
            k_matrix[[i, j]] = k_val;
            k_matrix[[j, i]] = k_val;
            progress_bar.inc(1);
        }

        progress_bar.finish_with_message("Kernel matrix computed");

        // Calculate mean for each row and the overall mean
        let row_means = k_matrix.mean_axis(Axis(1)).unwrap();
        let total_mean = k_matrix.mean().unwrap();
        self.row_means = Some(row_means.clone());
        self.total_mean = Some(total_mean);

        // Center the kernel matrix in-place: k_centered[i,j] = k_matrix[i,j] - row_means[i] - row_means[j] + total_mean
        let mut k_centered = Array2::<f64>::zeros((n_samples, n_samples));
        k_centered.indexed_iter_mut().for_each(|((i, j), val)| {
            *val = k_matrix[[i, j]] - row_means[i] - row_means[j] + total_mean;
        });

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
        eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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

        println!(
            "\nKernel PCA fitting completed: {} samples, {} features, {} components extracted",
            n_samples,
            x.ncols(),
            self.n_components
        );

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
    pub fn transform<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
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

        // Closure for computing kernel column (shared between sequential and parallel)
        let compute_kernel_column = |j: usize| {
            let x_j = x.row(j);
            let mut column_values = Vec::with_capacity(n_train);
            for i in 0..n_train {
                let k_val = compute_kernel(&x_fit.row(i), &x_j, &self.kernel);
                column_values.push(k_val);
            }
            column_values
        };

        // Calculate kernel values with adaptive parallelization
        let kernel_values: Vec<Vec<f64>> = if n_new < KERNEL_PCA_PARALLEL_THRESHOLD {
            (0..n_new).map(compute_kernel_column).collect()
        } else {
            (0..n_new)
                .into_par_iter()
                .map(compute_kernel_column)
                .collect()
        };

        // Fill the kernel matrix from computed values
        for (j, column_values) in kernel_values.iter().enumerate() {
            for (i, &k_val) in column_values.iter().enumerate() {
                k_new[[i, j]] = k_val;
            }
        }

        // Closure for computing column means (shared between sequential and parallel)
        let compute_mean = |j: usize| k_new.column(j).sum() / n_train as f64;

        // Precompute column means with adaptive parallelization
        let column_means: Vec<f64> = if n_new < KERNEL_PCA_PARALLEL_THRESHOLD {
            (0..n_new).map(compute_mean).collect()
        } else {
            (0..n_new).into_par_iter().map(compute_mean).collect()
        };

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
    pub fn fit_transform<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x)?;
        self.transform(x)
    }

    model_save_and_load_methods!(KernelPCA);
}

use super::*;
pub use crate::KernelType;
use std::cmp::Ordering;

/// Threshold for using parallel computation in Kernel PCA.
/// When the number of samples is below this threshold, sequential computation is used.
const KERNEL_PCA_PARALLEL_THRESHOLD: usize = 200;

/// Eigen solver options for Kernel PCA.
///
/// Selects the strategy used to compute eigenvalues and eigenvectors of the
/// centered kernel matrix.
///
/// # Variants
///
/// - `Dense` - Uses dense eigendecomposition via nalgebra
/// - `ARPACK` - Uses power-iteration based approximation
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum EigenSolver {
    Dense,
    ARPACK,
}

/// Kernel Principal Component Analysis (Kernel PCA).
///
/// Projects data into a lower-dimensional space using a nonlinear kernel.
///
/// Stores the fitted training data and kernel statistics needed to transform
/// new samples.
///
/// # Fields
///
/// - `kernel` - Kernel function configuration
/// - `n_components` - Number of components to keep
/// - `eigen_solver` - Eigen solver strategy
/// - `x_fit` - Training data used for fitting
/// - `eigenvalues` - Eigenvalues of the centered kernel matrix
/// - `eigenvectors` - Eigenvectors of the centered kernel matrix
/// - `kernel_row_means` - Per-row means of the training kernel matrix
/// - `kernel_all_mean` - Overall mean of the training kernel matrix
/// - `n_samples` - Number of samples seen during fitting
/// - `n_features` - Number of features seen during fitting
///
/// # Examples
/// ```rust
/// use rustyml::utility::{EigenSolver, KernelPCA, KernelType};
/// use ndarray::array;
///
/// let mut kpca = KernelPCA::new(KernelType::RBF { gamma: 0.1 }, 2, EigenSolver::Dense).unwrap();
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// kpca.fit(&x).unwrap();
/// let projected = kpca.transform(&x).unwrap();
/// assert_eq!(projected.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPCA {
    kernel: KernelType,
    n_components: usize,
    eigen_solver: EigenSolver,
    x_fit: Option<Array2<f64>>,
    eigenvalues: Option<Array1<f64>>,
    eigenvectors: Option<Array2<f64>>,
    kernel_row_means: Option<Array1<f64>>,
    kernel_all_mean: Option<f64>,
    n_samples: Option<usize>,
    n_features: Option<usize>,
}

impl Default for KernelPCA {
    /// Creates a default KernelPCA instance.
    ///
    /// # Default Values
    ///
    /// - `kernel` - RBF kernel with gamma=0.1
    /// - `n_components` - 2
    /// - `eigen_solver` - `EigenSolver::Dense`
    fn default() -> Self {
        KernelPCA::new(KernelType::RBF { gamma: 0.1 }, 2, EigenSolver::Dense)
            .expect("Default KernelPCA parameters should be valid")
    }
}

impl KernelPCA {
    /// Creates a new KernelPCA instance with validated hyperparameters.
    ///
    /// Validates kernel configuration and component count before initialization.
    ///
    /// # Parameters
    ///
    /// - `kernel` - Kernel function type
    /// - `n_components` - Number of components to keep (must be > 0)
    /// - `eigen_solver` - Eigen solver strategy
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new KernelPCA instance or validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `n_components` is 0 or kernel parameters are invalid
    pub fn new(
        kernel: KernelType,
        n_components: usize,
        eigen_solver: EigenSolver,
    ) -> Result<Self, ModelError> {
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        Self::validate_kernel(&kernel)?;

        Ok(Self {
            kernel,
            n_components,
            eigen_solver,
            x_fit: None,
            eigenvalues: None,
            eigenvectors: None,
            kernel_row_means: None,
            kernel_all_mean: None,
            n_samples: None,
            n_features: None,
        })
    }

    // Getters
    get_field!(get_kernel, kernel, KernelType);
    get_field!(get_n_components, n_components, usize);
    get_field!(get_eigen_solver, eigen_solver, EigenSolver);
    get_field!(get_n_samples, n_samples, Option<usize>);
    get_field!(get_n_features, n_features, Option<usize>);
    get_field!(get_kernel_all_mean, kernel_all_mean, Option<f64>);
    get_field_as_ref!(get_eigenvalues, eigenvalues, Option<&Array1<f64>>);
    get_field_as_ref!(get_eigenvectors, eigenvectors, Option<&Array2<f64>>);
    get_field_as_ref!(get_kernel_row_means, kernel_row_means, Option<&Array1<f64>>);

    /// Fits the KernelPCA model to the input data.
    ///
    /// Computes the kernel matrix, centers it, and extracts eigen components.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - Mutable reference to self for chaining
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the input is empty or invalid
    /// - `ModelError::ProcessingError` - If kernel computation or eigendecomposition fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `KERNEL_PCA_THRESHOLD`.
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit_internal(x, true)
    }

    /// Transforms data using the fitted KernelPCA model.
    ///
    /// Centers the cross-kernel matrix against the training statistics and
    /// projects it into component space.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed data matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted
    /// - `ModelError::InputValidationError` - If the input is invalid
    /// - `ModelError::ProcessingError` - If kernel centering or projection fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `KERNEL_PCA_THRESHOLD` (200)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.transform_internal(x, true)
    }

    /// Fits the model to the data and then transforms it.
    ///
    /// Computes eigen components and returns the projected data in one step.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed data matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the input is invalid
    /// - `ModelError::ProcessingError` - If kernel computation or eigendecomposition fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `KERNEL_PCA_THRESHOLD` (200)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        let progress_bar = Self::create_progress_bar(2, "Fitting model");
        self.fit_internal(x, false)?;
        progress_bar.inc(1);
        progress_bar.set_message("Transforming data");
        let transformed = self.transform_internal(x, false)?;
        progress_bar.inc(1);
        progress_bar.finish_with_message("Completed");
        Ok(transformed)
    }

    /// Fits the model and updates internal state without exposing progress logic
    fn fit_internal<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        show_progress: bool,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.validate_input(x)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(ModelError::InputValidationError(
                "KernelPCA requires at least 2 samples".to_string(),
            ));
        }

        if self.n_components > n_samples {
            return Err(ModelError::InputValidationError(format!(
                "n_components should be <= {}, got {}",
                n_samples, self.n_components
            )));
        }

        let use_parallel = n_samples >= KERNEL_PCA_PARALLEL_THRESHOLD;

        let progress_bar = if show_progress {
            Some(Self::create_progress_bar(5, "Validating input"))
        } else {
            None
        };

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing kernel matrix");
        }

        // Build the training kernel matrix
        let mut kernel_matrix = self.compute_kernel_matrix(x, use_parallel)?;
        Self::validate_kernel_matrix(&kernel_matrix, use_parallel)?;

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Centering kernel matrix");
        }

        // Compute centering statistics from the training kernel matrix
        let (row_means, overall_mean) = Self::kernel_means(&kernel_matrix, use_parallel)?;
        Self::center_kernel_matrix(&mut kernel_matrix, &row_means, overall_mean, use_parallel);

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing eigen decomposition");
        }

        // Extract eigenvalues and eigenvectors for projection
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&kernel_matrix)?;

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Finalizing model state");
        }

        // Persist fitted state for later transforms
        self.x_fit = Some(x.to_owned());
        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);
        self.kernel_row_means = Some(row_means);
        self.kernel_all_mean = Some(overall_mean);
        self.n_samples = Some(n_samples);
        self.n_features = Some(n_features);

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.finish_with_message("Completed");
        }

        Ok(self)
    }

    /// Transforms input data using the fitted model state
    fn transform_internal<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        show_progress: bool,
    ) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Load fitted state needed for transformation
        let x_fit = self.x_fit.as_ref().ok_or(ModelError::NotFitted)?;
        let eigenvectors = self.eigenvectors.as_ref().ok_or(ModelError::NotFitted)?;
        let eigenvalues = self.eigenvalues.as_ref().ok_or(ModelError::NotFitted)?;
        let kernel_row_means = self
            .kernel_row_means
            .as_ref()
            .ok_or(ModelError::NotFitted)?;
        let kernel_all_mean = self.kernel_all_mean.ok_or(ModelError::NotFitted)?;
        let n_features = self.n_features.ok_or(ModelError::NotFitted)?;

        self.validate_input(x)?;

        if x.ncols() != n_features {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                n_features
            )));
        }

        if eigenvectors.ncols() != eigenvalues.len() {
            return Err(ModelError::ProcessingError(
                "Eigenvectors and eigenvalues dimension mismatch".to_string(),
            ));
        }

        let use_parallel = x.nrows().max(x_fit.nrows()) >= KERNEL_PCA_PARALLEL_THRESHOLD;

        let progress_bar = if show_progress {
            Some(Self::create_progress_bar(4, "Validating input"))
        } else {
            None
        };

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing kernel matrix");
        }

        // Build the cross-kernel matrix with training samples
        let mut kernel_matrix = self.compute_cross_kernel_matrix(x, x_fit, use_parallel)?;
        Self::validate_kernel_matrix(&kernel_matrix, use_parallel)?;

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Centering kernel matrix");
        }

        // Center the cross-kernel matrix using training means
        Self::center_cross_kernel_matrix(
            &mut kernel_matrix,
            kernel_row_means,
            kernel_all_mean,
            use_parallel,
        )?;

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Projecting data");
        }

        // Scale by inverse sqrt of eigenvalues for projection
        let scales = Self::compute_scaling_factors(eigenvalues)?;
        let projected = if use_parallel {
            Self::project_parallel(&kernel_matrix, eigenvectors, &scales)?
        } else {
            let mut projected = kernel_matrix.dot(eigenvectors);
            for (idx, scale) in scales.iter().enumerate() {
                projected.column_mut(idx).mapv_inplace(|val| val * scale);
            }
            projected
        };

        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.finish_with_message("Completed");
        }

        Ok(projected)
    }

    /// Validates input data shape and numerical values
    fn validate_input<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), ModelError>
    where
        S: Data<Elem = f64>,
    {
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Number of features must be greater than 0".to_string(),
            ));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        Ok(())
    }

    /// Validates kernel hyperparameters for correctness
    fn validate_kernel(kernel: &KernelType) -> Result<(), ModelError> {
        match *kernel {
            KernelType::Linear => Ok(()),
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                if degree == 0 {
                    return Err(ModelError::InputValidationError(
                        "Poly kernel degree must be greater than 0".to_string(),
                    ));
                }
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(ModelError::InputValidationError(format!(
                        "Poly kernel gamma must be positive and finite, got {}",
                        gamma
                    )));
                }
                if !coef0.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Poly kernel coef0 must be finite, got {}",
                        coef0
                    )));
                }
                Ok(())
            }
            KernelType::RBF { gamma } => {
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(ModelError::InputValidationError(format!(
                        "RBF kernel gamma must be positive and finite, got {}",
                        gamma
                    )));
                }
                Ok(())
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
                Ok(())
            }
            KernelType::Cosine => Ok(()),
        }
    }

    /// Checks kernel matrix values for finiteness
    fn validate_kernel_matrix(
        kernel_matrix: &Array2<f64>,
        use_parallel: bool,
    ) -> Result<(), ModelError> {
        let invalid = if use_parallel {
            kernel_matrix.par_iter().any(|&val| !val.is_finite())
        } else {
            kernel_matrix.iter().any(|&val| !val.is_finite())
        };

        if invalid {
            return Err(ModelError::ProcessingError(
                "Kernel matrix contains NaN or infinite values".to_string(),
            ));
        }

        Ok(())
    }

    /// Evaluates the configured kernel on two samples
    fn kernel_function(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match self.kernel {
            KernelType::Linear => x1.dot(&x2),
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => (gamma * x1.dot(&x2) + coef0).powf(degree as f64),
            KernelType::RBF { gamma } => {
                let diff = &x1 - &x2;
                let squared_norm = diff.dot(&diff);
                (-gamma * squared_norm).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => (gamma * x1.dot(&x2) + coef0).tanh(),
            KernelType::Cosine => {
                let norm_product = (x1.dot(&x1) * x2.dot(&x2)).sqrt();
                if norm_product <= f64::EPSILON {
                    0.0
                } else {
                    x1.dot(&x2) / norm_product
                }
            }
        }
    }

    /// Computes the square kernel matrix for training data
    fn compute_kernel_matrix<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        use_parallel: bool,
    ) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        let n_samples = x.nrows();
        let rows = self.compute_kernel_rows(x, x, use_parallel);

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_samples), flat).map_err(|e| {
            ModelError::ProcessingError(format!("Failed to build kernel matrix: {}", e))
        })
    }

    /// Computes the cross-kernel matrix between new data and training data
    fn compute_cross_kernel_matrix<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        x_fit: &ArrayBase<S2, Ix2>,
        use_parallel: bool,
    ) -> Result<Array2<f64>, ModelError>
    where
        S1: Data<Elem = f64> + Send + Sync,
        S2: Data<Elem = f64> + Send + Sync,
    {
        let n_samples = x.nrows();
        let n_train = x_fit.nrows();
        let rows = self.compute_kernel_rows(x, x_fit, use_parallel);

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_train), flat).map_err(|e| {
            ModelError::ProcessingError(format!("Failed to build kernel matrix: {}", e))
        })
    }

    /// Computes kernel rows for a pair of datasets
    fn compute_kernel_rows<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix2>,
        use_parallel: bool,
    ) -> Vec<Vec<f64>>
    where
        S1: Data<Elem = f64> + Send + Sync,
        S2: Data<Elem = f64> + Send + Sync,
    {
        let n_rows = x.nrows();
        let n_cols = y.nrows();

        if use_parallel {
            (0..n_rows)
                .into_par_iter()
                .map(|i| {
                    let row_i = x.row(i);
                    let mut row = Vec::with_capacity(n_cols);
                    for j in 0..n_cols {
                        row.push(self.kernel_function(row_i, y.row(j)));
                    }
                    row
                })
                .collect()
        } else {
            let mut rows = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let row_i = x.row(i);
                let mut row = Vec::with_capacity(n_cols);
                for j in 0..n_cols {
                    row.push(self.kernel_function(row_i, y.row(j)));
                }
                rows.push(row);
            }
            rows
        }
    }

    /// Computes row means and overall mean for a kernel matrix
    fn kernel_means(
        kernel_matrix: &Array2<f64>,
        use_parallel: bool,
    ) -> Result<(Array1<f64>, f64), ModelError> {
        let n_samples = kernel_matrix.nrows();
        if n_samples == 0 {
            return Err(ModelError::ProcessingError(
                "Kernel matrix has zero rows".to_string(),
            ));
        }
        let n_cols = kernel_matrix.ncols();
        if n_cols == 0 {
            return Err(ModelError::ProcessingError(
                "Kernel matrix has zero columns".to_string(),
            ));
        }

        let row_means: Vec<f64> = if use_parallel {
            kernel_matrix
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| row.sum() / n_cols as f64)
                .collect()
        } else {
            kernel_matrix
                .axis_iter(Axis(0))
                .map(|row| row.sum() / n_cols as f64)
                .collect()
        };
        let row_means = Array1::from_vec(row_means);

        let total: f64 = if use_parallel {
            row_means.par_iter().copied().sum()
        } else {
            row_means.sum()
        };
        let overall_mean = total / n_samples as f64;

        if !overall_mean.is_finite() {
            return Err(ModelError::ProcessingError(
                "Kernel matrix mean is not finite".to_string(),
            ));
        }

        Ok((row_means, overall_mean))
    }

    /// Centers a kernel matrix in place using training means
    fn center_kernel_matrix(
        kernel_matrix: &mut Array2<f64>,
        row_means: &Array1<f64>,
        overall_mean: f64,
        use_parallel: bool,
    ) {
        let n_rows = kernel_matrix.nrows();
        if use_parallel {
            let row_means = row_means.to_owned();
            kernel_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    let row_mean = row_means[i];
                    for (j, val) in row.iter_mut().enumerate() {
                        *val = *val - row_mean - row_means[j] + overall_mean;
                    }
                });
        } else {
            for i in 0..n_rows {
                let row_mean = row_means[i];
                for j in 0..kernel_matrix.ncols() {
                    kernel_matrix[[i, j]] =
                        kernel_matrix[[i, j]] - row_mean - row_means[j] + overall_mean;
                }
            }
        }
    }

    /// Centers a cross-kernel matrix using training statistics
    fn center_cross_kernel_matrix(
        kernel_matrix: &mut Array2<f64>,
        train_row_means: &Array1<f64>,
        train_overall_mean: f64,
        use_parallel: bool,
    ) -> Result<(), ModelError> {
        let n_train = train_row_means.len();
        if n_train == 0 {
            return Err(ModelError::ProcessingError(
                "Training kernel means are empty".to_string(),
            ));
        }

        if kernel_matrix.ncols() != n_train {
            return Err(ModelError::ProcessingError(
                "Kernel matrix columns do not match training samples".to_string(),
            ));
        }

        let denom = n_train as f64;
        if use_parallel {
            let train_row_means = train_row_means.to_owned();
            kernel_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row| {
                    let row_mean = row.sum() / denom;
                    for (j, val) in row.iter_mut().enumerate() {
                        *val = *val - train_row_means[j] - row_mean + train_overall_mean;
                    }
                });
        } else {
            for mut row in kernel_matrix.axis_iter_mut(Axis(0)) {
                let row_mean = row.sum() / denom;
                for (j, val) in row.iter_mut().enumerate() {
                    *val = *val - train_row_means[j] - row_mean + train_overall_mean;
                }
            }
        }

        Ok(())
    }

    /// Dispatches to the configured eigensolver
    fn compute_eigendecomposition(
        &self,
        kernel_centered: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), ModelError> {
        match self.eigen_solver {
            EigenSolver::Dense => self.compute_dense_eigen(kernel_centered),
            EigenSolver::ARPACK => self.compute_arpack_eigen(kernel_centered),
        }
    }

    /// Computes eigenvalues and eigenvectors via dense decomposition
    fn compute_dense_eigen(
        &self,
        kernel_centered: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), ModelError> {
        let n_samples = kernel_centered.nrows();
        // Convert the kernel matrix to a dense nalgebra matrix
        let kernel_slice = kernel_centered.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert kernel matrix to slice".to_string())
        })?;
        let matrix = nalgebra::DMatrix::from_row_slice(n_samples, n_samples, kernel_slice);
        let eigen = nalgebra::linalg::SymmetricEigen::new(matrix);

        let mut pairs: Vec<(f64, usize)> = eigen
            .eigenvalues
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();

        // Sort eigenpairs by descending eigenvalue
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut eigenvalues = Vec::with_capacity(self.n_components);
        let mut eigenvectors = Array2::<f64>::zeros((n_samples, self.n_components));

        // Copy the top components into output arrays
        for (comp_idx, (_, idx)) in pairs.into_iter().take(self.n_components).enumerate() {
            let value = eigen.eigenvalues[idx];
            eigenvalues.push(value);
            for row in 0..n_samples {
                eigenvectors[[row, comp_idx]] = eigen.eigenvectors[(row, idx)];
            }
        }

        Self::validate_eigenvalues(&eigenvalues)?;

        Ok((Array1::from_vec(eigenvalues), eigenvectors))
    }

    /// Computes eigenvalues and eigenvectors using power iteration
    fn compute_arpack_eigen(
        &self,
        kernel_centered: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), ModelError> {
        let n_samples = kernel_centered.nrows();
        // Deflate the matrix to extract multiple eigenpairs
        let mut matrix = kernel_centered.to_owned();
        let mut eigenvectors = Array2::<f64>::zeros((n_samples, self.n_components));
        let mut eigenvalues = Vec::with_capacity(self.n_components);
        let mut rng = StdRng::seed_from_u64(0);
        let max_iter = 1000usize;
        let tol = 1e-6;
        let use_parallel = n_samples >= KERNEL_PCA_PARALLEL_THRESHOLD;

        for idx in 0..self.n_components {
            // Power iteration for the next dominant component
            let (eigenvector, eigenvalue) =
                Self::power_iteration(&matrix, &mut rng, max_iter, tol, use_parallel)?;
            eigenvectors.column_mut(idx).assign(&eigenvector);
            eigenvalues.push(eigenvalue);

            // Remove the extracted component from the matrix
            let v_col = eigenvector.view().insert_axis(Axis(1));
            let v_row = eigenvector.view().insert_axis(Axis(0));
            matrix -= &(v_col.dot(&v_row) * eigenvalue);
        }

        Self::validate_eigenvalues(&eigenvalues)?;

        Ok((Array1::from_vec(eigenvalues), eigenvectors))
    }

    /// Validates eigenvalues for positivity and finiteness
    fn validate_eigenvalues(eigenvalues: &[f64]) -> Result<(), ModelError> {
        for &value in eigenvalues {
            if !value.is_finite() || value <= 0.0 {
                return Err(ModelError::ProcessingError(format!(
                    "Kernel PCA requires positive finite eigenvalues, got {}",
                    value
                )));
            }
        }
        Ok(())
    }

    /// Runs power iteration to extract a dominant eigenpair
    fn power_iteration(
        matrix: &Array2<f64>,
        rng: &mut StdRng,
        max_iter: usize,
        tol: f64,
        use_parallel: bool,
    ) -> Result<(Array1<f64>, f64), ModelError> {
        let n = matrix.ncols();
        // Start from a random unit vector
        let mut v = Array1::<f64>::from_vec((0..n).map(|_| rng.random_range(-1.0..1.0)).collect());
        let norm = v.dot(&v).sqrt();
        if norm <= f64::EPSILON {
            v.fill(1.0 / (n as f64).sqrt());
        } else {
            v /= norm;
        }

        let mut prev_lambda = 0.0;
        for _ in 0..max_iter {
            // Iterate toward the dominant eigenvector
            let w = Self::mat_vec_mul(matrix, &v, use_parallel);
            let w_norm = w.dot(&w).sqrt();
            if w_norm <= f64::EPSILON || !w_norm.is_finite() {
                return Err(ModelError::ProcessingError(
                    "Power iteration failed to converge".to_string(),
                ));
            }
            let v_next = &w / w_norm;
            let lambda = v_next.dot(&Self::mat_vec_mul(matrix, &v_next, use_parallel));
            if !lambda.is_finite() {
                return Err(ModelError::ProcessingError(
                    "Power iteration produced non-finite eigenvalue".to_string(),
                ));
            }
            if (lambda - prev_lambda).abs() < tol {
                return Ok((v_next, lambda));
            }
            prev_lambda = lambda;
            v = v_next;
        }

        let lambda = v.dot(&Self::mat_vec_mul(matrix, &v, use_parallel));
        if !lambda.is_finite() {
            return Err(ModelError::ProcessingError(
                "Power iteration produced non-finite eigenvalue".to_string(),
            ));
        }
        Ok((v, lambda))
    }

    /// Multiplies a matrix by a vector with optional parallelism
    fn mat_vec_mul(matrix: &Array2<f64>, v: &Array1<f64>, use_parallel: bool) -> Array1<f64> {
        let rows: Vec<f64> = if use_parallel {
            matrix
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| row.dot(v))
                .collect()
        } else {
            matrix.axis_iter(Axis(0)).map(|row| row.dot(v)).collect()
        };
        Array1::from_vec(rows)
    }

    /// Computes scaling factors from eigenvalues for projection
    fn compute_scaling_factors(eigenvalues: &Array1<f64>) -> Result<Vec<f64>, ModelError> {
        let mut scales = Vec::with_capacity(eigenvalues.len());
        for &value in eigenvalues.iter() {
            if !value.is_finite() || value <= 0.0 {
                return Err(ModelError::ProcessingError(format!(
                    "Eigenvalue must be positive and finite, got {}",
                    value
                )));
            }
            scales.push(1.0 / value.sqrt());
        }
        Ok(scales)
    }

    /// Projects data in parallel using eigenvectors and scaling factors
    fn project_parallel(
        kernel_centered: &Array2<f64>,
        eigenvectors: &Array2<f64>,
        scales: &[f64],
    ) -> Result<Array2<f64>, ModelError> {
        let n_samples = kernel_centered.nrows();
        let n_components = eigenvectors.ncols();

        if scales.len() != n_components {
            return Err(ModelError::ProcessingError(
                "Scaling factors dimension mismatch".to_string(),
            ));
        }

        let rows: Vec<Vec<f64>> = kernel_centered
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                let mut projected = vec![0.0; n_components];
                for (idx, eigenvector) in eigenvectors.axis_iter(Axis(1)).enumerate() {
                    projected[idx] = row.dot(&eigenvector) * scales[idx];
                }
                projected
            })
            .collect();

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_components), flat).map_err(|e| {
            ModelError::ProcessingError(format!("Failed to build projected matrix: {}", e))
        })
    }

    /// Creates a progress bar with a consistent style
    fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
        let progress_bar = ProgressBar::new(len);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Stage: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("=>-"),
        );
        progress_bar.set_message(message.to_string());
        progress_bar
    }

    model_save_and_load_methods!(KernelPCA);
}

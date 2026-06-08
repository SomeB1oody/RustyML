use crate::error::Error;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, Axis, Data, Ix2, Zip};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::cmp::Ordering;

// Re-exported from the canonical `crate::types` home (not the crate-root back-compat
// alias) so the source of `KernelType` is unambiguous.
pub use crate::types::KernelType;

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
/// - `Dense` - Exact dense symmetric eigendecomposition via nalgebra; best for small
///   to mid-sized kernel matrices
/// - `Lanczos` - Krylov-subspace iterative solver (the pure-Rust counterpart of the
///   symmetric solver behind ARPACK); fast and accurate for a few leading components
///   of a large kernel matrix
/// - `PowerIteration` - Power iteration with Hotelling deflation, extracting one
///   component at a time; simplest iterative option
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum EigenSolver {
    Dense,
    Lanczos,
    PowerIteration,
}

impl EigenSolver {
    /// Computes the top `n_components` eigenpairs of the symmetric centered kernel
    /// matrix, returning the eigenvalues alongside the eigenvectors stored as columns
    /// (the layout the projection step expects).
    ///
    /// This is the single dispatch point over the solver strategies; the eigenvalue
    /// positivity check that Kernel PCA additionally requires is applied by the caller.
    fn decompose(
        &self,
        kernel_centered: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), Error> {
        match self {
            EigenSolver::Dense => Self::dense(kernel_centered, n_components),
            EigenSolver::Lanczos => Self::columns_from_pairs(
                super::linalg::top_eigenpairs_lanczos(kernel_centered, n_components, 0)?,
                kernel_centered.nrows(),
                n_components,
            ),
            EigenSolver::PowerIteration => Self::columns_from_pairs(
                super::linalg::top_eigenpairs_power_iteration(
                    kernel_centered.to_owned(),
                    n_components,
                    0,
                    1000,
                    1e-6,
                )?,
                kernel_centered.nrows(),
                n_components,
            ),
        }
    }

    /// Exact path: dense symmetric eigendecomposition, then take the leading components.
    fn dense(
        kernel_centered: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), Error> {
        let n_samples = kernel_centered.nrows();
        let kernel_slice = kernel_centered
            .as_slice()
            .ok_or_else(|| Error::computation("Failed to convert kernel matrix to slice"))?;
        let matrix = nalgebra::DMatrix::from_row_slice(n_samples, n_samples, kernel_slice);
        let eigen = nalgebra::linalg::SymmetricEigen::new(matrix);

        // Sort eigenpairs by descending eigenvalue.
        let mut pairs: Vec<(f64, usize)> = eigen
            .eigenvalues
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut eigenvalues = Vec::with_capacity(n_components);
        let mut eigenvectors = Array2::<f64>::zeros((n_samples, n_components));
        for (comp_idx, (_, idx)) in pairs.into_iter().take(n_components).enumerate() {
            eigenvalues.push(eigen.eigenvalues[idx]);
            for row in 0..n_samples {
                eigenvectors[[row, comp_idx]] = eigen.eigenvectors[(row, idx)];
            }
        }

        Ok((Array1::from_vec(eigenvalues), eigenvectors))
    }

    /// Iterative path shared by Lanczos and power iteration: arrange the returned
    /// eigenvectors as the columns of an `(n_samples × n_components)` matrix.
    fn columns_from_pairs(
        pairs: (Vec<f64>, Vec<Array1<f64>>),
        n_samples: usize,
        n_components: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), Error> {
        let (eigenvalues, eigenvectors) = pairs;
        if eigenvectors.len() < n_components {
            return Err(Error::computation(
                "Solver could not extract the requested number of components",
            ));
        }
        let mut matrix = Array2::<f64>::zeros((n_samples, n_components));
        for (idx, eigenvector) in eigenvectors.iter().enumerate() {
            matrix.column_mut(idx).assign(eigenvector);
        }
        Ok((Array1::from_vec(eigenvalues), matrix))
    }
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
/// use rustyml::utility::kernel_pca::{EigenSolver, KernelPCA, KernelType};
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
    /// - `Result<Self, Error>` - A new KernelPCA instance or validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n_components` is 0 or kernel parameters are invalid
    pub fn new(
        kernel: KernelType,
        n_components: usize,
        eigen_solver: EigenSolver,
    ) -> Result<Self, Error> {
        if n_components == 0 {
            return Err(Error::invalid_parameter(
                "n_components",
                "must be greater than 0",
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
    get_field_as_ref!(get_eigenvalues, eigenvalues, Option<&Array1<f64>>);
    get_field_as_ref!(get_eigenvectors, eigenvectors, Option<&Array2<f64>>);
    // `kernel_row_means` / `kernel_all_mean` are internal centering statistics used only
    // to center the cross-kernel matrix during `transform`; they are intentionally not
    // exposed via getters.

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
    /// - `Result<&mut Self, Error>` - Mutable reference to self for chaining
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::InvalidInput` - If the input is empty or invalid
    /// - `Error::InvalidParameter` - If `n_components` exceeds the number of samples
    /// - `Error::NonFinite` / `Error::Computation` - If kernel computation or eigendecomposition fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least
    /// `KERNEL_PCA_PARALLEL_THRESHOLD` (200).
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit_internal(x)
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
    /// - `Result<Array2<f64>, Error>` - Transformed data matrix
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::InvalidInput` / `Error::DimensionMismatch` - If the input is invalid
    /// - `Error::NonFinite` / `Error::Computation` - If kernel centering or projection fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `KERNEL_PCA_PARALLEL_THRESHOLD` (200)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.transform_internal(x)
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
    /// - `Result<Array2<f64>, Error>` - Transformed data matrix
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::InvalidInput` - If the input is invalid
    /// - `Error::InvalidParameter` - If `n_components` exceeds the number of samples
    /// - `Error::NonFinite` / `Error::Computation` - If kernel computation or eigendecomposition fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `KERNEL_PCA_PARALLEL_THRESHOLD` (200)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        #[cfg(feature = "show_progress")]
        let progress_bar = crate::create_progress_bar(
            2,
            "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
        );
        self.fit_internal(x)?;
        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Transforming data");
        }
        let transformed = self.transform_internal(x)?;
        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }
        Ok(transformed)
    }

    /// Fits the model and updates internal state without exposing progress logic
    fn fit_internal<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        super::validation::validate_fit_matrix(x)?;
        super::validation::check_min_samples(x, 2, "KernelPCA")?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_components > n_samples {
            return Err(Error::invalid_parameter(
                "n_components",
                format!("should be <= {}, got {}", n_samples, self.n_components),
            ));
        }

        let use_parallel = n_samples >= KERNEL_PCA_PARALLEL_THRESHOLD;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                5,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Computing kernel matrix");
        }

        // Build the training kernel (Gram) matrix
        let mut kernel_matrix = self.compute_kernel_matrix(x, x, use_parallel);
        Self::validate_kernel_matrix(&kernel_matrix, use_parallel)?;

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Centering kernel matrix");
        }

        // Compute centering statistics from the training kernel matrix
        let (row_means, overall_mean) = Self::kernel_means(&kernel_matrix, use_parallel)?;
        Self::center_kernel_matrix(&mut kernel_matrix, &row_means, overall_mean, use_parallel);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Computing eigen decomposition");
        }

        // Extract eigenvalues and eigenvectors for projection, then apply Kernel PCA's
        // positivity requirement on the resulting eigenvalues.
        let (eigenvalues, eigenvectors) = self
            .eigen_solver
            .decompose(&kernel_matrix, self.n_components)?;
        Self::validate_eigenvalues(&eigenvalues)?;

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Finalizing model state");
        }

        // Persist fitted state for later transforms
        self.x_fit = Some(x.to_owned());
        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);
        self.kernel_row_means = Some(row_means);
        self.kernel_all_mean = Some(overall_mean);
        self.n_samples = Some(n_samples);
        self.n_features = Some(n_features);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(self)
    }

    /// Transforms input data using the fitted model state
    fn transform_internal<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Load fitted state needed for transformation
        let x_fit = self
            .x_fit
            .as_ref()
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;
        let eigenvectors = self
            .eigenvectors
            .as_ref()
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;
        let kernel_row_means = self
            .kernel_row_means
            .as_ref()
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;
        let kernel_all_mean = self
            .kernel_all_mean
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;
        let n_features = self
            .n_features
            .ok_or_else(|| Error::not_fitted("KernelPCA"))?;

        super::validation::validate_transform_matrix(x, n_features)?;

        if eigenvectors.ncols() != eigenvalues.len() {
            return Err(Error::computation(
                "Eigenvectors and eigenvalues dimension mismatch",
            ));
        }

        let use_parallel = x.nrows().max(x_fit.nrows()) >= KERNEL_PCA_PARALLEL_THRESHOLD;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                4,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Computing kernel matrix");
        }

        // Build the cross-kernel matrix between new data and the fitted samples
        let mut kernel_matrix = self.compute_kernel_matrix(x, x_fit, use_parallel);
        Self::validate_kernel_matrix(&kernel_matrix, use_parallel)?;

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Centering kernel matrix");
        }

        // Center the cross-kernel matrix using training means
        Self::center_cross_kernel_matrix(
            &mut kernel_matrix,
            kernel_row_means,
            kernel_all_mean,
            use_parallel,
        )?;

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Projecting data");
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

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(projected)
    }

    /// Validates kernel hyperparameters for correctness
    fn validate_kernel(kernel: &KernelType) -> Result<(), Error> {
        match *kernel {
            KernelType::Linear => Ok(()),
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                if degree == 0 {
                    return Err(Error::invalid_parameter(
                        "degree",
                        "Poly kernel degree must be greater than 0",
                    ));
                }
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(Error::invalid_parameter(
                        "gamma",
                        format!(
                            "Poly kernel gamma must be positive and finite, got {}",
                            gamma
                        ),
                    ));
                }
                if !coef0.is_finite() {
                    return Err(Error::invalid_parameter(
                        "coef0",
                        format!("Poly kernel coef0 must be finite, got {}", coef0),
                    ));
                }
                Ok(())
            }
            KernelType::RBF { gamma } => {
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(Error::invalid_parameter(
                        "gamma",
                        format!(
                            "RBF kernel gamma must be positive and finite, got {}",
                            gamma
                        ),
                    ));
                }
                Ok(())
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                if !gamma.is_finite() {
                    return Err(Error::invalid_parameter(
                        "gamma",
                        format!("Sigmoid kernel gamma must be finite, got {}", gamma),
                    ));
                }
                if !coef0.is_finite() {
                    return Err(Error::invalid_parameter(
                        "coef0",
                        format!("Sigmoid kernel coef0 must be finite, got {}", coef0),
                    ));
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
    ) -> Result<(), Error> {
        let invalid = if use_parallel {
            kernel_matrix.par_iter().any(|&val| !val.is_finite())
        } else {
            kernel_matrix.iter().any(|&val| !val.is_finite())
        };

        if invalid {
            return Err(Error::non_finite("kernel matrix"));
        }

        Ok(())
    }

    /// Computes the kernel matrix `K[i, j] = kernel(x_i, y_j)` directly into an
    /// `Array2`, in parallel over rows when requested.
    ///
    /// Used both for the training Gram matrix (called with `y == x`) and for the
    /// cross-kernel matrix against the fitted data during `transform`.
    fn compute_kernel_matrix<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix2>,
        use_parallel: bool,
    ) -> Array2<f64>
    where
        S1: Data<Elem = f64> + Sync,
        S2: Data<Elem = f64> + Sync,
    {
        let mut matrix = Array2::<f64>::zeros((x.nrows(), y.nrows()));
        let fill = |mut out_row: ArrayViewMut1<f64>, x_row: ArrayView1<f64>| {
            for (j, y_row) in y.outer_iter().enumerate() {
                out_row[j] = self.kernel.compute(x_row, y_row);
            }
        };
        if use_parallel {
            Zip::from(matrix.outer_iter_mut())
                .and(x.outer_iter())
                .par_for_each(fill);
        } else {
            Zip::from(matrix.outer_iter_mut())
                .and(x.outer_iter())
                .for_each(fill);
        }
        matrix
    }

    /// Computes row means and overall mean for a kernel matrix
    fn kernel_means(
        kernel_matrix: &Array2<f64>,
        use_parallel: bool,
    ) -> Result<(Array1<f64>, f64), Error> {
        let n_samples = kernel_matrix.nrows();
        if n_samples == 0 {
            return Err(Error::computation("Kernel matrix has zero rows"));
        }
        let n_cols = kernel_matrix.ncols();
        if n_cols == 0 {
            return Err(Error::computation("Kernel matrix has zero columns"));
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
            return Err(Error::non_finite("kernel matrix mean"));
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
    ) -> Result<(), Error> {
        let n_train = train_row_means.len();
        if n_train == 0 {
            return Err(Error::computation("Training kernel means are empty"));
        }

        if kernel_matrix.ncols() != n_train {
            return Err(Error::computation(
                "Kernel matrix columns do not match training samples",
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

    /// Validates eigenvalues for positivity and finiteness
    fn validate_eigenvalues(eigenvalues: &Array1<f64>) -> Result<(), Error> {
        for &value in eigenvalues.iter() {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::computation(format!(
                    "Kernel PCA requires positive finite eigenvalues, got {}",
                    value
                )));
            }
        }
        Ok(())
    }

    /// Computes scaling factors from eigenvalues for projection
    fn compute_scaling_factors(eigenvalues: &Array1<f64>) -> Result<Vec<f64>, Error> {
        let mut scales = Vec::with_capacity(eigenvalues.len());
        for &value in eigenvalues.iter() {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::computation(format!(
                    "Eigenvalue must be positive and finite, got {}",
                    value
                )));
            }
            scales.push(1.0 / value.sqrt());
        }
        Ok(scales)
    }

    /// Projects the centered kernel matrix onto the eigenvectors (scaled by the
    /// inverse square root of the eigenvalues), one output row per thread.
    fn project_parallel(
        kernel_centered: &Array2<f64>,
        eigenvectors: &Array2<f64>,
        scales: &[f64],
    ) -> Result<Array2<f64>, Error> {
        let n_components = eigenvectors.ncols();
        if scales.len() != n_components {
            return Err(Error::computation("Scaling factors dimension mismatch"));
        }

        let mut projected = Array2::<f64>::zeros((kernel_centered.nrows(), n_components));
        Zip::from(projected.outer_iter_mut())
            .and(kernel_centered.outer_iter())
            .par_for_each(|mut out_row, k_row| {
                for (idx, eigenvector) in eigenvectors.axis_iter(Axis(1)).enumerate() {
                    out_row[idx] = k_row.dot(&eigenvector) * scales[idx];
                }
            });

        Ok(projected)
    }

    model_save_and_load_methods!(KernelPCA);
}

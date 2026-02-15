use crate::error::ModelError;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::IntoParallelRefIterator;

/// SVD solver options for Principal Component Analysis.
///
/// Selects the decomposition strategy used to compute principal components.
/// For small to mid-sized datasets (typically fewer than 10,000 samples or features),
/// `Full` is recommended for accuracy. For large datasets (10,000+ samples or features),
/// `Randomized` is recommended for speed with good accuracy. `ARPACK` is recommended when
/// you need only a few components from very large, sparse, or memory-constrained problems.
///
/// # Variants
///
/// - `Full` - Full SVD using deterministic decomposition
/// - `Randomized` - Randomized SVD with a fixed RNG seed
/// - `ARPACK` - Power-iteration based approximation
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum SVDSolver {
    Full,
    Randomized(u64),
    ARPACK,
}

/// Threshold for using parallel computation in PCA.
/// When the number of samples is below this threshold, sequential computation is used.
const PCA_PARALLEL_THRESHOLD: usize = 200;

/// Principal Component Analysis (PCA) model.
///
/// Reduces dimensionality by projecting data onto orthogonal components that capture the highest
/// variance.
///
/// # Fields
///
/// - `n_components` - Number of components to keep
/// - `svd_solver` - SVD solver strategy
/// - `mean` - Per-feature mean used for centering
/// - `components` - Principal axes in feature space
/// - `explained_variance` - Variance explained by each selected component
/// - `explained_variance_ratio` - Ratio of variance explained by each selected component
/// - `singular_values` - Singular values corresponding to each component
/// - `n_samples` - Number of samples seen during fitting
/// - `n_features` - Number of features seen during fitting
///
/// # Examples
/// ```rust
/// use rustyml::utility::*;
/// use ndarray::array;
///
/// let mut pca = principal_component_analysis::PCA::new(
///     2,
///     principal_component_analysis::SVDSolver::Full,
/// )
/// .unwrap();
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// pca.fit(&x).unwrap();
/// let projected = pca.transform(&x).unwrap();
/// assert_eq!(projected.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    n_components: usize,
    svd_solver: SVDSolver,
    mean: Option<Array1<f64>>,
    components: Option<Array2<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    singular_values: Option<Array1<f64>>,
    n_samples: Option<usize>,
    n_features: Option<usize>,
}

impl Default for PCA {
    /// Creates a default PCA instance.
    ///
    /// # Default Values
    ///
    /// - `n_components` - 2
    /// - `svd_solver` - `SVDSolver::Full`
    fn default() -> Self {
        PCA::new(2, SVDSolver::Full).expect("Default PCA parameters should be valid")
    }
}

impl PCA {
    /// Creates a new PCA instance with validated hyperparameters.
    ///
    /// Solver guidance: choose `SVDSolver::Full` for small to mid-sized datasets (typically fewer
    /// than 10,000 samples or features), `SVDSolver::Randomized` for large datasets (10,000+ samples
    /// or features) when speed matters, and `SVDSolver::ARPACK` when extracting only a few
    /// components from very large, sparse, or memory-constrained problems.
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of components to keep (must be > 0)
    /// - `svd_solver` - SVD solver strategy
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new PCA instance or validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `n_components` is 0
    pub fn new(n_components: usize, svd_solver: SVDSolver) -> Result<Self, ModelError> {
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        // Initialize model state with unset learned parameters
        Ok(Self {
            n_components,
            svd_solver,
            mean: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            n_samples: None,
            n_features: None,
        })
    }

    // Getters
    get_field!(get_n_components, n_components, usize);
    get_field!(get_svd_solver, svd_solver, SVDSolver);
    get_field!(get_n_samples, n_samples, Option<usize>);
    get_field!(get_n_features, n_features, Option<usize>);
    get_field_as_ref!(get_mean, mean, Option<&Array1<f64>>);
    get_field_as_ref!(get_components, components, Option<&Array2<f64>>);
    get_field_as_ref!(
        get_explained_variance,
        explained_variance,
        Option<&Array1<f64>>
    );
    get_field_as_ref!(
        get_explained_variance_ratio,
        explained_variance_ratio,
        Option<&Array1<f64>>
    );
    get_field_as_ref!(get_singular_values, singular_values, Option<&Array1<f64>>);

    /// Fits the PCA model.
    ///
    /// Computes the mean, principal components, and variance statistics for the input data.
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
    /// - `ModelError::InputValidationError` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `ModelError::ProcessingError` - If the decomposition fails or numerical issues occur
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `PCA_PARALLEL_THRESHOLD` (200).
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.fit_internal(x)
    }

    /// Transforms data into principal component space.
    ///
    /// Projects centered data onto the fitted principal components.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted
    /// - `ModelError::InputValidationError` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `ModelError::ProcessingError` - If projection fails
    ///
    /// # Performance
    ///
    /// Uses parallel projection when the number of samples is at least `PCA_PARALLEL_THRESHOLD` (200).
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.transform_internal(x)
    }

    /// Fits the model and transforms the data.
    ///
    /// Computes components and returns the projected data in a single step.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `ModelError::ProcessingError` - If the decomposition or projection fails
    ///
    /// # Performance
    ///
    /// Uses parallel computation when the number of samples is at least `PCA_PARALLEL_THRESHOLD` (200).
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                2,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Fitting model");
            pb
        };
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

    /// Transforms data from principal component space back to original feature space.
    ///
    /// Reconstructs approximate original data using the fitted components and mean.
    ///
    /// # Parameters
    ///
    /// - `x` - PCA-transformed data matrix
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Reconstructed data in original feature space
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted
    /// - `ModelError::InputValidationError` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `ModelError::ProcessingError` - If reconstruction fails
    ///
    /// # Performance
    ///
    /// Uses parallel reconstruction when the number of samples is at least `PCA_PARALLEL_THRESHOLD` (200).
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot inverse transform empty dataset".to_string(),
            ));
        }

        if x.ncols() != components.nrows() {
            return Err(ModelError::InputValidationError(format!(
                "Number of components does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                components.nrows()
            )));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                3,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };
        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Reconstructing data");
        }

        let reconstructed = if x.nrows() >= PCA_PARALLEL_THRESHOLD {
            let x_owned = x.to_owned();
            Self::reconstruct_parallel(&x_owned, components, mean)?
        } else {
            let mut reconstructed = x.dot(components);
            reconstructed += mean;
            reconstructed
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Finalizing output");
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(reconstructed)
    }

    /// Fits the model and updates internal state without exposing progress logic
    fn fit_internal<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
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

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(ModelError::InputValidationError(
                "PCA requires at least 2 samples".to_string(),
            ));
        }

        // Enforce component count against data rank limits
        let max_components = n_samples.min(n_features);
        if self.n_components > max_components {
            return Err(ModelError::InputValidationError(format!(
                "n_components should be <= {}, got {}",
                max_components, self.n_components
            )));
        }

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
            progress_bar.set_message("Centering data");
        }

        // Center data and keep the mean for later transforms
        let mut x_centered = x.to_owned();
        let mean = Self::compute_mean(&x_centered);
        Self::center_data(&mut x_centered, &mean);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Computing decomposition");
        }

        // Compute principal axes and singular values
        let (components, singular_values) = self.compute_components(&x_centered)?;

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Computing explained variance");
        }

        // Convert singular values into variance statistics
        let explained_variance = singular_values.mapv(|s| (s * s) / ((n_samples - 1) as f64));
        let total_variance = Self::total_variance(&x_centered, n_samples)?;
        let explained_variance_ratio = if total_variance > 0.0 && total_variance.is_finite() {
            explained_variance.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(self.n_components)
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Finalizing model state");
        }

        // Store learned parameters for future transforms
        self.mean = Some(mean);
        self.components = Some(components);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.singular_values = Some(singular_values);
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
    fn transform_internal<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot transform empty dataset".to_string(),
            ));
        }

        if x.ncols() != components.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, expected: {}",
                x.ncols(),
                components.ncols()
            )));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                3,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Centering data");
        }

        // Reuse training mean to center new samples
        let mut x_centered = x.to_owned();
        Self::center_data(&mut x_centered, mean);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Projecting data");
        }

        // Project into component space with optional parallelism
        let transformed = if x_centered.nrows() >= PCA_PARALLEL_THRESHOLD {
            Self::project_parallel(&x_centered, components)?
        } else {
            x_centered.dot(&components.t())
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(transformed)
    }

    /// Computes the per-feature mean for centering
    fn compute_mean(x: &Array2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute column means with a parallel path for large matrices
        if n_samples >= PCA_PARALLEL_THRESHOLD {
            let means: Vec<f64> = (0..n_features)
                .into_par_iter()
                .map(|col| x.column(col).sum() / n_samples as f64)
                .collect();
            Array1::from_vec(means)
        } else {
            x.mean_axis(Axis(0)).expect("Input data must be non-empty")
        }
    }

    /// Centers data in place by subtracting the mean
    fn center_data(x: &mut Array2<f64>, mean: &Array1<f64>) {
        // Subtract mean from each row in place
        if x.nrows() >= PCA_PARALLEL_THRESHOLD {
            let mean = mean.to_owned();
            x.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row| {
                    row -= &mean;
                });
        } else {
            for mut row in x.axis_iter_mut(Axis(0)) {
                row -= mean;
            }
        }
    }

    /// Computes total variance of centered data
    fn total_variance(x_centered: &Array2<f64>, n_samples: usize) -> Result<f64, ModelError> {
        let denom = (n_samples - 1) as f64;
        if denom <= 0.0 {
            return Err(ModelError::ProcessingError(
                "Variance computation requires at least 2 samples".to_string(),
            ));
        }

        // Sum of squares over centered data
        let sum_sq = if x_centered.nrows() >= PCA_PARALLEL_THRESHOLD {
            if let Some(slice) = x_centered.as_slice() {
                slice.par_iter().map(|v| v * v).sum::<f64>()
            } else {
                x_centered.iter().map(|v| v * v).sum::<f64>()
            }
        } else {
            x_centered.iter().map(|v| v * v).sum::<f64>()
        };

        Ok(sum_sq / denom)
    }

    /// Dispatches to the configured SVD solver
    fn compute_components(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), ModelError> {
        match self.svd_solver {
            SVDSolver::Full => self.compute_full_svd(x_centered),
            SVDSolver::Randomized(seed) => self.compute_randomized_svd(x_centered, seed),
            SVDSolver::ARPACK => self.compute_arpack_svd(x_centered),
        }
    }

    /// Computes components using full SVD
    fn compute_full_svd(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), ModelError> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        // Convert ndarray to nalgebra for SVD
        let x_slice = x_centered.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert centered data to slice".to_string())
        })?;
        let x_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_features, x_slice);
        let svd = nalgebra::linalg::SVD::new(x_mat, false, true);
        let v_t = svd.v_t.ok_or_else(|| {
            ModelError::ProcessingError("SVD did not compute V^T matrix".to_string())
        })?;

        let singular_values: Vec<f64> = svd
            .singular_values
            .iter()
            .take(self.n_components)
            .cloned()
            .collect();

        // Copy the top components from V^T into ndarray layout
        let mut components = Array2::<f64>::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                components[[i, j]] = v_t[(i, j)];
            }
        }

        Ok((components, Array1::from_vec(singular_values)))
    }

    /// Computes components using randomized SVD
    fn compute_randomized_svd(
        &self,
        x_centered: &Array2<f64>,
        seed: u64,
    ) -> Result<(Array2<f64>, Array1<f64>), ModelError> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        let max_rank = n_samples.min(n_features);
        let oversampling = 5usize;
        // Oversample to improve the randomized subspace
        let k = (self.n_components + oversampling).min(max_rank);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut omega = Vec::with_capacity(n_features * k);
        for _ in 0..(n_features * k) {
            omega.push(rng.random_range(-1.0..1.0));
        }

        // Build a random projection matrix and sketch X
        let x_slice = x_centered.as_slice().ok_or_else(|| {
            ModelError::ProcessingError("Failed to convert centered data to slice".to_string())
        })?;
        let x_mat = nalgebra::DMatrix::from_row_slice(n_samples, n_features, x_slice);
        let omega_mat = nalgebra::DMatrix::from_row_slice(n_features, k, &omega);
        let mut y_mat = &x_mat * &omega_mat;

        // Perform a few power iterations to improve spectral separation
        let n_iter = 2usize;
        for _ in 0..n_iter {
            let y_t = x_mat.transpose() * &y_mat;
            y_mat = &x_mat * y_t;
        }

        // Orthonormalize the sketch and compute SVD in the reduced space
        let qr = nalgebra::linalg::QR::new(y_mat);
        let q = qr.q();
        let b = q.transpose() * x_mat;

        let svd = nalgebra::linalg::SVD::new(b, false, true);
        let v_t = svd.v_t.ok_or_else(|| {
            ModelError::ProcessingError("Randomized SVD did not compute V^T matrix".to_string())
        })?;

        let singular_values: Vec<f64> = svd
            .singular_values
            .iter()
            .take(self.n_components)
            .cloned()
            .collect();

        // Expand V^T back to full feature space components
        let mut components = Array2::<f64>::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                components[[i, j]] = v_t[(i, j)];
            }
        }

        Ok((components, Array1::from_vec(singular_values)))
    }

    /// Computes components using power-iteration SVD
    fn compute_arpack_svd(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), ModelError> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        let denom = (n_samples - 1) as f64;
        // Form the covariance matrix for power iteration
        let mut cov = x_centered.t().dot(x_centered) / denom;

        let mut components = Array2::<f64>::zeros((self.n_components, n_features));
        let mut eigenvalues = Vec::with_capacity(self.n_components);
        let mut rng = StdRng::seed_from_u64(0);
        let max_iter = 1000usize;
        let tol = 1e-6;

        for idx in 0..self.n_components {
            // Deflate the covariance as components are extracted
            let (eigenvector, eigenvalue) = Self::power_iteration(&cov, &mut rng, max_iter, tol)?;
            components.row_mut(idx).assign(&eigenvector);
            eigenvalues.push(eigenvalue);

            let v_col = eigenvector.view().insert_axis(Axis(1));
            let v_row = eigenvector.view().insert_axis(Axis(0));
            cov -= &(v_col.dot(&v_row) * eigenvalue);
        }

        // Convert eigenvalues into singular values
        let singular_values: Vec<f64> = eigenvalues
            .into_iter()
            .map(|lambda| {
                let clamped = if lambda.is_finite() && lambda > 0.0 {
                    lambda
                } else {
                    0.0
                };
                (clamped * denom).sqrt()
            })
            .collect();

        Ok((components, Array1::from_vec(singular_values)))
    }

    /// Runs power iteration to extract a dominant eigenpair
    fn power_iteration(
        cov: &Array2<f64>,
        rng: &mut StdRng,
        max_iter: usize,
        tol: f64,
    ) -> Result<(Array1<f64>, f64), ModelError> {
        let n_features = cov.ncols();
        // Start with a random unit vector
        let mut v = Array1::<f64>::from_vec(
            (0..n_features)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect(),
        );
        let norm = v.dot(&v).sqrt();
        if norm <= f64::EPSILON {
            v.fill(1.0 / (n_features as f64).sqrt());
        } else {
            v /= norm;
        }

        let mut prev_lambda = 0.0;
        for _ in 0..max_iter {
            let w = cov.dot(&v);
            let w_norm = w.dot(&w).sqrt();
            if w_norm <= f64::EPSILON || !w_norm.is_finite() {
                return Err(ModelError::ProcessingError(
                    "Power iteration failed to converge".to_string(),
                ));
            }
            // Normalize the next vector estimate
            let v_next = &w / w_norm;
            let lambda = v_next.dot(&cov.dot(&v_next));
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

        let lambda = v.dot(&cov.dot(&v));
        if !lambda.is_finite() {
            return Err(ModelError::ProcessingError(
                "Power iteration produced non-finite eigenvalue".to_string(),
            ));
        }
        Ok((v, lambda))
    }

    /// Projects data in parallel using principal components
    fn project_parallel(
        x_centered: &Array2<f64>,
        components: &Array2<f64>,
    ) -> Result<Array2<f64>, ModelError> {
        let n_samples = x_centered.nrows();
        let n_components = components.nrows();

        // Project each row in parallel to build the output matrix
        let rows: Vec<Vec<f64>> = x_centered
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                let mut projected = vec![0.0; n_components];
                for (idx, comp) in components.outer_iter().enumerate() {
                    projected[idx] = row.dot(&comp);
                }
                projected
            })
            .collect();

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_components), flat).map_err(|e| {
            ModelError::ProcessingError(format!("Failed to build projected matrix: {}", e))
        })
    }

    /// Reconstructs data in parallel from component space
    fn reconstruct_parallel(
        x: &Array2<f64>,
        components: &Array2<f64>,
        mean: &Array1<f64>,
    ) -> Result<Array2<f64>, ModelError> {
        let n_samples = x.nrows();
        let n_features = components.ncols();
        let components_t = components.t().to_owned();
        let mean_vec = mean.to_owned();

        // Reconstruct each row in parallel and add the mean
        let rows: Vec<Vec<f64>> = x
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                let mut reconstructed = vec![0.0; n_features];
                for (j, comp_row) in components_t.outer_iter().enumerate() {
                    reconstructed[j] = row.dot(&comp_row) + mean_vec[j];
                }
                reconstructed
            })
            .collect();

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_features), flat).map_err(|e| {
            ModelError::ProcessingError(format!("Failed to build reconstructed matrix: {}", e))
        })
    }

    model_save_and_load_methods!(PCA);
}

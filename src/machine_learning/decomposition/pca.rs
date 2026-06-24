//! Principal Component Analysis (PCA)
//!
//! Provides the `PCA` model for linear dimensionality reduction and the `SVDSolver`
//! enum selecting the underlying decomposition strategy (full, randomized, or power iteration)

use crate::error::Error;
use crate::math::matmul::gemm_par_auto;
use crate::math::reduction::det_reduce;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, sum_f64_parallel_min_elems};
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// SVD solver options for Principal Component Analysis
///
/// Selects the decomposition strategy used to compute principal components. Use `Full` for
/// small to mid-sized datasets (typically fewer than 10,000 samples or features) when accuracy
/// matters. Use `Randomized` for large datasets (10,000+ samples or features) when speed
/// matters. Use `PowerIteration` when you need only a few components from very large or
/// memory-constrained problems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
pub enum SVDSolver {
    /// Full SVD using deterministic decomposition
    #[default]
    Full,
    /// Randomized SVD with a fixed RNG seed
    Randomized(u64),
    /// Power-iteration approximation with Hotelling deflation, extracting one component
    /// at a time (best when only a few components are needed)
    PowerIteration,
}

impl SVDSolver {
    /// Computes the top `n_components` principal axes and their singular values from
    /// the centered data, dispatching over the configured solver strategy
    ///
    /// Returns `(components, singular_values)`, where the principal axes are the rows of `components`
    fn compute_components(
        &self,
        x_centered: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array2<f64>, Array1<f64>), Error> {
        match *self {
            SVDSolver::Full => Self::full_svd(x_centered, n_components),
            SVDSolver::Randomized(seed) => Self::randomized_svd(x_centered, n_components, seed),
            SVDSolver::PowerIteration => Self::power_iteration_svd(x_centered, n_components),
        }
    }

    /// Exact, deterministic full decomposition via the symmetric eigendecomposition of the
    /// covariance matrix `Xᵀ X / (n - 1)`: its eigenvectors are the principal axes and its
    /// eigenvalues map back to the singular values of the centered data
    fn full_svd(
        x_centered: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array2<f64>, Array1<f64>), Error> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        let denom = (n_samples - 1) as f64;

        // Eigendecompose the covariance matrix: eigenvectors are the principal axes, eigenvalues
        // are the per-axis variances
        let cov = gemm_par_auto(&x_centered.t(), x_centered) / denom;
        let eigen = crate::machine_learning::linalg::symmetric_eigen(&cov);

        // The solver returns eigenpairs ascending; order them by descending eigenvalue
        let mut order: Vec<usize> = (0..n_features).collect();
        order.sort_by(|&a, &b| {
            eigen.eigenvalues[b]
                .partial_cmp(&eigen.eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Principal axes are the covariance eigenvectors (columns), stored as component rows
        let mut components = Array2::<f64>::zeros((n_components, n_features));
        let mut singular_values = Vec::with_capacity(n_components);
        for (row, &idx) in order.iter().take(n_components).enumerate() {
            for j in 0..n_features {
                components[[row, j]] = eigen.eigenvectors[[j, idx]];
            }
            // Convert each covariance eigenvalue back into a singular value of X
            let lambda = eigen.eigenvalues[idx];
            let clamped = if lambda.is_finite() && lambda > 0.0 {
                lambda
            } else {
                0.0
            };
            singular_values.push((clamped * denom).sqrt());
        }

        Ok((components, Array1::from_vec(singular_values)))
    }

    /// Randomized SVD with oversampling and a couple of power iterations
    fn randomized_svd(
        x_centered: &Array2<f64>,
        n_components: usize,
        seed: u64,
    ) -> Result<(Array2<f64>, Array1<f64>), Error> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        let max_rank = n_samples.min(n_features);
        let oversampling = 5usize;
        // Oversample to improve the randomized subspace
        let k = (n_components + oversampling).min(max_rank);

        use crate::machine_learning::linalg::{qr_q, svd};

        let mut rng = StdRng::seed_from_u64(seed);
        let mut omega = Vec::with_capacity(n_features * k);
        for _ in 0..(n_features * k) {
            omega.push(rng.random_range(-1.0..1.0));
        }
        // Random projection matrix Omega (n_features x k), row-major
        let omega = Array2::from_shape_vec((n_features, k), omega)
            .map_err(|_| Error::computation("Failed to build random projection matrix"))?;

        // Initial sketch Y = X * Omega, orthonormalized to an orthonormal basis Q
        let mut q = qr_q(&gemm_par_auto(x_centered, &omega));

        // Subspace (power) iterations with re-orthonormalization between each step
        let n_iter = 2usize;
        for _ in 0..n_iter {
            let w = qr_q(&gemm_par_auto(&x_centered.t(), &q));
            q = qr_q(&gemm_par_auto(x_centered, &w));
        }

        // Project X onto the orthonormal basis and compute the SVD in the reduced space
        let b = gemm_par_auto(&q.t(), x_centered);
        let decomp = svd(&b, false, true);
        let v_t = decomp
            .v_t
            .ok_or_else(|| Error::computation("Randomized SVD did not compute V^T matrix"))?;

        let singular_values: Vec<f64> = decomp
            .singular_values
            .iter()
            .take(n_components)
            .cloned()
            .collect();
        // Expand V^T back to full feature-space components
        let components =
            Array2::<f64>::from_shape_fn((n_components, n_features), |(i, j)| v_t[[i, j]]);

        Ok((components, Array1::from_vec(singular_values)))
    }

    /// Power iteration with Hotelling deflation on the covariance matrix
    fn power_iteration_svd(
        x_centered: &Array2<f64>,
        n_components: usize,
    ) -> Result<(Array2<f64>, Array1<f64>), Error> {
        let n_samples = x_centered.nrows();
        let n_features = x_centered.ncols();
        let denom = (n_samples - 1) as f64;
        // Extract the leading eigenpairs of the covariance matrix
        let cov = gemm_par_auto(&x_centered.t(), x_centered) / denom;
        let (eigenvalues, eigenvectors) =
            crate::machine_learning::linalg::top_eigenpairs_power_iteration(
                cov,
                n_components,
                0,
                1000,
                1e-6,
            )?;

        // Principal axes are the covariance eigenvectors, stored as rows
        let mut components = Array2::<f64>::zeros((n_components, n_features));
        for (idx, eigenvector) in eigenvectors.iter().enumerate() {
            components.row_mut(idx).assign(eigenvector);
        }

        // Convert covariance eigenvalues into the corresponding singular values
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
}

/// Principal Component Analysis (PCA) model
///
/// Reduces dimensionality by projecting data onto orthogonal components that capture the highest
/// variance
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::decomposition::PCA;
/// use ndarray::array;
///
/// let mut pca = PCA::new(2).unwrap();
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// pca.fit(&x).unwrap();
/// let projected = pca.transform(&x).unwrap();
/// assert_eq!(projected.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    /// Number of components to keep
    n_components: usize,
    /// SVD solver strategy
    svd_solver: SVDSolver,
    /// Per-feature mean used for centering
    mean: Option<Array1<f64>>,
    /// Principal axes in feature space
    components: Option<Array2<f64>>,
    /// Variance explained by each selected component
    explained_variance: Option<Array1<f64>>,
    /// Ratio of variance explained by each selected component
    explained_variance_ratio: Option<Array1<f64>>,
    /// Singular values corresponding to each component
    singular_values: Option<Array1<f64>>,
    /// Number of samples seen during fitting
    n_samples: Option<usize>,
    /// Number of features seen during fitting
    n_features: Option<usize>,
}

impl Default for PCA {
    /// Creates a default PCA instance
    ///
    /// # Default Values
    ///
    /// - `n_components` - 2
    /// - `svd_solver` - `SVDSolver::Full`
    fn default() -> Self {
        PCA::new(2).expect("Default PCA parameters should be valid")
    }
}

impl PCA {
    /// Creates a new PCA instance with validated hyperparameters
    ///
    /// Solver guidance: choose `SVDSolver::Full` for small to mid-sized datasets (typically fewer
    /// than 10,000 samples or features), `SVDSolver::Randomized` for large datasets (10,000+ samples
    /// or features) when speed matters, and `SVDSolver::PowerIteration` when extracting only a few
    /// components from very large or memory-constrained problems
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of components to keep (must be > 0)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new PCA instance or validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n_components` is 0
    ///
    /// # Notes
    ///
    /// The SVD solver defaults to `SVDSolver::Full`. To pick another strategy (e.g. a
    /// randomized or power-iteration solver for large data), use the builder method below:
    ///
    /// - [`with_svd_solver`](Self::with_svd_solver) - SVD solver strategy
    pub fn new(n_components: usize) -> Result<Self, Error> {
        if n_components == 0 {
            return Err(Error::invalid_parameter(
                "n_components",
                "must be greater than 0",
            ));
        }

        // Learned parameters stay unset until fit
        Ok(Self {
            n_components,
            svd_solver: SVDSolver::Full,
            mean: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            n_samples: None,
            n_features: None,
        })
    }

    /// Sets the SVD solver strategy (default: `SVDSolver::Full`)
    ///
    /// # Parameters
    ///
    /// - `svd_solver` - the solver strategy to use
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_svd_solver(mut self, svd_solver: SVDSolver) -> Self {
        self.svd_solver = svd_solver;
        self
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

    /// Fits the PCA model
    ///
    /// Computes the mean, principal components, and variance statistics for the input data
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
    /// - `Error::EmptyInput` / `Error::NonFinite` / `Error::InvalidParameter` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `Error::Computation` - If the decomposition fails or numerical issues occur
    ///
    /// # Performance
    ///
    /// The covariance/projection GEMMs run parallel above their FLOPs gates; centering
    /// and the variance reduction parallelize above the calibrated cheap-map/sum gates (see
    /// `crate::parallel_gates`)
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit_internal(x)
    }

    /// Transforms data into principal component space
    ///
    /// Projects centered data onto the fitted principal components
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::EmptyInput` / `Error::NonFinite` / `Error::DimensionMismatch` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `Error::Computation` - If projection fails
    ///
    /// # Performance
    ///
    /// The projection GEMM runs on the `gemm` backend (parallelized for large products)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.transform_internal(x)
    }

    /// Fits the model and transforms the data
    ///
    /// Computes components and returns the projected data in a single step
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::NonFinite` / `Error::InvalidParameter` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `Error::Computation` - If the decomposition or projection fails
    ///
    /// # Performance
    ///
    /// The covariance/projection GEMMs run parallel above their FLOPs gates; centering
    /// and the variance reduction parallelize above the calibrated cheap-map/sum gates (see
    /// `crate::parallel_gates`)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
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

    /// Transforms data from principal component space back to original feature space
    ///
    /// Reconstructs approximate original data using the fitted components and mean
    ///
    /// # Parameters
    ///
    /// - `x` - PCA-transformed data matrix
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - Reconstructed data in original feature space
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::EmptyInput` / `Error::NonFinite` / `Error::DimensionMismatch` - If the input is empty, has non-finite values, or has incompatible dimensions
    /// - `Error::Computation` - If reconstruction fails
    ///
    /// # Performance
    ///
    /// The reconstruction GEMM runs on the `gemm` backend (parallelized for large products)
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| Error::not_fitted("PCA"))?;
        let mean = self.mean.as_ref().ok_or_else(|| Error::not_fitted("PCA"))?;

        crate::machine_learning::validation::check_non_empty(x)?;
        if x.ncols() != components.nrows() {
            return Err(Error::dimension_mismatch(components.nrows(), x.ncols()));
        }
        crate::machine_learning::validation::check_finite(x)?;

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

        // Map back to feature space; the GEMM parallelizes above its FLOPs gate
        let mut reconstructed = gemm_par_auto(x, components);
        reconstructed += mean;

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
    fn fit_internal<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        crate::machine_learning::validation::validate_fit_matrix(x)?;
        crate::machine_learning::validation::check_min_samples(x, 2, "PCA")?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // n_components cannot exceed the data rank min(n_samples, n_features)
        let max_components = n_samples.min(n_features);
        if self.n_components > max_components {
            return Err(Error::invalid_parameter(
                "n_components",
                format!("should be <= {}, got {}", max_components, self.n_components),
            ));
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
        let (mut components, singular_values) = self
            .svd_solver
            .compute_components(&x_centered, self.n_components)?;

        // Fix component signs so the result is deterministic and identical across solvers
        Self::flip_component_signs(&mut components);

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
    fn transform_internal<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| Error::not_fitted("PCA"))?;
        let mean = self.mean.as_ref().ok_or_else(|| Error::not_fitted("PCA"))?;

        crate::machine_learning::validation::validate_transform_matrix(x, components.ncols())?;

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

        // Project into component space; the GEMM parallelizes above its FLOPs gate
        let transformed = gemm_par_auto(&x_centered, &components.t());

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(transformed)
    }

    /// Computes the per-feature mean for centering
    fn compute_mean(x: &Array2<f64>) -> Array1<f64> {
        // `mean_axis` sums each column in row-major (cache-friendly) order
        x.mean_axis(Axis(0)).expect("Input data must be non-empty")
    }

    /// Centers data in place by subtracting the mean (cheap-map class gate on the total
    /// element count)
    fn center_data(x: &mut Array2<f64>, mean: &Array1<f64>) {
        if x.len() >= cheap_map_f64_parallel_threshold() {
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

    /// Fixes the sign of each principal axis so the decomposition is deterministic
    ///
    /// Each component (row of `components`) is negated when its largest-magnitude loading is
    /// negative, making that entry non-negative. This removes the sign ambiguity inherent to
    /// eigen/SVD decompositions so all solvers and repeated runs agree on the orientation of every
    /// axis. The decision uses the component vectors themselves, not `U` (which the randomized and
    /// power-iteration solvers never form), so it is internally consistent but need not byte-match a
    /// `U`-based sign convention. Reconstructions are unaffected: flipping an axis flips its
    /// scores in step, leaving their product unchanged
    fn flip_component_signs(components: &mut Array2<f64>) {
        for mut row in components.axis_iter_mut(Axis(0)) {
            // Index of the largest-magnitude loading; ties keep the first (lowest) index
            let mut max_idx = 0;
            let mut max_abs = 0.0;
            for (j, &v) in row.iter().enumerate() {
                let abs = v.abs();
                if abs > max_abs {
                    max_abs = abs;
                    max_idx = j;
                }
            }
            if row[max_idx] < 0.0 {
                row.mapv_inplace(|x| -x);
            }
        }
    }

    /// Computes total variance of centered data
    fn total_variance(x_centered: &Array2<f64>, n_samples: usize) -> Result<f64, Error> {
        let denom = (n_samples - 1) as f64;
        if denom <= 0.0 {
            return Err(Error::computation(
                "Variance computation requires at least 2 samples",
            ));
        }

        // Sum of squares over centered data
        let sum_sq = match x_centered.as_slice() {
            Some(slice) => det_reduce(
                slice,
                slice.len() >= sum_f64_parallel_min_elems(),
                |block| block.iter().map(|v| v * v).sum::<f64>(),
                |a, b| a + b,
                0.0,
            ),
            // Non-contiguous storage: plain flat fold
            _ => x_centered.iter().map(|v| v * v).sum::<f64>(),
        };

        Ok(sum_sq / denom)
    }

    model_save_and_load_methods!(PCA);
}

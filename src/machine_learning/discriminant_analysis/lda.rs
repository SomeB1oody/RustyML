//! Linear Discriminant Analysis (LDA) for supervised dimensionality reduction and classification
//!
//! Contains the [`LDA`] model and its [`DiscriminantSolver`] and [`Shrinkage`] configuration
//! enums

use crate::error::Error;
use crate::math::matmul::{dot_par, matvec};
use crate::parallel_gates::scan_f64_parallel_min_elems;
use crate::{Deserialize, Serialize};
use ahash::{AHashMap, AHashSet};
use gemmkit_ndarray::{Bias, Parallelism, dot};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Numerical strategy for computing the per-class linear scoring coefficients used when scoring
/// with LDA
///
/// Selects the numerical method that derives the per-class linear scoring coefficients from the
/// shared covariance. The discriminant projection that `transform` uses does not depend on this
/// choice. It affects only the scoring path behind `predict`, `decision_function`, and
/// `predict_proba`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
pub enum DiscriminantSolver {
    /// Inverts the shared covariance through its SVD pseudo-inverse, then derives the per-class
    /// scoring coefficients from it
    #[default]
    SVD,
    /// Inverts the shared covariance through a symmetric eigendecomposition, then derives the
    /// per-class scoring coefficients from it
    Eigen,
    /// Solves each class scoring system `Sigma * coef = mu` directly with the iterative LSQR
    /// method (Paige and Saunders), forming no explicit inverse
    LSQR,
}

/// Returns the symmetric part `(m + m^T) / 2` of a square matrix
///
/// Callers apply this before a symmetric eigendecomposition, to correct any floating-point
/// asymmetry in covariance or scatter matrices
fn symmetric_part(m: &Array2<f64>) -> Array2<f64> {
    let mut out = m.to_owned();
    out.zip_mut_with(&m.t(), |a, &b| *a = (*a + b) * 0.5);
    out
}

impl DiscriminantSolver {
    /// Computes the per-class linear scoring coefficients `Sigma^-1 * mu_c` under this solver
    ///
    /// Returns an `(n_classes x n_features)` matrix whose row `c` is the coefficient vector for
    /// class `c`. `Eigen` and `SVD` form a symmetric inverse of the covariance and multiply it by
    /// the class means. `LSQR` solves each system `Sigma * coef_c = mu_c` directly with the
    /// iterative LSQR method, so it never materializes an explicit inverse
    fn scoring_coefficients(
        &self,
        cov: &Array2<f64>,
        means: &Array2<f64>,
    ) -> Result<Array2<f64>, Error> {
        match *self {
            DiscriminantSolver::LSQR => {
                let n_classes = means.nrows();
                let n_features = means.ncols();
                let max_iter = 4 * n_features + 100;
                let mut coefficients = Array2::<f64>::zeros((n_classes, n_features));
                for c in 0..n_classes {
                    let coef = lsqr_solve(cov, means.row(c), max_iter, 1e-12);
                    coefficients.row_mut(c).assign(&coef);
                }
                Ok(coefficients)
            }
            // The covariance inverse is symmetric, so `means . inv` row c equals `inv * mu_c`
            DiscriminantSolver::Eigen => Ok(dot(means, &Self::eigen_inverse(cov)?)),
            DiscriminantSolver::SVD => Ok(dot(means, &Self::svd_pseudo_inverse(cov)?)),
        }
    }

    /// Inverts a symmetric covariance through its eigendecomposition, zeroing eigenvalues that
    /// fall below a relative tolerance
    fn eigen_inverse(cov: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let n_features = cov.ncols();
        let eig = crate::machine_learning::linalg::symmetric_eigen(cov);

        let mut inv_vals = eig.eigenvalues.clone();
        let max_eval = inv_vals.iter().cloned().fold(0.0_f64, f64::max);
        let tol = (1e-12 * max_eval).max(1e-12);
        for v in inv_vals.iter_mut() {
            *v = if v.abs() > tol { 1.0 / *v } else { 0.0 };
        }

        // Symmetric inverse: U diag(1/lambda) U^T, formed by scaling each eigenvector column
        let mut scaled = eig.eigenvectors.clone();
        for j in 0..n_features {
            let s = inv_vals[j];
            for i in 0..n_features {
                scaled[[i, j]] *= s;
            }
        }
        Ok(dot(&scaled, &eig.eigenvectors.t()))
    }

    /// Inverts a symmetric covariance through its SVD pseudo-inverse
    fn svd_pseudo_inverse(cov: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let decomp = crate::machine_learning::linalg::svd(cov, true, true);
        let max_sv = decomp
            .singular_values
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let tol = (1e-12 * max_sv).max(1e-12);
        decomp.pseudo_inverse(tol)
    }

    /// Derives the LDA projection matrix by solving the generalized eigenproblem
    /// `S_b w = lambda * S_w w`. Here `cov` is the (shrunk or regularized) within-class
    /// covariance `S_w` and `sb` is the between-class scatter `S_b`
    ///
    /// A whitening transform keeps every step a symmetric eigendecomposition, which stays
    /// numerically stable. Write `S_w = U diag(d) U^T` and define the whitening
    /// `W = U diag(d^{-1/2})`. Then `A = W^T S_b W` is symmetric. Its eigenvectors `v` map back
    /// to the discriminant directions `w = W v`. This is the correct generalized-eigenvector
    /// solution
    ///
    /// Taking singular vectors of the non-symmetric `S_w^{-1} S_b` is not equivalent. Its left
    /// singular vectors are not the discriminant axes. The result does not depend on the
    /// `DiscriminantSolver`, which governs only the linear-scoring path behind `predict`,
    /// `decision_function`, and `predict_proba`
    fn project(
        cov: &Array2<f64>,
        sb: &Array2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, Error> {
        use crate::machine_learning::linalg::symmetric_eigen;

        let n_features = cov.nrows();

        // Eigendecompose the within-class covariance S_w (symmetrized defensively)
        let cov_sym = symmetric_part(cov);
        let cov_eig = symmetric_eigen(&cov_sym);

        // Whitening W = U diag(d^{-1/2})
        let max_d = cov_eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        let tol = (1e-12 * max_d).max(1e-12);
        let mut w_scale = cov_eig.eigenvectors.clone();
        for j in 0..n_features {
            let d_j = cov_eig.eigenvalues[j];
            let scale = if d_j > tol { 1.0 / d_j.sqrt() } else { 0.0 };
            for i in 0..n_features {
                w_scale[[i, j]] *= scale;
            }
        }

        // A = W^T S_b W is symmetric
        let sbw = dot(sb, &w_scale);
        let a = dot(&w_scale.t(), &sbw);
        let a_sym = symmetric_part(&a);
        let a_eig = symmetric_eigen(&a_sym);
        let directions = dot(&w_scale, &a_eig.eigenvectors);

        // Rank discriminant directions by eigenvalue (class separability), descending
        let mut order: Vec<usize> = (0..n_features).collect();
        order.sort_unstable_by(|&a_idx, &b_idx| {
            a_eig.eigenvalues[b_idx]
                .partial_cmp(&a_eig.eigenvalues[a_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Selects the top components, keeping each axis at the scale the whitening produced.
        // That scale gives the projected data unit within-class covariance, the convention
        // scikit-learn's `scalings_` follows. Rescaling the columns to unit L2 norm would undo
        // the whitening and put the projection on a different scale
        let selected: Vec<usize> = order.iter().take(n_components).copied().collect();

        // The degenerate check must be relative. These norms scale like the inverse square root
        // of the within-class variance, so they carry the data's units
        let max_norm = selected
            .iter()
            .map(|&idx| {
                let col = directions.column(idx);
                col.dot(&col).sqrt()
            })
            .fold(0.0_f64, f64::max);
        let norm_tol = 1e-12 * max_norm;

        let mut w = Array2::<f64>::zeros((n_features, n_components));
        for (component_idx, &idx) in selected.iter().enumerate() {
            let col = directions.column(idx);
            let norm = col.dot(&col).sqrt();
            if norm <= norm_tol {
                return Err(Error::computation(
                    "Discriminant direction norm too small for stable projection",
                ));
            }
            for i in 0..n_features {
                w[[i, component_idx]] = col[i];
            }
        }

        Ok(w)
    }
}

/// Solves `min ||a x - b||_2` with the iterative LSQR method of Paige and Saunders
///
/// `a` is the regularized within-class covariance, symmetric and positive definite, so the
/// iteration converges quickly. `b` is the right-hand side vector. The method runs a
/// Golub-Kahan bidiagonalization with a running Givens rotation. It never forms an explicit
/// inverse. `max_iter` caps the iteration count. `tol` is the relative residual threshold that
/// stops the iteration early
fn lsqr_solve(a: &Array2<f64>, b: ArrayView1<f64>, max_iter: usize, tol: f64) -> Array1<f64> {
    let n = a.ncols();
    let mut x = Array1::<f64>::zeros(n);

    // Start the bidiagonalization from the right-hand side
    let mut u = b.to_owned();
    let mut beta = u.dot(&u).sqrt();
    let b_norm = beta;
    if beta <= 0.0 {
        return x; // b = 0 gives x = 0
    }
    u.mapv_inplace(|v| v / beta);

    let mut v = matvec(&a.t(), &u, Parallelism::Rayon(0));
    let mut alpha = v.dot(&v).sqrt();
    if alpha <= 0.0 {
        return x; // a^T b = 0 leaves no descent direction
    }
    v.mapv_inplace(|val| val / alpha);

    let mut w = v.clone();
    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    for _ in 0..max_iter {
        // Advance the bidiagonalization to u_{k+1} and v_{k+1}
        let mut u_next = matvec(a, &v, Parallelism::Rayon(0));
        u_next.scaled_add(-alpha, &u);
        beta = u_next.dot(&u_next).sqrt();
        if beta > 0.0 {
            u_next.mapv_inplace(|val| val / beta);
        }

        let mut v_next = matvec(&a.t(), &u_next, Parallelism::Rayon(0));
        v_next.scaled_add(-beta, &v);
        alpha = v_next.dot(&v_next).sqrt();
        if alpha > 0.0 {
            v_next.mapv_inplace(|val| val / alpha);
        }

        // Givens rotation that eliminates beta
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let c = rho_bar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rho_bar = -c * alpha;
        let phi = c * phi_bar;
        phi_bar *= s;

        // Update the solution and the search direction
        x.scaled_add(phi / rho, &w);
        w.mapv_inplace(|val| val * (-theta / rho));
        w += &v_next;

        u = u_next;
        v = v_next;

        // phi_bar tracks the residual norm, so stop once it is negligible
        if phi_bar.abs() <= tol * b_norm || beta == 0.0 {
            break;
        }
    }

    x
}

/// Computes the Ledoit-Wolf optimal shrinkage intensity toward a scaled identity target
///
/// `sw` is the pooled within-class scatter. `n_samples` and `n_features` are the sample and
/// feature counts. `sum_z4` is the sum over all centered samples of `||z_k||^4`. The
/// maximum-likelihood covariance is `S = sw / n_samples`
///
/// Following Ledoit and Wolf (2004), the function returns `delta = b^2 / d^2` clamped to
/// `[0, 1]`. `d^2` is the dispersion of `S` around its identity target. `b^2` is the estimation
/// error of `S`. A return of `0` means no shrinkage, and a return of `1` means full shrinkage to
/// the target. The inner product used throughout is the trace inner product, normalized by the
/// feature count
fn ledoit_wolf_shrinkage(
    sw: &Array2<f64>,
    sum_z4: f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    let n = n_samples as f64;
    let p = n_features as f64;

    // Identity-target statistics of the maximum-likelihood covariance S = sw / n
    // Serial sums: the inputs are n_features x n_features, far below the parallel sum gate
    let sw_frob_sq: f64 = sw.iter().map(|&v| v * v).sum();
    let s_norm_sq = sw_frob_sq / (p * n * n); // ||S||^2
    let mu = sw.diag().sum() / (p * n); // <S, I> = trace(S) / p

    // d^2 = ||S - mu I||^2 reduces to ||S||^2 - mu^2
    let d2 = s_norm_sq - mu * mu;
    if d2 <= 0.0 {
        return 0.0;
    }

    // b^2 estimates how far the sample covariance scatters around S, capped by d^2
    let b_bar2 = sum_z4 / (p * n * n) - s_norm_sq / n;
    let b2 = b_bar2.clamp(0.0, d2);

    (b2 / d2).clamp(0.0, 1.0)
}

/// Shrinkage strategy for covariance estimation
///
/// Sets how much LDA shrinks the within-class covariance toward a scaled identity target, which
/// improves numerical stability
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Shrinkage {
    /// Ledoit-Wolf shrinkage intensity, estimated automatically from the training data
    Auto,
    /// Explicit shrinkage factor in the range `[0, 1]`
    Manual(f64),
}

/// Linear Discriminant Analysis (LDA) model
///
/// Provides supervised dimensionality reduction and classification by projecting
/// samples onto a lower-dimensional space that maximizes class separability
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::{LDA, Shrinkage, DiscriminantSolver};
///
/// let x = Array2::from_shape_vec(
///     (6, 2),
///     vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 5.0, 5.5, 4.5, 6.0, 5.0],
/// ).unwrap();
/// let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// let mut lda = LDA::new(1)
///     .unwrap()
///     .with_solver(DiscriminantSolver::SVD)
///     .with_shrinkage(Shrinkage::Manual(0.1))
///     .unwrap();
/// lda.fit(&x, &y).unwrap();
/// let _predictions = lda.predict(&x).unwrap();
/// let _x_transformed = lda.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDA {
    /// Number of components to keep after dimensionality reduction. `None` means "auto":
    /// the maximum possible, `min(n_classes - 1, n_features)`, resolved at fit time
    n_components: Option<usize>,
    /// Solver strategy for the LDA scoring computation
    solver: DiscriminantSolver,
    /// Shrinkage strategy for covariance estimation, or `None` for no shrinkage
    shrinkage: Option<Shrinkage>,
    /// Unique class labels from training data
    classes: Option<Array1<i32>>,
    /// Prior probabilities for each class
    priors: Option<Array1<f64>>,
    /// Mean vectors for each class
    means: Option<Array2<f64>>,
    /// Mean of the whole training matrix, scikit-learn's `xbar_`. [`LDA::transform`] subtracts it
    /// before projecting, which puts the projected coordinates on scikit-learn's scale
    ///
    /// Stored as the arithmetic row mean. That equals the prior-weighted average of the class
    /// means (`priors . means`), because the priors always equal `n_class / n_samples`
    overall_mean: Option<Array1<f64>>,
    /// Projection matrix for dimensionality reduction
    projection: Option<Array2<f64>>,
    /// Per-class linear discriminant scoring coefficients (`Sigma^-1 * mu_c`). Cached at fit
    /// time so `predict`, `decision_function`, and `predict_proba` do not recompute them
    coefficients: Option<Array2<f64>>,
    /// Per-class scoring intercepts (`-0.5 * mu_c . Sigma^-1 mu_c + ln prior_c`). Cached at fit
    /// time alongside the coefficients
    intercepts: Option<Array1<f64>>,
}

/// Default LDA configuration
///
/// # Default Values
///
/// - `n_components` - `None` (auto: `min(n_classes - 1, n_features)`, resolved at fit time)
/// - `solver` - `DiscriminantSolver::SVD`
/// - `shrinkage` - `None`
impl Default for LDA {
    fn default() -> Self {
        LDA {
            n_components: None,
            solver: DiscriminantSolver::SVD,
            shrinkage: None,
            classes: None,
            priors: None,
            means: None,
            overall_mean: None,
            projection: None,
            coefficients: None,
            intercepts: None,
        }
    }
}

impl LDA {
    /// Creates a new LDA instance with validated hyperparameters
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of components to keep (must be > 0)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new LDA instance or validation error
    ///
    /// # Notes
    ///
    /// The solver defaults to `DiscriminantSolver::SVD`, and no shrinkage applies by default.
    /// Override either with the builder methods below. `with_shrinkage` returns `Result` because
    /// it validates a manual shrinkage coefficient
    ///
    /// - [`with_solver`](Self::with_solver) - solver strategy for the LDA computation
    /// - [`with_shrinkage`](Self::with_shrinkage) - shrinkage strategy for covariance estimation
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n_components` is zero
    pub fn new(n_components: usize) -> Result<Self, Error> {
        if n_components == 0 {
            return Err(Error::invalid_parameter(
                "n_components",
                "must be greater than 0",
            ));
        }

        Ok(Self {
            n_components: Some(n_components),
            solver: DiscriminantSolver::SVD,
            shrinkage: None,
            classes: None,
            priors: None,
            means: None,
            overall_mean: None,
            projection: None,
            coefficients: None,
            intercepts: None,
        })
    }

    /// Sets the solver strategy for the LDA computation (default: `DiscriminantSolver::SVD`)
    ///
    /// # Parameters
    ///
    /// - `solver` - the solver to use
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_solver(mut self, solver: DiscriminantSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Sets the shrinkage strategy for covariance estimation (default: no shrinkage)
    ///
    /// # Parameters
    ///
    /// - `shrinkage` - the shrinkage strategy
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if a [`Shrinkage::Manual`] coefficient is outside `[0, 1]`
    ///   or not finite
    pub fn with_shrinkage(mut self, shrinkage: Shrinkage) -> Result<Self, Error> {
        if let Shrinkage::Manual(alpha) = shrinkage
            && (!alpha.is_finite() || !(0.0..=1.0).contains(&alpha))
        {
            return Err(Error::invalid_parameter(
                "shrinkage",
                format!("Manual(alpha) must be in [0, 1], got {}", alpha),
            ));
        }
        self.shrinkage = Some(shrinkage);
        Ok(self)
    }

    get_field!(get_n_components, n_components, Option<usize>);
    get_field!(get_solver, solver, DiscriminantSolver);
    get_field!(get_shrinkage, shrinkage, Option<Shrinkage>);
    get_field_as_ref!(get_classes, classes, Option<&Array1<i32>>);
    get_field_as_ref!(get_priors, priors, Option<&Array1<f64>>);
    get_field_as_ref!(get_means, means, Option<&Array2<f64>>);
    get_field_as_ref!(get_overall_mean, overall_mean, Option<&Array1<f64>>);
    get_field_as_ref!(get_projection, projection, Option<&Array2<f64>>);

    /// Fits the LDA model using training data
    ///
    /// Estimates class statistics, covariance, and projection matrices from labeled samples
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    /// - `y` - Class labels aligned with the rows of `x`
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - Mutable reference to self for chaining
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    /// - `Error::Computation` - If numerical computation fails during fitting
    ///
    /// # Performance
    ///
    /// Parallelizes the per-class statistics when the total work clears the calibrated
    /// scan-class gate (see `crate::parallel_gates`). The internal GEMMs gate themselves
    pub fn fit<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = i32>,
    {
        // Non-empty + finiteness checks on `x`
        crate::machine_learning::validation::preliminary_check(x, None)?;

        if x.nrows() != y.len() {
            return Err(Error::dimension_mismatch(x.nrows(), y.len()));
        }

        if x.ncols() == 0 {
            return Err(Error::empty_input("features"));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        // Decide execution mode from sample count
        let use_parallel = n_samples.saturating_mul(n_features) >= scan_f64_parallel_min_elems();

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                5,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input and extracting classes");
            Some(pb)
        };

        let mut classes_set = AHashSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }

        if classes_set.len() < 2 {
            return Err(Error::invalid_input(
                "At least two distinct classes are required",
            ));
        }

        let mut classes_vec: Vec<i32> = classes_set.into_iter().collect();
        classes_vec.sort_unstable();
        self.classes = Some(Array1::from_vec(classes_vec));
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        if n_samples <= n_classes {
            return Err(Error::invalid_input(format!(
                "Number of samples ({}) must be greater than number of classes ({})",
                n_samples, n_classes
            )));
        }

        // Resolve the number of components
        let max_components = (n_classes - 1).min(n_features);
        let n_components = match self.n_components {
            Some(requested) if requested > max_components => {
                return Err(Error::invalid_input(format!(
                    "n_components should be <= {}, got {}",
                    max_components, requested
                )));
            }
            Some(requested) => requested,
            None => max_components,
        };

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing class statistics and scatter matrices");
        }

        // Group row indices by class label
        let mut class_indices_map: AHashMap<i32, Vec<usize>> = AHashMap::with_capacity(n_classes);
        for &class in classes.iter() {
            class_indices_map.insert(class, Vec::new());
        }
        for (idx, &class) in y.iter().enumerate() {
            if let Some(indices) = class_indices_map.get_mut(&class) {
                indices.push(idx);
            }
        }
        for (&class, indices) in &class_indices_map {
            if indices.len() < 2 {
                return Err(Error::invalid_input(format!(
                    "Class {} has only {} sample(s). Each class must have at least 2 samples",
                    class,
                    indices.len()
                )));
            }
        }

        // Compute the overall mean for between-class scatter
        let overall_mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::computation("Error computing overall mean"))?;

        let class_pairs: Vec<_> = classes.iter().enumerate().collect();
        // Forces each class task's scatter GEMM serial once the class fan alone fills the pool
        // (`n_classes >= threads`)
        let scatter_par = if use_parallel && n_classes >= rayon::current_num_threads() {
            Parallelism::Serial
        } else {
            Parallelism::Rayon(0)
        };
        let class_results: Vec<_> = if use_parallel {
            // Compute per-class stats in parallel
            let x_owned = x.to_owned();
            class_pairs
                .par_iter()
                .map(|&(class_idx, &class)| {
                    let indices = &class_indices_map[&class];
                    let (prior, class_mean, class_sw, class_sb, class_z4) =
                        Self::compute_class_stats(
                            &x_owned,
                            indices,
                            &overall_mean,
                            n_samples,
                            scatter_par,
                        );
                    (class_idx, prior, class_mean, class_sw, class_sb, class_z4)
                })
                .collect()
        } else {
            // Compute per-class stats sequentially
            class_pairs
                .iter()
                .map(|&(class_idx, &class)| {
                    let indices = &class_indices_map[&class];
                    let (prior, class_mean, class_sw, class_sb, class_z4) =
                        Self::compute_class_stats(
                            x,
                            indices,
                            &overall_mean,
                            n_samples,
                            Parallelism::Rayon(0),
                        );
                    (class_idx, prior, class_mean, class_sw, class_sb, class_z4)
                })
                .collect()
        };

        let mut priors_vec = Vec::with_capacity(n_classes);
        let mut means_mat = Array2::<f64>::zeros((n_classes, n_features));
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        let mut sb = Array2::<f64>::zeros((n_features, n_features));
        // Sum of ||z_k||^4 over all centered samples, used by the Ledoit-Wolf shrinkage estimator
        let mut sum_z4 = 0.0;

        // Aggregate priors, class means, and scatter matrices
        for (class_idx, prior, class_mean, class_sw, class_sb, class_z4) in class_results {
            priors_vec.push(prior);
            means_mat.row_mut(class_idx).assign(&class_mean);
            sw += &class_sw;
            sb += &class_sb;
            sum_z4 += class_z4;
        }

        self.priors = Some(Array1::from_vec(priors_vec));
        self.means = Some(means_mat);

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Applying shrinkage and stabilizing covariance matrix");
        }

        // Estimate and stabilize the shared covariance
        let mut cov = sw.clone() / ((n_samples - n_classes) as f64);
        let alpha = self.shrinkage_alpha(&sw, sum_z4, n_samples, n_features);
        cov = Self::apply_shrinkage(&cov, alpha, n_features);
        self.regularize_covariance(&mut cov);

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Computing projection matrix");
        }

        // Build the discriminant projection by solving S_b w = lambda * S_w w
        let projection = DiscriminantSolver::project(&cov, &sb, n_components)?;
        self.projection = Some(projection);
        self.overall_mean = Some(overall_mean);

        // Cache the per-class linear-scoring parameters (Sigma^-1 * mu_c and the intercepts)
        {
            let means = self.means.as_ref().unwrap();
            let priors = self.priors.as_ref().unwrap();
            let coefficients = self.solver.scoring_coefficients(&cov, means)?;
            let mut intercepts = Array1::<f64>::zeros(n_classes);
            for j in 0..n_classes {
                let coef = coefficients.row(j);
                let prior_term = if priors[j] > 0.0 {
                    priors[j].ln()
                } else {
                    f64::NEG_INFINITY
                };
                intercepts[j] = -0.5 * means.row(j).dot(&coef) + prior_term;
            }
            self.coefficients = Some(coefficients);
            self.intercepts = Some(intercepts);
        }

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.set_message("Finalizing model state");
        }

        #[cfg(feature = "show_progress")]
        if let Some(pb) = &progress_bar {
            pb.inc(1);
            pb.finish_with_message("Completed");
        }

        Ok(self)
    }

    /// Predicts class labels for new samples using the trained model
    ///
    /// Scores each sample against every class with the cached linear coefficients, then returns
    /// the best-scoring class label per row
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - Predicted class labels
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    ///
    /// # Performance
    ///
    /// Scores through 1 parallel GEMM. The per-row label pick parallelizes when the total scan
    /// work clears the calibrated scan-class gate (see `crate::parallel_gates`)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64>,
    {
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LDA"))?;

        // Per-class discriminant scores. Columns align to `classes`
        let scores = self.decision_scores(x)?;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                x.nrows() as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Scoring samples");
            pb
        };

        let n_classes = classes.len();

        let predict_sample = |score_row: ArrayView1<f64>| {
            // Keeps the best-scoring label. Ties resolve to the lowest class index
            let mut best_score = f64::NEG_INFINITY;
            let mut best_class = classes[0];
            for j in 0..n_classes {
                if score_row[j] > best_score {
                    best_score = score_row[j];
                    best_class = classes[j];
                }
            }
            best_class
        };

        // Scan-class gate: n tasks, each an O(classes) best-score scan
        let scan_work = x.nrows().saturating_mul(n_classes);
        let predictions: Vec<i32> = if scan_work >= scan_f64_parallel_min_elems() {
            // Parallel label pick over the score rows
            #[cfg(feature = "show_progress")]
            let pb = progress_bar.clone();
            scores
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    let pred = predict_sample(row);
                    #[cfg(feature = "show_progress")]
                    pb.inc(1);
                    pred
                })
                .collect()
        } else {
            // Sequential label pick for smaller batches
            scores
                .outer_iter()
                .map(|row| {
                    let pred = predict_sample(row);
                    #[cfg(feature = "show_progress")]
                    progress_bar.inc(1);
                    pred
                })
                .collect()
        };

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Completed");
        Ok(Array1::from(predictions))
    }

    /// Computes the per-class linear discriminant scores for every sample
    ///
    /// Score `j` for a sample `x` is `delta_j(x) = x . (Sigma^-1 mu_j) - 0.5 mu_j . Sigma^-1 mu_j
    /// + ln(prior_j)`. The returned matrix has shape `(n_samples, n_classes)` with columns
    /// ordered to match [`get_classes`](Self::get_classes). This is the shared core of
    /// [`predict`](Self::predict), [`decision_function`](Self::decision_function), and
    /// [`predict_proba`](Self::predict_proba)
    fn decision_scores<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // `fit` precomputed these per-class scoring parameters
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LDA"))?;
        let intercepts = self
            .intercepts
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LDA"))?;

        let n_features = coefficients.ncols();
        crate::machine_learning::validation::validate_predict_input(x, n_features)?;

        // Computes `x . coefficients^T` plus the per-class intercept in one pass. The intercept
        // is constant down each column of the score matrix, so it rides the GEMM's fused
        // epilogue instead of a second broadcast sweep. The fused epilogue equals the product
        // followed by the same scalar add. `beta = 0` means the write never reads the freshly
        // allocated buffer. The intercepts are a contiguous owned array, so the bias slice below
        // is a plain borrow
        let mut scores = Array2::<f64>::zeros((x.nrows(), coefficients.nrows()));
        gemmkit_ndarray::gemm_fused(
            1.0,
            x,
            &coefficients.t(),
            0.0,
            &mut scores,
            Some(Bias::PerCol(
                intercepts
                    .as_slice()
                    .expect("LDA intercepts are stored contiguously"),
            )),
            None,
            Parallelism::Rayon(0),
        );
        Ok(scores)
    }

    /// Returns the per-class linear discriminant scores (the decision function)
    ///
    /// The result has shape `(n_samples, n_classes)`. Column `j` is the discriminant score for
    /// the class at index `j` of [`get_classes`](Self::get_classes). [`predict`](Self::predict)
    /// returns the per-row argmax of these scores
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - The `(n_samples, n_classes)` matrix of discriminant scores
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.decision_scores(x)
    }

    /// Returns the posterior class probabilities for each sample
    ///
    /// Under the LDA Gaussian model with a shared covariance, the posterior `P(class_j | x)` is
    /// the softmax of the discriminant scores from
    /// [`decision_function`](Self::decision_function). The result has shape
    /// `(n_samples, n_classes)`. Each row sums to 1, and the columns match the order of
    /// [`get_classes`](Self::get_classes). The softmax subtracts the per-row maximum before it
    /// exponentiates, for numerical stability
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - The `(n_samples, n_classes)` matrix of class posteriors
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    pub fn predict_proba<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let mut scores = self.decision_scores(x)?;
        // Row-wise numerically stable softmax
        for mut row in scores.outer_iter_mut() {
            let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - row_max).exp());
            let row_sum = row.sum();
            row.mapv_inplace(|v| v / row_sum);
        }
        Ok(scores)
    }

    /// Transforms data using the trained projection matrix
    ///
    /// Projects samples onto the learned discriminant components
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
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.transform_internal(x)
    }

    /// Fits the model and transforms the data in one step
    ///
    /// Trains the model, then returns the projected data
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    /// - `y` - Class labels aligned with the rows of `x`
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - Transformed feature matrix
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::InvalidInput` - If inputs
    ///   are empty, the shapes do not match, or values are not valid
    /// - `Error::Computation` - If numerical computation fails during fitting
    ///
    /// # Performance
    ///
    /// Parallelizes the per-class statistics when the total work clears the calibrated
    /// scan-class gate (see `crate::parallel_gates`). The internal GEMMs gate themselves
    pub fn fit_transform<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<Array2<f64>, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = i32>,
    {
        self.fit(x, y)?;
        self.transform_internal(x)
    }

    /// Transforms input data using the fitted projection
    fn transform_internal<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let projection = self
            .projection
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LDA"))?;

        crate::machine_learning::validation::validate_predict_input(x, projection.nrows())?;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                2,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Stage: {msg}",
            );
            pb.set_message("Validating input");
            pb
        };

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.set_message("Applying projection");
        }

        // scikit-learn projects `(X - xbar_) @ scalings_`. This materializes the centering
        // rather than folding it into the product as `X W - xbar W`. The 2 forms are equal in
        // real arithmetic. The folded form loses digits to cancellation when the data sits far
        // from the origin relative to its spread. Matching scikit-learn numerically is the
        // point here
        let overall_mean = self
            .overall_mean
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LDA"))?;
        let centered = x.to_owned() - overall_mean;
        let transformed = dot(&centered, projection);

        #[cfg(feature = "show_progress")]
        {
            progress_bar.inc(1);
            progress_bar.finish_with_message("Completed");
        }

        Ok(transformed)
    }

    /// Computes per-class statistics and scatter matrices
    ///
    /// The final tuple element is `sum_z4`, the sum over this class of `||x_i - mu_class||^4`,
    /// which the Ledoit-Wolf shrinkage estimator needs
    ///
    /// `scatter_par` is the parallelism the within-class scatter product runs with. Pass
    /// `Parallelism::Serial` when the per-class loop already fills the pool
    /// (`n_classes >= threads`), so each class task does not fork rayon again inside its
    /// scatter GEMM. Otherwise, `Parallelism::Rayon(0)` lets the backend spread it over the
    /// otherwise-idle cores
    fn compute_class_stats<S>(
        x: &ArrayBase<S, Ix2>,
        indices: &[usize],
        overall_mean: &Array1<f64>,
        n_samples: usize,
        scatter_par: Parallelism,
    ) -> (f64, Array1<f64>, Array2<f64>, Array2<f64>, f64)
    where
        S: Data<Elem = f64>,
    {
        let n_class = indices.len();
        let prior = n_class as f64 / n_samples as f64;

        let class_data = x.select(Axis(0), indices);
        let class_mean = class_data
            .mean_axis(Axis(0))
            .expect("Error computing class mean");

        let centered = &class_data - &class_mean;

        let class_sw = dot_par(&centered.t(), &centered, scatter_par);

        // Fourth power of each centered row norm, summed for the Ledoit-Wolf dispersion term
        let sum_z4 = centered
            .outer_iter()
            .map(|row| {
                let sq = row.dot(&row);
                sq * sq
            })
            .sum();

        // Between-class scatter from mean shift
        let mean_diff = &class_mean - overall_mean;
        let mean_diff_col = mean_diff.insert_axis(Axis(1));
        let class_sb = dot(&mean_diff_col, &mean_diff_col.t()) * (n_class as f64);

        (prior, class_mean, class_sw, class_sb, sum_z4)
    }

    /// Resolves the shrinkage intensity in `[0, 1]` for the configured strategy
    ///
    /// `Auto` uses the Ledoit-Wolf optimal estimator computed from the pooled within-class
    /// scatter `sw` and `sum_z4`
    fn shrinkage_alpha(
        &self,
        sw: &Array2<f64>,
        sum_z4: f64,
        n_samples: usize,
        n_features: usize,
    ) -> f64 {
        match self.shrinkage {
            None => 0.0,
            Some(Shrinkage::Manual(alpha)) => alpha,
            Some(Shrinkage::Auto) => ledoit_wolf_shrinkage(sw, sum_z4, n_samples, n_features),
        }
    }

    /// Shrinks the covariance toward a scaled identity target by the given intensity
    fn apply_shrinkage(cov: &Array2<f64>, alpha: f64, n_features: usize) -> Array2<f64> {
        if alpha <= 0.0 {
            return cov.clone();
        }

        let mut shrunk = cov.mapv(|v| v * (1.0 - alpha));
        let mu = cov.diag().sum() / n_features as f64;
        shrunk += &(Array2::<f64>::eye(n_features) * (alpha * mu));
        shrunk
    }

    /// Adds a small diagonal regularization term to the covariance matrix
    fn regularize_covariance(&self, cov: &mut Array2<f64>) {
        let n_features = cov.ncols().max(1);
        let trace = cov.diag().sum();
        let avg_var = if trace.is_finite() && trace > 0.0 {
            trace / n_features as f64
        } else {
            1.0
        };
        let regularization = avg_var * 1e-6;
        *cov += &(Array2::<f64>::eye(n_features) * regularization);
    }

    model_save_and_load_methods!(LDA);
}

/// Unit tests for the private numerical helpers that LDA fitting uses
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// LSQR recovers the exact solution of a small symmetric positive-definite system
    #[test]
    fn lsqr_solve_matches_direct_solution() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let x = lsqr_solve(&a, b.view(), 100, 1e-14);

        // A^-1 b works out to [1/11, 7/11]
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9, "x0 = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9, "x1 = {}", x[1]);

        // The residual A x - b must be negligible
        let r = a.dot(&x) - &b;
        assert!(r.dot(&r).sqrt() < 1e-9);
    }

    /// The Ledoit-Wolf intensity always lands inside the unit interval
    #[test]
    fn ledoit_wolf_shrinkage_in_unit_interval() {
        let sw = array![[2.0, 0.3], [0.3, 1.5]];
        let delta = ledoit_wolf_shrinkage(&sw, 6.0, 10, 2);
        assert!((0.0..=1.0).contains(&delta), "delta = {delta}");
    }
}

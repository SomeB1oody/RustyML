//! Support Vector Classifier (SVC)
//!
//! Provides the [`SVC`] binary classifier, trained with the Sequential Minimal
//! Optimization (SMO) algorithm and the kernel types re-exported as [`KernelType`](crate::machine_learning::KernelType)

use crate::error::Error;
use crate::machine_learning::parallel::map_collect;
use crate::machine_learning::validation::{
    preliminary_check, validate_max_iterations, validate_tolerance,
};
use crate::parallel_gates::SCAN_F64_PARALLEL_MIN_ELEMS;
pub use crate::types::KernelType;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::rngs::StdRng;

/// Sample-count threshold for switching SVC operations to parallel computation
///
/// Support Vector Machine Classifier
///
/// Support Vector Machines (SVM) are supervised learning methods used for classification, regression, and outlier detection. This implementation uses the Sequential Minimal Optimization (SMO) algorithm
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::{SVC, KernelType};
/// use ndarray::{Array2, Array1};
///
/// // Create training data
/// let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
/// let y_train = Array1::from_vec(vec![1.0, -1.0, -1.0, 1.0]);
///
/// // Initialize SVM classifier with RBF kernel
/// let mut svc = SVC::new(
///     KernelType::RBF { gamma: 0.5 },
///     1.0,  // regularization parameter
///     1e-3, // tolerance
///     100,  // max iterations
/// )
/// .expect("Failed to create SVC")
/// .with_random_state(42); // fixed seed for reproducible training
///
/// // Train the model
/// svc.fit(&x_train, &y_train).expect("Failed to train SVM");
///
/// // Make predictions
/// let x_test = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.8, 0.8]).unwrap();
/// let predictions = svc.predict(&x_test).expect("Failed to predict");
/// println!("Predictions: {:?}", predictions);
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SVC {
    /// Kernel function type that transforms input data to higher dimensions
    kernel: KernelType,
    /// Regularization parameter C, trading off margin width against classification error
    regularization_param: f64,
    /// Lagrange multipliers for the dual optimization problem
    alphas: Option<Array1<f64>>,
    /// Training samples that define the decision boundary
    support_vectors: Option<Array2<f64>>,
    /// Class labels corresponding to the support vectors
    support_vector_labels: Option<Array1<f64>>,
    /// Intercept term in the decision function
    bias: Option<f64>,
    /// Tolerance for the stopping criterion
    tol: f64,
    /// Maximum number of iterations for the optimization algorithm
    max_iter: usize,
    /// Small value for numerical stability in calculations
    eps: f64,
    /// Number of SMO iterations actually performed during the last fit
    n_iter: Option<usize>,
    /// Optional seed for the SMO working-set selection, enabling reproducible training
    random_state: Option<u64>,
}

impl Default for SVC {
    /// Creates an SVC instance with default parameters
    ///
    /// # Default Values
    ///
    /// - `kernel` - RBF (Radial Basis Function) with gamma=0.1
    /// - `regularization_param` - 1.0
    /// - `tol` - 0.001
    /// - `max_iter` - 1000
    /// - `eps` - 1e-8
    /// - `random_state` - None (non-deterministic working-set selection)
    fn default() -> Self {
        SVC {
            kernel: KernelType::RBF { gamma: 0.1 },
            regularization_param: 1.0,
            alphas: None,
            support_vectors: None,
            support_vector_labels: None,
            bias: None,
            tol: 0.001,
            max_iter: 1000,
            eps: 1e-8,
            n_iter: None,
            random_state: None,
        }
    }
}

impl SVC {
    /// Creates a new Support Vector Classifier (SVC) with specified parameters
    ///
    /// # Parameters
    ///
    /// - `kernel` - The kernel type to use for the algorithm
    /// - `regularization_param` - The regularization parameter (C) that trades off margin size and training error
    /// - `tol` - Tolerance for the stopping criterion
    /// - `max_iter` - Maximum number of iterations for the optimization algorithm
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SVC instance if parameters are valid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `regularization_param` is non-positive or non-finite, or if `tol` or `max_iter` fail validation
    ///
    /// # Notes
    ///
    /// Training is non-deterministic by default. For reproducible runs, set a fixed seed
    /// after construction with the builder method below:
    ///
    /// - [`with_random_state`](Self::with_random_state) - fixed seed for the SMO working-set selection
    ///
    /// ```
    /// use rustyml::machine_learning::{SVC, KernelType};
    ///
    /// let model = SVC::new(KernelType::RBF { gamma: 0.5 }, 1.0, 1e-3, 100)
    ///     .unwrap()
    ///     .with_random_state(42);
    /// ```
    pub fn new(
        kernel: KernelType,
        regularization_param: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<Self, Error> {
        // Validate regularization parameter
        if regularization_param <= 0.0 || !regularization_param.is_finite() {
            return Err(Error::invalid_parameter(
                "regularization_param",
                format!("must be positive and finite, got {}", regularization_param),
            ));
        }

        validate_tolerance(tol)?;
        validate_max_iterations(max_iter)?;

        Ok(SVC {
            kernel,
            regularization_param,
            alphas: None,
            support_vectors: None,
            support_vector_labels: None,
            bias: None,
            tol,
            max_iter,
            eps: 1e-8,
            n_iter: None,
            random_state: None,
        })
    }

    /// Sets a fixed RNG seed for the SMO working-set selection, making training
    /// reproducible (default: `None`, non-deterministic)
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the working-set selection RNG
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    // Getters
    get_field!(get_kernel, kernel, KernelType);
    get_field!(get_regularization_parameter, regularization_param, f64);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_epsilon, eps, f64);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field_as_ref!(get_alphas, alphas, Option<&Array1<f64>>);
    get_field_as_ref!(get_support_vectors, support_vectors, Option<&Array2<f64>>);
    get_field_as_ref!(
        get_support_vector_labels,
        support_vector_labels,
        Option<&Array1<f64>>
    );
    get_field!(get_bias, bias, Option<f64>);

    /// Computes the raw decision-function value for every row of `x` in one batched pass,
    /// returning one value per sample
    fn decision_values_batch<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        support_vectors: &Array2<f64>,
        alphas: &Array1<f64>,
        support_vector_labels: &Array1<f64>,
        bias: f64,
    ) -> Array1<f64>
    where
        S: Data<Elem = f64> + Sync,
    {
        let coef = alphas * support_vector_labels;
        let kernel_matrix = self.kernel.compute_matrix(x, support_vectors);
        let mut decision_values = kernel_matrix.dot(&coef);
        decision_values += bias;
        decision_values
    }

    /// Fits the SVC model to the training data
    ///
    /// Uses the Sequential Minimal Optimization (SMO) algorithm to find the optimal hyperplane
    ///
    /// # Parameters
    ///
    /// - `x` - Training data matrix where each row is a sample
    /// - `y` - Target labels (must be +1.0 or -1.0)
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - A mutable reference to the fitted model
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If input data is empty
    /// - `Error::InvalidInput` - If labels are not +1/-1
    /// - `Error::NotConverged` - If the model fails to converge and no support vectors are found
    /// - `Error::NonFinite` - If numerical instability produces non-finite values
    ///
    /// # Performance
    ///
    /// The kernel-matrix GEMM runs block-parallel above its FLOPs gate, and the error-cache
    /// initialization parallelizes above the calibrated scan-class gate; the SMO inner loops
    /// are sequential so the optimization trajectory is reproducible
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Basic input validation
        preliminary_check(x, Some(y))?;

        let (n_samples, n_features) = (x.nrows(), x.ncols());

        // Validate labels (SVC-specific)
        if !y.iter().all(|&yi| yi == 1.0 || yi == -1.0) {
            return Err(Error::invalid_input(
                "All labels must be either 1.0 or -1.0",
            ));
        }

        // Initialize optimization variables
        let mut alphas = Array1::<f64>::zeros(n_samples);
        let mut b = 0.0;

        // Symmetric Gram matrix via one GEMM + elementwise kernel transform
        let kernel_matrix = self.kernel.compute_matrix(x, x);

        if kernel_matrix.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("kernel matrix"));
        }

        // Initialize error cache
        let error_cache_parallel =
            n_samples.saturating_mul(n_samples) >= SCAN_F64_PARALLEL_MIN_ELEMS;
        let error_cache = map_collect(n_samples, error_cache_parallel, |i| {
            self.compute_error(i, &alphas, &kernel_matrix, y, b)
        });
        let mut error_cache = Array1::from(error_cache);

        // SMO main loop
        let mut num_changed_alphas;
        let mut examine_all = true;
        let mut iteration_count = 0;

        // RNG for SMO working-set selection
        let mut rng = crate::random::make_rng(self.random_state);

        // Progress bar for SMO iterations
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.max_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | {msg}",
            );
            pb.set_message("Alpha changes: 0 | Examine: All");
            pb
        };

        loop {
            if iteration_count >= self.max_iter {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message(
                    "Warning: Max iterations reached without full convergence",
                );
                break;
            }

            num_changed_alphas = 0;
            iteration_count += 1;
            #[cfg(feature = "show_progress")]
            progress_bar.inc(1);

            let sample_range: Vec<usize> = if examine_all {
                (0..n_samples).collect()
            } else {
                (0..n_samples)
                    .filter(|&i| alphas[i] > 0.0 && alphas[i] < self.regularization_param)
                    .collect()
            };

            for &i in &sample_range {
                num_changed_alphas += self.examine_example(
                    i,
                    &mut alphas,
                    &kernel_matrix,
                    y,
                    &mut b,
                    &mut error_cache,
                    &mut rng,
                );
            }

            #[cfg(feature = "show_progress")]
            progress_bar.set_message(format!(
                "Alpha changes: {} | Examine: {}",
                num_changed_alphas,
                if examine_all { "All" } else { "Non-bound" }
            ));

            // Alternate between examining all examples and only the non-bound ones
            if examine_all {
                examine_all = false;
            } else if num_changed_alphas == 0 {
                examine_all = true;
            }

            if !examine_all && num_changed_alphas == 0 {
                break;
            }
        }

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message(format!("Converged at iteration {}", iteration_count));

        let support_indices: Vec<usize> =
            (0..n_samples).filter(|&i| alphas[i] > self.eps).collect();

        if support_indices.is_empty() {
            return Err(Error::not_converged(
                "no support vectors found; try adjusting parameters",
            ));
        }

        if !b.is_finite() {
            return Err(Error::non_finite("bias term"));
        }

        let n_support_vectors = support_indices.len();
        let mut support_vectors = Array2::<f64>::zeros((n_support_vectors, n_features));
        let mut support_vector_labels = Array1::<f64>::zeros(n_support_vectors);
        let mut support_vector_alphas = Array1::<f64>::zeros(n_support_vectors);

        // Copy support vector data
        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_vector_labels[i] = y[idx];
            support_vector_alphas[i] = alphas[idx];
        }

        if support_vector_alphas.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("support vector alphas"));
        }

        // Store results
        self.alphas = Some(support_vector_alphas);
        self.support_vectors = Some(support_vectors);
        self.support_vector_labels = Some(support_vector_labels);
        self.bias = Some(b);
        self.n_iter = Some(iteration_count);

        Ok(self)
    }

    /// Predicts class labels for samples in X
    ///
    /// # Parameters
    ///
    /// - `x` - The input samples matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array containing predicted class labels (+1.0 or -1.0)
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model hasn't been fitted yet
    /// - `Error::EmptyInput` - If input data is empty
    /// - `Error::DimensionMismatch` - If feature dimensions mismatch
    /// - `Error::NonFinite` - If numerical issues produce a non-finite decision value during prediction
    ///
    /// # Performance
    ///
    /// The decision values come from one batched kernel-matrix GEMM (block-parallel above
    /// its FLOPs gate)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Ensure the model has been fitted
        let (support_vectors, support_vector_labels, alphas, bias) = match (
            &self.support_vectors,
            &self.support_vector_labels,
            &self.alphas,
            self.bias,
        ) {
            (Some(sv), Some(svl), Some(a), Some(b)) => (sv, svl, a, b),
            _ => return Err(Error::not_fitted("SVC")),
        };

        // Basic input validation
        preliminary_check(x, None)?;

        let n_features = x.ncols();

        if n_features != support_vectors.ncols() {
            return Err(Error::dimension_mismatch(
                support_vectors.ncols(),
                n_features,
            ));
        }

        let decision_values =
            self.decision_values_batch(x, support_vectors, alphas, support_vector_labels, bias);

        if decision_values.iter().any(|v| !v.is_finite()) {
            return Err(Error::non_finite("decision function during prediction"));
        }

        let predictions = decision_values.mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 });

        Ok(predictions)
    }

    /// Computes the decision function values for samples in X
    ///
    /// # Parameters
    ///
    /// - `x` - The input samples matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array containing the raw decision function values (distance to hyperplane)
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model hasn't been fitted yet
    /// - `Error::EmptyInput` - If input data is empty
    /// - `Error::DimensionMismatch` - If feature dimensions mismatch
    ///
    /// # Performance
    ///
    /// The decision values come from one batched kernel-matrix GEMM (block-parallel above
    /// its FLOPs gate)
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Ensure the model has been fitted
        let (support_vectors, support_vector_labels, alphas, bias) = match (
            &self.support_vectors,
            &self.support_vector_labels,
            &self.alphas,
            self.bias,
        ) {
            (Some(sv), Some(svl), Some(a), Some(b)) => (sv, svl, a, b),
            _ => return Err(Error::not_fitted("SVC")),
        };

        // Basic input validation
        preliminary_check(x, None)?;

        let n_features = x.ncols();

        if n_features != support_vectors.ncols() {
            return Err(Error::dimension_mismatch(
                support_vectors.ncols(),
                n_features,
            ));
        }

        let decision_values =
            self.decision_values_batch(x, support_vectors, alphas, support_vector_labels, bias);

        Ok(decision_values)
    }

    /// Examines an example for potential optimization as part of the SMO algorithm
    ///
    /// # Parameters
    ///
    /// - `i2` - Index of example to examine
    /// - `alphas` - Current alpha values
    /// - `kernel_matrix` - Pre-computed kernel matrix
    /// - `y` - Target labels
    /// - `b` - Current bias term
    /// - `error_cache` - Cached error values
    /// - `rng` - Random number generator used for the randomized working-set fallback
    ///
    /// # Returns
    ///
    /// - `usize` - Number of alpha values changed (0 or 1)
    #[allow(clippy::too_many_arguments)]
    fn examine_example<S>(
        &self,
        i2: usize,
        alphas: &mut Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: &ArrayBase<S, Ix1>,
        b: &mut f64,
        error_cache: &mut Array1<f64>,
        rng: &mut StdRng,
    ) -> usize
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        let y2 = y[i2];
        let alpha2 = alphas[i2];
        let e2 = error_cache[i2];
        let r2 = e2 * y2;

        // Check KKT conditions
        if (r2 < -self.tol && alpha2 < self.regularization_param) || (r2 > self.tol && alpha2 > 0.0)
        {
            // First try the second alpha that maximally violates the KKT conditions
            let mut i1 = self.select_second_alpha(i2, e2, alphas, error_cache);
            if i1 != i2 && self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache) {
                return 1;
            }

            // Try non-bound alphas, starting from a random offset
            let n_samples = alphas.len();
            let mut start = rng.random_range(0..n_samples);

            for _ in 0..n_samples {
                i1 = start;
                if alphas[i1] > 0.0
                    && alphas[i1] < self.regularization_param
                    && i1 != i2
                    && self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache)
                {
                    return 1;
                }
                start = (start + 1) % n_samples;
            }

            // Try all alphas, starting from a random offset
            start = rng.random_range(0..n_samples);
            for _ in 0..n_samples {
                i1 = start;
                if i1 != i2 && self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache) {
                    return 1;
                }
                start = (start + 1) % n_samples;
            }
        }

        0
    }

    /// Selects the second alpha for joint optimization in SMO
    ///
    /// # Parameters
    ///
    /// - `i2` - Index of the first alpha
    /// - `e2` - Error value for the first alpha
    /// - `alphas` - Current alpha values
    /// - `error_cache` - Cached error values
    ///
    /// # Returns
    ///
    /// - `usize` - Index of the selected second alpha
    fn select_second_alpha(
        &self,
        i2: usize,
        e2: f64,
        alphas: &Array1<f64>,
        error_cache: &Array1<f64>,
    ) -> usize {
        let n_samples = alphas.len();

        // Find the index with maximum |E1-E2|. Stays serial for the same reason as
        // `compute_error`: the kernel matrix bounds n far below any parallel scan gate
        let result = (0..n_samples)
            .filter(|&i| alphas[i] > 0.0 && alphas[i] < self.regularization_param)
            .map(|i| {
                let e1 = error_cache[i];
                let delta_e = (e1 - e2).abs();
                (i, delta_e)
            })
            .fold((i2, 0.0), |a, b| if b.1 > a.1 { b } else { a });

        // Return the index of the alpha that maximizes |E1-E2|
        result.0
    }

    /// Updates a pair of alpha values in the SMO algorithm
    ///
    /// # Parameters
    ///
    /// - `i1` - Index of first alpha to update
    /// - `i2` - Index of second alpha to update
    /// - `alphas` - Current alpha values
    /// - `kernel_matrix` - Pre-computed kernel matrix
    /// - `y` - Target labels
    /// - `b` - Current bias term (updated in place)
    /// - `error_cache` - Cached error values (updated in place)
    ///
    /// # Returns
    ///
    /// - `bool` - `true` if the alpha values were changed, `false` otherwise)
    #[allow(clippy::too_many_arguments)]
    fn take_step<S>(
        &self,
        i1: usize,
        i2: usize,
        alphas: &mut Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: &ArrayBase<S, Ix1>,
        b: &mut f64,
        error_cache: &mut Array1<f64>,
    ) -> bool
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        if i1 == i2 {
            return false;
        }

        let alpha1_old = alphas[i1];
        let alpha2_old = alphas[i2];
        let y1 = y[i1];
        let y2 = y[i2];
        let e1 = error_cache[i1];
        let e2 = error_cache[i2];
        let s = y1 * y2;

        // Compute the lower and upper bounds for alpha2
        let (l, h) = if y1 != y2 {
            (
                0.0f64.max(alpha2_old - alpha1_old),
                self.regularization_param
                    .min(self.regularization_param + alpha2_old - alpha1_old),
            )
        } else {
            (
                0.0f64.max(alpha1_old + alpha2_old - self.regularization_param),
                self.regularization_param.min(alpha1_old + alpha2_old),
            )
        };

        if l == h {
            return false;
        }

        let k11 = kernel_matrix[[i1, i1]];
        let k12 = kernel_matrix[[i1, i2]];
        let k22 = kernel_matrix[[i2, i2]];

        // Second derivative of the objective along the constraint line
        let eta = k11 + k22 - 2.0 * k12;

        let mut alpha2_new;
        if eta > 0.0 {
            // Standard case: unconstrained optimum, then clip to [l, h]
            alpha2_new = alpha2_old + y2 * (e1 - e2) / eta;
            if alpha2_new < l {
                alpha2_new = l;
            } else if alpha2_new > h {
                alpha2_new = h;
            }
        } else {
            // For eta <= 0, evaluate the objective at both endpoints and pick the larger
            let f1 = y1 * (e1 + *b) - alpha1_old * k11 - s * alpha2_old * k12;
            let f2 = y2 * (e2 + *b) - s * alpha1_old * k12 - alpha2_old * k22;
            let l1 = alpha1_old + s * (alpha2_old - l);
            let h1 = alpha1_old + s * (alpha2_old - h);
            let obj_l =
                l1 * f1 + l * f2 + 0.5 * l1 * l1 * k11 + 0.5 * l * l * k22 + s * l * l1 * k12;
            let obj_h =
                h1 * f1 + h * f2 + 0.5 * h1 * h1 * k11 + 0.5 * h * h * k22 + s * h * h1 * k12;

            if obj_l < obj_h - self.eps {
                alpha2_new = l;
            } else if obj_l > obj_h + self.eps {
                alpha2_new = h;
            } else {
                alpha2_new = alpha2_old;
            }
        }

        // Skip when the change in alpha2 is negligible
        if (alpha2_new - alpha2_old).abs() < self.eps * (alpha2_new + alpha2_old + self.eps) {
            return false;
        }

        let alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new);

        // Update bias under the u = sum + b convention (compute_error / compute_decision_value add + b)
        let b_old = *b;
        let b1 =
            *b - e1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12;
        let b2 =
            *b - e2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22;

        if alpha1_new > 0.0 && alpha1_new < self.regularization_param {
            *b = b1;
        } else if alpha2_new > 0.0 && alpha2_new < self.regularization_param {
            *b = b2;
        } else {
            *b = (b1 + b2) / 2.0;
        }

        // Update alpha values
        alphas[i1] = alpha1_new;
        alphas[i2] = alpha2_new;

        // Incremental O(n) error-cache update
        let coeff1 = y1 * (alpha1_new - alpha1_old);
        let coeff2 = y2 * (alpha2_new - alpha2_old);
        let delta_b = *b - b_old;
        let apply = |i: usize, e: &mut f64| {
            *e += coeff1 * kernel_matrix[[i1, i]] + coeff2 * kernel_matrix[[i2, i]] + delta_b;
        };
        error_cache
            .indexed_iter_mut()
            .for_each(|(i, e)| apply(i, e));

        true
    }

    /// Computes the prediction error `E_i = f(x_i) - y_i` for a training example
    ///
    /// This is the quantity cached in the SMO error cache, not the raw decision
    /// function `f(x_i)`, which omits the `- y_i` term
    ///
    /// # Parameters
    ///
    /// - `i` - Index of the example
    /// - `alphas` - Alpha values
    /// - `kernel_matrix` - Pre-computed kernel matrix
    /// - `y` - Target labels
    /// - `b` - Bias term
    ///
    /// # Returns
    ///
    /// - `f64` - The prediction error for example `i`
    fn compute_error<S>(
        &self,
        i: usize,
        alphas: &Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: &ArrayBase<S, Ix1>,
        b: f64,
    ) -> f64
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        let n_samples = alphas.len();

        // Decision-function sum over the non-zero alphas. Stays serial: the O(n^2) kernel
        // matrix bounds n far below the parallel sum gate (262_144 alphas would need a
        // ~550 GB kernel matrix)
        let sum: f64 = (0..n_samples)
            .filter(|&j| alphas[j] > 0.0)
            .map(|j| alphas[j] * y[j] * kernel_matrix[[i, j]])
            .sum();

        sum - y[i] + b
    }

    /// Fits the model to the training data and then predicts labels for the same data
    ///
    /// A convenience method that runs `fit` followed by `predict`
    ///
    /// # Parameters
    ///
    /// - `x` - Training data matrix where each row is a sample
    /// - `y` - Target labels (must be +1.0 or -1.0)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - Predicted class labels for the training data
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::InvalidInput` / `Error::InvalidParameter` - If input data is invalid
    /// - `Error::NotConverged` / `Error::NonFinite` / `Error::NotFitted` / `Error::DimensionMismatch` - If an error occurs during fitting or prediction
    pub fn fit_predict<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x, y)?;
        self.predict(x)
    }

    model_save_and_load_methods!(SVC);
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// predict() returns Error::NonFinite when finite input yields a non-finite decision value
    #[test]
    fn predict_non_finite_decision_value_returns_non_finite() {
        let svc = SVC {
            kernel: KernelType::Poly {
                degree: 400,
                gamma: 1.0,
                coef0: 1.0,
            },
            regularization_param: 1.0,
            alphas: Some(Array1::from_vec(vec![1.0])),
            support_vectors: Some(Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap()),
            support_vector_labels: Some(Array1::from_vec(vec![1.0])),
            bias: Some(0.0),
            tol: 1e-3,
            max_iter: 1000,
            eps: 1e-8,
            n_iter: Some(1),
            random_state: None,
        };

        // Finite input whose Poly-kernel value against the support vector overflows to +inf
        let x = Array2::from_shape_vec((1, 2), vec![40.0, 40.0]).unwrap();
        let result = svc.predict(&x);
        assert!(
            matches!(result, Err(Error::NonFinite(_))),
            "expected NonFinite for overflowing (finite-input) decision value, got {result:?}"
        );
    }
}

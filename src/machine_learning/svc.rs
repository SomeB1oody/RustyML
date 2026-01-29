use super::*;
pub use crate::KernelType;

/// Threshold for using parallel computation in SVC operations.
/// When the number of samples is below this threshold, sequential computation is used.
/// This avoids the overhead of thread spawning for small datasets.
const SVC_PARALLEL_THRESHOLD: usize = 100;

/// Support Vector Machine Classifier
///
/// Support Vector Machines (SVM) are a set of supervised learning methods used for classification, regression, and outlier detection. This implementation uses the Sequential Minimal Optimization (SMO) algorithm.
///
/// # Fields
///
/// - `kernel` - Kernel function type that transforms input data to higher dimensions
/// - `regularization_param` - Regularization parameter C, controls the trade-off between maximizing the margin and minimizing the classification error
/// - `alphas` - Lagrange multipliers for the dual optimization problem
/// - `support_vectors` - Training samples that define the decision boundary
/// - `support_vector_labels` - Class labels corresponding to the support vectors
/// - `bias` - Intercept term in the decision function
/// - `tol` - Tolerance for stopping criterion
/// - `max_iter` - Maximum number of iterations for the optimization algorithm
/// - `eps` - Small value for numerical stability in calculations
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::svc::{SVC, KernelType};
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
///     100   // max iterations
/// ).expect("Failed to create SVC");
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
    kernel: KernelType,
    regularization_param: f64,
    alphas: Option<Array1<f64>>,
    support_vectors: Option<Array2<f64>>,
    support_vector_labels: Option<Array1<f64>>,
    bias: Option<f64>,
    tol: f64,
    max_iter: usize,
    eps: f64,
    n_iter: Option<usize>,
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
    /// - `Result<Self, ModelError>` - A new SVC instance if parameters are valid
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `regularization_param` is non-positive or non-finite, or if `tol` or `max_iter` fail validation
    pub fn new(
        kernel: KernelType,
        regularization_param: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<Self, ModelError> {
        // Validate regularization parameter
        if regularization_param <= 0.0 || !regularization_param.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "Regularization parameter must be positive and finite, got {}",
                regularization_param
            )));
        }

        // Validate tolerance
        validate_tolerance(tol)?;

        // Validate maximum iterations
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
        })
    }

    // Getters
    get_field!(get_kernel, kernel, KernelType);
    get_field!(get_regularization_parameter, regularization_param, f64);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_epsilon, eps, f64);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_alphas, alphas, Option<&Array1<f64>>);
    get_field_as_ref!(get_support_vectors, support_vectors, Option<&Array2<f64>>);
    get_field_as_ref!(
        get_support_vector_labels,
        support_vector_labels,
        Option<&Array1<f64>>
    );
    get_field!(get_bias, bias, Option<f64>);

    /// Calculates the kernel function value between two vectors
    ///
    /// # Parameters
    ///
    /// - `x1` - First input vector
    /// - `x2` - Second input vector
    ///
    /// # Returns
    ///
    /// * `f64` - The kernel function value between the two input vectors
    fn kernel_function(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match self.kernel {
            KernelType::Linear => {
                // K(x, y) = x·y
                x1.dot(&x2)
            }
            KernelType::Poly {
                degree,
                gamma,
                coef0,
            } => {
                // K(x, y) = (gamma·x·y + coef0)^degree
                (gamma * x1.dot(&x2) + coef0).powf(degree as f64)
            }
            KernelType::RBF { gamma } => {
                // K(x, y) = exp(-gamma·|x-y|^2)
                let diff = &x1 - &x2;
                let squared_norm = diff.dot(&diff);
                (-gamma * squared_norm).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                // K(x, y) = tanh(gamma·x·y + coef0)
                (gamma * x1.dot(&x2) + coef0).tanh()
            }
        }
    }

    /// Computes the kernel matrix (Gram matrix) for the given data
    ///
    /// # Parameters
    ///
    /// * `x` - Input data matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// * `Array2<f64>` - The computed kernel matrix
    fn compute_kernel_matrix<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::<f64>::zeros((n_samples, n_samples));

        // Generate all (i,j) pairs where i <= j to compute only upper triangle + diagonal
        let pairs: Vec<(usize, usize)> = (0..n_samples)
            .flat_map(|i| (i..n_samples).map(move |j| (i, j)))
            .collect();

        // Compute kernel values (parallel for large datasets, sequential for small)
        let kernel_values: Vec<((usize, usize), f64)> = if n_samples >= SVC_PARALLEL_THRESHOLD {
            pairs
                .par_iter()
                .map(|&(i, j)| {
                    let k_val = self.kernel_function(x.row(i), x.row(j));
                    ((i, j), k_val)
                })
                .collect()
        } else {
            pairs
                .iter()
                .map(|&(i, j)| {
                    let k_val = self.kernel_function(x.row(i), x.row(j));
                    ((i, j), k_val)
                })
                .collect()
        };

        // Fill the matrix (including symmetric values)
        for ((i, j), val) in kernel_values {
            kernel_matrix[[i, j]] = val;
            if i != j {
                kernel_matrix[[j, i]] = val; // Symmetric
            }
        }

        kernel_matrix
    }

    /// Helper function to compute dual objective quadratic term
    ///
    /// # Parameters
    ///
    /// - `support_indices` - Indices of support vectors
    /// - `support_vector_alphas` - Alpha values for support vectors
    /// - `support_vector_labels` - Labels for support vectors
    /// - `kernel_matrix` - Pre-computed kernel matrix
    /// - `use_parallel` - Whether to use parallel computation
    ///
    /// # Returns
    ///
    /// * `f64` - The computed quadratic term value
    fn compute_quadratic_term(
        support_indices: &[usize],
        support_vector_alphas: &Array1<f64>,
        support_vector_labels: &Array1<f64>,
        kernel_matrix: &Array2<f64>,
        use_parallel: bool,
    ) -> f64 {
        let compute_fn = |(i, &idx_i): (usize, &usize)| {
            support_indices
                .iter()
                .enumerate()
                .map(|(j, &idx_j)| {
                    let kernel_val = kernel_matrix[[idx_i, idx_j]];
                    support_vector_alphas[i]
                        * support_vector_alphas[j]
                        * support_vector_labels[i]
                        * support_vector_labels[j]
                        * kernel_val
                })
                .sum::<f64>()
        };

        if use_parallel {
            support_indices.par_iter().enumerate().map(compute_fn).sum()
        } else {
            support_indices.iter().enumerate().map(compute_fn).sum()
        }
    }

    /// Helper function to compute decision value for a single sample
    ///
    /// # Parameters
    ///
    /// - `x_row` - Input sample
    /// - `support_vectors` - Support vector matrix
    /// - `alphas` - Alpha values
    /// - `support_vector_labels` - Support vector labels
    /// - `bias` - Bias term
    /// - `kernel_fn` - Kernel function closure
    ///
    /// # Returns
    ///
    /// * `f64` - The computed decision value
    fn compute_decision_value<F>(
        x_row: ArrayView1<f64>,
        support_vectors: &Array2<f64>,
        alphas: &Array1<f64>,
        support_vector_labels: &Array1<f64>,
        bias: f64,
        kernel_fn: F,
    ) -> f64
    where
        F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    {
        (0..support_vectors.nrows())
            .map(|j| {
                let kernel_val = kernel_fn(x_row, support_vectors.row(j));
                alphas[j] * support_vector_labels[j] * kernel_val
            })
            .sum::<f64>()
            + bias
    }

    /// Fits the SVC model to the training data
    ///
    /// Uses the Sequential Minimal Optimization (SMO) algorithm to find the optimal hyperplane.
    ///
    /// # Parameters
    ///
    /// - `x` - Training data matrix where each row is a sample
    /// - `y` - Target labels (must be +1.0 or -1.0)
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A mutable reference to the fitted model
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data is empty or labels are not +1/-1
    /// - `ModelError::ProcessingError` - If the model fails to converge, no support vectors are found, or numerical instability occurs
    ///
    /// # Performance
    ///
    /// Parallel computation is automatically enabled for various internal operations (kernel matrix calculation, error cache updates, etc.) when the number of samples exceeds 100.
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Use preliminary_check for basic input validation
        preliminary_check(x, Some(y))?;

        let (n_samples, n_features) = (x.nrows(), x.ncols());

        // Validate labels (SVC-specific requirement)
        let y_vec: Vec<f64> = y.to_vec();
        let label_check = if n_samples >= SVC_PARALLEL_THRESHOLD {
            y_vec.par_iter().all(|&yi| yi == 1.0 || yi == -1.0)
        } else {
            y_vec.iter().all(|&yi| yi == 1.0 || yi == -1.0)
        };
        if !label_check {
            return Err(ModelError::InputValidationError(
                "All labels must be either 1.0 or -1.0".to_string(),
            ));
        }

        // Initialize optimization variables
        let mut alphas = Array1::<f64>::zeros(n_samples);
        let mut b = 0.0;

        // Compute kernel matrix with error handling
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Validate kernel matrix
        let kernel_vec: Vec<f64> = kernel_matrix.iter().cloned().collect();
        let kernel_invalid = if n_samples >= SVC_PARALLEL_THRESHOLD {
            kernel_vec.par_iter().any(|&val| !val.is_finite())
        } else {
            kernel_vec.iter().any(|&val| !val.is_finite())
        };
        if kernel_invalid {
            return Err(ModelError::ProcessingError(
                "Kernel matrix contains invalid values - check kernel parameters".to_string(),
            ));
        }

        // Initialize error cache
        let error_cache: Vec<f64> = if n_samples >= SVC_PARALLEL_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .map(|i| self.decision_function_internal(i, &alphas, &kernel_matrix, y, b))
                .collect()
        } else {
            (0..n_samples)
                .map(|i| self.decision_function_internal(i, &alphas, &kernel_matrix, y, b))
                .collect()
        };
        let mut error_cache = Array1::from(error_cache);

        // SMO main loop with improved convergence tracking
        let mut num_changed_alphas;
        let mut examine_all = true;
        let mut iteration_count = 0;

        // Create progress bar for SMO iterations
        let progress_bar = ProgressBar::new(self.max_iter as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message("Alpha changes: 0 | Examine: All");

        loop {
            if iteration_count >= self.max_iter {
                progress_bar.finish_with_message(
                    "Warning: Max iterations reached without full convergence",
                );
                eprintln!(
                    "Warning: SVC reached maximum iterations ({}) without full convergence",
                    self.max_iter
                );
                break;
            }

            num_changed_alphas = 0;
            iteration_count += 1;
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
                );
            }

            // Update progress bar with current status
            let examine_mode = if examine_all { "All" } else { "Non-bound" };
            progress_bar.set_message(format!(
                "Alpha changes: {} | Examine: {}",
                num_changed_alphas, examine_mode
            ));

            // Update examination strategy
            if examine_all {
                examine_all = false;
            } else if num_changed_alphas == 0 {
                examine_all = true;
            }

            // Early termination check
            if !examine_all && num_changed_alphas == 0 {
                break;
            }
        }

        // Finish progress bar with convergence status
        progress_bar.finish_with_message(format!("Converged at iteration {}", iteration_count));

        // Extract support vectors
        let support_indices: Vec<usize> = if n_samples >= SVC_PARALLEL_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .filter_map(|i| if alphas[i] > self.eps { Some(i) } else { None })
                .collect()
        } else {
            (0..n_samples)
                .filter_map(|i| if alphas[i] > self.eps { Some(i) } else { None })
                .collect()
        };

        if support_indices.is_empty() {
            return Err(ModelError::ProcessingError(
                "No support vectors found - model failed to converge. Try adjusting parameters."
                    .to_string(),
            ));
        }

        // Validate bias term
        if !b.is_finite() {
            return Err(ModelError::ProcessingError(
                "Bias term is invalid - numerical instability detected".to_string(),
            ));
        }

        let n_support_vectors = support_indices.len();
        let mut support_vectors = Array2::<f64>::zeros((n_support_vectors, n_features));
        let mut support_vector_labels = Array1::<f64>::zeros(n_support_vectors);
        let mut support_vector_alphas = Array1::<f64>::zeros(n_support_vectors);

        // Efficiently copy support vector data
        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_vector_labels[i] = y[idx];
            support_vector_alphas[i] = alphas[idx];
        }

        // Final validation of extracted values
        let alphas_vec: Vec<f64> = support_vector_alphas.to_vec();
        let alphas_invalid = if support_indices.len() >= SVC_PARALLEL_THRESHOLD {
            alphas_vec.par_iter().any(|&val| !val.is_finite())
        } else {
            alphas_vec.iter().any(|&val| !val.is_finite())
        };
        if alphas_invalid {
            return Err(ModelError::ProcessingError(
                "Support vector alphas contain invalid values".to_string(),
            ));
        }

        // Calculate cost using margin-based objective function
        let cost = {
            // Calculate the primal objective function: 0.5 * ||w||^2 + C * sum(xi)
            // For SVM, we compute it as: 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)) - sum(alpha_i)

            // First term: 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
            let mut dual_objective = 0.0;

            // Compute the quadratic term
            let quadratic_term: f64 = Self::compute_quadratic_term(
                &support_indices,
                &support_vector_alphas,
                &support_vector_labels,
                &kernel_matrix,
                support_indices.len() >= SVC_PARALLEL_THRESHOLD,
            );

            dual_objective += 0.5 * quadratic_term;

            // Subtract the linear term: sum(alpha_i)
            let linear_term: f64 = support_vector_alphas.sum();
            dual_objective -= linear_term;

            // Return negative dual objective as cost (higher dual objective means better, so negation gives cost)
            -dual_objective
        };

        println!(
            "\nSVC training completed: {} samples, {} features, {} iterations, {} support vectors, final cost: {:.6}",
            n_samples, n_features, iteration_count, n_support_vectors, cost
        );

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
    /// - `Result<Array1<f64>, ModelError>` - A 1D array containing predicted class labels (+1.0 or -1.0)
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been fitted yet
    /// - `ModelError::InputValidationError` - If input data is invalid or feature dimensions mismatch
    /// - `ModelError::ProcessingError` - If numerical issues occur during decision function calculation
    ///
    /// # Performance
    ///
    /// Parallel computation is used for batch prediction when the number of samples exceeds 100.
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Check model fitting status
        let (support_vectors, support_vector_labels, alphas, bias) = match (
            &self.support_vectors,
            &self.support_vector_labels,
            &self.alphas,
            self.bias,
        ) {
            (Some(sv), Some(svl), Some(a), Some(b)) => (sv, svl, a, b),
            _ => return Err(ModelError::NotFitted),
        };

        // Basic input validation
        preliminary_check(x, None)?;

        let n_features = x.ncols();

        // Check feature dimension match
        if n_features != support_vectors.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "Input has {} features but model was trained on {} features",
                n_features,
                support_vectors.ncols()
            )));
        }

        let n_samples = x.nrows();

        // Compute predictions with improved error handling
        let compute_prediction = |i: usize| {
            let decision_value = Self::compute_decision_value(
                x.row(i),
                support_vectors,
                alphas,
                support_vector_labels,
                bias,
                |x1, x2| self.kernel_function(x1, x2),
            );

            // Handle numerical issues more robustly
            if !decision_value.is_finite() {
                Err(ModelError::ProcessingError(
                    "Decision function produced invalid value during prediction".to_string(),
                ))
            } else {
                Ok(if decision_value >= 0.0 { 1.0 } else { -1.0 })
            }
        };

        let prediction_results: Vec<Result<f64, ModelError>> =
            if n_samples >= SVC_PARALLEL_THRESHOLD {
                (0..n_samples)
                    .into_par_iter()
                    .map(compute_prediction)
                    .collect()
            } else {
                (0..n_samples).map(compute_prediction).collect()
            };

        // Check if any errors occurred during parallel computation
        let mut predictions = Vec::with_capacity(n_samples);
        for result in prediction_results {
            match result {
                Ok(pred) => predictions.push(pred),
                Err(e) => return Err(e),
            }
        }

        Ok(Array1::from(predictions))
    }

    /// Computes the decision function values for samples in X
    ///
    /// # Parameters
    ///
    /// - `x` - The input samples matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - A 1D array containing the raw decision function values (distance to hyperplane)
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been fitted yet
    /// - `ModelError::InputValidationError` - If input data is invalid or feature dimensions mismatch
    ///
    /// # Performance
    ///
    /// Parallel computation is used when the number of samples exceeds 100.
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Check model fitting status
        let (support_vectors, support_vector_labels, alphas, bias) = match (
            &self.support_vectors,
            &self.support_vector_labels,
            &self.alphas,
            self.bias,
        ) {
            (Some(sv), Some(svl), Some(a), Some(b)) => (sv, svl, a, b),
            _ => return Err(ModelError::NotFitted),
        };

        // Basic input validation
        preliminary_check(x, None)?;

        let n_features = x.ncols();

        // Check feature dimension match
        if n_features != support_vectors.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "Input has {} features but model was trained on {} features",
                n_features,
                support_vectors.ncols()
            )));
        }

        let n_samples = x.nrows();

        let mut decision_values = Array1::<f64>::zeros(n_samples);

        // Computation on each element of decision_values
        let compute_fn = |(i, mut val): (usize, ArrayViewMut0<f64>)| {
            let decision_val = Self::compute_decision_value(
                x.row(i),
                support_vectors,
                alphas,
                support_vector_labels,
                bias,
                |x1, x2| self.kernel_function(x1, x2),
            );
            val.fill(decision_val);
        };

        if n_samples >= SVC_PARALLEL_THRESHOLD {
            decision_values
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(compute_fn);
        } else {
            decision_values
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(compute_fn);
        }

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
    ///
    /// # Returns
    ///
    /// * `usize` - Number of alpha values changed (0 or 1)
    fn examine_example<S>(
        &self,
        i2: usize,
        alphas: &mut Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: &ArrayBase<S, Ix1>,
        b: &mut f64,
        error_cache: &mut Array1<f64>,
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
            // Find second alpha
            // First try the one that maximally violates KKT conditions
            let mut i1 = self.select_second_alpha(i2, e2, alphas, error_cache);
            if i1 != i2 && self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache) {
                return 1;
            }

            // Try non-bound alphas randomly
            let n_samples = alphas.len();
            let mut start = rand::random_range(0..n_samples);

            for _ in 0..n_samples {
                i1 = start;
                if alphas[i1] > 0.0 && alphas[i1] < self.regularization_param && i1 != i2 {
                    if self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache) {
                        return 1;
                    }
                }
                start = (start + 1) % n_samples;
            }

            // Try all alphas randomly
            start = rand::random_range(0..n_samples);
            for _ in 0..n_samples {
                i1 = start;
                if i1 != i2 {
                    if self.take_step(i1, i2, alphas, kernel_matrix, y, b, error_cache) {
                        return 1;
                    }
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
    /// * `usize` - Index of the selected second alpha
    fn select_second_alpha(
        &self,
        i2: usize,
        e2: f64,
        alphas: &Array1<f64>,
        error_cache: &Array1<f64>,
    ) -> usize {
        let n_samples = alphas.len();

        // Find the index with maximum |E1-E2|
        let result = if n_samples >= SVC_PARALLEL_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .filter(|&i| alphas[i] > 0.0 && alphas[i] < self.regularization_param)
                .map(|i| {
                    let e1 = error_cache[i];
                    let delta_e = (e1 - e2).abs();
                    (i, delta_e)
                })
                .reduce(
                    || (i2, 0.0), // Default to i2 if no better candidate is found
                    |a, b| if b.1 > a.1 { b } else { a },
                )
        } else {
            (0..n_samples)
                .filter(|&i| alphas[i] > 0.0 && alphas[i] < self.regularization_param)
                .map(|i| {
                    let e1 = error_cache[i];
                    let delta_e = (e1 - e2).abs();
                    (i, delta_e)
                })
                .fold((i2, 0.0), |a, b| if b.1 > a.1 { b } else { a })
        };

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
    /// * `bool` - `true` if the alpha values were changed, `false` otherwise
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

        // Calculate alpha boundaries
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

        // Calculate kernel values
        let k11 = kernel_matrix[[i1, i1]];
        let k12 = kernel_matrix[[i1, i2]];
        let k22 = kernel_matrix[[i2, i2]];

        // Calculate eta
        let eta = k11 + k22 - 2.0 * k12;

        let mut alpha2_new;
        if eta > 0.0 {
            // Standard case
            alpha2_new = alpha2_old + y2 * (e1 - e2) / eta;
            // Clip to boundaries
            if alpha2_new < l {
                alpha2_new = l;
            } else if alpha2_new > h {
                alpha2_new = h;
            }
        } else {
            // For eta <= 0 case, need to calculate objective function values at endpoints
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

        // Check for significant change
        if (alpha2_new - alpha2_old).abs() < self.eps * (alpha2_new + alpha2_old + self.eps) {
            return false;
        }

        // Calculate new value for alpha1
        let alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new);

        // Update bias
        let b1 =
            *b + e1 + y1 * (alpha1_new - alpha1_old) * k11 + y2 * (alpha2_new - alpha2_old) * k12;
        let b2 =
            *b + e2 + y1 * (alpha1_new - alpha1_old) * k12 + y2 * (alpha2_new - alpha2_old) * k22;

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

        // Update error cache
        self.update_error_cache(alphas, kernel_matrix, &y, *b, error_cache);

        true
    }

    /// Updates the error cache after changes to alpha values
    ///
    /// # Parameters
    ///
    /// - `alphas` - Current alpha values
    /// - `kernel_matrix` - Pre-computed kernel matrix
    /// - `y` - Target labels
    /// - `b` - Current bias term
    /// - `error_cache` - Error cache to update
    fn update_error_cache<S>(
        &self,
        alphas: &Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: &ArrayBase<S, Ix1>,
        b: f64,
        error_cache: &mut Array1<f64>,
    ) where
        S: Data<Elem = f64> + Send + Sync,
    {
        let n_samples = alphas.len();
        if n_samples >= SVC_PARALLEL_THRESHOLD {
            error_cache
                .indexed_iter_mut()
                .par_bridge()
                .for_each(|(i, error)| {
                    *error = self.decision_function_internal(i, alphas, kernel_matrix, y, b);
                });
        } else {
            error_cache.indexed_iter_mut().for_each(|(i, error)| {
                *error = self.decision_function_internal(i, alphas, kernel_matrix, y, b);
            });
        }
    }

    /// Calculates the decision function value for a single training example
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
    /// * `f64` - The decision function value
    fn decision_function_internal<S>(
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

        // Compute sum
        let sum: f64 = if n_samples >= SVC_PARALLEL_THRESHOLD {
            let indices: Vec<usize> = (0..n_samples).collect();
            indices
                .par_iter()
                .filter(|&&j| alphas[j] > 0.0) // Only consider non-zero alphas
                .map(|&j| alphas[j] * y[j] * kernel_matrix[[i, j]])
                .sum()
        } else {
            (0..n_samples)
                .filter(|&j| alphas[j] > 0.0) // Only consider non-zero alphas
                .map(|j| alphas[j] * y[j] * kernel_matrix[[i, j]])
                .sum()
        };

        sum - y[i] + b
    }

    model_save_and_load_methods!(SVC);
}

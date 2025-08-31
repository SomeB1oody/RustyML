pub use super::KernelType;
use crate::ModelError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// # Support Vector Machine Classifier
///
/// Support Vector Machines (SVM) are a set of supervised learning methods used for classification, regression, and outlier detection. This implementation uses the Sequential Minimal Optimization (SMO) algorithm.
///
/// ## Fields
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
/// ## Example
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
/// );
///
/// // Train the model
/// svc.fit(x_train.view(), y_train.view()).expect("Failed to train SVM");
///
/// // Make predictions
/// let x_test = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.8, 0.8]).unwrap();
/// let predictions = svc.predict(x_test.view()).expect("Failed to predict");
/// println!("Predictions: {:?}", predictions);
/// ```
#[derive(Debug, Clone)]
pub struct SVC {
    /// Kernel function type used to transform input data
    /// Common options include Linear, RBF, Polynomial, and Sigmoid
    kernel: KernelType,

    /// Regularization parameter C, controls the penalty for misclassification
    /// Smaller values specify stronger regularization, larger values focus on minimizing training error
    /// Typical range: 0.1 to 100
    regularization_param: f64,

    /// Lagrange multipliers from the dual form optimization problem
    /// Each non-zero alpha corresponds to a support vector
    alphas: Option<Array1<f64>>,

    /// Matrix of support vectors that define the decision boundary
    /// Shape: (n_support_vectors, n_features)
    support_vectors: Option<Array2<f64>>,

    /// Class labels (-1 or 1) for each support vector
    support_vector_labels: Option<Array1<f64>>,

    /// Bias term (intercept) in the decision function
    /// The hyperplane is defined as: w·x + bias = 0
    bias: Option<f64>,

    /// Convergence tolerance for stopping criterion
    /// The algorithm stops when KKT conditions are satisfied within this tolerance
    tol: f64,

    /// Maximum number of iterations for the optimization algorithm
    /// Prevents infinite loops in case of non-convergence
    max_iter: usize,

    /// Small value for numerical stability in calculations
    /// Helps prevent division by zero and other numerical issues
    eps: f64,

    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
}

impl Default for SVC {
    /// Creates an SVC instance with default parameters
    ///
    /// Default configuration:
    /// - Kernel function: RBF (Radial Basis Function) with gamma=0.1
    /// - Regularization parameter: 1.0
    /// - Convergence tolerance: 0.001
    /// - Maximum iterations: 1000
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
    /// * `Self` - A new SVC instance with the specified parameters
    pub fn new(kernel: KernelType, regularization_param: f64, tol: f64, max_iter: usize) -> Self {
        SVC {
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
        }
    }

    /// Returns the kernel type used by this SVC instance
    ///
    /// # Returns
    ///
    /// * `&KernelType` - A reference to the kernel type
    pub fn get_kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Returns the regularization parameter (C) used by this SVC instance
    ///
    /// # Returns
    ///
    /// * `f64` - The regularization parameter value
    pub fn get_regularization_param(&self) -> f64 {
        self.regularization_param
    }

    /// Returns the Lagrange multipliers (alphas) from the fitted model
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The array of Lagrange multipliers if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn get_alphas(&self) -> Result<&Array1<f64>, ModelError> {
        match &self.alphas {
            Some(alphas) => Ok(alphas),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the support vectors from the fitted model
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - The matrix of support vectors if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn get_support_vectors(&self) -> Result<&Array2<f64>, ModelError> {
        match &self.support_vectors {
            Some(support_vectors) => Ok(support_vectors),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the labels of the support vectors from the fitted model
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The array of support vector labels if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn get_support_vector_labels(&self) -> Result<&Array1<f64>, ModelError> {
        match &self.support_vector_labels {
            Some(support_vector_labels) => Ok(support_vector_labels),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the bias term (intercept) from the fitted model
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - The bias term if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn get_bias(&self) -> Result<f64, ModelError> {
        match self.bias {
            Some(bias) => Ok(bias),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the tolerance parameter used by this SVC instance
    ///
    /// # Returns
    ///
    /// * `f64` - The tolerance parameter value
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

    /// Returns the maximum number of iterations parameter used by this SVC instance
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of iterations
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the epsilon parameter used for numerical stability
    ///
    /// # Returns
    ///
    /// * `f64` - The epsilon value
    pub fn get_eps(&self) -> f64 {
        self.eps
    }

    /// Returns the epsilon parameter used for numerical stability
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` - number of iterations the algorithm ran for after fitting
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match &self.n_iter {
            Some(n_iter) => Ok(*n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

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
    fn compute_kernel_matrix(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute all values in parallel and collect results
        let values: Vec<((usize, usize), f64)> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let mut row_values = Vec::with_capacity(i + 1);
                for j in 0..=i {
                    let k_val = self.kernel_function(x.row(i), x.row(j));
                    row_values.push(((i, j), k_val));
                    if i != j {
                        row_values.push(((j, i), k_val)); // Add symmetric element
                    }
                }
                row_values
            })
            .collect();

        // Fill the matrix
        for ((i, j), val) in values {
            kernel_matrix[[i, j]] = val;
        }

        kernel_matrix
    }

    /// Fits the SVC model to the training data
    ///
    /// # Parameters
    ///
    /// - `x` - Training data matrix where each row is a sample
    /// - `y` - Target labels (should be +1 or -1)
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The fitted model (for method chaining)
    /// - `Err(ModelError)` - If there's an error during fitting
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        // Input validation checks
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "input data cannot be empty".to_string(),
            ));
        }

        if y.len() == 0 {
            return Err(ModelError::InputValidationError(
                "target labels cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(
                "x and y have different number of rows".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Ensure labels are +1 and -1
        if !y.iter().all(|&yi| yi == 1.0 || yi == -1.0) {
            return Err(ModelError::InputValidationError(
                "labels can only be either 1.0 or -1.0".to_string(),
            ));
        }

        if self.regularization_param <= 0.0 {
            return Err(ModelError::InputValidationError(
                "regularization parameter must be positive".to_string(),
            ));
        }

        if self.tol <= 0.0 {
            return Err(ModelError::InputValidationError(
                "tolerance must be positive".to_string(),
            ));
        }

        if self.max_iter <= 0 {
            return Err(ModelError::InputValidationError(
                "maximum number of iterations must be positive".to_string(),
            ));
        }

        // Initialize alpha and b
        let mut alphas = Array1::<f64>::zeros(n_samples);
        let mut b = 0.0;

        // Compute kernel matrix - this could potentially fail for certain kernel types
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Check if kernel matrix computation resulted in invalid values
        if kernel_matrix
            .iter()
            .any(|&val| val.is_nan() || val.is_infinite())
        {
            return Err(ModelError::ProcessingError(
                "kernel matrix contains invalid values (NaN or Infinity)".to_string(),
            ));
        }

        // Initialize error cache
        let mut error_cache = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            error_cache[i] = self.decision_function_internal(i, &alphas, &kernel_matrix, y, b);
        }

        // SMO main loop
        let mut iter = 0;
        let mut num_changed_alphas = 0;
        let mut examine_all = true;
        let mut n_iter = 0;

        while (iter < self.max_iter) && (num_changed_alphas > 0 || examine_all) {
            n_iter += 1;
            num_changed_alphas = 0;

            if examine_all {
                // Iterate through all samples
                for i in 0..n_samples {
                    num_changed_alphas += self.examine_example(
                        i,
                        &mut alphas,
                        &kernel_matrix,
                        y,
                        &mut b,
                        &mut error_cache,
                    );
                }
            } else {
                // Iterate through non-boundary alpha values
                for i in 0..n_samples {
                    if alphas[i] > 0.0 && alphas[i] < self.regularization_param {
                        num_changed_alphas += self.examine_example(
                            i,
                            &mut alphas,
                            &kernel_matrix,
                            y,
                            &mut b,
                            &mut error_cache,
                        );
                    }
                }
            }

            iter += 1;

            if examine_all {
                examine_all = false;
            } else if num_changed_alphas == 0 {
                examine_all = true;
            }
        }

        // Check for convergence - warn if max iterations reached
        if iter >= self.max_iter {
            eprintln!("Warning: SVC training reached maximum iterations without full convergence");
        }

        // Extract support vectors
        let mut support_vector_indices = Vec::new();
        for i in 0..n_samples {
            if alphas[i] > self.eps {
                support_vector_indices.push(i);
            }
        }

        let n_support_vectors = support_vector_indices.len();
        if n_support_vectors == 0 {
            return Err(ModelError::ProcessingError(
                "no support vectors found - model failed to converge".to_string(),
            ));
        }

        // Check for potential numerical instability
        if b.is_nan() || b.is_infinite() {
            return Err(ModelError::ProcessingError(
                "bias term contains invalid values (NaN or Infinity)".to_string(),
            ));
        }

        let mut support_vectors = Array2::<f64>::zeros((n_support_vectors, n_features));
        let mut support_vector_labels = Array1::<f64>::zeros(n_support_vectors);
        let mut support_vector_alphas = Array1::<f64>::zeros(n_support_vectors);

        for (i, &idx) in support_vector_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_vector_labels[i] = y[idx];
            support_vector_alphas[i] = alphas[idx];
        }

        // Final validation of extracted values
        if support_vector_alphas
            .iter()
            .any(|&val| val.is_nan() || val.is_infinite())
        {
            return Err(ModelError::ProcessingError(
                "support vector alphas contain invalid values".to_string(),
            ));
        }

        println!("SVC model training finished at iteration {}", n_iter);

        self.alphas = Some(support_vector_alphas);
        self.support_vectors = Some(support_vectors);
        self.support_vector_labels = Some(support_vector_labels);
        self.bias = Some(b);
        self.n_iter = Some(n_iter);

        Ok(self)
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
    fn examine_example(
        &self,
        i2: usize,
        alphas: &mut Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: ArrayView1<f64>,
        b: &mut f64,
        error_cache: &mut Array1<f64>,
    ) -> usize {
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
                i1 = (start + 1) % n_samples;
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
                i1 = (start + 1) % n_samples;
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

        // Find the index with maximum |E1-E2| in parallel
        let result = (0..n_samples)
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
            );

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
    fn take_step(
        &self,
        i1: usize,
        i2: usize,
        alphas: &mut Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: ArrayView1<f64>,
        b: &mut f64,
        error_cache: &mut Array1<f64>,
    ) -> bool {
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
        self.update_error_cache(alphas, kernel_matrix, y, *b, error_cache);

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
    fn update_error_cache(
        &self,
        alphas: &Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: ArrayView1<f64>,
        b: f64,
        error_cache: &mut Array1<f64>,
    ) {
        // Use zip_with_index for parallel updates
        error_cache
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|(i, error)| {
                *error = self.decision_function_internal(i, alphas, kernel_matrix, y, b);
            });
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
    fn decision_function_internal(
        &self,
        i: usize,
        alphas: &Array1<f64>,
        kernel_matrix: &Array2<f64>,
        y: ArrayView1<f64>,
        b: f64,
    ) -> f64 {
        // Create index range
        let indices: Vec<usize> = (0..alphas.len()).collect();

        // Compute sum in parallel
        let sum: f64 = indices
            .par_iter()
            .filter(|&&j| alphas[j] > 0.0) // Only consider non-zero alphas
            .map(|&j| alphas[j] * y[j] * kernel_matrix[[i, j]])
            .sum();

        sum - y[i] + b
    }

    /// Predicts class labels for samples in X
    ///
    /// # Parameters
    ///
    /// * `x` - The input samples, where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)` - The predicted class labels (+1 or -1)
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    /// - `Err(ModelError::InputValidationError)` - If input data is invalid
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        // Check if the model has been fitted
        if self.support_vectors.is_none()
            || self.support_vector_labels.is_none()
            || self.alphas.is_none()
        {
            return Err(ModelError::NotFitted);
        }

        let bias: f64 = match self.bias {
            Some(b) => b,
            None => return Err(ModelError::NotFitted),
        };

        // Input validation
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "input data cannot be empty".to_string(),
            ));
        }

        let support_vectors = self.support_vectors.as_ref().unwrap();

        // Check feature dimension consistency
        if x.ncols() != support_vectors.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "input features dimension {} does not match training features dimension {}",
                x.ncols(),
                support_vectors.ncols()
            )));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::<f64>::zeros(n_samples);

        let support_vector_labels = self.support_vector_labels.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();

        // Create an array of indices
        let indices: Vec<usize> = (0..n_samples).collect();

        // Calculate predictions in parallel
        let results: Vec<(usize, f64)> = indices
            .par_iter()
            .map(|&i| {
                // Compute the decision value
                let decision_value = (0..support_vectors.nrows())
                    .map(|j| {
                        let kernel_val = self.kernel_function(x.row(i), support_vectors.row(j));
                        alphas[j] * support_vector_labels[j] * kernel_val
                    })
                    .sum::<f64>()
                    + bias;

                // Check for numerical issues in decision value
                let prediction = if decision_value.is_nan() {
                    // Handle NaN case - could happen with certain kernel parameters
                    0.0 // or return an error, but this maintains robustness
                } else if decision_value >= 0.0 {
                    1.0
                } else {
                    -1.0
                };

                (i, prediction)
            })
            .collect();

        // Check if any predictions resulted in invalid values
        if results.iter().any(|(_, pred)| pred.is_nan()) {
            return Err(ModelError::ProcessingError(
                "prediction contains invalid values (NaN)".to_string(),
            ));
        }

        // Fill the predictions array with results
        for (i, pred) in results {
            predictions[i] = pred;
        }

        Ok(predictions)
    }

    /// Computes the decision function values for samples in X
    ///
    /// # Parameters
    ///
    /// * `x` - The input samples, where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)` - The decision function values
    /// - `Err(ModelError::NotFitted)` - If the model hasn't been fitted yet
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        // Check if the model has been fitted
        if self.support_vectors.is_none()
            || self.support_vector_labels.is_none()
            || self.alphas.is_none()
        {
            return Err(ModelError::NotFitted);
        }

        let bias: f64 = match self.bias {
            Some(b) => b,
            None => return Err(ModelError::NotFitted),
        };

        let n_samples = x.nrows();
        let mut decision_values = Array1::<f64>::zeros(n_samples);

        let support_vectors = self.support_vectors.as_ref().unwrap();
        let support_vector_labels = self.support_vector_labels.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();

        // Parallel computation on each element of decision_values
        decision_values
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut val)| {
                let x_i = x.row(i);
                let sum = (0..support_vectors.nrows())
                    .map(|j| {
                        alphas[j]
                            * support_vector_labels[j]
                            * self.kernel_function(x_i, support_vectors.row(j))
                    })
                    .sum::<f64>();

                *val.first_mut().unwrap() = sum + bias;
            });

        Ok(decision_values)
    }
}

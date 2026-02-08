pub use super::RegularizationType;
use super::helper_function::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_regulation_type,
    validate_tolerance,
};
use crate::error::ModelError;
use crate::math::sum_of_squared_errors;
use crate::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge,
    ParallelIterator,
};

/// Threshold for using parallel computation in Linear Regression.
/// When the number of samples or features is below this threshold, sequential computation is used.
const LINEAR_REGRESSION_PARALLEL_THRESHOLD: usize = 200;

/// Linear Regression model implementation
///
/// Trains a simple linear regression model using gradient descent algorithm. This implementation
/// supports multivariate regression, optional intercept term, and allows adjustment of learning rate,
/// maximum iterations, and convergence tolerance.
///
/// # Fields
///
/// - `coefficients` - Model coefficients (slopes), None before training
/// - `intercept` - Model intercept, None before training
/// - `fit_intercept` - Whether to include an intercept term in the model
/// - `learning_rate` - Learning rate for gradient descent
/// - `max_iter` - Maximum number of iterations for gradient descent
/// - `tol` - Convergence tolerance
/// - `n_iter` - Number of iterations the algorithm ran for after fitting
/// - `regularization_type` - Regularization type and strength
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::linear_regression::*;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();
///
/// // Prepare training data
/// let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let raw_y = vec![6.0, 9.0, 12.0];
///
/// // Convert Vec to ndarray types
/// let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
/// let y = Array1::from_vec(raw_y);
///
/// // Train the model
/// model.fit(&x, &y).unwrap();
///
/// // Make predictions
/// let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
/// let predictions = model.predict(&new_data);
///
/// // Save the trained model to a file
/// model.save_to_path("linear_regression_model.json").unwrap();
///
/// // Load the model from the file
/// let loaded_model = LinearRegression::load_from_path("linear_regression_model.json").unwrap();
///
/// // Use the loaded model for predictions
/// let loaded_predictions = loaded_model.predict(&new_data);
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
///
/// // Clean up: remove the created file
/// std::fs::remove_file("linear_regression_model.json").unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    fit_intercept: bool,
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
    n_iter: Option<usize>,
    regularization_type: Option<RegularizationType>,
}

impl Default for LinearRegression {
    /// Creates a new LinearRegression instance with default parameter values.
    ///
    /// # Default Values
    ///
    /// - `coefficients` - `None` - Model coefficients are not initialized until training
    /// - `intercept` - `None` - Model intercept is not initialized until training
    /// - `fit_intercept` - `true` - Include an intercept term in the linear model
    /// - `learning_rate` - `0.01` - Learning rate for gradient descent optimization
    /// - `max_iter` - `1000` - Maximum number of iterations for gradient descent
    /// - `tol` - `1e-5` - Convergence tolerance (0.00001) for stopping criteria
    /// - `n_iter` - `None` - Number of actual iterations performed (set after training)
    /// - `regularization_type` - `None` - No regularization applied by default
    ///
    /// # Returns
    ///
    /// - `LinearRegression` - A new instance with sensible default parameters for most use cases
    fn default() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-5,
            n_iter: None,
            regularization_type: None,
        }
    }
}

impl LinearRegression {
    /// Creates a new linear regression model with custom parameters.
    ///
    /// This constructor validates all input parameters to ensure they are within
    /// acceptable numerical ranges before returning the model instance.
    ///
    /// # Parameters
    ///
    /// - `fit_intercept` - Whether to calculate the intercept for this model
    /// - `learning_rate` - The learning rate for gradient descent optimization
    /// - `max_iterations` - Maximum number of iterations for gradient descent
    /// - `tolerance` - The tolerance for stopping criteria
    /// - `regularization_type` - Optional regularization to prevent overfitting (L1, L2, or None)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new instance of LinearRegression, or an error if parameters are invalid
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If learning_rate, max_iterations or tolerance is invalid
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        regularization_type: Option<RegularizationType>,
    ) -> Result<Self, ModelError> {
        // Input validation
        validate_learning_rate(learning_rate)?;
        validate_max_iterations(max_iterations)?;
        validate_tolerance(tolerance)?;
        validate_regulation_type(regularization_type)?;

        Ok(LinearRegression {
            coefficients: None,
            intercept: None,
            fit_intercept,
            learning_rate,
            max_iter: max_iterations,
            tol: tolerance,
            n_iter: None,
            regularization_type,
        })
    }

    // Getters
    get_field!(get_fit_intercept, fit_intercept, bool);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_max_iter, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(
        get_regularization_type,
        regularization_type,
        Option<RegularizationType>
    );
    get_field_as_ref!(get_coefficients, coefficients, Option<&Array1<f64>>);
    get_field!(get_intercept, intercept, Option<f64>);

    /// Fits the linear regression model using gradient descent.
    ///
    /// Iteratively updates the model's coefficients and intercept to minimize the cost function.
    /// The process includes real-time progress tracking and early stopping if convergence is reached.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix, each row is a sample, each column is a feature
    /// - `y` - Target variable vector
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - Returns mutable reference to self for method chaining
    ///
    /// # Errors
    ///
    /// - `ModelError::ProcessingError` - If numerical issues like NaN or infinity occur during training
    /// - `ModelError::InputValidationError` - If input dimensions are inconsistent
    ///
    /// # Performance
    ///
    /// Parallel computation is automatically used for L1 regularization and gradient updates
    /// when the number of features exceeds `LINEAR_REGRESSION_PARALLEL_THRESHOLD` (200).
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Use preliminary_check for input validation
        preliminary_check(x, Some(y))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize parameters
        let mut weights = Array1::<f64>::zeros(n_features); // Initialize weights to zero
        let mut intercept = 0.0; // Initialize intercept to zero

        let mut prev_cost = f64::INFINITY;
        let mut convergence_count = 0; // Track consecutive convergences for stability
        const CONVERGENCE_THRESHOLD: usize = 3; // Require 3 consecutive convergences

        let mut n_iter = 0;

        // Pre-allocate arrays to avoid repeated memory allocation
        let mut predictions = Array1::<f64>::zeros(n_samples);
        let mut error_vec = Array1::<f64>::zeros(n_samples);

        // Create progress bar for training iterations
        let progress_bar = ProgressBar::new(self.max_iter as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Cost: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message(format!(
            "{:.6} | Convergence: 0/{}",
            f64::INFINITY,
            CONVERGENCE_THRESHOLD
        ));

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            // Calculate predictions - vectorized operation
            predictions.assign(&x.dot(&weights));
            if self.fit_intercept {
                predictions += intercept;
            }

            // Calculate errors - use borrowing to avoid moving predictions
            error_vec.assign(&(&predictions - y));

            // Calculate cost using math module's sum_of_squared_errors
            let sse = sum_of_squared_errors(&predictions, &y);

            let regularization_term = match &self.regularization_type {
                None => 0.0,
                Some(RegularizationType::L1(alpha)) => {
                    // Use parallel computation for L1 regularization when feature count is large
                    if n_features >= LINEAR_REGRESSION_PARALLEL_THRESHOLD {
                        alpha * weights.iter().par_bridge().map(|w| w.abs()).sum::<f64>()
                    } else {
                        alpha * weights.iter().map(|w| w.abs()).sum::<f64>()
                    }
                }
                Some(RegularizationType::L2(alpha)) => alpha * weights.dot(&weights),
            };

            let cost = sse / (2.0 * n_samples as f64) + regularization_term;

            // Update progress bar with current cost and convergence status
            progress_bar.set_message(format!(
                "{:.6} | Convergence: {}/{}",
                cost, convergence_count, CONVERGENCE_THRESHOLD
            ));
            progress_bar.inc(1);

            // Check for numerical issues in cost
            if !cost.is_finite() {
                progress_bar.finish_with_message("Error: NaN or infinite cost");
                return Err(ModelError::ProcessingError(
                    "Cost calculation resulted in NaN or infinite value".to_string(),
                ));
            }

            // Calculate gradients using matrix operations
            let mut weight_gradients = x.t().dot(&error_vec) / (n_samples as f64);
            let intercept_gradient = if self.fit_intercept {
                error_vec.sum() / (n_samples as f64)
            } else {
                0.0
            };

            // Check for numerical issues in gradients
            if weight_gradients.iter().any(|&val| !val.is_finite())
                || !intercept_gradient.is_finite()
            {
                progress_bar.finish_with_message("Error: NaN or infinite gradients");
                return Err(ModelError::ProcessingError(
                    "Gradient calculation resulted in NaN or infinite values".to_string(),
                ));
            }

            // Add regularization terms to gradients
            match &self.regularization_type {
                None => {}
                Some(RegularizationType::L1(alpha)) => {
                    // Use parallel computation for L1 gradient when feature count is large
                    let alpha_val = *alpha;
                    if n_features >= LINEAR_REGRESSION_PARALLEL_THRESHOLD {
                        let weights_slice = weights.as_slice().unwrap();
                        let gradients_slice = weight_gradients.as_slice_mut().unwrap();

                        gradients_slice
                            .par_iter_mut()
                            .zip(weights_slice.par_iter())
                            .for_each(|(grad, w)| {
                                *grad += alpha_val * w.signum();
                            });
                    } else {
                        weight_gradients
                            .iter_mut()
                            .zip(weights.iter())
                            .for_each(|(grad, w)| {
                                *grad += alpha_val * w.signum();
                            });
                    }
                }
                Some(RegularizationType::L2(alpha)) => {
                    weight_gradients.scaled_add(*alpha, &weights);
                }
            }

            // Update parameters
            weights.scaled_add(-self.learning_rate, &weight_gradients);
            if self.fit_intercept {
                intercept -= self.learning_rate * intercept_gradient;
            }

            // Check for numerical issues in updated parameters
            if weights.iter().any(|&val| !val.is_finite()) || !intercept.is_finite() {
                progress_bar.finish_with_message("Error: NaN or infinite parameters");
                return Err(ModelError::ProcessingError(
                    "Parameter update resulted in NaN or infinite values".to_string(),
                ));
            }

            // Enhanced convergence check with stability requirement
            let cost_change = (prev_cost - cost).abs();
            if cost_change < self.tol {
                convergence_count += 1;
                if convergence_count >= CONVERGENCE_THRESHOLD {
                    break;
                }
            } else {
                convergence_count = 0; // Reset if not converged
            }

            prev_cost = cost;
        }

        // Finish progress bar with final statistics
        let convergence_status = if n_iter < self.max_iter {
            "Converged"
        } else {
            "Max iterations"
        };
        progress_bar.finish_with_message(format!(
            "{:.6} | {} | Iterations: {}",
            prev_cost, convergence_status, n_iter
        ));

        // Save training results
        self.coefficients = Some(weights);
        self.intercept = Some(if self.fit_intercept { intercept } else { 0.0 });
        self.n_iter = Some(n_iter);

        println!(
            "\nLinear Regression training completed: {} samples, {} features, {} iterations, final cost: {:.6}",
            n_samples, n_features, n_iter, prev_cost
        );

        Ok(self)
    }

    /// Makes predictions using the trained model.
    ///
    /// Applies the learned coefficients and intercept to the provided feature matrix.
    ///
    /// # Parameters
    ///
    /// - `x` - Prediction data, each row is a sample, each column is a feature
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - A vector of predictions
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been trained yet
    /// - `ModelError::InputValidationError` - If features count doesn't match or data contains invalid values
    /// - `ModelError::ProcessingError` - If prediction results in non-finite values
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted
        if self.coefficients.is_none() {
            return Err(ModelError::NotFitted);
        }

        let coeffs = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        // Check for empty input data
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        // Check feature dimension match
        if x.ncols() != coeffs.len() {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, coefficients: {}",
                x.ncols(),
                coeffs.len()
            )));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        // Calculate predictions using matrix operations
        let mut predictions = x.dot(coeffs);
        if self.fit_intercept {
            predictions += intercept;
        }

        // Check if predictions contain invalid values
        if predictions.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::ProcessingError(
                "Prediction calculation resulted in NaN or infinite values".to_string(),
            ));
        }

        Ok(predictions)
    }

    /// Fits the model to the training data and then makes predictions on the same data.
    ///
    /// A convenience method that sequentially executes `fit` and then `predict`.
    ///
    /// # Parameters
    ///
    /// - `x` - The input features matrix
    /// - `y` - The target values corresponding to each training example
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - The predicted values for the input data
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data is invalid
    /// - `ModelError::ProcessingError` - If an error occurs during fitting or prediction
    pub fn fit_predict<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.fit(x, y)?;
        Ok(self.predict(x)?)
    }

    model_save_and_load_methods!(LinearRegression);
}

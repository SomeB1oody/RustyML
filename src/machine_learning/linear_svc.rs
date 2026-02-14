pub use super::RegularizationType;
use super::helper_function::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_tolerance,
};
use crate::error::ModelError;
use crate::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, s};
use ndarray_rand::rand::{rng, seq::SliceRandom};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Threshold for batch size above which parallel processing is used
const LINEAR_SVC_PARALLEL_THRESHOLD: usize = 200;

/// Linear Support Vector Classifier (LinearSVC)
///
/// Implements a classifier similar to sklearn's LinearSVC, trained using the hinge loss function.
/// Supports L1 and L2 regularization for preventing overfitting.
///
/// # Fields
///
/// - `weights` - Weight coefficients for each feature
/// - `bias` - Bias term (intercept) of the model
/// - `max_iter` - Maximum number of iterations for the optimizer
/// - `learning_rate` - Learning rate (step size) for gradient descent
/// - `penalty` - Regularization type (L1 or L2) with strength parameter
/// - `fit_intercept` - Whether to calculate and use an intercept/bias term
/// - `tol` - Training convergence tolerance
/// - `n_iter` - Number of iterations that were actually performed during training
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::linear_svc::*;
///
/// // Create model with custom parameters
/// let mut model = LinearSVC::new(
///     1000,                           // max_iter
///     0.001,                          // learning_rate
///     RegularizationType::L2(1.0),    // penalty type with regularization strength
///     true,                           // fit_intercept
///     1e-4                            // tolerance
/// ).unwrap();
///
/// let x = Array2::from_shape_vec((8, 2), vec![
///         1.0, 2.0,
///         2.0, 3.0,
///         3.0, 4.0,
///         4.0, 5.0,
///         5.0, 1.0,
///         6.0, 2.0,
///         7.0, 3.0,
///         8.0, 4.0,
///     ]).unwrap();
///
/// let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
///
/// model.fit(&x, &y).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinearSVC {
    weights: Option<Array1<f64>>,
    bias: Option<f64>,
    max_iter: usize,
    learning_rate: f64,
    penalty: RegularizationType,
    fit_intercept: bool,
    tol: f64,
    n_iter: Option<usize>,
}

impl Default for LinearSVC {
    /// Creates a new LinearSVC with default parameters
    ///
    /// # Default values
    ///
    /// - `max_iter`: 1000
    /// - `learning_rate`: 0.001
    /// - `penalty`: RegularizationType::L2(1.0) - L2 regularization with strength 1.0
    /// - `fit_intercept`: true
    /// - `tol`: 1e-4
    ///
    /// # Returns
    ///
    /// - `Self` - A new LinearSVC instance with default parameters
    fn default() -> Self {
        LinearSVC {
            weights: None,
            bias: None,
            max_iter: 1000,
            learning_rate: 0.001,
            penalty: RegularizationType::L2(1.0),
            fit_intercept: true,
            tol: 1e-4,
            n_iter: None,
        }
    }
}

impl LinearSVC {
    /// Creates a new LinearSVC instance with custom parameters.
    ///
    /// # Parameters
    ///
    /// - `max_iter` - Maximum number of iterations for the optimizer
    /// - `learning_rate` - Step size for gradient descent updates
    /// - `penalty` - Type and strength of regularization (L1(lambda) or L2(lambda))
    /// - `fit_intercept` - Whether to calculate and use bias term
    /// - `tol` - Convergence tolerance that stops training when reached
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A Result containing the new LinearSVC instance if validation passes
    ///
    /// # Errors
    ///
    /// Returns `ModelError::InputValidationError` if:
    /// - `max_iter` is 0
    /// - `learning_rate` is not positive or not finite
    /// - `penalty` regularization parameter is negative or not finite
    /// - `tol` is not positive or not finite
    pub fn new(
        max_iter: usize,
        learning_rate: f64,
        penalty: RegularizationType,
        fit_intercept: bool,
        tol: f64,
    ) -> Result<Self, ModelError> {
        // Validate parameters
        validate_max_iterations(max_iter)?;
        validate_learning_rate(learning_rate)?;
        validate_tolerance(tol)?;

        // Validate regularization parameter
        let reg_param = match penalty {
            RegularizationType::L1(lambda) | RegularizationType::L2(lambda) => lambda,
        };
        if reg_param < 0.0 || !reg_param.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "Regularization parameter must be non-negative and finite, got {}",
                reg_param
            )));
        }

        Ok(LinearSVC {
            weights: None,
            bias: None,
            max_iter,
            learning_rate,
            penalty,
            fit_intercept,
            tol,
            n_iter: None,
        })
    }

    /// Validates input data dimensions and values
    fn validate_input_data<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check for empty input
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Check for NaN/Inf values in input
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input features contain NaN or infinite values".to_string(),
            ));
        }

        // Check feature dimension mismatch if model is trained
        if let Some(ref weights) = self.weights {
            if x.ncols() != weights.len() {
                return Err(ModelError::InputValidationError(format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    weights.len(),
                    x.ncols()
                )));
            }
        }

        Ok(())
    }

    /// Checks if weights contain invalid values (NaN or infinite)
    fn check_weights_validity(weights: &Array1<f64>, bias: f64) -> Result<(), ModelError> {
        if weights.iter().any(|&w| !w.is_finite()) || !bias.is_finite() {
            return Err(ModelError::ProcessingError(
                "Weights became NaN or infinite during training. Try reducing learning_rate or regularization_param".to_string()
            ));
        }
        Ok(())
    }

    /// Calculates the optimal batch size based on dataset size
    fn calculate_batch_size(n_samples: usize) -> usize {
        const MIN_BATCH_SIZE: usize = 32;
        const MAX_BATCH_SIZE: usize = 512;

        std::cmp::max(
            MIN_BATCH_SIZE,
            std::cmp::min(MAX_BATCH_SIZE, n_samples / 10),
        )
    }

    // Getters
    get_field!(get_fit_intercept, fit_intercept, bool);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_max_iter, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_weights, weights, Option<&Array1<f64>>);
    get_field!(get_bias, bias, Option<f64>);
    get_field!(get_penalty, penalty, RegularizationType);

    /// Trains the model on the provided data.
    ///
    /// Uses stochastic gradient descent to optimize the hinge loss function.
    /// The model will continue training until either maximum iterations are reached
    /// or convergence is detected based on tolerance.
    ///
    /// # Parameters
    ///
    /// - `x` - Input features as a 2D array where each row is a sample and each column is a feature
    /// - `y` - Target values as a 1D array (should contain only 0.0 and 1.0 values)
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A mutable reference to the trained model
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data is invalid or feature dimension mismatches
    /// - `ModelError::ProcessingError` - If numerical issues occur during training (e.g., weights become NaN)
    ///
    /// # Performance
    ///
    /// Parallel processing is automatically enabled when the batch size exceeds `LINEAR_SVC_PARALLEL_THRESHOLD` (200).
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Use preliminary_check for input validation
        preliminary_check(x, Some(y))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize weights and bias
        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;

        // Convert labels to -1 and 1
        let y_binary = y.mapv(|v| if v <= 0.0 { -1.0 } else { 1.0 });

        // Create index array for random sampling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rng();

        let mut prev_weights = weights.clone();
        let mut prev_bias = bias;

        let mut n_iter = 0;

        // Calculate optimal batch size
        let batch_size = Self::calculate_batch_size(n_samples);

        // Define cost calculation closure
        let calculate_cost = |x: &ArrayBase<S, Ix2>,
                              y: &Array1<f64>,
                              weights: &Array1<f64>,
                              bias: f64,
                              penalty: &RegularizationType|
         -> f64 {
            let n_samples = x.nrows() as f64;

            // Calculate hinge loss
            let hinge_loss: f64 = x
                .outer_iter()
                .zip(y.iter())
                .map(|(xi, &yi)| {
                    let margin = xi.dot(weights) + bias;
                    (1.0 - yi * margin).max(0.0)
                })
                .sum::<f64>()
                / n_samples;

            // Calculate regularization term
            let regularization_term = match penalty {
                RegularizationType::L2(lambda) => {
                    lambda * weights.iter().map(|&w| w * w).sum::<f64>() / 2.0
                }
                RegularizationType::L1(lambda) => {
                    lambda * weights.iter().map(|&w| w.abs()).sum::<f64>()
                }
            };

            hinge_loss + regularization_term
        };

        // Create progress bar for training iterations
        let progress_bar = ProgressBar::new(self.max_iter as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Cost: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message("Initializing...");

        while n_iter < self.max_iter {
            n_iter += 1;
            progress_bar.inc(1);

            // Randomly shuffle indices
            indices.shuffle(&mut rng);

            // Split data into batches
            for batch_indices in indices.chunks(batch_size) {
                let batch_len = batch_indices.len() as f64;

                // Gradient computation closure
                let compute_gradient = |&idx: &usize| {
                    let xi = x.slice(s![idx, ..]);
                    let yi = y_binary[idx];
                    let margin = xi.dot(&weights) + bias;

                    // Hinge loss gradient: only contribute if margin violation occurs
                    if yi * margin < 1.0 {
                        let weight_grad = xi.to_owned() * yi;
                        let bias_grad = yi;
                        (weight_grad, bias_grad)
                    } else {
                        (Array1::zeros(n_features), 0.0)
                    }
                };

                // Use parallel or sequential processing based on batch size
                let (weight_grad_sum, bias_grad_sum) =
                    if batch_indices.len() >= LINEAR_SVC_PARALLEL_THRESHOLD {
                        batch_indices.par_iter().map(compute_gradient).reduce(
                            || (Array1::zeros(n_features), 0.0),
                            |mut acc, (w_grad, b_grad)| {
                                acc.0 = &acc.0 + &w_grad;
                                acc.1 += b_grad;
                                acc
                            },
                        )
                    } else {
                        batch_indices.iter().map(compute_gradient).fold(
                            (Array1::zeros(n_features), 0.0),
                            |mut acc, (w_grad, b_grad)| {
                                acc.0 = &acc.0 + &w_grad;
                                acc.1 += b_grad;
                                acc
                            },
                        )
                    };

                // Update weights with hinge loss gradient
                weights = &weights + &(weight_grad_sum * (self.learning_rate / batch_len));

                // Apply regularization
                match self.penalty {
                    RegularizationType::L2(lambda) => {
                        // L2 regularization: gradient is lambda * weights
                        weights = &weights * (1.0 - self.learning_rate * lambda);
                    }
                    RegularizationType::L1(lambda) => {
                        // L1 regularization: subgradient update
                        let l1_grad = weights.mapv(|w| {
                            if w > 0.0 {
                                1.0
                            } else if w < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        });
                        weights = &weights - &(l1_grad * (self.learning_rate * lambda));
                    }
                }

                if self.fit_intercept {
                    bias += self.learning_rate * bias_grad_sum / batch_len;
                }

                // Check for weight explosion early
                Self::check_weights_validity(&weights, bias)?;
            }

            // Convergence check: compute mean squared difference
            let weight_diff = (&weights - &prev_weights)
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                / n_features as f64;

            let bias_diff = if self.fit_intercept {
                (bias - prev_bias).powi(2)
            } else {
                0.0
            };

            let total_diff = (weight_diff + bias_diff).sqrt();

            // Calculate and display current cost
            if n_iter % 10 == 0 || total_diff < self.tol {
                let current_cost = calculate_cost(x, &y_binary, &weights, bias, &self.penalty);
                progress_bar.set_message(format!("{:.6}", current_cost));
            }

            if total_diff < self.tol {
                break;
            }

            prev_weights.assign(&weights);
            prev_bias = bias;
        }

        // Calculate final cost
        let final_cost = calculate_cost(x, &y_binary, &weights, bias, &self.penalty);

        // Finish progress bar with final statistics
        let convergence_status = if n_iter < self.max_iter {
            "Converged"
        } else {
            "Max iterations"
        };
        progress_bar.finish_with_message(format!(
            "{:.6} | {} | Iterations: {}",
            final_cost, convergence_status, n_iter
        ));

        println!(
            "\nLinear SVC training completed: {} samples, {} features, {} iterations, final cost: {:.6}",
            n_samples, n_features, n_iter, final_cost
        );

        self.weights = Some(weights);
        self.bias = Some(bias);
        self.n_iter = Some(n_iter);

        Ok(self)
    }

    /// Predicts the class for each sample in the provided data.
    ///
    /// # Parameters
    ///
    /// - `x` - Input features as a 2D array where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - Array of predicted class labels (0.0 or 1.0) for each sample
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet
    /// - `ModelError::InputValidationError` - If input data is invalid or feature dimension mismatches
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted
        if self.weights.is_none() {
            return Err(ModelError::NotFitted);
        }

        // Validate input data
        self.validate_input_data(x)?;

        // Get decision values and convert to predictions
        let decision = self.decision_function(x)?;
        Ok(decision.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
    }

    /// Calculates the decision function values for each sample.
    ///
    /// This method provides raw scores representing the distance to the decision hyperplane.
    /// Positive values indicate class 1, negative values indicate class 0.
    ///
    /// # Parameters
    ///
    /// - `x` - Input features as a 2D array where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - Raw decision scores for each sample
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet
    /// - `ModelError::InputValidationError` - If input data is invalid or feature dimension mismatches
    /// - `ModelError::ProcessingError` - If the computation produced NaN or infinite values
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted
        let weights = match self.get_weights() {
            Some(weights) => weights,
            None => {
                return Err(ModelError::NotFitted);
            }
        };
        let bias = self.bias.unwrap_or(0.0);

        // Validate input data
        self.validate_input_data(x)?;

        let decision = x.dot(weights) + bias;

        // Check for NaN/Inf in decision values
        if decision.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::ProcessingError(
                "Decision function produced NaN or infinite values".to_string(),
            ));
        }

        Ok(decision)
    }

    model_save_and_load_methods!(LinearSVC);
}

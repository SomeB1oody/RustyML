pub use super::RegularizationType;
use super::*;

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
/// - `regularization_param` - Regularization strength parameter
/// - `penalty` - Regularization type: L1 or L2
/// - `fit_intercept` - Whether to calculate and use an intercept/bias term
/// - `tol` - Training convergence tolerance
/// - `n_iter` - Number of iterations that were actually performed during training
///
/// # Features
///
/// - Binary classification
/// - Stochastic gradient descent optimization
/// - L1 or L2 regularization
/// - Configurable convergence tolerance
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::linear_svc::*;
/// use rustyml::utility::train_test_split::train_test_split;
///
/// // Create model with custom parameters
/// let mut model = LinearSVC::new(
///     1000,                // max_iter
///     0.001,               // learning_rate
///     1.0,                 // regularization_param
///     RegularizationType::L2(0.0),     // penalty type, number here means nothing
///     true,                // fit_intercept
///     1e-4                 // tolerance
/// );
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
/// let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.25), Some(42)).unwrap();
///
/// model.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(x_test.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearSVC {
    weights: Option<Array1<f64>>,
    bias: Option<f64>,
    max_iter: usize,
    learning_rate: f64,
    regularization_param: f64,
    penalty: RegularizationType,
    fit_intercept: bool,
    tol: f64,
    n_iter: Option<usize>,
}

/// Creates a new LinearSVC with default parameters
///
/// # Default values
///
/// - `weights`: None (not trained)
/// - `bias`: None (not trained)
/// - `max_iter`: 1000
/// - `learning_rate`: 0.001
/// - `regularization_param`: 1.0
/// - `penalty`: PenaltyType::L2
/// - `fit_intercept`: true
/// - `tol`: 1e-4
/// - `n_iter`: None (not trained)
///
/// # Returns
///
/// - `LinearSVC` - A new LinearSVC instance with default parameters
impl Default for LinearSVC {
    fn default() -> Self {
        LinearSVC {
            weights: None,
            bias: None,
            max_iter: 1000,
            learning_rate: 0.001,
            regularization_param: 1.0,
            penalty: RegularizationType::L2(0.0), // the number here means nothing
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
    /// - `max_iter`: Maximum number of iterations for the optimizer
    /// - `learning_rate`: Step size for gradient descent updates
    /// - `regularization_param`: Strength of regularization (higher = stronger)
    /// - `penalty`: Type of regularization (L1 or L2)
    /// - `fit_intercept`: Whether to calculate and use bias term
    /// - `tol`: Convergence tolerance that stops training when reached
    ///
    /// # Returns
    ///
    /// * `Self` - A new LinearSVC instance with specified parameters
    pub fn new(
        max_iter: usize,
        learning_rate: f64,
        regularization_param: f64,
        penalty: RegularizationType,
        fit_intercept: bool,
        tol: f64,
    ) -> Self {
        LinearSVC {
            weights: None,
            bias: None,
            max_iter,
            learning_rate,
            regularization_param,
            penalty,
            fit_intercept,
            tol,
            n_iter: None,
        }
    }

    /// Validates input parameters for training
    fn validate_training_params(&self) -> Result<(), ModelError> {
        if self.max_iter == 0 {
            return Err(ModelError::InputValidationError(
                "max_iter must be greater than 0".to_string(),
            ));
        }

        if self.learning_rate <= 0.0 {
            return Err(ModelError::InputValidationError(format!(
                "learning_rate must be greater than 0.0, got {}",
                self.learning_rate
            )));
        }

        if self.regularization_param <= 0.0 {
            return Err(ModelError::InputValidationError(format!(
                "regularization_param must be greater than 0.0, got {}",
                self.regularization_param
            )));
        }

        if self.tol <= 0.0 {
            return Err(ModelError::InputValidationError(format!(
                "tol must be greater than 0.0, got {}",
                self.tol
            )));
        }

        Ok(())
    }

    /// Validates input data dimensions and values
    fn validate_input_data(&self, x: ArrayView2<f64>) -> Result<(), ModelError> {
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
    get_field_as_ref!(get_weights, weights, &Option<Array1<f64>>);
    get_field!(get_bias, bias, Option<f64>);
    get_field!(get_regularization_parameter, regularization_param, f64);
    get_field!(get_penalty, penalty, RegularizationType);

    /// Trains the model on the provided data.
    ///
    /// Uses stochastic gradient descent to optimize the hinge loss function.
    /// The model will continue training until either:
    /// - Maximum iterations are reached
    /// - Convergence is detected based on tolerance
    ///
    /// # Parameters
    ///
    /// - `x`: Input features as a 2D array where each row is a sample and each column is a feature
    /// - `y`: Target values as a 1D array (should contain only 0.0 and 1.0 values)
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)`: Reference to self if training succeeds
    /// - `Err(ModelError)`: Error if validation fails or training encounters problems
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        // Input shape validation
        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(format!(
                "Input data size mismatch: x.shape={}, y.shape={}",
                x.nrows(),
                y.len()
            )));
        }

        // Validate training parameters
        self.validate_training_params()?;
        self.validate_input_data(x)?;

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

        while n_iter < self.max_iter {
            n_iter += 1;
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

                    if yi * margin < 1.0 {
                        let weight_grad = xi.to_owned() * yi;
                        let bias_grad = if self.fit_intercept { yi } else { 0.0 };
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

                // Apply regularization and update weights
                match self.penalty {
                    RegularizationType::L2(_) => {
                        weights = &weights * (1.0 - self.learning_rate * self.regularization_param)
                            + &(weight_grad_sum * (self.learning_rate / batch_len));
                    }
                    RegularizationType::L1(_) => {
                        // L1 regularization subgradient update
                        let l1_grad = weights.mapv(|w| {
                            if w > 0.0 {
                                1.0
                            } else if w < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        });

                        weights = &weights
                            - &(l1_grad * (self.learning_rate * self.regularization_param))
                            + &(weight_grad_sum * (self.learning_rate / batch_len));
                    }
                }

                if self.fit_intercept {
                    bias += self.learning_rate * bias_grad_sum / batch_len;
                }

                // Check for weight explosion early
                Self::check_weights_validity(&weights, bias)?;
            }

            // Convergence check
            let weight_diff = {
                let diff = &weights - &prev_weights;
                (diff.iter().map(|&x| x * x).sum::<f64>() / weights.len() as f64).sqrt()
            };

            let bias_diff = if self.fit_intercept {
                (bias - prev_bias).abs()
            } else {
                0.0
            };

            if weight_diff < self.tol && bias_diff < self.tol {
                break;
            }

            prev_weights.assign(&weights);
            prev_bias = bias;
        }

        // Calculate final cost for reporting using closure
        let calculate_cost = |x: ArrayView2<f64>,
                              y: &Array1<f64>,
                              weights: &Array1<f64>,
                              bias: f64,
                              penalty: &RegularizationType,
                              regularization_param: f64|
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
                RegularizationType::L2(_) => {
                    regularization_param * weights.iter().map(|&w| w * w).sum::<f64>() / 2.0
                }
                RegularizationType::L1(_) => {
                    regularization_param * weights.iter().map(|&w| w.abs()).sum::<f64>()
                }
            };

            hinge_loss + regularization_term
        };

        let final_cost = calculate_cost(
            x,
            &y_binary,
            &weights,
            bias,
            &self.penalty,
            self.regularization_param,
        );

        println!(
            "Linear SVC model computing finished at iteration {}, cost: {}",
            n_iter, final_cost
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
    /// - `x`: Input features as a 2D array where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)`: Array of predictions (0.0 or 1.0) for each sample
    /// - `Err(ModelError::NotFitted)`: If the model hasn't been trained yet
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        // Validate input data
        self.validate_input_data(x)?;

        // Get decision values and convert to predictions
        let decision = self.decision_function(x)?;
        Ok(decision.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
    }

    /// Calculates the decision function values (distance to the hyperplane) for each sample.
    ///
    /// This method provides raw scores rather than class predictions.
    /// Positive values indicate class 1, negative values indicate class 0,
    /// and the magnitude indicates confidence (distance from the decision boundary).
    ///
    /// # Parameters
    ///
    /// - `x`: Input features as a 2D array where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)`: Raw decision scores for each sample
    /// - `Err(ModelError::NotFitted)`: If the model hasn't been trained yet
    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
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
}

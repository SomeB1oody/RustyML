pub use super::RegularizationType;
use super::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_tolerance,
};
use crate::error::Error;
use crate::math::hinge_loss;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, s};
use ndarray_rand::rand::seq::SliceRandom;
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
/// - `random_state` - Optional seed for the per-epoch minibatch shuffling, enabling reproducible training
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
///     1e-4,                           // tolerance
///     Some(42),                       // random_state for reproducible training
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
    random_state: Option<u64>,
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
    /// - `random_state`: None (non-deterministic minibatch shuffling)
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
            random_state: None,
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
    /// - `random_state` - Optional seed for the per-epoch minibatch shuffling. Pass `Some(seed)` for reproducible training, or `None` for non-deterministic behavior
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A Result containing the new LinearSVC instance if validation passes
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParameter` if:
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
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        // Validate parameters
        validate_max_iterations(max_iter)?;
        validate_learning_rate(learning_rate)?;
        validate_tolerance(tol)?;

        // Validate regularization parameter
        let reg_param = match penalty {
            RegularizationType::L1(lambda) | RegularizationType::L2(lambda) => lambda,
        };
        if reg_param < 0.0 || !reg_param.is_finite() {
            return Err(Error::invalid_parameter(
                "penalty",
                format!(
                    "regularization parameter must be non-negative and finite, got {reg_param}"
                ),
            ));
        }

        Ok(LinearSVC {
            weights: None,
            bias: None,
            max_iter,
            learning_rate,
            penalty,
            fit_intercept,
            tol,
            random_state,
            n_iter: None,
        })
    }

    /// Checks if weights contain invalid values (NaN or infinite)
    fn check_weights_validity(weights: &Array1<f64>, bias: f64) -> Result<(), Error> {
        if weights.iter().any(|&w| !w.is_finite()) || !bias.is_finite() {
            return Err(Error::non_finite(
                "weights during training (try reducing learning_rate or regularization_param)",
            ));
        }
        Ok(())
    }

    /// Calculates the optimal batch size based on dataset size
    fn calculate_batch_size(n_samples: usize) -> usize {
        const MIN_BATCH_SIZE: usize = 32;
        const MAX_BATCH_SIZE: usize = 512;

        (n_samples / 10).clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
    }

    // Getters
    get_field!(get_fit_intercept, fit_intercept, bool);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_weights, weights, Option<&Array1<f64>>);
    get_field!(get_bias, bias, Option<f64>);
    get_field!(get_penalty, penalty, RegularizationType);
    get_field!(get_random_state, random_state, Option<u64>);

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
    /// - `Result<&mut Self, Error>` - A mutable reference to the trained model
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - If input data is invalid or feature dimension mismatches
    /// - `Error::NonFinite` - If numerical issues occur during training (e.g., weights become NaN)
    ///
    /// # Performance
    ///
    /// Parallel processing is automatically enabled when the batch size exceeds `LINEAR_SVC_PARALLEL_THRESHOLD` (200).
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
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

        // Create index array for random sampling. Seeding the RNG from `random_state`
        // makes the per-epoch minibatch shuffling reproducible.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = crate::random::make_rng(self.random_state);

        let mut prev_weights = weights.clone();
        let mut prev_bias = bias;

        let mut n_iter = 0;

        // Calculate optimal batch size
        let batch_size = Self::calculate_batch_size(n_samples);

        // Define cost calculation closure
        #[allow(unused_variables)]
        let calculate_cost = |x: &ArrayBase<S, Ix2>,
                              y: &Array1<f64>,
                              weights: &Array1<f64>,
                              bias: f64,
                              penalty: &RegularizationType|
         -> f64 {
            // Hinge loss via the shared math primitive (margins = w·xᵢ + b)
            let margins: Array1<f64> = x.outer_iter().map(|xi| xi.dot(weights) + bias).collect();
            let hinge = hinge_loss(&margins, y);

            // Calculate regularization term
            let regularization_term = match penalty {
                RegularizationType::L2(lambda) => {
                    lambda * weights.iter().map(|&w| w * w).sum::<f64>() / 2.0
                }
                RegularizationType::L1(lambda) => {
                    lambda * weights.iter().map(|&w| w.abs()).sum::<f64>()
                }
            };

            hinge + regularization_term
        };

        // Create progress bar for training iterations
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.max_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Cost: {msg}",
            );
            pb.set_message("Initializing...");
            pb
        };

        while n_iter < self.max_iter {
            n_iter += 1;
            #[cfg(feature = "show_progress")]
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
            #[cfg(feature = "show_progress")]
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

        // Finish progress bar with final statistics
        #[cfg(feature = "show_progress")]
        {
            let final_cost = calculate_cost(x, &y_binary, &weights, bias, &self.penalty);
            let convergence_status = if n_iter < self.max_iter {
                "Converged"
            } else {
                "Max iterations"
            };
            progress_bar.finish_with_message(format!(
                "{:.6} | {} | Iterations: {}",
                final_cost, convergence_status, n_iter
            ));
        }

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
    /// - `Result<Array1<f64>, Error>` - Array of predicted class labels (0.0 or 1.0) for each sample
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model hasn't been trained yet
    /// - `Error::DimensionMismatch` - If input data is invalid or feature dimension mismatches
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // Fitting checks and input validation are handled by `decision_function`
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
    /// - `Result<Array1<f64>, Error>` - Raw decision scores for each sample
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model hasn't been trained yet
    /// - `Error::DimensionMismatch` - If input data is invalid or feature dimension mismatches
    /// - `Error::NonFinite` - If the computation produced NaN or infinite values
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted, then validate the prediction input
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LinearSVC"))?;
        let bias = self.bias.unwrap_or(0.0);

        validate_predict_input(x, weights.len())?;

        let decision = x.dot(weights) + bias;

        // Check for NaN/Inf in decision values
        if decision.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("decision function"));
        }

        Ok(decision)
    }

    /// Fits the model to the training data and then predicts labels for the same data.
    ///
    /// A convenience method that sequentially executes `fit` and then `predict`.
    ///
    /// # Parameters
    ///
    /// - `x` - Input features as a 2D array where each row is a sample and each column is a feature
    /// - `y` - Target values as a 1D array (should contain only 0.0 and 1.0 values)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - Predicted class labels (0.0 or 1.0) for the training data
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - If input data is invalid
    /// - `Error::NonFinite` - If numerical issues occur during training
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

    model_save_and_load_methods!(LinearSVC);
}

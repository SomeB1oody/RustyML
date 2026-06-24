//! Linear Support Vector Classifier
//!
//! Provides [`LinearSVC`], a hinge-loss classifier trained with stochastic gradient
//! descent and L1 or L2 regularization, along with the [`RegularizationType`](crate::machine_learning::RegularizationType) enum
//! that selects the penalty

use crate::error::Error;
pub use crate::machine_learning::RegularizationType;
use crate::machine_learning::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_tolerance,
};
use crate::math::matmul::gemv_par_auto;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, s};
use ndarray_rand::rand::seq::SliceRandom;

/// Loss function minimized by [`LinearSVC`]
///
/// `Hinge` is `max(0, 1 - y * f(x))`; `SquaredHinge` is its square, `max(0, 1 - y * f(x))^2`,
/// which penalizes margin violations quadratically and is differentiable everywhere
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Loss {
    /// Standard hinge loss `max(0, 1 - y * f(x))`
    Hinge,
    /// Squared hinge loss `max(0, 1 - y * f(x))^2`
    SquaredHinge,
}

/// Linear Support Vector Classifier (LinearSVC)
///
/// A linear classifier trained with the hinge loss function
/// and L1 or L2 regularization to prevent overfitting
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::*;
///
/// // Create model with custom parameters
/// let mut model = LinearSVC::new(
///     1000,                        // max_iter
///     0.001,                       // learning_rate
///     RegularizationType::L2(1.0), // penalty type with regularization strength
///     true,                        // fit_intercept
///     1e-4,                        // tolerance
/// )
/// .unwrap()
/// .with_random_state(42); // fixed seed for reproducible training
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
    /// Weight coefficients for each feature
    weights: Option<Array1<f64>>,
    /// Bias term (intercept) of the model
    bias: Option<f64>,
    /// Maximum number of iterations for the optimizer
    max_iter: usize,
    /// Learning rate (step size) for gradient descent
    learning_rate: f64,
    /// Inverse-scaling learning-rate decay: the effective rate at epoch `t` (0-indexed) is
    /// `learning_rate / (1 + learning_rate_decay * t)`. `0.0` means a constant rate
    learning_rate_decay: f64,
    /// Regularization type (L1 or L2) with strength parameter
    penalty: RegularizationType,
    /// Whether to calculate and use an intercept/bias term
    fit_intercept: bool,
    /// Training convergence tolerance
    tol: f64,
    /// Loss function (hinge or squared hinge)
    loss: Loss,
    /// Optional seed for the per-epoch minibatch shuffling, enabling reproducible training
    random_state: Option<u64>,
    /// Number of iterations that were actually performed during training
    n_iter: Option<usize>,
}

impl Default for LinearSVC {
    /// Creates a new LinearSVC with default parameters
    ///
    /// # Default Values
    ///
    /// - `max_iter`: 1000
    /// - `learning_rate`: 0.001
    /// - `learning_rate_decay`: 0.0 (constant rate)
    /// - `penalty`: RegularizationType::L2(1.0) - L2 regularization with strength 1.0
    /// - `fit_intercept`: true
    /// - `tol`: 1e-4
    /// - `loss`: Loss::Hinge
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
            learning_rate_decay: 0.0,
            penalty: RegularizationType::L2(1.0),
            fit_intercept: true,
            tol: 1e-4,
            loss: Loss::Hinge,
            random_state: None,
            n_iter: None,
        }
    }
}

impl LinearSVC {
    /// Creates a new LinearSVC instance with custom parameters
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
    /// - `Result<Self, Error>` - A Result containing the new LinearSVC instance if validation passes
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParameter` if:
    /// - `max_iter` is 0
    /// - `learning_rate` is not positive or not finite
    /// - `penalty` regularization parameter is negative or not finite
    /// - `tol` is not positive or not finite
    ///
    /// # Notes
    ///
    /// The per-epoch minibatch shuffling is non-deterministic by default. For reproducible
    /// runs, set a fixed seed after construction with the builder method below:
    ///
    /// - [`with_random_state`](Self::with_random_state) - fixed seed for minibatch shuffling
    pub fn new(
        max_iter: usize,
        learning_rate: f64,
        penalty: RegularizationType,
        fit_intercept: bool,
        tol: f64,
    ) -> Result<Self, Error> {
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
            learning_rate_decay: 0.0,
            penalty,
            fit_intercept,
            tol,
            loss: Loss::Hinge,
            random_state: None,
            n_iter: None,
        })
    }

    /// Selects the loss function (default: [`Loss::Hinge`])
    ///
    /// [`Loss::SquaredHinge`] penalizes margin violations quadratically
    ///
    /// # Parameters
    ///
    /// - `loss` - the loss function to minimize
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Sets an inverse-scaling learning-rate decay (default: `0.0`, i.e. a constant rate)
    ///
    /// The effective learning rate at epoch `t` (0-indexed) is
    /// `learning_rate / (1 + learning_rate_decay * t)`. A positive decay shrinks the step
    /// size over time, which lets stochastic gradient descent settle closer to the optimum
    /// instead of hovering at a fixed-step distance from it
    ///
    /// # Parameters
    ///
    /// - `decay` - Non-negative, finite decay rate
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated instance, or an error if `decay` is invalid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `decay` is negative or not finite
    pub fn with_learning_rate_decay(mut self, decay: f64) -> Result<Self, Error> {
        if decay < 0.0 || !decay.is_finite() {
            return Err(Error::invalid_parameter(
                "learning_rate_decay",
                format!("must be non-negative and finite, got {decay}"),
            ));
        }
        self.learning_rate_decay = decay;
        Ok(self)
    }

    /// Sets a fixed RNG seed for the per-epoch minibatch shuffling, making training
    /// reproducible (default: `None`, non-deterministic)
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the minibatch-shuffling RNG
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
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
    get_field!(get_learning_rate_decay, learning_rate_decay, f64);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_weights, weights, Option<&Array1<f64>>);
    get_field!(get_bias, bias, Option<f64>);
    get_field!(get_penalty, penalty, RegularizationType);
    get_field!(get_loss, loss, Loss);
    get_field!(get_random_state, random_state, Option<u64>);

    /// Trains the model on the provided data
    ///
    /// Uses stochastic gradient descent to optimize the hinge loss function,
    /// continuing until either the maximum iterations are reached or convergence
    /// is detected based on tolerance
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
    /// - `Error::InvalidInput` - If any label is not 0.0 or 1.0 (LinearSVC is a binary classifier)
    /// - `Error::NonFinite` - If numerical issues occur during training (e.g., weights become NaN)
    ///
    /// # Performance
    ///
    /// The full-batch cost GEMV runs parallel above its FLOPs gate; the per-batch
    /// gradient accumulation is sequential so the f64 result is reproducible
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        preliminary_check(x, Some(y))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if !y.iter().all(|&yi| yi == 0.0 || yi == 1.0) {
            return Err(Error::invalid_input(
                "LinearSVC is a binary classifier; all labels must be either 0.0 or 1.0",
            ));
        }

        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;

        // Convert labels to -1 and 1
        let y_binary = y.mapv(|v| if v <= 0.0 { -1.0 } else { 1.0 });

        // Index array for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = crate::random::make_rng(self.random_state);

        let mut prev_weights = weights.clone();
        let mut prev_bias = bias;

        let mut n_iter = 0;

        let batch_size = Self::calculate_batch_size(n_samples);

        #[allow(unused_variables)]
        let calculate_cost = |x: &ArrayBase<S, Ix2>,
                              y: &Array1<f64>,
                              weights: &Array1<f64>,
                              bias: f64,
                              penalty: &RegularizationType|
         -> f64 {
            let margins: Array1<f64> = gemv_par_auto(x, weights) + bias;
            // Mean data loss, matching the configured loss function
            let data_loss = match self.loss {
                Loss::Hinge => {
                    margins
                        .iter()
                        .zip(y.iter())
                        .map(|(&m, &yi)| (1.0 - yi * m).max(0.0))
                        .sum::<f64>()
                        / margins.len() as f64
                }
                Loss::SquaredHinge => {
                    margins
                        .iter()
                        .zip(y.iter())
                        .map(|(&m, &yi)| {
                            let s = (1.0 - yi * m).max(0.0);
                            s * s
                        })
                        .sum::<f64>()
                        / margins.len() as f64
                }
            };

            // Regularization term
            let regularization_term = match penalty {
                RegularizationType::L2(lambda) => {
                    lambda * weights.iter().map(|&w| w * w).sum::<f64>() / 2.0
                }
                RegularizationType::L1(lambda) => {
                    lambda * weights.iter().map(|&w| w.abs()).sum::<f64>()
                }
            };

            data_loss + regularization_term
        };

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

            // Inverse-scaling learning-rate schedule
            let lr = self.learning_rate / (1.0 + self.learning_rate_decay * (n_iter - 1) as f64);

            indices.shuffle(&mut rng);

            for batch_indices in indices.chunks(batch_size) {
                let batch_len = batch_indices.len() as f64;

                let loss = self.loss;
                let accumulate = |idx: usize, w_grad: &mut Array1<f64>, b_grad: &mut f64| {
                    let xi = x.slice(s![idx, ..]);
                    let yi = y_binary[idx];
                    let margin = xi.dot(&weights) + bias;
                    let slack = 1.0 - yi * margin;
                    if slack > 0.0 {
                        let coef = match loss {
                            Loss::Hinge => yi,
                            Loss::SquaredHinge => 2.0 * slack * yi,
                        };
                        w_grad.scaled_add(coef, &xi);
                        *b_grad += coef;
                    }
                };

                // Serial accumulation: one minibatch is far below the parallel sum gate,
                // and the accumulator is an n_features vector, so blocked merging would
                // allocate more than it saves
                let (weight_grad_sum, bias_grad_sum) = {
                    let mut acc = (Array1::<f64>::zeros(n_features), 0.0);
                    for &idx in batch_indices {
                        accumulate(idx, &mut acc.0, &mut acc.1);
                    }
                    acc
                };

                // Update weights with the averaged loss gradient (in place, no allocation)
                weights.scaled_add(lr / batch_len, &weight_grad_sum);

                match self.penalty {
                    RegularizationType::L2(lambda) => {
                        // L2 gradient is lambda * weights
                        weights = &weights * (1.0 - lr * lambda);
                    }
                    RegularizationType::L1(lambda) => {
                        // L1 subgradient update
                        let l1_grad = weights.mapv(|w| {
                            if w > 0.0 {
                                1.0
                            } else if w < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        });
                        weights = &weights - &(l1_grad * (lr * lambda));
                    }
                }

                if self.fit_intercept {
                    bias += lr * bias_grad_sum / batch_len;
                }

                // Catch weight explosion early
                Self::check_weights_validity(&weights, bias)?;
            }

            // Convergence check: mean squared difference
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

            // Compute and display current cost
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

    /// Predicts the class for each sample in the provided data
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

    /// Calculates the decision function values for each sample
    ///
    /// Raw scores represent the distance to the decision hyperplane: positive values
    /// indicate class 1 and negative values indicate class 0
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
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LinearSVC"))?;
        let bias = self.bias.unwrap_or(0.0);

        validate_predict_input(x, weights.len())?;

        let decision = gemv_par_auto(x, weights) + bias;

        // Reject NaN/Inf in decision values
        if decision.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("decision function"));
        }

        Ok(decision)
    }

    /// Fits the model to the training data and then predicts labels for the same data
    ///
    /// A convenience method that sequentially executes `fit` and then `predict`
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

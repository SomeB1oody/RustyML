//! Linear regression via gradient descent
//!
//! Provides the [`LinearRegression`] model supporting multivariate regression, an
//! optional intercept term, and L1/L2 regularization

pub use super::RegularizationType;
use super::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_regularization_type, validate_tolerance,
};
use crate::error::Error;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge,
    ParallelIterator,
};

/// Feature/sample count at or above which parallel computation is used; below it computation is sequential
const LINEAR_REGRESSION_PARALLEL_THRESHOLD: usize = 200;

/// Linear regression model implementation
///
/// Trains a linear regression model using gradient descent. Supports multivariate regression, an
/// optional intercept term, and adjustment of the learning rate, maximum iterations, and convergence
/// tolerance
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::linear_regression::*;
/// use ndarray::{Array1, Array2};
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
    /// Model coefficients (slopes), `None` before training
    coefficients: Option<Array1<f64>>,
    /// Model intercept, `None` before training
    intercept: Option<f64>,
    /// Whether to include an intercept term in the model
    fit_intercept: bool,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Maximum number of iterations for gradient descent
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
    /// Regularization type and strength
    regularization_type: Option<RegularizationType>,
}

impl Default for LinearRegression {
    /// Creates a new LinearRegression instance with default parameter values
    ///
    /// # Default Values
    ///
    /// - `coefficients` - `None` - model coefficients are not initialized until training
    /// - `intercept` - `None` - model intercept is not initialized until training
    /// - `fit_intercept` - `true` - include an intercept term in the linear model
    /// - `learning_rate` - `0.01` - learning rate for gradient descent optimization
    /// - `max_iter` - `1000` - maximum number of iterations for gradient descent
    /// - `tol` - `1e-5` - convergence tolerance (0.00001) for stopping criteria
    /// - `n_iter` - `None` - number of actual iterations performed (set after training)
    /// - `regularization_type` - `None` - no regularization applied by default
    ///
    /// # Returns
    ///
    /// - `LinearRegression` - a new instance with default parameters
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
    /// Creates a new linear regression model with custom parameters
    ///
    /// Validates all input parameters to ensure they are within acceptable numerical ranges
    /// before returning the model instance
    ///
    /// # Parameters
    ///
    /// - `fit_intercept` - whether to calculate the intercept for this model
    /// - `learning_rate` - the learning rate for gradient descent optimization
    /// - `max_iterations` - maximum number of iterations for gradient descent
    /// - `tolerance` - the tolerance for stopping criteria
    /// - `regularization_type` - optional regularization to prevent overfitting (L1, L2, or None)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - a new instance of LinearRegression, or an error if parameters are invalid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if learning_rate, max_iterations, tolerance, or the regularization alpha is invalid
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        regularization_type: Option<RegularizationType>,
    ) -> Result<Self, Error> {
        // Input validation
        validate_learning_rate(learning_rate)?;
        validate_max_iterations(max_iterations)?;
        validate_tolerance(tolerance)?;
        validate_regularization_type(regularization_type)?;

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

    get_field!(get_fit_intercept, fit_intercept, bool);
    get_field!(get_learning_rate, learning_rate, f64);
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

    /// Fits the linear regression model using gradient descent
    ///
    /// Iteratively updates the model's coefficients and intercept to minimize the cost function,
    /// with early stopping once convergence is reached
    ///
    /// # Parameters
    ///
    /// - `x` - feature matrix, each row is a sample, each column is a feature
    /// - `y` - target variable vector
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - a mutable reference to self for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::NonFinite` - if numerical issues like NaN or infinity occur during training
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - if input dimensions are inconsistent
    ///
    /// # Performance
    ///
    /// Parallel computation is used for L1 regularization and gradient updates when the number of
    /// features is at least `LINEAR_REGRESSION_PARALLEL_THRESHOLD` (200)
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        preliminary_check(x, Some(y))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut weights = Array1::<f64>::zeros(n_features);
        let mut intercept = 0.0;

        let mut prev_cost = f64::INFINITY;
        // Track consecutive convergences for stability
        let mut convergence_count = 0;
        const CONVERGENCE_THRESHOLD: usize = 3;

        let mut n_iter = 0;

        // Pre-allocate arrays to avoid repeated memory allocation
        let mut predictions = Array1::<f64>::zeros(n_samples);
        let mut error_vec = Array1::<f64>::zeros(n_samples);

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.max_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Cost: {msg}",
            );
            pb.set_message(format!(
                "{:.6} | Convergence: 0/{}",
                f64::INFINITY,
                CONVERGENCE_THRESHOLD
            ));
            pb
        };

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            // Vectorized prediction
            predictions.assign(&x.dot(&weights));
            if self.fit_intercept {
                predictions += intercept;
            }

            // Calculate errors once; the same vector feeds both the cost and the gradient
            error_vec.assign(&(&predictions - y));

            // Cost (sum of squared errors) reuses the error vector: SSE = e dot e
            let sse = error_vec.dot(&error_vec);

            let regularization_term = match &self.regularization_type {
                None => 0.0,
                Some(RegularizationType::L1(alpha)) => {
                    if n_features >= LINEAR_REGRESSION_PARALLEL_THRESHOLD {
                        alpha * weights.iter().par_bridge().map(|w| w.abs()).sum::<f64>()
                    } else {
                        alpha * weights.iter().map(|w| w.abs()).sum::<f64>()
                    }
                }
                // Penalty (alpha/2)*||w||^2; the 1/2 matches the gradient alpha*w applied below
                // and mirrors the 1/2 in the sse/(2n) data term
                Some(RegularizationType::L2(alpha)) => 0.5 * alpha * weights.dot(&weights),
            };

            let cost = sse / (2.0 * n_samples as f64) + regularization_term;

            #[cfg(feature = "show_progress")]
            progress_bar.set_message(format!(
                "{:.6} | Convergence: {}/{}",
                cost, convergence_count, CONVERGENCE_THRESHOLD
            ));
            #[cfg(feature = "show_progress")]
            progress_bar.inc(1);

            if !cost.is_finite() {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite cost");
                return Err(Error::non_finite("cost calculation"));
            }

            // Gradients via matrix operations
            let mut weight_gradients = x.t().dot(&error_vec) / (n_samples as f64);
            let intercept_gradient = if self.fit_intercept {
                error_vec.sum() / (n_samples as f64)
            } else {
                0.0
            };

            if weight_gradients.iter().any(|&val| !val.is_finite())
                || !intercept_gradient.is_finite()
            {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite gradients");
                return Err(Error::non_finite("gradient calculation"));
            }

            // Add regularization terms to gradients
            match &self.regularization_type {
                None => {}
                Some(RegularizationType::L1(alpha)) => {
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
                    // d/dw [(alpha/2) * ||w||^2] = alpha * w, matching the cost term above
                    weight_gradients.scaled_add(*alpha, &weights);
                }
            }

            // Update parameters
            weights.scaled_add(-self.learning_rate, &weight_gradients);
            if self.fit_intercept {
                intercept -= self.learning_rate * intercept_gradient;
            }

            if weights.iter().any(|&val| !val.is_finite()) || !intercept.is_finite() {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite parameters");
                return Err(Error::non_finite("parameter update"));
            }

            // Require several consecutive small cost changes before declaring convergence
            let cost_change = (prev_cost - cost).abs();
            if cost_change < self.tol {
                convergence_count += 1;
                if convergence_count >= CONVERGENCE_THRESHOLD {
                    break;
                }
            } else {
                convergence_count = 0;
            }

            prev_cost = cost;
        }

        #[cfg(feature = "show_progress")]
        let convergence_status = if n_iter < self.max_iter {
            "Converged"
        } else {
            "Max iterations"
        };
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message(format!(
            "{:.6} | {} | Iterations: {}",
            prev_cost, convergence_status, n_iter
        ));

        // Save training results
        self.coefficients = Some(weights);
        self.intercept = Some(if self.fit_intercept { intercept } else { 0.0 });
        self.n_iter = Some(n_iter);

        Ok(self)
    }

    /// Makes predictions using the trained model
    ///
    /// Applies the learned coefficients and intercept to the provided feature matrix
    ///
    /// # Parameters
    ///
    /// - `x` - prediction data, each row is a sample, each column is a feature
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - a vector of predictions
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - if the model has not been trained yet
    /// - `Error::EmptyInput` - if the feature matrix has no rows
    /// - `Error::DimensionMismatch` - if the feature count does not match the trained model
    /// - `Error::NonFinite` - if the input data or the predictions contain non-finite values
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted, then validate the prediction input
        let coeffs = self
            .coefficients
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LinearRegression"))?;
        let intercept = self.intercept.unwrap_or(0.0);

        validate_predict_input(x, coeffs.len())?;

        let mut predictions = x.dot(coeffs);
        if self.fit_intercept {
            predictions += intercept;
        }

        if predictions.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("prediction calculation"));
        }

        Ok(predictions)
    }

    /// Fits the model to the training data and then makes predictions on the same data
    ///
    /// A convenience method that runs `fit` followed by `predict`
    ///
    /// # Parameters
    ///
    /// - `x` - the input features matrix
    /// - `y` - the target values corresponding to each training example
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - the predicted values for the input data
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - if input data is invalid
    /// - `Error::NonFinite` - if an error occurs during fitting or prediction
    pub fn fit_predict<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit(x, y)?;
        self.predict(x)
    }

    model_save_and_load_methods!(LinearRegression);
}

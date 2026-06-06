pub use super::RegularizationType;
use super::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_regularization_type, validate_tolerance,
};
use crate::error::ModelError;
use crate::math::{logistic_loss, sigmoid};
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, s};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Threshold for enabling parallel computation in logistic regression.
/// When the number of samples exceeds this value, parallel processing is used
/// to improve performance. For smaller datasets, sequential processing is used
/// to avoid parallelization overhead.
const LOGISTIC_REGRESSION_PARALLEL_THRESHOLD: usize = 1000;

/// Logistic Regression model implementation
///
/// This model uses gradient descent to train a binary classification logistic regression model.
///
/// # Fields
///
/// - `weights` - Model weights vector, None before training
/// - `fit_intercept` - Whether to use intercept term (bias)
/// - `learning_rate` - Controls gradient descent step size
/// - `max_iter` - Maximum number of iterations for gradient descent
/// - `tol` - Convergence tolerance, stops iteration when loss change is smaller than this value
/// - `n_iter` - Actual number of iterations the algorithm ran for after fitting, None before training
/// - `regularization_type` - Regularization type and strength
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::logistic_regression::LogisticRegression;
/// use ndarray::{Array1, Array2};
///
/// // Create a logistic regression model
/// let mut model = LogisticRegression::default();
///
/// // Create some simple training data
/// // Two features: x1 and x2
/// // This data represents a simple logical AND function
/// let x_train = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  // [0,0] -> 0
///     0.0, 1.0,  // [0,1] -> 0
///     1.0, 0.0,  // [1,0] -> 0
///     1.0, 1.0,  // [1,1] -> 1
/// ]).unwrap();
///
/// let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
///
/// // Train the model
/// model.fit(&x_train, &y_train).unwrap();
///
/// // Create test data
/// let x_test = Array2::from_shape_vec((2, 2), vec![
///     1.0, 0.0,  // Should predict 0
///     1.0, 1.0,  // Should predict 1
/// ]).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x_test);
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogisticRegression {
    weights: Option<Array1<f64>>,
    fit_intercept: bool,
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
    n_iter: Option<usize>,
    regularization_type: Option<RegularizationType>,
}

impl Default for LogisticRegression {
    /// Creates a logistic regression model with default parameters.
    ///
    /// This implementation provides sensible default values that work well for most binary
    /// classification problems as a starting point for experimentation and training.
    ///
    /// # Default Values
    ///
    /// - `fit_intercept`: `true` - Include a bias/intercept term in the model, which is generally recommended for most datasets
    /// - `learning_rate`: `0.01` - A moderate learning rate that provides stable convergence for most problems without being too slow
    /// - `max_iter`: `100` - Maximum number of gradient descent iterations, sufficient for many simple to moderately complex problems
    /// - `tol`: `1e-4` - Convergence tolerance (0.0001), stops training when the loss change between iterations is smaller than this value
    ///
    /// # Returns
    ///
    /// * `Self` - A new `LogisticRegression` instance with default configuration
    fn default() -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-4,
            n_iter: None,
            regularization_type: None,
        }
    }
}

impl LogisticRegression {
    /// Creates a new logistic regression model with specified parameters
    ///
    /// # Parameters
    ///
    /// - `fit_intercept` - Whether to add intercept term (bias)
    /// - `learning_rate` - Learning rate for gradient descent, must be positive and finite
    /// - `max_iterations` - Maximum number of iterations, must be greater than 0
    /// - `tolerance` - Convergence tolerance, stops when loss change is below this value, must be positive and finite
    /// - `regularization_type` - Optional regularization to prevent overfitting. Alpha must be non-negative and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - An untrained logistic regression model instance or validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If any parameter is invalid (e.g., non-positive learning rate)
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
        validate_regularization_type(regularization_type)?;

        Ok(LogisticRegression {
            weights: None,
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
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(
        get_regularization_type,
        regularization_type,
        Option<RegularizationType>
    );
    get_field_as_ref!(get_weights, weights, Option<&Array1<f64>>);

    /// Trains the logistic regression model
    ///
    /// Uses gradient descent to minimize the logistic loss function.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample and each column is a feature
    /// - `y` - Target variable containing 0 or 1 indicating sample class
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A mutable reference to the trained model or error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If target vector contains values other than 0 or 1
    /// - `ModelError::ProcessingError` - If numerical issues (NaN/Infinity) occur during training
    ///
    /// # Performance
    ///
    /// Parallel processing is automatically enabled when the number of samples exceeds 1000.
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Preliminary check
        preliminary_check(x, Some(y))?;

        // Check target values are binary
        for &val in y.iter() {
            if val != 0.0 && val != 1.0 {
                return Err(ModelError::InputValidationError(
                    "Target vector must contain only 0 or 1".to_string(),
                ));
            }
        }

        let (n_samples, mut n_features) = x.dim();

        // Decide the source of x_train based on whether to use intercept
        let x_train_view: ArrayView2<f64>;
        let _x_train_owned: Option<Array2<f64>>;

        if self.fit_intercept {
            n_features += 1;
            let mut x_with_bias = Array2::ones((n_samples, n_features));
            x_with_bias.slice_mut(s![.., 1..]).assign(x);
            _x_train_owned = Some(x_with_bias);
            x_train_view = _x_train_owned.as_ref().unwrap().view();
        } else {
            _x_train_owned = None;
            x_train_view = x.view();
        }

        // Initialize weight vector
        let mut weights = Array1::zeros(n_features);
        let mut prev_cost = f64::INFINITY;
        #[cfg(feature = "show_progress")]
        let mut final_cost = prev_cost;
        let mut n_iter = 0;

        // Create progress bar for training iterations
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.max_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Loss: {msg}",
            );
            pb.set_message(format!("{:.6}", f64::INFINITY));
            pb
        };

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            // Compute linear predictions (reuse for both gradient and loss calculation)
            let predictions = x_train_view.dot(&weights);

            // Compute sigmoid activation with conditional parallelization
            let sigmoid_preds = if n_samples >= LOGISTIC_REGRESSION_PARALLEL_THRESHOLD {
                // Parallel computation for large datasets
                let sigmoid_vec = (0..n_samples)
                    .into_par_iter()
                    .map(|i| sigmoid(predictions[i]))
                    .collect::<Vec<f64>>();
                Array1::from(sigmoid_vec)
            } else {
                // Sequential computation for small datasets
                predictions.mapv(sigmoid)
            };

            // Calculate prediction errors
            let errors = &sigmoid_preds - y;

            // Calculate gradients
            let mut gradients = x_train_view.t().dot(&errors) / n_samples as f64;

            // Check for numerical issues in gradients
            if gradients.iter().any(|&val| !val.is_finite()) {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite gradients");
                return Err(ModelError::ProcessingError(
                    "Gradient calculation resulted in NaN or infinite values".to_string(),
                ));
            }

            if let Some(reg_type) = &self.regularization_type {
                let start_idx = if self.fit_intercept { 1 } else { 0 };

                match reg_type {
                    RegularizationType::L1(regularization_strength) => {
                        for i in start_idx..n_features {
                            let sign = if weights[i] > 0.0 {
                                1.0
                            } else if weights[i] < 0.0 {
                                -1.0
                            } else {
                                0.0
                            };
                            gradients[i] += regularization_strength * sign / n_samples as f64;
                        }
                    }
                    RegularizationType::L2(regularization_strength) => {
                        for i in start_idx..n_features {
                            gradients[i] += regularization_strength * weights[i] / n_samples as f64;
                        }
                    }
                }
            }

            // Update weights
            weights = &weights - self.learning_rate * &gradients;

            // Check for numerical issues in updated weights
            if weights.iter().any(|&val| !val.is_finite()) {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite weights");
                return Err(ModelError::ProcessingError(
                    "Weight update resulted in NaN or infinite values".to_string(),
                ));
            }

            // Calculate loss using existing predictions
            let mut cost = logistic_loss(&predictions, y);

            if let Some(reg_type) = &self.regularization_type {
                let start_idx = if self.fit_intercept { 1 } else { 0 };

                match reg_type {
                    RegularizationType::L1(regularization_strength) => {
                        let l1_penalty: f64 =
                            weights.slice(s![start_idx..]).mapv(|w| w.abs()).sum();
                        cost += regularization_strength * l1_penalty / n_samples as f64;
                    }
                    RegularizationType::L2(regularization_strength) => {
                        let l2_penalty: f64 = weights.slice(s![start_idx..]).mapv(|w| w * w).sum();
                        cost += regularization_strength * l2_penalty / (2.0 * n_samples as f64);
                    }
                }
            }

            #[cfg(feature = "show_progress")]
            {
                final_cost = cost;
            }

            // Check for numerical issues in cost
            if !cost.is_finite() {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite cost");
                return Err(ModelError::ProcessingError(
                    "Cost calculation resulted in NaN or infinite value".to_string(),
                ));
            }

            // Update progress bar with current loss
            #[cfg(feature = "show_progress")]
            {
                progress_bar.set_message(format!("{:.6}", cost));
                progress_bar.inc(1);
            }

            // Check convergence condition
            if (prev_cost - cost).abs() < self.tol {
                break;
            }
            prev_cost = cost;
        }

        // Finish progress bar with final statistics
        #[cfg(feature = "show_progress")]
        let convergence_status = if n_iter < self.max_iter {
            "Converged"
        } else {
            "Max iterations"
        };
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message(format!(
            "{:.6} | {} | Iterations: {}",
            final_cost, convergence_status, n_iter
        ));

        self.weights = Some(weights);
        self.n_iter = Some(n_iter);

        Ok(self)
    }

    /// Predicts class labels for samples
    ///
    /// Performs classification by applying a 0.5 threshold to probability values.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample and each column is a feature (without bias term)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, ModelError>` - A 1D array containing predicted class labels (0 or 1)
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted yet
    /// - `ModelError::InputValidationError` - If input is empty, dimensions mismatch, or contains invalid values
    /// - `ModelError::ProcessingError` - If numerical issues occur during probability calculation
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Probabilities (with full input validation) then a 0.5 decision threshold
        let probs = self.predict_proba(x)?;
        Ok(probs.mapv(|prob| if prob >= 0.5 { 1 } else { 0 }))
    }

    /// Predicts the positive-class probability for each sample.
    ///
    /// Applies the sigmoid function to the linear decision values to produce
    /// probabilities in the range (0, 1).
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample and each column is a feature (without the bias term)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - Probability of the positive class for each sample
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted yet
    /// - `ModelError::InputValidationError` - If input is empty, dimensions mismatch, or data contains non-finite values
    /// - `ModelError::ProcessingError` - If numerical issues occur during probability calculation
    pub fn predict_proba<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Check if model has been fitted
        let weights = self.weights.as_ref().ok_or(ModelError::NotFitted)?;

        // Validate input against the feature count the model was trained on
        // (excluding the implicit bias column when an intercept was fitted)
        let expected_features = if self.fit_intercept {
            weights.len() - 1
        } else {
            weights.len()
        };
        validate_predict_input(x, expected_features)?;

        // Prepend the bias column when the model was trained with an intercept
        let probs = if self.fit_intercept {
            let (n_samples, n_features) = x.dim();
            let mut x_with_bias = Array2::ones((n_samples, n_features + 1));
            x_with_bias.slice_mut(s![.., 1..]).assign(x);
            self.sigmoid_decision(&x_with_bias)
        } else {
            self.sigmoid_decision(x)
        };

        // Check if probabilities contain invalid values
        if probs.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::ProcessingError(
                "Probability calculation resulted in NaN or infinite values".to_string(),
            ));
        }

        Ok(probs)
    }

    /// Applies the sigmoid to the raw linear decision values `x · weights`.
    ///
    /// `x` must already include the bias column when an intercept was fitted; this
    /// is an internal helper used by [`Self::predict_proba`].
    fn sigmoid_decision<S>(&self, x: &ArrayBase<S, Ix2>) -> Array1<f64>
    where
        S: Data<Elem = f64>,
    {
        let weights = self.weights.as_ref().unwrap();
        let mut predictions = x.dot(weights);

        // Apply sigmoid with conditional parallelization (mutate in place)
        if predictions.len() >= LOGISTIC_REGRESSION_PARALLEL_THRESHOLD {
            predictions.par_mapv_inplace(sigmoid);
        } else {
            predictions.mapv_inplace(sigmoid);
        }

        predictions
    }

    /// Fits the logistic regression model to the training data and then makes predictions.
    ///
    /// This is a convenience method that combines `fit` and `predict` operations in a single call.
    ///
    /// # Parameters
    ///
    /// - `train_x` - Training features as a 2D array
    /// - `train_y` - Target values as a 1D array corresponding to the training samples
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, ModelError>` - Predicted class labels for the training samples
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data does not match expectations
    /// - `ModelError::ProcessingError` - If errors occur during fitting or prediction
    pub fn fit_predict<S>(
        &mut self,
        train_x: &ArrayBase<S, Ix2>,
        train_y: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<i32>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.fit(train_x, train_y)?;
        self.predict(train_x)
    }

    model_save_and_load_methods!(LogisticRegression);
}

/// Generates polynomial features from input features.
///
/// This function transforms the input feature matrix into a new feature matrix containing
/// polynomial combinations of the input features up to the specified degree.
///
/// # Parameters
///
/// - `x` - Input feature matrix with shape (n_samples, n_features)
/// - `degree` - The maximum degree of polynomial features to generate
///
/// # Returns
///
/// - `Array2<f64>` - A new feature matrix containing polynomial combinations of the input features with shape (n_samples, n_output_features)
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::machine_learning::logistic_regression::{generate_polynomial_features, LogisticRegression};
///
/// // Example of using polynomial features with logistic regression
/// // Create a simple dataset for binary classification
/// let training_x = array![[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 2.0], [2.5, 1.0]];
/// let training_y = array![0.0, 0.0, 1.0, 1.0, 0.0];
///
/// // Transform features to polynomial features
/// let poly_training_x = generate_polynomial_features(&training_x, 2);
///
/// // Create and train a logistic regression model with polynomial features
/// let mut model = LogisticRegression::default();
/// model.fit(&poly_training_x, &training_y).unwrap();
/// ```
pub fn generate_polynomial_features<S>(x: &ArrayBase<S, Ix2>, degree: usize) -> Array2<f64>
where
    S: Data<Elem = f64> + Send + Sync,
{
    let (n_samples, n_features) = x.dim();

    // Calculate the number of output features (excluding constant term)
    // Formula: C(n+d,d) = (n+d)!/(n!*d!) where n is feature count and d is degree
    let n_output_features = {
        let mut count = 0; // No constant term
        for d in 1..=degree {
            let mut term = 1;
            for i in 0..d {
                term = term * (n_features + i) / (i + 1);
            }
            count += term;
        }
        count
    };

    // Initialize result matrix (without the constant term column)
    let mut result = Array2::<f64>::zeros((n_samples, n_output_features));

    // Add first-order features (original features)
    // Process samples in parallel using Rayon
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n_features {
                row[j] = x[[i, j]]; // Index starts from 0, no +1 offset
            }
        });

    // If degree >= 2, add higher-order features
    if degree >= 2 {
        let mut col_idx = n_features; // Start from n_features, no +1 offset

        // Define an inner recursive function to generate combinations.
        // The argument count is inherent to the recursion (shared mutable state plus
        // the current position), so the lint is allowed here for clarity.
        #[allow(clippy::too_many_arguments)]
        fn add_combinations<S>(
            x: &ArrayBase<S, Ix2>,
            result: &mut Array2<f64>,
            col_idx: &mut usize,
            n_features: usize,
            degree: usize,
            current_degree: usize,
            start_feature: usize,
            combination: &mut Vec<usize>,
        ) where
            S: Data<Elem = f64> + Send + Sync,
        {
            // If we've reached the target degree, compute the feature value
            if current_degree == degree {
                // Store current column index and increment to avoid race conditions
                let current_col = *col_idx;
                *col_idx += 1;

                // Process all samples in parallel
                result
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut value = 1.0;
                        for &feat_idx in combination.iter() {
                            value *= x[[i, feat_idx]];
                        }
                        row[current_col] = value;
                    });
                return;
            }

            // Recursively build combinations (sequential since it modifies shared state)
            for j in start_feature..n_features {
                combination.push(j);
                add_combinations(
                    x,
                    result,
                    col_idx,
                    n_features,
                    degree,
                    current_degree + 1,
                    j,
                    combination,
                );
                combination.pop();
            }
        }

        // Generate combinations for each degree
        for d in 2..=degree {
            add_combinations(
                x,
                &mut result,
                &mut col_idx,
                n_features,
                d,
                0,
                0,
                &mut vec![],
            );
        }
    }

    result
}

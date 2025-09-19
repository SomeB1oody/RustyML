pub use super::RegularizationType;
use super::preliminary_check;
use crate::ModelError;
use crate::math::{logistic_loss, sigmoid};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;

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
/// model.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Create test data
/// let x_test = Array2::from_shape_vec((2, 2), vec![
///     1.0, 0.0,  // Should predict 0
///     1.0, 1.0,  // Should predict 1
/// ]).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(x_test.view());
/// ```
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    weights: Option<Array1<f64>>,
    fit_intercept: bool,
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
    n_iter: Option<usize>,
    regularization_type: Option<RegularizationType>,
}

/// Creates a logistic regression model with default parameters.
///
/// This implementation provides sensible default values that work well for most binary
/// classification problems as a starting point for experimentation and training.
///
/// # Default Values
///
/// - `weights`: `None` - Model weights are not initialized until training begins
/// - `fit_intercept`: `true` - Include a bias/intercept term in the model, which is generally recommended for most datasets
/// - `learning_rate`: `0.01` - A moderate learning rate that provides stable convergence for most problems without being too slow
/// - `max_iter`: `100` - Maximum number of gradient descent iterations, sufficient for many simple to moderately complex problems
/// - `tol`: `1e-4` - Convergence tolerance (0.0001), stops training when the loss change between iterations is smaller than this value
/// - `n_iter`: `None` - Actual number of iterations performed is unknown before training
/// - `regularization_type`: `None` - No regularization applied by default, suitable for datasets without overfitting issues
///
/// # Returns
///
/// * `Self` - A new `LogisticRegression` instance with default configuration
impl Default for LogisticRegression {
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
    /// - `learning_rate` - Learning rate for gradient descent
    /// - `max_iterations` - Maximum number of iterations
    /// - `tolerance` - Convergence tolerance, stops when loss change is below this value
    ///
    /// # Returns
    ///
    /// * `Self` - An untrained logistic regression model instance
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        regularization_type: Option<RegularizationType>,
    ) -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept,
            learning_rate,
            max_iter: max_iterations,
            tol: tolerance,
            n_iter: None,
            regularization_type,
        }
    }

    get_fit_intercept!();

    get_learning_rate!();

    get_max_iterations!();

    get_tolerance!();

    get_actual_iterations!();

    get_regularization_type!();

    /// Returns the model weights
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - A reference to the weight array if the model has been trained, or None otherwise
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_weights(&self) -> Result<&Array1<f64>, ModelError> {
        match &self.weights {
            Some(weights) => Ok(weights),
            None => Err(ModelError::NotFitted),
        }
    }

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
    /// - `Ok(&mut Self)` - A mutable reference to the trained model, allowing for method chaining
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        // Preliminary check
        preliminary_check(x, Some(y))?;

        // Check learning rate bounds
        if self.learning_rate <= 0.0 {
            return Err(ModelError::InputValidationError(
                "Learning rate must be greater than 0.0".to_string(),
            ));
        }

        // Check target values are binary
        for &val in y.iter() {
            if val != 0.0 && val != 1.0 {
                return Err(ModelError::InputValidationError(
                    "Target vector must contain only 0 or 1".to_string(),
                ));
            }
        }

        let (n_samples, mut n_features) = x.dim();

        // Check max iterations
        if self.max_iter == 0 {
            return Err(ModelError::InputValidationError(
                "Maximum number of iterations must be greater than 0".to_string(),
            ));
        }

        // Decide the source of x_train based on whether to use intercept
        let x_train_view: ArrayView2<f64>;
        let _x_train_owned: Option<Array2<f64>>;

        if self.fit_intercept {
            n_features += 1;
            let mut x_with_bias = Array2::ones((n_samples, n_features));
            x_with_bias.slice_mut(s![.., 1..]).assign(&x);
            _x_train_owned = Some(x_with_bias);
            x_train_view = _x_train_owned.as_ref().unwrap().view();
        } else {
            _x_train_owned = None;
            x_train_view = x.view();
        }

        // Initialize weight vector
        let mut weights = Array1::zeros(n_features);
        let mut prev_cost = f64::INFINITY;
        let mut final_cost = prev_cost;
        let mut n_iter = 0;

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            let predictions = x_train_view.dot(&weights);

            // Parallel computation of sigmoid activation
            let sigmoid_preds = (0..n_samples)
                .into_par_iter()
                .map(|i| sigmoid(predictions[i]))
                .collect::<Vec<f64>>();
            let sigmoid_preds = Array1::from(sigmoid_preds);

            // Calculate prediction errors
            let errors = sigmoid_preds - y;

            // Calculate gradients
            let mut gradients = x_train_view.t().dot(&errors) / n_samples as f64;

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

            // Calculate loss - propagate error from logistic_loss
            let raw_preds = x_train_view.dot(&weights);
            let mut cost = logistic_loss(raw_preds.view(), y.view())?;

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

            final_cost = cost;

            // Check convergence condition
            if (prev_cost - cost).abs() < self.tol {
                break;
            }
            prev_cost = cost;
        }

        self.weights = Some(weights);
        self.n_iter = Some(n_iter);

        println!(
            "Logistic regression training finished at iteration {}, cost: {}",
            n_iter, final_cost
        );

        Ok(self)
    }

    /// Predicts class labels for samples
    ///
    /// Performs classification by applying a 0.5 threshold to probability values.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<i32>)` - A 1D array containing predicted class labels (0 or 1) for each sample
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<i32>, ModelError> {
        let (n_samples, n_features) = x.dim();

        // Check for empty input data
        if n_samples == 0 {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        // Prepare test data with optional bias term
        let x_test = if self.fit_intercept {
            let mut x_with_bias = Array2::ones((n_samples, n_features + 1));
            x_with_bias.slice_mut(s![.., 1..]).assign(&x);
            x_with_bias
        } else {
            x.to_owned()
        };

        let probs = self.predict_proba(&x_test.view())?;

        // Check if probabilities contain invalid values
        if probs.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::ProcessingError(
                "Probability calculation resulted in NaN or infinite values".to_string(),
            ));
        }

        // Apply threshold for classification
        Ok(probs.mapv(|prob| if prob >= 0.5 { 1 } else { 0 }))
    }

    /// Predicts probability scores for samples
    ///
    /// Uses the sigmoid function to convert linear predictions to probabilities between 0-1.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix view where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)`A 1D array containing the probability of each sample belonging to the positive class
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    fn predict_proba(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        use crate::math::sigmoid;

        if let Some(weights) = &self.weights {
            let mut predictions = x.dot(weights);

            // sigmoid
            predictions.par_mapv_inplace(|x| sigmoid(x));

            Ok(predictions)
        } else {
            Err(ModelError::NotFitted)
        }
    }

    /// Fits the logistic regression model to the training data and then makes predictions.
    ///
    /// This is a convenience method that combines `fit` and `predict` operations in a single call.
    ///
    /// # Parameters
    ///
    /// - `train_x` - Training features as a 2D array where each row represents a sample and each column represents a feature
    /// - `train_y` - Target values as a 1D array corresponding to the training samples
    /// - `test_x` - Test features for which predictions are to be made
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<i32>)` - Predicted class labels for the test samples
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(
        &mut self,
        train_x: ArrayView2<f64>,
        train_y: ArrayView1<f64>,
        test_x: ArrayView2<f64>,
    ) -> Result<Array1<i32>, ModelError> {
        self.fit(train_x, train_y)?;
        Ok(self.predict(test_x)?)
    }
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
/// * `Array2<f64>` - A new feature matrix containing polynomial combinations of the input features with shape (n_samples, n_output_features)
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
/// let poly_training_x = generate_polynomial_features(training_x.view(), 2);
///
/// // Create and train a logistic regression model with polynomial features
/// let mut model = LogisticRegression::default();
/// model.fit(poly_training_x.view(), training_y.view()).unwrap();
/// ```
pub fn generate_polynomial_features(x: ArrayView2<f64>, degree: usize) -> Array2<f64> {
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

        // Define an inner recursive function to generate combinations
        fn add_combinations(
            x: ArrayView2<f64>,
            result: &mut Array2<f64>,
            col_idx: &mut usize,
            n_samples: usize,
            n_features: usize,
            degree: usize,
            current_degree: usize,
            start_feature: usize,
            combination: &mut Vec<usize>,
        ) {
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
                    n_samples,
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
                n_samples,
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

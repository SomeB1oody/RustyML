//! Logistic regression for binary classification
//!
//! Provides the [`LogisticRegression`] model, trained with gradient descent and
//! optional L1/L2 regularization, plus the [`generate_polynomial_features`]
//! helper for building polynomial feature expansions

use crate::error::Error;
pub use crate::machine_learning::RegularizationType;
use crate::machine_learning::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_regularization_type, validate_tolerance,
};
use crate::math::matmul::gemv_par_auto;
use crate::math::{logistic_loss, sigmoid};
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, exp_map_f64_parallel_threshold};
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, s};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Logistic regression model for binary classification
///
/// Trained with gradient descent to minimize the logistic loss
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::LogisticRegression;
/// use ndarray::{Array1, Array2};
///
/// // Create a logistic regression model
/// let mut model = LogisticRegression::default();
///
/// // Training data with 2 features (x1, x2) for a logical AND function
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
    /// Model weights vector, `None` before training
    weights: Option<Array1<f64>>,
    /// Whether to use an intercept term (bias)
    fit_intercept: bool,
    /// Gradient descent step size
    learning_rate: f64,
    /// Maximum number of gradient descent iterations
    max_iter: usize,
    /// Convergence tolerance; iteration stops when the loss change is smaller than this value
    tol: f64,
    /// Number of iterations actually run after fitting, `None` before training
    n_iter: Option<usize>,
    /// Regularization type and strength
    regularization_type: Option<RegularizationType>,
}

impl Default for LogisticRegression {
    /// Creates a logistic regression model with default parameters
    ///
    /// # Default Values
    ///
    /// - `fit_intercept`: `true` - include a bias/intercept term
    /// - `learning_rate`: `0.01` - a moderate rate giving stable convergence for most problems
    /// - `max_iter`: `100` - maximum number of gradient descent iterations
    /// - `tol`: `1e-4` - convergence tolerance; training stops when the loss change between iterations is smaller than this value
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
    /// - `fit_intercept` - whether to add an intercept term (bias)
    /// - `learning_rate` - learning rate for gradient descent, must be positive and finite
    /// - `max_iterations` - maximum number of iterations, must be greater than 0
    /// - `tolerance` - convergence tolerance, stops when the loss change is below this value, must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - an untrained logistic regression model, or a validation error
    ///
    /// # Notes
    ///
    /// No regularization is applied by default. To add L1/L2 regularization, use
    /// [`with_regularization`](Self::with_regularization)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if any parameter is invalid (e.g. non-positive learning rate)
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_max_iterations(max_iterations)?;
        validate_tolerance(tolerance)?;

        Ok(LogisticRegression {
            weights: None,
            fit_intercept,
            learning_rate,
            max_iter: max_iterations,
            tol: tolerance,
            n_iter: None,
            regularization_type: None,
        })
    }

    /// Enables L1 or L2 regularization to prevent overfitting (default: no regularization)
    ///
    /// The penalty is added to the mean log-loss as `alpha * R(w)` (with `R = ||w||_1` for L1
    /// and `R = 0.5 * ||w||^2` for L2), not divided by the sample count
    ///
    /// # Parameters
    ///
    /// - `regularization` - the regularization variant and strength ([`RegularizationType::L1`] or [`RegularizationType::L2`])
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if the regularization alpha is negative or not finite
    pub fn with_regularization(
        mut self,
        regularization: RegularizationType,
    ) -> Result<Self, Error> {
        validate_regularization_type(Some(regularization))?;
        self.regularization_type = Some(regularization);
        Ok(self)
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
    /// Uses gradient descent to minimize the logistic loss function
    ///
    /// # Parameters
    ///
    /// - `x` - feature matrix where each row is a sample and each column is a feature
    /// - `y` - target variable containing 0 or 1 indicating sample class
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - a mutable reference to the trained model, or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - if the target vector contains values other than 0 or 1
    /// - `Error::NonFinite` - if numerical issues (NaN/Infinity) occur during training
    ///
    /// # Performance
    ///
    /// The per-iteration logits and gradient run as parallel GEMVs above their FLOPs
    /// gates, the sigmoid above the exp-map gate, and the loss as a deterministic blocked fold
    /// above its exp-reduction gate (see [`crate::math::logistic_loss`]), so re-running on the
    /// same machine reproduces the result (not necessarily bit-for-bit)
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        preliminary_check(x, Some(y))?;

        // Check target values are binary
        for &val in y.iter() {
            if val != 0.0 && val != 1.0 {
                return Err(Error::invalid_input(
                    "Target vector must contain only 0 or 1",
                ));
            }
        }

        let (n_samples, mut n_features) = x.dim();

        // Source of the training matrix depends on whether an intercept is used
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

        let mut weights = Array1::zeros(n_features);
        let mut prev_cost = f64::INFINITY;
        #[cfg(feature = "show_progress")]
        let mut final_cost = prev_cost;
        let mut n_iter = 0;

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

            // Linear predictions (the raw logits feed the loss below), then the sigmoid
            // activations (exp-map class gate: 1 f64 exp per element)
            let predictions = gemv_par_auto(&x_train_view, &weights);
            let mut sigmoid_preds = predictions.clone();
            if n_samples >= exp_map_f64_parallel_threshold() {
                sigmoid_preds.par_mapv_inplace(sigmoid);
            } else {
                sigmoid_preds.mapv_inplace(sigmoid);
            }

            let errors = &sigmoid_preds - y;

            let mut gradients = gemv_par_auto(&x_train_view.t(), &errors) / n_samples as f64;

            // Check for numerical issues in gradients
            if gradients.iter().any(|&val| !val.is_finite()) {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite gradients");
                return Err(Error::non_finite("gradient calculation"));
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
                            gradients[i] += regularization_strength * sign;
                        }
                    }
                    RegularizationType::L2(regularization_strength) => {
                        for i in start_idx..n_features {
                            gradients[i] += regularization_strength * weights[i];
                        }
                    }
                }
            }

            // Loss at the CURRENT weights (before this step's update)
            let mut cost = logistic_loss(&predictions, y);

            if let Some(reg_type) = &self.regularization_type {
                let start_idx = if self.fit_intercept { 1 } else { 0 };

                match reg_type {
                    RegularizationType::L1(regularization_strength) => {
                        let l1_penalty: f64 =
                            weights.slice(s![start_idx..]).mapv(|w| w.abs()).sum();
                        cost += regularization_strength * l1_penalty;
                    }
                    RegularizationType::L2(regularization_strength) => {
                        let l2_penalty: f64 = weights.slice(s![start_idx..]).mapv(|w| w * w).sum();
                        cost += regularization_strength * l2_penalty / 2.0;
                    }
                }
            }

            // In-place gradient step, avoiding a fresh weight array every iteration
            weights.scaled_add(-self.learning_rate, &gradients);

            // Check for numerical issues in updated weights
            if weights.iter().any(|&val| !val.is_finite()) {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite weights");
                return Err(Error::non_finite("weight update"));
            }

            #[cfg(feature = "show_progress")]
            {
                final_cost = cost;
            }

            // Check for numerical issues in cost
            if !cost.is_finite() {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite cost");
                return Err(Error::non_finite("cost calculation"));
            }

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
    /// Performs classification by applying a 0.5 threshold to probability values
    ///
    /// # Parameters
    ///
    /// - `x` - feature matrix where each row is a sample and each column is a feature (without bias term)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - a 1D array containing predicted class labels (0 or 1)
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - if the model has not been fitted yet
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::NonFinite` - if input is empty, dimensions mismatch, or contains invalid values
    /// - `Error::NonFinite` - if numerical issues occur during probability calculation
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64>,
    {
        // Probabilities (with full input validation) then a 0.5 decision threshold
        let probs = self.predict_proba(x)?;
        Ok(probs.mapv(|prob| if prob >= 0.5 { 1 } else { 0 }))
    }

    /// Predicts the positive-class probability for each sample
    ///
    /// Applies the sigmoid function to the linear decision values to produce
    /// probabilities in the range (0, 1)
    ///
    /// # Parameters
    ///
    /// - `x` - feature matrix where each row is a sample and each column is a feature (without the bias term)
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - probability of the positive class for each sample
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - if the model has not been fitted yet
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::NonFinite` - if input is empty, dimensions mismatch, or data contains non-finite values
    /// - `Error::NonFinite` - if numerical issues occur during probability calculation
    pub fn predict_proba<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| Error::not_fitted("LogisticRegression"))?;

        // Validate against trained feature count, excluding the implicit bias column when an intercept was fitted
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

        if probs.iter().any(|&val| !val.is_finite()) {
            return Err(Error::non_finite("probability calculation"));
        }

        Ok(probs)
    }

    /// Applies the sigmoid to the raw linear decision values `x.dot(weights)`
    ///
    /// `x` must already include the bias column when an intercept was fitted; this
    /// is an internal helper used by [`Self::predict_proba`]
    ///
    /// # Parameters
    ///
    /// - `x` - feature matrix, including the bias column when an intercept was fitted
    ///
    /// # Returns
    ///
    /// - `Array1<f64>` - positive-class probability for each sample
    fn sigmoid_decision<S>(&self, x: &ArrayBase<S, Ix2>) -> Array1<f64>
    where
        S: Data<Elem = f64>,
    {
        let weights = self.weights.as_ref().unwrap();
        let mut predictions = gemv_par_auto(x, weights);

        // Apply sigmoid with conditional parallelization (mutate in place; exp-map class)
        if predictions.len() >= exp_map_f64_parallel_threshold() {
            predictions.par_mapv_inplace(sigmoid);
        } else {
            predictions.mapv_inplace(sigmoid);
        }

        predictions
    }

    /// Fits the logistic regression model to the training data and then makes predictions
    ///
    /// Convenience method that combines `fit` and `predict` in a single call
    ///
    /// # Parameters
    ///
    /// - `train_x` - training features as a 2D array
    /// - `train_y` - target values as a 1D array corresponding to the training samples
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - predicted class labels for the training samples
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - if input data does not match expectations
    /// - `Error::NonFinite` - if numerical issues occur during fitting or prediction
    pub fn fit_predict<S>(
        &mut self,
        train_x: &ArrayBase<S, Ix2>,
        train_y: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit(train_x, train_y)?;
        self.predict(train_x)
    }

    model_save_and_load_methods!(LogisticRegression);
}

/// Generates polynomial features from input features
///
/// Transforms the input feature matrix into a new feature matrix containing
/// polynomial combinations of the input features up to the specified degree
///
/// # Parameters
///
/// - `x` - input feature matrix with shape (n_samples, n_features)
/// - `degree` - the maximum degree of polynomial features to generate
///
/// # Returns
///
/// - `Array2<f64>` - a new feature matrix of polynomial combinations of the input features, with shape (n_samples, n_output_features)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::machine_learning::{generate_polynomial_features, LogisticRegression};
///
/// // A simple dataset for binary classification
/// let training_x = array![[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 2.0], [2.5, 1.0]];
/// let training_y = array![0.0, 0.0, 1.0, 1.0, 0.0];
///
/// // Transform features to polynomial features
/// let poly_training_x = generate_polynomial_features(&training_x, 2);
///
/// // Train a logistic regression model with polynomial features
/// let mut model = LogisticRegression::default();
/// model.fit(&poly_training_x, &training_y).unwrap();
/// ```
pub fn generate_polynomial_features<S>(x: &ArrayBase<S, Ix2>, degree: usize) -> Array2<f64>
where
    S: Data<Elem = f64> + Send + Sync,
{
    let (n_samples, n_features) = x.dim();

    // Output feature count excludes the constant term: each degree d adds C(n+d-1, d) monomials
    let n_output_features = {
        let mut count = 0;
        for d in 1..=degree {
            let mut term = 1;
            for i in 0..d {
                term = term * (n_features + i) / (i + 1);
            }
            count += term;
        }
        count
    };

    // Result matrix, without a constant-term column
    let mut result = Array2::<f64>::zeros((n_samples, n_output_features));

    // Copy the original (first-order) features
    if n_samples.saturating_mul(n_features) >= cheap_map_f64_parallel_threshold() {
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_features {
                    row[j] = x[[i, j]];
                }
            });
    } else {
        result
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_features {
                    row[j] = x[[i, j]];
                }
            });
    }

    // For degree >= 2, add the higher-order features
    if degree >= 2 {
        let mut col_idx = n_features;

        // Recursive combination helper: the argument count is inherent to the recursion, so the lint is allowed
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
            // At the target degree, compute the feature value
            if current_degree == degree {
                // Capture the current column index, then increment for the next call
                let current_col = *col_idx;
                *col_idx += 1;

                // Cheap-map class
                if result.nrows().saturating_mul(combination.len())
                    >= cheap_map_f64_parallel_threshold()
                {
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
                } else {
                    result
                        .axis_iter_mut(Axis(0))
                        .enumerate()
                        .for_each(|(i, mut row)| {
                            let mut value = 1.0;
                            for &feat_idx in combination.iter() {
                                value *= x[[i, feat_idx]];
                            }
                            row[current_col] = value;
                        });
                }
                return;
            }

            // Build combinations recursively, sequentially since shared state is mutated
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

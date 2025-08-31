use crate::ModelError;
pub use crate::machine_learning::RegularizationType;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// # Linear Support Vector Classifier (LinearSVC)
///
/// Implements a classifier similar to sklearn's LinearSVC, trained using the hinge loss function.
/// Supports L1 and L2 regularization for preventing overfitting.
///
/// ## Fields
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
/// ## Features
///
/// - Binary classification
/// - Stochastic gradient descent optimization
/// - L1 or L2 regularization
/// - Configurable convergence tolerance
///
/// ## Example
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
    /// Weight coefficients for each feature.
    /// `None` if model is not trained yet.
    weights: Option<Array1<f64>>,

    /// Bias term (intercept) of the model.
    /// `None` if model is not trained yet.
    bias: Option<f64>,

    /// Maximum number of iterations for the optimizer.
    /// Higher values may lead to better convergence but longer training time.
    max_iter: usize,

    /// Learning rate (step size) for gradient descent.
    /// Controls how quickly the model adapts to the problem.
    learning_rate: f64,

    /// Regularization strength parameter.
    /// Higher values specify stronger regularization.
    regularization_param: f64,

    /// Regularization type: L1 or L2.
    /// L1 can produce sparse models, L2 typically works better for most problems.
    penalty: RegularizationType,

    /// Whether to calculate and use an intercept/bias term.
    /// If false, the decision boundary passes through origin.
    fit_intercept: bool,

    /// Training convergence tolerance.
    /// The algorithm stops when the improvement is less than this value.
    tol: f64,

    /// Number of iterations that were actually performed during training.
    /// `None` if model is not trained yet.
    n_iter: Option<usize>,
}

impl Default for LinearSVC {
    /// Creates a new LinearSVC with default parameters.
    ///
    /// ## Default values
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
    /// ## Returns
    /// A new LinearSVC instance with default parameters
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

    /// Returns the weight coefficients of the trained model.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)`: Reference to weight coefficients if model is trained
    /// - `Err(ModelError::NotFitted)`: If model hasn't been trained yet
    pub fn get_weights(&self) -> Result<&Array1<f64>, ModelError> {
        match &self.weights {
            Some(w) => Ok(w),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the bias term (intercept) of the trained model.
    ///
    /// # Returns
    ///
    /// - `Ok(f64)`: Bias value if model is trained
    /// - `Err(ModelError::NotFitted)`: If model hasn't been trained yet
    pub fn get_bias(&self) -> Result<f64, ModelError> {
        match &self.bias {
            Some(b) => Ok(*b),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the number of iterations performed during training.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)`: Number of iterations if model is trained
    /// - `Err(ModelError::NotFitted)`: If model hasn't been trained yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match &self.n_iter {
            Some(n) => Ok(*n),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Returns
    ///
    /// * `usize` - Maximum iterations the model will use during training
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the learning rate.
    ///
    /// # Returns
    ///
    /// * `f64` - Current learning rate value
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the regularization parameter.
    ///
    /// # Returns
    ///
    /// * `f64` - Current regularization strength value
    pub fn get_regularization_param(&self) -> f64 {
        self.regularization_param
    }

    /// Returns the regularization penalty type.
    ///
    /// # Returns
    ///
    /// * `&RegularizationType` - Reference to the current penalty type (L1 or L2)
    pub fn get_penalty(&self) -> &RegularizationType {
        &self.penalty
    }

    /// Returns whether the model uses an intercept/bias term.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the model uses a bias term, `false` otherwise
    pub fn get_fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Returns the convergence tolerance.
    ///
    /// # Returns
    ///
    /// * `f64` - Current tolerance value
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

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
        // Basic input validation
        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(format!(
                "Input data size mismatch: x.shape={}, y.shape={}",
                x.nrows(),
                y.len()
            )));
        }

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

        // Define mini-batch size, can be adjusted based on dataset size
        let batch_size = std::cmp::max(1, n_samples / 10);

        while n_iter < self.max_iter {
            n_iter += 1;
            // Randomly shuffle indices
            indices.shuffle(&mut rng);

            // Split data into batches
            for batch_indices in indices.chunks(batch_size) {
                // Use Rayon to compute gradients for each sample in parallel
                let gradients: Vec<(Array1<f64>, f64)> = batch_indices
                    .par_iter()
                    .map(|&idx| {
                        let xi = x.slice(ndarray::s![idx, ..]);
                        let yi = y_binary[idx];

                        // Calculate prediction
                        let margin = xi.dot(&weights) + bias;

                        // If misclassified or on the margin, calculate gradient
                        if yi * margin < 1.0 {
                            let weight_grad = xi.to_owned() * yi;
                            let bias_grad = if self.fit_intercept { yi } else { 0.0 };
                            (weight_grad, bias_grad)
                        } else {
                            (Array1::zeros(n_features), 0.0)
                        }
                    })
                    .collect();

                // Accumulate gradients and update weights
                let mut weight_update = Array1::<f64>::zeros(n_features);
                let mut bias_update = 0.0;

                for (w_grad, b_grad) in gradients {
                    weight_update = &weight_update + &w_grad;
                    bias_update += b_grad;
                }

                // Apply regularization and learning rate
                match self.penalty {
                    RegularizationType::L2(_) => {
                        weights = &weights * (1.0 - self.learning_rate * self.regularization_param)
                            + &(weight_update * (self.learning_rate / batch_indices.len() as f64));
                    }
                    RegularizationType::L1(_) => {
                        // L1 regularization subgradient update
                        weights = &weights
                            - &(weights.mapv(|w| {
                                if w > 0.0 {
                                    1.0
                                } else if w < 0.0 {
                                    -1.0
                                } else {
                                    0.0
                                }
                            }) * (self.learning_rate * self.regularization_param))
                            + &(weight_update * (self.learning_rate / batch_indices.len() as f64));
                    }
                }

                if self.fit_intercept {
                    bias += self.learning_rate * bias_update / batch_indices.len() as f64;
                }

                // Check for weight explosion - potential error source
                if weights.iter().any(|&w| !w.is_finite()) || !bias.is_finite() {
                    return Err(ModelError::ProcessingError(
                        "Weights became NaN or infinite during training. Try reducing learning_rate or regularization_param".to_string()
                    ));
                }
            }

            // Convergence check
            let diff = &weights - &prev_weights;
            let weight_diff =
                diff.iter().map(|&x| x * x).sum::<f64>().sqrt() / weights.len() as f64;
            let bias_diff = if self.fit_intercept {
                (bias - prev_bias).abs()
            } else {
                0.0
            };

            if weight_diff < self.tol && bias_diff < self.tol {
                break;
            }

            prev_weights = weights.clone();
            prev_bias = bias;
        }

        println!("Linear SVC model training finished at iteration {}", n_iter);

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
        let weights = match self.weights.as_ref() {
            Some(w) => w,
            None => return Err(ModelError::NotFitted),
        };
        let bias = self.bias.unwrap_or(0.0);

        // Check for empty input - potential error source
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Check feature dimension mismatch - potential error source
        if x.ncols() != weights.len() {
            return Err(ModelError::InputValidationError(format!(
                "Feature dimension mismatch: expected {}, got {}",
                weights.len(),
                x.ncols()
            )));
        }

        // Check for NaN/Inf values in input - potential error source
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input features contain NaN or infinite values".to_string(),
            ));
        }

        let decision = x.dot(weights) + bias;

        // Check for NaN/Inf in decision values - potential error source
        if decision.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::ProcessingError(
                "Decision function produced NaN or infinite values".to_string(),
            ));
        }

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
        let weights = match self.weights.as_ref() {
            Some(w) => w,
            None => return Err(ModelError::NotFitted),
        };
        let bias = self.bias.unwrap_or(0.0);

        Ok(x.dot(weights) + bias)
    }
}

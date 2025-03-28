use ndarray::{Array1, Array2, ArrayView2};
use crate::ModelError;

/// # Logistic Regression model implementation
///
/// This model uses gradient descent to train a binary classification logistic regression model.
///
/// ## Fields
///
/// * `weights` - Model weights vector, None before training
/// * `fit_intercept` - Whether to use intercept term (bias)
/// * `learning_rate` - Controls gradient descent step size
/// * `max_iter` - Maximum number of iterations for gradient descent
/// * `tol` - Convergence tolerance, stops iteration when loss change is smaller than this value
/// * `n_iter` - Actual number of iterations the algorithm ran for after fitting, None before training
///
/// ## Examples
///
/// ```
/// use rust_ai::machine_learning::logistic_regression::LogisticRegression;
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
/// model.fit(&x_train, &y_train);
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
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Model weights, None before training
    weights: Option<Array1<f64>>,
    /// Whether to use intercept term (bias)
    fit_intercept: bool,
    /// Learning rate, controls gradient descent step size
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance, stops iteration when loss change is smaller than this value
    tol: f64,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-4,
            n_iter: None,
        }
    }
}

impl LogisticRegression {
    /// Creates a new logistic regression model with specified parameters
    ///
    /// # Parameters
    ///
    /// * `fit_intercept` - Whether to add intercept term (bias)
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance, stops when loss change is below this value
    ///
    /// # Returns
    ///
    /// * `Self` - An untrained logistic regression model instance
    pub fn new(fit_intercept: bool,
               learning_rate: f64,
               max_iterations: usize,
               tolerance: f64,
    ) -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept,
            learning_rate,
            max_iter: max_iterations,
            tol : tolerance,
            n_iter: None,
        }
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

            // Apply sigmoid activation function
            for val in predictions.iter_mut() {
                *val = sigmoid(*val);
            }

            Ok(predictions)
        } else {
            Err(ModelError::NotFitted)
        }
    }

    /// Gets the current setting for fitting the intercept term
    ///
    /// # Returns
    ///
    /// * `bool` - Returns `true` if the model includes an intercept term, `false` otherwise
    pub fn get_fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Gets the current learning rate
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    ///
    /// # Returns
    ///
    /// * `f64` - The current learning rate value
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Gets the maximum number of iterations
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of iterations for the gradient descent algorithm
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Gets the convergence tolerance threshold
    ///
    /// The convergence tolerance is used to determine when to stop the training process.
    /// Training stops when the change in the loss function between consecutive iterations
    /// is less than this value.
    ///
    /// # Returns
    ///
    /// * `f64` - The current convergence tolerance value
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

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

    /// Get number of iterations the algorithm ran for after fitting
    ///
    /// # Returns
    ///
    /// - `usize` - number of iterations the algorithm ran for after fitting
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match self.n_iter {
            Some(n_iter) => Ok(n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Trains the logistic regression model
    ///
    /// Uses gradient descent to minimize the logistic loss function.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    /// * `y` - Target variable containing 0 or 1 indicating sample class
    ///
    /// # Returns
    ///
    /// * `&mut Self` - A mutable reference to the trained model, allowing for method chaining
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> &mut Self {
        use crate::math::sigmoid;
        let (n_samples, mut n_features) = x.dim();

        // If using intercept, add a column of ones
        let x_train = if self.fit_intercept {
            n_features += 1;
            let mut x_with_bias = Array2::ones((n_samples, n_features));
            for i in 0..n_samples {
                for j in 0..n_features - 1 {
                    x_with_bias[[i, j + 1]] = x[[i, j]];
                }
            }
            x_with_bias
        } else {
            x.clone()
        };

        // Initialize weights
        let mut weights = Array1::zeros(n_features);

        let mut prev_cost = f64::INFINITY;

        let mut final_cost = prev_cost;

        let mut n_iter = 0;

        // Gradient descent optimization
        while n_iter < self.max_iter {
            n_iter += 1;

            let predictions = x_train.dot(&weights);
            let mut sigmoid_preds = Array1::zeros(n_samples);

            for i in 0..n_samples {
                sigmoid_preds[i] = sigmoid(predictions[i]);
            }

            // Calculate error
            let errors = &sigmoid_preds - y;

            // Calculate gradients
            let gradients = x_train.t().dot(&errors) / n_samples as f64;

            // Update weights
            weights = &weights - self.learning_rate * &gradients;

            // Calculate loss function using math module's logistic_loss
            let raw_preds: Vec<f64> = x_train.dot(&weights)
                .iter()
                .copied()
                .collect();

            let y_vec: Vec<f64> = y.iter().copied().collect();
            let cost = crate::math::logistic_loss(&raw_preds, &y_vec);
            final_cost = cost;

            // Check convergence
            if (prev_cost - cost).abs() < self.tol {
                break;
            }

            prev_cost = cost;
        }

        self.weights = Some(weights);
        self.n_iter = Some(n_iter);

        // print training info
        println!("Logistic regression training finished at iteration {}, cost: {}",
                 n_iter, final_cost);

        self
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
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, ModelError> {
        let (n_samples, n_features) = x.dim();

        // Handle intercept term
        let x_test = if self.fit_intercept {
            let mut x_with_bias = Array2::ones((n_samples, n_features + 1));
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_with_bias[[i, j + 1]] = x[[i, j]];
                }
            }
            x_with_bias
        } else {
            x.clone()
        };

        let probs = self.predict_proba(&x_test.view());

        match probs {
            // Apply threshold (0.5) for classification
            Ok(probs) => Ok(probs.mapv(|prob| if prob >= 0.5 { 1 } else { 0 })),
            Err(e) => Err(e),
        }
    }

    /// Fits the logistic regression model to the training data and then makes predictions.
    ///
    /// This is a convenience method that combines `fit` and `predict` operations in a single call.
    ///
    /// # Arguments
    ///
    /// * `train_x` - Training features as a 2D array where each row represents a sample
    ///               and each column represents a feature
    /// * `train_y` - Target values as a 1D array corresponding to the training samples
    /// * `test_x` - Test features for which predictions are to be made
    ///
    /// # Returns
    ///
    /// * `Array1<i32>` - Predicted class labels for the test samples
    pub fn fit_predict(&mut self, 
                       train_x: &Array2<f64>, 
                       train_y: &Array1<f64>, 
                       test_x: &Array2<f64>
    ) -> Array1<i32> {
        self.fit(train_x, train_y);
        self.predict(test_x).unwrap()
    }
}

/// Generates polynomial features from input features.
///
/// This function transforms the input feature matrix into a new feature matrix containing
/// polynomial combinations of the input features up to the specified degree.
///
/// # Arguments
///
/// * `x` - Input feature matrix with shape (n_samples, n_features)
/// * `degree` - The maximum degree of polynomial features to generate
///
/// # Examples
/// Following codes show how this function works with `LogisticRegression`:
/// ```
/// use ndarray::array;
/// use rust_ai::machine_learning::logistic_regression::{generate_polynomial_features, LogisticRegression};
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
/// model.fit(&poly_training_x, &training_y);
/// ```
///
/// # Returns
///
/// * `Array2<f64>` - A new feature matrix containing polynomial combinations of the input features with shape (n_samples, n_output_features)
pub fn generate_polynomial_features(x: &Array2<f64>, degree: usize) -> Array2<f64> {
    let (n_samples, n_features) = x.dim();

    // Calculate the number of output features (including constant term)
    // Formula: C(n+d,d) = (n+d)!/(n!*d!) where n is feature count and d is degree
    let n_output_features = {
        let mut count = 1; // Constant term
        for d in 1..=degree {
            let mut term = 1;
            for i in 0..d {
                term = term * (n_features + i) / (i + 1);
            }
            count += term;
        }
        count
    };

    // Initialize result matrix with ones in the first column (constant term)
    let mut result = Array2::<f64>::ones((n_samples, n_output_features));

    // Add first-order features (original features)
    for i in 0..n_samples {
        for j in 0..n_features {
            result[[i, j+1]] = x[[i, j]];
        }
    }

    // If degree >= 2, add higher-order features
    if degree >= 2 {
        let mut col_idx = n_features + 1;

        // Define an inner recursive function to generate combinations
        fn add_combinations(
            x: &Array2<f64>,
            result: &mut Array2<f64>,
            col_idx: &mut usize,
            n_samples: usize,
            n_features: usize,
            degree: usize,
            current_degree: usize,
            start_feature: usize,
            combination: &mut Vec<usize>
        ) {
            // If we've reached the target degree, compute the feature value
            if current_degree == degree {
                for i in 0..n_samples {
                    let mut value = 1.0;
                    for &feat_idx in combination.iter() {
                        value *= x[[i, feat_idx]];
                    }
                    result[[i, *col_idx]] = value;
                }
                *col_idx += 1;
                return;
            }

            // Recursively build combinations
            for j in start_feature..n_features {
                combination.push(j);
                add_combinations(
                    x, result, col_idx, n_samples, n_features,
                    degree, current_degree + 1, j, combination
                );
                combination.pop();
            }
        }

        // Generate combinations for each degree
        for d in 2..=degree {
            add_combinations(x, &mut result, &mut col_idx, n_samples, n_features, d, 0, 0, &mut vec![]);
        }
    }

    result
}
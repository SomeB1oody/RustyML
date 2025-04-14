use crate::{math, ModelError};
use ndarray::{Array1, ArrayView2, ArrayView1};
use rayon::prelude::*;

/// # Linear Regression model implementation
///
/// Trains a simple linear regression model using gradient descent algorithm. This implementation
/// supports multivariate regression, optional intercept term, and allows adjustment of learning rate,
/// maximum iterations, and convergence tolerance.
///
/// ## Fields
///
/// - `coefficients` - Model coefficients (slopes), None before training
/// - `intercept` - Model intercept, None before training
/// - `fit_intercept` - Whether to include an intercept term in the model
/// - `learning_rate` - Learning rate for gradient descent
/// - `max_iter` - Maximum number of iterations for gradient descent
/// - `tol` - Convergence tolerance
/// - `n_iter` - Number of iterations the algorithm ran for after fitting
///
/// ## Examples
/// ```rust
/// use rustyml::machine_learning::linear_regression::LinearRegression;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);
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
/// model.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
/// let predictions = model.predict(new_data.view());
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients (slopes)
    coefficients: Option<Array1<f64>>,
    /// Intercept
    intercept: Option<f64>,
    /// Whether to fit an intercept
    fit_intercept: bool,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-5,
            n_iter: None,
        }
    }
}

impl LinearRegression {
    /// Creates a new linear regression model with custom parameters
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
            fit_intercept,
            learning_rate,
            max_iter: max_iterations,
            tol: tolerance,
            n_iter: None,
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

    /// Returns the model coefficients if the model has been fitted
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - A vector of model coefficients
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_coefficients(&self) -> Result<Array1<f64>, ModelError> {
        match &self.coefficients {
            Some(coeffs) => Ok(coeffs.clone()),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the intercept term if the model has been fitted
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - The intercept value
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_intercept(&self) -> Result<f64, ModelError> {
        match self.intercept {
            Some(intercept) => Ok(intercept),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the actual number of iterations performed during the last model fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` - The number of iterations if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match self.n_iter {
            Some(n_iter) => Ok(n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Fits the linear regression model using gradient descent
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix, each row is a sample, each column is a feature
    /// - `y` - Target variable vector
    ///
    /// # Returns
    ///
    /// - `Ok(&mut self)` - Returns mutable reference to self for method chaining
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        // Ensure x and y have the same number of samples
        use super::preliminary_check;

        preliminary_check(x, Some(y))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize parameters
        let mut weights = Array1::<f64>::zeros(n_features); // Initialize weights to zero
        let mut intercept = 0.0;                 // Initialize intercept to zero

        let mut prev_cost = f64::INFINITY;
        let mut final_cost = prev_cost;

        let mut n_iter = 0;

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            // Calculate predictions using parallel iterator
            let predictions: Array1<f64> = (0..n_samples).into_par_iter()
                .map(|i| {
                    let row = x.row(i);
                    let pred = row.dot(&weights) + if self.fit_intercept { intercept } else { 0.0 };
                    pred
                })
                .collect::<Vec<f64>>()
                .into();

            // Calculate mean squared error
            let sse = math::sum_of_squared_errors(predictions.view(), y.view())?;
            let cost = sse / (2.0 * n_samples as f64); // Mean squared error divided by 2
            final_cost = cost;

            // Calculate gradients in parallel
            let gradients_result: (Array1<f64>, f64) = (0..n_samples).into_par_iter()
                .map(|i| {
                    let error = predictions[i] - y[i];
                    let row = x.row(i);

                    // Compute weight gradients
                    let weight_grad = row.to_owned() * error;

                    // Compute intercept gradient
                    let intercept_grad = if self.fit_intercept { error } else { 0.0 };

                    (weight_grad, intercept_grad)
                })
                .reduce(
                    || (Array1::<f64>::zeros(n_features), 0.0),
                    |(mut acc_w, acc_i), (w_grad, i_grad)| {
                        acc_w += &w_grad;
                        (acc_w, acc_i + i_grad)
                    }
                );

            // Extract and normalize gradients
            let gradients = gradients_result.0 / (n_samples as f64);
            let intercept_gradient = gradients_result.1 / (n_samples as f64);

            // Update parameters
            weights -= &(&gradients * self.learning_rate);
            if self.fit_intercept {
                intercept -= self.learning_rate * intercept_gradient;
            }

            // Check convergence
            if (prev_cost - cost).abs() < self.tol {
                break;
            }

            prev_cost = cost;
        }

        // Save training results
        self.coefficients = Some(weights);
        self.intercept = Some(if self.fit_intercept { intercept } else { 0.0 });
        self.n_iter = Some(n_iter);

        // print training info
        println!("Linear regression model training finished at iteration {}, cost: {}",
                 n_iter, final_cost);

        Ok(self)
    }

    /// Makes predictions using the trained model
    ///
    /// # Parameters
    ///
    /// * `x` - Prediction data, each row is a sample, each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - A vector of predictions
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::InputValidationError)` - If number of features does not match training data
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        if self.coefficients.is_none() {
            return Err(ModelError::NotFitted);
        }

        let coeffs = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        if x.ncols() != coeffs.len() {
            return Err(ModelError::InputValidationError(
                format!("Number of features does not match training data, x columns: {}, coefficients: {}",
                        x.ncols(), coeffs.len())
            ));
        }

        let predictions = (0..x.nrows()).into_par_iter()
            .map(|i| {
                let row = x.row(i);
                intercept + row.dot(coeffs)
            })
            .collect::<Vec<f64>>();
        let predictions = Array1::from(predictions);

        Ok(predictions)
    }

    /// Fits the model to the training data and then makes predictions on the same data.
    ///
    /// This is a convenience method that combines the `fit` and `predict` methods into one call.
    ///
    /// # Parameters
    ///
    /// - `x` - The input features matrix where each inner vector represents a training example
    /// - `y` - The target values corresponding to each training example
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - The predicted values for the input data
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>, ModelError> {
        self.fit(x, y)?;
        Ok(self.predict(x)?)
    }
}
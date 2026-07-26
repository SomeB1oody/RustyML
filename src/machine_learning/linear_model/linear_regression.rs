//! Linear regression with gradient-descent and closed-form solvers
//!
//! Provides the [`LinearRegression`] model supporting multivariate regression, an
//! optional intercept term, and L1/L2 regularization

use crate::error::Error;
pub use crate::machine_learning::RegularizationType;
use crate::machine_learning::validation::{
    preliminary_check, validate_learning_rate, validate_max_iterations, validate_predict_input,
    validate_regularization_type, validate_tolerance,
};
use crate::math::matmul::matvec;
use crate::math::reduction::det_reduce;
use crate::parallel_gates::sum_f64_parallel_min_elems;
use crate::{Deserialize, Serialize};
use gemmkit_ndarray::Parallelism;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

/// Optimization strategy used to fit [`LinearRegression`], carrying that strategy's own settings
///
/// Each variant owns exactly the knobs it uses, so there is no way to hand a learning rate to the
/// closed form or to leave one unset for the iterative path. Validate a variant by passing it to
/// [`LinearRegression::with_solver`], which rejects an unusable configuration
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize, Serialize)]
pub enum LeastSquaresSolver {
    /// Closed-form normal-equation / ridge solution via SVD least squares
    ///
    /// The default, and the analogue of scikit-learn's `LinearRegression()`. Exact and
    /// hyperparameter-free, but it supports only no regularization or L2 - L1 has no closed form
    #[default]
    Normal,
    /// Iterative gradient descent, supporting L1, L2, or no regularization
    ///
    /// The analogue of a scikit-learn iterative solver such as `Ridge(solver="sag")`. Needed for
    /// L1, since [`Normal`](LeastSquaresSolver::Normal) cannot express it
    GradientDescent {
        /// Step size applied to each gradient update; must be positive and finite
        learning_rate: f64,
        /// Iteration cap; must be greater than 0
        max_iter: usize,
        /// Cost-change threshold below which the fit is treated as converged; must be positive
        /// and finite
        tol: f64,
    },
}

/// Linear regression model implementation
///
/// Trains a linear regression model using gradient descent. Supports multivariate regression, an
/// optional intercept term. The iteration settings - learning rate, iteration cap, convergence
/// tolerance - are carried by [`LeastSquaresSolver::GradientDescent`], the only strategy that uses them
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::*;
/// use rustyml::machine_learning::linear_model::LeastSquaresSolver;
/// use ndarray::{Array1, Array2};
///
/// // Create a linear regression model. `LeastSquaresSolver::GradientDescent` carries the settings only it
/// // uses; `LinearRegression::new(true)` alone would give the closed-form `LeastSquaresSolver::Normal`
/// let mut model = LinearRegression::new(true)
///     .with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 })
///     .unwrap();
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
/// model.save_to_path("linear_regression_model.bin").unwrap();
///
/// // Load the model from the file
/// let loaded_model = LinearRegression::load_from_path("linear_regression_model.bin").unwrap();
///
/// // Use the loaded model for predictions
/// let loaded_predictions = loaded_model.predict(&new_data);
///
/// // Clone is implemented, so the model can be copied
/// let model_copy = model.clone();
///
/// // Debug is implemented, so model details can be printed
/// println!("{:?}", model);
///
/// // Clean up the created file
/// std::fs::remove_file("linear_regression_model.bin").unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinearRegression {
    /// Model coefficients (slopes), `None` before training
    coefficients: Option<Array1<f64>>,
    /// Model intercept, `None` before training
    intercept: Option<f64>,
    /// Whether to include an intercept term in the model
    fit_intercept: bool,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
    /// Regularization type and strength
    regularization_type: Option<RegularizationType>,
    /// Optimization strategy, together with the settings that strategy uses
    solver: LeastSquaresSolver,
}

impl Default for LinearRegression {
    /// Creates a `LinearRegression` with default parameter values
    ///
    /// # Default Values
    ///
    /// - `fit_intercept` - `true` - include an intercept term in the linear model
    /// - `solver` - [`LeastSquaresSolver::Normal`] - the exact closed-form least-squares solution
    /// - `regularization_type` - `None` - no regularization applied by default
    ///
    /// Identical to [`LinearRegression::new(true)`](Self::new), so the two constructors always
    /// agree on which algorithm they build
    ///
    /// # scikit-learn parity
    ///
    /// The analogue of Python's `LinearRegression()`, which is exact OLS - hence
    /// [`LeastSquaresSolver::Normal`]
    ///
    /// # Returns
    ///
    /// - `LinearRegression` - a new instance with default parameters
    fn default() -> Self {
        Self::new(true)
    }
}

impl LinearRegression {
    /// Creates a linear regression model
    ///
    /// The solver and any regularization are chosen afterwards through
    /// [`with_solver`](Self::with_solver) and [`with_regularization`](Self::with_regularization).
    /// Each solver carries its own settings, so there is nothing here to validate and no `Result`
    /// to unwrap
    ///
    /// # Parameters
    ///
    /// - `fit_intercept` - whether to fit an intercept term
    ///
    /// # Returns
    ///
    /// - `Self` - a new instance using [`LeastSquaresSolver::Normal`] and no regularization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustyml::machine_learning::linear_model::{LinearRegression, LeastSquaresSolver};
    ///
    /// // Exact OLS, the default
    /// let ols = LinearRegression::new(true);
    ///
    /// // Gradient descent, which carries the settings only it uses
    /// let gd = LinearRegression::new(true)
    ///     .with_solver(LeastSquaresSolver::GradientDescent {
    ///         learning_rate: 0.01,
    ///         max_iter: 1000,
    ///         tol: 1e-6,
    ///     })
    ///     .unwrap();
    /// ```
    pub fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
            fit_intercept,
            n_iter: None,
            regularization_type: None,
            solver: LeastSquaresSolver::Normal,
        }
    }

    /// Selects the optimization strategy (default: [`LeastSquaresSolver::Normal`])
    ///
    /// The variant carries its own settings, so a [`LeastSquaresSolver::GradientDescent`] is configured in the
    /// same expression that selects it, and [`LeastSquaresSolver::Normal`] has nothing to configure. Those
    /// settings are validated here, which is why this returns `Result` while the other builders on
    /// this type do not
    ///
    /// [`LeastSquaresSolver::Normal`] supports only no regularization or L2; pairing it with L1 makes
    /// [`fit`](Self::fit) return an error, since L1 has no closed form
    ///
    /// # Parameters
    ///
    /// - `solver` - the optimization strategy, with its settings
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] - If a [`LeastSquaresSolver::GradientDescent`] carries a non-positive or
    ///   non-finite `learning_rate` or `tol`, or a `max_iter` of 0
    pub fn with_solver(mut self, solver: LeastSquaresSolver) -> Result<Self, Error> {
        if let LeastSquaresSolver::GradientDescent {
            learning_rate,
            max_iter,
            tol,
        } = solver
        {
            validate_learning_rate(learning_rate)?;
            validate_max_iterations(max_iter)?;
            validate_tolerance(tol)?;
        }
        self.solver = solver;
        Ok(self)
    }

    /// Enables L1 or L2 regularization to prevent overfitting (default: no regularization)
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

    get_field!(get_fit_intercept, fit_intercept, bool);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(
        get_regularization_type,
        regularization_type,
        Option<RegularizationType>
    );
    get_field_as_ref!(get_coefficients, coefficients, Option<&Array1<f64>>);
    get_field!(get_intercept, intercept, Option<f64>);
    get_field!(get_solver, solver, LeastSquaresSolver);

    /// Fits the linear regression model using the configured [`LeastSquaresSolver`]
    ///
    /// With [`LeastSquaresSolver::GradientDescent`] (the default), iteratively updates the coefficients and
    /// intercept to minimize the cost function, with early stopping once convergence is reached;
    /// with [`LeastSquaresSolver::Normal`], delegates to the closed-form ridge solution instead
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
    /// The prediction and gradient matrix-vector products run in parallel above an internal size
    /// gate. The SSE and intercept-gradient sums use deterministic blocked folds above the sum
    /// gate (see `crate::parallel_gates`), so re-running on the same machine reproduces the result
    /// (not necessarily bit-for-bit)
    pub fn fit<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        preliminary_check(x, Some(y.len()))?;

        // Closed-form path; the iterative one below unpacks its own settings
        let LeastSquaresSolver::GradientDescent {
            learning_rate,
            max_iter,
            tol,
        } = self.solver
        else {
            return self.fit_normal(x, y);
        };

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut weights = Array1::<f64>::zeros(n_features);
        let mut intercept = 0.0;

        let mut prev_cost = f64::INFINITY;
        // Track consecutive convergences for stability
        let mut convergence_count = 0;
        const CONVERGENCE_THRESHOLD: usize = 3;

        let mut n_iter = 0;

        // Pre-allocate to avoid repeated allocation
        let mut predictions = Array1::<f64>::zeros(n_samples);
        let mut error_vec = Array1::<f64>::zeros(n_samples);

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                max_iter as u64,
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
        while n_iter < max_iter {
            n_iter += 1;

            // Vectorized prediction
            predictions.assign(&matvec(x, &weights, Parallelism::Rayon(0)));
            if self.fit_intercept {
                predictions += intercept;
            }

            // Calculate errors once
            error_vec.assign(&(&predictions - y));

            // Cost (sum of squared errors) reuses the error vector: SSE = e dot e
            let sse = match error_vec.as_slice() {
                Some(slice) => det_reduce(
                    slice,
                    slice.len() >= sum_f64_parallel_min_elems(),
                    |block| block.iter().map(|v| v * v).sum::<f64>(),
                    |a, b| a + b,
                    0.0,
                ),
                // Non-contiguous storage: ndarray's serial kernel
                _ => error_vec.dot(&error_vec),
            };

            let regularization_term = match &self.regularization_type {
                None => 0.0,
                Some(RegularizationType::L1(alpha)) => {
                    alpha * weights.iter().map(|w| w.abs()).sum::<f64>()
                }
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
            let mut weight_gradients =
                matvec(&x.t(), &error_vec, Parallelism::Rayon(0)) / (n_samples as f64);
            let intercept_gradient = if self.fit_intercept {
                let error_sum = match error_vec.as_slice() {
                    Some(slice) => det_reduce(
                        slice,
                        slice.len() >= sum_f64_parallel_min_elems(),
                        |block| block.iter().sum::<f64>(),
                        |a, b| a + b,
                        0.0,
                    ),
                    // Non-contiguous storage: ndarray's serial kernel
                    _ => error_vec.sum(),
                };
                error_sum / (n_samples as f64)
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
            //
            // L1 is deliberately absent here: it is applied after the step by the proximal
            // operator below. Folding `alpha * sign(w)` into the gradient only lets a weight
            // approach zero asymptotically, so a sub-gradient "Lasso" never delivers the exact
            // zeros that make L1 a feature selector
            match &self.regularization_type {
                None | Some(RegularizationType::L1(_)) => {}
                Some(RegularizationType::L2(alpha)) => {
                    // d/dw [(alpha/2) * ||w||^2] = alpha * w, matching the cost term above
                    weight_gradients.scaled_add(*alpha, &weights);
                }
            }

            // Update parameters
            weights.scaled_add(-learning_rate, &weight_gradients);
            if self.fit_intercept {
                intercept -= learning_rate * intercept_gradient;
            }

            // Proximal step for L1 (ISTA): soft-thresholding by `learning_rate * alpha` is the
            // exact minimizer of `0.5 * ||w - v||^2 + learning_rate * alpha * ||w||_1`, so a
            // weight the data cannot justify lands on exactly 0.0 and stays there. The intercept
            // is left alone, since it carries no penalty
            if let Some(RegularizationType::L1(alpha)) = &self.regularization_type {
                let shrink = learning_rate * alpha;
                weights.mapv_inplace(|w| {
                    if w > shrink {
                        w - shrink
                    } else if w < -shrink {
                        w + shrink
                    } else {
                        0.0
                    }
                });
            }

            if weights.iter().any(|&val| !val.is_finite()) || !intercept.is_finite() {
                #[cfg(feature = "show_progress")]
                progress_bar.finish_with_message("Error: NaN or infinite parameters");
                return Err(Error::non_finite("parameter update"));
            }

            // Require several consecutive small cost changes before declaring convergence
            let cost_change = (prev_cost - cost).abs();
            if cost_change < tol {
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
        let convergence_status = if n_iter < max_iter {
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

    /// Fits the model with the closed-form normal-equation (ridge) solution
    ///
    /// Minimizes the same objective as the gradient-descent path,
    /// `(1/2n)||Xw + b - y||^2 + (alpha/2)||w||^2`, whose minimizer satisfies the ridge normal
    /// equations with effective penalty `lambda = n * alpha`. When `fit_intercept` is set, the
    /// features and target are mean-centered so the intercept is not penalized, and the
    /// intercept is recovered as `mean(y) - mean(x) . w`. The system is solved via an SVD least
    /// squares on the augmented design `[Xc; sqrt(lambda) I]`, which yields the minimum-norm
    /// solution even when `X^T X` is singular (e.g. collinear or wide data)
    fn fit_normal<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        let n_samples = x.nrows();

        // The penalty matches the GD objective
        let ridge_lambda = match &self.regularization_type {
            None => 0.0,
            Some(RegularizationType::L2(alpha)) => *alpha * n_samples as f64,
            Some(RegularizationType::L1(_)) => {
                return Err(Error::invalid_input(
                    "the Normal solver does not support L1 regularization (no closed form); \
                     use LeastSquaresSolver::GradientDescent",
                ));
            }
        };

        // Center features and target when fitting an intercept (keeps the intercept unpenalized)
        let (x_design, y_target, x_means, y_mean) = if self.fit_intercept {
            let x_means = x
                .mean_axis(Axis(0))
                .ok_or_else(|| Error::empty_input("feature matrix"))?;
            let y_mean = y.sum() / n_samples as f64;
            let xc = &x.to_owned() - &x_means;
            let yc = y.mapv(|v| v - y_mean);
            (xc, yc, Some(x_means), y_mean)
        } else {
            (x.to_owned(), y.to_owned(), None, 0.0)
        };

        let weights = solve_ridge_lstsq(&x_design, &y_target, ridge_lambda)?;

        let intercept = match &x_means {
            Some(x_means) => y_mean - x_means.dot(&weights),
            None => 0.0,
        };

        if weights.iter().any(|v| !v.is_finite()) || !intercept.is_finite() {
            return Err(Error::non_finite("closed-form solution"));
        }

        self.coefficients = Some(weights);
        self.intercept = Some(intercept);
        // Closed form: no iterative steps
        self.n_iter = Some(0);

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

        let mut predictions = matvec(x, coeffs, Parallelism::Rayon(0));
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
    pub fn fit_predict<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<Array1<f64>, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        self.fit(x, y)?;
        self.predict(x)
    }

    /// Returns the coefficient of determination R^2 of the prediction on `(x, y)`
    ///
    /// `R^2 = 1 - SS_res / SS_tot`, where `SS_res = sum((y_i - y_pred_i)^2)` is the residual
    /// sum of squares and `SS_tot = sum((y_i - mean(y))^2)` is the total sum of squares. The
    /// best possible score is `1.0`; a model that always predicts the mean of `y` scores `0.0`,
    /// and an arbitrarily worse model scores negative
    ///
    /// # Parameters
    ///
    /// - `x` - input features with samples as rows and features as columns
    /// - `y` - true target values aligned with the rows of `x`
    ///
    /// # Returns
    ///
    /// - `Result<f64, Error>` - the R^2 score
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - if the model has not been fitted
    /// - `Error::EmptyInput` / `Error::DimensionMismatch` - if inputs are empty or mismatched
    /// - `Error::NonFinite` - if `x` or `y` contain NaN or infinite values
    pub fn score<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<f64, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
    {
        // `predict` validates the model is fitted and that `x` is non-empty and finite
        let predictions = self.predict(x)?;

        if y.len() != predictions.len() {
            return Err(Error::dimension_mismatch(predictions.len(), y.len()));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(Error::non_finite("target vector"));
        }

        let y_mean = y.sum() / y.len() as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for (yi, pi) in y.iter().zip(predictions.iter()) {
            ss_res += (yi - pi).powi(2);
            ss_tot += (yi - y_mean).powi(2);
        }

        // Constant-target handling
        let r2 = if ss_tot != 0.0 {
            1.0 - ss_res / ss_tot
        } else if ss_res == 0.0 {
            1.0
        } else {
            0.0
        };
        Ok(r2)
    }

    model_save_and_load_methods!(LinearRegression);
}

/// Solves the ridge least-squares problem `min ||x w - y||^2 + ridge_lambda ||w||^2`
///
/// Stacks the design as `[x; sqrt(ridge_lambda) I]` with target `[y; 0]` and solves the
/// resulting least-squares system with an SVD, which yields the minimum-norm solution even
/// when `x` is rank-deficient (collinear or wide). `ridge_lambda == 0` reduces to ordinary
/// least squares
fn solve_ridge_lstsq(
    x: &Array2<f64>,
    y: &Array1<f64>,
    ridge_lambda: f64,
) -> Result<Array1<f64>, Error> {
    let n = x.nrows();
    let p = x.ncols();
    let extra = if ridge_lambda > 0.0 { p } else { 0 };
    let total_rows = n + extra;

    // Augmented design matrix D and target t
    let mut d = Array2::<f64>::zeros((total_rows, p));
    for i in 0..n {
        for j in 0..p {
            d[[i, j]] = x[[i, j]];
        }
    }
    if extra > 0 {
        let s = ridge_lambda.sqrt();
        for j in 0..p {
            d[[n + j, j]] = s;
        }
    }
    let mut t = Array1::<f64>::zeros(total_rows);
    for (i, &yi) in y.iter().enumerate() {
        t[i] = yi;
    }

    // SVD least-squares solve, yielding the minimum-norm solution
    crate::machine_learning::linalg::svd(&d, true, true).solve(&t, 1e-12)
}

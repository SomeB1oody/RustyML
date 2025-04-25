use crate::ModelError;
use crate::machine_learning::RegularizationType;

pub trait RegressorCommonGetterFunctions {
    /// Gets the current setting for fitting the intercept term
    ///
    /// # Returns
    ///
    /// * `bool` - Returns `true` if the model includes an intercept term, `false` otherwise
    fn get_fit_intercept(&self) -> bool;

    /// Gets the current learning rate
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    ///
    /// # Returns
    ///
    /// * `f64` - The current learning rate value
    fn get_learning_rate(&self) -> f64;

    /// Gets the maximum number of iterations
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of iterations for the gradient descent algorithm
    fn get_max_iterations(&self) -> usize;

    /// Gets the convergence tolerance threshold
    ///
    /// The convergence tolerance is used to determine when to stop the training process.
    /// Training stops when the change in the loss function between consecutive iterations
    /// is less than this value.
    ///
    /// # Returns
    ///
    /// * `f64` - The current convergence tolerance value
    fn get_tolerance(&self) -> f64;

    /// Returns the actual number of actual iterations performed during the last model fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` - The number of iterations if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    fn get_actual_iterations(&self) -> Result<usize, ModelError>;

    /// Returns a reference to the regularization type of the model
    ///
    /// This method provides access to the regularization configuration of the model,
    /// which can be None (no regularization), L1 (LASSO), or L2 (Ridge).
    ///
    /// # Returns
    ///
    /// * `&Option<RegularizationType>` - A reference to the regularization type, which will be None if no regularization is applied
    fn get_regularization_type(&self) -> &Option<RegularizationType>;
}

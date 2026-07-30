//! Linear models for regression and classification
//!
//! Groups [`LinearRegression`] for regression and [`LogisticRegression`] for binary
//! classification. [`LinearRegression`] supports gradient-descent and closed-form solvers.
//! [`LogisticRegression`] trains only by gradient descent and adds the
//! [`generate_polynomial_features`] helper

/// Linear regression with gradient-descent and closed-form solvers
pub mod linear_regression;
/// Logistic regression for binary classification
pub mod logistic_regression;

pub use linear_regression::{LeastSquaresSolver, LinearRegression};
pub use logistic_regression::{LogisticRegression, generate_polynomial_features};

//! Linear models trained by gradient descent
//!
//! Groups [`LinearRegression`] for regression and [`LogisticRegression`] for binary
//! classification (with its [`generate_polynomial_features`] helper)

/// Linear regression via gradient descent
pub mod linear_regression;
/// Logistic regression for binary classification
pub mod logistic_regression;

pub use linear_regression::{LinearRegression, Solver};
pub use logistic_regression::{LogisticRegression, generate_polynomial_features};

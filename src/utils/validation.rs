//! Shared input-validation helpers for the `utils` module
//!
//! Transformers (PCA, Kernel PCA, t-SNE) and preprocessing functions
//! (`normalize`, `standardize`) share the same checks: non-empty input,
//! no NaN/infinite values, matching feature counts. These helpers keep
//! the error messages consistent across the module
//!
//! Intentionally a separate, `utils`-local copy of the role played by
//! [`machine_learning::validation`](crate::machine_learning): the `utils`
//! feature does not depend on `machine_learning`, so the two cannot share
//! one module without coupling the features

use crate::error::Error;
use ndarray::{ArrayBase, Data, Dimension, Ix2};

/// Returns an error if `x` contains no elements
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty
pub(super) fn check_non_empty<S, D>(x: &ArrayBase<S, D>) -> Result<(), Error>
where
    S: Data,
    D: Dimension,
{
    if x.is_empty() {
        return Err(Error::empty_input("input data"));
    }
    Ok(())
}

/// Returns an error if `x` contains any NaN or infinite value
///
/// # Errors
///
/// - [`Error::NonFinite`] - If any element is not finite
pub(super) fn check_finite<S, D>(x: &ArrayBase<S, D>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::non_finite("input data"));
    }
    Ok(())
}

/// Returns an error if `x` has fewer than `min_samples` rows
///
/// `model` names the estimator so the message reads naturally (e.g. `"PCA
/// requires at least 2 samples"`)
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If `x` has fewer than `min_samples` rows
pub(super) fn check_min_samples<S>(
    x: &ArrayBase<S, Ix2>,
    min_samples: usize,
    model: &str,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.nrows() < min_samples {
        return Err(Error::invalid_input(format!(
            "{} requires at least {} samples",
            model, min_samples
        )));
    }
    Ok(())
}

/// Validates a feature matrix passed to a `fit`-style method
///
/// Checks, in order, that the matrix is non-empty, has at least one feature
/// column, and contains only finite values
///
/// # Parameters
///
/// - `x` - Feature matrix with samples as rows and features as columns
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty or has zero columns
/// - [`Error::NonFinite`] - If `x` contains non-finite values
pub(super) fn validate_fit_matrix<S>(x: &ArrayBase<S, Ix2>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    check_non_empty(x)?;
    if x.ncols() == 0 {
        return Err(Error::empty_input("features"));
    }
    check_finite(x)
}

/// Validates a feature matrix passed to a `transform`-style method
///
/// Checks that the matrix is non-empty, that its feature count matches the
/// fitted model, and that it contains only finite values
///
/// # Parameters
///
/// - `x` - Feature matrix with samples as rows and features as columns
/// - `expected_features` - Number of features the fitted model was trained on
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` is empty
/// - [`Error::DimensionMismatch`] - If `x` has a mismatched feature count
/// - [`Error::NonFinite`] - If `x` contains non-finite values
pub(super) fn validate_transform_matrix<S>(
    x: &ArrayBase<S, Ix2>,
    expected_features: usize,
) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    check_non_empty(x)?;
    if x.ncols() != expected_features {
        return Err(Error::dimension_mismatch(expected_features, x.ncols()));
    }
    check_finite(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use ndarray::Array2;

    // check_non_empty

    /// An empty array produces EmptyInput
    #[test]
    fn check_non_empty_empty_array_gives_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 3));
        let err = check_non_empty(&empty).unwrap_err();
        match err {
            Error::EmptyInput(_) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    /// A non-empty array returns Ok
    #[test]
    fn check_non_empty_non_empty_array_gives_ok() {
        let a: Array2<f64> = Array2::zeros((2, 3));
        assert!(check_non_empty(&a).is_ok());
    }

    // check_finite

    /// An array containing NaN produces NonFinite
    #[test]
    fn check_finite_nan_gives_non_finite() {
        let a = ndarray::array![[1.0, f64::NAN], [2.0, 3.0]];
        let err = check_finite(&a).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An array containing positive infinity produces NonFinite
    #[test]
    fn check_finite_inf_gives_non_finite() {
        let a = ndarray::array![[f64::INFINITY, 2.0]];
        let err = check_finite(&a).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An array containing negative infinity produces NonFinite
    #[test]
    fn check_finite_neg_inf_gives_non_finite() {
        let a = ndarray::array![[1.0, f64::NEG_INFINITY]];
        let err = check_finite(&a).unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }

    /// An array of finite values returns Ok
    #[test]
    fn check_finite_all_finite_gives_ok() {
        let a = ndarray::array![[0.0, -1.5, 1e10], [f64::MIN, f64::MAX, 0.0]];
        assert!(check_finite(&a).is_ok());
    }

    // check_min_samples

    /// Too few rows produces InvalidInput
    #[test]
    fn check_min_samples_too_few_gives_invalid_input() {
        let a: Array2<f64> = Array2::zeros((1, 3));
        let err = check_min_samples(&a, 2, "TestModel").unwrap_err();
        match err {
            Error::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    /// Rows equal to min_samples returns Ok
    #[test]
    fn check_min_samples_exactly_min_gives_ok() {
        let a: Array2<f64> = Array2::zeros((2, 3));
        assert!(check_min_samples(&a, 2, "TestModel").is_ok());
    }

    /// More rows than min_samples returns Ok
    #[test]
    fn check_min_samples_more_than_min_gives_ok() {
        let a: Array2<f64> = Array2::zeros((5, 3));
        assert!(check_min_samples(&a, 2, "TestModel").is_ok());
    }
}

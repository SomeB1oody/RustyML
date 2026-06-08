//! Shared input-validation helpers for the `utility` module.
//!
//! Every transformer (PCA, Kernel PCA, t-SNE) and preprocessing function
//! (`normalize`, `standardize`) needs the same handful of checks — non-empty
//! input, no NaN/infinite values, matching feature counts. These helpers are
//! the single source of truth for those checks so the error messages stay
//! consistent across the module.
//!
//! This mirrors the role of [`machine_learning::validation`](crate::machine_learning)
//! but is intentionally a separate, `utility`-local copy: the `utility` feature
//! does not depend on `machine_learning`, so the two cannot share one module
//! without coupling the features.

use crate::error::Error;
use ndarray::{ArrayBase, Data, Dimension, Ix2};

/// Returns an error if `x` contains no elements.
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

/// Returns an error if `x` contains any NaN or infinite value.
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

/// Returns an error if `x` has fewer than `min_samples` rows.
///
/// `model` names the estimator so the message reads naturally (e.g. `"PCA requires
/// at least 2 samples"`). Several transformers share the same minimum-sample guard,
/// differing only in that name.
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

/// Validates a feature matrix passed to a `fit`-style method.
///
/// Checks, in order, that the matrix is non-empty, has at least one feature
/// column, and contains only finite values.
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

/// Validates a feature matrix passed to a `transform`-style method.
///
/// Checks that the matrix is non-empty, that its feature count matches the
/// fitted model, and that it contains only finite values.
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

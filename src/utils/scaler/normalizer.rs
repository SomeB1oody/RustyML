//! Per-sample unit-norm scaling behind the estimator contract
//!
//! Provides [`Normalizer`], the `fit`/`transform` face of the stateless
//! [`normalize`](crate::utils::normalize::normalize) function restricted to rows. It is the one
//! transformer in this module that learns nothing statistical - a sample's norm comes from that
//! sample alone - so it exists to let row normalization take part in the same
//! [`Fit`](crate::traits::Fit) / [`Transform`](crate::traits::Transform) contract as the
//! per-feature scalers, and to pin the feature count so a mis-shaped batch is an error rather
//! than a silent wrong answer

use super::{validate_matrix, validate_transform_matrix};
use crate::error::Error;
use crate::utils::normalize::{NormalizationAxis, NormalizationOrder, normalize};
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayBase, Data, Ix2};

/// Scales each sample to unit norm
///
/// Rows are samples and columns are features. [`transform`](Self::transform) divides every row
/// by its own norm under the configured [`NormalizationOrder`], so only a sample's *direction*
/// survives - the transform a cosine-similarity or dot-product model wants. A row whose norm is
/// below `10 * f64::EPSILON` has no direction to rescale and is left exactly as it is
///
/// Because each row is scaled by its own norm, there is no cross-sample statistic to leak
/// across a train/test boundary: fitting is a formality that records the feature count.
/// Mirrors scikit-learn's `Normalizer`, whose `fit` is likewise a no-op
///
/// This is the row-wise counterpart to the per-feature scalers in this module. For column or
/// whole-array normalization, or for N-D arrays, use the free [`normalize`] function
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::Normalizer;
/// use rustyml::utils::normalize::NormalizationOrder;
///
/// let x = array![[3.0, 4.0], [1.0, 0.0]];
///
/// let mut normalizer = Normalizer::new(NormalizationOrder::L2).unwrap();
/// let z = normalizer.fit_transform(&x).unwrap();
///
/// assert_eq!(z, array![[0.6, 0.8], [1.0, 0.0]]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    /// Norm each sample is divided by
    order: NormalizationOrder,
    /// Number of features seen during fitting (scikit-learn's `n_features_in_`)
    n_features: Option<usize>,
}

impl Default for Normalizer {
    /// Creates an L2 normalizer, the usual choice for direction-only comparisons
    fn default() -> Self {
        Self {
            order: NormalizationOrder::L2,
            n_features: None,
        }
    }
}

impl Normalizer {
    /// Creates an unfitted normalizer using the given norm
    ///
    /// # Parameters
    ///
    /// - `order` - The norm each sample is divided by (L1, L2, Max, or a custom Lp)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new normalizer, or a validation error
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] - If `order` is `Lp(p)` with a non-positive or non-finite `p`
    pub fn new(order: NormalizationOrder) -> Result<Self, Error> {
        if matches!(order, NormalizationOrder::Lp(p) if p <= 0.0 || !p.is_finite()) {
            return Err(Error::invalid_parameter(
                "p",
                "Lp norm parameter must be positive and finite",
            ));
        }

        Ok(Self {
            order,
            n_features: None,
        })
    }

    // Getters
    get_field!(get_order, order, NormalizationOrder);
    get_field!(get_n_features, n_features, Option<usize>);

    /// Validates `x` and records its feature count
    ///
    /// Nothing statistical is learned: a sample's norm depends only on that sample, so this
    /// exists to complete the estimator contract and to fix the width that later
    /// [`transform`](Self::transform) calls are checked against
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - Mutable reference to self for chaining
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;

        self.n_features = Some(x.ncols());
        Ok(self)
    }

    /// Scales each row of `x` to unit norm
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new normalized matrix; `x` is not modified
    ///
    /// # Errors
    ///
    /// - [`Error::NotFitted`] - If the normalizer has not been fitted
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::DimensionMismatch`] - If `x`'s feature count differs from the fitted one
    /// - [`Error::NonFinite`] - If `x` contains non-finite values, or a norm overflows
    ///
    /// # Performance
    ///
    /// One norm pass plus one scaling pass per row, parallelized across rows above the
    /// calibrated scan gate (see `crate::parallel_gates`)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let n_features = self
            .n_features
            .ok_or_else(|| Error::not_fitted("Normalizer"))?;
        validate_transform_matrix(x, n_features)?;

        normalize(x, NormalizationAxis::Row, self.order)
    }

    /// Records `x`'s feature count and returns the normalized `x`
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new normalized matrix; `x` is not modified
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::NonFinite`] - If `x` contains non-finite values, or a norm overflows
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit(x)?;
        self.transform(x)
    }

    model_save_and_load_methods!(Normalizer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Each row is divided by its own L2 norm
    #[test]
    fn scales_rows_to_unit_l2_norm() {
        let x = array![[3.0, 4.0], [1.0, 0.0]];

        let mut normalizer = Normalizer::default();
        let z = normalizer.fit_transform(&x).unwrap();

        assert_eq!(z, array![[0.6, 0.8], [1.0, 0.0]]);
        assert_eq!(normalizer.get_n_features(), Some(2));
    }

    /// The norm order is honoured
    #[test]
    fn honours_the_configured_order() {
        let x = array![[3.0, 4.0]];

        let l1 = Normalizer::new(NormalizationOrder::L1)
            .unwrap()
            .fit_transform(&x)
            .unwrap();
        assert!((l1[[0, 0]] - 3.0 / 7.0).abs() < 1e-12);

        let max = Normalizer::new(NormalizationOrder::Max)
            .unwrap()
            .fit_transform(&x)
            .unwrap();
        assert_eq!(max, array![[0.75, 1.0]]);
    }

    /// An invalid Lp order is rejected at construction
    #[test]
    fn invalid_lp_order_is_rejected() {
        let err = Normalizer::new(NormalizationOrder::Lp(0.0)).unwrap_err();
        match err {
            Error::InvalidParameter { name, .. } => assert_eq!(name, "p"),
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    /// Row normalization is per-sample, so train and test batches agree by construction
    #[test]
    fn transform_is_independent_of_the_batch() {
        let x = array![[3.0, 4.0], [1.0, 1.0], [0.0, 5.0]];

        let mut normalizer = Normalizer::default();
        normalizer.fit(&x.slice(ndarray::s![0..2, ..])).unwrap();

        let whole = normalizer.transform(&x).unwrap();
        let single = normalizer.transform(&array![[0.0, 5.0]]).unwrap();

        assert_eq!(whole.row(2).to_owned(), single.row(0).to_owned());
    }

    /// A zero row has no direction to rescale and is left untouched
    #[test]
    fn zero_row_is_left_untouched() {
        let x = array![[3.0, 4.0], [0.0, 0.0]];

        let z = Normalizer::default().fit_transform(&x).unwrap();

        assert_eq!(z.row(1).to_owned(), array![0.0, 0.0]);
    }

    /// Transforming before fitting reports `NotFitted`
    #[test]
    fn transform_before_fit_gives_not_fitted() {
        let err = Normalizer::default().transform(&array![[1.0]]).unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "Normalizer"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }
}

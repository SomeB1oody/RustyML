//! Magnitude scaling that preserves zeros and signs
//!
//! Provides [`MaxAbsScaler`], which divides each feature by the largest absolute value seen at
//! fit time. It neither centers nor shifts, so a zero stays a zero and every sign survives.
//! This is the property that makes it the scaler of choice for data where zero means "absent".
//! Examples include count vectors, one-hot blocks, and TF-IDF matrices, where centering would
//! destroy that meaning

use super::{
    column_min_max, fitted, for_each_row, handle_zero_scale, validate_matrix,
    validate_transform_matrix,
};
use crate::error::Error;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Scales each feature by its maximum absolute value, learned from the training data
///
/// Rows are samples and columns are features. [`fit`](Self::fit) records each feature's
/// `max(|x|)`. [`transform`](Self::transform) divides by it, so every training value lands in
/// `[-1, 1]` and the feature's sign structure stays untouched. An all-zero feature has no
/// magnitude to divide by. The scaler forces its divisor to `1.0` instead, so the column stays
/// zero rather than becoming `NaN`. This mirrors scikit-learn's `MaxAbsScaler`
///
/// Prefer it over [`MinMaxScaler`](super::MinMaxScaler) when zero has to keep meaning "absent".
/// Min-max shifts every feature, which turns structural zeros into an arbitrary nonzero value
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::MaxAbsScaler;
///
/// let x_train = array![[1.0, -4.0], [0.0, 2.0], [-2.0, 0.0]];
///
/// let mut scaler = MaxAbsScaler::new();
/// let z = scaler.fit_transform(&x_train).unwrap();
///
/// // Divisors are 2 and 4. Zeros stay zero and signs survive
/// assert_eq!(scaler.get_max_abs().unwrap(), &array![2.0, 4.0]);
/// assert_eq!(z, array![[0.5, -1.0], [0.0, 0.5], [-1.0, 0.0]]);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaxAbsScaler {
    /// Per-feature maximum absolute value seen during fitting (scikit-learn's `max_abs_`)
    max_abs: Option<Array1<f64>>,
    /// Per-feature divisor: `max_abs`, or `1.0` for an all-zero feature (`scale_`)
    scale: Option<Array1<f64>>,
    /// Number of samples folded into the statistics so far (`n_samples_seen_`)
    n_samples_seen: usize,
}

impl MaxAbsScaler {
    /// Creates an unfitted scaler
    ///
    /// # Returns
    ///
    /// - `Self` - A new scaler with no learned magnitudes
    pub fn new() -> Self {
        Self::default()
    }

    // Getters
    get_field!(get_n_samples_seen, n_samples_seen, usize);
    get_field_as_ref!(get_max_abs, max_abs, Option<&Array1<f64>>);
    get_field_as_ref!(get_scale, scale, Option<&Array1<f64>>);

    /// Gets the number of features the scaler was fitted on (scikit-learn's `n_features_in_`)
    ///
    /// # Returns
    ///
    /// - `Option<usize>` - The feature count, or `None` if the scaler is not fitted
    #[inline]
    pub fn get_n_features(&self) -> Option<usize> {
        self.max_abs.as_ref().map(|max_abs| max_abs.len())
    }

    /// Fits the scaler, recording each feature's maximum absolute value
    ///
    /// This discards any magnitudes from a previous fit. Call this on the training matrix only.
    /// Fitting on the full dataset before splitting leaks test-set information into the
    /// transform
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
    ///
    /// # Performance
    ///
    /// One min/max pass per feature, where the magnitude is the wider end of the 2. The pass runs
    /// in parallel across features above the scan gate (see `crate::parallel_gates`). The result
    /// never depends on the thread count
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;

        self.max_abs = Some(column_max_abs(x));
        self.n_samples_seen = x.nrows();
        self.recompute_scale();
        Ok(self)
    }

    /// Folds another batch of samples into the recorded magnitudes
    ///
    /// Keeps the larger of the stored magnitude and the batch magnitude, per feature. A scaler can
    /// therefore be fitted over data that never exists in memory at once. On an unfitted scaler
    /// this behaves exactly like [`fit`](Self::fit). The result after `n` batches is identical to
    /// a single `fit` over their concatenation
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
    /// - [`Error::DimensionMismatch`] - If `x` has a different feature count than the previous
    ///   batches
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    pub fn partial_fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;
        if let Some(n_features) = self.get_n_features()
            && n_features != x.ncols()
        {
            return Err(Error::dimension_mismatch(n_features, x.ncols()));
        }

        let mut max_abs = column_max_abs(x);
        if let Some(seen) = &self.max_abs {
            max_abs.zip_mut_with(seen, |value, &seen| *value = value.max(seen));
        }

        self.max_abs = Some(max_abs);
        self.n_samples_seen += x.nrows();
        self.recompute_scale();
        Ok(self)
    }

    /// Divides `x` by the stored training magnitudes
    ///
    /// Applies `x / max_abs` feature by feature, using the numbers learned at fit time and
    /// never recomputing them from `x`. Values larger than anything seen during training leave
    /// `[-1, 1]`, which is how an out-of-range batch makes itself visible
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new scaled matrix. `x` is not modified
    ///
    /// # Errors
    ///
    /// - [`Error::NotFitted`] - If the scaler has not been fitted
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::DimensionMismatch`] - If `x`'s feature count differs from the fitted one
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    ///
    /// # Performance
    ///
    /// One pass per row, parallelized across rows above the calibrated cheap-map gate (see
    /// `crate::parallel_gates`)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let scale = fitted(&self.scale, "MaxAbsScaler")?;
        validate_transform_matrix(x, scale.len())?;

        let mut result = x.to_owned();
        for_each_row(&mut result, |mut row| {
            for (value, &s) in row.iter_mut().zip(scale) {
                *value /= s;
            }
        });

        Ok(result)
    }

    /// Fits the scaler on `x` and returns the scaled `x`
    ///
    /// Equivalent to [`fit`](Self::fit) followed by [`transform`](Self::transform). This is the
    /// call for the training matrix, after which [`transform`](Self::transform) handles every
    /// other batch
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new scaled matrix. `x` is not modified
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Multiplies scaled data back into the original units
    ///
    /// Applies `x * max_abs`, the exact inverse of [`transform`](Self::transform). An all-zero
    /// feature is the only degenerate case, and it round-trips too. Zero maps to zero either
    /// way
    ///
    /// # Parameters
    ///
    /// - `x` - Scaled matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new matrix in the original units. `x` is not modified
    ///
    /// # Errors
    ///
    /// - [`Error::NotFitted`] - If the scaler has not been fitted
    /// - [`Error::EmptyInput`] - If `x` has no rows or no columns
    /// - [`Error::DimensionMismatch`] - If `x`'s feature count differs from the fitted one
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    pub fn inverse_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let scale = fitted(&self.scale, "MaxAbsScaler")?;
        validate_transform_matrix(x, scale.len())?;

        let mut result = x.to_owned();
        for_each_row(&mut result, |mut row| {
            for (value, &s) in row.iter_mut().zip(scale) {
                *value *= s;
            }
        });

        Ok(result)
    }

    model_save_and_load_methods!(MaxAbsScaler);

    /// Recomputes the divisors from the recorded magnitudes, guarding an all-zero feature
    fn recompute_scale(&mut self) {
        self.scale = self
            .max_abs
            .as_ref()
            .map(|max_abs| max_abs.mapv(handle_zero_scale));
    }
}

/// Computes each feature's maximum absolute value
///
/// The magnitude is the wider end of the column's `(min, max)`, so this reuses the shared
/// extrema pass rather than sweeping the data a second time
fn column_max_abs<S>(x: &ArrayBase<S, Ix2>) -> Array1<f64>
where
    S: Data<Elem = f64>,
{
    column_min_max(&x.view())
        .into_iter()
        .map(|(lo, hi)| lo.abs().max(hi.abs()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The scaler divides every feature by its own magnitude, leaving zeros and signs intact
    #[test]
    fn divides_by_column_magnitude() {
        let x = array![[1.0, -4.0], [0.0, 2.0], [-2.0, 0.0]];

        let mut scaler = MaxAbsScaler::new();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(scaler.get_max_abs().unwrap(), &array![2.0, 4.0]);
        assert_eq!(z, array![[0.5, -1.0], [0.0, 0.5], [-1.0, 0.0]]);
        assert_eq!(scaler.get_n_samples_seen(), 3);
    }

    /// Training values all land within [-1, 1]
    #[test]
    fn training_values_stay_within_unit_magnitude() {
        let x = array![[10.0, -3.0], [-7.0, 1.0], [4.0, 2.0]];

        let z = MaxAbsScaler::new().fit_transform(&x).unwrap();

        assert!(z.iter().all(|v| v.abs() <= 1.0));
    }

    /// An all-zero feature gets a divisor of 1.0 and stays zero
    #[test]
    fn all_zero_feature_stays_zero() {
        let x = array![[0.0, 1.0], [0.0, 2.0]];

        let mut scaler = MaxAbsScaler::new();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
        assert!(z.column(0).iter().all(|&v| v == 0.0));
    }

    /// `partial_fit` keeps the larger magnitude and matches a single fit
    #[test]
    fn partial_fit_keeps_the_larger_magnitude() {
        let batch_a = array![[1.0, -9.0]];
        let batch_b = array![[-5.0, 2.0]];
        let full = array![[1.0, -9.0], [-5.0, 2.0]];

        let mut incremental = MaxAbsScaler::new();
        incremental.partial_fit(&batch_a).unwrap();
        incremental.partial_fit(&batch_b).unwrap();

        let mut single = MaxAbsScaler::new();
        single.fit(&full).unwrap();

        assert_eq!(incremental.get_max_abs(), single.get_max_abs());
        assert_eq!(incremental.get_n_samples_seen(), 2);
    }

    /// `inverse_transform` round-trips back to the original values
    #[test]
    fn inverse_transform_round_trips() {
        let x = array![[1.0, -5.0], [2.0, 7.5], [0.0, 0.5]];

        let mut scaler = MaxAbsScaler::new();
        let z = scaler.fit_transform(&x).unwrap();
        let restored = scaler.inverse_transform(&z).unwrap();

        for (original, back) in x.iter().zip(restored.iter()) {
            assert!((original - back).abs() < 1e-9);
        }
    }

    /// Transforming before fitting reports `NotFitted`
    #[test]
    fn transform_before_fit_gives_not_fitted() {
        let err = MaxAbsScaler::new().transform(&array![[1.0]]).unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "MaxAbsScaler"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }
}

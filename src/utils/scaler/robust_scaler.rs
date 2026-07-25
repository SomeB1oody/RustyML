//! Outlier-resistant scaling built on quantiles rather than moments
//!
//! Provides [`RobustScaler`], which centers on the median and divides by the interquartile
//! range instead of using the mean and standard deviation. Both statistics are order-based, so
//! a handful of extreme values moves them barely at all - where a single outlier can drag
//! [`StandardScaler`](super::StandardScaler)'s mean and inflate its standard deviation, or
//! stretch [`MinMaxScaler`](super::MinMaxScaler)'s denominator until every other sample is
//! squeezed into a sliver of the range

use super::{
    column_quantiles, fitted, for_each_row, handle_zero_scale, validate_matrix,
    validate_transform_matrix,
};
use crate::error::Error;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Scales features by their median and interquartile range
///
/// Rows are samples and columns are features. [`fit`](Self::fit) records each feature's median
/// and the spread between the two quantiles of `quantile_range` (the interquartile range by
/// default); [`transform`](Self::transform) applies `(x - center) / scale` with those stored
/// values. A feature whose quantile spread is degenerate gets a divisor of `1.0`, so it centers
/// to zeros rather than producing `NaN`
///
/// This is scikit-learn's `RobustScaler`, and it is the answer when the data has outliers you
/// do not want to remove: the median ignores how far away the extremes are, and the IQR
/// measures the bulk of the distribution rather than its tails. The trade-off is that the
/// output has no fixed range and no unit variance - only a comparable middle
///
/// # Differences from the rest of the family
///
/// - **No `partial_fit`.** Quantiles cannot be merged across batches the way moments and
///   extrema can, so the whole training matrix has to be in hand (scikit-learn has no
///   `partial_fit` here either)
/// - **No `unit_variance` option.** scikit-learn can additionally rescale so that normally
///   distributed features come out with unit variance; that needs the inverse normal CDF, which
///   this crate does not provide
/// - As with [`StandardScaler`](super::StandardScaler), `center_` and `scale_` are computed and
///   kept even when `with_centering` / `with_scaling` are `false`; the flags decide only what
///   [`transform`](Self::transform) applies
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::RobustScaler;
///
/// // Two identical features, except that column 1's last value is a wild outlier
/// let x = array![
///     [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
///     [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 1000.0],
/// ];
///
/// let mut scaler = RobustScaler::new();
/// let z = scaler.fit_transform(&x).unwrap();
///
/// // Both columns get the SAME center and scale: the outlier never enters either statistic
/// assert_eq!(scaler.get_center().unwrap(), &array![5.0, 5.0]);
/// assert_eq!(scaler.get_scale().unwrap(), &array![4.0, 4.0]);
///
/// // So the bulk of the data lands on a sane scale, and the outlier stays visible as one
/// assert!(z.rows().into_iter().take(8).all(|row| row.iter().all(|v| v.abs() <= 1.0)));
/// assert!(z[[8, 1]] > 200.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustScaler {
    /// Whether [`transform`](Self::transform) subtracts the median
    with_centering: bool,
    /// Whether [`transform`](Self::transform) divides by the quantile spread
    with_scaling: bool,
    /// Quantile percentages `(low, high)` whose spread becomes the divisor
    quantile_range: (f64, f64),
    /// Per-feature median learned during fitting (scikit-learn's `center_`)
    center: Option<Array1<f64>>,
    /// Per-feature divisor: the quantile spread, or `1.0` when it is degenerate (`scale_`)
    scale: Option<Array1<f64>>,
    /// Number of samples the statistics were computed over
    ///
    /// A RustyML addition for symmetry with the rest of the family; scikit-learn's
    /// `RobustScaler` has no such attribute
    n_samples_seen: usize,
}

impl Default for RobustScaler {
    /// Creates a scaler centering on the median and dividing by the IQR, equivalent to
    /// [`RobustScaler::new`]
    fn default() -> Self {
        Self::new()
    }
}

impl RobustScaler {
    /// Creates an unfitted scaler using the median and the interquartile range
    ///
    /// # Returns
    ///
    /// - `Self` - A new scaler with `quantile_range = (25.0, 75.0)`, centering and scaling on
    ///
    /// # Notes
    ///
    /// Either step can be switched off with [`with_centering`](Self::with_centering) and
    /// [`with_scaling`](Self::with_scaling); the quantiles themselves move with
    /// [`with_quantile_range`](Self::with_quantile_range)
    pub fn new() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            center: None,
            scale: None,
            n_samples_seen: 0,
        }
    }

    /// Sets whether [`transform`](Self::transform) subtracts the median (default: `true`)
    ///
    /// # Parameters
    ///
    /// - `with_centering` - `false` leaves the data uncentered
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_centering(mut self, with_centering: bool) -> Self {
        self.with_centering = with_centering;
        self
    }

    /// Sets whether [`transform`](Self::transform) divides by the quantile spread
    /// (default: `true`)
    ///
    /// # Parameters
    ///
    /// - `with_scaling` - `false` leaves the data unscaled
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_scaling(mut self, with_scaling: bool) -> Self {
        self.with_scaling = with_scaling;
        self
    }

    /// Sets the quantile percentages whose spread becomes the divisor (default: `(25.0, 75.0)`)
    ///
    /// A wider range (say `(10.0, 90.0)`) uses more of the distribution and reacts more to the
    /// tails; a narrower one is more resistant but rests on fewer samples
    ///
    /// # Parameters
    ///
    /// - `low` - Lower quantile as a percentage in `[0, 100)`
    /// - `high` - Upper quantile as a percentage in `(low, 100]`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] - If the bounds are not finite, out of `[0, 100]`, or
    ///   `low >= high`
    ///
    /// # Notes
    ///
    /// Unlike [`MinMaxScaler::with_feature_range`](super::MinMaxScaler::with_feature_range),
    /// this **discards any fitted statistics**: the stored quantiles were read at the old
    /// positions and cannot be moved without the training data. Refit after changing it
    pub fn with_quantile_range(mut self, low: f64, high: f64) -> Result<Self, Error> {
        if !low.is_finite() || !high.is_finite() || low < 0.0 || high > 100.0 || low >= high {
            return Err(Error::invalid_parameter(
                "quantile_range",
                format!(
                    "must satisfy 0 <= low < high <= 100, got ({}, {})",
                    low, high
                ),
            ));
        }

        self.quantile_range = (low, high);
        // The stored statistics were read at the previous positions, so they no longer apply
        self.center = None;
        self.scale = None;
        self.n_samples_seen = 0;
        Ok(self)
    }

    // Getters
    get_field!(get_with_centering, with_centering, bool);
    get_field!(get_with_scaling, with_scaling, bool);
    get_field!(get_quantile_range, quantile_range, (f64, f64));
    get_field!(get_n_samples_seen, n_samples_seen, usize);
    get_field_as_ref!(get_center, center, Option<&Array1<f64>>);
    get_field_as_ref!(get_scale, scale, Option<&Array1<f64>>);

    /// Gets the number of features the scaler was fitted on (scikit-learn's `n_features_in_`)
    ///
    /// # Returns
    ///
    /// - `Option<usize>` - The feature count, or `None` if the scaler is not fitted
    #[inline]
    pub fn get_n_features(&self) -> Option<usize> {
        self.center.as_ref().map(|center| center.len())
    }

    /// Fits the scaler, recording each feature's median and quantile spread
    ///
    /// Any statistics from a previous fit are discarded. Call this on the training matrix only -
    /// fitting on the full dataset before splitting leaks test-set information into the
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
    /// One sort per feature serves all three quantiles, parallelized across features above the
    /// calibrated scan gate (see `crate::parallel_gates`), so the statistics never depend on
    /// the thread count. This is the one scaler that copies a column at a time (a borrowed lane
    /// cannot be sorted in place), so it costs `threads * n_samples` of scratch memory
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;

        let (low, high) = self.quantile_range;
        let per_feature = column_quantiles(&x.view(), &[low / 100.0, 0.5, high / 100.0]);

        self.center = Some(per_feature.iter().map(|q| q[1]).collect());
        self.scale = Some(
            per_feature
                .iter()
                .map(|q| handle_zero_scale(q[2] - q[0]))
                .collect(),
        );
        self.n_samples_seen = x.nrows();
        Ok(self)
    }

    /// Scales `x` with the stored median and quantile spread
    ///
    /// Applies `(x - center) / scale` feature by feature, using the numbers learned at fit time
    /// and never recomputing them from `x`
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new scaled matrix; `x` is not modified
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
    /// One fused pass per row (subtract and divide together), parallelized across rows above
    /// the calibrated cheap-map gate (see `crate::parallel_gates`)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let (center, scale) = self.fitted_stats()?;
        validate_transform_matrix(x, center.len())?;

        let mut result = x.to_owned();
        let (with_centering, with_scaling) = (self.with_centering, self.with_scaling);
        for_each_row(&mut result, |mut row| {
            match (with_centering, with_scaling) {
                (true, true) => {
                    for ((value, &c), &s) in row.iter_mut().zip(center).zip(scale) {
                        *value = (*value - c) / s;
                    }
                }
                (true, false) => {
                    for (value, &c) in row.iter_mut().zip(center) {
                        *value -= c;
                    }
                }
                (false, true) => {
                    for (value, &s) in row.iter_mut().zip(scale) {
                        *value /= s;
                    }
                }
                (false, false) => {}
            }
        });

        Ok(result)
    }

    /// Fits the scaler on `x` and returns the scaled `x`
    ///
    /// Equivalent to [`fit`](Self::fit) followed by [`transform`](Self::transform); this is the
    /// call for the training matrix, after which [`transform`](Self::transform) handles every
    /// other batch
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new scaled matrix; `x` is not modified
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

    /// Maps scaled data back to the original feature space
    ///
    /// Applies `x * scale + center`, the exact inverse of [`transform`](Self::transform). A
    /// feature whose quantile spread was degenerate comes back as its median
    ///
    /// # Parameters
    ///
    /// - `x` - Scaled matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new matrix in the original units; `x` is not modified
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
        let (center, scale) = self.fitted_stats()?;
        validate_transform_matrix(x, center.len())?;

        let mut result = x.to_owned();
        let (with_centering, with_scaling) = (self.with_centering, self.with_scaling);
        for_each_row(&mut result, |mut row| {
            match (with_centering, with_scaling) {
                (true, true) => {
                    for ((value, &c), &s) in row.iter_mut().zip(center).zip(scale) {
                        *value = *value * s + c;
                    }
                }
                (true, false) => {
                    for (value, &c) in row.iter_mut().zip(center) {
                        *value += c;
                    }
                }
                (false, true) => {
                    for (value, &s) in row.iter_mut().zip(scale) {
                        *value *= s;
                    }
                }
                (false, false) => {}
            }
        });

        Ok(result)
    }

    model_save_and_load_methods!(RobustScaler);

    /// Borrows the fitted center and scale, or reports that the scaler is unfitted
    fn fitted_stats(&self) -> Result<(&Array1<f64>, &Array1<f64>), Error> {
        Ok((
            fitted(&self.center, "RobustScaler")?,
            fitted(&self.scale, "RobustScaler")?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Reproduces the worked example from scikit-learn's `RobustScaler` documentation
    #[test]
    fn matches_scikit_learn_reference_example() {
        let x = array![[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]];

        let z = RobustScaler::new().fit_transform(&x).unwrap();

        let expected = array![[0.0, -2.0, 0.0], [-1.0, 0.0, 0.4], [1.0, 0.0, -1.6]];
        for (actual, want) in z.iter().zip(expected.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} != {want}");
        }
    }

    /// The median and IQR come out of the linear-interpolation quantile rule
    #[test]
    fn learns_median_and_iqr() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];

        let mut scaler = RobustScaler::new();
        scaler.fit(&x).unwrap();

        // Quantiles of [1, 2, 3, 4]: q25 = 1.75, median = 2.5, q75 = 3.25
        assert!((scaler.get_center().unwrap()[0] - 2.5).abs() < 1e-12);
        assert!((scaler.get_scale().unwrap()[0] - 1.5).abs() < 1e-12);
        assert_eq!(scaler.get_n_samples_seen(), 4);
        assert_eq!(scaler.get_n_features(), Some(1));
    }

    /// Replacing an extreme value leaves the robust statistics untouched, unlike the moments
    ///
    /// Note that this needs enough samples for the quantile positions to sit away from the
    /// tail: with only four rows, the 75th percentile interpolates straight into the outlier,
    /// which is the rule working as specified rather than a failure of robustness
    #[test]
    fn resists_an_outlier() {
        let clean = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let spoiled = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [1000.0]
        ];

        let mut on_clean = RobustScaler::new();
        on_clean.fit(&clean).unwrap();
        let mut on_spoiled = RobustScaler::new();
        on_spoiled.fit(&spoiled).unwrap();

        // Median 5 and IQR 4 on both: the extreme value never enters either statistic
        assert_eq!(on_clean.get_center(), on_spoiled.get_center());
        assert_eq!(on_clean.get_scale(), on_spoiled.get_scale());
        assert!((on_spoiled.get_center().unwrap()[0] - 5.0).abs() < 1e-12);
        assert!((on_spoiled.get_scale().unwrap()[0] - 4.0).abs() < 1e-12);

        // The mean, by contrast, moves from 5 to over 100
        let mean = spoiled.iter().sum::<f64>() / spoiled.len() as f64;
        assert!(mean > 100.0);
    }

    /// A custom quantile range widens the divisor and drops any previous fit
    #[test]
    fn custom_quantile_range() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let mut narrow = RobustScaler::new();
        narrow.fit(&x).unwrap();
        let iqr = narrow.get_scale().unwrap()[0];

        let mut wide = RobustScaler::new().with_quantile_range(10.0, 90.0).unwrap();
        assert!(wide.get_center().is_none(), "changing the range unfits");
        wide.fit(&x).unwrap();

        assert!(wide.get_scale().unwrap()[0] > iqr);
        assert_eq!(wide.get_quantile_range(), (10.0, 90.0));
    }

    /// An invalid quantile range is rejected
    #[test]
    fn invalid_quantile_range_is_rejected() {
        for (low, high) in [(75.0, 25.0), (25.0, 25.0), (-1.0, 75.0), (25.0, 101.0)] {
            let err = RobustScaler::new()
                .with_quantile_range(low, high)
                .unwrap_err();
            match err {
                Error::InvalidParameter { name, .. } => assert_eq!(name, "quantile_range"),
                other => panic!("expected InvalidParameter, got {:?}", other),
            }
        }
    }

    /// A constant feature has no spread, so it centers to zeros instead of dividing by zero
    #[test]
    fn constant_feature_maps_to_zeros() {
        let x = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];

        let mut scaler = RobustScaler::new();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
        assert!(z.column(0).iter().all(|&v| v == 0.0));
        assert!(z.iter().all(|v| v.is_finite()));
    }

    /// `inverse_transform` round-trips back to the original values
    #[test]
    fn inverse_transform_round_trips() {
        let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5], [4.0, 100.0]];

        let mut scaler = RobustScaler::new();
        let z = scaler.fit_transform(&x).unwrap();
        let restored = scaler.inverse_transform(&z).unwrap();

        for (original, back) in x.iter().zip(restored.iter()) {
            assert!((original - back).abs() < 1e-9);
        }
    }

    /// The flags switch centering and scaling off independently
    #[test]
    fn flags_control_what_transform_applies() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];

        let centered = RobustScaler::new()
            .with_scaling(false)
            .fit_transform(&x)
            .unwrap();
        assert!((centered[[0, 0]] + 1.5).abs() < 1e-12);

        let scaled = RobustScaler::new()
            .with_centering(false)
            .fit_transform(&x)
            .unwrap();
        assert!((scaled[[0, 0]] - 1.0 / 1.5).abs() < 1e-12);
    }

    /// A single sample has no spread at all and still transforms cleanly
    #[test]
    fn single_sample_fit() {
        let mut scaler = RobustScaler::new();
        scaler.fit(&array![[5.0, -2.0]]).unwrap();

        assert_eq!(scaler.get_center().unwrap(), &array![5.0, -2.0]);
        assert_eq!(scaler.get_scale().unwrap(), &array![1.0, 1.0]);
    }

    /// Transforming before fitting reports `NotFitted`
    #[test]
    fn transform_before_fit_gives_not_fitted() {
        let err = RobustScaler::new().transform(&array![[1.0]]).unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "RobustScaler"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }
}

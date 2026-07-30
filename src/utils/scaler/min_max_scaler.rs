//! Range scaling that remembers its training minimum and maximum
//!
//! Provides [`MinMaxScaler`], which maps each feature onto a fixed interval (`[0, 1]` by
//! default) using the extrema learned at fit time. Unlike a z-score it makes no distributional
//! assumption. It is the transform to reach for when a downstream component needs bounded
//! inputs. Examples include an image-style pipeline or an activation with a limited useful
//! domain

use super::{
    column_min_max, fitted, for_each_row, handle_zero_scale, validate_matrix,
    validate_transform_matrix,
};
use crate::error::Error;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Scales features onto a fixed range, learned from the training extrema
///
/// Rows are samples and columns are features. [`fit`](Self::fit) records each feature's minimum
/// and maximum. [`transform`](Self::transform) maps them onto `feature_range` (default
/// `[0.0, 1.0]`) with the affine map
///
/// ```text
/// x_std    = (x - data_min) / (data_max - data_min)
/// x_scaled = x_std * (range_max - range_min) + range_min
/// ```
///
/// A feature with no spread at all (`data_max == data_min`) would otherwise divide by zero.
/// The scaler forces its divisor to `1.0` instead, so the column lands flat on `range_min`
/// rather than producing `NaN`. This mirrors scikit-learn's `MinMaxScaler`, including the
/// `clip` option
///
/// Values outside the training extrema land outside `feature_range`. It shows that a later
/// batch left the range the model was trained on. Enable [`with_clip`](Self::with_clip) when
/// a downstream component genuinely requires bounded input
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::MinMaxScaler;
///
/// let x_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
///
/// let mut scaler = MinMaxScaler::new();
/// let z = scaler.fit_transform(&x_train).unwrap();
/// assert_eq!(z, array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
///
/// // The scaler maps this later batch using the TRAINING extrema, so it may exceed the range
/// let z_new = scaler.transform(&array![[4.0, 5.0]]).unwrap();
/// assert_eq!(z_new, array![[1.5, -0.25]]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Target interval `(min, max)` that the training range is mapped onto
    feature_range: (f64, f64),
    /// Whether [`transform`](Self::transform) clamps its output to `feature_range`
    clip: bool,
    /// Per-feature minimum seen during fitting (scikit-learn's `data_min_`)
    data_min: Option<Array1<f64>>,
    /// Per-feature maximum seen during fitting (scikit-learn's `data_max_`)
    data_max: Option<Array1<f64>>,
    /// Per-feature multiplier applied by `transform` (scikit-learn's `scale_`)
    scale: Option<Array1<f64>>,
    /// Per-feature offset applied by `transform` (scikit-learn's `min_`)
    min: Option<Array1<f64>>,
    /// Number of samples folded into the extrema so far (`n_samples_seen_`)
    n_samples_seen: usize,
}

impl Default for MinMaxScaler {
    /// Creates a scaler onto `[0.0, 1.0]`, equivalent to [`MinMaxScaler::new`]
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxScaler {
    /// Creates an unfitted scaler mapping onto `[0.0, 1.0]`
    ///
    /// # Returns
    ///
    /// - `Self` - A new scaler with `feature_range = (0.0, 1.0)` and clipping off
    ///
    /// # Notes
    ///
    /// Use [`with_feature_range`](Self::with_feature_range) for another interval, for example
    /// `(-1.0, 1.0)`, and [`with_clip`](Self::with_clip) to clamp out-of-range values
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            clip: false,
            data_min: None,
            data_max: None,
            scale: None,
            min: None,
            n_samples_seen: 0,
        }
    }

    /// Sets the interval the training range is mapped onto (default: `(0.0, 1.0)`)
    ///
    /// # Parameters
    ///
    /// - `min` - Lower end of the target interval
    /// - `max` - Upper end of the target interval, strictly greater than `min`
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] - If `min >= max`, or either bound is not finite
    pub fn with_feature_range(mut self, min: f64, max: f64) -> Result<Self, Error> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(Error::invalid_parameter(
                "feature_range",
                format!("must be finite with min < max, got ({}, {})", min, max),
            ));
        }
        self.feature_range = (min, max);
        // The stored multiplier and offset are derived from the range, so retarget them here
        // rather than forcing a refit on an already-fitted scaler
        self.recompute_affine();
        Ok(self)
    }

    /// Sets whether [`transform`](Self::transform) clamps its output to `feature_range`
    /// (default: `false`)
    ///
    /// # Parameters
    ///
    /// - `clip` - `true` clamps every transformed value into the target interval
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    ///
    /// # Notes
    ///
    /// Clipping is lossy: a clamped value cannot be recovered by
    /// [`inverse_transform`](Self::inverse_transform)
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    // Getters
    get_field!(get_feature_range, feature_range, (f64, f64));
    get_field!(get_clip, clip, bool);
    get_field!(get_n_samples_seen, n_samples_seen, usize);
    get_field_as_ref!(get_data_min, data_min, Option<&Array1<f64>>);
    get_field_as_ref!(get_data_max, data_max, Option<&Array1<f64>>);
    get_field_as_ref!(get_scale, scale, Option<&Array1<f64>>);
    get_field_as_ref!(get_min, min, Option<&Array1<f64>>);

    /// Gets the per-feature spread seen during fitting (scikit-learn's `data_range_`)
    ///
    /// # Returns
    ///
    /// - `Option<Array1<f64>>` - `data_max - data_min`, or `None` if the scaler is not fitted
    pub fn get_data_range(&self) -> Option<Array1<f64>> {
        match (&self.data_min, &self.data_max) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        }
    }

    /// Gets the number of features the scaler was fitted on (scikit-learn's `n_features_in_`)
    ///
    /// # Returns
    ///
    /// - `Option<usize>` - The feature count, or `None` if the scaler is not fitted
    #[inline]
    pub fn get_n_features(&self) -> Option<usize> {
        self.data_min.as_ref().map(|min| min.len())
    }

    /// Fits the scaler, recording each feature's minimum and maximum
    ///
    /// This discards any extrema from a previous fit. Call this on the training matrix only.
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
    /// One min/max pass per feature, parallelized across features above the calibrated scan
    /// gate (see `crate::parallel_gates`), so the extrema never depend on the thread count
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;

        let extrema = column_min_max(&x.view());
        self.data_min = Some(extrema.iter().map(|&(lo, _)| lo).collect());
        self.data_max = Some(extrema.iter().map(|&(_, hi)| hi).collect());
        self.n_samples_seen = x.nrows();
        self.recompute_affine();
        Ok(self)
    }

    /// Folds another batch of samples into the recorded extrema
    ///
    /// Widens the stored minimum and maximum to cover `x` as well. This lets a scaler fit over
    /// data that never exists in memory at once. On an unfitted scaler this behaves exactly
    /// like [`fit`](Self::fit). The result after `n` batches is identical to a single `fit`
    /// over their concatenation
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

        let extrema = column_min_max(&x.view());
        let mut data_min: Array1<f64> = extrema.iter().map(|&(lo, _)| lo).collect();
        let mut data_max: Array1<f64> = extrema.iter().map(|&(_, hi)| hi).collect();

        // Widen the stored interval to cover this batch as well
        if let (Some(seen_min), Some(seen_max)) = (&self.data_min, &self.data_max) {
            data_min.zip_mut_with(seen_min, |value, &seen| *value = value.min(seen));
            data_max.zip_mut_with(seen_max, |value, &seen| *value = value.max(seen));
        }

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.n_samples_seen += x.nrows();
        self.recompute_affine();
        Ok(self)
    }

    /// Maps `x` onto the target range using the stored training extrema
    ///
    /// Applies `x * scale + min` feature by feature, using the numbers learned at fit time and
    /// never recomputing them from `x`. Values beyond the training extrema fall outside
    /// `feature_range` unless [`with_clip`](Self::with_clip) is enabled
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
    /// One fused pass per row (multiply and add together), parallelized across rows above the
    /// calibrated cheap-map gate (see `crate::parallel_gates`)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        let scale = fitted(&self.scale, "MinMaxScaler")?;
        let offset = fitted(&self.min, "MinMaxScaler")?;
        validate_transform_matrix(x, scale.len())?;

        let mut result = x.to_owned();
        let (low, high) = self.feature_range;
        let clip = self.clip;
        for_each_row(&mut result, |mut row| {
            for ((value, &s), &o) in row.iter_mut().zip(scale).zip(offset) {
                let scaled = *value * s + o;
                *value = if clip {
                    scaled.clamp(low, high)
                } else {
                    scaled
                };
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

    /// Maps scaled data back to the original feature space
    ///
    /// Applies `(x - min) / scale`, the exact inverse of [`transform`](Self::transform). This
    /// holds as long as clipping did not discard information and the feature had a real spread
    /// at fit time. A flat feature comes back as its training minimum
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
        let scale = fitted(&self.scale, "MinMaxScaler")?;
        let offset = fitted(&self.min, "MinMaxScaler")?;
        validate_transform_matrix(x, scale.len())?;

        let mut result = x.to_owned();
        for_each_row(&mut result, |mut row| {
            for ((value, &s), &o) in row.iter_mut().zip(scale).zip(offset) {
                *value = (*value - o) / s;
            }
        });

        Ok(result)
    }

    model_save_and_load_methods!(MinMaxScaler);

    /// Recomputes the stored multiplier and offset from the extrema and the target range
    ///
    /// A feature whose spread is degenerate gets a divisor of `1.0` (see
    /// [`handle_zero_scale`]), which lands the whole column on `feature_range.0`
    fn recompute_affine(&mut self) {
        let (Some(data_min), Some(data_max)) = (&self.data_min, &self.data_max) else {
            return;
        };

        let (low, high) = self.feature_range;
        let span = high - low;
        let scale: Array1<f64> = data_min
            .iter()
            .zip(data_max)
            .map(|(&lo, &hi)| span / handle_zero_scale(hi - lo))
            .collect();
        let min: Array1<f64> = data_min
            .iter()
            .zip(&scale)
            .map(|(&lo, &s)| low - lo * s)
            .collect();

        self.scale = Some(scale);
        self.min = Some(min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The default range maps the training minimum to 0 and the maximum to 1
    #[test]
    fn default_range_maps_training_extrema_to_unit_interval() {
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let mut scaler = MinMaxScaler::new();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(z, array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
        assert_eq!(scaler.get_data_min().unwrap(), &array![1.0, 10.0]);
        assert_eq!(scaler.get_data_max().unwrap(), &array![3.0, 30.0]);
        assert_eq!(scaler.get_data_range().unwrap(), array![2.0, 20.0]);
    }

    /// The scaler honors a custom feature range end to end
    #[test]
    fn custom_feature_range() {
        let x = array![[0.0], [10.0]];

        let mut scaler = MinMaxScaler::new().with_feature_range(-1.0, 1.0).unwrap();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(z, array![[-1.0], [1.0]]);
        assert_eq!(scaler.get_feature_range(), (-1.0, 1.0));
    }

    /// `with_feature_range` rejects an invalid range
    #[test]
    fn invalid_feature_range_is_rejected() {
        for (low, high) in [(1.0, 1.0), (2.0, 1.0), (f64::NAN, 1.0)] {
            let err = MinMaxScaler::new()
                .with_feature_range(low, high)
                .unwrap_err();
            match err {
                Error::InvalidParameter { name, .. } => assert_eq!(name, "feature_range"),
                other => panic!("expected InvalidParameter, got {:?}", other),
            }
        }
    }

    /// Out-of-range values escape the target interval unless clipping is enabled
    #[test]
    fn clip_bounds_out_of_range_values() {
        let x_train = array![[1.0], [3.0]];

        let mut open = MinMaxScaler::new();
        open.fit(&x_train).unwrap();
        assert_eq!(open.transform(&array![[5.0]]).unwrap(), array![[2.0]]);

        let mut clipped = MinMaxScaler::new().with_clip(true);
        clipped.fit(&x_train).unwrap();
        assert_eq!(clipped.transform(&array![[5.0]]).unwrap(), array![[1.0]]);
        assert_eq!(clipped.transform(&array![[-3.0]]).unwrap(), array![[0.0]]);
    }

    /// A constant feature lands flat on the low end of the range instead of dividing by zero
    #[test]
    fn constant_feature_maps_to_range_start() {
        let x = array![[3.0, 1.0], [3.0, 3.0]];

        let mut scaler = MinMaxScaler::new().with_feature_range(-2.0, 2.0).unwrap();
        let z = scaler.fit_transform(&x).unwrap();

        assert!(z.iter().all(|v| v.is_finite()));
        assert_eq!(z.column(0).to_vec(), vec![-2.0, -2.0]);
    }

    /// `partial_fit` widens the interval and matches a single fit over the concatenation
    #[test]
    fn partial_fit_widens_the_interval() {
        let batch_a = array![[2.0, 5.0], [3.0, 4.0]];
        let batch_b = array![[1.0, 9.0]];
        let full = array![[2.0, 5.0], [3.0, 4.0], [1.0, 9.0]];

        let mut incremental = MinMaxScaler::new();
        incremental.partial_fit(&batch_a).unwrap();
        incremental.partial_fit(&batch_b).unwrap();

        let mut single = MinMaxScaler::new();
        single.fit(&full).unwrap();

        assert_eq!(incremental.get_data_min(), single.get_data_min());
        assert_eq!(incremental.get_data_max(), single.get_data_max());
        assert_eq!(incremental.get_n_samples_seen(), 3);
    }

    /// `inverse_transform` round-trips back to the original values
    #[test]
    fn inverse_transform_round_trips() {
        let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5]];

        let mut scaler = MinMaxScaler::new();
        let z = scaler.fit_transform(&x).unwrap();
        let restored = scaler.inverse_transform(&z).unwrap();

        for (original, back) in x.iter().zip(restored.iter()) {
            assert!((original - back).abs() < 1e-9);
        }
    }

    /// Transforming before fitting reports `NotFitted`
    #[test]
    fn transform_before_fit_gives_not_fitted() {
        let err = MinMaxScaler::new().transform(&array![[1.0]]).unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "MinMaxScaler"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }
}

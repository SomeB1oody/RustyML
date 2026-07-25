//! Stateful feature scaling that remembers its training statistics
//!
//! Provides [`StandardScaler`], the fit/transform counterpart to the stateless
//! [`standardize`](crate::utils::standardize::standardize) function: it learns the per-feature
//! mean and standard deviation once, on the training matrix, and reuses those frozen numbers
//! for every later batch - the test split, a validation fold, or a single sample arriving at
//! inference time. This is the scikit-learn `StandardScaler` contract, and it is what keeps a
//! train/test boundary honest: rescaling a test set by its own column statistics applies a
//! different linear map than the one the model was trained under

use crate::error::Error;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems};
use crate::utils::standardize::{WelfordState, scale_from_variance, welford_merge, welford_step};
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Data, Ix2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Standardizes features by removing the mean and scaling to unit variance
///
/// Rows are samples and columns are features. [`fit`](Self::fit) computes each feature's mean
/// and population standard deviation and stores them; [`transform`](Self::transform) applies
/// `(x - mean) / scale` using those stored values, whatever data it is handed. A feature whose
/// spread is within floating-point noise gets a scale of `1.0`, so a constant column maps to
/// zeros instead of `NaN`
///
/// The divisor is the **population** standard deviation (variance divided by `n`, not `n - 1`),
/// matching scikit-learn's `StandardScaler` (`ddof=0`), and the statistics come from the same
/// numerically stable Welford pass that
/// [`standardize`](crate::utils::standardize::standardize) uses - so
/// `StandardScaler::default().fit_transform(&x)` is bit-for-bit identical to
/// `standardize(&x, StandardizationAxis::Column)` on a 2-D array
///
/// # Deviations from scikit-learn
///
/// - Non-finite input is rejected up front with [`Error::NonFinite`]; scikit-learn's scaler
///   instead ignores `NaN` during `fit` and passes it through `transform`. Impute or drop
///   missing values before scaling
/// - `mean_` / `var_` / `scale_` are computed and kept even when `with_mean` / `with_std` are
///   `false` (scikit-learn sets the unused ones to `None`); the flags decide only what
///   [`transform`](Self::transform) applies, so the statistics stay available for inspection
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::StandardScaler;
///
/// let x_train = array![[1.0, 100.0], [2.0, 150.0], [3.0, 200.0]];
/// let x_test = array![[4.0, 250.0]];
///
/// // Fit ONCE on the training matrix, then reuse those statistics everywhere.
/// let mut scaler = StandardScaler::new();
/// let z_train = scaler.fit_transform(&x_train).unwrap();
/// let z_test = scaler.transform(&x_test).unwrap();
///
/// assert_eq!(z_train.dim(), (3, 2));
/// assert_eq!(z_test.dim(), (1, 2));
/// // The test row is scaled by the TRAINING mean/std, not its own
/// assert!((z_test[[0, 0]] - 2.449489742783178).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Whether [`transform`](Self::transform) centers the data
    with_mean: bool,
    /// Whether [`transform`](Self::transform) divides by the standard deviation
    with_std: bool,
    /// Per-feature mean learned during fitting (scikit-learn's `mean_`)
    mean: Option<Array1<f64>>,
    /// Per-feature population variance learned during fitting (scikit-learn's `var_`)
    var: Option<Array1<f64>>,
    /// Per-feature divisor: `sqrt(var)`, or `1.0` for a constant feature (`scale_`)
    scale: Option<Array1<f64>>,
    /// Number of samples folded into the statistics so far (`n_samples_seen_`)
    n_samples_seen: usize,
}

impl Default for StandardScaler {
    /// Creates a scaler that both centers and scales, equivalent to [`StandardScaler::new`]
    fn default() -> Self {
        Self::new()
    }
}

impl StandardScaler {
    /// Creates an unfitted scaler that centers and scales
    ///
    /// # Returns
    ///
    /// - `Self` - A new scaler with `with_mean = true` and `with_std = true`
    ///
    /// # Notes
    ///
    /// Either step can be switched off with the builder methods
    /// [`with_mean`](Self::with_mean) and [`with_std`](Self::with_std)
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean: None,
            var: None,
            scale: None,
            n_samples_seen: 0,
        }
    }

    /// Sets whether [`transform`](Self::transform) subtracts the training mean (default: `true`)
    ///
    /// # Parameters
    ///
    /// - `with_mean` - `false` leaves the data uncentered
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Sets whether [`transform`](Self::transform) divides by the training standard deviation
    /// (default: `true`)
    ///
    /// # Parameters
    ///
    /// - `with_std` - `false` leaves the data unscaled
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    // Getters
    get_field!(get_with_mean, with_mean, bool);
    get_field!(get_with_std, with_std, bool);
    get_field!(get_n_samples_seen, n_samples_seen, usize);
    get_field_as_ref!(get_mean, mean, Option<&Array1<f64>>);
    get_field_as_ref!(get_var, var, Option<&Array1<f64>>);
    get_field_as_ref!(get_scale, scale, Option<&Array1<f64>>);

    /// Gets the number of features the scaler was fitted on (scikit-learn's `n_features_in_`)
    ///
    /// # Returns
    ///
    /// - `Option<usize>` - The feature count, or `None` if the scaler is not fitted
    #[inline]
    pub fn get_n_features(&self) -> Option<usize> {
        self.mean.as_ref().map(|mean| mean.len())
    }

    /// Fits the scaler, computing and storing each feature's mean and standard deviation
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
    /// Each feature is folded independently in row order, parallelized across features above
    /// the calibrated scan gate (see `crate::parallel_gates`), so the statistics never depend
    /// on the thread count
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        validate_matrix(x)?;

        let states = column_states(&x.view());
        self.store(&states);
        Ok(self)
    }

    /// Folds another batch of samples into the existing statistics
    ///
    /// Merges the batch's per-feature moments into the stored ones (Chan et al.), so a scaler
    /// can be fitted over data that never exists in memory at once - streaming batches, or one
    /// chunk of a file at a time. On an unfitted scaler this behaves exactly like
    /// [`fit`](Self::fit); the mean and variance after `n` batches equal what a single `fit`
    /// over their concatenation would produce, up to floating-point rounding
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
    /// - [`Error::DimensionMismatch`] - If `x` has a different feature count than the previous batches
    /// - [`Error::NonFinite`] - If `x` contains NaN or infinite values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rustyml::utils::StandardScaler;
    ///
    /// let mut scaler = StandardScaler::new();
    /// scaler.partial_fit(&array![[1.0], [2.0]]).unwrap();
    /// scaler.partial_fit(&array![[3.0], [4.0]]).unwrap();
    ///
    /// assert_eq!(scaler.get_n_samples_seen(), 4);
    /// assert!((scaler.get_mean().unwrap()[0] - 2.5).abs() < 1e-12);
    /// ```
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

        let mut states = column_states(&x.view());

        // Merge the batch's moments into the stored ones, recovering each feature's sum of
        // squared deviations from the population variance it was stored as
        if let (Some(mean), Some(var)) = (&self.mean, &self.var) {
            let seen = self.n_samples_seen as f64;
            for (j, state) in states.iter_mut().enumerate() {
                *state = welford_merge((seen, mean[j], var[j] * seen), *state);
            }
        }

        self.store(&states);
        Ok(self)
    }

    /// Standardizes `x` with the stored training statistics
    ///
    /// Applies `(x - mean) / scale` feature by feature, using the numbers learned at fit time
    /// and never recomputing them from `x`. A single-row `x` is fine: it is scaled by the
    /// training statistics, which is exactly what inference needs
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix with samples as rows and features as columns
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A new standardized matrix; `x` is not modified
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
        let (mean, scale) = self.fitted_stats()?;
        validate_matrix(x)?;
        if mean.len() != x.ncols() {
            return Err(Error::dimension_mismatch(mean.len(), x.ncols()));
        }

        let mut result = x.to_owned();
        let (with_mean, with_std) = (self.with_mean, self.with_std);
        for_each_row(&mut result, |mut row| match (with_mean, with_std) {
            (true, true) => {
                for ((value, &m), &s) in row.iter_mut().zip(mean).zip(scale) {
                    *value = (*value - m) / s;
                }
            }
            (true, false) => {
                for (value, &m) in row.iter_mut().zip(mean) {
                    *value -= m;
                }
            }
            (false, true) => {
                for (value, &s) in row.iter_mut().zip(scale) {
                    *value /= s;
                }
            }
            (false, false) => {}
        });

        Ok(result)
    }

    /// Fits the scaler on `x` and returns the standardized `x`
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
    /// - `Result<Array2<f64>, Error>` - A new standardized matrix; `x` is not modified
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

    /// Maps standardized data back to the original feature space
    ///
    /// Applies `x * scale + mean`, the exact inverse of [`transform`](Self::transform) - useful
    /// for reading a model's outputs or a reconstruction back in the units the data arrived in.
    /// A feature that was constant at fit time cannot be recovered from its zeros: it comes
    /// back as its training mean
    ///
    /// # Parameters
    ///
    /// - `x` - Standardized matrix with samples as rows and features as columns
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
        let (mean, scale) = self.fitted_stats()?;
        validate_matrix(x)?;
        if mean.len() != x.ncols() {
            return Err(Error::dimension_mismatch(mean.len(), x.ncols()));
        }

        let mut result = x.to_owned();
        let (with_mean, with_std) = (self.with_mean, self.with_std);
        for_each_row(&mut result, |mut row| match (with_mean, with_std) {
            (true, true) => {
                for ((value, &m), &s) in row.iter_mut().zip(mean).zip(scale) {
                    *value = *value * s + m;
                }
            }
            (true, false) => {
                for (value, &m) in row.iter_mut().zip(mean) {
                    *value += m;
                }
            }
            (false, true) => {
                for (value, &s) in row.iter_mut().zip(scale) {
                    *value *= s;
                }
            }
            (false, false) => {}
        });

        Ok(result)
    }

    model_save_and_load_methods!(StandardScaler);

    /// Borrows the fitted mean and scale, or reports that the scaler is unfitted
    fn fitted_stats(&self) -> Result<(&Array1<f64>, &Array1<f64>), Error> {
        match (&self.mean, &self.scale) {
            (Some(mean), Some(scale)) => Ok((mean, scale)),
            _ => Err(Error::not_fitted("StandardScaler")),
        }
    }

    /// Replaces the stored statistics with those of `states`
    ///
    /// Every feature is folded over the same rows, so the sample count is shared
    fn store(&mut self, states: &[WelfordState]) {
        let n = states.first().map_or(0.0, |&(count, _, _)| count);

        self.mean = Some(states.iter().map(|&(_, mean, _)| mean).collect());
        self.var = Some(states.iter().map(|&(_, _, m2)| m2 / n).collect());
        self.scale = Some(
            states
                .iter()
                .map(|&(_, mean, m2)| scale_from_variance(m2 / n, mean, n))
                .collect(),
        );
        self.n_samples_seen = n as usize;
    }
}

/// Computes one Welford accumulator per feature, folding each column over the rows of `x`
///
/// Columns are independent, so the serial and parallel paths produce identical results - the
/// gate only decides who does the work
fn column_states(x: &ArrayView2<f64>) -> Vec<WelfordState> {
    let fold_lane = |lane: ArrayView1<f64>| {
        lane.iter()
            .fold((0.0, 0.0, 0.0), |acc, &value| welford_step(acc, value))
    };

    let lanes: Vec<ArrayView1<f64>> = x.lanes(Axis(0)).into_iter().collect();

    // Scan-class gate: one O(n_samples) Welford pass per feature, so the work is the element count
    if x.len() >= scan_f64_parallel_min_elems() {
        lanes.into_par_iter().map(fold_lane).collect()
    } else {
        lanes.into_iter().map(fold_lane).collect()
    }
}

/// Applies `row_op` to every row of `x` in place, on rayon above the cheap-map gate
fn for_each_row<F>(x: &mut Array2<f64>, row_op: F)
where
    F: Fn(ArrayViewMut1<f64>) + Send + Sync,
{
    if x.len() >= cheap_map_f64_parallel_threshold() {
        x.axis_iter_mut(Axis(0)).into_par_iter().for_each(row_op);
    } else {
        x.axis_iter_mut(Axis(0)).for_each(row_op);
    }
}

/// Rejects a matrix that is empty, featureless, or non-finite
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If `x` has no rows or no columns
/// - [`Error::NonFinite`] - If any element is NaN or infinite
fn validate_matrix<S>(x: &ArrayBase<S, Ix2>) -> Result<(), Error>
where
    S: Data<Elem = f64>,
{
    if x.nrows() == 0 {
        return Err(Error::empty_input("input data"));
    }
    if x.ncols() == 0 {
        return Err(Error::empty_input("features"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(Error::non_finite("input data"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::standardize::{StandardizationAxis, standardize};
    use ndarray::array;

    /// `fit_transform` agrees bit-for-bit with the stateless column standardization
    #[test]
    fn fit_transform_matches_standardize_column() {
        let x = array![[1.0, 2000.0], [2.0, 3000.0], [3.0, 4000.0], [4.0, 5000.0]];

        let scaled = StandardScaler::new().fit_transform(&x).unwrap();
        let stateless = standardize(&x, StandardizationAxis::Column).unwrap();

        assert_eq!(scaled, stateless);
    }

    /// `transform` reuses the training statistics rather than recomputing them per batch
    #[test]
    fn transform_reuses_training_statistics() {
        let x_train = array![[1.0], [2.0], [3.0]];
        let mut scaler = StandardScaler::new();
        scaler.fit(&x_train).unwrap();

        // Training mean 2, population std sqrt(2/3); a lone sample must map through those
        let z = scaler.transform(&array![[3.0]]).unwrap();
        let expected = (3.0 - 2.0) / (2.0f64 / 3.0).sqrt();
        assert!((z[[0, 0]] - expected).abs() < 1e-12);

        // Standardizing the batch on its own would have produced 0.0 instead
        assert!(z[[0, 0]] > 1.0);
    }

    /// `partial_fit` over two batches matches a single `fit` over their concatenation
    #[test]
    fn partial_fit_matches_single_fit() {
        let batch_a = array![[1.0, 10.0], [2.0, 20.0]];
        let batch_b = array![[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]];
        let full = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0]
        ];

        let mut incremental = StandardScaler::new();
        incremental.partial_fit(&batch_a).unwrap();
        incremental.partial_fit(&batch_b).unwrap();

        let mut single = StandardScaler::new();
        single.fit(&full).unwrap();

        assert_eq!(incremental.get_n_samples_seen(), 5);
        for j in 0..2 {
            assert!(
                (incremental.get_mean().unwrap()[j] - single.get_mean().unwrap()[j]).abs() < 1e-9
            );
            assert!(
                (incremental.get_var().unwrap()[j] - single.get_var().unwrap()[j]).abs() < 1e-9
            );
        }
    }

    /// `inverse_transform` round-trips back to the original values
    #[test]
    fn inverse_transform_round_trips() {
        let x = array![[1.0, -5.0], [2.0, 7.5], [3.0, 0.5]];
        let mut scaler = StandardScaler::new();
        let z = scaler.fit_transform(&x).unwrap();
        let restored = scaler.inverse_transform(&z).unwrap();

        for (original, back) in x.iter().zip(restored.iter()) {
            assert!((original - back).abs() < 1e-9);
        }
    }

    /// A constant feature gets a scale of 1.0 and maps to zeros, leaving other features alone
    #[test]
    fn constant_feature_maps_to_zeros() {
        let x = array![[3.0, 1.0], [3.0, 3.0], [3.0, 5.0]];
        let mut scaler = StandardScaler::new();
        let z = scaler.fit_transform(&x).unwrap();

        assert_eq!(scaler.get_scale().unwrap()[0], 1.0);
        assert!(z.column(0).iter().all(|&v| v == 0.0));
        assert!(z.column(1).iter().all(|v| v.is_finite()));
    }

    /// The `with_mean` / `with_std` flags switch off centering and scaling independently
    #[test]
    fn flags_control_what_transform_applies() {
        let x = array![[1.0], [2.0], [3.0]];

        let centered_only = StandardScaler::new()
            .with_std(false)
            .fit_transform(&x)
            .unwrap();
        assert_eq!(centered_only, array![[-1.0], [0.0], [1.0]]);

        let scaled_only = StandardScaler::new()
            .with_mean(false)
            .fit_transform(&x)
            .unwrap();
        let std = (2.0f64 / 3.0).sqrt();
        assert!((scaled_only[[0, 0]] - 1.0 / std).abs() < 1e-12);
    }

    /// Transforming before fitting reports `NotFitted`
    #[test]
    fn transform_before_fit_gives_not_fitted() {
        let err = StandardScaler::new()
            .transform(&array![[1.0, 2.0]])
            .unwrap_err();
        match err {
            Error::NotFitted(model) => assert_eq!(model, "StandardScaler"),
            other => panic!("expected NotFitted, got {:?}", other),
        }
    }

    /// A feature-count mismatch at transform time reports `DimensionMismatch`
    #[test]
    fn transform_feature_mismatch_gives_dimension_mismatch() {
        let mut scaler = StandardScaler::new();
        scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();

        let err = scaler.transform(&array![[1.0, 2.0, 3.0]]).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 2);
                assert_eq!(found, 3);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// Non-finite input is rejected before any statistics are computed
    #[test]
    fn non_finite_input_is_rejected() {
        let err = StandardScaler::new()
            .fit(&array![[1.0, f64::NAN], [3.0, 4.0]])
            .unwrap_err();
        match err {
            Error::NonFinite(_) => {}
            other => panic!("expected NonFinite, got {:?}", other),
        }
    }
}

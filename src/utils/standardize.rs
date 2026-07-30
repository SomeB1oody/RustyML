//! Feature standardization (zero mean, unit variance) along row, column, or global axes
//!
//! Provides [`standardize()`] and the [`StandardizationAxis`] selector, with sequential
//! and parallel paths chosen by data size
//!
//! This is the one-shot form. Every call recomputes the statistics from the array it is
//! handed. When the same statistics must apply to more than one batch, use
//! [`StandardScaler`](crate::utils::StandardScaler) instead. Examples include a test split
//! scaled by the training mean, or a single sample at inference time. It stores the
//! statistics at fit time

use crate::error::Error;
use crate::math::reduction::det_reduce;
use crate::parallel_gates::{
    cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems, sum_f64_parallel_min_elems,
};
use ndarray::{Array, ArrayBase, ArrayViewMut1, Axis, Data, Dimension};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;

/// Axis along which standardization is applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardizationAxis {
    /// Standardize each row independently
    Row,
    /// Standardize each column independently
    Column,
    /// Standardize the entire array as one dataset
    Global,
}

impl StandardizationAxis {
    /// Standardizes `data` in place along this axis
    ///
    /// Routes to whole-array standardization for [`StandardizationAxis::Global`] or to
    /// the per-lane path for [`StandardizationAxis::Row`] / [`StandardizationAxis::Column`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] - If row/column standardization is requested on an
    ///   array with fewer than 2 dimensions
    fn apply<D>(&self, data: &mut Array<f64, D>) -> Result<(), Error>
    where
        D: Dimension,
    {
        match self {
            StandardizationAxis::Global => standardize_global(data),
            StandardizationAxis::Row => standardize_lanes(data, 1, "Row standardization"),
            StandardizationAxis::Column => standardize_lanes(data, 2, "Column standardization"),
        }
    }
}

/// Standardizes data to have zero mean and unit variance
///
/// Subtracts the mean and divides by the standard deviation per the chosen axis, so the
/// standardized values have mean 0 and standard deviation 1
///
/// # Parameters
///
/// - `data` - Input array as `ArrayBase` with arbitrary dimensions and f64 elements
/// - `axis` - Axis along which to standardize (Row/Column/Global)
///
/// # Returns
///
/// - `Result<Array<f64, D>, Error>` - Standardized array with the same dimensions as the input
///
/// # Notes
///
/// The statistics come from `data` itself on every call, so standardizing a test set this way
/// scales it by its own column statistics. This is a different linear map than the one applied
/// to the training data. Fit a [`StandardScaler`](crate::utils::StandardScaler) on the training
/// matrix and `transform` the rest when the batches have to agree.
/// `StandardScaler::default().fit_transform(&x)` returns exactly what
/// [`StandardizationAxis::Column`] produces here
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::standardize::{standardize, StandardizationAxis};
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let result = standardize(&data, StandardizationAxis::Column).unwrap();
/// ```
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If the input array is empty
/// - [`Error::NonFinite`] - If the input contains NaN or infinite values
/// - [`Error::Computation`] - If the global-axis path has no values to standardize
///
/// # Performance
///
/// The global-axis path computes its moments through a blocked parallel reduction above the
/// sum gate in `crate::parallel_gates`. The reduction order can vary with the thread count, so
/// results stay close but are not always bit for bit identical
pub fn standardize<S, D>(
    data: &ArrayBase<S, D>,
    axis: StandardizationAxis,
) -> Result<Array<f64, D>, Error>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if data.is_empty() {
        return Err(Error::empty_input("Cannot standardize empty array"));
    }

    if data.iter().any(|&x| !x.is_finite()) {
        return Err(Error::non_finite("input data"));
    }

    let mut result = data.to_owned();
    axis.apply(&mut result)?;
    Ok(result)
}

/// Running (count, mean, sum-of-squared-deviations) statistics for Welford's online
/// algorithm, which computes mean and variance in a single numerically stable pass
pub(super) type WelfordState = (f64, f64, f64);

/// Folds one value into a Welford accumulator
#[inline]
pub(super) fn welford_step((count, mean, m2): WelfordState, x: f64) -> WelfordState {
    let count = count + 1.0;
    let delta = x - mean;
    let mean = mean + delta / count;
    let m2 = m2 + delta * (x - mean);
    (count, mean, m2)
}

/// Merges 2 Welford accumulators (Chan et al.), enabling a parallel one-pass reduction
/// and the incremental refits behind
/// [`StandardScaler::partial_fit`](crate::utils::StandardScaler::partial_fit)
#[inline]
pub(super) fn welford_merge(a: WelfordState, b: WelfordState) -> WelfordState {
    let (na, ma, m2a) = a;
    let (nb, mb, m2b) = b;
    if na == 0.0 {
        return b;
    }
    if nb == 0.0 {
        return a;
    }
    let n = na + nb;
    let delta = mb - ma;
    let mean = ma + delta * nb / n;
    let m2 = m2a + m2b + delta * delta * na * nb / n;
    (n, mean, m2)
}

/// Detects whether a lane is indistinguishable from constant
///
/// Uses the error bound of the 2-pass variance algorithm (Chan, Golub and LeVeque). A lane
/// counts as constant when `variance <= n*eps*variance + (n*mean*eps)^2`, with
/// `eps = f64::EPSILON`. The bound is variance-based and magnitude-relative, so it flags
/// features whose spread sits within floating-point noise regardless of their scale
#[inline]
fn is_constant_feature(variance: f64, mean: f64, n: f64) -> bool {
    let eps = f64::EPSILON;
    let upper_bound = n * eps * variance + (n * mean * eps).powi(2);
    variance <= upper_bound
}

/// Converts a lane's population variance and mean into the standardization divisor
///
/// Returns the raw `sqrt(variance)`, or `1.0` for a constant lane (see [`is_constant_feature`]).
/// This keeps a constant feature's centered values at zero instead of amplifying them through
/// a vanishing divisor
#[inline]
pub(super) fn scale_from_variance(variance: f64, mean: f64, n: f64) -> f64 {
    if is_constant_feature(variance, mean, n) {
        1.0
    } else {
        variance.sqrt()
    }
}

/// Standardizes the entire array as a single dataset
fn standardize_global<D>(data: &mut Array<f64, D>) -> Result<(), Error>
where
    D: Dimension,
{
    let n = data.len() as f64;

    if n == 0.0 {
        return Err(Error::computation("No values to standardize"));
    }

    let (_, mean, m2) = match data.as_slice() {
        Some(slice) => det_reduce(
            slice,
            slice.len() >= sum_f64_parallel_min_elems(),
            |block| {
                block
                    .iter()
                    .fold((0.0, 0.0, 0.0), |acc, &x| welford_step(acc, x))
            },
            welford_merge,
            (0.0, 0.0, 0.0),
        ),
        // Non-contiguous storage: plain flat fold
        _ => data
            .iter()
            .fold((0.0, 0.0, 0.0), |acc, &x| welford_step(acc, x)),
    };
    let scale = scale_from_variance(m2 / n, mean, n);

    // Cheap-map class gate for the transform pass
    if data.len() >= cheap_map_f64_parallel_threshold() {
        data.par_mapv_inplace(|x| (x - mean) / scale);
    } else {
        data.mapv_inplace(|x| (x - mean) / scale);
    }

    Ok(())
}

/// Computes the population mean and the standardization scale of a lane
///
/// The returned scale is the raw `sqrt(variance)`, or `1.0` for a constant lane (see
/// [`scale_from_variance`])
fn lane_mean_and_std(lane: &ArrayViewMut1<f64>) -> (f64, f64) {
    let n = lane.len() as f64;
    // Single stable pass (Welford) for both mean and population variance
    let (_, mean, m2) = lane
        .iter()
        .fold((0.0, 0.0, 0.0), |acc, &x| welford_step(acc, x));
    (mean, scale_from_variance(m2 / n, mean, n))
}

/// Standardizes each lane along the axis `axis_from_end` positions from the end.
/// `1` means the last axis (rows), `2` means the second-to-last axis (columns)
///
/// Parallelizes across lanes once there are enough of them. Each lane is then
/// processed sequentially, so the 2 levels never nest
fn standardize_lanes<D>(
    data: &mut Array<f64, D>,
    axis_from_end: usize,
    operation_name: &str,
) -> Result<(), Error>
where
    D: Dimension,
{
    let ndim = data.ndim();
    if ndim < 2 {
        return Err(Error::invalid_input(format!(
            "{} requires at least 2 dimensions",
            operation_name
        )));
    }
    // ndim >= 2 above and axis_from_end in {1, 2}, so the subtraction cannot underflow
    let axis = Axis(ndim - axis_from_end);

    let data_len = data.len();
    let mut lanes: Vec<ArrayViewMut1<f64>> = data.lanes_mut(axis).into_iter().collect();
    let process = |lane: &mut ArrayViewMut1<f64>| {
        let (mean, scale) = lane_mean_and_std(lane);
        lane.mapv_inplace(|x| (x - mean) / scale);
    };

    // Scan-class gate: one O(lane) Welford pass + map per lane, so the work is the element count
    if data_len >= scan_f64_parallel_min_elems() {
        lanes.par_iter_mut().for_each(process);
    } else {
        lanes.iter_mut().for_each(process);
    }
    Ok(())
}

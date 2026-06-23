//! Data normalization along configurable axes and norm orders
//!
//! Provides [`normalize()`] to scale arrays so the chosen norm equals 1 along each
//! row, column, or globally, with [`NormalizationAxis`] and [`NormalizationOrder`]
//! selecting the axis and norm (L1, L2, Max, or Lp)

use crate::error::Error;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems};
use ndarray::{Array, ArrayBase, ArrayViewMut1, Axis, Data, Dimension};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

/// Norm magnitude below which a lane is treated as (near-)constant and left unscaled
const NORM_CONSTANT_THRESHOLD: f64 = 10.0 * f64::EPSILON;

/// Axis along which normalization is applied
///
/// For arrays with 3 or more dimensions, `Row` and `Column` operate on the last two
/// axes: `Row` normalizes along axis N-1 (last axis) and `Column` along axis N-2
/// (second-to-last). The last axis holds features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationAxis {
    /// Normalize each row independently
    ///
    /// For 2D arrays, normalizes each row (samples); for N-D arrays (N>2), normalizes
    /// along the last axis (features). For shape (batch, height, width), along width
    Row,
    /// Normalize each column independently
    ///
    /// For 2D arrays, normalizes each column (features); for N-D arrays (N>2), normalizes
    /// along the second-to-last axis. For shape (batch, height, width), along height
    Column,
    /// Normalize the entire array as a single vector
    Global,
}

/// Order of the norm used for normalization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationOrder {
    /// L1 norm (Manhattan norm): sum of absolute values
    L1,
    /// L2 norm (Euclidean norm): square root of sum of squares
    L2,
    /// Max norm (infinity norm): maximum absolute value
    Max,
    /// Lp norm with custom p value: (sum of |x|^p)^(1/p)
    Lp(f64),
}

/// Normalize data along the specified axis using the given norm order
///
/// Divides each element by the norm computed along the chosen axis, so the norm of the
/// data along that axis equals 1. A lane whose norm is below `10 * f64::EPSILON` (a (near-)zero
/// lane) is left unchanged, matching scikit-learn's `normalize`, which treats such norms as 1.0
///
/// # Parameters
///
/// - `data` - Input array with arbitrary dimensions and f64 elements
/// - `axis` - Axis along which to normalize (Row/Column/Global)
/// - `order` - Norm order to use (L1/L2/Max/Lp)
///
/// # Returns
///
/// - `Result<Array<f64, D>, Error>` - Normalized array with the same dimensions as the input
///
/// # Errors
///
/// - [`Error::EmptyInput`] if the input array is empty
/// - [`Error::NonFinite`] if the input contains non-finite values (NaN/Inf)
/// - [`Error::InvalidParameter`] if the Lp norm parameter `p` is not positive and finite
/// - [`Error::NonFinite`] if the normalization computation produces non-finite values
///
/// # Performance
///
/// Row/column normalization runs across lanes in parallel once the per-lane scan work
/// clears the scan-class gate; global normalization divides in parallel above the
/// cheap-map gate (see `crate::parallel_gates`)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::normalize::{normalize, NormalizationAxis, NormalizationOrder};
///
/// let data = array![[3.0, 4.0], [1.0, 2.0]];
/// let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
/// // Each row has L2 norm = 1
/// ```
pub fn normalize<S, D>(
    data: &ArrayBase<S, D>,
    axis: NormalizationAxis,
    order: NormalizationOrder,
) -> Result<Array<f64, D>, Error>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if data.is_empty() {
        return Err(Error::empty_input("Cannot normalize empty array"));
    }

    if data.iter().any(|&x| !x.is_finite()) {
        return Err(Error::non_finite("input data"));
    }

    if matches!(order, NormalizationOrder::Lp(p) if p <= 0.0 || !p.is_finite()) {
        return Err(Error::invalid_parameter(
            "p",
            "Lp norm parameter must be positive and finite",
        ));
    }

    let mut result = data.to_owned();
    axis.apply(&mut result, order)?;
    Ok(result)
}

/// Divides a single lane by its norm in place
///
/// Matches scikit-learn's `normalize`: a lane is scaled unless its norm is below
/// [`NORM_CONSTANT_THRESHOLD`] (`10 * f64::EPSILON`), in which case the norm is treated as `1.0`
/// and the lane is left untouched (a (near-)zero lane has no meaningful direction to rescale)
fn normalize_lane(lane: &mut ArrayViewMut1<f64>, norm: f64) {
    if norm >= NORM_CONSTANT_THRESHOLD {
        lane.mapv_inplace(|x| x / norm);
    }
}

/// Normalizes the entire array as a single flat vector
fn normalize_global<D>(data: &mut Array<f64, D>, order: NormalizationOrder) -> Result<(), Error>
where
    D: Dimension,
{
    let norm = order.norm(data.iter().copied())?;

    if norm >= NORM_CONSTANT_THRESHOLD {
        // Cheap-map class gate
        if data.len() >= cheap_map_f64_parallel_threshold() {
            data.par_mapv_inplace(|x| x / norm);
        } else {
            data.mapv_inplace(|x| x / norm);
        }
    }

    Ok(())
}

/// Normalizes each lane along the axis `axis_from_end` positions from the end
/// (`1` = last axis, rows; `2` = second-to-last, columns)
///
/// Parallelizes across lanes once there are enough of them; each lane is then
/// processed sequentially, so the two levels never nest
///
/// # Errors
///
/// - [`Error::InvalidInput`] if `data` has fewer than 2 dimensions
/// - [`Error::NonFinite`] if a norm computation overflows
fn normalize_lanes<D>(
    data: &mut Array<f64, D>,
    axis_from_end: usize,
    order: NormalizationOrder,
    operation_name: &str,
) -> Result<(), Error>
where
    D: Dimension,
{
    let ndim = data.ndim();
    let data_len = data.len();
    if ndim < 2 {
        return Err(Error::invalid_input(format!(
            "{} requires at least 2 dimensions",
            operation_name
        )));
    }
    // ndim >= 2 guaranteed above, so ndim - axis_from_end cannot underflow for axis_from_end in {1, 2}
    let axis = Axis(ndim - axis_from_end);

    let mut lanes: Vec<ArrayViewMut1<f64>> = data.lanes_mut(axis).into_iter().collect();
    let process = |lane: &mut ArrayViewMut1<f64>| -> Result<(), Error> {
        let norm = order.norm(lane.iter().copied())?;
        normalize_lane(lane, norm);
        Ok(())
    };

    // Scan-class gate: one O(lane) norm + map per lane, so the work is the element count
    if data_len >= scan_f64_parallel_min_elems() {
        lanes.par_iter_mut().try_for_each(process)
    } else {
        lanes.iter_mut().try_for_each(process)
    }
}

impl NormalizationOrder {
    /// Computes this norm over a sequence of values
    ///
    /// Single source for each variant's norm formula, shared by the global and per-lane
    /// normalization paths
    ///
    /// # Errors
    ///
    /// - [`Error::NonFinite`] if the accumulation overflows to a non-finite value
    fn norm<I>(&self, values: I) -> Result<f64, Error>
    where
        I: Iterator<Item = f64>,
    {
        match *self {
            NormalizationOrder::L1 => {
                let norm: f64 = values.map(|x| x.abs()).sum();
                if norm.is_finite() {
                    Ok(norm)
                } else {
                    Err(Error::non_finite(
                        "L1 norm computation resulted in non-finite value",
                    ))
                }
            }
            NormalizationOrder::L2 => {
                let norm_squared: f64 = values.map(|x| x * x).sum();
                if norm_squared.is_finite() && norm_squared >= 0.0 {
                    Ok(norm_squared.sqrt())
                } else {
                    Err(Error::non_finite(
                        "L2 norm computation resulted in non-finite value",
                    ))
                }
            }
            NormalizationOrder::Max => {
                let norm = values.map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
                if norm.is_finite() && norm >= 0.0 {
                    Ok(norm)
                } else if norm == f64::NEG_INFINITY {
                    // empty iterator
                    Ok(0.0)
                } else {
                    Err(Error::non_finite(
                        "Max norm computation resulted in non-finite value",
                    ))
                }
            }
            NormalizationOrder::Lp(p) => {
                let sum: f64 = values.map(|x| x.abs().powf(p)).sum();
                if sum.is_finite() && sum >= 0.0 {
                    Ok(sum.powf(1.0 / p))
                } else {
                    Err(Error::non_finite(format!(
                        "Lp norm (p={}) computation resulted in non-finite value",
                        p
                    )))
                }
            }
        }
    }
}

impl NormalizationAxis {
    /// Normalizes `data` in place along this axis under the given norm order
    ///
    /// Routes to whole-array normalization for [`NormalizationAxis::Global`] or to the
    /// per-lane path for [`NormalizationAxis::Row`] / [`NormalizationAxis::Column`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if row/column normalization is requested on an array
    ///   with fewer than 2 dimensions
    /// - [`Error::NonFinite`] if a norm computation overflows
    fn apply<D>(&self, data: &mut Array<f64, D>, order: NormalizationOrder) -> Result<(), Error>
    where
        D: Dimension,
    {
        match self {
            NormalizationAxis::Global => normalize_global(data, order),
            NormalizationAxis::Row => normalize_lanes(data, 1, order, "Row normalization"),
            NormalizationAxis::Column => normalize_lanes(data, 2, order, "Column normalization"),
        }
    }
}

use crate::error::ModelError;
use ndarray::{Array, ArrayBase, ArrayViewMut1, Axis, Data, Dimension};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

/// Tolerance for considering a norm as effectively zero
const NORM_ZERO_THRESHOLD: f64 = 1e-15;

/// Element-count threshold above which the global-axis path divides in parallel.
const NORMALIZE_PARALLEL_THRESHOLD: usize = 10000;

/// Lane-count threshold above which row/column normalization runs across lanes in parallel.
const NORMALIZE_PARALLEL_LANES: usize = 100;

/// Defines the axis along which the normalization is applied
///
/// # Variants
///
/// - `Row` - Normalize across rows (each row is normalized independently)
///     - For 2D arrays: normalizes each row (samples) independently
///     - For N-D arrays (N>2): normalizes along the last axis (features dimension)
///     - Example: For shape (batch, height, width), normalizes along width axis
/// - `Column` - Normalize across columns (each column is normalized independently)
///     - For 2D arrays: normalizes each column (features) independently
///     - For N-D arrays (N>2): normalizes along the second-to-last axis
///     - Example: For shape (batch, height, width), normalizes along height axis
/// - `Global` - Normalize the entire array globally (treats all elements as a single vector)
///
/// # Note on High-Dimensional Arrays
///
/// For arrays with 3 or more dimensions, `Row` and `Column` operate on the last two axes:
/// - `Row`: normalizes along axis N-1 (last axis)
/// - `Column`: normalizes along axis N-2 (second-to-last axis)
///
/// This convention follows common machine learning practices where the last axis represents features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationAxis {
    Row,
    Column,
    Global,
}

/// Defines the order of the norm used for normalization
///
/// # Variants
///
/// - `L1` - L1 norm (Manhattan norm): sum of absolute values
/// - `L2` - L2 norm (Euclidean norm): square root of sum of squares  
/// - `Max` - Max norm (infinity norm): maximum absolute value
/// - `Lp(f64)` - Lp norm with custom p value: (sum of |x|^p)^(1/p)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationOrder {
    L1,
    L2,
    Max,
    Lp(f64),
}

/// Normalize data along specified axis using the given norm order
///
/// This function normalizes input data by dividing each element by the norm
/// computed along the specified axis. The normalization ensures that the
/// norm of the data along the chosen axis equals 1.
///
/// # Parameters
///
/// - `data` - Input array data with arbitrary dimensions and f64 elements
/// - `axis` - The axis along which to perform normalization (Row/Column/Global)
/// - `order` - The norm order to use for normalization (L1/L2/Max/Lp)
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::normalize::{normalize, NormalizationAxis, NormalizationOrder};
///
/// let data = array![[3.0, 4.0], [1.0, 2.0]];
/// let result = normalize(&data, NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
/// // Each row will have L2 norm = 1
/// ```
///
/// # Returns
///
/// - `Result<Array<f64, D>, ModelError>` - Normalized array with same dimensions as input
///
/// # Errors
///
/// - Returns `ModelError::InputValidationError` if the input array is empty, contains non-finite values (NaN/Inf), or if the Lp norm parameter `p` is not positive/finite.
/// - Returns `ModelError::ProcessingError` if the normalization computation results in non-finite values.
///
/// # Performance
///
/// - Row/column normalization runs across lanes in parallel once there are at least
///   `NORMALIZE_PARALLEL_LANES` (100) lanes; global normalization divides in parallel once the
///   array has at least `NORMALIZE_PARALLEL_THRESHOLD` (10,000) elements.
///
/// # Implementation Details
///
/// - For Row axis: Each row is normalized independently
/// - For Column axis: Each column is normalized independently  
/// - For Global axis: The entire array is normalized as a single vector
/// - Zero norms are handled by setting the corresponding elements to zero
/// - NaN and infinite values in input will result in an error
pub fn normalize<S, D>(
    data: &ArrayBase<S, D>,
    axis: NormalizationAxis,
    order: NormalizationOrder,
) -> Result<Array<f64, D>, ModelError>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    // Input validation
    if data.is_empty() {
        return Err(ModelError::InputValidationError(
            "Cannot normalize empty array".to_string(),
        ));
    }

    // Check for NaN or infinite values
    if data.iter().any(|&x| !x.is_finite()) {
        return Err(ModelError::InputValidationError(
            "Input contains NaN or infinite values".to_string(),
        ));
    }

    // Validate Lp norm parameter
    if matches!(order, NormalizationOrder::Lp(p) if p <= 0.0 || !p.is_finite()) {
        return Err(ModelError::InputValidationError(
            "Lp norm parameter must be positive and finite".to_string(),
        ));
    }

    let mut result = data.to_owned();
    axis.apply(&mut result, order)?;
    Ok(result)
}

/// Divides a single lane by its norm in place, or zeros it when the norm is
/// effectively zero (the lane is all near-zero, so it has no direction to keep).
fn normalize_lane(lane: &mut ArrayViewMut1<f64>, norm: f64) {
    if norm > NORM_ZERO_THRESHOLD {
        lane.mapv_inplace(|x| x / norm);
    } else {
        lane.fill(0.0);
    }
}

/// Normalizes the entire array as a single flat vector.
fn normalize_global<D>(
    data: &mut Array<f64, D>,
    order: NormalizationOrder,
) -> Result<(), ModelError>
where
    D: Dimension,
{
    let norm = order.norm(data.iter().copied())?;

    if norm > NORM_ZERO_THRESHOLD {
        if data.len() >= NORMALIZE_PARALLEL_THRESHOLD {
            data.par_mapv_inplace(|x| x / norm);
        } else {
            data.mapv_inplace(|x| x / norm);
        }
    }

    Ok(())
}

/// Normalizes each lane along the axis `axis_from_end` positions from the end
/// (`1` = last axis → rows, `2` = second-to-last → columns).
///
/// Parallelizes across lanes once there are enough of them; each lane is then
/// processed sequentially, so the two levels never nest.
fn normalize_lanes<D>(
    data: &mut Array<f64, D>,
    axis_from_end: usize,
    order: NormalizationOrder,
    operation_name: &str,
) -> Result<(), ModelError>
where
    D: Dimension,
{
    let ndim = data.ndim();
    if ndim < 2 {
        return Err(ModelError::InputValidationError(format!(
            "{} requires at least 2 dimensions",
            operation_name
        )));
    }
    // `ndim >= 2` is guaranteed above, so `ndim - axis_from_end` cannot underflow for
    // `axis_from_end` in {1, 2}.
    let axis = Axis(ndim - axis_from_end);

    let mut lanes: Vec<ArrayViewMut1<f64>> = data.lanes_mut(axis).into_iter().collect();
    let process = |lane: &mut ArrayViewMut1<f64>| -> Result<(), ModelError> {
        let norm = order.norm(lane.iter().copied())?;
        normalize_lane(lane, norm);
        Ok(())
    };

    if lanes.len() >= NORMALIZE_PARALLEL_LANES {
        lanes.par_iter_mut().try_for_each(process)
    } else {
        lanes.iter_mut().try_for_each(process)
    }
}

impl NormalizationOrder {
    /// Computes this norm over a sequence of values.
    ///
    /// This is the single source of truth for each variant's norm formula, shared by
    /// the global and per-lane normalization paths.
    ///
    /// # Errors
    ///
    /// - [`ModelError::ProcessingError`] - If the accumulation overflows to a non-finite value
    fn norm<I>(&self, values: I) -> Result<f64, ModelError>
    where
        I: Iterator<Item = f64>,
    {
        match *self {
            NormalizationOrder::L1 => {
                let norm: f64 = values.map(|x| x.abs()).sum();
                if norm.is_finite() {
                    Ok(norm)
                } else {
                    Err(ModelError::ProcessingError(
                        "L1 norm computation resulted in non-finite value".to_string(),
                    ))
                }
            }
            NormalizationOrder::L2 => {
                let norm_squared: f64 = values.map(|x| x * x).sum();
                if norm_squared.is_finite() && norm_squared >= 0.0 {
                    Ok(norm_squared.sqrt())
                } else {
                    Err(ModelError::ProcessingError(
                        "L2 norm computation resulted in non-finite value".to_string(),
                    ))
                }
            }
            NormalizationOrder::Max => {
                let norm = values.map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
                // Handle the case where the iterator was empty or all values were zero
                if norm.is_finite() && norm >= 0.0 {
                    Ok(norm)
                } else if norm == f64::NEG_INFINITY {
                    // Empty iterator case
                    Ok(0.0)
                } else {
                    Err(ModelError::ProcessingError(
                        "Max norm computation resulted in non-finite value".to_string(),
                    ))
                }
            }
            NormalizationOrder::Lp(p) => {
                let sum: f64 = values.map(|x| x.abs().powf(p)).sum();
                if sum.is_finite() && sum >= 0.0 {
                    Ok(sum.powf(1.0 / p))
                } else {
                    Err(ModelError::ProcessingError(format!(
                        "Lp norm (p={}) computation resulted in non-finite value",
                        p
                    )))
                }
            }
        }
    }
}

impl NormalizationAxis {
    /// Normalizes `data` in place along this axis under the given norm order.
    ///
    /// Routes to whole-array normalization for [`NormalizationAxis::Global`] or to the
    /// per-lane path for [`NormalizationAxis::Row`] / [`NormalizationAxis::Column`].
    ///
    /// # Errors
    ///
    /// - [`ModelError::InputValidationError`] - If row/column normalization is requested
    ///   on an array with fewer than 2 dimensions
    /// - [`ModelError::ProcessingError`] - If a norm computation overflows
    fn apply<D>(
        &self,
        data: &mut Array<f64, D>,
        order: NormalizationOrder,
    ) -> Result<(), ModelError>
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

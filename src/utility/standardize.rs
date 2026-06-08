use crate::error::Error;
use ndarray::{Array, ArrayBase, ArrayViewMut1, Axis, Data, Dimension};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::IntoParallelRefMutIterator;

/// Element-count threshold above which the global-axis path computes in parallel.
const STANDARDIZE_PARALLEL_THRESHOLD: usize = 10000;

/// Lane-count threshold above which row/column standardization runs across lanes in parallel.
const STANDARDIZE_PARALLEL_LANES: usize = 100;

/// Defines the axis along which the standardization is applied
///
/// # Variants
///
/// - `Row` - Standardize across rows (each row is standardized independently)
/// - `Column` - Standardize across columns (each column is standardized independently)
/// - `Global` - Standardize the entire array globally
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardizationAxis {
    Row,
    Column,
    Global,
}

impl StandardizationAxis {
    /// Standardizes `data` in place along this axis.
    ///
    /// Routes to whole-array standardization for [`StandardizationAxis::Global`] or to
    /// the per-lane path for [`StandardizationAxis::Row`] / [`StandardizationAxis::Column`].
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] - If row/column standardization is requested
    ///   on an array with fewer than 2 dimensions
    fn apply<D>(&self, data: &mut Array<f64, D>, epsilon: f64) -> Result<(), Error>
    where
        D: Dimension,
    {
        match self {
            StandardizationAxis::Global => standardize_global(data, epsilon),
            StandardizationAxis::Row => standardize_lanes(data, 1, epsilon, "Row standardization"),
            StandardizationAxis::Column => {
                standardize_lanes(data, 2, epsilon, "Column standardization")
            }
        }
    }
}

/// Standardizes data to have zero mean and unit variance
///
/// This function transforms input data by subtracting the mean and dividing
/// by the standard deviation for each feature, resulting in standardized data
/// where each feature has a mean of 0 and a standard deviation of 1.
///
/// # Parameters
///
/// - `data` - Input array data as `ArrayBase` with arbitrary dimensions and f64 elements
/// - `axis` - The axis along which to perform standardization (Row/Column/Global)
/// - `epsilon` - Small value that floors the standard deviation, added in quadrature as
///   `sqrt(variance + epsilon²)`, to prevent division by zero
///
/// # Returns
///
/// - `Result<Array<f64, D>, Error>` - Standardized array with same dimensions as input
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::standardize::{standardize, StandardizationAxis};
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let result = standardize(&data, StandardizationAxis::Column, 1e-8).unwrap();
/// ```
///
/// # Errors
///
/// - [`Error::EmptyInput`] - If input array is empty
/// - [`Error::NonFinite`] - If input contains NaN/Infinite values
/// - [`Error::InvalidParameter`] - If epsilon is non-positive or non-finite
/// - [`Error::Computation`] - If standardization computation fails (e.g., zero values in global axis)
///
/// # Performance
///
/// - Parallel computation is enabled when the number of elements exceeds `STANDARDIZE_PARALLEL_THRESHOLD` (10,000)
///
/// # Implementation Details
///
/// - For Row axis: Each row is standardized independently
/// - For Column axis: Each column is standardized independently
/// - For Global axis: The entire array is standardized as a single dataset
/// - The standard deviation is floored via `sqrt(variance + epsilon²)` to prevent division by zero
/// - Uses parallel computation for improved performance on large datasets
/// - NaN and infinite values in input will result in an error
pub fn standardize<S, D>(
    data: &ArrayBase<S, D>,
    axis: StandardizationAxis,
    epsilon: f64,
) -> Result<Array<f64, D>, Error>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    // Input validation
    if data.is_empty() {
        return Err(Error::empty_input("Cannot standardize empty array"));
    }

    // Check for NaN or infinite values
    if data.iter().any(|&x| !x.is_finite()) {
        return Err(Error::non_finite("input data"));
    }

    // Validate epsilon parameter
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(Error::invalid_parameter(
            "epsilon",
            "Epsilon must be positive and finite",
        ));
    }

    let mut result = data.to_owned();
    axis.apply(&mut result, epsilon)?;
    Ok(result)
}

/// Helper function to standardize the entire array globally
fn standardize_global<D>(data: &mut Array<f64, D>, epsilon: f64) -> Result<(), Error>
where
    D: Dimension,
{
    let n = data.len() as f64;

    if n == 0.0 {
        return Err(Error::computation("No values to standardize"));
    }

    // Use parallel computation for large datasets
    if n as usize >= STANDARDIZE_PARALLEL_THRESHOLD {
        // Calculate mean
        let mean = data.par_iter().sum::<f64>() / n;

        // Calculate variance
        let variance = data.par_iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        // Add epsilon to variance for numerical stability, then take sqrt
        let std_dev = (variance + epsilon * epsilon).sqrt();

        // Apply standardization
        data.par_mapv_inplace(|x| (x - mean) / std_dev);
    } else {
        // Same process as above, but sequential
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = (variance + epsilon * epsilon).sqrt();
        data.mapv_inplace(|x| (x - mean) / std_dev);
    }

    Ok(())
}

/// Computes the population mean and the epsilon-floored standard deviation of a lane.
///
/// The returned standard deviation is `sqrt(variance + epsilon²)`, which stays
/// strictly positive so the subsequent division is always well defined.
fn lane_mean_and_std(lane: &ArrayViewMut1<f64>, epsilon: f64) -> (f64, f64) {
    let n = lane.len() as f64;
    let mean = lane.sum() / n;
    let variance = lane.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = (variance + epsilon * epsilon).sqrt();
    (mean, std_dev)
}

/// Standardizes each lane along the axis `axis_from_end` positions from the end
/// (`1` = last axis → rows, `2` = second-to-last → columns).
///
/// Parallelizes across lanes once there are enough of them; each lane is then
/// processed sequentially, so the two levels never nest.
fn standardize_lanes<D>(
    data: &mut Array<f64, D>,
    axis_from_end: usize,
    epsilon: f64,
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
    // `ndim >= 2` is guaranteed above, so `ndim - axis_from_end` cannot underflow for
    // `axis_from_end` in {1, 2}.
    let axis = Axis(ndim - axis_from_end);

    let mut lanes: Vec<ArrayViewMut1<f64>> = data.lanes_mut(axis).into_iter().collect();
    let process = |lane: &mut ArrayViewMut1<f64>| {
        let (mean, std_dev) = lane_mean_and_std(lane, epsilon);
        lane.mapv_inplace(|x| (x - mean) / std_dev);
    };

    if lanes.len() >= STANDARDIZE_PARALLEL_LANES {
        lanes.par_iter_mut().for_each(process);
    } else {
        lanes.iter_mut().for_each(process);
    }
    Ok(())
}

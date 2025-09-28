use super::*;

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

/// Standardizes data to have zero mean and unit variance
///
/// This function transforms input data by subtracting the mean and dividing
/// by the standard deviation for each feature, resulting in standardized data
/// where each feature has a mean of 0 and a standard deviation of 1.
///
/// # Parameters
///
/// - `data` - Input array data as `ArrayView` with arbitrary dimensions and f64 elements
/// - `axis` - The axis along which to perform standardization (Row/Column/Global)
/// - `epsilon` - Small value added to standard deviation to prevent division by zero
///
/// # Returns
///
/// * `Result<Array<f64, D>, ModelError>` - Standardized array with same dimensions as input
///   - `Ok(Array<f64, D>)` - Successfully standardized array
///   - `Err(ModelError::InputValidationError)` - If input validation fails
///   - `Err(ModelError::ProcessingError)` - If standardization computation fails
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::{standardize, StandardizationAxis};
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let result = standardize(data.view(), StandardizationAxis::Column, 1e-8).unwrap();
/// // Each column will have mean ≈ 0 and std ≈ 1
/// ```
///
/// # Implementation Details
///
/// - For Row axis: Each row is standardized independently
/// - For Column axis: Each column is standardized independently
/// - For Global axis: The entire array is standardized as a single dataset
/// - Epsilon is added to standard deviation to prevent division by zero
/// - Uses parallel computation for improved performance on large datasets
/// - NaN and infinite values in input will result in an error
pub fn standardize<S, D>(
    data: ArrayBase<S, D>,
    axis: StandardizationAxis,
    epsilon: f64,
) -> Result<Array<f64, D>, ModelError>
where
    S: ndarray::Data<Elem = f64>,
    D: Dimension,
{
    // Input validation
    if data.is_empty() {
        return Err(ModelError::InputValidationError(
            "Cannot standardize empty array".to_string(),
        ));
    }

    // Check for NaN or infinite values
    if data.iter().any(|&x| !x.is_finite()) {
        return Err(ModelError::InputValidationError(
            "Input contains NaN or infinite values".to_string(),
        ));
    }

    // Validate epsilon parameter
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(ModelError::InputValidationError(
            "Epsilon must be positive and finite".to_string(),
        ));
    }

    let mut result = data.to_owned();

    match axis {
        StandardizationAxis::Global => {
            standardize_global(&mut result, epsilon)?;
        }
        StandardizationAxis::Row => {
            standardize_by_rows(&mut result, epsilon)?;
        }
        StandardizationAxis::Column => {
            standardize_by_columns(&mut result, epsilon)?;
        }
    }

    Ok(result)
}

/// Helper function to standardize the entire array globally
fn standardize_global<D>(data: &mut Array<f64, D>, epsilon: f64) -> Result<(), ModelError>
where
    D: Dimension,
{
    let values: Vec<f64> = data.iter().copied().collect();
    let n = values.len() as f64;

    if n == 0.0 {
        return Err(ModelError::ProcessingError(
            "No values to standardize".to_string(),
        ));
    }

    // Calculate mean
    let mean = values.iter().sum::<f64>() / n;

    // Calculate standard deviation
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt() + epsilon;

    // Apply standardization
    data.mapv_inplace(|x| (x - mean) / std_dev);

    Ok(())
}

/// Helper function to compute mean and standard deviation for a collection of values
fn compute_mean_and_std(values: &[f64], epsilon: f64) -> (f64, f64) {
    let n = values.len() as f64;

    if n == 0.0 {
        return (0.0, epsilon);
    }

    // Calculate mean
    let mean = values.iter().sum::<f64>() / n;

    // Calculate standard deviation
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt() + epsilon;

    (mean, std_dev)
}

/// Generic helper function to standardize lanes along a specified axis
fn standardize_lanes<D>(
    data: &mut Array<f64, D>,
    axis: Axis,
    epsilon: f64,
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

    // Process each lane along the specified axis using parallel iteration
    let mut lanes: Vec<_> = data.lanes_mut(axis).into_iter().collect();

    lanes
        .par_iter_mut()
        .try_for_each(|lane| -> Result<(), ModelError> {
            let values: Vec<f64> = lane.iter().copied().collect();
            let (mean, std_dev) = compute_mean_and_std(&values, epsilon);

            // Apply standardization
            lane.mapv_inplace(|x| (x - mean) / std_dev);

            Ok(())
        })?;

    Ok(())
}

/// Helper function to standardize each row independently
fn standardize_by_rows<D>(data: &mut Array<f64, D>, epsilon: f64) -> Result<(), ModelError>
where
    D: Dimension,
{
    let ndim = data.ndim();
    // Get the last axis (assuming it represents features/columns)
    let last_axis = Axis(ndim - 1);

    standardize_lanes(data, last_axis, epsilon, "Row standardization")
}

/// Helper function to standardize each column independently
fn standardize_by_columns<D>(data: &mut Array<f64, D>, epsilon: f64) -> Result<(), ModelError>
where
    D: Dimension,
{
    let ndim = data.ndim();

    // Column standardization requires at least 2 dimensions
    if ndim < 2 {
        return Err(ModelError::InputValidationError(
            "Column standardization requires at least 2 dimensions".to_string(),
        ));
    }

    // Get the second-to-last axis (assuming it represents samples/rows)
    let axis = Axis(ndim - 2);

    standardize_lanes(data, axis, epsilon, "Column standardization")
}

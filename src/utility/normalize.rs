use super::*;

/// Defines the axis along which the normalization is applied
///
/// # Variants
///
/// - `Row` - Normalize across rows (each row is normalized independently)
/// - `Column` - Normalize across columns (each column is normalized independently)
/// - `Global` - Normalize the entire array globally
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
/// - `data` - Input array data as `ArrayBase` with arbitrary dimensions and f64 elements
/// - `axis` - The axis along which to perform normalization (Row/Column/Global)
/// - `order` - The norm order to use for normalization (L1/L2/Max/Lp)
///
/// # Returns
///
/// * `Result<Array<f64, D>, ModelError>` - Normalized array with same dimensions as input
///   - `Ok(Array<f64, D>)` - Successfully normalized array
///   - `Err(ModelError::InputValidationError)` - If input validation fails
///   - `Err(ModelError::ProcessingError)` - If normalization computation fails
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::{normalize, NormalizationAxis, NormalizationOrder};
///
/// let data = array![[3.0, 4.0], [1.0, 2.0]];
/// let result = normalize(data.view(), NormalizationAxis::Row, NormalizationOrder::L2).unwrap();
/// // Each row will have L2 norm = 1
/// ```
///
/// # Implementation Details
///
/// - For Row axis: Each row is normalized independently
/// - For Column axis: Each column is normalized independently  
/// - For Global axis: The entire array is normalized as a single vector
/// - Zero norms are handled by setting the corresponding elements to zero
/// - NaN and infinite values in input will result in an error
pub fn normalize<S, D>(
    data: ArrayBase<S, D>,
    axis: NormalizationAxis,
    order: NormalizationOrder,
) -> Result<Array<f64, D>, ModelError>
where
    S: ndarray::Data<Elem = f64>,
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
    if let NormalizationOrder::Lp(p) = order {
        if p <= 0.0 || !p.is_finite() {
            return Err(ModelError::InputValidationError(
                "Lp norm parameter must be positive and finite".to_string(),
            ));
        }
    }

    let mut result = data.to_owned();

    match axis {
        NormalizationAxis::Global => {
            normalize_global(&mut result, order)?;
        }
        NormalizationAxis::Row => {
            normalize_by_rows(&mut result, order)?;
        }
        NormalizationAxis::Column => {
            normalize_by_columns(&mut result, order)?;
        }
    }

    Ok(result)
}

/// Helper function to normalize the entire array globally
fn normalize_global<D>(
    data: &mut Array<f64, D>,
    order: NormalizationOrder,
) -> Result<(), ModelError>
where
    D: Dimension,
{
    let norm = compute_norm(data.view().into_iter().copied(), order)?;

    if norm > 1e-15 {
        data.mapv_inplace(|x| x / norm);
    } else {
        // If norm is effectively zero, set all elements to zero
        data.fill(0.0);
    }

    Ok(())
}

/// Helper function to normalize each row independently
fn normalize_by_rows<D>(
    data: &mut Array<f64, D>,
    order: NormalizationOrder,
) -> Result<(), ModelError>
where
    D: Dimension,
{
    let ndim = data.ndim();
    if ndim < 2 {
        return Err(ModelError::InputValidationError(
            "Row normalization requires at least 2 dimensions".to_string(),
        ));
    }

    // Get the last axis (assuming it represents features/columns)
    let last_axis = Axis(ndim - 1);

    // Process each lane along the last axis
    for mut row in data.lanes_mut(last_axis) {
        let norm = compute_norm(row.iter().copied(), order)?;

        if norm > 1e-15 {
            row.mapv_inplace(|x| x / norm);
        } else {
            // If norm is effectively zero, set all elements in row to zero
            row.fill(0.0);
        }
    }

    Ok(())
}

/// Helper function to normalize each column independently
fn normalize_by_columns<D>(
    data: &mut Array<f64, D>,
    order: NormalizationOrder,
) -> Result<(), ModelError>
where
    D: Dimension,
{
    let ndim = data.ndim();
    if ndim < 2 {
        return Err(ModelError::InputValidationError(
            "Column normalization requires at least 2 dimensions".to_string(),
        ));
    }

    // Get the second-to-last axis (assuming it represents samples/rows)
    let axis = Axis(ndim - 2);

    // Process each lane along the specified axis
    for mut col in data.lanes_mut(axis) {
        let norm = compute_norm(col.iter().copied(), order)?;

        if norm > 1e-15 {
            col.mapv_inplace(|x| x / norm);
        } else {
            // If norm is effectively zero, set all elements in column to zero
            col.fill(0.0);
        }
    }

    Ok(())
}

/// Compute the norm of a sequence of values according to the specified order
fn compute_norm<I>(values: I, order: NormalizationOrder) -> Result<f64, ModelError>
where
    I: Iterator<Item = f64>,
{
    match order {
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
            let norm = values
                .map(|x| x.abs())
                .fold(0.0, |max, x| if x > max { x } else { max });
            if norm.is_finite() {
                Ok(norm)
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

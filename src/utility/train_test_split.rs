use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::ModelError;

/// Splits a dataset into training and test sets
///
/// # Parameters
///
/// - `x` - Feature matrix with shape (n_samples, n_features)
/// - `y` - Target values with shape (n_samples,)
/// - `test_size` - Size of the test set, default is 0.3 (30%)
/// - `random_state` - Random seed, default is None
///
/// # Returns
///
/// Returns a tuple `(x_train, x_test, y_train, y_test)`
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::train_test_split::train_test_split;
///
/// let x = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
/// let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
/// let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.4), Some(42)).unwrap();
/// ```
pub fn train_test_split(
    x: Array2<f64>,
    y: Array1<f64>,
    test_size: Option<f64>,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), ModelError> {
    /// Extract a subset from Array2 based on indices using ndarray's select method
    fn create_subset_array2(array: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        array.select(Axis(0), indices)
    }

    /// Extract a subset from Array1 based on indices using ndarray's select method
    fn create_subset_array1(array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        array.select(Axis(0), indices)
    }

    let n_samples = x.nrows();

    // Ensure x and y have the same number of samples
    if n_samples != y.len() {
        return Err(ModelError::InputValidationError(
            format!("x and y must have the same number of samples, x rows: {}, y length: {}",
                    n_samples, y.len())
        ));
    }

    // Set test size, default is 0.3
    let test_size = test_size.unwrap_or(0.3);
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(ModelError::InputValidationError(
            format!("test_size must be between 0 and 1, got {}", test_size)
        ));
    }

    // Calculate the number of test samples using ceil to ensure at least the expected proportion
    let n_test = std::cmp::max(1, (n_samples as f64 * test_size).ceil() as usize);
    if n_test > n_samples {
        return Err(ModelError::InputValidationError(
            format!("test_size={} is too big for n_samples={}", test_size, n_samples)
        ));
    }

    // Create random indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle indices based on random state
    if let Some(seed) = random_state {
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    } else {
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
    }

    // Split indices into train and test sets
    let test_indices = &indices[0..n_test];
    let train_indices = &indices[n_test..];

    // Create train and test datasets using ndarray's select method for more concise code
    let x_train = create_subset_array2(&x, train_indices);
    let x_test = create_subset_array2(&x, test_indices);
    let y_train = create_subset_array1(&y, train_indices);
    let y_test = create_subset_array1(&y, test_indices);

    Ok((x_train, x_test, y_train, y_test))
}
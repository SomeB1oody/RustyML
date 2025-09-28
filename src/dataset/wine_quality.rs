use super::raw_data::{load_red_wine_quality_raw_data, load_white_wine_quality_raw_data};
use ndarray::prelude::*;
use std::sync::OnceLock;

// Use `OnceLock` for thread-safe delayed initialization of red wine dataset
static RED_WINE_DATA: OnceLock<(Array1<&'static str>, Array2<f64>)> = OnceLock::new();

// Use `OnceLock` for thread-safe delayed initialization of white wine dataset
static WHITE_WINE_DATA: OnceLock<(Array1<&'static str>, Array2<f64>)> = OnceLock::new();

/// Parses wine quality dataset from raw string data into structured arrays.
///
/// This internal function extracts feature headers and wine quality data from raw string format,
/// converting them into ndarray structures suitable for machine learning operations.
///
/// # Parameters
///
/// - `raw_data` - Raw string containing both headers and wine data
/// - `n_samples` - Number of data samples expected
///
/// # Returns
///
/// * A tuple containing:
///     - `Array1<&'static str>` - Array of feature names (headers)
///     - `Array2<f64>` - 2D array of wine quality features with shape (n_samples, 12)
fn parse_wine_data(
    raw_data: &'static str,
    n_samples: usize,
) -> (Array1<&'static str>, Array2<f64>) {
    let lines: Vec<&str> = raw_data.trim().lines().collect();

    // First line contains headers (comma-separated)
    let headers_line = lines[0];
    let headers_array: Vec<&str> = headers_line.split(',').collect();

    // Remaining lines contain data (semicolon-separated)
    let mut features_array = Vec::with_capacity(n_samples * 12);

    for line in &lines[1..] {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(';').collect();

        for i in 0..12 {
            features_array.push(cols[i].parse::<f64>().unwrap());
        }
    }

    let features_array =
        Array::from_shape_vec((features_array.len() / 12, 12), features_array).unwrap();
    let headers_array = Array::from_vec(headers_array);

    (headers_array, features_array)
}

/// Internal function to load and process the raw red wine quality dataset.
///
/// This function loads the raw red wine quality dataset, parses the comma-separated headers
/// and semicolon-separated data format, and converts it into structured ndarray arrays.
///
/// # Returns
///
/// * A tuple containing:
///     - `Array1<&'static str>`: Array of column headers from the dataset
///     - `Array2<f64>`: Feature matrix where each row represents a wine sample
///       and each column represents a feature
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format
/// - Memory allocation fails during array creation
fn load_red_wine_quality_internal() -> (Array1<&'static str>, Array2<f64>) {
    let red_wine_raw_data = load_red_wine_quality_raw_data();
    parse_wine_data(red_wine_raw_data, 1599)
}

/// Internal function to load and process the raw white wine quality dataset.
///
/// This function loads the raw white wine quality dataset, parses the comma-separated headers
/// and semicolon-separated data format, and converts it into structured ndarray arrays.
///
/// # Returns
///
/// * A tuple containing:
///     - `Array1<&'static str>`: Array of column headers from the dataset
///     - `Array2<f64>`: Feature matrix where each row represents a wine sample
///       and each column represents a feature
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format
/// - Memory allocation fails during array creation
fn load_white_wine_quality_internal() -> (Array1<&'static str>, Array2<f64>) {
    let white_wine_raw_data = load_white_wine_quality_raw_data();
    parse_wine_data(white_wine_raw_data, 4898)
}

/// Loads the red wine quality dataset with memoization for machine learning tasks.
///
/// This function provides access to a curated red wine quality dataset containing
/// physicochemical properties and quality ratings. The dataset includes 11 features
/// such as acidity levels, sugar content, pH, and alcohol percentage, along with
/// quality scores ranging from 3 to 8. Uses memoization for improved performance
/// on repeated calls.
///
/// # Returns
///
/// A tuple containing:
/// * `&'static Array1<&'static str>` - Static reference to array of feature names including:
///   - fixed acidity, volatile acidity, citric acid
///   - residual sugar, chlorides
///   - free sulfur dioxide, total sulfur dioxide
///   - density, pH, sulphates, alcohol, quality
/// * `&'static Array2<f64>` - Static reference to 2D feature matrix with shape (1599, 12)
///   containing normalized wine quality measurements
///
/// # Example
/// ```rust
/// use rustyml::dataset::wine_quality::load_red_wine_quality;
///
/// let (headers, features) = load_red_wine_quality();
///
/// // Access feature names
/// println!("Features: {:?}", headers);
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.ncols(), 12);  // 12 features
/// assert_eq!(features.nrows(), 1599); // 1599 samples
///
/// // Example: Extract quality scores (last column)
/// let quality_scores = features.column(11);  // Quality is the 12th column (index 11)
/// ```
pub fn load_red_wine_quality() -> (&'static Array1<&'static str>, &'static Array2<f64>) {
    let (headers, features) = RED_WINE_DATA.get_or_init(load_red_wine_quality_internal);
    (headers, features)
}

/// Loads the white wine quality dataset with memoization for machine learning tasks.
///
/// This function provides access to a curated white wine quality dataset with
/// the same structure as the red wine dataset. It contains physicochemical
/// properties and quality ratings specifically for white wine samples.
/// The dataset uses the same 12 features but with different value ranges
/// typical for white wine characteristics. Uses memoization for improved
/// performance on repeated calls.
///
/// # Returns
///
/// A tuple containing:
/// * `&'static Array1<&'static str>` - Static reference to array of feature names including:
///   - fixed acidity, volatile acidity, citric acid
///   - residual sugar, chlorides
///   - free sulfur dioxide, total sulfur dioxide
///   - density, pH, sulphates, alcohol, quality
/// * `&'static Array2<f64>` - Static reference to 2D feature matrix with shape (4898, 12)
///   containing normalized white wine quality measurements
///
/// # Example
/// ```rust
/// use rustyml::dataset::wine_quality::load_white_wine_quality;
/// use ndarray::prelude::*;
///
/// let (headers, features) = load_white_wine_quality();
///
/// // Access feature names
/// println!("Features: {:?}", headers);
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.ncols(), 12);  // 12 features
/// assert_eq!(features.nrows(), 4898); // 4898 samples
///
/// // Example: Extract quality scores (last column)
/// let quality_scores = features.column(11);  // Quality is the 12th column (index 11)
/// ```
pub fn load_white_wine_quality() -> (&'static Array1<&'static str>, &'static Array2<f64>) {
    let (headers, features) = WHITE_WINE_DATA.get_or_init(load_white_wine_quality_internal);
    (headers, features)
}

/// Loads the red wine quality dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_red_wine_quality()` which returns references.
///
/// # Returns
///
/// * A tuple containing owned copies of:
///     - `Array1<&'static str>`: Owned array of column headers from the dataset, containing 12 feature names
///     - `Array2<f64>`: Owned feature matrix with shape (1599, 12) where each row represents a wine sample and each column represents a feature (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality)
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_red_wine_quality()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::wine_quality::load_red_wine_quality_owned;
///
/// let (mut headers, mut features) = load_red_wine_quality_owned();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(headers.len(), 12);
/// assert_eq!(features.shape(), &[1599, 12]);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
/// ```
pub fn load_red_wine_quality_owned() -> (Array1<&'static str>, Array2<f64>) {
    let (headers, features) = load_red_wine_quality();
    (headers.clone(), features.clone())
}

/// Loads the white wine quality dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_white_wine_quality()` which returns references.
///
/// # Returns
///
/// * A tuple containing owned copies of:
///     - `Array1<&'static str>`: Owned array of column headers from the dataset, containing 12 feature names
///     - `Array2<f64>`: Owned feature matrix with shape (4898, 12) where each row represents a wine sample and each column represents a feature (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality)
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_white_wine_quality()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::wine_quality::load_white_wine_quality_owned;
/// use ndarray::prelude::*;
///
/// let (mut headers, mut features) = load_white_wine_quality_owned();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(headers.len(), 12);
/// assert_eq!(features.shape(), &[4898, 12]);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
/// ```
pub fn load_white_wine_quality_owned() -> (Array1<&'static str>, Array2<f64>) {
    let (headers, features) = load_white_wine_quality();
    (headers.clone(), features.clone())
}

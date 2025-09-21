use super::raw_data::{load_red_wine_quality_raw_data, load_white_wine_quality_raw_data};
use ndarray::prelude::*;

/// Parses wine quality dataset from raw string data into structured arrays.
///
/// This internal function extracts feature headers and wine quality data from raw string format,
/// converting them into ndarray structures suitable for machine learning operations.
///
/// # Parameters
///
/// - `headers_raw` - Raw string containing feature names, one per line
/// - `data_raw` - Raw string containing wine data with semicolon-separated values
///
/// # Returns
///
/// * A tuple containing:
///     - `Array1<&'static str>` - Array of feature names (headers)
///     - `Array2<f64>` - 2D array of wine quality features with shape (n_samples, 12)
fn parse_wine_data(
    headers_raw: &'static str,
    data_raw: &str,
    n_samples: usize,
) -> (Array1<&'static str>, Array2<f64>) {
    let mut features_array = Vec::with_capacity(n_samples * 12);
    let headers_array = headers_raw.trim().lines().collect::<Vec<&str>>();

    for line in data_raw.trim().lines() {
        let cols: Vec<&str> = line.split(';').collect();

        for i in 0..12 {
            features_array.push(cols[i].parse::<f64>().unwrap());
        }
    }

    let features_array = Array::from_shape_vec((n_samples, 12), features_array).unwrap();
    let headers_array = Array::from_vec(headers_array);

    (headers_array, features_array)
}

/// Loads the red wine quality dataset for machine learning tasks.
///
/// This function provides access to a curated red wine quality dataset containing
/// physicochemical properties and quality ratings. The dataset includes 11 features
/// such as acidity levels, sugar content, pH, and alcohol percentage, along with
/// quality scores ranging from 3 to 8.
///
/// # Returns
///
/// A tuple containing:
/// * `Array1<&'static str>` - Array of feature names including:
///   - fixed acidity, volatile acidity, citric acid
///   - residual sugar, chlorides
///   - free sulfur dioxide, total sulfur dioxide
///   - density, pH, sulphates, alcohol, quality
/// * `Array2<f64>` - 2D feature matrix with shape (n_samples, 12) containing
///   normalized wine quality measurements
///
/// # Example
///
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
/// assert!(features.nrows() > 0);     // Has sample data
///
/// // Example: Extract quality scores (last column)
/// let quality_scores = features.column(11);  // Quality is the 12th column (index 11)
/// ```
pub fn load_red_wine_quality() -> (Array1<&'static str>, Array2<f64>) {
    let (red_wine_data_headers_raw, red_wine_data_raw) = load_red_wine_quality_raw_data();

    parse_wine_data(red_wine_data_headers_raw, red_wine_data_raw, 1599)
}

/// Loads the white wine quality dataset for machine learning tasks.
///
/// This function provides access to a curated white wine quality dataset with
/// the same structure as the red wine dataset. It contains physicochemical
/// properties and quality ratings specifically for white wine samples.
/// The dataset uses the same 12 features but with different value ranges
/// typical for white wine characteristics.
///
/// # Returns
///
/// A tuple containing:
/// * `Array1<&'static str>` - Array of feature names including:
///   - fixed acidity, volatile acidity, citric acid
///   - residual sugar, chlorides
///   - free sulfur dioxide, total sulfur dioxide
///   - density, pH, sulphates, alcohol, quality
/// * `Array2<f64>` - 2D feature matrix with shape (n_samples, 12) containing
///   normalized white wine quality measurements
///
/// # Example
///
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
/// assert!(features.nrows() > 0);     // Has sample data
///
/// // Example: Extract quality scores (last column)
/// let quality_scores = features.column(11);  // Quality is the 12th column (index 11)
/// ```
pub fn load_white_wine_quality() -> (Array1<&'static str>, Array2<f64>) {
    let (white_wine_data_headers_raw, white_wine_data_raw) = load_white_wine_quality_raw_data();

    parse_wine_data(white_wine_data_headers_raw, white_wine_data_raw, 4898)
}

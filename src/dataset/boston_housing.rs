use super::raw_data::boston_housing_raw::*;
use ndarray::prelude::*;
use std::sync::OnceLock;

// Use `OnceLock` for thread-safe delayed initialization
static BOSTON_HOUSING_DATA: OnceLock<(Array1<&'static str>, Array2<f64>, Array1<f64>)> =
    OnceLock::new();

/// Internal function to load and process the raw boston housing dataset.
///
/// This function loads the raw boston housing dataset, parses the CSV-like format,
/// and converts it into structured ndarray arrays. It handles the parsing
/// of headers and data rows, extracting features and target values from the dataset.
///
/// # Returns
///
/// * A tuple containing:
///     - `Array1<&'static str>`: Array of column headers from the dataset
///     - `Array2<f64>`: Feature matrix with shape (2, 13) where each row represents
///     a housing sample and each column represents a feature
///     - `Array1<f64>`: Target values array with shape (2,) containing median home
///     values in $1000s (MEDV)
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (2 samples, 14 columns total)
/// - Memory allocation fails during array creation
fn load_boston_housing_internal() -> (Array1<&'static str>, Array2<f64>, Array1<f64>) {
    let (boston_housing_data_headers_raw, boston_housing_data_raw) = load_boston_housing_raw_data();

    let headers = boston_housing_data_headers_raw
        .trim()
        .lines()
        .collect::<Vec<&str>>();
    let mut features = Vec::new();
    let mut labels = Vec::new();

    let lines: Vec<&str> = boston_housing_data_raw.trim().lines().collect();
    let num_samples = lines.len();

    for line in lines {
        let cols: Vec<&str> = line.split(',').collect();

        // Features are columns 0-12 (13 features)
        for i in 0..13 {
            features.push(cols[i].parse::<f64>().unwrap());
        }

        // Target is column 13 (MEDV)
        labels.push(cols[13].parse::<f64>().unwrap());
    }

    let headers_array = Array1::from_vec(headers);
    let features_array = Array2::from_shape_vec((num_samples, 13), features).unwrap();
    let labels_array = Array1::from_vec(labels);

    (headers_array, features_array, labels_array)
}

/// Loads the Boston Housing dataset with memoization
///
/// The Boston Housing dataset contains information about housing values in
/// suburbs of Boston. The dataset includes 13 features for predicting
/// the median value of owner-occupied homes (MEDV).
///
/// This function uses memoization to ensure the dataset is loaded only once and returns
/// static references to the cached data for optimal performance.
///
/// # Features
///
/// - CRIM: per capita crime rate by town
/// - ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
/// - INDUS: proportion of non-retail business acres per town
/// - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
/// - NOX: nitric oxides concentration (parts per 10 million)
/// - RM: average number of rooms per dwelling
/// - AGE: proportion of owner-occupied units built prior to 1940
/// - DIS: weighted distances to five Boston employment centres
/// - RAD: index of accessibility to radial highways
/// - TAX: full-value property-tax rate per $10,000
/// - PTRATIO: pupil-teacher ratio by town
/// - B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
/// - LSTAT: % lower status of the population
///
/// # Target
///
/// - MEDV: Median value of owner-occupied homes in $1000's
///
/// # Returns
///
/// * A tuple containing:
///     - `&'static Array1<&'static str>`: Static reference to the headers of the dataset
///     - `&'static Array2<f64>`: Static reference to the feature matrix where each row is a sample and each column is a feature
///     - `&'static Array1<f64>`: Static reference to median home values (MEDV) in $1000s
///
/// # Examples
/// ```rust
/// use rustyml::dataset::boston_housing::load_boston_housing;
///
/// let (headers, features, medv) = load_boston_housing();
/// assert_eq!(headers.len(), 14);
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(medv.len(), 506);
/// ```
pub fn load_boston_housing() -> (
    &'static Array1<&'static str>,
    &'static Array2<f64>,
    &'static Array1<f64>,
) {
    let (headers, features, labels) = BOSTON_HOUSING_DATA.get_or_init(load_boston_housing_internal);
    (headers, features, labels)
}

/// Loads the Boston Housing dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_boston_housing()` which returns references.
///
/// The Boston Housing dataset contains information about housing values in
/// suburbs of Boston. The dataset includes 13 features for predicting
/// the median value of owner-occupied homes (MEDV).
///
/// # Returns
///
/// * A tuple containing owned copies of:
///     - `Array1<&'static str>`: Owned array of column headers from the dataset, containing 14 names (13 features plus MEDV target)
///     - `Array2<f64>`: Owned feature matrix where each row represents a housing sample and each column represents a feature
///     - `Array1<f64>`: Owned target values array containing median home values (MEDV) in $1000s
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_boston_housing()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::boston_housing::load_boston_housing_owned;
///
/// let (mut headers, mut features, mut medv) = load_boston_housing_owned();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(headers.len(), 14);
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(medv.len(), 506);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 0.1;
/// medv[0] = 25.5;
/// ```
pub fn load_boston_housing_owned() -> (Array1<&'static str>, Array2<f64>, Array1<f64>) {
    let (headers, features, labels) = load_boston_housing();
    (headers.clone(), features.clone(), labels.clone())
}

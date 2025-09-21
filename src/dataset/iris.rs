use super::raw_data::iris_raw::*;
use ndarray::prelude::*;

/// Loads the Iris dataset
///
/// The Iris dataset contains measurements of 150 iris flowers from three different species:
/// - Iris-setosa
/// - Iris-versicolor
/// - Iris-virginica
///
/// # Returns
///
/// A tuple containing:
/// - `Array1<&'static str>`: The headers of the dataset
/// - `Array2<f64>`: A 2D array of shape (150, 4) containing the feature measurements:
///   - sepal length in cm
///   - sepal width in cm
///   - petal length in cm
///   - petal width in cm
/// - `Array1<&'static str>`: A 1D array of length 150 containing the species labels
///
/// # Example
///
/// ```
/// use rustyml::dataset::iris::load_iris;
///
/// let (headers, features, labels) = load_iris();
/// assert_eq!(headers.len(), 5);
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
/// ```
pub fn load_iris() -> (Array1<&'static str>, Array2<f64>, Array1<&'static str>) {
    let (iris_data_headers_raw, iris_data_raw) = load_iris_raw_data();

    let headers = iris_data_headers_raw.trim().lines().collect::<Vec<&str>>();
    let mut features = Vec::with_capacity(150 * 4);
    let mut labels = Vec::with_capacity(150);

    for line in iris_data_raw.trim().lines() {
        let cols: Vec<&str> = line.split(',').collect();

        for i in 0..4 {
            features.push(cols[i].parse::<f64>().unwrap());
        }

        labels.push(cols[4]);
    }

    let headers_array = Array1::from_vec(headers);
    let features_array = Array2::from_shape_vec((150, 4), features).unwrap();
    let labels_array = Array1::from_vec(labels);

    (headers_array, features_array, labels_array)
}

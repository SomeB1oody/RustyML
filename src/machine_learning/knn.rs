use super::DistanceCalculationMetric as Metric;
use super::preliminary_check;
use crate::ModelError;
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Represents the strategy used for weighting neighbors in KNN algorithm.
///
/// # Variants
///
/// - `Uniform` - Each neighbor is weighted equally
/// - `Distance` - Neighbors are weighted by the inverse of their distance (closer neighbors have greater influence)
#[derive(Debug, Clone, PartialEq)]
pub enum WeightingStrategy {
    Uniform,
    Distance,
}

/// K-Nearest Neighbors (KNN) Classifier
///
/// A non-parametric classification algorithm that classifies new data points
/// based on the majority class of its k nearest neighbors.
///
/// # Type Parameters
///
/// * `T` - The type of target values. Must implement `Clone`, `Hash`, and `Eq` traits.
///
/// # Fields
///
/// - `k` - Number of neighbors to consider for classification
/// - `x_train` - Training data features as a 2D array
/// - `y_train` - Training data labels/targets
/// - `weights` - Weight function for neighbor votes. Options: Uniform, Distance
/// - `metric` - Distance metric used for finding neighbors. Options: Euclidean, Manhattan, Minkowski(p=3)
///
/// # Examples
/// ```rust
/// use ndarray::{array, Array1, Array2};
/// use rustyml::machine_learning::knn::{KNN, WeightingStrategy};
/// use rustyml::machine_learning::DistanceCalculationMetric as Metric;
///
/// // Create a simple dataset
/// let x_train = array![
///     [1.0, 2.0],
///     [2.0, 3.0],
///     [3.0, 4.0],
///     [5.0, 6.0],
///     [6.0, 7.0]
/// ];
///
/// // Target values (classification)
/// let y_train = array!["A", "A", "A", "B", "B"];
///
/// // Create KNN model with k=3 and default settings
/// let mut knn = KNN::new(3, WeightingStrategy::Uniform, Metric::Euclidean);
///
/// // Fit the model
/// knn.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Predict new samples
/// let x_test = array![
///     [1.5, 2.5],  // Should be closer to class "A" points
///     [5.5, 6.5]   // Should be closer to class "B" points
/// ];
///
/// let predictions = knn.predict(x_test.view()).unwrap();
/// println!("Predictions: {:?}", predictions);  // Should print ["A", "B"]
/// ```
#[derive(Debug, Clone)]
pub struct KNN<T> {
    k: usize,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<T>>,
    weights: WeightingStrategy,
    metric: Metric,
}

impl<T: Clone + std::hash::Hash + Eq> Default for KNN<T> {
    /// Creates a new KNN classifier with default parameters:
    /// - k = 5
    /// - weights = Uniform
    /// - metric = Euclidean
    fn default() -> Self {
        KNN {
            k: 5,
            x_train: None,
            y_train: None,
            weights: WeightingStrategy::Uniform,
            metric: Metric::Euclidean,
        }
    }
}

impl<T: Clone + std::hash::Hash + Eq + Send + Sync> KNN<T> {
    /// Creates a new KNN classifier with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `k` - Number of neighbors to use for classification
    /// - `weights` - Weighting strategy for neighbor votes (Uniform or Distance)
    /// - `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// * `KNN` - A new KNN classifier instance
    pub fn new(k: usize, weights: WeightingStrategy, metric: Metric) -> Self {
        KNN {
            k,
            x_train: None,
            y_train: None,
            weights,
            metric,
        }
    }

    /// Returns the number of neighbors (k) used in the KNN algorithm
    ///
    /// # Returns
    ///
    /// * `usize` - The value of k, representing how many nearest neighbors are considered for predictions
    pub fn get_k(&self) -> usize {
        self.k
    }

    /// Returns the weighting strategy used in the KNN algorithm
    ///
    /// # Returns
    ///
    /// * `&WeightingStrategy` - A reference to the `WeightingStrategy` enum used by this instance
    pub fn get_weights(&self) -> &WeightingStrategy {
        &self.weights
    }

    /// Returns the distance metric used for calculating point similarities
    ///
    /// # Returns
    ///
    /// * `&DistanceCalculationMetric` - A reference to the Metric enum used by this instance
    pub fn get_metric(&self) -> &Metric {
        &self.metric
    }

    /// Returns a reference to the training features if available
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - A reference to the training data features if the model has been trained
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_x_train(&self) -> Result<&Array2<f64>, ModelError> {
        match self.x_train {
            Some(ref x) => Ok(x),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns a reference to the training labels if available
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<T>)` - A reference to the training data labels if the model has been trained
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_y_train(&self) -> Result<&Array1<T>, ModelError> {
        match self.y_train {
            Some(ref y) => Ok(y),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Fits the KNN classifier to the training data
    ///
    /// # Parameters
    ///
    /// - `x` - Training features as a 2D array (samples × features)
    /// - `y` - Training targets/labels as a 1D array
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The instance
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    ///
    /// # Notes
    ///
    /// KNN is a lazy learning algorithm, and the calculation is done in the prediction phase.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<T>) -> Result<&mut Self, ModelError> {
        preliminary_check(x, None)?;

        if x.nrows() < self.k {
            return Err(ModelError::InputValidationError(
                "The number of samples is less than k".to_string(),
            ));
        }

        // check if k is 0
        if self.k == 0 {
            return Err(ModelError::InputValidationError(
                "k must be greater than 0".to_string(),
            ));
        }

        self.x_train = Some(x.to_owned());
        self.y_train = Some(y.to_owned());

        Ok(self)
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<T>, ModelError> {
        use super::preliminary_check;

        // check if model is fitted
        if self.x_train.is_none() || self.y_train.is_none() {
            return Err(ModelError::NotFitted);
        }

        // validate input data
        preliminary_check(x, None)?;

        // check if feature dimension matches training data
        let x_train = self.x_train.as_ref().unwrap();
        if x.ncols() != x_train.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "Feature dimension mismatch: expected {}, got {}",
                x_train.ncols(),
                x.ncols()
            )));
        }

        // check if input data is empty
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input array is empty".to_string(),
            ));
        }

        let y_train = self.y_train.as_ref().unwrap();

        // Use rayon for parallel prediction
        let results: Vec<T> = (0..x.nrows())
            .into_par_iter() // Convert to parallel iterator
            .map(|i| {
                let sample = x.row(i);
                self.predict_one(sample, x_train.view(), y_train)
            })
            .collect();

        Ok(Array1::from(results))
    }

    /// Calculates the distance between two points based on the selected metric
    ///
    /// # Parameters
    ///
    /// - `a` - First point as a 1D array
    /// - `b` - Second point as a 1D array
    ///
    /// # Returns
    ///
    /// * `f64` - The calculated distance between points `a` and `b`
    ///
    /// # Notes
    /// Minkowski distance with p=3
    fn calculate_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        use crate::math::{
            manhattan_distance_row, minkowski_distance_row, squared_euclidean_distance_row,
        };

        match self.metric {
            Metric::Euclidean => squared_euclidean_distance_row(a, b).sqrt(),
            Metric::Manhattan => manhattan_distance_row(a, b),
            Metric::Minkowski => minkowski_distance_row(a, b, 3.0),
        }
    }

    /// Predicts the class for a single data point
    ///
    /// # Parameters
    ///
    /// - `x` - The data point to classify as a 1D array
    /// - `x_train` - Training data features
    /// - `y_train` - Training data labels
    ///
    /// # Returns
    ///
    /// * `T` - The predicted class for the data point
    fn predict_one(&self, x: ArrayView1<f64>, x_train: ArrayView2<f64>, y_train: &Array1<T>) -> T {
        let n_samples = x_train.nrows();
        let k = self.k.min(n_samples); // Ensure k doesn't exceed available samples

        // Calculate distances to all training samples in parallel
        let mut distances: Vec<(f64, usize)> = (0..n_samples)
            .into_par_iter() // Convert to parallel iterator
            .map(|i| {
                let distance = self.calculate_distance(x, x_train.row(i));
                (distance, i)
            })
            .collect();

        // Use partial sorting to get only k smallest elements instead of full sort
        distances.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let k_neighbors = &distances[..k];

        // Calculate based on weight strategy
        match self.weights {
            WeightingStrategy::Uniform => {
                // Count class occurrences using pre-sized HashMap
                let mut class_counts: AHashMap<&T, usize> = AHashMap::with_capacity(k);
                for &(_, idx) in k_neighbors {
                    let class = &y_train[idx];
                    *class_counts.entry(class).or_insert(0) += 1;
                }

                // Find the most common class
                class_counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(class, _)| class.clone())
                    .unwrap()
            }
            WeightingStrategy::Distance => {
                // Weight by inverse distance using pre-sized HashMap
                let mut class_weights: AHashMap<&T, f64> = AHashMap::with_capacity(k);
                for &(distance, idx) in k_neighbors {
                    // Handle zero distance case more efficiently
                    let weight = if distance == 0.0 {
                        // If distance is exactly zero, this sample gets maximum weight
                        return y_train[idx].clone();
                    } else {
                        1.0 / distance
                    };
                    let class = &y_train[idx];
                    *class_weights.entry(class).or_insert(0.0) += weight;
                }

                // Find the class with highest weight
                class_weights
                    .into_iter()
                    .max_by(|(_, weight_a), (_, weight_b)| weight_a.partial_cmp(weight_b).unwrap())
                    .map(|(class, _)| class.clone())
                    .unwrap()
            }
        }
    }

    /// Fits the model with the training data and immediately predicts on the given test data.
    ///
    /// This is a convenience method that combines the `fit` and `predict` steps into one operation
    ///
    /// # Parameters
    ///
    /// - `x_train` - The training feature matrix with shape (n_samples, n_features)
    /// - `y_train` - The training target values
    /// - `x_test` - The test feature matrix with shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Array of predicted values
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(
        &mut self,
        x_train: ArrayView2<f64>,
        y_train: ArrayView1<T>,
        x_test: ArrayView2<f64>,
    ) -> Result<Array1<T>, ModelError> {
        self.fit(x_train, y_train)?;
        Ok(self.predict(x_test)?)
    }
}

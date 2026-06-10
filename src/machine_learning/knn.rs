//! K-Nearest Neighbors (KNN) classification
//!
//! Provides the [`KNN`] classifier and the [`WeightingStrategy`] enum that controls how
//! neighbor votes are weighted

pub use super::DistanceCalculationMetric;
use super::validation::{check_is_fitted, preliminary_check, validate_predict_input};
use crate::error::Error;
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Strategy used for weighting neighbors in the KNN algorithm
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightingStrategy {
    /// Each neighbor is weighted equally
    Uniform,
    /// Neighbors are weighted by the inverse of their distance, so closer neighbors have greater influence
    Distance,
}

/// K-Nearest Neighbors (KNN) classifier
///
/// A non-parametric classification algorithm that classifies new data points
/// based on the majority class of their k nearest neighbors
///
/// # Examples
///
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
/// let mut knn = KNN::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
///
/// // Fit the model
/// knn.fit(&x_train, &y_train).unwrap();
///
/// // Predict new samples
/// let x_test = array![
///     [1.5, 2.5],  // Should be closer to class "A" points
///     [5.5, 6.5]   // Should be closer to class "B" points
/// ];
///
/// let predictions = knn.predict(&x_test).unwrap();
/// println!("Predictions: {:?}", predictions);  // Should print ["A", "B"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNN<T> {
    /// Number of neighbors to consider for classification
    k: usize,
    /// Training data features as a 2D array
    x_train: Option<Array2<f64>>,
    /// Encoded training labels as indices for efficient parallel computation
    y_train_encoded: Option<Array1<usize>>,

    /// Bidirectional mapping between original labels and their encoded indices: (label -> index, index -> label)
    #[serde(bound(
        serialize = "T: Serialize + Eq + std::hash::Hash",
        deserialize = "T: Deserialize<'de> + Eq + std::hash::Hash"
    ))]
    label_map: Option<(AHashMap<T, usize>, Vec<T>)>,

    /// Weight function for neighbor votes
    weighting_strategy: WeightingStrategy,
    /// Distance metric used for finding neighbors
    metric: DistanceCalculationMetric,
}

impl<T: Clone + std::hash::Hash + Eq> Default for KNN<T> {
    /// Creates a new KNN classifier with default parameters
    ///
    /// # Default Values
    ///
    /// - `k` - 5
    /// - `weighting_strategy` - WeightingStrategy::Uniform
    /// - `metric` - DistanceCalculationMetric::Euclidean
    fn default() -> Self {
        KNN {
            k: 5,
            x_train: None,
            y_train_encoded: None,
            label_map: None,
            weighting_strategy: WeightingStrategy::Uniform,
            metric: DistanceCalculationMetric::Euclidean,
        }
    }
}

impl<T: Clone + std::hash::Hash + Eq> KNN<T> {
    /// Creates a new KNN classifier with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `k` - Number of neighbors to use for classification
    /// - `weighting_strategy` - Weighting strategy for neighbor votes (Uniform or Distance)
    /// - `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - A new KNN classifier instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If k is 0
    pub fn new(
        k: usize,
        weighting_strategy: WeightingStrategy,
        metric: DistanceCalculationMetric,
    ) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::invalid_parameter("k", "must be greater than 0"));
        }

        Ok(KNN {
            k,
            x_train: None,
            y_train_encoded: None,
            label_map: None,
            weighting_strategy,
            metric,
        })
    }

    // Getters
    get_field!(get_k, k, usize);
    get_field!(
        get_weighting_strategy,
        weighting_strategy,
        WeightingStrategy
    );
    get_field!(get_metric, metric, DistanceCalculationMetric);
    get_field_as_ref!(get_x_train, x_train, Option<&Array2<f64>>);

    /// Fits the KNN classifier to the training data
    ///
    /// KNN is a lazy learning algorithm, and the actual calculation is done in the prediction phase;
    /// labels are internally encoded as indices for efficient parallel computation
    ///
    /// # Parameters
    ///
    /// - `x` - Training features as a 2D array (samples x features)
    /// - `y` - Training targets/labels as a 1D array
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The instance of KNN after being fitted
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` - If `x` has no rows
    /// - `Error::NonFinite` - If `x` contains NaN or infinite values
    /// - `Error::DimensionMismatch` - If the number of labels in `y` differs from the number of rows in `x`
    /// - `Error::InvalidInput` - If the number of samples is less than k
    pub fn fit<S1, S2>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        y: &ArrayBase<S2, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = T>,
    {
        preliminary_check(x, None)?;

        // `preliminary_check` cannot validate generic labels `T`, so check row counts here:
        // a mismatched `y` is otherwise silently stored and panics out-of-bounds at predict time
        if y.len() != x.nrows() {
            return Err(Error::dimension_mismatch(x.nrows(), y.len()));
        }

        if x.nrows() < self.k {
            return Err(Error::invalid_input("The number of samples is less than k"));
        }

        // Build label encoding map: label -> index and index -> label
        let mut label_to_idx: AHashMap<T, usize> = AHashMap::new();
        let mut idx_to_label: Vec<T> = Vec::new();
        let mut next_idx = 0;

        // Encode labels as indices
        let mut encoded_labels = Vec::with_capacity(y.len());
        for label in y.iter() {
            let idx = if let Some(&existing_idx) = label_to_idx.get(label) {
                existing_idx
            } else {
                let new_idx = next_idx;
                label_to_idx.insert(label.clone(), new_idx);
                idx_to_label.push(label.clone());
                next_idx += 1;
                new_idx
            };
            encoded_labels.push(idx);
        }

        self.x_train = Some(x.to_owned());
        self.y_train_encoded = Some(Array1::from(encoded_labels));
        self.label_map = Some((label_to_idx, idx_to_label));

        Ok(self)
    }

    /// Predicts class labels for input samples (sequential version)
    ///
    /// This method works with any type `T` without requiring `Sync + Send` bounds;
    /// for large datasets with types that implement `Sync + Send`, consider using
    /// `predict_parallel` for better performance
    ///
    /// # Parameters
    ///
    /// - `x` - Test features as a 2D array (samples x features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Predicted labels for each input sample
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been trained using `fit`
    /// - `Error::EmptyInput` - If `x` has no elements
    /// - `Error::DimensionMismatch` - If the number of features in `x` differs from the training data
    /// - `Error::NonFinite` - If `x` contains NaN or infinite values
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<T>, Error>
    where
        S: Data<Elem = f64>,
    {
        // check if model is fitted, then validate the prediction input
        check_is_fitted(
            self.x_train.is_some() && self.y_train_encoded.is_some() && self.label_map.is_some(),
            "KNN",
        )?;
        let x_train = self.x_train.as_ref().unwrap();
        validate_predict_input(x, x_train.ncols())?;

        let y_train_encoded = self.y_train_encoded.as_ref().unwrap();
        let (_, idx_to_label) = self.label_map.as_ref().unwrap();

        // Sequential prediction on encoded indices
        let encoded_results: Result<Vec<usize>, Error> = (0..x.nrows())
            .map(|i| {
                let sample = x.row(i);
                self.predict_one(sample, x_train.view(), y_train_encoded)
            })
            .collect();

        // Decode the predictions back to original labels
        encoded_results.map(|encoded_preds| {
            Array1::from(
                encoded_preds
                    .into_iter()
                    .map(|idx| idx_to_label[idx].clone())
                    .collect::<Vec<_>>(),
            )
        })
    }
}

impl<T: Clone + std::hash::Hash + Eq + Sync + Send> KNN<T> {
    /// Predicts class labels for input samples (parallel version)
    ///
    /// This method uses parallel computation for faster prediction on large datasets;
    /// requires `T` to implement `Sync + Send` for thread safety
    ///
    /// # Parameters
    ///
    /// - `x` - Test features as a 2D array (samples x features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Predicted labels for each input sample
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been trained
    /// - `Error::EmptyInput` - If `x` has no elements
    /// - `Error::DimensionMismatch` - If the number of features in `x` differs from the training data
    /// - `Error::NonFinite` - If `x` contains NaN or infinite values
    pub fn predict_parallel<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<T>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // check if model is fitted, then validate the prediction input
        check_is_fitted(
            self.x_train.is_some() && self.y_train_encoded.is_some() && self.label_map.is_some(),
            "KNN",
        )?;
        let x_train = self.x_train.as_ref().unwrap();
        validate_predict_input(x, x_train.ncols())?;

        let y_train_encoded = self.y_train_encoded.as_ref().unwrap();
        let (_, idx_to_label) = self.label_map.as_ref().unwrap();

        // Parallel prediction on encoded indices
        let encoded_results: Result<Vec<usize>, Error> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let sample = x.row(i);
                self.predict_one(sample, x_train.view(), y_train_encoded)
            })
            .collect();

        // Decode the predictions back to original labels
        encoded_results.map(|encoded_preds| {
            Array1::from(
                encoded_preds
                    .into_par_iter()
                    .map(|idx| idx_to_label[idx].clone())
                    .collect::<Vec<_>>(),
            )
        })
    }
}

impl<T: Clone + std::hash::Hash + Eq> KNN<T> {
    /// Predicts the encoded class index for a single data point
    fn predict_one(
        &self,
        x: ArrayView1<f64>,
        x_train: ArrayView2<f64>,
        y_train_encoded: &Array1<usize>,
    ) -> Result<usize, Error> {
        let n_samples = x_train.nrows();
        let k = self.k.min(n_samples); // Ensure k doesn't exceed available samples

        // Distances to all training samples; kept sequential because callers parallelize across
        // query samples (see `predict_parallel`), and nesting Rayon pools gives no real speedup
        let mut distances: Vec<(f64, usize)> = (0..n_samples)
            .map(|i| (self.metric.distance(x, x_train.row(i)), i))
            .collect();

        // Use partial sorting to get only k smallest elements instead of full sort
        distances.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal) // Handle NaN by treating as equal
        });
        let k_neighbors = &distances[..k];

        // Calculate based on weight strategy
        let result = match self.weighting_strategy {
            WeightingStrategy::Uniform => {
                // Threshold for using parallel voting aggregation
                const VOTING_PARALLEL_THRESHOLD: usize = 100;

                if k >= VOTING_PARALLEL_THRESHOLD {
                    // Parallel aggregation for large k values
                    let class_counts = k_neighbors
                        .par_iter()
                        .fold(
                            AHashMap::new,
                            |mut acc: AHashMap<usize, usize>, &(_, idx)| {
                                let class_idx = y_train_encoded[idx];
                                *acc.entry(class_idx).or_insert(0) += 1;
                                acc
                            },
                        )
                        .reduce(AHashMap::new, |mut a, b| {
                            for (class_idx, count) in b {
                                *a.entry(class_idx).or_insert(0) += count;
                            }
                            a
                        });

                    // Find the most common class
                    class_counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(class_idx, _)| class_idx)
                        .ok_or_else(|| {
                            Error::computation("No valid neighbors found for classification")
                        })?
                } else {
                    // Sequential counting for small k values
                    let mut class_counts: AHashMap<usize, usize> = AHashMap::with_capacity(k);
                    for &(_, idx) in k_neighbors {
                        let class_idx = y_train_encoded[idx];
                        *class_counts.entry(class_idx).or_insert(0) += 1;
                    }

                    // Find the most common class
                    class_counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(class_idx, _)| class_idx)
                        .ok_or_else(|| {
                            Error::computation("No valid neighbors found for classification")
                        })?
                }
            }
            WeightingStrategy::Distance => {
                // Check for zero distance early (exact match)
                if let Some(&(distance, idx)) = k_neighbors.first()
                    && distance == 0.0
                {
                    return Ok(y_train_encoded[idx]);
                }

                // Threshold for using parallel weight aggregation
                const WEIGHT_PARALLEL_THRESHOLD: usize = 100;

                if k >= WEIGHT_PARALLEL_THRESHOLD {
                    // Parallel weight aggregation for large k values
                    let class_weights = k_neighbors
                        .par_iter()
                        .fold(
                            AHashMap::new,
                            |mut acc: AHashMap<usize, f64>, &(distance, idx)| {
                                let weight = 1.0 / distance;
                                let class_idx = y_train_encoded[idx];
                                *acc.entry(class_idx).or_insert(0.0) += weight;
                                acc
                            },
                        )
                        .reduce(AHashMap::new, |mut a, b| {
                            for (class_idx, weight) in b {
                                *a.entry(class_idx).or_insert(0.0) += weight;
                            }
                            a
                        });

                    // Find the class with highest weight
                    class_weights
                        .into_iter()
                        .max_by(|(_, weight_a), (_, weight_b)| {
                            weight_a
                                .partial_cmp(weight_b)
                                .unwrap_or(std::cmp::Ordering::Equal) // Handle NaN/Inf by treating as equal
                        })
                        .map(|(class_idx, _)| class_idx)
                        .ok_or_else(|| {
                            Error::computation("No valid neighbors found for classification")
                        })?
                } else {
                    // Sequential weight calculation for small k values
                    let mut class_weights: AHashMap<usize, f64> = AHashMap::with_capacity(k);
                    for &(distance, idx) in k_neighbors {
                        let weight = 1.0 / distance;
                        let class_idx = y_train_encoded[idx];
                        *class_weights.entry(class_idx).or_insert(0.0) += weight;
                    }

                    // Find the class with highest weight
                    class_weights
                        .into_iter()
                        .max_by(|(_, weight_a), (_, weight_b)| {
                            weight_a
                                .partial_cmp(weight_b)
                                .unwrap_or(std::cmp::Ordering::Equal) // Handle NaN/Inf by treating as equal
                        })
                        .map(|(class_idx, _)| class_idx)
                        .ok_or_else(|| {
                            Error::computation("No valid neighbors found for classification")
                        })?
                }
            }
        };

        Ok(result)
    }

    /// Fits the model with the training data and immediately predicts on the given training data
    ///
    /// This is a convenience method that combines the `fit` and `predict` steps into one operation
    ///
    /// # Parameters
    ///
    /// - `x_train` - The training feature matrix with shape (n_samples, n_features)
    /// - `y_train` - The training target values
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Array of predicted values for the training data
    ///
    /// # Errors
    ///
    /// Returns any error produced by [`fit`](Self::fit) or [`predict`](Self::predict)
    pub fn fit_predict<S1, S2>(
        &mut self,
        x_train: &ArrayBase<S1, Ix2>,
        y_train: &ArrayBase<S2, Ix1>,
    ) -> Result<Array1<T>, Error>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = T>,
    {
        self.fit(x_train, y_train)?;
        self.predict(x_train)
    }
}

impl<T: Clone + std::hash::Hash + Eq + Serialize + for<'de> Deserialize<'de>> KNN<T> {
    model_save_and_load_methods!(KNN<T>);
}

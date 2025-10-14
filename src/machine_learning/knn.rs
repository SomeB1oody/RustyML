pub use super::*;

/// Threshold for using parallel distance calculation
const PARALLEL_THRESHOLD: usize = 1000;

/// Represents the strategy used for weighting neighbors in KNN algorithm.
///
/// # Variants
///
/// - `Uniform` - Each neighbor is weighted equally
/// - `Distance` - Neighbors are weighted by the inverse of their distance (closer neighbors have greater influence)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
/// - `y_train_encoded` - Encoded training labels as indices for efficient parallel computation
/// - `label_map` - Bidirectional mapping between original labels and their encoded indices
/// - `weighting_strategy` - Weight function for neighbor votes. Options: Uniform, Distance
/// - `metric` - Distance metric used for finding neighbors. Options: Euclidean, Manhattan, Minkowski(user can specify p)
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
/// let mut knn = KNN::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNN<T> {
    k: usize,
    x_train: Option<Array2<f64>>,
    y_train_encoded: Option<Array1<usize>>,

    #[serde(bound(
        serialize = "T: Serialize + Eq + std::hash::Hash",
        deserialize = "T: Deserialize<'de> + Eq + std::hash::Hash"
    ))]
    label_map: Option<(AHashMap<T, usize>, Vec<T>)>, // (label -> index, index -> label)

    weighting_strategy: WeightingStrategy,
    metric: DistanceCalculationMetric,
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
    /// - `weights` - Weighting strategy for neighbor votes (Uniform or Distance)
    /// - `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - A new KNN classifier instance
    /// - `Err(ModelError::InputValidationError)` - If k is 0
    pub fn new(
        k: usize,
        weighting_strategy: WeightingStrategy,
        metric: DistanceCalculationMetric,
    ) -> Result<Self, ModelError> {
        // Validate k parameter
        if k == 0 {
            return Err(ModelError::InputValidationError(
                "k must be greater than 0".to_string(),
            ));
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
    get_field_as_ref!(get_y_train_encoded, y_train_encoded, Option<&Array1<usize>>);
    get_field_as_ref!(
        get_label_map,
        label_map,
        Option<&(AHashMap<T, usize>, Vec<T>)>
    );

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
    /// Labels are internally encoded as indices for efficient parallel computation.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<T>) -> Result<&mut Self, ModelError> {
        preliminary_check(x, None)?;

        if x.nrows() < self.k {
            return Err(ModelError::InputValidationError(
                "The number of samples is less than k".to_string(),
            ));
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

    /// Predicts class labels for input samples (sequential version).
    ///
    /// This method works with any type `T` without requiring `Sync + Send` bounds.
    /// For large datasets with types that implement `Sync + Send`, consider using
    /// `predict_parallel` for better performance.
    ///
    /// # Parameters
    ///
    /// - `x` - Test features as a 2D array (samples × features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Predicted labels
    /// - `Err(ModelError)` - If model is not fitted or input is invalid
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<T>, ModelError> {
        use super::preliminary_check;

        // check if model is fitted
        if self.x_train.is_none() || self.y_train_encoded.is_none() || self.label_map.is_none() {
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

        let y_train_encoded = self.y_train_encoded.as_ref().unwrap();
        let (_, idx_to_label) = self.label_map.as_ref().unwrap();

        // Sequential prediction on encoded indices
        let encoded_results: Result<Vec<usize>, ModelError> = (0..x.nrows())
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
    /// Predicts class labels for input samples (parallel version).
    ///
    /// This method uses parallel computation for faster prediction on large datasets.
    /// Requires `T` to implement `Sync + Send` for thread safety.
    ///
    /// # Parameters
    ///
    /// - `x` - Test features as a 2D array (samples × features)
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<T>)` - Predicted labels
    /// - `Err(ModelError)` - If model is not fitted or input is invalid
    pub fn predict_parallel(&self, x: ArrayView2<f64>) -> Result<Array1<T>, ModelError> {
        use super::preliminary_check;

        // check if model is fitted
        if self.x_train.is_none() || self.y_train_encoded.is_none() || self.label_map.is_none() {
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

        let y_train_encoded = self.y_train_encoded.as_ref().unwrap();
        let (_, idx_to_label) = self.label_map.as_ref().unwrap();

        // Parallel prediction on encoded indices
        let encoded_results: Result<Vec<usize>, ModelError> = (0..x.nrows())
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
    /// Calculates the distance between two points based on the selected metric
    fn calculate_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match self.metric {
            DistanceCalculationMetric::Euclidean => squared_euclidean_distance_row(a, b).sqrt(),
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(a, b),
            DistanceCalculationMetric::Minkowski(p) => minkowski_distance_row(a, b, p),
        }
    }

    /// Predicts the encoded class index for a single data point
    fn predict_one(
        &self,
        x: ArrayView1<f64>,
        x_train: ArrayView2<f64>,
        y_train_encoded: &Array1<usize>,
    ) -> Result<usize, ModelError> {
        let n_samples = x_train.nrows();
        let k = self.k.min(n_samples); // Ensure k doesn't exceed available samples

        // Calculate distances to all training samples
        // Use parallel computation only when n_samples is large enough to benefit from it
        let mut distances: Vec<(f64, usize)> = if n_samples >= PARALLEL_THRESHOLD {
            (0..n_samples)
                .into_iter()
                .map(|i| -> Result<(f64, usize), ModelError> {
                    let distance = self.calculate_distance(x, x_train.row(i));
                    Ok((distance, i))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..n_samples)
                .map(|i| -> Result<(f64, usize), ModelError> {
                    let distance = self.calculate_distance(x, x_train.row(i));
                    Ok((distance, i))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

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
                            || AHashMap::new(),
                            |mut acc: AHashMap<usize, usize>, &(_, idx)| {
                                let class_idx = y_train_encoded[idx];
                                *acc.entry(class_idx).or_insert(0) += 1;
                                acc
                            },
                        )
                        .reduce(
                            || AHashMap::new(),
                            |mut a, b| {
                                for (class_idx, count) in b {
                                    *a.entry(class_idx).or_insert(0) += count;
                                }
                                a
                            },
                        );

                    // Find the most common class
                    class_counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(class_idx, _)| class_idx)
                        .ok_or(ModelError::ProcessingError(
                            "No valid neighbors found for classification".to_string(),
                        ))?
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
                        .ok_or(ModelError::ProcessingError(
                            "No valid neighbors found for classification".to_string(),
                        ))?
                }
            }
            WeightingStrategy::Distance => {
                // Check for zero distance early (exact match)
                if let Some(&(distance, idx)) = k_neighbors.first() {
                    if distance == 0.0 {
                        return Ok(y_train_encoded[idx]);
                    }
                }

                // Threshold for using parallel weight aggregation
                const WEIGHT_PARALLEL_THRESHOLD: usize = 100;

                if k >= WEIGHT_PARALLEL_THRESHOLD {
                    // Parallel weight aggregation for large k values
                    let class_weights = k_neighbors
                        .par_iter()
                        .fold(
                            || AHashMap::new(),
                            |mut acc: AHashMap<usize, f64>, &(distance, idx)| {
                                let weight = 1.0 / distance;
                                let class_idx = y_train_encoded[idx];
                                *acc.entry(class_idx).or_insert(0.0) += weight;
                                acc
                            },
                        )
                        .reduce(
                            || AHashMap::new(),
                            |mut a, b| {
                                for (class_idx, weight) in b {
                                    *a.entry(class_idx).or_insert(0.0) += weight;
                                }
                                a
                            },
                        );

                    // Find the class with highest weight
                    class_weights
                        .into_iter()
                        .max_by(|(_, weight_a), (_, weight_b)| {
                            weight_a
                                .partial_cmp(weight_b)
                                .unwrap_or(std::cmp::Ordering::Equal) // Handle NaN/Inf by treating as equal
                        })
                        .map(|(class_idx, _)| class_idx)
                        .ok_or(ModelError::ProcessingError(
                            "No valid neighbors found for classification".to_string(),
                        ))?
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
                        .ok_or(ModelError::ProcessingError(
                            "No valid neighbors found for classification".to_string(),
                        ))?
                }
            }
        };

        Ok(result)
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

impl<T: Clone + std::hash::Hash + Eq + Serialize + for<'de> Deserialize<'de>> KNN<T> {
    model_save_and_load_methods!(KNN<T>);
}

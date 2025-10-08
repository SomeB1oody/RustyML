use super::*;

/// Default minimum number of trees required to enable parallel tree construction
const DEFAULT_PARALLEL_THRESHOLD_TREES: usize = 10;

/// Default minimum number of samples required to enable parallel prediction
const DEFAULT_PARALLEL_THRESHOLD_SAMPLES: usize = 100;

/// An Isolation Forest implementation for anomaly detection.
///
/// Isolation Forest is an unsupervised learning algorithm that detects anomalies by isolating
/// outliers in the data. The algorithm works by randomly selecting a feature and then randomly
/// selecting a split value between the maximum and minimum values of the selected feature.
/// Anomalies are more susceptible to isolation and thus have shorter average path lengths in
/// the trees.
///
/// # Fields
///
/// - `trees` - Collection of isolation trees built during training
/// - `n_estimators` - Number of isolation trees in the forest
/// - `max_samples` - Maximum number of samples to draw from the dataset for each tree
/// - `max_depth` - Maximum depth of each isolation tree
/// - `random_state` - Random seed for reproducibility
/// - `n_features` - Number of features in the training data
///
/// # Example
/// ```rust
/// use rustyml::machine_learning::IsolationForest;
/// use ndarray::array;
///
/// let mut model = IsolationForest::new(100, 256, None, Some(42));
/// let data = array![[1.0, 2.0], [2.0, 3.0], [10.0, 15.0]];
/// model.fit(data.view()).unwrap();
/// let scores = model.predict(data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct IsolationForest {
    trees: Option<Vec<Box<Node>>>,
    n_estimators: usize,
    max_samples: usize,
    max_depth: usize,
    random_state: Option<u64>,
    n_features: usize,
}

/// Creates an Isolation Forest with default parameters.
///
/// # Default Values
///
/// - `trees` - None (not fitted)
/// - `n_estimators` - 100
/// - `max_samples` - 256
/// - `max_depth` - 8 (ceil(log2(256)))
/// - `random_state` - None
/// - `n_features` - 0
/// - `parallel_threshold_trees` - DEFAULT_PARALLEL_THRESHOLD_TREES
/// - `parallel_threshold_samples` - DEFAULT_PARALLEL_THRESHOLD_SAMPLES
impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            trees: None,
            n_estimators: 100,
            max_samples: 256,
            max_depth: 8, // ceil(log2(256)) = 8
            random_state: None,
            n_features: 0,
        }
    }
}

impl IsolationForest {
    /// Creates a new Isolation Forest with specified parameters.
    ///
    /// # Parameters
    ///
    /// - `n_estimators` - Number of isolation trees to build
    /// - `max_samples` - Maximum number of samples to draw from the dataset for each tree
    /// - `max_depth` - Maximum depth of each tree. If None, defaults to ceil(log2(max_samples))
    /// - `random_state` - Random seed for reproducibility. If None, uses non-deterministic seed
    ///
    /// # Returns
    ///
    /// * `IsolationForest` - A new unfitted `IsolationForest` instance
    pub fn new(
        n_estimators: usize,
        max_samples: usize,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        let computed_max_depth = max_depth.unwrap_or_else(|| {
            // ceil(log2(max_samples))
            (max_samples as f64).log2().ceil() as usize
        });

        Self {
            trees: None,
            n_estimators,
            max_samples,
            max_depth: computed_max_depth,
            random_state,
            n_features: 0,
        }
    }

    // Getters
    get_field!(get_n_estimators, n_estimators, usize);
    get_field!(get_max_samples, max_samples, usize);
    get_field!(get_max_depth, max_depth, usize);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field!(get_n_features, n_features, usize);
    get_field_as_ref!(get_trees, trees, &Option<Vec<Box<Node>>>);

    /// Trains the Isolation Forest model on the provided dataset.
    ///
    /// Builds multiple isolation trees by randomly sampling subsets of the data and
    /// recursively partitioning them using random feature splits. Uses parallelization
    /// when the number of trees exceeds the threshold.
    ///
    /// # Parameters
    ///
    /// * `x` - Training data as a 2D array where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, ModelError>` - A mutable reference to self if successful, or `ModelError` if:
    ///     - Input data is empty
    ///     - Input contains NaN or infinite values
    ///     - Tree building fails
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<&mut Self, ModelError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Check for NaN or infinite values
        if x.iter().any(|&v| v.is_nan() || v.is_infinite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        self.n_features = x.ncols();

        // Build multiple isolation trees
        let build_tree = |i: usize| -> Result<Box<Node>, ModelError> {
            // Create an independent RNG for each tree to maintain reproducibility
            let mut rng = if let Some(seed) = self.random_state {
                StdRng::seed_from_u64(seed.wrapping_add(i as u64))
            } else {
                StdRng::from_rng(&mut rng())
            };

            // Sample a subset of data for this tree
            let sample_size = self.max_samples.min(x.nrows());
            let sample_indices = self.sample_indices(x.nrows(), sample_size, &mut rng);

            // Build isolation tree
            self.build_isolation_tree(x, &sample_indices, 0, &mut rng)
                .map(Box::new)
        };

        let trees: Result<Vec<Box<Node>>, ModelError> =
            if self.n_estimators >= DEFAULT_PARALLEL_THRESHOLD_TREES {
                // Use parallelization for large number of trees
                (0..self.n_estimators)
                    .into_par_iter()
                    .map(build_tree)
                    .collect()
            } else {
                // Sequential execution for small number of trees
                (0..self.n_estimators).map(build_tree).collect()
            };

        self.trees = Some(trees?);
        Ok(self)
    }

    /// Randomly samples indices from the dataset using Fisher-Yates shuffle.
    fn sample_indices(&self, n: usize, sample_size: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle for the first sample_size elements
        for i in 0..sample_size {
            let j = rng.random_range(i..n);
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices
    }

    /// Recursively builds an isolation tree by randomly selecting features and split points.
    fn build_isolation_tree(
        &self,
        x: ArrayView2<f64>,
        indices: &[usize],
        current_depth: usize,
        rng: &mut StdRng,
    ) -> Result<Node, ModelError> {
        // Stopping criteria: max depth reached or node has only one sample
        if current_depth >= self.max_depth || indices.len() <= 1 {
            // Create a leaf node with the sample size as the value
            return Ok(Node::new_leaf(indices.len() as f64, None, None));
        }

        // Randomly select a feature
        let feature_index = rng.random_range(0..self.n_features);

        // Find min and max values for the selected feature in the current subset
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &idx in indices {
            let val = x[[idx, feature_index]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If all values are the same, create a leaf
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(Node::new_leaf(indices.len() as f64, None, None));
        }

        // Randomly select a split point between min and max
        let threshold = rng.random_range(min_val..max_val);

        // Split samples based on the threshold
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&idx| x[[idx, feature_index]] < threshold);

        // If split results in empty partition, create a leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(Node::new_leaf(indices.len() as f64, None, None));
        }

        // Create internal node and recursively build children
        let mut node = Node::new_internal(feature_index, threshold);
        node.left = Some(Box::new(self.build_isolation_tree(
            x,
            &left_indices,
            current_depth + 1,
            rng,
        )?));
        node.right = Some(Box::new(self.build_isolation_tree(
            x,
            &right_indices,
            current_depth + 1,
            rng,
        )?));

        Ok(node)
    }

    /// Computes the path length of a sample through an isolation tree with average adjustment for unresolved samples.
    fn path_length(&self, sample: &[f64], node: &Node, current_depth: usize) -> f64 {
        match &node.node_type {
            NodeType::Leaf { value, .. } => {
                // value stores the number of samples in this leaf
                // Add the average path length adjustment for unresolved samples
                current_depth as f64 + c(*value as usize)
            }
            NodeType::Internal {
                feature_index,
                threshold,
                ..
            } => {
                // Traverse to left or right child based on feature value
                if sample[*feature_index] < *threshold {
                    if let Some(ref left) = node.left {
                        self.path_length(sample, left, current_depth + 1)
                    } else {
                        current_depth as f64
                    }
                } else {
                    if let Some(ref right) = node.right {
                        self.path_length(sample, right, current_depth + 1)
                    } else {
                        current_depth as f64
                    }
                }
            }
        }
    }

    /// Computes the anomaly score for a single sample.
    ///
    /// The anomaly score is normalized to the range [0, 1], where values close to 1
    /// indicate anomalies and values close to 0 indicate normal samples.
    ///
    /// # Parameters
    ///
    /// * `sample` - A single data sample as a slice of feature values
    ///
    /// # Returns
    ///
    /// * `Result<f64, ModelError>` - The anomaly score between 0 and 1, or `ModelError` if:
    ///     - Model has not been fitted
    ///     - Sample feature dimension does not match training data
    pub fn anomaly_score(&self, sample: &[f64]) -> Result<f64, ModelError> {
        if self.trees.is_none() {
            return Err(ModelError::NotFitted);
        }

        if sample.len() != self.n_features {
            return Err(ModelError::InputValidationError(
                "Sample feature dimension mismatch".to_string(),
            ));
        }

        let trees = self.trees.as_ref().unwrap();

        // Compute average path length across all trees
        let avg_path_length: f64 = trees
            .iter()
            .map(|tree| self.path_length(sample, tree, 0))
            .sum::<f64>()
            / trees.len() as f64;

        // Normalize using c(max_samples)
        let c_n = c(self.max_samples);

        // Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
        let score = 2.0_f64.powf(-avg_path_length / c_n);

        Ok(score)
    }

    /// Predicts anomaly scores for multiple samples.
    ///
    /// Uses parallelization when the number of samples exceeds the threshold.
    ///
    /// # Parameters
    ///
    /// * `x` - Input data as a 2D array where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, ModelError>` - A 1D array of anomaly scores, or `ModelError` if:
    ///     - Model has not been fitted
    ///     - Input data is empty
    ///     - Feature dimension does not match training data
    ///     - Input contains NaN or infinite values
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        if self.trees.is_none() {
            return Err(ModelError::NotFitted);
        }

        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        if x.ncols() != self.n_features {
            return Err(ModelError::InputValidationError(
                "Feature dimension mismatch".to_string(),
            ));
        }

        // Check for NaN or infinite values
        if x.iter().any(|&v| v.is_nan() || v.is_infinite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        // Compute anomaly scores for all samples
        let scores: Result<Vec<f64>, ModelError> =
            if x.nrows() >= DEFAULT_PARALLEL_THRESHOLD_SAMPLES {
                // Use parallelization for large number of samples
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| self.anomaly_score(row.as_slice().unwrap()))
                    .collect()
            } else {
                // Sequential execution for small number of samples
                x.axis_iter(Axis(0))
                    .map(|row| self.anomaly_score(row.as_slice().unwrap()))
                    .collect()
            };

        Ok(Array1::from_vec(scores?))
    }

    /// Trains the model on the dataset and immediately predicts anomaly scores.
    ///
    /// This is a convenience method that combines `fit` and `predict` in one call.
    ///
    /// # Parameters
    ///
    /// * `x` - Training and prediction data as a 2D array
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, ModelError>` - A 1D array of anomaly scores for the training data, or `ModelError` if fitting or prediction fails
    pub fn fit_predict(&mut self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        self.fit(x)?;
        self.predict(x)
    }
}

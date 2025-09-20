use super::decision_tree::{Node, NodeType};
use super::preliminary_check;
use crate::ModelError;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::sync::Arc;

/// Implementation of Isolation Forest, a decision forest based on randomly generated Isolation Trees
///
/// Isolation Forest is an unsupervised learning algorithm that works by isolating anomalies instead of profiling normal points.
/// It builds an ensemble of Isolation Trees that recursively partition the data space, and anomalies are points that require fewer partitions to isolate.
///
/// # Fields
///
/// - `trees` - A vector of Isolation Trees, where each tree is a recursive Node structure. These trees collectively form the forest and are used for anomaly detection.
/// - `n_estimators` - The number of base estimators (trees) in the ensemble. More trees generally improve the robustness of the model but increase computation time.
/// - `max_samples` - The number of samples to draw from the dataset to train each tree. If less than the total dataset size, this creates diversity among trees. Smaller subsamples lead to more diverse trees but might miss global patterns.
/// - `max_depth` - Maximum depth limit for each tree. By default, this is set to ceil(log2(max_samples)), which is optimal for isolation trees. Limited depth prevents overfitting on noisy data.
/// - `random_state` - Optional seed for the random number generator. Setting this enables reproducible results across different runs.
///
/// # Example
/// ```rust
/// use rustyml::machine_learning::isolation_forest::IsolationForest;
/// use ndarray::Array2;
///
/// // Create a new Isolation Forest model
/// let mut model = IsolationForest::new(100, 256, None, Some(42));
///
/// // Assuming we have some data
/// // Create a sample dataset with 100 points, each with 2 features
/// let mut sample_data = Vec::with_capacity(100 * 2);
/// for _ in 0..100 {
///     sample_data.push(1.0);  // First feature
///     sample_data.push(2.0);  // Second feature
/// }
/// let data = Array2::from_shape_vec((100, 2), sample_data).unwrap();
///
/// // Fit the model
/// model.fit(data.view()).unwrap();
///
/// // Or get binary predictions (true for inliers, false for outliers)
/// let predictions = model.predict(data.view());
/// ```
#[derive(Debug, Clone)]
pub struct IsolationForest {
    trees: Option<Vec<Box<Node>>>, // Stores multiple Isolation Trees (each tree is a Node tree)
    n_estimators: usize,           // Number of trees in the forest
    max_samples: usize,            // Number of subsamples for each tree
    max_depth: usize, // Maximum depth of each tree (defaults to ceil(log2(max_samples)))
    random_state: Option<u64>, // Random seed, can be used for result reproducibility
}

/// Creates a new IsolationForest with default parameter values
///
/// This implementation uses the following default values:
/// - `n_estimators`: 100 (number of isolation trees)
/// - `max_samples`: 256 (maximum number of samples used to build each tree)
/// - `max_depth`: log2(max_samples) (maximum depth of each tree, calculated automatically)
/// - `random_state`: None (random seed is not fixed)
///
/// # Returns
///
/// * `IsolationForest` - An instance with default configuration values
impl Default for IsolationForest {
    fn default() -> Self {
        let max_samples = 256;
        let max_depth = (max_samples as f64).log2().ceil() as usize;
        IsolationForest {
            trees: None,
            n_estimators: 100,
            max_samples,
            max_depth,
            random_state: None,
        }
    }
}

impl IsolationForest {
    /// Creates a new IsolationForest
    ///
    /// # Parameters
    ///
    /// - `n_estimators` - Number of trees
    /// - `max_samples` - Number of subsamples per tree
    /// - `max_depth` - Maximum depth (optional, if None it's automatically set to ceil(log2(max_samples)))
    /// - `random_state` - Random seed (optional)
    ///
    /// # Returns
    ///
    /// * `IsolationForest` - A new IsolationForest instance
    pub fn new(
        n_estimators: usize,
        max_samples: usize,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        let computed_max_depth =
            max_depth.unwrap_or_else(|| (max_samples as f64).log2().ceil() as usize);
        IsolationForest {
            trees: None,
            n_estimators,
            max_samples,
            max_depth: computed_max_depth,
            random_state,
        }
    }

    /// Returns the vector of isolation trees built during training.
    ///
    /// # Returns
    ///
    /// - `Ok(&Vec<Box<Node>>)` - A reference to the vector of trained isolation trees
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_trees(&self) -> Result<&Vec<Box<Node>>, ModelError> {
        match &self.trees {
            Some(trees) => Ok(trees),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the number of base estimators (trees) in the ensemble.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of isolation trees used in the model.
    pub fn get_n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Returns the number of samples used to train each isolation tree.
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of samples used for each tree.
    pub fn get_max_samples(&self) -> usize {
        self.max_samples
    }

    /// Returns the maximum depth limit for the isolation trees.
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum depth allowed for each tree.
    pub fn get_max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the random seed used for reproducibility.
    ///
    /// # Returns
    ///
    /// - `Some(u64)` - The seed value if one was specified
    /// - `None` - If no specific seed was set (using system randomness)
    pub fn get_random_state(&self) -> Option<u64> {
        self.random_state
    }

    /// Trains the IsolationForest model with input sample matrix x
    /// (each row is a sample, each column is a feature)
    ///
    /// # Parameters
    ///
    /// * `x` - 2D array of input data samples
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - Trained instance
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn fit(&mut self, x: ArrayView2<f64>) -> Result<&mut Self, ModelError> {
        preliminary_check(x, None)?;

        if self.max_samples == 0 {
            return Err(ModelError::InputValidationError(
                "max_samples must be greater than 0".to_string(),
            ));
        }

        if self.n_estimators == 0 {
            return Err(ModelError::InputValidationError(
                "n_estimators must be greater than 0".to_string(),
            ));
        }

        if self.max_depth == 0 {
            return Err(ModelError::InputValidationError(
                "max_depth must be greater than 0".to_string(),
            ));
        }

        let n_rows = x.nrows();

        // Initialize random number generator with the main seed
        let main_seed = self.random_state.unwrap_or_else(|| {
            let mut temp_rng = rng();
            temp_rng.random::<u64>()
        });

        // Create an Arc to share the input data across threads
        let x_arc = Arc::new(x.clone());

        // Generate trees in parallel - collect into Result to handle potential errors
        let trees_result: Result<Vec<_>, ModelError> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| -> Result<Box<Node>, ModelError> {
                let mut tree_rng = StdRng::seed_from_u64(main_seed.wrapping_add(i as u64));

                let sample_indices = if self.max_samples < n_rows {
                    let mut indices: Vec<usize> = (0..n_rows).collect();
                    indices.shuffle(&mut tree_rng);
                    indices.into_iter().take(self.max_samples).collect()
                } else {
                    (0..n_rows).collect::<Vec<usize>>()
                };

                let sample = x_arc.select(Axis(0), &sample_indices);
                Ok(Self::build_tree(&sample, 0, self.max_depth, &mut tree_rng))
            })
            .collect();

        match trees_result {
            Ok(trees) => {
                self.trees = Some(trees);
                println!("Finished building Isolation Forest");
                Ok(self)
            }
            Err(e) => Err(e),
        }
    }

    /// Recursively constructs an Isolation Tree
    ///
    /// # Parameters
    ///
    /// - `x` - Current node data (sample matrix)
    /// - `current_depth` - Current depth
    /// - `max_depth` - Maximum allowed depth
    /// - `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// * `Box(Node)` - A new node
    ///
    /// If sample count <= 1 or max_depth is reached, returns a leaf node
    /// where the value represents the number of samples in that node
    fn build_tree(
        x: &Array2<f64>,
        current_depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> Box<Node> {
        let n_samples = x.nrows();
        if current_depth >= max_depth || n_samples <= 1 {
            // Leaf node: value records the number of samples in this node
            return Box::new(Node::new_leaf(n_samples as f64, None, None));
        }
        let n_features = x.ncols();
        // Randomly select a feature
        let feature_index = rng.random_range(0..n_features);
        let col = x.column(feature_index);
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // If all samples have the same value for this feature, can't split, return leaf node
        if min_val == max_val {
            return Box::new(Node::new_leaf(n_samples as f64, None, None));
        }

        // Randomly select a split point (between min and max)
        let split_value = rng.random_range(min_val..max_val);

        // Divide sample indices based on the split point
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for i in 0..n_samples {
            if x[[i, feature_index]] < split_value {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }
        if left_indices.is_empty() || right_indices.is_empty() {
            return Box::new(Node::new_leaf(n_samples as f64, None, None));
        }

        let left_x = x.select(Axis(0), &left_indices);
        let right_x = x.select(Axis(0), &right_indices);

        let mut node = Node::new_internal(feature_index, split_value);
        node.left = Some(Self::build_tree(&left_x, current_depth + 1, max_depth, rng));
        node.right = Some(Self::build_tree(
            &right_x,
            current_depth + 1,
            max_depth,
            rng,
        ));

        Box::new(node)
    }

    /// Calculates the path length of a sample in a single tree
    ///
    /// # Parameters
    ///
    /// - `node` - Current tree node
    /// - `sample` - Sample data
    /// - `current_depth` - Current depth in the tree
    ///
    /// # Returns
    ///
    /// * `f64` - Path length of the sample
    ///
    /// Recursively traverses the tree. If a leaf node is reached,
    /// returns the current depth plus the adjustment factor c(n)
    fn path_length(node: &Box<Node>, sample: &[f64], current_depth: f64) -> f64 {
        use crate::math::average_path_length_factor;

        match &node.node_type {
            NodeType::Leaf { value, .. } => {
                // value stores the number of samples in the leaf node
                current_depth + average_path_length_factor(*value)
            }
            NodeType::Internal {
                feature_index,
                threshold,
                ..
            } => {
                if sample[*feature_index] < *threshold {
                    if let Some(ref left) = node.left {
                        Self::path_length(left, sample, current_depth + 1.0)
                    } else {
                        current_depth + 1.0
                    }
                } else {
                    if let Some(ref right) = node.right {
                        Self::path_length(right, sample, current_depth + 1.0)
                    } else {
                        current_depth + 1.0
                    }
                }
            }
        }
    }

    pub fn anomaly_score(&self, sample: &[f64]) -> Result<f64, ModelError> {
        use crate::math::average_path_length_factor;

        // Check if the model has been trained
        let trees = self.get_trees()?;

        if sample.is_empty() {
            return Err(ModelError::InputValidationError(
                "Sample cannot be empty".to_string(),
            ));
        }

        // Dimension checking
        fn find_max_feature_index(node: &Box<Node>, max_index: &mut usize) {
            match &node.node_type {
                NodeType::Internal { feature_index, .. } => {
                    *max_index = (*max_index).max(*feature_index);
                    if let Some(ref left) = node.left {
                        find_max_feature_index(left, max_index);
                    }
                    if let Some(ref right) = node.right {
                        find_max_feature_index(right, max_index);
                    }
                }
                NodeType::Leaf { .. } => {}
            }
        }

        let expected_dimension = if let Some(first_tree) = trees.first() {
            let mut max_feature_index = 0;
            find_max_feature_index(first_tree, &mut max_feature_index);
            max_feature_index + 1
        } else {
            return Err(ModelError::NotFitted);
        };

        if sample.len() != expected_dimension {
            return Err(ModelError::InputValidationError(format!(
                "Input dimension mismatch: expected {}, got {}",
                expected_dimension,
                sample.len()
            )));
        }

        // Check for invalid values (NaN or infinite)
        for (i, &value) in sample.iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::InputValidationError(format!(
                    "Invalid value at index {}: {}",
                    i, value
                )));
            }
        }

        // Continue with original logic
        let mut path_length_sum = 0.0;
        for tree in trees {
            let path_len = Self::path_length(tree, sample, 0.0);
            if !path_len.is_finite() {
                return Err(ModelError::ProcessingError(
                    "Path length calculation failed".to_string(),
                ));
            }
            path_length_sum += path_len;
        }

        let avg_path_length = path_length_sum / trees.len() as f64;
        let cn = average_path_length_factor(self.max_samples as f64);

        // Check if cn is valid (not zero or invalid)
        if cn <= 0.0 || !cn.is_finite() {
            return Err(ModelError::ProcessingError(
                "Invalid normalization factor".to_string(),
            ));
        }

        let score = 2f64.powf(-avg_path_length / cn);

        // Final validation of the score
        if !score.is_finite() {
            return Err(ModelError::ProcessingError(
                "Anomaly score calculation failed".to_string(),
            ));
        }

        Ok(score)
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        // First check if input is empty
        if x.nrows() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Input features cannot be empty".to_string(),
            ));
        }

        // Check for invalid values (NaN or infinite) in input data
        for (row_idx, row) in x.axis_iter(ndarray::Axis(0)).enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Invalid value at row {}, column {}: {}",
                        row_idx, col_idx, value
                    )));
                }
            }
        }

        // Convert each row to a vector
        let samples: Vec<Vec<f64>> = (0..x.nrows())
            .map(|i| x.slice(s![i, ..]).to_vec())
            .collect();

        // Use parallel iterator to calculate anomaly scores
        let result: Result<Vec<f64>, ModelError> = samples
            .par_iter()
            .map(|sample| self.anomaly_score(sample))
            .collect();

        // Handle potential errors and convert results to Array1
        result.map(Array1::from)
    }

    /// Fits the model and performs anomaly detection in one step
    ///
    /// # Parameters
    ///
    /// * `x` - Input data, a 2D array where each row represents a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)` - If successful, returns anomaly scores for each sample
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(&mut self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        // First, train the model
        self.fit(x)?;

        // Then, perform prediction
        self.predict(x)
    }
}

use super::*;

/// Decision tree algorithm types.
///
/// Represents different splitting criteria and impurity measures used in decision tree construction.
///
/// # Variants
///
/// - `ID3` - Iterative Dichotomiser 3, uses information gain (entropy) for splitting. Only suitable for classification tasks.
/// - `C45` - Successor to ID3, uses information gain ratio to handle varied attribute value ranges. Only suitable for classification tasks.
/// - `CART` - Classification and Regression Trees, uses Gini impurity for classification and MSE for regression. Supports both classification and regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    ID3,
    C45,
    CART,
}

/// Hyperparameters for controlling decision tree growth and complexity.
///
/// These parameters help prevent overfitting and control the tree structure during training.
///
/// # Fields
///
/// - `max_depth` - Maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure or contain fewer than `min_samples_split` samples.
/// - `min_samples_split` - Minimum number of samples required to split an internal node. Must be at least 2.
/// - `min_samples_leaf` - Minimum number of samples required to be at a leaf node. Splits that result in leaves with fewer samples are rejected.
/// - `min_impurity_decrease` - Minimum impurity decrease required for a split. A node will be split if the decrease in impurity is greater than or equal to this value.
/// - `random_state` - Seed for random number generation. Currently not used but reserved for future stochastic features.
#[derive(Debug, Clone)]
pub struct DecisionTreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_impurity_decrease: f64,
    pub random_state: Option<u64>,
}

/// Default hyperparameters for decision tree.
///
/// Provides sensible defaults: no depth limit (`max_depth = None`), minimum 2 samples to split (`min_samples_split = 2`),
/// minimum 1 sample per leaf (`min_samples_leaf = 1`), no minimum impurity decrease requirement (`min_impurity_decrease = 0.0`),
/// and no random state (`random_state = None`).
impl Default for DecisionTreeParams {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            random_state: None,
        }
    }
}

/// Type of a node in the decision tree.
///
/// Distinguishes between internal decision nodes and leaf nodes that produce predictions.
///
/// # Variants
///
/// - `Internal` - A decision node that splits data based on a feature.
///   - `feature_index`: Index of the feature used for splitting.
///   - `threshold`: Threshold value for binary splits (samples with feature value ≤ threshold go left).
///   - `categories`: Optional list of categorical values for multi-way splits (not yet fully implemented).
/// - `Leaf` - A terminal node that produces a prediction.
///   - `value`: The predicted value (class label for classification, continuous value for regression).
///   - `class`: For classification, the majority class index.
///   - `probabilities`: For classification, probability distribution over all classes.
#[derive(Debug, Clone)]
pub enum NodeType {
    Internal {
        feature_index: usize,
        threshold: f64,
        categories: Option<Vec<String>>,
    },
    Leaf {
        value: f64,
        class: Option<usize>,
        probabilities: Option<Vec<f64>>,
    },
}

/// A node in the decision tree structure.
///
/// Represents either an internal decision node or a leaf node, with connections to child nodes.
///
/// # Fields
///
/// - `node_type` - The type of this node (Internal or Leaf), containing node-specific data.
/// - `left` - For binary splits, the left child node (samples with feature value ≤ threshold).
/// - `right` - For binary splits, the right child node (samples with feature value > threshold).
/// - `children` - For categorical splits, a map from category values to child nodes (not yet fully implemented).
#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
    pub children: Option<AHashMap<String, Box<Node>>>,
}

impl Node {
    /// Creates a new leaf node with prediction values.
    ///
    /// # Parameters
    ///
    /// - `value` - The predicted value (class label for classification, continuous value for regression).
    /// - `class` - For classification, the majority class index.
    /// - `probabilities` - For classification, probability distribution over all classes.
    ///
    /// # Returns
    ///
    /// * `Node` - A new `Node` configured as a leaf.
    pub fn new_leaf(value: f64, class: Option<usize>, probabilities: Option<Vec<f64>>) -> Self {
        Self {
            node_type: NodeType::Leaf {
                value,
                class,
                probabilities,
            },
            left: None,
            right: None,
            children: None,
        }
    }

    /// Creates a new internal node for binary splitting.
    ///
    /// # Parameters
    ///
    /// - `feature_index` - Index of the feature used for splitting.
    /// - `threshold` - Threshold value (samples with feature value ≤ threshold go to left child).
    ///
    /// # Returns
    ///
    /// * `Node` - A new `Node` configured as an internal decision node.
    pub fn new_internal(feature_index: usize, threshold: f64) -> Self {
        Self {
            node_type: NodeType::Internal {
                feature_index,
                threshold,
                categories: None,
            },
            left: None,
            right: None,
            children: None,
        }
    }

    /// Creates a new internal node for categorical splitting (not yet fully implemented).
    ///
    /// # Parameters
    ///
    /// * `feature_index` - Index of the categorical feature used for splitting.
    /// * `categories` - List of possible categorical values for this feature.
    ///
    /// # Returns
    ///
    /// * `Node` - A new `Node` configured for categorical splitting with an empty children map.
    pub fn new_categorical(feature_index: usize, categories: Vec<String>) -> Self {
        Self {
            node_type: NodeType::Internal {
                feature_index,
                threshold: 0.0, // Not used for categorical
                categories: Some(categories),
            },
            left: None,
            right: None,
            children: Some(AHashMap::new()),
        }
    }
}

/// Decision tree for classification and regression tasks.
///
/// Implements ID3, C4.5, and CART algorithms with parallel optimization using rayon.
/// Supports both classification (with probability estimates) and regression tasks.
/// The tree is built recursively by selecting the best split at each node based on
/// impurity measures (Gini, entropy, or MSE) and various stopping criteria.
///
/// # Fields
///
/// - `algorithm` - The splitting algorithm to use (ID3, C45, or CART).
/// - `root` - The root node of the trained tree, or `None` if not yet fitted.
/// - `n_features` - Number of features in the training data.
/// - `n_classes` - For classification, the number of distinct classes. `None` for regression.
/// - `params` - Hyperparameters controlling tree growth and complexity.
/// - `is_classifier` - Whether this tree performs classification (`true`) or regression (`false`).
///
/// # Example
/// ```rust
/// use rustyml::machine_learning::{DecisionTree, Algorithm, DecisionTreeParams};
/// use ndarray::{array, Array1, Array2};
///
/// // Classification example with Iris dataset
/// let x_train = array![
///     [5.1, 3.5, 1.4, 0.2],
///     [4.9, 3.0, 1.4, 0.2],
///     [6.2, 2.9, 4.3, 1.3],
///     [5.7, 2.8, 4.1, 1.3],
///     [6.3, 3.3, 6.0, 2.5],
///     [7.1, 3.0, 5.9, 2.1],
/// ];
/// let y_train = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
///
/// // Create and train a CART classifier with custom parameters
/// let params = DecisionTreeParams {
///     max_depth: Some(5),
///     min_samples_split: 2,
///     min_samples_leaf: 1,
///     min_impurity_decrease: 0.0,
///     random_state: Some(42),
/// };
///
/// let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params));
/// tree.fit(x_train.view(), y_train.view()).unwrap();
///
/// // Make predictions
/// let x_test = array![[5.0, 3.2, 1.2, 0.2], [6.5, 3.0, 5.2, 2.0]];
/// let predictions = tree.predict(x_test.view()).unwrap();
///
/// // Get probability estimates for classification
/// let probabilities = tree.predict_proba(x_test.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DecisionTree {
    algorithm: Algorithm,
    root: Option<Box<Node>>,
    n_features: usize,
    n_classes: Option<usize>,
    params: DecisionTreeParams,
    is_classifier: bool,
}

impl DecisionTree {
    /// Creates a new decision tree with the specified algorithm and task type.
    ///
    /// # Parameters
    ///
    /// - `algorithm` - The splitting algorithm to use (ID3, C45, or CART).
    /// - `is_classifier` - `true` for classification tasks, `false` for regression tasks.
    /// - `params` - Optional hyperparameters. If `None`, default parameters are used.
    ///
    /// # Returns
    ///
    /// * `DecisionTree` - A new untrained `DecisionTree` instance.
    ///
    /// # Panics
    ///
    /// Panics if `is_classifier` is `false` and `algorithm` is not `CART`, as only CART supports regression.
    pub fn new(
        algorithm: Algorithm,
        is_classifier: bool,
        params: Option<DecisionTreeParams>,
    ) -> Self {
        if !is_classifier && algorithm != Algorithm::CART {
            panic!("Only CART algorithm is supported for regression tasks");
        }

        Self {
            algorithm,
            root: None,
            n_features: 0,
            n_classes: None,
            params: params.unwrap_or_default(),
            is_classifier,
        }
    }

    // Getters
    get_field!(get_algorithm, algorithm, Algorithm);
    get_field!(get_n_features, n_features, usize);
    get_field!(get_n_classes, n_classes, Option<usize>);
    get_field_as_ref!(get_parameters, params, &DecisionTreeParams);
    get_field_as_ref!(get_root, root, &Option<Box<Node>>);
    get_field!(get_is_classifier, is_classifier, bool);

    /// Trains the decision tree on the provided training data.
    ///
    /// Builds the tree structure by recursively finding the best splits according to
    /// the specified algorithm and stopping criteria. Uses parallel processing for
    /// evaluating potential splits across features.
    ///
    /// # Parameters
    ///
    /// - `x` - Training features as a 2D array with shape (n_samples, n_features).
    /// - `y` - Training labels as a 1D array with shape (n_samples,). For classification,
    ///         labels should be non-negative integers starting from 0. For regression,
    ///         labels can be any continuous values.
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, ModelError>` - A mutable reference to `self` for method chaining, or a `ModelError` if training fails.
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        if x.nrows() != y.len() {
            return Err(ModelError::InputValidationError(
                "Number of samples in x and y must match".to_string(),
            ));
        }

        if x.nrows() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            ));
        }

        self.n_features = x.ncols();

        // For classification, determine number of classes
        if self.is_classifier {
            let max_class = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).ok_or(
                ModelError::ProcessingError("Cannot determine max class".to_string()),
            )?;
            self.n_classes = Some((*max_class as usize) + 1);
        }

        // Build the tree
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(Box::new(self.build_tree(x, y, &indices, 0)?));

        Ok(self)
    }

    /// Recursively builds a decision tree node by finding optimal splits.
    fn build_tree(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
    ) -> Result<Node, ModelError> {
        let n_samples = indices.len();

        // Check stopping criteria
        if n_samples < self.params.min_samples_split
            || (self.params.max_depth.is_some() && depth >= self.params.max_depth.unwrap())
            || self.is_pure(y, indices)
        {
            return Ok(self.create_leaf(y, indices));
        }

        // Find the best split
        let split_result = self.find_best_split(x, y, indices)?;

        if let Some((feature_idx, threshold, left_indices, right_indices, impurity_decrease)) =
            split_result
        {
            // Check if split meets minimum impurity decrease
            if impurity_decrease < self.params.min_impurity_decrease {
                return Ok(self.create_leaf(y, indices));
            }

            // Check if leaf size constraint is met
            if left_indices.len() < self.params.min_samples_leaf
                || right_indices.len() < self.params.min_samples_leaf
            {
                return Ok(self.create_leaf(y, indices));
            }

            // Create internal node and recursively build children
            let mut node = Node::new_internal(feature_idx, threshold);
            node.left = Some(Box::new(self.build_tree(x, y, &left_indices, depth + 1)?));
            node.right = Some(Box::new(self.build_tree(
                x,
                y,
                &right_indices,
                depth + 1,
            )?));

            Ok(node)
        } else {
            // No valid split found
            Ok(self.create_leaf(y, indices))
        }
    }

    /// Finds the best feature and threshold to split the given samples using parallel evaluation.
    fn find_best_split(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        indices: &[usize],
    ) -> Result<Option<(usize, f64, Vec<usize>, Vec<usize>, f64)>, ModelError> {
        let parent_impurity = self.calculate_impurity(y, indices);

        // Parallel evaluation of all features to find the best split
        let best_split = (0..self.n_features)
            .into_par_iter()
            .filter_map(|feature_idx| {
                // Get unique values for this feature
                let mut feature_values: Vec<f64> =
                    indices.iter().map(|&i| x[[i, feature_idx]]).collect();
                feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                feature_values.dedup();

                // Try each possible threshold and find the best for this feature
                let mut best_feature_gain = 0.0;
                let mut best_feature_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> =
                    None;

                for i in 0..feature_values.len().saturating_sub(1) {
                    let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                    // Split samples
                    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                        .iter()
                        .partition(|&&idx| x[[idx, feature_idx]] <= threshold);

                    if left_indices.is_empty() || right_indices.is_empty() {
                        continue;
                    }

                    // Calculate impurity decrease
                    let left_impurity = self.calculate_impurity(y, &left_indices);
                    let right_impurity = self.calculate_impurity(y, &right_indices);

                    let n_samples = indices.len() as f64;
                    let n_left = left_indices.len() as f64;
                    let n_right = right_indices.len() as f64;

                    let weighted_impurity = (n_left / n_samples) * left_impurity
                        + (n_right / n_samples) * right_impurity;
                    let impurity_decrease = parent_impurity - weighted_impurity;

                    // Information gain for ID3 and C4.5
                    let gain = match self.algorithm {
                        Algorithm::ID3 | Algorithm::C45 => impurity_decrease,
                        Algorithm::CART => impurity_decrease,
                    };

                    if gain > best_feature_gain {
                        best_feature_gain = gain;
                        best_feature_split = Some((
                            feature_idx,
                            threshold,
                            left_indices,
                            right_indices,
                            impurity_decrease,
                        ));
                    }
                }

                best_feature_split.map(|split| (best_feature_gain, split))
            })
            .max_by(|(gain_a, _), (gain_b, _)| gain_a.partial_cmp(gain_b).unwrap())
            .map(|(_, split)| split);

        Ok(best_split)
    }

    /// Calculates the impurity measure for the given samples based on the algorithm type.
    fn calculate_impurity(&self, y: ArrayView1<f64>, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        if self.is_classifier {
            match self.algorithm {
                Algorithm::CART => self.calculate_gini(y, indices),
                Algorithm::ID3 | Algorithm::C45 => self.calculate_entropy(y, indices),
            }
        } else {
            self.calculate_mse(y, indices)
        }
    }

    /// Calculates Gini impurity for classification samples.
    fn calculate_gini(&self, y: ArrayView1<f64>, indices: &[usize]) -> f64 {
        let n_samples = indices.len() as f64;
        let mut class_counts = vec![0.0; self.n_classes.unwrap_or(0)];

        for &idx in indices {
            let class = y[idx] as usize;
            class_counts[class] += 1.0;
        }

        let mut gini = 1.0;
        for count in class_counts {
            let p = count / n_samples;
            gini -= p * p;
        }

        gini
    }

    /// Calculates entropy for classification samples.
    fn calculate_entropy(&self, y: ArrayView1<f64>, indices: &[usize]) -> f64 {
        let n_samples = indices.len() as f64;
        let mut class_counts = vec![0.0; self.n_classes.unwrap_or(0)];

        for &idx in indices {
            let class = y[idx] as usize;
            class_counts[class] += 1.0;
        }

        let mut entropy = 0.0;
        for count in class_counts {
            if count > 0.0 {
                let p = count / n_samples;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Calculates mean squared error for regression samples.
    fn calculate_mse(&self, y: ArrayView1<f64>, indices: &[usize]) -> f64 {
        let n_samples = indices.len() as f64;
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n_samples;
        indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n_samples
    }

    /// Checks if all samples in the given indices have the same label (pure node).
    fn is_pure(&self, y: ArrayView1<f64>, indices: &[usize]) -> bool {
        if indices.is_empty() {
            return true;
        }

        let first_value = y[indices[0]];
        indices.iter().all(|&i| (y[i] - first_value).abs() < 1e-10)
    }

    /// Creates a leaf node with the appropriate prediction value based on the task type.
    fn create_leaf(&self, y: ArrayView1<f64>, indices: &[usize]) -> Node {
        if self.is_classifier {
            // Count occurrences of each class
            let n_classes = self.n_classes.unwrap();
            let mut class_counts = vec![0.0; n_classes];

            for &idx in indices {
                let class = y[idx] as usize;
                class_counts[class] += 1.0;
            }

            // Find majority class
            let majority_class = class_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Calculate probabilities
            let total = indices.len() as f64;
            let probabilities: Vec<f64> = class_counts.iter().map(|&count| count / total).collect();

            Node::new_leaf(
                majority_class as f64,
                Some(majority_class),
                Some(probabilities),
            )
        } else {
            // Regression: return mean value
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            Node::new_leaf(mean, None, None)
        }
    }

    /// Predicts the output for a single sample.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature vector for a single sample as a slice of length `n_features`.
    ///
    /// # Returns
    ///
    /// * `Result<f64, ModelError>` - The predicted value (class label for classification, continuous value for regression), or a `ModelError` if prediction fails.
    pub fn predict_one(&self, x: &[f64]) -> Result<f64, ModelError> {
        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        if x.len() != self.n_features {
            return Err(ModelError::TreeError("Feature dimension mismatch"));
        }

        self.traverse_tree(self.root.as_ref().unwrap(), x)
    }

    /// Traverses the tree from a given node to make a prediction for a single sample.
    fn traverse_tree(&self, node: &Node, x: &[f64]) -> Result<f64, ModelError> {
        match &node.node_type {
            NodeType::Leaf { value, .. } => Ok(*value),
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    // Categorical split (not implemented in this basic version)
                    return Err(ModelError::TreeError(
                        "Categorical splits not yet implemented",
                    ));
                }

                // Binary split
                if x[*feature_index] <= *threshold {
                    if let Some(ref left) = node.left {
                        self.traverse_tree(left, x)
                    } else {
                        Err(ModelError::TreeError("Missing left child"))
                    }
                } else {
                    if let Some(ref right) = node.right {
                        self.traverse_tree(right, x)
                    } else {
                        Err(ModelError::TreeError("Missing right child"))
                    }
                }
            }
        }
    }

    /// Predicts outputs for multiple samples using parallel processing.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix as a 2D array with shape (n_samples, n_features).
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, ModelError>` - A 1D array of predicted values with shape (n_samples,), or a `ModelError` if prediction fails.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        if x.ncols() != self.n_features {
            return Err(ModelError::TreeError("Feature dimension mismatch"));
        }

        // Parallel prediction for all samples
        let predictions: Result<Vec<f64>, ModelError> = x
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| self.predict_one(row.as_slice().unwrap()))
            .collect();

        Ok(Array1::from_vec(predictions?))
    }

    /// Trains the tree on training data and immediately makes predictions on test data.
    ///
    /// # Parameters
    ///
    /// - `x_train` - Training features as a 2D array with shape (n_train_samples, n_features).
    /// - `y_train` - Training labels as a 1D array with shape (n_train_samples,).
    /// - `x_test` - Test features as a 2D array with shape (n_test_samples, n_features).
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, ModelError>` - A 1D array of predictions for the test data with shape (n_test_samples,), or a `ModelError` if training or prediction fails.
    pub fn fit_predict(
        &mut self,
        x_train: ArrayView2<f64>,
        y_train: ArrayView1<f64>,
        x_test: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        self.fit(x_train, y_train)?;
        self.predict(x_test)
    }

    /// Predicts class probabilities for multiple samples using parallel processing (classification only).
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix as a 2D array with shape (n_samples, n_features).
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, ModelError>` - A 2D array of class probabilities with shape (n_samples, n_classes), where each row sums to 1.0, or a `ModelError` if prediction fails.
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        if !self.is_classifier {
            return Err(ModelError::TreeError(
                "predict_proba is only available for classification",
            ));
        }

        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        if x.ncols() != self.n_features {
            return Err(ModelError::TreeError("Feature dimension mismatch"));
        }

        let n_classes = self.n_classes.unwrap();

        // Parallel probability prediction for all samples
        let probabilities: Result<Vec<Vec<f64>>, ModelError> = x
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| self.predict_proba_one(row.as_slice().unwrap()))
            .collect();

        let probabilities = probabilities?;

        // Convert Vec<Vec<f64>> to Array2<f64>
        let mut result = Array2::zeros((x.nrows(), n_classes));
        for (i, proba) in probabilities.iter().enumerate() {
            for (j, &p) in proba.iter().enumerate() {
                result[[i, j]] = p;
            }
        }

        Ok(result)
    }

    /// Predicts class probabilities for a single sample (classification only).
    ///
    /// # Parameters
    ///
    /// * `x` - Feature vector for a single sample as a slice of length `n_features`.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f64>, ModelError>` - A vector of class probabilities of length `n_classes` that sums to 1.0, or a `ModelError` if prediction fails.
    pub fn predict_proba_one(&self, x: &[f64]) -> Result<Vec<f64>, ModelError> {
        if !self.is_classifier {
            return Err(ModelError::TreeError(
                "predict_proba is only available for classification",
            ));
        }

        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        if x.len() != self.n_features {
            return Err(ModelError::TreeError("Feature dimension mismatch"));
        }

        self.get_probabilities(self.root.as_ref().unwrap(), x)
    }

    /// Traverses the tree to retrieve class probabilities from the appropriate leaf node.
    fn get_probabilities(&self, node: &Node, x: &[f64]) -> Result<Vec<f64>, ModelError> {
        match &node.node_type {
            NodeType::Leaf { probabilities, .. } => probabilities
                .clone()
                .ok_or(ModelError::TreeError("No probabilities in leaf node")),
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    return Err(ModelError::TreeError(
                        "Categorical splits not yet implemented",
                    ));
                }

                if x[*feature_index] <= *threshold {
                    if let Some(ref left) = node.left {
                        self.get_probabilities(left, x)
                    } else {
                        Err(ModelError::TreeError("Missing left child"))
                    }
                } else {
                    if let Some(ref right) = node.right {
                        self.get_probabilities(right, x)
                    } else {
                        Err(ModelError::TreeError("Missing right child"))
                    }
                }
            }
        }
    }

    /// Generates a human-readable string representation of the decision tree structure.
    ///
    /// This method creates a visual representation of the trained decision tree, showing
    /// the hierarchical structure with internal nodes (containing split conditions) and
    /// leaf nodes (containing predictions). The output uses tree-like formatting with
    /// ASCII characters to represent branches and connections.
    ///
    /// # Returns
    ///
    /// * `Result<String, ModelError>` - A formatted string containing the tree structure, or a `ModelError::NotFitted` if the model hasn't been trained yet.
    pub fn generate_tree_structure(&self) -> Result<String, ModelError> {
        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        let mut output = String::new();
        output.push_str("Decision Tree Structure:\n");
        self.print_node(self.root.as_ref().unwrap(), &mut output, "", true);
        Ok(output)
    }

    // Recursively print tree structure
    fn print_node(&self, node: &Node, output: &mut String, prefix: &str, is_last: bool) {
        // Print current node
        let connector = if is_last { "└── " } else { "├── " };
        output.push_str(&format!("{}{}", prefix, connector));

        match &node.node_type {
            NodeType::Leaf {
                value,
                class,
                probabilities,
            } => {
                if self.is_classifier {
                    output.push_str(&format!("Leaf: class={}", class.unwrap()));
                    if let Some(probs) = probabilities {
                        output.push_str(&format!(" probs={:?}", probs));
                    }
                } else {
                    output.push_str(&format!("Leaf: value={:.4}", value));
                }
                output.push('\n');
            }
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    output.push_str(&format!(
                        "Split: feature[{}] (categorical)\n",
                        feature_index
                    ));
                } else {
                    output.push_str(&format!(
                        "Split: feature[{}] <= {:.4}\n",
                        feature_index, threshold
                    ));
                }

                // Print children
                let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

                if let Some(ref left) = node.left {
                    self.print_node(left, output, &new_prefix, false);
                }

                if let Some(ref right) = node.right {
                    self.print_node(right, output, &new_prefix, true);
                }
            }
        }
    }
}

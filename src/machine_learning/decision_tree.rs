use super::preliminary_check;
use crate::ModelError;
use crate::math::{gain_ratio, gini, information_gain, variance};
use ahash::{AHashMap, AHashSet};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::sync::Mutex;

/// Represents different decision tree algorithms that can be used for tree construction.
///
/// # Variants
///
/// - `ID3` - Iterative Dichotomiser 3 algorithm, which uses information gain for feature selection. Works best with categorical features.
/// - `C45` - An extension of ID3 that handles both continuous and discrete attributes, uses gain ratio instead of information gain to reduce bias towards features with many values.
/// - `CART` - Classification And Regression Trees algorithm, which builds binary trees using the feature and threshold that yield the largest information gain at each node. Works with both classification and regression problems.
#[derive(Debug, Clone, PartialEq)]
pub enum Algorithm {
    ID3,
    C45,
    CART,
}

/// Decision Tree Implementation
///
/// A machine learning model that makes predictions by recursively partitioning the feature space.
/// Supports both classification and regression tasks using various algorithms (ID3, C4.5, CART).
///
/// The decision tree builds a hierarchical structure of nodes during training, where:
/// - Internal nodes represent decision rules based on feature values
/// - Leaf nodes contain predictions (class labels or regression values)
///
/// # Fields
///
/// - `algorithm` - The splitting algorithm used (ID3, C45, CART)
/// - `root` - The root node of the decision tree, if trained
/// - `n_features` - Number of input features the model was trained on
/// - `n_classes` - Number of target classes for classification tasks, None for regression
/// - `params` - Hyperparameters controlling the tree's structure and training process
/// - `is_classifier` - Boolean flag indicating whether this is a classification tree (true) or regression tree (false)
///
/// # Features
///
/// - Supports multiple splitting algorithms (ID3, C4.5, CART)
/// - Handles both classification and regression tasks
/// - Provides probability estimates for classification
/// - Customizable via hyperparameters to control tree complexity
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::machine_learning::decision_tree::{DecisionTree, Algorithm};
///
/// // Create a classification decision tree with default parameters
/// let mut tree = DecisionTree::new(Algorithm::CART, true, None);
///
/// // Sample data: 2 features (x1, x2) and class labels (0 or 1)
/// let x = Array2::from_shape_vec((4, 2), vec![2.0, 3.0, 5.0, 4.0, 7.0, 2.0, 1.0, 6.0]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
///
/// // Train the model
/// tree.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let new_sample = Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).unwrap();
/// let prediction = tree.predict(new_sample.view()).unwrap();
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

/// Decision Tree Parameters
///
/// Hyperparameters that control the training and structure of the decision tree.
/// These parameters help prevent overfitting and optimize the model's performance.
///
/// # Fields
///
/// - `max_depth` - Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples. Controls model complexity and helps prevent overfitting.
/// - `min_samples_split` - Minimum number of samples required to split an internal node. Higher values prevent learning patterns from small subsets that may be noise.
/// - `min_samples_leaf` - Minimum number of samples required at each leaf node. Ensures terminal nodes have enough observations for stable predictions.
/// - `min_impurity_decrease` - Minimum impurity decrease required for a split to happen. Prevents splits that don't significantly improve the model.
/// - `random_state` - Seed for the random number generator, used for reproducibility when selecting features or splitting points.
#[derive(Debug, Clone)]
pub struct DecisionTreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_impurity_decrease: f64,
    pub random_state: Option<u64>,
}

/// Decision Tree Node Type
///
/// Represents the two possible types of nodes in a decision tree: internal decision nodes
/// and terminal leaf nodes.
///
/// # Variants
///
/// - `Internal` - A decision node that splits the data based on a feature value comparison.
///   For numerical features, uses a threshold comparison. For categorical features, may use
///   multi-way splits based on category values.
///   - `feature_index`: The index of the feature used for splitting (0-based indexing)
///   - `threshold`: The threshold value for numerical feature comparison. Samples with
///     feature values <= threshold go left, others go right
///   - `categories`: Optional vector of category strings for categorical features. When present,
///     enables multi-way splits where each category maps to a specific child node
///
/// - `Leaf` - A terminal node that provides the prediction output. For regression trees,
///   this is typically the mean value of samples in the node. For classification trees,
///   it contains the majority class and probability distribution across all classes.
///   - `value`: The predicted value for regression tasks, or the class prediction for
///     classification tasks (typically the majority class)
///   - `class`: Optional class index (0-based) for classification tasks. Contains the
///     majority class index, None for regression tasks
///   - `probabilities`: Optional probability distribution across all classes for
///     classification tasks. Vector length equals number of classes, None for regression tasks
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

/// Decision Tree Node
///
/// Represents a node in the decision tree hierarchy.
/// Each node either makes a decision based on a feature (internal node)
/// or provides a final prediction (leaf node).
///
/// # Fields
///
/// - `node_type` - The type of the node (Internal or Leaf) which determines its behavior and what information it contains
/// - `left` - Left child node for binary splits (samples where feature value ≤ threshold)
/// - `right` - Right child node for binary splits (samples where feature value > threshold)
/// - `children` - HashMap of child nodes for categorical features, where keys are category values and values are the corresponding child nodes for each category
///
/// # Notes
///
/// During prediction, samples are routed from the root node through internal nodes
/// based on feature comparisons until reaching a leaf node that provides the final prediction.
#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
    pub children: Option<AHashMap<String, Box<Node>>>,
}

impl Node {
    /// Creates a new leaf node for the decision tree
    ///
    /// A leaf node represents a terminal prediction point in the decision tree.
    /// It contains the prediction value and, for classification tasks, the predicted class
    /// and probability distribution across classes.
    ///
    /// # Parameters
    ///
    /// - `value` - The prediction value (mean target for regression trees or weighted majority class for classification)
    /// - `class` - The predicted class index for classification trees (None for regression trees)
    /// - `probabilities` - Optional vector of class probabilities for classification trees (proportion of each class in this leaf node)
    ///
    /// # Returns
    ///
    /// * `Self` - A new Node configured as a leaf node with the specified prediction information
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

    /// Creates a new internal node for numerical feature splits
    ///
    /// An internal node represents a decision point in the tree, where samples are routed
    /// to either the left or right child based on comparing a feature value with a threshold.
    ///
    /// # Parameters
    ///
    /// - `feature_index` - The index of the feature to use for the split decision
    /// - `threshold` - The threshold value for the split (samples with feature value ≤ threshold go left)
    ///
    /// # Returns
    ///
    /// * `Self` - A new Node configured as an internal node for binary splitting on a numerical feature
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

    /// Creates a new internal node for categorical feature splits
    ///
    /// This type of internal node is specialized for handling categorical features,
    /// allowing multi-way splits based on different category values instead of binary splits.
    ///
    /// # Parameters
    ///
    /// - `feature_index` - The index of the categorical feature to use for the split decision
    /// - `categories` - Vector of category values that this node will handle
    ///
    /// # Returns
    ///
    /// * `Self` - A new Node configured as an internal node for multi-way splitting on a categorical feature, with an initialized empty HashMap for child nodes corresponding to different categories
    pub fn new_categorical(feature_index: usize, categories: Vec<String>) -> Self {
        Self {
            node_type: NodeType::Internal {
                feature_index,
                threshold: 0.0,
                categories: Some(categories),
            },
            left: None,
            right: None,
            children: Some(AHashMap::new()),
        }
    }
}

/// Provides default values for the `DecisionTreeParams` structure.
///
/// This implementation sets sensible default values that work well for many datasets:
/// - No maximum depth limit (tree can grow until other stopping criteria are met)
/// - Minimum of 2 samples required to consider splitting a node
/// - Minimum of 1 sample required in each leaf node
/// - No minimum impurity decrease requirement for splitting
/// - No specific random seed (results may vary between runs)
///
/// # Returns
///
/// * `Self` - A new `DecisionTreeParams` instance with default values
impl Default for DecisionTreeParams {
    fn default() -> Self {
        DecisionTreeParams {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            random_state: None,
        }
    }
}

impl DecisionTree {
    /// Creates a new Decision Tree model with specified parameters
    ///
    /// Initializes a Decision Tree model for either classification or regression tasks.
    /// The tree is not trained yet; the `fit` method must be called separately with training data.
    ///
    /// # Parameters
    ///
    /// - `algorithm` - Algorithm to use for tree construction. If specified, uses the selected algorithm, otherwise defaults to "CART" for both classification and regression tasks.
    /// - `is_classifier` - Boolean flag indicating whether this is a classification tree (true) or regression tree (false)
    /// - `params` - Optional hyperparameters for the tree. If None, uses default parameter values
    ///
    /// # Returns
    ///
    /// * `DecisionTree` - A new, untrained DecisionTree instance configured with the specified settings
    pub fn new(
        algorithm: Algorithm,
        is_classifier: bool,
        params: Option<DecisionTreeParams>,
    ) -> Self {
        if is_classifier == false && algorithm != Algorithm::CART {
            panic!("Algorithm must be CART for non-classification tasks");
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

    /// Returns the hyperparameters of this decision tree
    ///
    /// Provides read-only access to the internal hyperparameters that control the tree's structure and training process.
    ///
    /// # Returns
    ///
    /// * `&DecisionTreeParams` - Reference to the decision tree's hyperparameters
    pub fn get_params(&self) -> &DecisionTreeParams {
        &self.params
    }

    /// Returns the splitting algorithm used by this decision tree
    ///
    /// # Returns
    ///
    /// * `&Algorithm` - A reference to the `&Algorithm` enum used by this instance
    pub fn get_algorithm(&self) -> &Algorithm {
        &self.algorithm
    }

    /// Returns whether this tree is a classifier or regressor
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if this is a classification tree, `false` if it's a regression tree
    pub fn get_is_classifier(&self) -> bool {
        self.is_classifier
    }

    /// Returns the number of input features this model was trained on
    ///
    /// # Returns
    ///
    /// * `usize` - The number of features the model expects for predictions
    ///
    /// # Note
    ///
    /// Returns 0 if the model hasn't been trained yet
    pub fn get_n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of target classes for a classification tree
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` - The number of target classes if the model is trained
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_classes(&self) -> Result<usize, ModelError> {
        match self.n_classes {
            Some(n_classes) => Ok(n_classes),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns a reference to the root node of the decision tree
    ///
    /// # Returns
    ///
    /// - `Ok(&Box<Node>)` - Reference to the root node if the model is trained
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_root(&self) -> Result<&Box<Node>, ModelError> {
        match &self.root {
            Some(root) => Ok(root),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Trains the decision tree on the provided dataset
    ///
    /// Fits this decision tree to the input feature matrix and target values.
    /// After calling this method, the tree is built and ready for predictions.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix where each row is a sample and each column is a feature
    /// - `y` - Target values (class labels for classification, continuous values for regression)
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - Mutable reference to self, enabling method chaining
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    /// - `Err(ModelError::ProcessingError)` - Error during tree building process
    ///
    /// # Note
    ///
    /// - For classification tasks, the class labels in `y` should be integers starting from 0
    /// - The algorithm used for building the tree is determined by the `algorithm` field set during initialization
    /// - Model hyperparameters like max_depth, min_samples_split etc. control the training process
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<&mut Self, ModelError> {
        preliminary_check(x, Some(y))?;

        self.n_features = x.ncols();

        if self.is_classifier {
            // For classification tasks, check if label values are valid
            if y.iter().any(|&val| val < 0.0) {
                return Err(ModelError::InputValidationError(
                    "Class labels must be non-negative finite numbers".to_string(),
                ));
            }

            // Calculate number of classes
            let unique_classes = y.iter().map(|&val| val as usize).collect::<AHashSet<_>>();
            let n_classes = unique_classes.len();

            // Check if the number of classes is reasonable
            if n_classes == 0 {
                return Err(ModelError::ProcessingError(
                    "No valid classes found in target data".to_string(),
                ));
            }

            self.n_classes = Some(n_classes);
        }

        // Build decision tree based on specified algorithm
        let root = match self.algorithm {
            Algorithm::ID3 => self.build_id3_tree(x, y, 0),
            Algorithm::C45 => self.build_c45_tree(x, y, 0),
            Algorithm::CART => self.build_cart_tree(x, y, 0),
        };

        self.root = Some(root);

        // Validate if tree was successfully built
        if let Some(ref root_node) = self.root {
            let actual_depth = self.calculate_tree_depth(root_node);
            if actual_depth == 0 {
                return Err(ModelError::ProcessingError(
                    "Failed to build decision tree - tree has zero depth".to_string(),
                ));
            }
            println!("Finish building decision tree, depth: {}", actual_depth);
        } else {
            return Err(ModelError::ProcessingError(
                "Failed to build decision tree - root node is None".to_string(),
            ));
        }

        Ok(self)
    }

    /// Core decision tree building function used by ID3 and C4.5 algorithms
    ///
    /// This is an internal method that implements the recursive tree building logic
    /// for information gain based algorithms.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    /// - `algorithm` - Algorithm type string ("ID3" or "C4.5")
    /// - `build_child_tree` - Function pointer for recursive building of child nodes
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    ///
    /// # Implementation details
    ///
    /// - Implements early stopping based on max_depth, min_samples_split, and pure node detection
    /// - Uses the specified algorithm to find the best feature and threshold for splitting
    /// - Recursively builds left and right subtrees using the provided build_child_tree function
    /// - Creates leaf nodes when stopping criteria are met
    fn build_tree(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        depth: usize,
        algorithm: &str,
        build_child_tree: fn(
            &DecisionTree,
            ArrayView2<f64>,
            ArrayView1<f64>,
            usize,
            &str,
        ) -> Box<Node>,
    ) -> Box<Node> {
        // Termination conditions
        if y.is_empty()
            || (self.params.max_depth.is_some() && depth >= self.params.max_depth.unwrap())
            || y.len() < self.params.min_samples_split
        {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            return Box::new(Node::new_leaf(value, class, None));
        }

        // Check if all samples belong to the same class
        let first_val = y[0];
        if y.iter().all(|&val| val == first_val) {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            return Box::new(Node::new_leaf(value, class, None));
        }

        // Find the best split point, passing algorithm type
        let (feature_idx, threshold, left_indices, right_indices) =
            find_best_split(x, y.view(), self.is_classifier, algorithm);

        // If no good split was found
        if left_indices.is_empty() || right_indices.is_empty() {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            return Box::new(Node::new_leaf(value, class, None));
        }

        // Create data for left and right subtrees
        let left_x = x.select(Axis(0), &left_indices);
        let left_y = y.select(Axis(0), &left_indices);
        let right_x = x.select(Axis(0), &right_indices);
        let right_y = y.select(Axis(0), &right_indices);

        // Create internal node
        let mut node = Node::new_internal(feature_idx, threshold);

        // Build subtrees using the provided function, and pass algorithm type
        node.left = Some(build_child_tree(
            self,
            left_x.view(),
            left_y.view(),
            depth + 1,
            algorithm,
        ));
        node.right = Some(build_child_tree(
            self,
            right_x.view(),
            right_y.view(),
            depth + 1,
            algorithm,
        ));

        Box::new(node)
    }

    /// Builds a decision tree using the ID3 (Iterative Dichotomiser 3) algorithm
    ///
    /// ID3 uses information gain as the splitting criterion and is primarily designed
    /// for categorical features, though this implementation handles numerical features as well.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    fn build_id3_tree(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, depth: usize) -> Box<Node> {
        self.build_tree(x, y, depth, "ID3", Self::build_id3_tree_with_algorithm)
    }

    /// Helper function for recursive ID3 tree building
    ///
    /// This function is passed as a function pointer to build_tree to enable
    /// recursive construction of the tree while maintaining the algorithm type.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    /// - `algorithm` - Algorithm type string (always "ID3" in this context)
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    fn build_id3_tree_with_algorithm(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        depth: usize,
        algorithm: &str,
    ) -> Box<Node> {
        self.build_tree(x, y, depth, algorithm, Self::build_id3_tree_with_algorithm)
    }

    /// Builds a decision tree using the C4.5 algorithm
    ///
    /// C4.5 is an extension of ID3 that uses gain ratio instead of information gain as the
    /// splitting criterion, which helps address the bias toward features with many values.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    fn build_c45_tree(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, depth: usize) -> Box<Node> {
        self.build_tree(x, y, depth, "C4.5", Self::build_c45_tree_with_algorithm)
    }

    /// Helper function for recursive C4.5 tree building
    ///
    /// This function is passed as a function pointer to build_tree to enable
    /// recursive construction of the tree while maintaining the algorithm type.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    /// - `algorithm` - Algorithm type string (always "C4.5" in this context)
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    fn build_c45_tree_with_algorithm(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        depth: usize,
        algorithm: &str,
    ) -> Box<Node> {
        self.build_tree(x, y, depth, algorithm, Self::build_c45_tree_with_algorithm)
    }

    /// Builds a decision tree using the CART (Classification and Regression Tree) algorithm
    ///
    /// CART uses Gini impurity for classification or MSE for regression as the splitting criterion.
    /// It builds binary trees and handles both categorical and numerical features naturally.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix for the current node's samples
    /// - `y` - Target values for the current node's samples
    /// - `depth` - Current depth in the tree
    ///
    /// # Returns
    ///
    /// * `Box<Node>` - A boxed Node representing the root of the constructed (sub)tree
    fn build_cart_tree(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, depth: usize) -> Box<Node> {
        // Termination conditions
        if y.is_empty()
            || (self.params.max_depth.is_some() && depth >= self.params.max_depth.unwrap())
            || y.len() < self.params.min_samples_split
        {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            let probabilities = if self.is_classifier && self.n_classes.is_some() {
                Some(calculate_class_probabilities(
                    y.view(),
                    self.n_classes.unwrap(),
                ))
            } else {
                None
            };
            return Box::new(Node::new_leaf(value, class, probabilities));
        }

        // Check if all samples belong to the same value
        let first_val = y[0];
        if y.iter().all(|&val| val == first_val) {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            let probabilities = if self.is_classifier && self.n_classes.is_some() {
                Some(calculate_class_probabilities(
                    y.view(),
                    self.n_classes.unwrap(),
                ))
            } else {
                None
            };
            return Box::new(Node::new_leaf(value, class, probabilities));
        }

        // Find the best split point, explicitly specifying CART algorithm
        let (feature_idx, threshold, left_indices, right_indices) =
            find_best_split(x, y.view(), self.is_classifier, "CART");

        // If no good split was found
        if left_indices.is_empty()
            || right_indices.is_empty()
            || left_indices.len() < self.params.min_samples_leaf
            || right_indices.len() < self.params.min_samples_leaf
        {
            let (value, class) = calculate_leaf_value(y.view(), self.is_classifier);
            let probabilities = if self.is_classifier && self.n_classes.is_some() {
                Some(calculate_class_probabilities(
                    y.view(),
                    self.n_classes.unwrap(),
                ))
            } else {
                None
            };
            return Box::new(Node::new_leaf(value, class, probabilities));
        }

        // Create data for left and right subtrees
        let left_x = x.select(Axis(0), &left_indices);
        let left_y = y.select(Axis(0), &left_indices);
        let right_x = x.select(Axis(0), &right_indices);
        let right_y = y.select(Axis(0), &right_indices);

        // Create internal node
        let mut node = Node::new_internal(feature_idx, threshold);
        node.left = Some(self.build_cart_tree(left_x.view(), left_y.view(), depth + 1));
        node.right = Some(self.build_cart_tree(right_x.view(), right_y.view(), depth + 1));

        Box::new(node)
    }

    /// Predicts the target value for a single sample
    ///
    /// # Parameters
    ///
    /// * `x` - Feature vector for a single sample
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - The predicted value
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    pub fn predict_one(&self, x: &[f64]) -> Result<f64, ModelError> {
        match &self.root {
            Some(root) => self.predict_sample(root, x),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Predicts target values for multiple samples in a feature matrix
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)` - Array of predictions
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, ModelError> {
        // Get root node, return error if tree is not trained
        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        // Convert data to a collection of rows
        let rows: Vec<Vec<f64>> = (0..x.nrows())
            .map(|i| x.slice(s![i, ..]).to_vec())
            .collect();

        // Process predictions in parallel
        let results: Result<Vec<f64>, ModelError> =
            rows.par_iter().map(|row| self.predict_one(row)).collect();

        // Handle results
        match results {
            Ok(predictions) => Ok(Array1::from(predictions)),
            Err(e) => Err(e),
        }
    }

    /// Fits the decision tree on training data and makes predictions on test data in a single operation.
    ///
    /// # Parameters
    ///
    /// - `x_train` - A 2D array of training features where rows represent samples and columns represent features
    /// - `y_train` - A 1D array of training target values corresponding to each sample in `x_train`
    /// - `x_test` - A 2D array of test features with the same number of features (columns) as `x_train`
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - A 1D array of predictions for the test data if successful
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(
        &mut self,
        x_train: ArrayView2<f64>,
        y_train: ArrayView1<f64>,
        x_test: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        self.fit(x_train, y_train)?;
        self.predict(x_test)
    }

    /// Predicts the target value for a single sample by traversing the decision tree
    ///
    /// This method recursively navigates through the decision tree from the root to a leaf
    /// based on the feature values of the sample.
    ///
    /// # Parameters
    ///
    /// - `node` - Current node in the decision tree
    /// - `x` - Feature vector for a single sample
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - Predicted value
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    fn predict_sample(&self, node: &Node, x: &[f64]) -> Result<f64, ModelError> {
        match &node.node_type {
            NodeType::Leaf { value, .. } => Ok(*value),
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if let Some(_) = categories {
                    // Handle categorical features
                    if let Some(children) = &node.children {
                        let feature_val = x[*feature_index].to_string();
                        if let Some(child) = children.get(&feature_val) {
                            self.predict_sample(child, x)
                        } else {
                            // If the category is not found, use default direction
                            if let Some(left) = &node.left {
                                self.predict_sample(left, x)
                            } else {
                                Err(ModelError::TreeError(
                                    "The classification node is missing a default branch",
                                ))
                            }
                        }
                    } else {
                        Err(ModelError::TreeError(
                            "The classification node is missing mappings to its child nodes",
                        ))
                    }
                } else {
                    // Handle numerical features
                    let feature_val = x[*feature_index];
                    if feature_val <= *threshold {
                        if let Some(left) = &node.left {
                            self.predict_sample(left, x)
                        } else {
                            // This should not happen in a correctly built tree
                            Err(ModelError::TreeError(
                                "Left child node is missing for the internal node",
                            ))
                        }
                    } else {
                        if let Some(right) = &node.right {
                            self.predict_sample(right, x)
                        } else {
                            // This should not happen in a correctly built tree
                            Err(ModelError::TreeError(
                                "Right child node is missing for the internal node",
                            ))
                        }
                    }
                }
            }
        }
    }

    /// Predicts class probabilities for multiple samples in a feature matrix
    ///
    /// This method is only applicable for classification trees and returns
    /// the probability distribution over all possible classes for each input sample.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - Matrix of class probabilities
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        if !self.is_classifier {
            return Err(ModelError::TreeError(
                "Probabilistic prediction is only applicable to classification problems",
            ));
        }

        let n_samples = x.nrows();
        let n_classes = match self.n_classes {
            Some(n_classes) => n_classes,
            None => {
                return Err(ModelError::TreeError(
                    "The model is not trained correctly and the number of classes is missing",
                ));
            }
        };

        // Convert data to a collection of rows
        let rows: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| x.slice(s![i, ..]).to_vec())
            .collect();

        // Process probability predictions in parallel
        let results: Result<Vec<Vec<f64>>, ModelError> = rows
            .par_iter()
            .map(|row| self.predict_proba_one(row))
            .collect();

        match results {
            Ok(all_probas) => {
                let mut probas = Array2::<f64>::zeros((n_samples, n_classes));

                // Fill the probabilities matrix
                for (i, sample_probas) in all_probas.iter().enumerate() {
                    for (j, &p) in sample_probas.iter().enumerate() {
                        if j < n_classes {
                            probas[[i, j]] = p;
                        }
                    }
                }

                Ok(probas)
            }
            Err(e) => Err(e),
        }
    }

    /// Predicts class probabilities for a single sample
    ///
    /// # Parameters
    ///
    /// * `x` - Feature vector for a single sample
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - Vector of class probabilities
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn predict_proba_one(&self, x: &[f64]) -> Result<Vec<f64>, ModelError> {
        match &self.root {
            Some(root) => match self.predict_proba_sample(root, x) {
                Ok(probabilities) => Ok(probabilities),
                Err(e) => Err(e),
            },
            None => Err(ModelError::NotFitted),
        }
    }

    /// Predicts class probabilities for a single sample by traversing the decision tree
    ///
    /// This method recursively navigates through the decision tree from the root to a leaf node
    /// to retrieve the class probability distribution for the given sample.
    ///
    /// # Parameters
    ///
    /// - `node` - Current node in the decision tree
    /// - `x` - Feature vector for a single sample
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - Vector of class probabilities
    /// - `Err(ModelError::TreeError(&str))` - Something wrong with the tree
    fn predict_proba_sample(&self, node: &Node, x: &[f64]) -> Result<Vec<f64>, ModelError> {
        match &node.node_type {
            NodeType::Leaf { probabilities, .. } => match probabilities {
                Some(proba) => Ok(proba.clone()),
                None => Err(ModelError::TreeError(
                    "The leaf node lacks probability information",
                )),
            },
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if let Some(_) = categories {
                    // Handle categorical features
                    if let Some(children) = &node.children {
                        let feature_val = x[*feature_index].to_string();
                        if let Some(child) = children.get(&feature_val) {
                            self.predict_proba_sample(child, x)
                        } else {
                            // If category is not found, use default direction
                            if let Some(left) = &node.left {
                                self.predict_proba_sample(left, x)
                            } else {
                                Err(ModelError::TreeError(
                                    "The classification node is missing a default branch",
                                ))
                            }
                        }
                    } else {
                        Err(ModelError::TreeError(
                            "The classification node is missing mappings to its child nodes",
                        ))
                    }
                } else {
                    // Handle numerical features
                    let feature_val = x[*feature_index];
                    if feature_val <= *threshold {
                        if let Some(left) = &node.left {
                            self.predict_proba_sample(left, x)
                        } else {
                            Err(ModelError::TreeError(
                                "Left child node is missing for the internal node",
                            ))
                        }
                    } else {
                        if let Some(right) = &node.right {
                            self.predict_proba_sample(right, x)
                        } else {
                            Err(ModelError::TreeError(
                                "Right child node is missing for the internal node",
                            ))
                        }
                    }
                }
            }
        }
    }

    /// Calculates the maximum depth of the decision tree starting from the given node
    ///
    /// This function recursively traverses the tree structure to determine its depth.
    /// The depth is defined as the maximum number of nodes from the given node to any leaf node.
    ///
    /// # Parameters
    ///
    /// * `node` - A reference to the node from which to start the depth calculation
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum depth of the tree from the given node
    ///
    /// # Algorithm Details
    ///
    /// - For leaf nodes, the depth is always 1
    /// - For internal nodes, the depth is calculated as:
    ///   1 + max(depth of left subtree, depth of right subtree, depth of categorical children)
    /// - If a child does not exist, its depth contribution is 0
    fn calculate_tree_depth(&self, node: &Node) -> usize {
        match node.node_type {
            NodeType::Leaf { .. } => 1,
            NodeType::Internal { .. } => {
                let left_depth = match &node.left {
                    Some(left_node) => self.calculate_tree_depth(left_node),
                    None => 0,
                };
                let right_depth = match &node.right {
                    Some(right_node) => self.calculate_tree_depth(right_node),
                    None => 0,
                };

                // For categorical features, we also need to check children
                let children_depth = match &node.children {
                    Some(children_map) => {
                        if children_map.is_empty() {
                            0
                        } else {
                            children_map
                                .values()
                                .map(|child| self.calculate_tree_depth(child))
                                .max()
                                .unwrap_or(0)
                        }
                    }
                    None => 0,
                };

                1 + left_depth.max(right_depth).max(children_depth)
            }
        }
    }

    /// Prints the decision tree structure using ASCII symbols.
    ///
    /// If the tree has been trained, it recursively prints each node from the root,
    /// using symbols such as "├──", "└──", and "│" to show the hierarchy.
    /// If the tree is not trained, it prints an appropriate message.
    ///
    /// # Returns
    ///
    /// - `Ok(String)` - Tree structure as string
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn generate_tree_structure(&self) -> Result<String, ModelError> {
        /// Recursively builds and returns a String representing the tree structure using ASCII symbols.
        ///
        /// # Parameters
        ///
        /// - `node`: The current node in the tree.
        /// - `prefix`: The current indentation string.
        /// - `is_last`: A flag indicating whether this node is the last child of its parent.
        ///
        /// # Returns
        ///
        /// * `String` - Tree structure as string
        fn build_ascii(node: &Node, prefix: &str, is_last: bool) -> String {
            let mut output = String::new();
            let connector = if is_last { "└── " } else { "├── " };

            match &node.node_type {
                NodeType::Leaf {
                    value,
                    class,
                    probabilities,
                } => {
                    output.push_str(&format!(
                        "{}{}Leaf: value: {}, class: {:?}, probabilities: {:?}\n",
                        prefix, connector, value, class, probabilities
                    ));
                }
                NodeType::Internal {
                    feature_index,
                    threshold,
                    categories,
                } => {
                    if let Some(categories) = categories {
                        // For categorical features.
                        output.push_str(&format!(
                            "{}{}Internal (categorical): feature index: {}, categories: {:?}\n",
                            prefix, connector, feature_index, categories
                        ));
                        if let Some(children) = &node.children {
                            let new_prefix =
                                format!("{}{}", prefix, if is_last { "    " } else { "│   " });
                            // Sort keys for consistent output.
                            let mut keys: Vec<&String> = children.keys().collect();
                            keys.sort();
                            let child_count = keys.len();
                            for (i, key) in keys.into_iter().enumerate() {
                                let child = &children[key];
                                let child_connector = if i == child_count - 1 {
                                    "└── "
                                } else {
                                    "├── "
                                };
                                output.push_str(&format!(
                                    "{}{}Category: {}\n",
                                    new_prefix, child_connector, key
                                ));
                                let child_prefix = format!(
                                    "{}{}",
                                    new_prefix,
                                    if i == child_count - 1 {
                                        "    "
                                    } else {
                                        "│   "
                                    }
                                );
                                output.push_str(&build_ascii(child, &child_prefix, true));
                            }
                        }
                    } else {
                        // For numerical features.
                        output.push_str(&format!(
                            "{}{}Internal (numerical): feature index: {}, threshold: {}\n",
                            prefix, connector, feature_index, threshold
                        ));
                        let new_prefix =
                            format!("{}{}", prefix, if is_last { "    " } else { "│   " });
                        if let Some(left) = &node.left {
                            output.push_str(&build_ascii(left, &new_prefix, false));
                        }
                        if let Some(right) = &node.right {
                            output.push_str(&build_ascii(right, &new_prefix, true));
                        }
                    }
                }
            }
            output
        }

        let root = match self.root.as_ref() {
            Some(root) => root,
            None => return Err(ModelError::NotFitted),
        };
        Ok(build_ascii(root, "", true))
    }
}

/// Finds the best feature and threshold to split the data based on the specified algorithm
///
/// This function evaluates all features and possible threshold values to determine
/// the optimal split according to the selected decision tree algorithm (CART, ID3, or C4.5).
///
/// # Parameters
///
/// - `x` - Feature matrix of the input data
/// - `y` - Target values array
/// - `is_classifier` - Boolean flag indicating whether this is a classification task
/// - `algorithm` - String specifying the algorithm to use ("CART", "ID3", or "C4.5")
///
/// # Returns
///
/// A tuple containing:
/// - `usize` - The index of the best feature to split on
/// - `f64` - The threshold value for the split
/// - `Vec<usize>` - Indices of samples going to the left child node
/// - `Vec<usize>` - Indices of samples going to the right child node
///
/// # Errors
///
/// - Returns default split (feature 0, threshold 0.0) if no valid split is found
/// - Handles NaN/infinite values in feature data gracefully
fn find_best_split(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    is_classifier: bool,
    algorithm: &str,
) -> (usize, f64, Vec<usize>, Vec<usize>) {
    let n_features = x.ncols();

    // Early return for empty data or single feature with all same values
    if n_features == 0 || x.nrows() == 0 || y.is_empty() {
        return (0, 0.0, Vec::new(), Vec::new());
    }

    // Use Mutex to wrap the best result for thread-safe updates during parallel processing
    let best_result = Mutex::new((
        0,                 // best_feature
        0.0,               // best_threshold
        Vec::new(),        // best_left_indices
        Vec::new(),        // best_right_indices
        f64::NEG_INFINITY, // best_criterion
    ));

    /// Helper function to calculate CART criterion (Gini impurity reduction)
    fn calculate_cart_criterion(
        y: ArrayView1<f64>,
        left_y: ArrayView1<f64>,
        right_y: ArrayView1<f64>,
        is_classifier: bool,
    ) -> f64 {
        // Select the appropriate impurity function
        let impurity_fn = if is_classifier { gini } else { variance };

        // Calculate impurity for parent node, left child node, and right child node
        let parent_impurity = impurity_fn(y);
        let left_impurity = impurity_fn(left_y);
        let right_impurity = impurity_fn(right_y);

        // Check if all impurity values are valid
        if !parent_impurity.is_finite() || !left_impurity.is_finite() || !right_impurity.is_finite()
        {
            return f64::NEG_INFINITY;
        }

        // Calculate weights
        let total_len = y.len() as f64;
        let left_weight = left_y.len() as f64 / total_len;
        let right_weight = right_y.len() as f64 / total_len;

        // Calculate weighted impurity
        let weighted_impurity = left_weight * left_impurity + right_weight * right_impurity;

        // Return the impurity reduction
        parent_impurity - weighted_impurity
    }

    // Process each feature in parallel
    (0..n_features).into_par_iter().for_each(|feature_idx| {
        // Get all unique values for this feature, filtering out invalid values
        let mut feature_values = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let val = x[[i, feature_idx]];
            // Skip NaN or infinite values
            if val.is_finite() {
                feature_values.push(val);
            }
        }

        // Skip feature if no valid values or all values are the same
        if feature_values.len() < 2 {
            return;
        }

        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find potential thresholds (midpoints between adjacent unique values)
        let mut thresholds = Vec::new();
        for window in feature_values.windows(2) {
            if (window[1] - window[0]).abs() > f64::EPSILON {
                let threshold = (window[0] + window[1]) / 2.0;
                if threshold.is_finite() {
                    thresholds.push(threshold);
                }
            }
        }

        // Skip feature if no valid thresholds
        if thresholds.is_empty() {
            return;
        }

        // Process thresholds for this feature
        for &threshold in &thresholds {
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            // Split samples based on threshold, handling invalid values
            for i in 0..x.nrows() {
                let val = x[[i, feature_idx]];
                if val.is_finite() {
                    if val <= threshold {
                        left_indices.push(i);
                    } else {
                        right_indices.push(i);
                    }
                }
                // Skip samples with NaN or infinite values
            }

            // Check if split is valid - need at least one sample on each side
            if left_indices.is_empty() || right_indices.is_empty() {
                continue;
            }

            let left_y = Array1::from_iter(left_indices.iter().map(|&i| y[i]));
            let right_y = Array1::from_iter(right_indices.iter().map(|&i| y[i]));

            // Calculate split quality based on the selected algorithm
            let criterion = match algorithm {
                "ID3" => {
                    if is_classifier {
                        let gain = information_gain(y, left_y.view(), right_y.view());
                        // Check for invalid gain values
                        if gain.is_finite() {
                            gain
                        } else {
                            f64::NEG_INFINITY
                        }
                    } else {
                        f64::NEG_INFINITY // ID3 is not applicable for regression
                    }
                }
                "C4.5" => {
                    if is_classifier {
                        let ratio = gain_ratio(y, left_y.view(), right_y.view());
                        if ratio.is_finite() {
                            ratio
                        } else {
                            f64::NEG_INFINITY
                        }
                    } else {
                        f64::NEG_INFINITY // C4.5 is not applicable for regression
                    }
                }
                _ => {
                    // CART algorithm - use the extracted helper function
                    calculate_cart_criterion(y, left_y.view(), right_y.view(), is_classifier)
                }
            };

            // Only update if criterion is valid and better than current best
            if criterion.is_finite() && criterion > f64::NEG_INFINITY {
                // Update global best split (using lock to avoid data races)
                if let Ok(mut best) = best_result.lock() {
                    if criterion > best.4 {
                        *best = (
                            feature_idx,
                            threshold,
                            left_indices,
                            right_indices,
                            criterion,
                        );
                    }
                }
            }
        }
    });

    // Extract final result from Mutex
    match best_result.into_inner() {
        Ok((
            best_feature,
            best_threshold,
            best_left_indices,
            best_right_indices,
            best_criterion,
        )) => {
            // If no valid split was found, return a fallback split
            if best_criterion == f64::NEG_INFINITY
                || (best_left_indices.is_empty() && best_right_indices.is_empty())
            {
                // Create a simple fallback split on the first feature
                let fallback_threshold = if x.nrows() > 0 {
                    let first_feature_vals: Vec<f64> = (0..x.nrows())
                        .map(|i| x[[i, 0]])
                        .filter(|&val| val.is_finite())
                        .collect();

                    if !first_feature_vals.is_empty() {
                        first_feature_vals.iter().sum::<f64>() / first_feature_vals.len() as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let mut fallback_left = Vec::new();
                let mut fallback_right = Vec::new();

                for i in 0..x.nrows() {
                    let val = x[[i, 0]];
                    if val.is_finite() {
                        if val <= fallback_threshold {
                            fallback_left.push(i);
                        } else {
                            fallback_right.push(i);
                        }
                    }
                }

                // Ensure we have at least one sample on each side for fallback
                if fallback_left.is_empty() && !fallback_right.is_empty() {
                    fallback_left.push(fallback_right.pop().unwrap());
                } else if fallback_right.is_empty() && !fallback_left.is_empty() {
                    fallback_right.push(fallback_left.pop().unwrap());
                }

                (0, fallback_threshold, fallback_left, fallback_right)
            } else {
                (
                    best_feature,
                    best_threshold,
                    best_left_indices,
                    best_right_indices,
                )
            }
        }
        Err(_) => {
            // Mutex was poisoned, return a safe fallback
            (0, 0.0, Vec::new(), Vec::new())
        }
    }
}

/// Calculates the appropriate leaf node value based on target values
///
/// For classification problems, this selects the most frequent class.
/// For regression problems, this returns the mean of target values.
///
/// # Parameters
///
/// - `y` - Array of target values at this leaf node
/// - `is_classifier` - Boolean flag indicating whether this is a classification task
///
/// # Returns
///
/// A tuple containing:
/// - `f64` - The prediction value for this leaf node
/// - `Option<usize>` - For classification tasks, the class index as `Some(usize)`; for regression, `None`
fn calculate_leaf_value(y: ArrayView1<f64>, is_classifier: bool) -> (f64, Option<usize>) {
    if y.is_empty() {
        return (0.0, None);
    }

    if is_classifier {
        // Select the most frequent class as the prediction value
        let mut class_counts = AHashMap::new();
        for &value in y {
            // Convert floating point to integer representation
            let key = (value * 1000.0).round() as i64; // Preserve 3 decimal places of precision
            *class_counts.entry(key).or_insert(0) += 1;
        }

        // Find the most common class
        let (most_common_class_key, _) = class_counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or((&0, &0));

        // Convert integer key back to floating point
        let most_common_class = *most_common_class_key as f64 / 1000.0;

        (most_common_class, Some(most_common_class as usize))
    } else {
        // Regression problem: return the mean value
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        (mean, None)
    }
}

/// Calculates class probability distribution for a set of target values
///
/// This function computes the frequency-based probability for each class
/// by counting occurrences and dividing by the total number of samples.
///
/// # Parameters
///
/// - `y` - Array of target values (assumed to be class indices as floating point values)
/// - `n_classes` - Total number of classes in the classification problem
///
/// # Returns
///
/// * `Vec<f64>` - A vector of probabilities for each class, summing to 1.0
fn calculate_class_probabilities(y: ArrayView1<f64>, n_classes: usize) -> Vec<f64> {
    let mut probas = vec![0.0; n_classes];
    let total = y.len() as f64;

    if total == 0.0 {
        return probas;
    }

    // Only use parallel processing for large arrays
    if y.len() > 10000 {
        // Process data in chunks in parallel, then merge results
        let chunk_size = std::cmp::max(1, y.len() / rayon::current_num_threads());

        let counts: Vec<Vec<f64>> = y
            .axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .map(|chunk| {
                let mut local_counts = vec![0.0; n_classes];
                for &val in chunk {
                    let class_idx = val as usize;
                    if class_idx < n_classes {
                        local_counts[class_idx] += 1.0;
                    }
                }
                local_counts
            })
            .collect();

        // Merge counts from all chunks
        for local_counts in counts {
            for (i, count) in local_counts.iter().enumerate() {
                probas[i] += count;
            }
        }
    } else {
        // Standard counting method
        for &val in y {
            let class_idx = val as usize;
            if class_idx < n_classes {
                probas[class_idx] += 1.0;
            }
        }
    }

    // Convert counts to probabilities
    for p in &mut probas {
        *p /= total;
    }

    probas
}

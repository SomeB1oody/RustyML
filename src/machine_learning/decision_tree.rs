use super::validation::{check_is_fitted, preliminary_check, validate_predict_input};
use crate::error::ModelError;
use crate::math::{entropy, gini, variance};
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "show_progress")]
use indicatif::ProgressBar;

/// Minimum number of samples required to enable parallel processing in decision tree operations.
/// When the number of samples is below this threshold, sequential processing is used instead
/// to avoid parallelization overhead.
const DECISION_TREE_PARALLEL_THRESHOLD: usize = 1000;

/// The best split found for a node while growing the tree.
enum Split {
    /// Binary numeric split: samples with `feature <= threshold` go left, the rest right.
    Numeric {
        feature: usize,
        threshold: f64,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    /// Multi-way categorical split: one partition (and child branch) per distinct value.
    Categorical {
        feature: usize,
        partitions: Vec<(String, Vec<usize>)>,
    },
}

/// Canonical string key for a categorical feature value.
///
/// Rounds to 6 decimal places so values that are equal in practice map to the same
/// branch despite minor floating-point noise. The same key is used both to group
/// samples during training and to route samples in [`Node`]'s `children` at predict time.
fn category_key(value: f64) -> String {
    format!("{}", (value * 1e6).round() / 1e6)
}

/// Split information (intrinsic value) of a partition, used for the C4.5 gain ratio:
/// `-Σ (cᵢ/total) · log2(cᵢ/total)`.
fn split_information(counts: &[f64], total: f64) -> f64 {
    let mut info = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            info -= p * p.log2();
        }
    }
    info
}

/// Decision tree algorithm types.
///
/// Represents different splitting criteria and impurity measures used in decision tree construction.
///
/// # Variants
///
/// - `ID3` - Iterative Dichotomiser 3, uses information gain (entropy) for splitting. Only suitable for classification tasks.
/// - `C45` - Successor to ID3, uses information gain ratio to handle varied attribute value ranges. Only suitable for classification tasks.
/// - `CART` - Classification and Regression Trees, uses Gini impurity for classification and MSE for regression. Supports both classification and regression.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Algorithm {
    ID3,
    C45,
    CART,
}

impl Algorithm {
    /// Whether this algorithm can be used for regression tasks.
    ///
    /// Only CART supports regression; ID3 and C4.5 are classification-only.
    fn supports_regression(&self) -> bool {
        matches!(self, Algorithm::CART)
    }

    /// Whether this algorithm splits categorical features multi-way (one branch per value).
    ///
    /// ID3 and C4.5 do; CART is always binary.
    fn allows_multiway_categorical(&self) -> bool {
        matches!(self, Algorithm::ID3 | Algorithm::C45)
    }

    /// Impurity measure for a classification subset under this algorithm.
    ///
    /// CART uses Gini impurity; ID3 and C4.5 use entropy. (Regression always uses MSE,
    /// which is independent of the algorithm and handled by the caller.)
    fn classification_impurity(&self, y: &ArrayView1<f64>) -> f64 {
        match self {
            Algorithm::CART => gini(y),
            Algorithm::ID3 | Algorithm::C45 => entropy(y),
        }
    }

    /// Selection score for a candidate split, given its impurity decrease and the sample
    /// counts of its partitions.
    ///
    /// C4.5 normalizes the gain by split information (gain ratio) to curb the bias toward
    /// features with many distinct values; ID3 and CART use the raw impurity decrease.
    /// Returns `None` when C4.5's split information is degenerate (≈ 0), leaving the
    /// caller to skip the split.
    fn selection_score(&self, impurity_decrease: f64, counts: &[f64], total: f64) -> Option<f64> {
        match self {
            Algorithm::C45 => {
                let split_info = split_information(counts, total);
                (split_info > f64::EPSILON).then(|| impurity_decrease / split_info)
            }
            Algorithm::ID3 | Algorithm::CART => Some(impurity_decrease),
        }
    }
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
#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct DecisionTreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_impurity_decrease: f64,
    pub random_state: Option<u64>,
}

impl Default for DecisionTreeParams {
    /// Default hyperparameters for decision tree.
    ///
    /// Provides sensible defaults for a standard decision tree.
    ///
    /// # Default Values
    ///
    /// - `max_depth` - `None` (no depth limit)
    /// - `min_samples_split` - `2`
    /// - `min_samples_leaf` - `1`
    /// - `min_impurity_decrease` - `0.0`
    /// - `random_state` - `None`
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
///   - `categories`: For categorical splits, the list of category-value keys (one per branch).
/// - `Leaf` - A terminal node that produces a prediction.
///   - `value`: The predicted value (class label for classification, continuous value for regression).
///   - `class`: For classification, the majority class index.
///   - `probabilities`: For classification, probability distribution over all classes.
#[derive(Debug, Clone, Deserialize, Serialize)]
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
/// - `left` - For binary splits, the left child (feature value ≤ threshold); for categorical splits, the fallback child used for category values unseen during training.
/// - `right` - For binary splits, the right child node (samples with feature value > threshold).
/// - `children` - For categorical splits, a map from category-value keys to child nodes.
#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// - `Node` - A new `Node` configured as a leaf.
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
    /// - `Node` - A new `Node` configured as an internal decision node.
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

    /// Creates a new internal node for a multi-way categorical split.
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
/// - `categorical_features` - Indices of feature columns treated as categorical (multi-way splits for ID3/C4.5).
///
/// # Example
/// ```rust
/// use rustyml::machine_learning::decision_tree::{DecisionTree, Algorithm, DecisionTreeParams};
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
/// let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
/// tree.fit(&x_train, &y_train).unwrap();
///
/// // Make predictions
/// let x_test = array![[5.0, 3.2, 1.2, 0.2], [6.5, 3.0, 5.2, 2.0]];
/// let predictions = tree.predict(&x_test).unwrap();
///
/// // Get probability estimates for classification
/// let probabilities = tree.predict_proba(&x_test).unwrap();
///
/// // Categorical features: mark which columns hold discrete category codes.
/// // ID3 and C4.5 then split them multi-way (one branch per distinct value),
/// // which can separate classes a single binary threshold cannot.
/// let x_cat = array![[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]];
/// let y_cat = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // value 1 -> class 1, values 0/2 -> class 0
///
/// let mut cat_tree = DecisionTree::new(Algorithm::C45, true, None).unwrap();
/// cat_tree.set_categorical_features(vec![0]); // treat column 0 as categorical
/// cat_tree.fit(&x_cat, &y_cat).unwrap();
/// let cat_predictions = cat_tree.predict(&x_cat).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    algorithm: Algorithm,
    root: Option<Box<Node>>,
    n_features: usize,
    n_classes: Option<usize>,
    params: DecisionTreeParams,
    is_classifier: bool,
    categorical_features: Vec<usize>,
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
    /// - `Result<Self, ModelError>` - A new untrained `DecisionTree` instance, or a `ModelError` if validation fails.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the algorithm is incompatible with the task type, or if hyperparameters are invalid.
    pub fn new(
        algorithm: Algorithm,
        is_classifier: bool,
        params: Option<DecisionTreeParams>,
    ) -> Result<Self, ModelError> {
        // Validate algorithm compatibility with task type
        if !is_classifier && !algorithm.supports_regression() {
            return Err(ModelError::InputValidationError(
                "Only CART algorithm is supported for regression tasks".to_string(),
            ));
        }

        let params = params.unwrap_or_default();

        // Validate hyperparameters
        if params.min_samples_split < 2 {
            return Err(ModelError::InputValidationError(
                "min_samples_split must be at least 2".to_string(),
            ));
        }

        if params.min_samples_leaf < 1 {
            return Err(ModelError::InputValidationError(
                "min_samples_leaf must be at least 1".to_string(),
            ));
        }

        if params.min_samples_leaf > params.min_samples_split {
            return Err(ModelError::InputValidationError(format!(
                "min_samples_leaf ({}) cannot be greater than min_samples_split ({})",
                params.min_samples_leaf, params.min_samples_split
            )));
        }

        if params.min_impurity_decrease < 0.0 || !params.min_impurity_decrease.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "min_impurity_decrease must be non-negative and finite, got {}",
                params.min_impurity_decrease
            )));
        }

        Ok(Self {
            algorithm,
            root: None,
            n_features: 0,
            n_classes: None,
            params,
            is_classifier,
            categorical_features: Vec::new(),
        })
    }

    // Getters
    get_field!(get_algorithm, algorithm, Algorithm);
    get_field!(get_n_features, n_features, usize);
    get_field!(get_n_classes, n_classes, Option<usize>);
    get_field!(get_parameters, params, DecisionTreeParams);
    /// Gets the root node of the trained tree.
    ///
    /// # Returns
    ///
    /// * `Option<&Node>` - The root node, or `None` if the model has not been fitted
    pub fn get_root(&self) -> Option<&Node> {
        self.root.as_deref()
    }
    get_field!(get_is_classifier, is_classifier, bool);

    /// Designates which feature columns are categorical.
    ///
    /// Categorical features are split multi-way (one branch per distinct value) by the
    /// ID3 and C4.5 algorithms. CART always uses binary splits, so it ignores this
    /// designation. This must be set before calling [`fit`](Self::fit).
    ///
    /// # Parameters
    ///
    /// - `features` - Indices of the feature columns to treat as categorical
    ///
    /// # Returns
    ///
    /// - `&mut Self` - A mutable reference to `self` for method chaining
    pub fn set_categorical_features(&mut self, features: Vec<usize>) -> &mut Self {
        self.categorical_features = features;
        self
    }

    /// Returns the indices of the feature columns treated as categorical.
    ///
    /// # Returns
    ///
    /// - `&[usize]` - The configured categorical feature indices (empty if none)
    pub fn get_categorical_features(&self) -> &[usize] {
        &self.categorical_features
    }

    /// Trains the decision tree on the provided training data.
    ///
    /// Builds the tree structure by recursively finding the best splits.
    ///
    /// # Parameters
    ///
    /// - `x` - Training features as a 2D array with shape (n_samples, n_features).
    /// - `y` - Training labels as a 1D array.
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A mutable reference to `self` for method chaining.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the input data is invalid or doesn't meet minimum requirements.
    /// - `ModelError::ProcessingError` - If class information cannot be determined.
    ///
    /// # Performance
    ///
    /// Uses parallel processing for evaluating potential splits when the number of samples exceeds `DECISION_TREE_PARALLEL_THRESHOLD`.
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Use preliminary_check for input validation
        preliminary_check(x, Some(y))?;

        // Check minimum samples requirement for tree construction
        if x.nrows() < self.params.min_samples_split {
            return Err(ModelError::InputValidationError(format!(
                "Number of samples ({}) is less than min_samples_split ({})",
                x.nrows(),
                self.params.min_samples_split
            )));
        }

        // Check that there are enough features
        if x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data must have at least one feature".to_string(),
            ));
        }

        self.n_features = x.ncols();

        // For classification, determine number of classes and validate labels
        if self.is_classifier {
            let max_class = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).ok_or(
                ModelError::ProcessingError("Cannot determine max class".to_string()),
            )?;

            // Validate that all labels are non-negative integers
            for &label in y.iter() {
                if label < 0.0 || label.fract() != 0.0 {
                    return Err(ModelError::InputValidationError(
                        "Class labels must be non-negative integers starting from 0".to_string(),
                    ));
                }
            }

            self.n_classes = Some((*max_class as usize) + 1);
        }

        // Estimate maximum possible nodes for progress bar
        // For a balanced binary tree with depth d, max nodes = 2^(d+1) - 1
        #[cfg(feature = "show_progress")]
        let estimated_max_depth = self.params.max_depth.unwrap_or(20).min(20);
        #[cfg(feature = "show_progress")]
        let estimated_nodes = (1 << (estimated_max_depth + 1)) - 1;

        // Create progress bar for tree building
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                estimated_nodes as u64,
                "[{elapsed_precise}] {bar:40} {pos} nodes | Depth: {msg}",
            );
            pb.set_message("0");
            pb
        };

        // Build the tree
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(Box::new(self.build_tree(
            x,
            y,
            &indices,
            0,
            #[cfg(feature = "show_progress")]
            &progress_bar,
        )?));

        // Finish progress bar
        #[cfg(feature = "show_progress")]
        progress_bar
            .finish_with_message(format!("{}", self.count_nodes(self.root.as_ref().unwrap())));

        Ok(self)
    }

    /// Recursively builds a decision tree node by finding optimal splits with progress tracking.
    fn build_tree<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
        depth: usize,
        #[cfg(feature = "show_progress")] progress_bar: &ProgressBar,
    ) -> Result<Node, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        #[cfg(feature = "show_progress")]
        progress_bar.inc(1);
        #[cfg(feature = "show_progress")]
        progress_bar.set_message(format!("{}", depth));

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

        let Some((split, impurity_decrease)) = split_result else {
            // No valid split found
            return Ok(self.create_leaf(y, indices));
        };

        // Check if split meets minimum impurity decrease
        if impurity_decrease < self.params.min_impurity_decrease {
            return Ok(self.create_leaf(y, indices));
        }

        match split {
            Split::Numeric {
                feature,
                threshold,
                left,
                right,
            } => {
                // Check if leaf size constraint is met
                if left.len() < self.params.min_samples_leaf
                    || right.len() < self.params.min_samples_leaf
                {
                    return Ok(self.create_leaf(y, indices));
                }

                // Create internal node and recursively build children
                let mut node = Node::new_internal(feature, threshold);
                node.left = Some(Box::new(self.build_tree(
                    x,
                    y,
                    &left,
                    depth + 1,
                    #[cfg(feature = "show_progress")]
                    progress_bar,
                )?));
                node.right = Some(Box::new(self.build_tree(
                    x,
                    y,
                    &right,
                    depth + 1,
                    #[cfg(feature = "show_progress")]
                    progress_bar,
                )?));

                Ok(node)
            }
            Split::Categorical {
                feature,
                partitions,
            } => {
                // Every branch must satisfy the leaf size constraint
                if partitions
                    .iter()
                    .any(|(_, idx)| idx.len() < self.params.min_samples_leaf)
                {
                    return Ok(self.create_leaf(y, indices));
                }

                let keys: Vec<String> = partitions.iter().map(|(key, _)| key.clone()).collect();
                let mut node = Node::new_categorical(feature, keys);

                // Fallback leaf for categories not seen during training (used at predict time)
                node.left = Some(Box::new(self.create_leaf(y, indices)));

                let children = node.children.as_mut().unwrap();
                for (key, idx) in partitions {
                    let child = self.build_tree(
                        x,
                        y,
                        &idx,
                        depth + 1,
                        #[cfg(feature = "show_progress")]
                        progress_bar,
                    )?;
                    children.insert(key, Box::new(child));
                }

                Ok(node)
            }
        }
    }

    /// Finds the best split for the given samples across all features.
    ///
    /// Numeric features are evaluated as binary threshold splits. Features marked as
    /// categorical are evaluated as multi-way splits for ID3 and C4.5 (CART is always
    /// binary). Selection uses information gain (ID3/CART) or gain ratio (C4.5); the
    /// returned `impurity_decrease` drives the `min_impurity_decrease` stopping rule.
    /// Uses parallel evaluation when the number of samples exceeds DECISION_TREE_PARALLEL_THRESHOLD.
    fn find_best_split<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
    ) -> Result<Option<(Split, f64)>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Calculate parent impurity once
        let parent_impurity = self.calculate_impurity(y, indices);

        // ID3 and C4.5 support multi-way categorical splits; CART is always binary.
        let allow_categorical = self.algorithm.allows_multiway_categorical();

        // For each feature, returns (selection_score, split, impurity_decrease).
        let process_feature = |feature_idx: usize| -> Option<(f64, Split, f64)> {
            if allow_categorical && self.categorical_features.contains(&feature_idx) {
                self.evaluate_categorical_split(x, y, indices, feature_idx, parent_impurity)
            } else {
                self.evaluate_numeric_split(x, y, indices, feature_idx, parent_impurity)
            }
        };

        // Use parallel evaluation only when sample size exceeds threshold
        let best = if indices.len() >= DECISION_TREE_PARALLEL_THRESHOLD {
            (0..self.n_features)
                .into_par_iter()
                .filter_map(process_feature)
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        } else {
            (0..self.n_features)
                .filter_map(process_feature)
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        };

        Ok(best.map(|(_score, split, impurity_decrease)| (split, impurity_decrease)))
    }

    /// Evaluates the best binary numeric split for a single feature.
    ///
    /// Returns `(selection_score, split, impurity_decrease)`, or `None` if no
    /// impurity-reducing threshold exists.
    fn evaluate_numeric_split<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
        feature_idx: usize,
        parent_impurity: f64,
    ) -> Option<(f64, Split, f64)>
    where
        S: Data<Elem = f64>,
    {
        // Unique sorted values give the candidate thresholds (midpoints)
        let mut feature_values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_values.dedup();

        let n = indices.len() as f64;
        let mut best_score = 0.0;
        let mut best: Option<(Split, f64)> = None;

        for i in 0..feature_values.len().saturating_sub(1) {
            let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

            let (left, right): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&idx| x[[idx, feature_idx]] <= threshold);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let n_left = left.len() as f64;
            let n_right = right.len() as f64;
            let weighted = (n_left / n) * self.calculate_impurity(y, &left)
                + (n_right / n) * self.calculate_impurity(y, &right);
            let impurity_decrease = parent_impurity - weighted;

            // C4.5 normalizes the gain by split information (gain ratio); ID3 and CART use
            // the raw impurity decrease. A degenerate split-info score skips this threshold.
            let Some(score) =
                self.algorithm
                    .selection_score(impurity_decrease, &[n_left, n_right], n)
            else {
                continue;
            };

            if score > best_score {
                best_score = score;
                best = Some((
                    Split::Numeric {
                        feature: feature_idx,
                        threshold,
                        left,
                        right,
                    },
                    impurity_decrease,
                ));
            }
        }

        best.map(|(split, decrease)| (best_score, split, decrease))
    }

    /// Evaluates a multi-way categorical split for a single feature (one branch per value).
    ///
    /// Returns `(selection_score, split, impurity_decrease)`, or `None` if the feature
    /// has fewer than two distinct values or the split does not reduce impurity.
    fn evaluate_categorical_split<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
        feature_idx: usize,
        parent_impurity: f64,
    ) -> Option<(f64, Split, f64)>
    where
        S: Data<Elem = f64>,
    {
        // Partition samples by distinct category value
        let mut groups: AHashMap<String, Vec<usize>> = AHashMap::new();
        for &idx in indices {
            groups
                .entry(category_key(x[[idx, feature_idx]]))
                .or_default()
                .push(idx);
        }

        // A useful split needs at least two distinct values
        if groups.len() < 2 {
            return None;
        }

        let n = indices.len() as f64;
        let mut weighted = 0.0;
        let mut counts: Vec<f64> = Vec::with_capacity(groups.len());
        let mut partitions: Vec<(String, Vec<usize>)> = Vec::with_capacity(groups.len());
        for (key, group) in groups {
            let n_group = group.len() as f64;
            counts.push(n_group);
            weighted += (n_group / n) * self.calculate_impurity(y, &group);
            partitions.push((key, group));
        }

        let impurity_decrease = parent_impurity - weighted;
        if impurity_decrease <= 0.0 {
            return None;
        }

        // C4.5 uses gain ratio (impurity decrease / split information); ID3 uses the raw
        // decrease. A degenerate split-info score rejects this feature entirely.
        let score = self
            .algorithm
            .selection_score(impurity_decrease, &counts, n)?;

        Some((
            score,
            Split::Categorical {
                feature: feature_idx,
                partitions,
            },
            impurity_decrease,
        ))
    }

    /// Calculates the impurity measure for the given samples based on the algorithm type.
    fn calculate_impurity<S>(&self, y: &ArrayBase<S, Ix1>, indices: &[usize]) -> f64
    where
        S: Data<Elem = f64>,
    {
        if indices.is_empty() {
            return 0.0;
        }

        if self.is_classifier {
            let subset = y.select(Axis(0), indices);
            self.algorithm.classification_impurity(&subset.view())
        } else {
            self.calculate_mse(y, indices)
        }
    }

    /// Calculates mean squared error for regression samples.
    fn calculate_mse<S>(&self, y: &ArrayBase<S, Ix1>, indices: &[usize]) -> f64
    where
        S: Data<Elem = f64>,
    {
        // Collect subset values into an Array1 to use math::variance
        let subset: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        variance(&subset)
    }

    /// Checks if all samples in the given indices have the same label (pure node).
    fn is_pure<S>(&self, y: &ArrayBase<S, Ix1>, indices: &[usize]) -> bool
    where
        S: Data<Elem = f64>,
    {
        if indices.is_empty() {
            return true;
        }

        let first_value = y[indices[0]];
        indices.iter().all(|&i| (y[i] - first_value).abs() < 1e-10)
    }

    /// Creates a leaf node with the appropriate prediction value based on the task type.
    fn create_leaf<S>(&self, y: &ArrayBase<S, Ix1>, indices: &[usize]) -> Node
    where
        S: Data<Elem = f64>,
    {
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
    /// - `x` - Feature vector for a single sample as a slice of length `n_features`.
    ///
    /// # Returns
    ///
    /// - `Result<f64, ModelError>` - The predicted value, or a `ModelError` if prediction fails.
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet.
    /// - `ModelError::TreeError` - If the feature dimension mismatches or tree structure is broken.
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
                    // Multi-way categorical split: route by the sample's category value,
                    // falling back to the default leaf for categories unseen during training.
                    let key = category_key(x[*feature_index]);
                    if let Some(child) = node.children.as_ref().and_then(|c| c.get(&key)) {
                        return self.traverse_tree(child, x);
                    }
                    return match &node.left {
                        Some(fallback) => self.traverse_tree(fallback, x),
                        None => Err(ModelError::TreeError(
                            "Categorical node has no matching child and no fallback",
                        )),
                    };
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

    /// Predicts outputs for multiple samples.
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix as a 2D array.
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - A 1D array of predicted values.
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet.
    /// - `ModelError::InputValidationError` - If input data is empty or dimensions mismatch.
    ///
    /// # Performance
    ///
    /// Uses parallel processing for predictions when the number of samples exceeds `DECISION_TREE_PARALLEL_THRESHOLD`.
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        check_is_fitted(self.root.is_some())?;
        validate_predict_input(x, self.n_features)?;

        // Use parallel processing only when sample size exceeds threshold
        let predictions: Result<Vec<f64>, ModelError> =
            if x.nrows() >= DECISION_TREE_PARALLEL_THRESHOLD {
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| {
                        let row_slice = row.to_vec();
                        self.predict_one(&row_slice)
                    })
                    .collect()
            } else {
                x.axis_iter(Axis(0))
                    .map(|row| {
                        let row_slice = row.to_vec();
                        self.predict_one(&row_slice)
                    })
                    .collect()
            };

        Ok(Array1::from_vec(predictions?))
    }

    /// Trains the tree on training data and immediately makes predictions.
    ///
    /// # Parameters
    ///
    /// - `x_train` - Training features as a 2D array.
    /// - `y_train` - Training labels as a 1D array.
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, ModelError>` - A 1D array of predictions for the training data.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If training or prediction inputs are invalid.
    pub fn fit_predict<S>(
        &mut self,
        x_train: &ArrayBase<S, Ix2>,
        y_train: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x_train, y_train)?;
        self.predict(x_train)
    }

    /// Predicts class probabilities for multiple samples (classification only).
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix as a 2D array.
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - A 2D array of class probabilities where each row sums to 1.0.
    ///
    /// # Errors
    ///
    /// - `ModelError::TreeError` - If called on a regression tree.
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet.
    ///
    /// # Performance
    ///
    /// Uses parallel processing for predictions when the number of samples exceeds `DECISION_TREE_PARALLEL_THRESHOLD`.
    pub fn predict_proba<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        if !self.is_classifier {
            return Err(ModelError::TreeError(
                "predict_proba is only available for classification",
            ));
        }

        check_is_fitted(self.root.is_some())?;
        validate_predict_input(x, self.n_features)?;

        let n_classes = self.n_classes.unwrap();

        // Use parallel processing only when sample size exceeds threshold
        let probabilities: Result<Vec<Vec<f64>>, ModelError> =
            if x.nrows() >= DECISION_TREE_PARALLEL_THRESHOLD {
                x.axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| {
                        let row_slice = row.to_vec();
                        self.predict_proba_one(&row_slice)
                    })
                    .collect()
            } else {
                x.axis_iter(Axis(0))
                    .map(|row| {
                        let row_slice = row.to_vec();
                        self.predict_proba_one(&row_slice)
                    })
                    .collect()
            };

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
    /// - `x` - Feature vector for a single sample as a slice.
    ///
    /// # Returns
    ///
    /// - `Result<Vec<f64>, ModelError>` - A vector of class probabilities.
    ///
    /// # Errors
    ///
    /// - `ModelError::TreeError` - If called on a regression tree or dimension mismatches.
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet.
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
                .as_ref()
                .cloned()
                .ok_or(ModelError::TreeError("No probabilities in leaf node")),
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    // Multi-way categorical split: route by category value, falling back
                    // to the default leaf for categories unseen during training.
                    let key = category_key(x[*feature_index]);
                    if let Some(child) = node.children.as_ref().and_then(|c| c.get(&key)) {
                        return self.get_probabilities(child, x);
                    }
                    return match &node.left {
                        Some(fallback) => self.get_probabilities(fallback, x),
                        None => Err(ModelError::TreeError(
                            "Categorical node has no matching child and no fallback",
                        )),
                    };
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
    /// # Returns
    ///
    /// - `Result<String, ModelError>` - A formatted string containing the tree structure.
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model hasn't been trained yet.
    pub fn generate_tree_structure(&self) -> Result<String, ModelError> {
        if self.root.is_none() {
            return Err(ModelError::NotFitted);
        }

        let mut output = String::new();
        output.push_str("Decision Tree Structure:\n");
        self.print_node(self.root.as_ref().unwrap(), &mut output, "", true);
        Ok(output)
    }

    /// Counts the total number of nodes in the tree.
    #[cfg(feature = "show_progress")]
    fn count_nodes(&self, node: &Node) -> usize {
        let mut count = 1; // Count current node
        match &node.node_type {
            NodeType::Leaf { .. } => count,
            NodeType::Internal { .. } => {
                if let Some(ref left) = node.left {
                    count += self.count_nodes(left);
                }
                if let Some(ref right) = node.right {
                    count += self.count_nodes(right);
                }
                if let Some(ref children) = node.children {
                    for child in children.values() {
                        count += self.count_nodes(child);
                    }
                }
                count
            }
        }
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
                let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

                if categories.is_some() {
                    output.push_str(&format!(
                        "Split: feature[{}] (categorical)\n",
                        feature_index
                    ));

                    // One branch per category value (sorted by key for stable output),
                    // plus a default branch for values unseen during training.
                    let mut branches: Vec<(String, &Node)> = node
                        .children
                        .as_ref()
                        .map(|c| c.iter().map(|(k, v)| (k.clone(), v.as_ref())).collect())
                        .unwrap_or_default();
                    branches.sort_by(|a, b| a.0.cmp(&b.0));

                    let total = branches.len() + node.left.is_some() as usize;
                    for (i, (key, child)) in branches.into_iter().enumerate() {
                        let last = i + 1 == total;
                        let connector = if last { "└── " } else { "├── " };
                        output.push_str(&format!("{}{}= {}:\n", new_prefix, connector, key));
                        let child_prefix =
                            format!("{}{}", new_prefix, if last { "    " } else { "│   " });
                        self.print_node(child, output, &child_prefix, true);
                    }

                    if let Some(ref fallback) = node.left {
                        output.push_str(&format!("{}└── = (default):\n", new_prefix));
                        let child_prefix = format!("{}    ", new_prefix);
                        self.print_node(fallback, output, &child_prefix, true);
                    }
                } else {
                    output.push_str(&format!(
                        "Split: feature[{}] <= {:.4}\n",
                        feature_index, threshold
                    ));

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

    model_save_and_load_methods!(DecisionTree);
}

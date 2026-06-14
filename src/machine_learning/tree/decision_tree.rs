//! Decision tree models for classification and regression
//!
//! Provides the [`DecisionTree`] estimator backed by the ID3, C4.5, and CART
//! algorithms, along with its node types ([`Node`], [`NodeType`]), the
//! [`Algorithm`] selector, and the [`DecisionTreeParams`] hyperparameters

use crate::error::{Error, TreeError};
use crate::machine_learning::validation::{
    check_is_fitted, preliminary_check, validate_predict_input,
};
use crate::math::{entropy, gini, variance};
use crate::parallel_gates::{SORT_SCAN_MIN_ELEMS, TREE_TRAVERSAL_MIN_VISITS};
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};
use ndarray_rand::rand::{Rng, rngs::StdRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "show_progress")]
use indicatif::ProgressBar;

/// Assumed average root-to-leaf walk length for the prediction parallel gate
///
/// The fitted tree does not record its depth, so the traversal-work estimate
/// (`samples x walk length`) uses this fixed stand-in. A grown CART tree on real data
/// typically walks 10-30 nodes per sample
const DECISION_TREE_ASSUMED_DEPTH: usize = 16;

/// The best split found for a node while growing the tree
enum Split {
    /// Binary numeric split: samples with `feature <= threshold` go left, the rest right
    Numeric {
        feature: usize,
        threshold: f64,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    /// Multi-way categorical split: one partition (and child branch) per distinct value
    Categorical {
        feature: usize,
        partitions: Vec<(String, Vec<usize>)>,
    },
}

/// Canonical string key for a categorical feature value
///
/// Rounds to 6 decimal places so values that are equal in practice map to the same
/// branch despite minor floating-point noise. The same key groups samples during
/// training and routes samples in [`Node`]'s `children` at predict time
fn category_key(value: f64) -> String {
    format!("{}", (value * 1e6).round() / 1e6)
}

/// Split information (intrinsic value) of a partition, used for the C4.5 gain ratio:
/// `-sum (c_i/total) * log2(c_i/total)`
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

/// Decision tree algorithm types
///
/// The splitting criteria and impurity measures used in decision tree construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Algorithm {
    /// Iterative Dichotomiser 3: information gain (entropy) splitting, classification only
    ID3,
    /// Successor to ID3: information gain ratio to handle varied attribute value ranges, classification only
    C45,
    /// Classification and Regression Trees: Gini impurity for classification, MSE for regression
    CART,
}

impl Algorithm {
    /// Whether this algorithm can be used for regression tasks
    ///
    /// Only CART supports regression; ID3 and C4.5 are classification-only
    fn supports_regression(&self) -> bool {
        matches!(self, Algorithm::CART)
    }

    /// Whether this algorithm splits categorical features multi-way (one branch per value)
    ///
    /// ID3 and C4.5 do; CART is always binary
    fn allows_multiway_categorical(&self) -> bool {
        matches!(self, Algorithm::ID3 | Algorithm::C45)
    }

    /// Impurity measure for a classification subset under this algorithm
    ///
    /// CART uses Gini impurity; ID3 and C4.5 use entropy. Regression always uses MSE,
    /// which is independent of the algorithm and handled by the caller
    fn classification_impurity(&self, y: &ArrayView1<f64>) -> f64 {
        match self {
            Algorithm::CART => gini(y),
            Algorithm::ID3 | Algorithm::C45 => entropy(y),
        }
    }

    /// Classification impurity computed directly from per-class counts, where `n_side` is the
    /// number of samples on this side of a split
    ///
    /// Equivalent to [`classification_impurity`] (Gini for CART, entropy for ID3/C4.5) but
    /// driven by running counts, letting the numeric split search update impurity
    /// incrementally instead of re-scanning the label subset for every candidate threshold
    fn impurity_from_counts(&self, counts: &[f64], n_side: f64) -> f64 {
        match self {
            Algorithm::CART => {
                // Gini: 1 - sum (c/n)^2
                let mut sum_sq = 0.0;
                for &c in counts {
                    if c > 0.0 {
                        let p = c / n_side;
                        sum_sq += p * p;
                    }
                }
                1.0 - sum_sq
            }
            Algorithm::ID3 | Algorithm::C45 => {
                // Entropy: -sum (c/n) log2(c/n)
                let mut ent = 0.0;
                for &c in counts {
                    if c > 0.0 {
                        let p = c / n_side;
                        ent -= p * p.log2();
                    }
                }
                ent
            }
        }
    }

    /// Selection score for a candidate split, given its impurity decrease and the sample
    /// counts of its partitions
    ///
    /// C4.5 normalizes the gain by split information (gain ratio) to curb the bias toward
    /// features with many distinct values; ID3 and CART use the raw impurity decrease. It
    /// returns `None` when C4.5's split information is degenerate (~= 0), leaving the
    /// caller to skip the split
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

/// Hyperparameters controlling decision tree growth and complexity
///
/// These parameters help prevent overfitting and shape the tree structure during training
#[derive(Debug, Copy, Clone, PartialEq, Deserialize, Serialize)]
pub struct DecisionTreeParams {
    /// Maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure or
    /// contain fewer than `min_samples_split` samples
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node. Must be at least 2
    pub min_samples_split: usize,
    /// Minimum number of samples required at a leaf node. Splits that produce leaves with
    /// fewer samples are rejected
    pub min_samples_leaf: usize,
    /// Minimum impurity decrease required for a split. A node is split when the impurity
    /// decrease is greater than or equal to this value
    pub min_impurity_decrease: f64,
    /// Seed for breaking ties between equally-scoring splits
    pub random_state: Option<u64>,
}

impl Default for DecisionTreeParams {
    /// Default hyperparameters for a decision tree
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

/// Type of a node in the decision tree
///
/// Distinguishes internal decision nodes from leaf nodes that produce predictions
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum NodeType {
    /// A decision node that splits data based on a feature
    Internal {
        /// Index of the feature used for splitting
        feature_index: usize,
        /// Threshold for binary splits (samples with feature value <= threshold go left)
        threshold: f64,
        /// For categorical splits, the list of category-value keys (one per branch)
        categories: Option<Vec<String>>,
    },
    /// A terminal node that produces a prediction
    Leaf {
        /// The predicted value (class label for classification, continuous value for regression)
        value: f64,
        /// For classification, the majority class index
        class: Option<usize>,
        /// For classification, the probability distribution over all classes
        probabilities: Option<Vec<f64>>,
    },
}

/// A node in the decision tree structure
///
/// Either an internal decision node or a leaf node, with connections to child nodes
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Node {
    /// The type of this node (Internal or Leaf), containing node-specific data
    pub node_type: NodeType,
    /// For binary splits, the left child (feature value <= threshold); for categorical splits,
    /// the fallback child used for category values unseen during training
    pub left: Option<Box<Node>>,
    /// For binary splits, the right child (samples with feature value > threshold)
    pub right: Option<Box<Node>>,
    /// For categorical splits, a map from category-value keys to child nodes
    pub children: Option<AHashMap<String, Box<Node>>>,
}

impl Node {
    /// Creates a new leaf node with prediction values
    ///
    /// # Parameters
    ///
    /// - `value` - The predicted value (class label for classification, continuous value for regression)
    /// - `class` - For classification, the majority class index
    /// - `probabilities` - For classification, the probability distribution over all classes
    ///
    /// # Returns
    ///
    /// - `Node` - A new `Node` configured as a leaf
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

    /// Creates a new internal node for binary splitting
    ///
    /// # Parameters
    ///
    /// - `feature_index` - Index of the feature used for splitting
    /// - `threshold` - Threshold value (samples with feature value <= threshold go to the left child)
    ///
    /// # Returns
    ///
    /// - `Node` - A new `Node` configured as an internal decision node
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

    /// Creates a new internal node for a multi-way categorical split
    ///
    /// # Parameters
    ///
    /// - `feature_index` - Index of the categorical feature used for splitting
    /// - `categories` - List of possible categorical values for this feature
    ///
    /// # Returns
    ///
    /// - `Node` - A new `Node` configured for categorical splitting with an empty children map
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

/// Decision tree for classification and regression tasks
///
/// Implements the ID3, C4.5, and CART algorithms with parallel optimization using rayon,
/// and supports classification (with probability estimates) and regression. The tree is
/// built recursively by selecting the best split at each node based on impurity measures
/// (Gini, entropy, or MSE) and various stopping criteria
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::{DecisionTree, Algorithm};
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
/// let mut tree = DecisionTree::new(Algorithm::CART, true)
///     .unwrap()
///     .with_max_depth(5)
///     .with_random_state(42);
/// tree.fit(&x_train, &y_train).unwrap();
///
/// // Make predictions
/// let x_test = array![[5.0, 3.2, 1.2, 0.2], [6.5, 3.0, 5.2, 2.0]];
/// let predictions = tree.predict(&x_test).unwrap();
///
/// // Get probability estimates for classification
/// let probabilities = tree.predict_proba(&x_test).unwrap();
///
/// // Categorical features: mark which columns hold discrete category codes
/// // ID3 and C4.5 then split them multi-way (one branch per distinct value),
/// // which can separate classes a single binary threshold cannot
/// let x_cat = array![[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]];
/// let y_cat = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // value 1 -> class 1, values 0/2 -> class 0
///
/// let mut cat_tree = DecisionTree::new(Algorithm::C45, true).unwrap();
/// cat_tree.set_categorical_features(vec![0]); // treat column 0 as categorical
/// cat_tree.fit(&x_cat, &y_cat).unwrap();
/// let cat_predictions = cat_tree.predict(&x_cat).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// The splitting algorithm to use (ID3, C45, or CART)
    algorithm: Algorithm,
    /// The root node of the trained tree, or `None` if not yet fitted
    root: Option<Box<Node>>,
    /// Number of features in the training data
    n_features: usize,
    /// For classification, the number of distinct classes; `None` for regression
    n_classes: Option<usize>,
    /// Hyperparameters controlling tree growth and complexity
    params: DecisionTreeParams,
    /// Whether this tree performs classification (`true`) or regression (`false`)
    is_classifier: bool,
    /// Indices of feature columns treated as categorical (multi-way splits for ID3/C4.5)
    categorical_features: Vec<usize>,
}

impl DecisionTree {
    /// Creates a new decision tree with the specified algorithm and task type
    ///
    /// # Parameters
    ///
    /// - `algorithm` - The splitting algorithm to use (ID3, C45, or CART)
    /// - `is_classifier` - `true` for classification tasks, `false` for regression tasks
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new untrained `DecisionTree` instance, or an `Error` if validation fails
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the algorithm is incompatible with the task type (only CART supports regression)
    ///
    /// # Notes
    ///
    /// Growth and pruning hyperparameters use sensible defaults (no depth limit,
    /// `min_samples_split = 2`, `min_samples_leaf = 1`, `min_impurity_decrease = 0`, no fixed
    /// seed). Tune them with the builder methods below; the bounded ones return `Result`:
    ///
    /// - [`with_max_depth`](Self::with_max_depth) - maximum tree depth (default: unlimited)
    /// - [`with_min_samples_split`](Self::with_min_samples_split) - minimum samples to split a node (returns `Result`; must be >= 2)
    /// - [`with_min_samples_leaf`](Self::with_min_samples_leaf) - minimum samples at a leaf (returns `Result`; must be >= 1)
    /// - [`with_min_impurity_decrease`](Self::with_min_impurity_decrease) - minimum impurity decrease for a split (returns `Result`; must be >= 0)
    /// - [`with_random_state`](Self::with_random_state) - fixed seed for tie-breaking between equal splits
    ///
    /// `min_samples_leaf` must not exceed `min_samples_split`; because the two are set
    /// independently, that cross-field constraint is checked at [`fit`](Self::fit) time
    pub fn new(algorithm: Algorithm, is_classifier: bool) -> Result<Self, Error> {
        // Validate algorithm compatibility with task type
        if !is_classifier && !algorithm.supports_regression() {
            return Err(Error::invalid_input(
                "Only CART algorithm is supported for regression tasks",
            ));
        }

        Ok(Self {
            algorithm,
            root: None,
            n_features: 0,
            n_classes: None,
            params: DecisionTreeParams::default(),
            is_classifier,
            categorical_features: Vec::new(),
        })
    }

    /// Sets the maximum tree depth (default: unlimited)
    ///
    /// # Parameters
    ///
    /// - `max_depth` - maximum depth; nodes are not expanded beyond it
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.params.max_depth = Some(max_depth);
        self
    }

    /// Sets the minimum number of samples required to split an internal node (default: `2`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if `min_samples_split` is less than 2
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Result<Self, Error> {
        if min_samples_split < 2 {
            return Err(Error::invalid_parameter(
                "min_samples_split",
                "must be at least 2",
            ));
        }
        self.params.min_samples_split = min_samples_split;
        Ok(self)
    }

    /// Sets the minimum number of samples required at a leaf node (default: `1`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if `min_samples_leaf` is less than 1
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Result<Self, Error> {
        if min_samples_leaf < 1 {
            return Err(Error::invalid_parameter(
                "min_samples_leaf",
                "must be at least 1",
            ));
        }
        self.params.min_samples_leaf = min_samples_leaf;
        Ok(self)
    }

    /// Sets the minimum impurity decrease required for a split (default: `0.0`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if `min_impurity_decrease` is negative or not finite
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Result<Self, Error> {
        if min_impurity_decrease < 0.0 || !min_impurity_decrease.is_finite() {
            return Err(Error::invalid_parameter(
                "min_impurity_decrease",
                format!(
                    "must be non-negative and finite, got {}",
                    min_impurity_decrease
                ),
            ));
        }
        self.params.min_impurity_decrease = min_impurity_decrease;
        Ok(self)
    }

    /// Sets a fixed seed for breaking ties between equally-scoring splits
    /// (default: `None`, non-deterministic tie-breaking)
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.params.random_state = Some(seed);
        self
    }

    // Getters
    get_field!(get_algorithm, algorithm, Algorithm);
    get_field!(get_n_features, n_features, usize);
    get_field!(get_n_classes, n_classes, Option<usize>);
    get_field!(get_parameters, params, DecisionTreeParams);
    /// Gets the root node of the trained tree
    ///
    /// # Returns
    ///
    /// - `Option<&Node>` - The root node, or `None` if the model has not been fitted
    pub fn get_root(&self) -> Option<&Node> {
        self.root.as_deref()
    }
    get_field!(get_is_classifier, is_classifier, bool);

    /// Designates which feature columns are categorical
    ///
    /// The ID3 and C4.5 algorithms split categorical features multi-way (one branch per
    /// distinct value), while CART always uses binary splits and ignores this
    /// designation. Set this before calling [`fit`](Self::fit)
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

    /// Returns the indices of the feature columns treated as categorical
    ///
    /// # Returns
    ///
    /// - `&[usize]` - The configured categorical feature indices (empty if none)
    pub fn get_categorical_features(&self) -> &[usize] {
        &self.categorical_features
    }

    /// Trains the decision tree on the provided training data
    ///
    /// Builds the tree structure by recursively finding the best splits
    ///
    /// # Parameters
    ///
    /// - `x` - Training features as a 2D array with shape (n_samples, n_features)
    /// - `y` - Training labels as a 1D array
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - A mutable reference to `self` for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the input data is invalid or does not meet minimum requirements
    /// - `Error::Computation` - If class information cannot be determined
    ///
    /// # Performance
    ///
    /// Parallelizes the per-feature split search when the sort work clears the calibrated
    /// sort-scan gate (see `crate::parallel_gates`)
    pub fn fit<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
    ) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        preliminary_check(x, Some(y))?;

        // Cross-field constraint: the two are set independently through the builder
        if self.params.min_samples_leaf > self.params.min_samples_split {
            return Err(Error::invalid_parameter(
                "min_samples_leaf",
                format!(
                    "min_samples_leaf ({}) cannot be greater than min_samples_split ({})",
                    self.params.min_samples_leaf, self.params.min_samples_split
                ),
            ));
        }

        // Need at least min_samples_split samples to attempt a split
        if x.nrows() < self.params.min_samples_split {
            return Err(Error::invalid_input(format!(
                "Number of samples ({}) is less than min_samples_split ({})",
                x.nrows(),
                self.params.min_samples_split
            )));
        }

        // At least one feature is required
        if x.ncols() == 0 {
            return Err(Error::invalid_input(
                "Input data must have at least one feature",
            ));
        }

        self.n_features = x.ncols();

        // For classification, determine number of classes and validate labels
        if self.is_classifier {
            let max_class = y
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .ok_or_else(|| Error::computation("Cannot determine max class"))?;

            // Validate that all labels are non-negative integers
            for &label in y.iter() {
                if label < 0.0 || label.fract() != 0.0 {
                    return Err(Error::invalid_input(
                        "Class labels must be non-negative integers starting from 0",
                    ));
                }
            }

            self.n_classes = Some((*max_class as usize) + 1);
        }

        // Progress-bar node estimate: a balanced binary tree of depth d has 2^(d+1) - 1 nodes
        #[cfg(feature = "show_progress")]
        let estimated_max_depth = self.params.max_depth.unwrap_or(20).min(20);
        #[cfg(feature = "show_progress")]
        let estimated_nodes = (1 << (estimated_max_depth + 1)) - 1;

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                estimated_nodes as u64,
                "[{elapsed_precise}] {bar:40} {pos} nodes | Depth: {msg}",
            );
            pb.set_message("0");
            pb
        };

        // RNG is `Some` only when a local `random_state` or the global seed is set; otherwise tie-breaking stays deterministic
        let mut rng = crate::random::make_rng_opt(self.params.random_state);
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(Box::new(self.build_tree(
            x,
            y,
            &indices,
            0,
            &mut rng,
            #[cfg(feature = "show_progress")]
            &progress_bar,
        )?));

        #[cfg(feature = "show_progress")]
        progress_bar
            .finish_with_message(format!("{}", self.count_nodes(self.root.as_ref().unwrap())));

        Ok(self)
    }

    /// Recursively builds a decision tree node by finding optimal splits
    fn build_tree<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
        depth: usize,
        rng: &mut Option<StdRng>,
        #[cfg(feature = "show_progress")] progress_bar: &ProgressBar,
    ) -> Result<Node, Error>
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
        let split_result = self.find_best_split(x, y, indices, rng)?;

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
                // Both children must satisfy the leaf size constraint
                if left.len() < self.params.min_samples_leaf
                    || right.len() < self.params.min_samples_leaf
                {
                    return Ok(self.create_leaf(y, indices));
                }

                let mut node = Node::new_internal(feature, threshold);
                node.left = Some(Box::new(self.build_tree(
                    x,
                    y,
                    &left,
                    depth + 1,
                    rng,
                    #[cfg(feature = "show_progress")]
                    progress_bar,
                )?));
                node.right = Some(Box::new(self.build_tree(
                    x,
                    y,
                    &right,
                    depth + 1,
                    rng,
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
                        rng,
                        #[cfg(feature = "show_progress")]
                        progress_bar,
                    )?;
                    children.insert(key, Box::new(child));
                }

                Ok(node)
            }
        }
    }

    /// Finds the best split for the given samples across all features
    ///
    /// Numeric features are evaluated as binary threshold splits. Features marked as
    /// categorical are evaluated as multi-way splits for ID3 and C4.5 (CART is always
    /// binary). Selection uses information gain (ID3/CART) or gain ratio (C4.5); the
    /// returned `impurity_decrease` drives the `min_impurity_decrease` stopping rule. The
    /// search parallelizes when the sort work clears the calibrated sort-scan gate (see
    /// `crate::parallel_gates`)
    fn find_best_split<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        y: &ArrayBase<S, Ix1>,
        indices: &[usize],
        rng: &mut Option<StdRng>,
    ) -> Result<Option<(Split, f64)>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Calculate parent impurity once
        let parent_impurity = self.calculate_impurity(y, indices);

        // ID3 and C4.5 support multi-way categorical splits; CART is always binary
        let allow_categorical = self.algorithm.allows_multiway_categorical();

        // For each feature, returns (selection_score, split, impurity_decrease)
        let process_feature = |feature_idx: usize| -> Option<(f64, Split, f64)> {
            if allow_categorical && self.categorical_features.contains(&feature_idx) {
                self.evaluate_categorical_split(x, y, indices, feature_idx, parent_impurity)
            } else {
                self.evaluate_numeric_split(x, y, indices, feature_idx, parent_impurity)
            }
        };

        // Collect every feature's best candidate, then pick the winner (sort-scan class
        // gate: one copy + sort + scan task per feature over the node's samples)
        let sort_work = indices.len().saturating_mul(self.n_features);
        let candidates: Vec<(f64, Split, f64)> = if sort_work >= SORT_SCAN_MIN_ELEMS {
            (0..self.n_features)
                .into_par_iter()
                .filter_map(process_feature)
                .collect()
        } else {
            (0..self.n_features).filter_map(process_feature).collect()
        };

        let best = Self::select_best_split(candidates, rng);
        Ok(best.map(|(_score, split, impurity_decrease)| (split, impurity_decrease)))
    }

    /// Picks the highest-scoring split candidate, resolving exact-score ties
    ///
    /// With `rng == None` the last tied candidate wins, which makes the parallel path
    /// deterministic. With `rng == Some`, a uniformly random tied candidate is chosen -
    /// seeded tie-breaking active only when a local `random_state` or the global seed is set
    fn select_best_split(
        mut candidates: Vec<(f64, Split, f64)>,
        rng: &mut Option<StdRng>,
    ) -> Option<(f64, Split, f64)> {
        // Ignore non-finite (NaN) scores
        let max_score = candidates
            .iter()
            .map(|(score, _, _)| *score)
            .filter(|score| !score.is_nan())
            .max_by(|a, b| a.partial_cmp(b).unwrap())?;
        // Indices of all candidates tied at the maximum score; NaN scores never equal `max_score` so they are excluded
        let tied: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, (score, _, _))| *score == max_score)
            .map(|(i, _)| i)
            .collect();
        let chosen = match rng {
            Some(r) => tied[r.random_range(0..tied.len())],
            None => *tied.last().unwrap(),
        };
        Some(candidates.swap_remove(chosen))
    }

    /// Evaluates the best binary numeric split for a single feature
    ///
    /// Returns `(selection_score, split, impurity_decrease)`, or `None` when no
    /// impurity-reducing threshold exists
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
        let n = indices.len();
        if n < 2 {
            return None;
        }
        let n_f = n as f64;

        // Sort the samples by the feature value once (O(n log n)), then sweep left to right
        // maintaining running left/right impurity statistics incrementally
        let mut order: Vec<(f64, usize)> =
            indices.iter().map(|&i| (x[[i, feature_idx]], i)).collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Start below any real score so the best candidate is found regardless of magnitude
        let mut best_score = f64::NEG_INFINITY;
        // (split position, threshold, impurity_decrease); left = order[..pos], right = order[pos..]
        let mut best: Option<(usize, f64, f64)> = None;

        // Evaluates the candidate split between sorted positions `pos-1` and `pos`
        let consider = |pos: usize,
                        imp_left: f64,
                        imp_right: f64,
                        best_score: &mut f64,
                        best: &mut Option<(usize, f64, f64)>| {
            let n_left = pos as f64;
            let n_right = (n - pos) as f64;
            let weighted = (n_left / n_f) * imp_left + (n_right / n_f) * imp_right;
            let impurity_decrease = parent_impurity - weighted;

            // C4.5 uses gain ratio (gain / split information)
            if let Some(score) =
                self.algorithm
                    .selection_score(impurity_decrease, &[n_left, n_right], n_f)
                && score > *best_score
            {
                *best_score = score;
                let threshold = (order[pos - 1].0 + order[pos].0) / 2.0;
                *best = Some((pos, threshold, impurity_decrease));
            }
        };

        if self.is_classifier {
            let n_classes = self.n_classes.unwrap();
            let mut left_counts = vec![0.0_f64; n_classes];
            let mut right_counts = vec![0.0_f64; n_classes];
            for &(_, idx) in &order {
                right_counts[y[idx] as usize] += 1.0;
            }

            for pos in 1..n {
                // Move the sample at position pos-1 from the right child to the left child
                let cls = y[order[pos - 1].1] as usize;
                left_counts[cls] += 1.0;
                right_counts[cls] -= 1.0;

                // A split is only valid between two distinct feature values
                if order[pos - 1].0 == order[pos].0 {
                    continue;
                }

                let imp_left = self
                    .algorithm
                    .impurity_from_counts(&left_counts, pos as f64);
                let imp_right = self
                    .algorithm
                    .impurity_from_counts(&right_counts, (n - pos) as f64);
                consider(pos, imp_left, imp_right, &mut best_score, &mut best);
            }
        } else {
            // Regression (CART/MSE): maintain a running sum and sum of squares for the left side
            // The sums stay serial: this runs inside the per-feature parallel split search, and
            // the prefix scan below is inherently sequential anyway
            let total_sum: f64 = order.iter().map(|&(_, idx)| y[idx]).sum();
            let total_sumsq: f64 = order.iter().map(|&(_, idx)| y[idx] * y[idx]).sum();
            let mut left_sum = 0.0;
            let mut left_sumsq = 0.0;

            for pos in 1..n {
                let val = y[order[pos - 1].1];
                left_sum += val;
                left_sumsq += val * val;

                if order[pos - 1].0 == order[pos].0 {
                    continue;
                }

                let n_left = pos as f64;
                let n_right = (n - pos) as f64;
                let right_sum = total_sum - left_sum;
                let right_sumsq = total_sumsq - left_sumsq;
                let var_left = (left_sumsq / n_left - (left_sum / n_left).powi(2)).max(0.0);
                let var_right = (right_sumsq / n_right - (right_sum / n_right).powi(2)).max(0.0);
                consider(pos, var_left, var_right, &mut best_score, &mut best);
            }
        }

        best.map(|(pos, threshold, decrease)| {
            let left: Vec<usize> = order[..pos].iter().map(|&(_, idx)| idx).collect();
            let right: Vec<usize> = order[pos..].iter().map(|&(_, idx)| idx).collect();
            (
                best_score,
                Split::Numeric {
                    feature: feature_idx,
                    threshold,
                    left,
                    right,
                },
                decrease,
            )
        })
    }

    /// Evaluates a multi-way categorical split for a single feature (one branch per value)
    ///
    /// Returns `(selection_score, split, impurity_decrease)`, or `None` when the feature
    /// has fewer than 2 distinct values or the split does not reduce impurity
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

        // A useful split needs at least 2 distinct values
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

        // Defer the acceptance threshold to the single `min_impurity_decrease` gate in
        // build_tree (consistent with the numeric path)
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

    /// Calculates the impurity measure for the given samples based on the algorithm type
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

    /// Calculates mean squared error for regression samples
    fn calculate_mse<S>(&self, y: &ArrayBase<S, Ix1>, indices: &[usize]) -> f64
    where
        S: Data<Elem = f64>,
    {
        // Collect subset values into an Array1 to use math::variance
        let subset: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        variance(&subset)
    }

    /// Checks if all samples in the given indices have the same label (pure node)
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

    /// Creates a leaf node with the appropriate prediction value based on the task type
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

    /// Predicts the output for a single sample
    ///
    /// # Parameters
    ///
    /// - `x` - Feature vector for a single sample as a slice of length `n_features`
    ///
    /// # Returns
    ///
    /// - `Result<f64, Error>` - The predicted value, or an `Error` if prediction fails
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been trained yet
    /// - `Error::DimensionMismatch` - If the feature dimension mismatches
    /// - `Error::Tree` - If the tree structure is broken
    pub fn predict_one(&self, x: &[f64]) -> Result<f64, Error> {
        if self.root.is_none() {
            return Err(Error::not_fitted("DecisionTree"));
        }

        if x.len() != self.n_features {
            return Err(Error::dimension_mismatch(self.n_features, x.len()));
        }

        self.traverse_tree(self.root.as_ref().unwrap(), x)
    }

    /// Traverses the tree from a given node to make a prediction for a single sample
    fn traverse_tree(&self, node: &Node, x: &[f64]) -> Result<f64, Error> {
        match &node.node_type {
            NodeType::Leaf { value, .. } => Ok(*value),
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    // Route by category value, falling back to the default leaf for values unseen during training
                    let key = category_key(x[*feature_index]);
                    if let Some(child) = node.children.as_ref().and_then(|c| c.get(&key)) {
                        return self.traverse_tree(child, x);
                    }
                    return match &node.left {
                        Some(fallback) => self.traverse_tree(fallback, x),
                        None => Err(Error::Tree(TreeError::CorruptStructure(
                            "Categorical node has no matching child and no fallback",
                        ))),
                    };
                }

                // Binary split
                if x[*feature_index] <= *threshold {
                    if let Some(ref left) = node.left {
                        self.traverse_tree(left, x)
                    } else {
                        Err(Error::Tree(TreeError::CorruptStructure(
                            "Missing left child",
                        )))
                    }
                } else {
                    if let Some(ref right) = node.right {
                        self.traverse_tree(right, x)
                    } else {
                        Err(Error::Tree(TreeError::CorruptStructure(
                            "Missing right child",
                        )))
                    }
                }
            }
        }
    }

    /// Predicts outputs for multiple samples
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array of predicted values
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been trained yet
    /// - `Error::EmptyInput` - If input data is empty
    /// - `Error::DimensionMismatch` - If feature dimensions mismatch
    ///
    /// # Performance
    ///
    /// Parallelizes when the traversal work clears the calibrated tree-traversal gate (see
    /// `crate::parallel_gates`)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        check_is_fitted(self.root.is_some(), "DecisionTree")?;
        validate_predict_input(x, self.n_features)?;

        // Tree-traversal gate: one root-to-leaf walk per sample
        let visit_work = x.nrows().saturating_mul(DECISION_TREE_ASSUMED_DEPTH);
        let predictions: Result<Vec<f64>, Error> = if visit_work >= TREE_TRAVERSAL_MIN_VISITS {
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

    /// Trains the tree on training data and immediately makes predictions
    ///
    /// # Parameters
    ///
    /// - `x_train` - Training features as a 2D array
    /// - `y_train` - Training labels as a 1D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array of predictions for the training data
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If training or prediction inputs are invalid
    pub fn fit_predict<S>(
        &mut self,
        x_train: &ArrayBase<S, Ix2>,
        y_train: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x_train, y_train)?;
        self.predict(x_train)
    }

    /// Predicts class probabilities for multiple samples (classification only)
    ///
    /// # Parameters
    ///
    /// - `x` - Feature matrix as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, Error>` - A 2D array of class probabilities where each row sums to 1.0
    ///
    /// # Errors
    ///
    /// - `Error::Tree` - If called on a regression tree
    /// - `Error::NotFitted` - If the model has not been trained yet
    ///
    /// # Performance
    ///
    /// Parallelizes when the traversal work clears the calibrated tree-traversal gate (see
    /// `crate::parallel_gates`)
    pub fn predict_proba<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        if !self.is_classifier {
            return Err(Error::Tree(TreeError::NotClassificationTree));
        }

        check_is_fitted(self.root.is_some(), "DecisionTree")?;
        validate_predict_input(x, self.n_features)?;

        let n_classes = self.n_classes.unwrap();

        let probabilities: Result<Vec<Vec<f64>>, Error> =
            if x.nrows().saturating_mul(DECISION_TREE_ASSUMED_DEPTH) >= TREE_TRAVERSAL_MIN_VISITS {
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

        let mut result = Array2::zeros((x.nrows(), n_classes));
        for (i, proba) in probabilities.iter().enumerate() {
            for (j, &p) in proba.iter().enumerate() {
                result[[i, j]] = p;
            }
        }

        Ok(result)
    }

    /// Predicts class probabilities for a single sample (classification only)
    ///
    /// # Parameters
    ///
    /// - `x` - Feature vector for a single sample as a slice
    ///
    /// # Returns
    ///
    /// - `Result<Vec<f64>, Error>` - A vector of class probabilities
    ///
    /// # Errors
    ///
    /// - `Error::Tree` - If called on a regression tree
    /// - `Error::DimensionMismatch` - If the feature dimension mismatches
    /// - `Error::NotFitted` - If the model has not been trained yet
    pub fn predict_proba_one(&self, x: &[f64]) -> Result<Vec<f64>, Error> {
        if !self.is_classifier {
            return Err(Error::Tree(TreeError::NotClassificationTree));
        }

        if self.root.is_none() {
            return Err(Error::not_fitted("DecisionTree"));
        }

        if x.len() != self.n_features {
            return Err(Error::dimension_mismatch(self.n_features, x.len()));
        }

        self.get_probabilities(self.root.as_ref().unwrap(), x)
    }

    /// Traverses the tree to retrieve class probabilities from the appropriate leaf node
    fn get_probabilities(&self, node: &Node, x: &[f64]) -> Result<Vec<f64>, Error> {
        match &node.node_type {
            NodeType::Leaf { probabilities, .. } => {
                probabilities.as_ref().cloned().ok_or_else(|| {
                    Error::Tree(TreeError::CorruptStructure("No probabilities in leaf node"))
                })
            }
            NodeType::Internal {
                feature_index,
                threshold,
                categories,
            } => {
                if categories.is_some() {
                    // Route by category value, falling back to the default leaf for values unseen during training
                    let key = category_key(x[*feature_index]);
                    if let Some(child) = node.children.as_ref().and_then(|c| c.get(&key)) {
                        return self.get_probabilities(child, x);
                    }
                    return match &node.left {
                        Some(fallback) => self.get_probabilities(fallback, x),
                        None => Err(Error::Tree(TreeError::CorruptStructure(
                            "Categorical node has no matching child and no fallback",
                        ))),
                    };
                }

                if x[*feature_index] <= *threshold {
                    if let Some(ref left) = node.left {
                        self.get_probabilities(left, x)
                    } else {
                        Err(Error::Tree(TreeError::CorruptStructure(
                            "Missing left child",
                        )))
                    }
                } else {
                    if let Some(ref right) = node.right {
                        self.get_probabilities(right, x)
                    } else {
                        Err(Error::Tree(TreeError::CorruptStructure(
                            "Missing right child",
                        )))
                    }
                }
            }
        }
    }

    /// Generates a human-readable string representation of the decision tree structure
    ///
    /// # Returns
    ///
    /// - `Result<String, Error>` - A formatted string containing the tree structure
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been trained yet
    pub fn generate_tree_structure(&self) -> Result<String, Error> {
        if self.root.is_none() {
            return Err(Error::not_fitted("DecisionTree"));
        }

        let mut output = String::new();
        output.push_str("Decision Tree Structure:\n");
        self.print_node(self.root.as_ref().unwrap(), &mut output, "", true);
        Ok(output)
    }

    /// Counts the total number of nodes in the tree
    #[cfg(feature = "show_progress")]
    fn count_nodes(&self, node: &Node) -> usize {
        let mut count = 1; // current node
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

    /// Recursively prints the tree structure
    fn print_node(&self, node: &Node, output: &mut String, prefix: &str, is_last: bool) {
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

                    // One branch per category value (sorted by key for stable output), plus a default branch for unseen values
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A numeric-split candidate with the given selection score and feature index
    fn numeric_candidate(score: f64, feature: usize) -> (f64, Split, f64) {
        (
            score,
            Split::Numeric {
                feature,
                threshold: 0.0,
                left: Vec::new(),
                right: Vec::new(),
            },
            score,
        )
    }

    /// select_best_split skips NaN-scoring candidates and picks the finite best
    #[test]
    fn select_best_split_ignores_nan_and_picks_finite_best() {
        let candidates = vec![
            numeric_candidate(f64::NAN, 0),
            numeric_candidate(0.25, 1),
            numeric_candidate(0.75, 2), // the finite best
            numeric_candidate(f64::NAN, 3),
        ];
        let mut rng: Option<StdRng> = None;
        let (score, split, _) = DecisionTree::select_best_split(candidates, &mut rng)
            .expect("a finite-scoring split exists");
        assert_eq!(score, 0.75);
        match split {
            Split::Numeric { feature, .. } => assert_eq!(feature, 2),
            _ => panic!("expected the finite best numeric split"),
        }
    }

    /// select_best_split returns None when every candidate scores NaN
    #[test]
    fn select_best_split_all_nan_scores_is_none() {
        let candidates = vec![
            numeric_candidate(f64::NAN, 0),
            numeric_candidate(f64::NAN, 1),
        ];
        let mut rng: Option<StdRng> = None;
        assert!(DecisionTree::select_best_split(candidates, &mut rng).is_none());
    }
    /// category_key collapses sub-1e-6 noise to one key but keeps distinct values distinct
    #[test]
    fn category_key_collapses_subkey_noise_but_separates_distinct_values() {
        // Sub-1e-6 noise rounds to the same canonical key
        assert_eq!(category_key(1.0000001), category_key(1.0000002));
        // ... and that shared key is exactly "1"
        assert_eq!(category_key(1.0000001), "1");
        // Distinct integer categories stay distinct
        assert_ne!(category_key(1.0), category_key(2.0));
        assert_eq!(category_key(1.0), "1");
        assert_eq!(category_key(2.0), "2");
    }

    /// split_information is 1 bit for an even 2/2 split and 0 for a single branch
    #[test]
    fn split_information_even_two_way_is_one_bit_and_single_branch_is_zero() {
        assert!((split_information(&[2.0, 2.0], 4.0) - 1.0).abs() < 1e-12);
        assert!(split_information(&[4.0], 4.0).abs() < 1e-12);
    }

    /// selection_score returns None for C4.5 on a degenerate split, the gain ratio otherwise, and the raw decrease for ID3/CART
    #[test]
    fn selection_score_c45_none_on_degenerate_split_info() {
        // Degenerate single-branch partition => split_info ~= 0 => None for C4.5
        assert!(
            Algorithm::C45.selection_score(0.5, &[4.0], 4.0).is_none(),
            "C4.5 must reject a split whose intrinsic value is ~0"
        );
        // A well-formed 2/2 split has split_info = 1.0, so the gain ratio is decrease / 1.0
        let ratio = Algorithm::C45
            .selection_score(0.4, &[2.0, 2.0], 4.0)
            .expect("non-degenerate split has a gain ratio");
        assert!((ratio - 0.4).abs() < 1e-12);
        // ID3 and CART return the raw decrease unconditionally (no split-info guard)
        assert_eq!(Algorithm::ID3.selection_score(0.4, &[4.0], 4.0), Some(0.4));
        assert_eq!(Algorithm::CART.selection_score(0.4, &[4.0], 4.0), Some(0.4));
    }

    /// traverse_tree returns CorruptStructure when a sample routes to a missing left child
    #[test]
    fn traverse_tree_missing_left_child_is_corrupt_structure() {
        let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
        // Internal binary node, threshold 0.5, with NO left child
        let mut node = Node::new_internal(0, 0.5);
        node.right = Some(Box::new(Node::new_leaf(1.0, Some(1), Some(vec![0.0, 1.0]))));
        // 0.0 <= 0.5 routes left, where there is no child
        let err = tree.traverse_tree(&node, &[0.0]).unwrap_err();
        assert!(
            matches!(err, Error::Tree(TreeError::CorruptStructure(_))),
            "expected CorruptStructure for missing left child, got {err:?}"
        );
    }

    /// traverse_tree returns CorruptStructure when a sample routes to a missing right child
    #[test]
    fn traverse_tree_missing_right_child_is_corrupt_structure() {
        let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
        let mut node = Node::new_internal(0, 0.5);
        node.left = Some(Box::new(Node::new_leaf(0.0, Some(0), Some(vec![1.0, 0.0]))));
        // 1.0 > 0.5 routes right, where there is no child
        let err = tree.traverse_tree(&node, &[1.0]).unwrap_err();
        assert!(
            matches!(err, Error::Tree(TreeError::CorruptStructure(_))),
            "expected CorruptStructure for missing right child, got {err:?}"
        );
    }

    /// traverse_tree returns CorruptStructure for a categorical node with no matching child and no fallback
    #[test]
    fn traverse_tree_categorical_no_match_no_fallback_is_corrupt_structure() {
        let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
        // Categorical internal node: children map present but empty, and left (fallback) is None
        let node = Node::new_categorical(0, vec!["0".to_string()]);
        // No category key can match an empty map; with no fallback this is corrupt
        let err = tree.traverse_tree(&node, &[0.0]).unwrap_err();
        assert!(
            matches!(err, Error::Tree(TreeError::CorruptStructure(_))),
            "expected CorruptStructure for categorical node with no child and no fallback, got {err:?}"
        );
    }

    /// get_probabilities returns CorruptStructure for a leaf with no stored probability vector
    #[test]
    fn get_probabilities_leaf_without_probabilities_is_corrupt_structure() {
        let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
        // Leaf with class set but no stored probability distribution
        let leaf = Node::new_leaf(0.0, Some(0), None);
        let err = tree.get_probabilities(&leaf, &[0.0]).unwrap_err();
        assert!(
            matches!(err, Error::Tree(TreeError::CorruptStructure(_))),
            "expected CorruptStructure for a leaf missing probabilities, got {err:?}"
        );
    }

    /// get_probabilities returns CorruptStructure for a categorical node with no matching child and no fallback
    #[test]
    fn get_probabilities_categorical_no_match_no_fallback_is_corrupt_structure() {
        let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
        let node = Node::new_categorical(0, vec!["0".to_string()]);
        let err = tree.get_probabilities(&node, &[0.0]).unwrap_err();
        assert!(
            matches!(err, Error::Tree(TreeError::CorruptStructure(_))),
            "expected CorruptStructure for categorical get_probabilities with no child and no fallback, got {err:?}"
        );
    }
}

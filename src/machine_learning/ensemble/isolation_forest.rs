//! Isolation Forest for unsupervised anomaly detection
//!
//! Provides the [`IsolationForest`] estimator and its underlying [`IsolationTree`]
//! node type. They isolate outliers via random feature splits and score samples by
//! their average path length across the forest

use crate::error::Error;
use crate::machine_learning::validation::{
    check_is_fitted, preliminary_check, validate_predict_input,
};
use crate::math::average_path_length_factor;
use crate::parallel_gates::tree_traversal_min_visits;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::rngs::StdRng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Default minimum number of trees required to enable parallel tree construction
///
/// A synthetic ~0.7 us-per-tree build crosses over at 16-32 tasks, roughly 22 us of total
/// work, about one rayon fork/join. A real iTree build (subsample gather, allocations) costs
/// well over 2.2 us per tree, so at this gate the parallel path's total work clears the fork
/// cost with margin
const DEFAULT_PARALLEL_THRESHOLD_TREES: usize = 10;

/// Average isolation-tree path length for the prediction work estimate, `c(256) ~= 10`
/// (`2 * H(psi - 1) - 2(psi - 1)/psi` at the default subsample size 256)
const ISOLATION_TREE_AVG_PATH: usize = 10;

/// A node in an isolation tree
///
/// Unlike a decision-tree node, an isolation tree only needs to record where it
/// split and how many samples ended up unresolved at a leaf, so it uses this
/// minimal structure rather than reusing the classification-oriented decision
/// tree node
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum IsolationTree {
    /// A terminal node reached by samples that could not be split further
    Leaf {
        /// Number of samples that reached this leaf, used by the average-path-length correction
        size: usize,
    },
    /// A split node directing samples to one of the two children
    Internal {
        /// Index of the feature this node splits on
        feature: usize,
        /// Split value; samples with a feature value below this go left
        threshold: f64,
        /// Subtree for samples below the threshold
        left: Box<IsolationTree>,
        /// Subtree for samples at or above the threshold
        right: Box<IsolationTree>,
    },
}

/// Isolation Forest for anomaly detection
///
/// An unsupervised algorithm that detects anomalies by isolating outliers in the
/// data. Each tree randomly selects a feature, then randomly selects a split value between
/// the minimum and maximum values of that feature. Anomalies are isolated more easily and so
/// have shorter average path lengths in the trees
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::IsolationForest;
/// use ndarray::array;
///
/// let mut model = IsolationForest::new(100, 256).unwrap().with_random_state(42);
/// let data = array![[1.0, 2.0], [2.0, 3.0], [10.0, 15.0]];
/// model.fit(&data).unwrap();
/// let scores = model.predict(&data).unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IsolationForest {
    /// Collection of isolation trees built during training
    trees: Option<Vec<IsolationTree>>,
    /// Number of isolation trees in the forest
    n_estimators: usize,
    /// Maximum number of samples to draw from the dataset for each tree
    max_samples: usize,
    /// Maximum depth of each isolation tree
    max_depth: usize,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Number of features in the training data
    n_features: usize,
    /// Actual sub-sample size used per tree at fit time, `min(max_samples, n_rows)`. The
    /// anomaly-score normalization `c(n)` uses this realized size, not `max_samples`, so the
    /// scores stay correct when the dataset is smaller than `max_samples`
    sample_size: usize,
}

impl Default for IsolationForest {
    /// Creates an Isolation Forest with default parameters
    ///
    /// # Default Values
    ///
    /// - `trees` - None (not fitted)
    /// - `n_estimators` - 100
    /// - `max_samples` - 256
    /// - `max_depth` - 8 (ceil(log2(256)))
    /// - `random_state` - None
    /// - `n_features` - 0
    fn default() -> Self {
        Self {
            trees: None,
            n_estimators: 100,
            max_samples: 256,
            sample_size: 0,
            max_depth: 8, // ceil(log2(256)) = 8
            random_state: None,
            n_features: 0,
        }
    }
}

impl IsolationForest {
    /// Creates a new Isolation Forest with specified parameters
    ///
    /// # Parameters
    ///
    /// - `n_estimators` - Number of isolation trees to build
    /// - `max_samples` - Maximum number of samples to draw from the dataset for each tree
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new unfitted `IsolationForest` instance
    ///
    /// # Notes
    ///
    /// By default the tree depth is derived as `ceil(log2(max_samples))` and the forest is
    /// seeded non-deterministically. Override either with the builder methods below:
    ///
    /// - [`with_max_depth`](Self::with_max_depth) - maximum depth of each tree (returns `Result`; the depth is validated)
    /// - [`with_random_state`](Self::with_random_state) - fixed RNG seed for reproducible forests
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParameter` if `n_estimators` or `max_samples` is 0
    pub fn new(n_estimators: usize, max_samples: usize) -> Result<Self, Error> {
        if n_estimators == 0 {
            return Err(Error::invalid_parameter(
                "n_estimators",
                "must be greater than 0",
            ));
        }

        if max_samples == 0 {
            return Err(Error::invalid_parameter(
                "max_samples",
                "must be greater than 0",
            ));
        }

        // ceil(log2(max_samples))
        let computed_max_depth = (max_samples as f64).log2().ceil() as usize;

        Ok(Self {
            trees: None,
            n_estimators,
            max_samples,
            sample_size: 0,
            max_depth: computed_max_depth,
            random_state: None,
            n_features: 0,
        })
    }

    /// Overrides the maximum depth of each isolation tree (default: `ceil(log2(max_samples))`)
    ///
    /// # Parameters
    ///
    /// - `max_depth` - Maximum depth of each tree; must be greater than 0
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if `max_depth` is 0
    pub fn with_max_depth(mut self, max_depth: usize) -> Result<Self, Error> {
        if max_depth == 0 {
            return Err(Error::invalid_parameter(
                "max_depth",
                "must be greater than 0",
            ));
        }
        self.max_depth = max_depth;
        Ok(self)
    }

    /// Sets a fixed RNG seed, making forest construction reproducible (default: `None`, non-deterministic)
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the tree-building RNG
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    // Getters
    get_field!(get_n_estimators, n_estimators, usize);
    get_field!(get_max_samples, max_samples, usize);
    get_field!(get_sample_size, sample_size, usize);
    get_field!(get_max_depth, max_depth, usize);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field!(get_n_features, n_features, usize);
    get_field_as_ref!(get_trees, trees, Option<&Vec<IsolationTree>>);

    /// Trains the Isolation Forest model on the provided dataset
    ///
    /// Builds multiple isolation trees by randomly sampling subsets of the data and
    /// recursively partitioning them using random feature splits
    ///
    /// # Parameters
    ///
    /// - `x` - Training data as a 2D array where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - A mutable reference to self if successful
    ///
    /// # Errors
    ///
    /// Returns `Error` if:
    /// - Input data is empty
    /// - Input contains NaN or infinite values
    /// - Tree building fails
    ///
    /// # Performance
    ///
    /// Uses parallelization when the number of trees reaches `DEFAULT_PARALLEL_THRESHOLD_TREES`
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        preliminary_check(x, None)?;

        self.n_features = x.ncols();
        self.sample_size = self.max_samples.min(x.nrows());

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.n_estimators as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | {msg}",
            );
            pb.set_message("Building isolation trees");
            pb
        };

        let build_tree = |i: usize| -> Result<IsolationTree, Error> {
            // An independent RNG per tree keeps results reproducible
            let mut rng =
                crate::random::make_rng(self.random_state.map(|s| s.wrapping_add(i as u64)));

            // Sample a subset of data for this tree (same realized size for every tree)
            let sample_indices = self.sample_indices(x.nrows(), self.sample_size, &mut rng);

            let result = self.build_isolation_tree(x, &sample_indices, 0, &mut rng);

            #[cfg(feature = "show_progress")]
            progress_bar.inc(1);

            result
        };

        let trees: Result<Vec<IsolationTree>, Error> =
            if self.n_estimators >= DEFAULT_PARALLEL_THRESHOLD_TREES {
                (0..self.n_estimators)
                    .into_par_iter()
                    .map(build_tree)
                    .collect()
            } else {
                (0..self.n_estimators).map(build_tree).collect()
            };

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("Trees built successfully");

        self.trees = Some(trees?);

        Ok(self)
    }

    /// Randomly samples indices from the dataset using Fisher-Yates shuffle
    fn sample_indices(&self, n: usize, sample_size: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle over the first sample_size elements
        for i in 0..sample_size {
            let j = rng.random_range(i..n);
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices
    }

    /// Recursively builds an isolation tree by randomly selecting features and split points
    fn build_isolation_tree<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        indices: &[usize],
        current_depth: usize,
        rng: &mut StdRng,
    ) -> Result<IsolationTree, Error>
    where
        S: Data<Elem = f64>,
    {
        // Stop when max depth is reached or the node has at most one sample
        if current_depth >= self.max_depth || indices.len() <= 1 {
            return Ok(IsolationTree::Leaf {
                size: indices.len(),
            });
        }

        let feature_index = rng.random_range(0..self.n_features);

        // Find min and max values for the selected feature in the current subset
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &idx in indices {
            let val = x[[idx, feature_index]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // All values equal: nothing to split on, so make a leaf
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(IsolationTree::Leaf {
                size: indices.len(),
            });
        }

        let threshold = rng.random_range(min_val..max_val);

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&idx| x[[idx, feature_index]] < threshold);

        // An empty partition means no real split, so make a leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(IsolationTree::Leaf {
                size: indices.len(),
            });
        }

        let left = self.build_isolation_tree(x, &left_indices, current_depth + 1, rng)?;
        let right = self.build_isolation_tree(x, &right_indices, current_depth + 1, rng)?;

        Ok(IsolationTree::Internal {
            feature: feature_index,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Computes the path length of a sample through an isolation tree with average adjustment for unresolved samples
    fn path_length(&self, sample: &[f64], node: &IsolationTree, current_depth: usize) -> f64 {
        match node {
            IsolationTree::Leaf { size } => {
                // Add the average-path-length correction for samples unresolved at this leaf
                current_depth as f64 + average_path_length_factor(*size)
            }
            IsolationTree::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    self.path_length(sample, left, current_depth + 1)
                } else {
                    self.path_length(sample, right, current_depth + 1)
                }
            }
        }
    }

    /// Computes the anomaly score for a single sample
    ///
    /// The anomaly score is normalized to the range \[0, 1\], where values close to 1
    /// indicate anomalies and values close to 0 indicate normal samples
    ///
    /// # Parameters
    ///
    /// - `sample` - A single data sample as a slice of feature values
    ///
    /// # Returns
    ///
    /// - `Result<f64, Error>` - The anomaly score between 0 and 1
    ///
    /// # Errors
    ///
    /// Returns `Error` if:
    /// - Model has not been fitted
    /// - Sample feature dimension does not match training data
    pub fn anomaly_score(&self, sample: &[f64]) -> Result<f64, Error> {
        if self.trees.is_none() {
            return Err(Error::not_fitted("IsolationForest"));
        }

        if sample.len() != self.n_features {
            return Err(Error::dimension_mismatch(self.n_features, sample.len()));
        }

        let trees = self.trees.as_ref().unwrap();

        // Normalize using c(sample_size), the actual sub-sample size used at fit time
        let c_n = average_path_length_factor(self.sample_size);

        Ok(self.normalized_score(sample, trees, c_n))
    }

    /// Computes the normalized anomaly score for one sample, given a precomputed
    /// normalization constant `c_n = c(sample_size)`
    ///
    /// Taking `c_n` as a parameter lets batch prediction compute it once instead of
    /// recomputing it for every sample
    fn normalized_score(&self, sample: &[f64], trees: &[IsolationTree], c_n: f64) -> f64 {
        // Average path length across all trees; one term per tree stays below any parallel
        // reduction gate, and the per-sample batch loop parallelizes one level up
        let avg_path_length: f64 = trees
            .iter()
            .map(|tree| self.path_length(sample, tree, 0))
            .sum::<f64>()
            / trees.len() as f64;

        // A degenerate sub-sample of size <= 1 makes c(n) = 0
        if c_n <= 0.0 {
            return 1.0;
        }

        // Anomaly score: s(x, n) = 2^(-E(h(x)) / c(n))
        2.0_f64.powf(-avg_path_length / c_n)
    }

    /// Predicts anomaly scores for multiple samples
    ///
    /// # Parameters
    ///
    /// - `x` - Input data as a 2D array where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array of anomaly scores
    ///
    /// # Errors
    ///
    /// Returns `Error` if:
    /// - Model has not been fitted
    /// - Input data is empty
    /// - Feature dimension does not match training data
    /// - Input contains NaN or infinite values
    ///
    /// # Performance
    ///
    /// Parallelizes when the traversal work (samples x trees x average path length) clears
    /// the calibrated tree-traversal gate (see `crate::parallel_gates`)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        check_is_fitted(self.trees.is_some(), "IsolationForest")?;
        validate_predict_input(x, self.n_features)?;

        // Precompute the normalization constant once for the whole batch
        let trees = self.trees.as_ref().unwrap();
        let c_n = average_path_length_factor(self.sample_size);

        // Tree-traversal gate
        let visit_work = x
            .nrows()
            .saturating_mul(trees.len())
            .saturating_mul(ISOLATION_TREE_AVG_PATH);

        let score_row = |row: ndarray::ArrayView1<f64>| match row.as_slice() {
            Some(slice) => self.normalized_score(slice, trees, c_n),
            None => self.normalized_score(&row.to_vec(), trees, c_n),
        };
        let scores: Vec<f64> = if visit_work >= tree_traversal_min_visits() {
            x.axis_iter(Axis(0))
                .into_par_iter()
                .map(score_row)
                .collect()
        } else {
            x.axis_iter(Axis(0)).map(score_row).collect()
        };

        Ok(Array1::from_vec(scores))
    }

    /// Classifies each sample as an inlier (`+1`) or outlier (`-1`) using a contamination rate
    ///
    /// `contamination` is the expected proportion of outliers in `x`. The
    /// `ceil(contamination * n_samples)` highest-scoring samples (see [`predict`](Self::predict))
    /// are labelled `-1` (outlier) and the rest `+1` (inlier). Ties at the threshold score are
    /// all labelled `-1`, so at least `ceil(contamination * n_samples)` samples are flagged
    ///
    /// # Parameters
    ///
    /// - `x` - Input data matrix where each row is a sample
    /// - `contamination` - Expected proportion of outliers, in `(0.0, 0.5]`
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - Per-sample labels: `-1` for outliers, `+1` for inliers
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `contamination` is not finite or not in `(0.0, 0.5]`
    /// - `Error::NotFitted` / `Error::EmptyInput` / `Error::DimensionMismatch` / `Error::NonFinite` -
    ///   propagated from [`predict`](Self::predict)
    pub fn predict_labels<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        contamination: f64,
    ) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64>,
    {
        if !contamination.is_finite() || contamination <= 0.0 || contamination > 0.5 {
            return Err(Error::invalid_parameter(
                "contamination",
                format!("must be in (0.0, 0.5], got {contamination}"),
            ));
        }

        // Anomaly scores in [0, 1]; higher means more anomalous. `predict` validates input
        let scores = self.predict(x)?;
        let n = scores.len();

        // Flag the ceil(contamination * n) highest-scoring samples (at least 1)
        let n_outliers = (((n as f64) * contamination).ceil() as usize).clamp(1, n);

        // Threshold = n_outliers-th largest score, via quickselect at ascending index n - n_outliers
        let mut sorted: Vec<f64> = scores.to_vec();
        let kth = n - n_outliers;
        sorted.select_nth_unstable_by(kth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = sorted[kth];

        Ok(scores.mapv(|s| if s >= threshold { -1 } else { 1 }))
    }

    /// Trains the model on the dataset and immediately predicts anomaly scores
    ///
    /// This is a convenience method that combines `fit` and `predict` in one call
    ///
    /// # Parameters
    ///
    /// - `x` - Training and prediction data as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - A 1D array of anomaly scores for the training data
    ///
    /// # Errors
    ///
    /// Returns `Error` if fitting or prediction fails
    pub fn fit_predict<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x)?;
        self.predict(x)
    }

    model_save_and_load_methods!(IsolationForest);
}

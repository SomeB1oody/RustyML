//! Isolation Forest for unsupervised anomaly detection
//!
//! Provides the [`IsolationForest`] estimator and its underlying [`IsolationTree`]
//! node type. They isolate outliers via random feature splits and score samples by
//! their average path length across the forest.
//!
//! The inlier/outlier decision threshold is [`Contamination`], resolved against the
//! *training* scores at fit time. [`IsolationForest::predict`] is therefore a fitted rule: it
//! gives the same answer whether the caller scores a sample alone or in a batch.

use crate::error::Error;
use crate::machine_learning::validation::{
    check_is_fitted, preliminary_check, validate_predict_input,
};
use crate::parallel_gates::tree_traversal_min_visits;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::rngs::StdRng;

/// Euler-Mascheroni constant, used in the harmonic-number approximation of
/// [`average_path_length_factor`]
const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243104215933593992;

/// The inlier/outlier decision threshold for [`IsolationForest`]
///
/// Resolved into a concrete score cutoff at fit time from the *training* scores (see
/// [`IsolationForest::get_offset`]). The resulting rule does not depend on how you later
/// batch the samples you predict on
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Contamination {
    /// `'auto'`: the `-0.5` cutoff from the original Isolation Forest paper. A sample is an
    /// outlier when it isolates faster than the forest's expected path length.
    Auto,
    /// The expected proportion of outliers in the training data, in `(0.0, 0.5]`
    ///
    /// The cutoff becomes the `100 * fraction`-th percentile of the training scores, using
    /// NumPy's default linear interpolation so it equals scikit-learn's `offset_`
    Fraction(f64),
}

/// Average path-length normalization factor `c(n)` for isolation trees
///
/// Equals the expected path length of an unsuccessful search in a binary search tree of `n`
/// points. Normalizes raw path lengths into anomaly scores. Returns 0.0 for `n <= 1` and 1.0
/// for `n == 2`.
#[inline]
fn average_path_length_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }

    let h_n_minus_1 = if n > 50 {
        // Asymptotic expansion of the harmonic number H_m (m = n - 1)
        let m = (n - 1) as f64;
        m.ln() + EULER_GAMMA + 0.5 / m
    } else {
        (1..n).map(|i| 1.0 / i as f64).sum::<f64>()
    };

    2.0 * h_n_minus_1 - 2.0 * (n - 1) as f64 / n as f64
}
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Minimum number of trees needed before the forest builds trees in parallel
const DEFAULT_PARALLEL_THRESHOLD_TREES: usize = 10;

/// Average isolation-tree path length, used to estimate per-sample work for the parallel gate
///
/// `c(256)` is about 10, the value at the default subsample size of 256
const ISOLATION_TREE_AVG_PATH: usize = 10;

/// Score cutoff for [`Contamination::Auto`]: the threshold from Liu et al. A raw
/// normalized score above `0.5` means the sample isolates faster than the forest average.
/// Negated to match the sign convention of [`IsolationForest::score_samples`], and identical
/// to scikit-learn's `offset_` under `contamination='auto'`
const AUTO_CONTAMINATION_OFFSET: f64 = -0.5;

/// A node in an isolation tree
///
/// An isolation tree only needs to record where it split and how many samples ended up
/// unresolved at a leaf. It uses this minimal structure instead of the classification-oriented
/// decision tree node.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum IsolationTree {
    /// A terminal node that holds samples the tree could not split further
    Leaf {
        /// Number of samples that reached this leaf, used by the average-path-length correction
        size: usize,
    },
    /// A split node that directs samples to one of the 2 children
    Internal {
        /// Index of the feature this node splits on
        feature: usize,
        /// Split value. Samples with a feature value below this go left.
        threshold: f64,
        /// Subtree for samples below the threshold
        left: Box<IsolationTree>,
        /// Subtree for samples at or above the threshold
        right: Box<IsolationTree>,
    },
}

/// Isolation Forest for anomaly detection
///
/// An unsupervised algorithm that detects anomalies by isolating outliers in the data. Each
/// tree randomly selects a feature, then randomly selects a split value between the minimum
/// and maximum values of that feature. Trees isolate anomalies more easily, so anomalies get
/// shorter average path lengths.
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
///
/// // Anomaly scores in [-1, 0). Lower means more anomalous, as in scikit-learn.
/// let scores = model.score_samples(&data).unwrap();
///
/// // Same scores, shifted by the fitted offset. Negative marks an outlier.
/// let decision = model.decision_function(&data).unwrap();
///
/// // Hard labels: -1 for outliers, +1 for inliers. This matches the sign of `decision_function`.
/// let labels = model.predict(&data).unwrap();
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
    /// Rule for choosing the inlier/outlier cutoff, resolved into `offset` at fit time
    contamination: Contamination,
    /// Score cutoff separating inliers from outliers, `None` before training.
    /// [`IsolationForest::predict`] labels a sample `-1` when it scores at or above this value.
    offset: Option<f64>,
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
    /// - `contamination` - `Contamination::Auto` (the paper's 0.5 cutoff)
    fn default() -> Self {
        Self {
            trees: None,
            n_estimators: 100,
            max_samples: 256,
            sample_size: 0,
            max_depth: 8, // ceil(log2(256)) = 8
            random_state: None,
            n_features: 0,
            contamination: Contamination::Auto,
            offset: None,
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
    /// By default, `new` derives the tree depth as `ceil(log2(max_samples))` and seeds the
    /// forest non-deterministically. Override either with the builder methods below:
    ///
    /// - [`with_max_depth`](Self::with_max_depth) - maximum depth of each tree (returns
    ///   `Result` because it validates the depth)
    /// - [`with_random_state`](Self::with_random_state) - fixed RNG seed for reproducible forests
    /// - [`with_contamination`](Self::with_contamination) - inlier/outlier cutoff rule (default
    ///   [`Contamination::Auto`])
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

        let computed_max_depth = (max_samples as f64).log2().ceil() as usize;

        Ok(Self {
            trees: None,
            n_estimators,
            max_samples,
            sample_size: 0,
            max_depth: computed_max_depth,
            random_state: None,
            n_features: 0,
            contamination: Contamination::Auto,
            offset: None,
        })
    }

    /// Overrides the maximum depth of each isolation tree (default: `ceil(log2(max_samples))`)
    ///
    /// # Parameters
    ///
    /// - `max_depth` - Maximum depth of each tree. Must be greater than 0
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
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

    /// Sets a fixed RNG seed, so forest construction is reproducible (default: `None`,
    /// non-deterministic)
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

    /// Sets the inlier/outlier cutoff rule used by [`predict`](Self::predict)
    /// (default: [`Contamination::Auto`])
    ///
    /// `fit` resolves this rule into a concrete score cutoff, so changing it after fitting
    /// requires a new call to `fit`.
    ///
    /// # Parameters
    ///
    /// - `contamination` - [`Contamination::Auto`] or [`Contamination::Fraction`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if a `Fraction` is not finite or not in `(0.0, 0.5]`
    pub fn with_contamination(mut self, contamination: Contamination) -> Result<Self, Error> {
        if let Contamination::Fraction(c) = contamination
            && (!c.is_finite() || c <= 0.0 || c > 0.5)
        {
            return Err(Error::invalid_parameter(
                "contamination",
                format!("must be in (0.0, 0.5], got {c}"),
            ));
        }
        self.contamination = contamination;
        Ok(self)
    }

    // Getters
    get_field!(get_n_estimators, n_estimators, usize);
    get_field!(get_max_samples, max_samples, usize);
    get_field!(get_sample_size, sample_size, usize);
    get_field!(get_max_depth, max_depth, usize);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field!(get_n_features, n_features, usize);
    get_field!(get_contamination, contamination, Contamination);
    get_field!(get_offset, offset, Option<f64>);
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

        // Resolve the decision threshold against the training scores, so `predict` is a
        // fitted rule, not a per-batch quantile of whatever the caller hands it.
        self.offset = Some(match self.contamination {
            Contamination::Auto => AUTO_CONTAMINATION_OFFSET,
            Contamination::Fraction(c) => {
                // scikit-learn resolves the cutoff as `percentile(score_samples(X), 100 * c)`
                // with NumPy's default linear interpolation, so `offset` matches its `offset_`
                // numerically rather than merely flagging the same count
                let scores = self.score_samples(x)?;
                let mut sorted = scores.to_vec();
                sorted.sort_unstable_by(f64::total_cmp);

                let position = (sorted.len() - 1) as f64 * c;
                let lower = position.floor() as usize;
                let fraction = position - lower as f64;
                if fraction == 0.0 || lower + 1 >= sorted.len() {
                    sorted[lower]
                } else {
                    sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower])
                }
            }
        });

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
        // Stop when the recursion reaches max depth or the node has at most 1 sample
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

    /// Computes the path length of a sample through an isolation tree, with the average-path
    /// correction for samples unresolved at a leaf
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
    /// The single-sample form of [`score_samples`](Self::score_samples). It takes a slice
    /// instead of a matrix, so a streaming caller does not have to build a 1-row array. The
    /// score lies in `[-1, 0)`, and *lower* means more anomalous, exactly as in the batch form.
    ///
    /// # Parameters
    ///
    /// - `sample` - A single data sample as a slice of feature values
    ///
    /// # Returns
    ///
    /// - `Result<f64, Error>` - The anomaly score, in `[-1, 0)`
    ///
    /// # Errors
    ///
    /// Returns `Error` if:
    /// - Model has not been fitted
    /// - Sample feature dimension does not match training data
    pub fn score_sample(&self, sample: &[f64]) -> Result<f64, Error> {
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

    /// Computes the normalized anomaly score for a single sample, using a precomputed
    /// normalization constant `c_n = c(sample_size)`
    ///
    /// `c_n` is a parameter so batch prediction can compute it once and reuse it, instead of
    /// recomputing it for every sample.
    fn normalized_score(&self, sample: &[f64], trees: &[IsolationTree], c_n: f64) -> f64 {
        // Average path length across all trees. Each tree contributes 1 term, which stays
        // below any parallel reduction gate. The per-sample batch loop parallelizes one level up.
        let avg_path_length: f64 = trees
            .iter()
            .map(|tree| self.path_length(sample, tree, 0))
            .sum::<f64>()
            / trees.len() as f64;

        // A degenerate sub-sample of size <= 1 makes c(n) = 0, which is maximally anomalous
        if c_n <= 0.0 {
            return -1.0;
        }

        // Liu et al.'s score is s(x, n) = 2^(-E(h(x)) / c(n)), which rises toward 1 for more
        // anomalous samples. This function returns the negated score, matching scikit-learn's
        // convention where *lower* means more anomalous and negative marks the rejected class.
        -(2.0_f64.powf(-avg_path_length / c_n))
    }

    /// Computes the anomaly score for each sample
    ///
    /// Scores lie in `[-1, 0)`, and *lower* means more anomalous. The formula is
    /// `-(2^(-E[h(x)] / c(n)))`, exactly scikit-learn's `score_samples`. For the same values
    /// shifted by the fitted cutoff, use [`decision_function`](Self::decision_function). For a
    /// hard inlier/outlier label, use [`predict`](Self::predict).
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
    /// Parallelizes when the traversal work (samples * trees * average path length) clears
    /// the calibrated tree-traversal gate (see `crate::parallel_gates`)
    pub fn score_samples<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
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

    /// Shifts each sample's score by the fitted cutoff, so that negative means outlier
    ///
    /// `score_samples(x) - offset`, matching scikit-learn's `decision_function`. The
    /// [`offset`](Self::get_offset) is model state rather than a quantile of `x`. A given
    /// sample gets the same decision value whether the caller scores it alone or inside a
    /// larger batch.
    ///
    /// # Parameters
    ///
    /// - `x` - Input data matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<f64>, Error>` - Per-sample decision values. Negative marks an outlier
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` / `Error::EmptyInput` / `Error::DimensionMismatch` /
    ///   `Error::NonFinite` - propagated from [`score_samples`](Self::score_samples)
    pub fn decision_function<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // `score_samples` validates the fitted state, so `offset` is Some by the time this
        // function reads it.
        let scores = self.score_samples(x)?;
        let offset = self
            .offset
            .ok_or_else(|| Error::not_fitted("IsolationForest"))?;

        Ok(scores.mapv(|s| s - offset))
    }

    /// Classifies each sample as an inlier (`+1`) or outlier (`-1`)
    ///
    /// Exactly the sign of [`decision_function`](Self::decision_function): a sample is an outlier
    /// when its decision value is strictly negative. A sample landing exactly on the cutoff is
    /// therefore an inlier, as in scikit-learn
    ///
    /// # Parameters
    ///
    /// - `x` - Input data matrix where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - Per-sample labels: `-1` for outliers, `+1` for inliers
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` / `Error::EmptyInput` / `Error::DimensionMismatch` /
    ///   `Error::NonFinite` - propagated from [`decision_function`](Self::decision_function)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64>,
    {
        let decision = self.decision_function(x)?;

        Ok(decision.mapv(|d| if d < 0.0 { -1 } else { 1 }))
    }

    /// Trains the model on the dataset and immediately labels it
    ///
    /// This is a convenience method that combines `fit` and `predict` in a single call
    ///
    /// # Parameters
    ///
    /// - `x` - Training and prediction data as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<i32>, Error>` - Per-sample labels for the training data: `-1` for
    ///   outliers, `+1` for inliers
    ///
    /// # Errors
    ///
    /// Returns `Error` if fitting or prediction fails
    pub fn fit_predict<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(x)?;
        self.predict(x)
    }

    model_save_and_load_methods!(IsolationForest);
}

/// Unit tests for `average_path_length_factor`
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_average_path_length_factor_n0() {
        assert_abs_diff_eq!(average_path_length_factor(0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_path_length_factor_n1() {
        assert_abs_diff_eq!(average_path_length_factor(1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_path_length_factor_n2() {
        assert_abs_diff_eq!(average_path_length_factor(2), 1.0, epsilon = 1e-10);
    }

    /// c(3) uses the exact harmonic branch and equals 5/3
    #[test]
    fn test_average_path_length_factor_n3() {
        let expected = 5.0_f64 / 3.0;
        assert_abs_diff_eq!(average_path_length_factor(3), expected, epsilon = 1e-9);
    }

    /// Factor increases with n: c(10) < c(100) < c(1000)
    #[test]
    fn test_average_path_length_factor_monotone() {
        let f10 = average_path_length_factor(10);
        let f100 = average_path_length_factor(100);
        let f1000 = average_path_length_factor(1000);
        assert!(
            f10 < f100,
            "expected factor(10) < factor(100), got {f10} vs {f100}"
        );
        assert!(
            f100 < f1000,
            "expected factor(100) < factor(1000), got {f100} vs {f1000}"
        );
    }

    /// Factor stays strictly increasing across the n=50/51 switch from the exact harmonic sum
    /// to the ln(n-1)+gamma approximation
    #[test]
    fn test_average_path_length_factor_monotone_across_branch_boundary() {
        let f49 = average_path_length_factor(49); // exact-harmonic branch
        let f50 = average_path_length_factor(50); // exact-harmonic branch (last)
        let f51 = average_path_length_factor(51); // ln+gamma branch (first)
        let f52 = average_path_length_factor(52); // ln+gamma branch
        assert!(f49 < f50, "factor(49)={f49} should be < factor(50)={f50}");
        assert!(f50 < f51, "factor(50)={f50} should be < factor(51)={f51}");
        assert!(f51 < f52, "factor(51)={f51} should be < factor(52)={f52}");
    }

    /// For n>50 the ln(n-1)+gamma+1/(2(n-1)) approximation stays within a tight error bound of the
    /// exact value c(n) = 2*H_{n-1} - 2(n-1)/n. The `1/(2(n-1))` correction term shrinks the error
    /// from ~2e-2 (ln+gamma only) to ~1e-5, so a `1e-3` tolerance now comfortably holds
    #[test]
    fn test_average_path_length_factor_matches_exact_harmonic_within_tolerance() {
        for &n in &[51usize, 100, 1000] {
            let exact_h: f64 = (1..n).map(|i| 1.0 / i as f64).sum();
            let theoretical = 2.0 * exact_h - 2.0 * (n - 1) as f64 / n as f64;
            let got = average_path_length_factor(n);
            assert!(
                (got - theoretical).abs() < 1e-3,
                "n={n}: factor={got} deviates from theoretical {theoretical} by more than the documented approximation bound",
            );
        }
    }
}

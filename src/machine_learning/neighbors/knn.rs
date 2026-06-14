//! K-Nearest Neighbors (KNN) classification
//!
//! Provides the [`KNN`] classifier and the [`WeightingStrategy`] enum that controls how
//! neighbor votes are weighted

use crate::error::Error;
pub use crate::machine_learning::DistanceCalculationMetric;
use crate::machine_learning::spatial::KdTree;
use crate::machine_learning::validation::{
    check_is_fitted, preliminary_check, validate_predict_input,
};
use crate::math::matmul::{cache_resident, gemm_chunk_rows, gemm_internal};
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2, s};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::sync::OnceLock;

/// Feature-count ceiling for using the kd-tree neighbor index
///
/// Above this many features the tree no longer prunes effectively, so the brute-force search
/// is used instead. On uniform data (20k points, k = 8) the kd-tree beats the brute-force scan
/// up to d = 8 (2.6x at d = 8) and loses from d = 12 on (2.2-2.6x slower), so the ceiling sits
/// at the proven-win end of the 8-12 bracket. The boundary shifts with data distribution
/// (clustered data favors the tree) and dataset size, so this is a single-shape calibration,
/// not a universal constant
const KNN_KD_TREE_MAX_DIMS: usize = 8;

/// Selects the class with the greatest accumulated score (vote count or summed weight),
/// breaking ties in favor of the smallest class index
///
/// This keeps the prediction deterministic: `AHashMap` iteration order is randomized per
/// process, so a plain `max_by`/`max_by_key` would resolve tied classes differently across
/// runs. Tying by the smallest class index matches the conventional scikit-learn behavior
fn select_top_class<T>(scores: &AHashMap<usize, T>) -> Option<usize>
where
    T: PartialOrd + Copy,
{
    scores
        .iter()
        .max_by(|a, b| {
            let (sa, sb): (T, T) = (*a.1, *b.1);
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(a.0)) // smaller class index wins ties
        })
        .map(|(&idx, _)| idx)
}

/// Strategy used for weighting neighbors in the KNN algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WeightingStrategy {
    /// Each neighbor is weighted equally
    #[default]
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
/// use rustyml::machine_learning::KNN;
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
/// // Create KNN model with k=3 and default settings (uniform weighting, Euclidean distance)
/// let mut knn = KNN::new(3).unwrap();
///
/// // Fit the model
/// knn.fit(&x_train, &y_train).unwrap();
///
/// // Predict new samples
/// let x_test = array![
///     [1.5, 2.5],  // Closer to class "A" points
///     [5.5, 6.5]   // Closer to class "B" points
/// ];
///
/// let predictions = knn.predict(&x_test).unwrap();
/// println!("Predictions: {:?}", predictions);  // Prints ["A", "B"]
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

    /// Lazily-built kd-tree over the training data, accelerating neighbor search in low
    /// dimensions. Derived from `x_train`, so it is not serialized; it is rebuilt on first use
    /// after loading. `Some(None)` records "high-dimensional, use brute force" so the decision
    /// is made once
    #[serde(skip)]
    tree: OnceLock<Option<KdTree>>,
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
            tree: OnceLock::new(),
        }
    }
}

impl<T: Clone + std::hash::Hash + Eq> KNN<T> {
    /// Creates a new KNN classifier with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `k` - Number of neighbors to use for classification
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - A new KNN classifier instance
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `k` is 0
    ///
    /// # Notes
    ///
    /// Neighbor votes default to uniform weighting and the Euclidean distance
    /// metric. Override either with the builder methods below (`with_metric` returns
    /// `Result` because the Minkowski order is validated):
    ///
    /// - [`with_weighting_strategy`](Self::with_weighting_strategy) - uniform or distance-weighted votes
    /// - [`with_metric`](Self::with_metric) - distance metric: Euclidean, Manhattan, or Minkowski
    pub fn new(k: usize) -> Result<Self, Error> {
        if k == 0 {
            return Err(Error::invalid_parameter("k", "must be greater than 0"));
        }

        Ok(KNN {
            k,
            x_train: None,
            y_train_encoded: None,
            label_map: None,
            weighting_strategy: WeightingStrategy::Uniform,
            metric: DistanceCalculationMetric::Euclidean,
            tree: OnceLock::new(),
        })
    }

    /// Sets the weighting strategy for neighbor votes (default: [`WeightingStrategy::Uniform`])
    ///
    /// # Parameters
    ///
    /// - `weighting_strategy` - uniform or distance-weighted votes
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_weighting_strategy(mut self, weighting_strategy: WeightingStrategy) -> Self {
        self.weighting_strategy = weighting_strategy;
        self
    }

    /// Overrides the distance metric used for neighbor searches (default: Euclidean)
    ///
    /// # Parameters
    ///
    /// - `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - if Minkowski `p` is less than 1 or not finite; orders below 1
    ///   are not valid metrics and would break kd-tree pruning
    pub fn with_metric(mut self, metric: DistanceCalculationMetric) -> Result<Self, Error> {
        // Reject NaN/inf (not finite) and orders below 1, which break the triangle inequality
        if let DistanceCalculationMetric::Minkowski(p) = metric
            && (p < 1.0 || !p.is_finite())
        {
            return Err(Error::invalid_parameter(
                "p",
                format!("Minkowski p must be at least 1 and finite, got {}", p),
            ));
        }
        self.metric = metric;
        Ok(self)
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
        // Drop any kd-tree built from previous training data so it is rebuilt lazily on demand
        self.tree = OnceLock::new();

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

        // Neighbor index (kd-tree in low dimensions, else None for the brute-force fallback)
        let tree = self.neighbor_tree(x_train.view());
        // Training squared norms feed the brute-force Euclidean fast path
        let train_sq_norms = if tree.is_some() {
            None
        } else {
            self.euclidean_train_sq_norms(x_train)
        };

        // Sequential prediction on encoded indices
        let encoded_results: Result<Vec<usize>, Error> = if let Some(train_sq_norms) =
            train_sq_norms.as_ref()
        {
            if cache_resident::<f64>(x_train.nrows(), x_train.ncols()) {
                (0..x.nrows())
                    .map(|i| {
                        self.predict_one(
                            x.row(i),
                            x_train.view(),
                            y_train_encoded,
                            Some((train_sq_norms, None)),
                            tree,
                        )
                    })
                    .collect()
            } else {
                let chunk_rows = gemm_chunk_rows(x_train.nrows());
                let mut encoded = Vec::with_capacity(x.nrows());
                for chunk_start in (0..x.nrows()).step_by(chunk_rows) {
                    let chunk_end = (chunk_start + chunk_rows).min(x.nrows());
                    let projections =
                        gemm_internal(&x.slice(s![chunk_start..chunk_end, ..]), &x_train.t());
                    for i in chunk_start..chunk_end {
                        encoded.push(self.predict_one(
                            x.row(i),
                            x_train.view(),
                            y_train_encoded,
                            Some((train_sq_norms, Some(projections.row(i - chunk_start)))),
                            tree,
                        )?);
                    }
                }
                Ok(encoded)
            }
        } else {
            (0..x.nrows())
                .map(|i| self.predict_one(x.row(i), x_train.view(), y_train_encoded, None, tree))
                .collect()
        };

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

        // Neighbor index built once (single-threaded) before the parallel queries fan out
        let tree = self.neighbor_tree(x_train.view());
        // Training squared norms feed the brute-force Euclidean fast path; skip them when the
        // tree handles the search
        let train_sq_norms = if tree.is_some() {
            None
        } else {
            self.euclidean_train_sq_norms(x_train)
        };

        let encoded_results: Result<Vec<usize>, Error> = if let Some(train_sq_norms) =
            train_sq_norms.as_ref()
        {
            if cache_resident::<f64>(x_train.nrows(), x_train.ncols()) {
                (0..x.nrows())
                    .into_par_iter()
                    .map(|i| {
                        self.predict_one(
                            x.row(i),
                            x_train.view(),
                            y_train_encoded,
                            Some((train_sq_norms, None)),
                            tree,
                        )
                    })
                    .collect()
            } else {
                let chunk_rows = gemm_chunk_rows(x_train.nrows());
                let mut encoded = Vec::with_capacity(x.nrows());
                for chunk_start in (0..x.nrows()).step_by(chunk_rows) {
                    let chunk_end = (chunk_start + chunk_rows).min(x.nrows());
                    let projections =
                        gemm_internal(&x.slice(s![chunk_start..chunk_end, ..]), &x_train.t());
                    let chunk_results: Result<Vec<usize>, Error> = (chunk_start..chunk_end)
                        .into_par_iter()
                        .map(|i| {
                            self.predict_one(
                                x.row(i),
                                x_train.view(),
                                y_train_encoded,
                                Some((train_sq_norms, Some(projections.row(i - chunk_start)))),
                                tree,
                            )
                        })
                        .collect();
                    encoded.extend(chunk_results?);
                }
                Ok(encoded)
            }
        } else {
            (0..x.nrows())
                .into_par_iter()
                .map(|i| self.predict_one(x.row(i), x_train.view(), y_train_encoded, None, tree))
                .collect()
        };

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
    /// Precomputes per-training-sample squared norms for the Euclidean fast path,
    /// or `None` for metrics (Manhattan / Minkowski) that have no GEMV form
    ///
    /// Shared across every query, so the `||t||^2` term is paid once, not per query
    fn euclidean_train_sq_norms<S>(&self, x_train: &ArrayBase<S, Ix2>) -> Option<Array1<f64>>
    where
        S: Data<Elem = f64>,
    {
        match self.metric {
            DistanceCalculationMetric::Euclidean => {
                Some(x_train.map_axis(Axis(1), |row| row.dot(&row)))
            }
            _ => None,
        }
    }

    /// Returns the kd-tree neighbor index over the training data, building and caching it on
    /// first use. Returns `None` in high dimensions (where the tree no longer helps), so the
    /// caller falls back to the brute-force search
    fn neighbor_tree(&self, x_train: ArrayView2<f64>) -> Option<&KdTree> {
        self.tree
            .get_or_init(|| {
                if x_train.ncols() <= KNN_KD_TREE_MAX_DIMS {
                    Some(KdTree::build(x_train, self.metric))
                } else {
                    None
                }
            })
            .as_ref()
    }

    /// Predicts the encoded class index for a single data point
    ///
    /// `euclidean_fast` carries the brute-force Euclidean fast path's inputs: the training
    /// squared norms, plus this query's projection row `X_train . x` when the caller
    /// precomputed it through a chunked GEMM (the cache-overflow path); `None` in the second
    /// slot computes the projection here as one GEMV against the cache-resident training
    /// matrix. A `None` overall selects the kd-tree path or the per-pair metric scan
    fn predict_one(
        &self,
        x: ArrayView1<f64>,
        x_train: ArrayView2<f64>,
        y_train_encoded: &Array1<usize>,
        euclidean_fast: Option<(&Array1<f64>, Option<ArrayView1<f64>>)>,
        tree: Option<&KdTree>,
    ) -> Result<usize, Error> {
        let n_samples = x_train.nrows();
        let k = self.k.min(n_samples); // Ensure k doesn't exceed available samples

        let k_neighbors_owned: Vec<(f64, usize)> = if let Some(tree) = tree {
            tree.k_nearest(x, k)
                .into_iter()
                .map(|(idx, cmp)| (self.metric.distance_from_comparable(cmp), idx))
                .collect()
        } else {
            let mut distances: Vec<(f64, usize)> = match euclidean_fast {
                Some((train_sq_norms, precomputed)) => {
                    let x_sq = x.dot(&x);
                    let projections_owned;
                    let projections = match precomputed {
                        Some(p) => p,
                        None => {
                            projections_owned = x_train.dot(&x);
                            projections_owned.view()
                        }
                    };
                    projections
                        .iter()
                        .zip(train_sq_norms.iter())
                        .enumerate()
                        .map(|(i, (&proj, &t_sq))| {
                            let dist_sq = (x_sq + t_sq - 2.0 * proj).max(0.0);
                            (dist_sq.sqrt(), i)
                        })
                        .collect()
                }
                None => (0..n_samples)
                    .map(|i| (self.metric.distance(x, x_train.row(i)), i))
                    .collect(),
            };

            // Partial selection of the k smallest under the (distance, index) total order
            if k < distances.len() {
                distances
                    .select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            }
            distances.truncate(k);
            distances
        };
        let k_neighbors = &k_neighbors_owned[..];

        // Calculate based on weight strategy
        let result = match self.weighting_strategy {
            WeightingStrategy::Uniform => {
                let mut class_counts: AHashMap<usize, usize> = AHashMap::with_capacity(k);
                for &(_, idx) in k_neighbors {
                    *class_counts.entry(y_train_encoded[idx]).or_insert(0) += 1;
                }

                // Most common class, ties broken deterministically by smallest class index
                select_top_class(&class_counts).ok_or_else(|| {
                    Error::computation("No valid neighbors found for classification")
                })?
            }
            WeightingStrategy::Distance => {
                let exact_matches: AHashMap<usize, usize> = k_neighbors
                    .iter()
                    .filter(|&&(distance, _)| distance == 0.0)
                    .fold(AHashMap::new(), |mut acc, &(_, idx)| {
                        *acc.entry(y_train_encoded[idx]).or_insert(0) += 1;
                        acc
                    });

                if !exact_matches.is_empty() {
                    select_top_class(&exact_matches).ok_or_else(|| {
                        Error::computation("No valid neighbors found for classification")
                    })?
                } else {
                    let mut class_weights: AHashMap<usize, f64> = AHashMap::with_capacity(k);
                    for &(distance, idx) in k_neighbors {
                        *class_weights.entry(y_train_encoded[idx]).or_insert(0.0) += 1.0 / distance;
                    }

                    // Highest summed weight, ties broken deterministically by class index
                    select_top_class(&class_weights).ok_or_else(|| {
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

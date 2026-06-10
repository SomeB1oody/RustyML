pub use super::DistanceCalculationMetric;
use super::parallel::map_collect;
use super::validation::{preliminary_check, validate_predict_input};
use crate::error::{Context, Error};
use crate::{Deserialize, Serialize};
use ahash::AHashSet;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::collections::VecDeque;

/// Threshold for parallelization: only use parallel processing for larger datasets
const DBSCAN_PARALLEL_THRESHOLD: usize = 1000;

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm implementation
///
/// DBSCAN is a popular density-based clustering algorithm that can discover clusters of arbitrary shapes
/// without requiring the number of clusters to be specified beforehand.
///
/// # Fields
///
/// - `eps` - Neighborhood radius used to find neighbors
/// - `min_samples` - Minimum number of neighbors required to form a core point
/// - `metric` - Distance metric, options: Euclidean, Manhattan, Minkowski(p=3)
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::dbscan::DBSCAN;
/// use ndarray::Array2;
/// use rustyml::machine_learning::DistanceCalculationMetric;
///
/// let data = Array2::from_shape_vec((5, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     1.0, 1.0,
///     1.1, 1.1,
///     2.0, 2.0,
/// ]).unwrap();
///
/// let mut dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
/// let labels = dbscan.fit_predict(&data).unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DBSCAN {
    eps: f64,
    min_samples: usize,
    metric: DistanceCalculationMetric,
    labels: Option<Array1<isize>>,
    core_sample_indices: Option<Array1<usize>>,
    /// Core-point coordinates, stored so `predict` needs only the new data rather
    /// than the original training set.
    core_points: Option<Array2<f64>>,
    /// Cluster label of each stored core point (parallel to `core_points`).
    core_point_labels: Option<Array1<isize>>,
}

impl Default for DBSCAN {
    /// Default parameters for DBSCAN model
    ///
    /// # Default Values
    /// - eps = 0.5
    /// - min_samples = 5
    /// - metric = Euclidean
    fn default() -> Self {
        DBSCAN {
            eps: 0.5,
            min_samples: 5,
            metric: DistanceCalculationMetric::Euclidean,
            labels: None,
            core_sample_indices: None,
            core_points: None,
            core_point_labels: None,
        }
    }
}

impl DBSCAN {
    /// Creates a new DBSCAN instance with specified parameters
    ///
    /// # Parameters
    ///
    /// - `eps` - Neighborhood radius used to find neighbors
    /// - `min_samples` - Minimum number of neighbors required to form a core point
    /// - `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// - `Ok(DBSCAN)` - A new DBSCAN instance with the specified parameters
    /// - `Err(Error::InvalidParameter)` - If parameters are invalid (e.g., eps <= 0)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParameter` if `eps` is non-positive or not finite,
    /// if `min_samples` is 0, or if Minkowski `p` is non-positive or not finite.
    pub fn new(
        eps: f64,
        min_samples: usize,
        metric: DistanceCalculationMetric,
    ) -> Result<Self, Error> {
        // Validate eps parameter
        if eps <= 0.0 || !eps.is_finite() {
            return Err(Error::invalid_parameter(
                "eps",
                format!("eps must be positive and finite, got {}", eps),
            ));
        }

        // Validate min_samples parameter
        if min_samples == 0 {
            return Err(Error::invalid_parameter(
                "min_samples",
                "min_samples must be greater than 0",
            ));
        }

        // Validate metric parameter
        match metric {
            DistanceCalculationMetric::Minkowski(p) if (p <= 0.0 || !p.is_finite()) => {
                return Err(Error::invalid_parameter(
                    "p",
                    format!("Minkowski p must be positive and finite, got {}", p),
                ));
            }
            _ => {} // Euclidean and Manhattan don't need additional validation
        }

        Ok(DBSCAN {
            eps,
            min_samples,
            metric,
            labels: None,
            core_sample_indices: None,
            core_points: None,
            core_point_labels: None,
        })
    }

    // Getters
    get_field!(get_epsilon, eps, f64);
    get_field!(get_min_samples, min_samples, usize);
    get_field!(get_metric, metric, DistanceCalculationMetric);
    get_field_as_ref!(get_labels, labels, Option<&Array1<isize>>);
    get_field_as_ref!(
        get_core_sample_indices,
        core_sample_indices,
        Option<&Array1<usize>>
    );

    /// Find all neighbors of point `p` (points within eps distance)
    ///
    /// Uses parallelization for datasets larger than a threshold to improve performance
    fn region_query<S>(&self, data: &ArrayBase<S, Ix2>, p: usize) -> Result<Vec<usize>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Bounds check
        if p >= data.nrows() {
            return Err(Error::computation(format!(
                "Point index {} is out of bounds (max: {})",
                p,
                data.nrows() - 1
            )));
        }

        // Pre-compute row p (read-only view) to avoid fetching it repeatedly in each iteration
        let p_row = data.row(p);
        let eps = self.eps;
        let n_samples = data.nrows();

        let neighbors: Vec<usize> = if n_samples >= DBSCAN_PARALLEL_THRESHOLD {
            // Parallel iteration through all rows, calculating distances and filtering points that satisfy the eps condition
            (0..n_samples)
                .into_par_iter()
                .filter(|&q| {
                    let q_row = data.row(q);
                    let dist = self.metric.distance(p_row, q_row);
                    dist <= eps
                })
                .collect()
        } else {
            // Sequential iteration for smaller datasets
            (0..n_samples)
                .filter(|&q| {
                    let q_row = data.row(q);
                    let dist = self.metric.distance(p_row, q_row);
                    dist <= eps
                })
                .collect()
        };

        Ok(neighbors)
    }

    /// Performs DBSCAN clustering on the input data
    ///
    /// # Parameters
    /// - `data` - Input data as a reference 2D array in ndarray where each row is a sample
    ///
    /// # Returns
    /// - `Ok(&mut Self)` - The trained instance containing cluster labels and core sample indices
    /// - `Err(Error::EmptyInput)` - If the dataset is empty
    /// - `Err(Error::InvalidInput)` - If the dataset fails validation
    /// - `Err(Error::Computation)` - If numerical issues occur or cluster ID overflows
    ///
    /// # Errors
    /// - `Error::Computation` - If the number of discovered clusters exceeds `isize::MAX`
    ///
    /// # Performance
    /// Uses parallel processing for region queries if the number of samples is greater than or equal to 1000.
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        preliminary_check(data, None)?;

        // Check if dataset is empty
        let n_samples = data.nrows();

        let mut labels = Array1::from(vec![-1isize; n_samples]); // -1 represents unclassified or noise
        let mut core_samples = AHashSet::with_capacity(n_samples / 4); // Estimate 25% core samples
        let mut cluster_id = 0isize;

        // Initialize progress bar for tracking clustering progress
        #[cfg(feature = "show_progress")]
        let pb = {
            let progress = crate::create_progress_bar(
                n_samples as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Clusters: {msg}",
            );
            progress.set_message("0 | Core points: 0");
            progress
        };

        // Main loop processes each point sequentially, the algorithm as a whole remains sequential
        for p in 0..n_samples {
            #[cfg(feature = "show_progress")]
            pb.inc(1);
            if labels[p] != -1 {
                continue;
            }

            let neighbors = self.region_query(data, p).context("region query failed")?;

            if neighbors.len() < self.min_samples {
                labels[p] = -1; // Mark as noise
                continue;
            }

            // Start a new cluster
            labels[p] = cluster_id;
            core_samples.insert(p);
            let mut seeds: VecDeque<usize> = neighbors.into_iter().collect();

            // Expand cluster (the expansion process is still sequential)
            while let Some(q) = seeds.pop_front() {
                // If already processed in this cluster, skip
                if labels[q] == cluster_id {
                    continue;
                }

                // Assign to current cluster (could be noise or unvisited)
                labels[q] = cluster_id;

                let q_neighbors = self
                    .region_query(data, q)
                    .with_context(|| format!("region query failed for point {q}"))?;

                if q_neighbors.len() >= self.min_samples {
                    core_samples.insert(q);
                    for r in q_neighbors {
                        if labels[r] != cluster_id {
                            seeds.push_back(r);
                        }
                    }
                }
            }

            cluster_id += 1;

            // Update progress bar message with current statistics
            #[cfg(feature = "show_progress")]
            pb.set_message(format!(
                "{} | Core points: {}",
                cluster_id,
                core_samples.len()
            ));

            // Check for cluster_id overflow (the next increment would overflow)
            if cluster_id == isize::MAX {
                #[cfg(feature = "show_progress")]
                pb.finish_with_message("Error: cluster ID overflow");
                return Err(Error::computation("Too many clusters: cluster ID overflow"));
            }
        }

        // Finish progress bar with final statistics
        #[cfg(feature = "show_progress")]
        pb.finish_with_message(format!(
            "{} | Core points: {} | Noise points: {}",
            cluster_id,
            core_samples.len(),
            labels.iter().filter(|&&x| x == -1).count()
        ));

        // Convert HashSet to sorted Vec for consistent ordering
        let mut core_indices: Vec<usize> = core_samples.into_iter().collect();
        core_indices.sort_unstable();

        // Store core-point coordinates and their labels so `predict` only needs the
        // new data; the original training set is not retained.
        let n_features = data.ncols();
        let mut core_points = Array2::<f64>::zeros((core_indices.len(), n_features));
        let mut core_point_labels = Array1::<isize>::zeros(core_indices.len());
        for (i, &idx) in core_indices.iter().enumerate() {
            core_points.row_mut(i).assign(&data.row(idx));
            core_point_labels[i] = labels[idx];
        }

        self.labels = Some(labels);
        self.core_sample_indices = Some(Array1::from(core_indices));
        self.core_points = Some(core_points);
        self.core_point_labels = Some(core_point_labels);

        Ok(self)
    }

    /// Predicts cluster labels for new data points based on the trained model.
    ///
    /// Each new point is assigned to the cluster of its nearest core point when that
    /// core point is within `eps`; otherwise it is labeled as noise (`-1`). Only the
    /// core points found during `fit` are needed, so the original training set does
    /// not have to be passed in.
    ///
    /// # Parameters
    ///
    /// - `new_data` - New data points to classify, each row a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<isize>)` - Array of predicted cluster labels (`-1` for noise)
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted yet
    /// - `Error::DimensionMismatch` - If feature dimensions don't match
    /// - `Error::NonFinite` - If data contains non-finite values
    ///
    /// # Performance
    ///
    /// New points are scored in parallel when the number of samples is large.
    pub fn predict<S>(&self, new_data: &ArrayBase<S, Ix2>) -> Result<Array1<isize>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        // Ensure the model has been trained
        let core_points = self
            .core_points
            .as_ref()
            .ok_or_else(|| Error::not_fitted("DBSCAN"))?;
        let core_point_labels = self
            .core_point_labels
            .as_ref()
            .ok_or_else(|| Error::not_fitted("DBSCAN"))?;

        // Empty input yields an empty result
        if new_data.nrows() == 0 {
            return Ok(Array1::from(vec![]));
        }

        // Validate feature dimensions and finiteness against the fitted model
        validate_predict_input(new_data, core_points.ncols())?;

        // Assign each point to the cluster of its nearest core point within `eps`
        let predictions = map_collect(new_data.nrows(), DBSCAN_PARALLEL_THRESHOLD, |i| {
            let row = new_data.row(i);
            let mut min_dist = f64::MAX;
            let mut label = -1isize;

            for (j, core_row) in core_points.rows().into_iter().enumerate() {
                let dist = self.metric.distance(row, core_row);
                if dist < min_dist {
                    min_dist = dist;
                    label = core_point_labels[j];
                }
            }

            if min_dist <= self.eps { label } else { -1 }
        });

        Ok(Array1::from(predictions))
    }

    /// Performs clustering and returns the labels in one step
    ///
    /// # Parameters
    ///
    /// - `data` - Input data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<isize>)` - Array of cluster labels for each sample
    /// - `Err(Error)` - If fitting fails due to validation or processing errors
    ///
    /// # Performance
    ///
    /// Inherits parallelization behavior from the `fit` method.
    pub fn fit_predict<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<Array1<isize>, Error>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        self.fit(data)?;
        Ok(self.labels.clone().unwrap())
    }

    model_save_and_load_methods!(DBSCAN);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Calling the private `region_query` with a point index >= data.nrows()
    /// must return Error::Computation (the defensive bounds-check arm).
    ///
    /// Derivation: the guard at the top of region_query is `if p >= data.nrows()`.
    /// With a 3-row matrix, passing p = 3 makes `3 >= 3` true, so the code returns
    /// Err(Error::computation(...)), whose discriminant is the Computation variant.
    /// This arm is unreachable through the public fit/predict API (which only ever
    /// feeds indices drawn from 0..n_samples), so it is exercised here by calling
    /// the private method directly from an in-module test. The index 3 == nrows is
    /// chosen because the message formats `data.nrows() - 1`, which would underflow
    /// if nrows were 0; a non-empty matrix keeps the message well-formed.
    ///
    /// Note: Error::Computation is a STRUCT variant (`{ context, source }`), so it
    /// is matched with `{ .. }`, not as a tuple.
    #[test]
    fn region_query_out_of_bounds_index_gives_computation() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]; // 3 rows
        let model = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean).unwrap();
        let err = model.region_query(&data, 3).unwrap_err(); // 3 >= nrows (3)
        match err {
            Error::Computation { .. } => {}
            other => panic!("expected Computation, got {:?}", other),
        }
    }
}

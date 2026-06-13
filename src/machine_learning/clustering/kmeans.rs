//! K-Means clustering
//!
//! Provides the [`KMeans`] estimator, which partitions samples into k clusters
//! using k-means++ initialization and Lloyd's iteration

use crate::error::Error;
use crate::machine_learning::validation::{
    preliminary_check, validate_max_iterations, validate_predict_input, validate_tolerance,
};
use crate::math::matmul::gemm_internal;
use crate::math::reduction::{DET_REDUCE_BLOCK, det_reduce, det_reduce_range};
use crate::math::squared_euclidean_distance_row;
use crate::parallel_gates::{SCAN_F64_PARALLEL_MIN_ELEMS, SUM_F64_PARALLEL_MIN_ELEMS};
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix2};
use ndarray_rand::rand::Rng;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use std::ops::AddAssign;

/// KMeans clustering algorithm implementation
///
/// Partitions n observations into k clusters where each observation belongs to
/// the cluster with the nearest mean (centroid)
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::KMeans;
/// use ndarray::Array2;
/// use ndarray_rand::rand::random; // or `use rand::random;`
///
/// // Create a sample dataset with 100 points in 2D space
/// // The dataset contains 3 distinct clusters
/// let mut data = vec![];
///
/// // First cluster around (2.0, 2.0)
/// for _ in 0..30 {
///     data.push(2.0 + random::<f64>() * 0.5);
///     data.push(2.0 + random::<f64>() * 0.5);
/// }
///
/// // Second cluster around (8.0, 8.0)
/// for _ in 0..40 {
///     data.push(8.0 + random::<f64>() * 0.5);
///     data.push(8.0 + random::<f64>() * 0.5);
/// }
///
/// // Third cluster around (2.0, 8.0)
/// for _ in 0..30 {
///     data.push(2.0 + random::<f64>() * 0.5);
///     data.push(8.0 + random::<f64>() * 0.5);
/// }
///
/// let data = Array2::<f64>::from_shape_vec((100, 2), data).unwrap();
///
/// // Create a KMeans instance with 3 clusters
/// let mut kmeans = KMeans::new(
///     3,        // Number of clusters
///     300,      // Maximum iterations
///     1e-4,     // Convergence tolerance
///     Some(42)  // Random seed for reproducibility
/// ).unwrap();
///
/// // Fit the model to the data
/// kmeans.fit(&data).unwrap();
///
/// // Get cluster labels for all training samples
/// let labels = kmeans.get_labels().unwrap();
/// println!("Cluster labels: {:?}", labels);
///
/// // Get the computed centroids
/// let centroids = kmeans.get_centroids().unwrap();
/// println!("Cluster centroids:\n{:?}", centroids);
///
/// // Get the inertia (sum of squared distances to nearest centroid)
/// let inertia = kmeans.get_inertia().unwrap();
/// println!("Inertia: {:.4}", inertia);
///
/// // Get the number of iterations performed
/// let n_iter = kmeans.get_actual_iterations().unwrap();
/// println!("Iterations: {}", n_iter);
///
/// // Predict clusters for new data points
/// let new_data = Array2::<f64>::from_shape_vec((3, 2),
///     vec![2.1, 2.2,  // Close to first cluster
///          7.9, 8.1,  // Close to second cluster
///          2.2, 7.8]) // Close to third cluster
///     .unwrap();
///
/// let predicted_labels = kmeans.predict(&new_data).unwrap();
/// println!("Predicted labels for new data: {:?}", predicted_labels);
///
/// // Alternative: fit and predict in one step
/// let mut kmeans2 = KMeans::default(); // Uses default parameters
/// let labels = kmeans2.fit_predict(&data).unwrap();
/// println!("Labels from fit_predict: {:?}", labels);
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KMeans {
    /// Number of clusters to form
    n_clusters: usize,
    /// Maximum number of iterations for a single run
    max_iter: usize,
    /// Tolerance for declaring convergence
    tol: f64,
    /// Optional seed for random number generation
    random_state: Option<u64>,
    /// Computed cluster centers after fitting
    centroids: Option<Array2<f64>>,
    /// Cluster labels for training data after fitting
    labels: Option<Array1<usize>>,
    /// Sum of squared distances to the closest centroid after fitting
    inertia: Option<f64>,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
}

impl Default for KMeans {
    /// Creates a `KMeans` instance with default parameters
    ///
    /// # Default Values
    ///
    /// - `n_clusters` - 8
    /// - `max_iter` - 300
    /// - `tolerance` - 1e-4
    /// - `random_state` - None
    ///
    /// # Returns
    ///
    /// - `KMeans` - a new `KMeans` instance with default values
    fn default() -> Self {
        // Default parameters are always valid
        KMeans::new(8, 300, 1e-4, None).expect("Default KMeans parameters should be valid")
    }
}

impl KMeans {
    /// Creates a new KMeans instance with the specified parameters
    ///
    /// # Parameters
    ///
    /// - `n_clusters` - Number of clusters to form (must be greater than 0)
    /// - `max_iterations` - Maximum number of iterations for the algorithm (must be greater than 0)
    /// - `tolerance` - Convergence tolerance; the algorithm stops when centroids move less than this value
    /// - `random_state` - Optional seed for random number generation to ensure reproducibility
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new KMeans instance if parameters are valid
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n_clusters` or `max_iterations` is 0, or `tolerance` is non-positive/non-finite
    pub fn new(
        n_clusters: usize,
        max_iterations: usize,
        tolerance: f64,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        if n_clusters == 0 {
            return Err(Error::invalid_parameter(
                "n_clusters",
                "must be greater than 0",
            ));
        }

        validate_max_iterations(max_iterations)?;
        validate_tolerance(tolerance)?;

        Ok(KMeans {
            n_clusters,
            max_iter: max_iterations,
            tol: tolerance,
            random_state,
            centroids: None,
            labels: None,
            inertia: None,
            n_iter: None,
        })
    }

    // Getters
    get_field!(get_n_clusters, n_clusters, usize);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_labels, labels, Option<&Array1<usize>>);
    get_field!(get_inertia, inertia, Option<f64>);
    get_field_as_ref!(get_centroids, centroids, Option<&Array2<f64>>);

    /// Index of the closest centroid given a sample's projection row (`proj[j] = x . c_j`)
    ///
    /// Ranks centroids by `||c_j||^2 - 2 x.c_j`, which orders identically to the squared
    /// distance (the `||x||^2` term is constant per sample). The projections for all samples
    /// come from one block-parallel GEMM, so per-sample work is a plain scan
    fn argmin_centroid(proj_row: ArrayView1<f64>, centroid_sq_norms: &Array1<f64>) -> usize {
        let mut min_cluster = 0;
        let mut min_val = f64::MAX;
        for (cluster_idx, (&c_sq, &proj)) in
            centroid_sq_norms.iter().zip(proj_row.iter()).enumerate()
        {
            let val = c_sq - 2.0 * proj;
            if val < min_val {
                min_val = val;
                min_cluster = cluster_idx;
            }
        }
        min_cluster
    }

    /// Initializes cluster centroids using the k-means++ algorithm
    ///
    /// K-means++ provides better initialization than random selection by choosing
    /// initial centers that are spread out from each other, leading to better convergence
    ///
    /// # Parameters
    ///
    /// - `data` - Training data as a 2D array
    fn init_centroids<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<(), Error>
    where
        S: Data<Elem = f64>,
    {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Initialize cluster centers matrix
        let mut centroids = Array2::<f64>::zeros((self.n_clusters, n_features));

        let mut rng = crate::random::make_rng(self.random_state);

        // Randomly select the first center
        let first_center_idx = rng.random_range(0..n_samples);
        centroids.row_mut(0).assign(&data.row(first_center_idx));

        let init_parallel = n_samples.saturating_mul(n_features) >= SCAN_F64_PARALLEL_MIN_ELEMS;
        let mut min_dists: Vec<f64> = if init_parallel {
            data.outer_iter()
                .into_par_iter()
                .map(|sample| squared_euclidean_distance_row(&sample, &centroids.row(0)))
                .collect()
        } else {
            data.outer_iter()
                .map(|sample| squared_euclidean_distance_row(&sample, &centroids.row(0)))
                .collect()
        };

        // Select the remaining centers
        for k in 1..self.n_clusters {
            // Fold the most recently selected center into the running minima
            if k > 1 {
                let latest = centroids.row(k - 1);
                let fold_min = |(sample, min_dist): (ArrayView1<f64>, &mut f64)| {
                    let dist = squared_euclidean_distance_row(&sample, &latest);
                    if dist < *min_dist {
                        *min_dist = dist;
                    }
                };
                if init_parallel {
                    data.outer_iter()
                        .into_par_iter()
                        .zip(min_dists.par_iter_mut())
                        .for_each(fold_min);
                } else {
                    data.outer_iter()
                        .zip(min_dists.iter_mut())
                        .for_each(fold_min);
                }
            }
            let distances = &min_dists;

            // Deterministic blocked sum, on rayon above the sum gate; the roulette walk
            // below stays serial (prefix scan)
            let total_dist: f64 = det_reduce(
                distances,
                distances.len() >= SUM_F64_PARALLEL_MIN_ELEMS,
                |block| block.iter().sum::<f64>(),
                |a, b| a + b,
                0.0,
            );

            // All distances zero: fall back to random selection
            if total_dist == 0.0 {
                let random_idx = rng.random_range(0..n_samples);
                centroids.row_mut(k).assign(&data.row(random_idx));
                continue;
            }

            // Roulette wheel selection of the next center
            let mut cumulative_dist = 0.0;
            let choice = rng.random::<f64>() * total_dist;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative_dist += dist;
                if dist > 0.0 && cumulative_dist >= choice {
                    centroids.row_mut(k).assign(&data.row(i));
                    break;
                }
            }
        }

        self.centroids = Some(centroids);

        Ok(())
    }

    /// Fits the KMeans model to the training data
    ///
    /// Computes cluster centroids and assigns each data point to its closest centroid
    ///
    /// # Parameters
    ///
    /// - `data` - Training data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - A mutable reference to self for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the number of samples is less than `n_clusters`
    /// - `Error::EmptyInput` - If the data has no rows
    /// - `Error::NonFinite` - If the data contains NaN or infinite values
    ///
    /// # Performance
    ///
    /// The per-iteration assignment runs as one block-parallel GEMM; the arg-min scan
    /// parallelizes above the calibrated scan-class gate, and the centroid accumulation
    /// runs as a deterministic blocked fold above the sum gate (see
    /// `crate::parallel_gates`), so results are bitwise identical at any thread count
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
    where
        S: Data<Elem = f64>,
    {
        preliminary_check(data, None)?;

        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples < self.n_clusters {
            return Err(Error::invalid_input(
                "Number of samples is less than number of clusters",
            ));
        }

        self.init_centroids(data)?;

        let mut labels = Array1::<usize>::zeros(n_samples);
        let mut prev_inertia: Option<f64> = None;
        let mut iter_count = 0;

        // Pre-allocate the cluster-update buffers to avoid repeated allocations
        let mut new_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        let mut counts = vec![0usize; self.n_clusters];

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.max_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | Inertia: {msg}",
            );
            pb.set_message(format!("{:.6}", f64::INFINITY));
            pb
        };

        // Main iteration loop
        for i in 0..self.max_iter {
            // Squared centroid norms, shared by every sample this iteration
            let centroids = self.centroids.as_ref().unwrap();
            let centroid_sq_norms = centroids.map_axis(Axis(1), |row| row.dot(&row));

            let data_view = data.view();
            let projections = gemm_internal(&data_view, &centroids.t());

            // Closest cluster center and distance for a single sample
            let compute_assignments = |sample_idx: usize| -> Result<(usize, f64), Error> {
                let min_cluster =
                    Self::argmin_centroid(projections.row(sample_idx), &centroid_sq_norms);

                let dist = squared_euclidean_distance_row(
                    &data_view.row(sample_idx),
                    &centroids.row(min_cluster),
                );
                Ok((min_cluster, dist))
            };

            // Scan-class gate: n tasks, each an O(k) arg-min scan plus an O(d) exact distance
            let scan_work = n_samples.saturating_mul(self.n_clusters + n_features);
            let results: Result<Vec<(usize, f64)>, Error> =
                if scan_work >= SCAN_F64_PARALLEL_MIN_ELEMS {
                    (0..n_samples)
                        .into_par_iter()
                        .map(compute_assignments)
                        .collect()
                } else {
                    (0..n_samples).map(compute_assignments).collect()
                };

            let results = results?;

            // Fold every sample's row into its cluster's running sum, plus counts and
            // inertia, as a deterministic blocked range fold - on rayon above the sum gate
            // (work metric: samples x features; see benches/RESULTS.md
            // "k-means assign-accumulate")
            let n_clusters = self.n_clusters;
            let accumulate_parallel =
                n_samples.saturating_mul(n_features) >= SUM_F64_PARALLEL_MIN_ELEMS;
            let (sums, new_counts, inertia) = det_reduce_range(
                n_samples,
                accumulate_parallel,
                |range| {
                    let mut sums = Array2::<f64>::zeros((n_clusters, n_features));
                    let mut counts = vec![0usize; n_clusters];
                    let mut inertia = 0.0;
                    for i in range {
                        let (cluster, dist) = results[i];
                        inertia += dist;
                        sums.row_mut(cluster).add_assign(&data_view.row(i));
                        counts[cluster] += 1;
                    }
                    (sums, counts, inertia)
                },
                |(mut sums_a, mut counts_a, inertia_a), (sums_b, counts_b, inertia_b)| {
                    sums_a += &sums_b;
                    for (a, b) in counts_a.iter_mut().zip(counts_b) {
                        *a += b;
                    }
                    (sums_a, counts_a, inertia_a + inertia_b)
                },
                (
                    Array2::<f64>::zeros((n_clusters, n_features)),
                    vec![0usize; n_clusters],
                    0.0,
                ),
            );
            new_centroids.assign(&sums);
            counts.copy_from_slice(&new_counts);

            // The label store is a per-index map (no accumulation order to preserve);
            // chunking the parallel write like the fold keeps the pass cheap for small n
            let label_slice = labels
                .as_slice_mut()
                .expect("labels are freshly allocated and contiguous");
            if accumulate_parallel {
                label_slice
                    .par_chunks_mut(DET_REDUCE_BLOCK)
                    .zip(results.par_chunks(DET_REDUCE_BLOCK))
                    .for_each(|(label_block, result_block)| {
                        for (label, &(cluster, _)) in label_block.iter_mut().zip(result_block) {
                            *label = cluster;
                        }
                    });
            } else {
                for (label, &(cluster, _)) in label_slice.iter_mut().zip(&results) {
                    *label = cluster;
                }
            }

            #[cfg(feature = "show_progress")]
            {
                progress_bar.set_message(format!("{:.6}", inertia));
                progress_bar.inc(1);
            }

            // Check convergence condition
            if let Some(prev) = prev_inertia
                && (prev - inertia).abs() < self.tol * prev.max(self.tol)
            {
                iter_count = i + 1;
                self.inertia = Some(inertia);
                break;
            }
            prev_inertia = Some(inertia);
            iter_count = i + 1;

            // Average each cluster center in parallel
            new_centroids
                .outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(idx, mut centroid_row)| {
                    if counts[idx] > 0 {
                        let count_f = counts[idx] as f64;
                        centroid_row.par_mapv_inplace(|x| x / count_f);
                    }
                });

            // For each empty cluster, seed it with the point furthest from its assigned centroid
            for (cluster_idx, &count) in counts.iter().enumerate() {
                if count == 0 {
                    let result: Result<Option<usize>, Error> = results
                        .iter()
                        .enumerate()
                        .try_fold(
                            None,
                            |acc, (sample_idx, &(assigned_cluster, dist))| match acc {
                                None => Ok(Some((sample_idx, assigned_cluster, dist))),
                                Some((best_idx, best_cluster, best_dist)) => {
                                    if dist > best_dist {
                                        Ok(Some((sample_idx, assigned_cluster, dist)))
                                    } else {
                                        Ok(Some((best_idx, best_cluster, best_dist)))
                                    }
                                }
                            },
                        )
                        .map(|opt| opt.map(|(idx, _, _)| idx));

                    if let Some(farthest_idx) = result? {
                        new_centroids
                            .row_mut(cluster_idx)
                            .assign(&data.row(farthest_idx));
                    } else {
                        // No samples exist (should not happen): keep the old centroid
                        new_centroids
                            .row_mut(cluster_idx)
                            .assign(&self.centroids.as_ref().unwrap().row(cluster_idx));
                    }
                }
            }

            // Install the new centroids and recycle the previous buffer for the next iteration
            let previous = self.centroids.replace(new_centroids);
            new_centroids = previous.expect("centroids are initialized before the loop starts");
        }

        #[cfg(feature = "show_progress")]
        {
            let final_inertia = self.inertia.unwrap_or_else(|| prev_inertia.unwrap_or(0.0));
            let convergence_status = if iter_count < self.max_iter {
                "Converged"
            } else {
                "Max iterations"
            };
            progress_bar.finish_with_message(format!(
                "{:.6} | {} | Iterations: {}",
                final_inertia, convergence_status, iter_count
            ));
        }

        self.labels = Some(labels);
        // Set inertia if max_iter was reached without convergence
        if self.inertia.is_none() {
            self.inertia = prev_inertia;
        }
        self.n_iter = Some(iter_count);

        Ok(self)
    }

    /// Predicts the closest cluster for each sample in the input data
    ///
    /// # Parameters
    ///
    /// - `data` - New data points for which to predict cluster assignments
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, Error>` - An array of cluster indices for each input data point
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted yet
    /// - `Error::EmptyInput` - If the input data is empty
    /// - `Error::DimensionMismatch` - If the feature count does not match the fitted data
    /// - `Error::NonFinite` - If the input data contains NaN or infinite values
    pub fn predict<S>(&self, data: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, Error>
    where
        S: Data<Elem = f64>,
    {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| Error::not_fitted("KMeans"))?;
        validate_predict_input(data, centroids.ncols())?;

        let centroid_sq_norms = centroids.map_axis(Axis(1), |row| row.dot(&row));
        let projections = gemm_internal(data, &centroids.t());

        // Scan-class gate: n tasks, each an O(k) arg-min scan
        let scan_work = data.nrows().saturating_mul(centroids.nrows());
        let labels: Vec<usize> = if scan_work >= SCAN_F64_PARALLEL_MIN_ELEMS {
            (0..data.nrows())
                .into_par_iter()
                .map(|i| Self::argmin_centroid(projections.row(i), &centroid_sq_norms))
                .collect()
        } else {
            (0..data.nrows())
                .map(|i| Self::argmin_centroid(projections.row(i), &centroid_sq_norms))
                .collect()
        };

        Ok(Array1::from(labels))
    }

    /// Fits the model and predicts cluster indices for the input data
    ///
    /// Equivalent to calling `fit` followed by `predict`
    ///
    /// # Parameters
    ///
    /// - `data` - Training data as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, Error>` - An array of cluster indices for each input data point
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the number of samples is less than `n_clusters`
    /// - `Error::EmptyInput` - If the data has no rows
    /// - `Error::NonFinite` - If the data contains NaN or infinite values
    pub fn fit_predict<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.fit(data)?;
        Ok(self.labels.clone().unwrap())
    }

    model_save_and_load_methods!(KMeans);
}

use super::helper_function::{preliminary_check, validate_max_iterations, validate_tolerance};
use crate::error::ModelError;
use crate::math::squared_euclidean_distance_row;
use crate::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand::{RngCore, rng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::ops::AddAssign;

/// Threshold for parallelization in KMeans clustering.
/// When the number of samples is below this threshold, sequential processing is used.
/// When the number of samples is at or above this threshold, parallel processing is used.
const KMEANS_PARALLEL_THRESHOLD: usize = 1000;

/// KMeans clustering algorithm implementation.
///
/// This struct implements the K-Means clustering algorithm, which partitions
/// n observations into k clusters where each observation belongs to the cluster
/// with the nearest mean (centroid).
///
/// # Fields
///
/// - `n_clusters` - Number of clusters to form
/// - `max_iter` - Maximum number of iterations for a single run
/// - `tol` - Tolerance for declaring convergence
/// - `random_seed` - Optional seed for random number generation
/// - `centroids` - Computed cluster centers after fitting
/// - `labels` - Cluster labels for training data after fitting
/// - `inertia` - Sum of squared distances to the closest centroid after fitting
/// - `n_iter` - Number of iterations the algorithm ran for after fitting
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::kmeans::KMeans;
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
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    random_seed: Option<u64>,
    centroids: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    inertia: Option<f64>,
    n_iter: Option<usize>,
}

impl Default for KMeans {
    /// implement Default for KMeans
    ///
    /// # Default Values
    ///
    /// - `n_clusters` - 8
    /// - `max_iter` - 300
    /// - `tolerance` - 1e-4
    /// - `random_seed` - None
    ///
    /// # Returns
    ///
    /// * `KMeans` - a new `KMeans` instance with default values
    fn default() -> Self {
        // Default values are guaranteed to be valid, so unwrap is safe here
        KMeans::new(8, 300, 1e-4, None).expect("Default KMeans parameters should be valid")
    }
}

impl KMeans {
    /// Creates a new KMeans instance with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `n_clusters` - Number of clusters to form (must be greater than 0)
    /// - `max_iterations` - Maximum number of iterations for the algorithm (must be greater than 0)
    /// - `tolerance` - Convergence tolerance; the algorithm stops when centroids move less than this value
    /// - `random_seed` - Optional seed for random number generation to ensure reproducibility
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new KMeans instance if parameters are valid
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `n_clusters` or `max_iterations` is 0, or `tolerance` is non-positive/non-finite
    pub fn new(
        n_clusters: usize,
        max_iterations: usize,
        tolerance: f64,
        random_seed: Option<u64>,
    ) -> Result<Self, ModelError> {
        // Input validation
        if n_clusters == 0 {
            return Err(ModelError::InputValidationError(
                "n_clusters must be greater than 0".to_string(),
            ));
        }

        validate_max_iterations(max_iterations)?;
        validate_tolerance(tolerance)?;

        Ok(KMeans {
            n_clusters,
            max_iter: max_iterations,
            tol: tolerance,
            random_seed,
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
    get_field!(get_random_seed, random_seed, Option<u64>);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field_as_ref!(get_labels, labels, Option<&Array1<usize>>);
    get_field!(get_inertia, inertia, Option<f64>);
    get_field_as_ref!(get_centroids, centroids, Option<&Array2<f64>>);

    /// Finds the closest centroid to a given data point and returns its index and distance.
    ///
    /// # Parameters
    ///
    /// * `x` - Data point as a 2D array view
    ///
    /// # Returns
    ///
    /// * `(usize, f64)` - A tuple containing the index of the closest centroid and the squared distance to it
    fn closest_centroid(&self, x: &ArrayView2<f64>) -> Result<(usize, f64), ModelError> {
        let sample = x.row(0);
        let centroids = self.centroids.as_ref().unwrap();

        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (i, centroid) in centroids.outer_iter().enumerate() {
            let dist = squared_euclidean_distance_row(&sample, &centroid);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Ok((min_idx, min_dist))
    }

    /// Initializes cluster centroids using K-means++ algorithm.
    ///
    /// K-means++ provides better initialization than random selection by choosing
    /// initial centers that are spread out from each other, leading to better convergence.
    ///
    /// # Parameters
    ///
    /// * `data` - Training data as a 2D array
    fn init_centroids<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<(), ModelError>
    where
        S: Data<Elem = f64>,
    {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Initialize cluster centers matrix
        let mut centroids = Array2::<f64>::zeros((self.n_clusters, n_features));

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rng().next_u64()),
        };

        // K-means++ initialization method

        // Randomly select the first center point
        let first_center_idx = rng.random_range(0..n_samples);
        centroids.row_mut(0).assign(&data.row(first_center_idx));

        // Select the remaining center points
        for k in 1..self.n_clusters {
            // Calculate the distance from each point to the nearest center
            let distances: Vec<f64> = data
                .outer_iter()
                .into_par_iter()
                .map(|sample| {
                    // Find the closest already selected center point
                    centroids
                        .rows()
                        .into_iter()
                        .take(k)
                        .map(|centroid| squared_euclidean_distance_row(&sample, &centroid))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .fold(f64::MAX, f64::min)
                })
                .collect();

            let total_dist: f64 = distances.iter().sum();

            // Handle edge case where all distances are zero
            if total_dist == 0.0 {
                // Fallback to random selection
                let random_idx = rng.random_range(0..n_samples);
                centroids.row_mut(k).assign(&data.row(random_idx));
                continue;
            }

            // Use roulette wheel selection to choose the next center point
            let mut cumulative_dist = 0.0;
            let choice = rng.random::<f64>() * total_dist;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative_dist += dist;
                if cumulative_dist >= choice {
                    centroids.row_mut(k).assign(&data.row(i));
                    break;
                }
            }
        }

        self.centroids = Some(centroids);

        Ok(())
    }

    /// Fits the KMeans model to the training data.
    ///
    /// This method computes cluster centroids and assigns each data point to its closest centroid.
    ///
    /// # Parameters
    ///
    /// - `data` - Training data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A mutable reference to self for method chaining
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the number of samples is less than `n_clusters` or data contains invalid values
    ///
    /// # Performance
    ///
    /// Parallel processing is used when the number of samples is greater than or equal to 1000.
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64>,
    {
        preliminary_check(data, None)?;

        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples < self.n_clusters {
            return Err(ModelError::InputValidationError(
                "Number of samples is less than number of clusters".to_string(),
            ));
        }

        // Initialize cluster centers
        self.init_centroids(data)?;

        let mut labels = Array1::<usize>::zeros(n_samples);
        let mut prev_inertia: Option<f64> = None;
        let mut iter_count = 0;

        // Pre-allocate arrays for cluster updates to avoid repeated allocations
        let mut new_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        let mut counts = vec![0usize; self.n_clusters];

        // Create progress bar for clustering iterations
        let progress_bar = ProgressBar::new(self.max_iter as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Inertia: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message(format!("{:.6}", f64::INFINITY));

        // Main iteration loop
        for i in 0..self.max_iter {
            // Reset for this iteration
            new_centroids.fill(0.0);
            counts.fill(0);

            // Find the closest cluster center and distance for each sample
            let compute_assignments =
                |sample: ArrayView1<f64>| -> Result<(usize, f64), ModelError> {
                    let mut min_dist = f64::MAX;
                    let mut min_cluster = 0;

                    // Find closest centroid for this sample
                    for (cluster_idx, centroid) in
                        self.centroids.as_ref().unwrap().outer_iter().enumerate()
                    {
                        let dist = squared_euclidean_distance_row(&sample, &centroid);
                        if dist < min_dist {
                            min_dist = dist;
                            min_cluster = cluster_idx;
                        }
                    }

                    Ok((min_cluster, min_dist))
                };

            let results: Result<Vec<(usize, f64)>, ModelError> =
                if n_samples >= KMEANS_PARALLEL_THRESHOLD {
                    // Parallel computation for large datasets
                    data.outer_iter()
                        .into_par_iter()
                        .map(compute_assignments)
                        .collect()
                } else {
                    // Sequential computation for small datasets
                    data.outer_iter().map(compute_assignments).collect()
                };

            let results = results?;

            // Update labels and compute centroids in a single pass
            let mut inertia = 0.0;
            for (sample_idx, &(cluster_idx, dist)) in results.iter().enumerate() {
                labels[sample_idx] = cluster_idx;
                inertia += dist;

                // Accumulate sample contributions to new centroids
                let sample = data.row(sample_idx);
                new_centroids.row_mut(cluster_idx).add_assign(&sample);
                counts[cluster_idx] += 1;
            }

            // Update progress bar with current inertia
            progress_bar.set_message(format!("{:.6}", inertia));
            progress_bar.inc(1);

            // Check convergence condition
            if let Some(prev) = prev_inertia {
                if (prev - inertia).abs() < self.tol * prev.max(self.tol) {
                    iter_count = i + 1;
                    self.inertia = Some(inertia);
                    break;
                }
            }
            prev_inertia = Some(inertia);
            iter_count = i + 1;

            // Calculate the mean for each cluster center using parallel processing
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

            // Handle empty clusters: for empty clusters, select the point furthest from its assigned centroid
            for (cluster_idx, &count) in counts.iter().enumerate() {
                if count == 0 {
                    // Find the sample with maximum distance to its assigned centroid
                    let result: Result<Option<usize>, ModelError> = results
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
                        // Fallback: if no samples exist (shouldn't happen), keep the old centroid
                        new_centroids
                            .row_mut(cluster_idx)
                            .assign(&self.centroids.as_ref().unwrap().row(cluster_idx));
                    }
                }
            }

            self.centroids = Some(new_centroids);
            // Re-allocate for next iteration if needed
            new_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        }

        // Finish progress bar with final statistics
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

        self.labels = Some(labels);
        // Set inertia if not already set (i.e., if max_iter was reached without convergence)
        if self.inertia.is_none() {
            self.inertia = prev_inertia;
        }
        self.n_iter = Some(iter_count);

        println!(
            "\nKMeans clustering completed: {} samples, {} clusters, {} iterations, final inertia: {:.6}",
            n_samples, self.n_clusters, iter_count, final_inertia
        );

        Ok(self)
    }

    /// Predicts the closest cluster for each sample in the input data.
    ///
    /// # Parameters
    ///
    /// - `data` - New data points for which to predict cluster assignments
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, ModelError>` - An array of cluster indices for each input data point
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted yet
    /// - `ModelError::InputValidationError` - If input data is empty, contains invalid values, or has incorrect feature dimensions
    pub fn predict<S>(&self, data: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        if self.centroids.is_none() {
            return Err(ModelError::NotFitted);
        }

        // Check for empty input data
        if data.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        // Check for invalid values in input data
        if data.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        let n_features = data.shape()[1];

        // Verify feature dimensions match
        let expected_features = self.centroids.as_ref().unwrap().shape()[1];
        if n_features != expected_features {
            return Err(ModelError::InputValidationError(format!(
                "Feature dimension mismatch: expected {}, got {}",
                expected_features, n_features
            )));
        }

        let labels: Result<Vec<usize>, ModelError> = data
            .outer_iter()
            .into_par_iter()
            .map(|sample| {
                let sample_shaped = sample.to_shape((1, n_features)).map_err(|_| {
                    ModelError::InputValidationError(
                        "Failed to reshape sample during prediction".to_string(),
                    )
                })?;
                let sample_view = sample_shaped.view();
                let (closest_idx, _) = self.closest_centroid(&sample_view)?;
                Ok(closest_idx)
            })
            .collect();

        Ok(Array1::from(labels?))
    }

    /// Fits the model and predicts cluster indices for the input data.
    ///
    /// This is equivalent to calling `fit` followed by `predict`.
    ///
    /// # Parameters
    ///
    /// - `data` - Training data as a 2D array
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, ModelError>` - An array of cluster indices for each input data point
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data is invalid or smaller than the number of clusters
    pub fn fit_predict<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        self.fit(data)?;
        Ok(self.labels.clone().unwrap())
    }

    model_save_and_load_methods!(KMeans);
}

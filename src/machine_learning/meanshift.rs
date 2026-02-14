use super::helper_function::{preliminary_check, validate_max_iterations, validate_tolerance};
use crate::error::ModelError;
use crate::math::squared_euclidean_distance_row;
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Data, Ix2};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{SeedableRng, rng, seq::SliceRandom};
use rayon::prelude::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};

/// Threshold for determining when to use parallel processing.
/// Parallel processing is used only when the number of samples exceeds this threshold.
const MEANSHIFT_PARALLEL_THRESHOLD: usize = 1000;

/// Mean Shift clustering algorithm implementation.
///
/// Mean Shift is a centroid-based clustering algorithm that works by iteratively shifting
/// data points towards areas of higher density. Each data point moves in the direction of
/// the mean of points within its current window until convergence. The algorithm does not
/// require specifying the number of clusters in advance.
///
/// # Fields
///
/// # Fields
///
/// - `bandwidth` - The kernel bandwidth parameter that determines the search radius. Larger values lead to fewer clusters.
/// - `max_iter` - Maximum number of iterations to prevent infinite loops.
/// - `tol` - Convergence tolerance threshold. Points are considered converged when they move less than this value.
/// - `bin_seeding` - Whether to use bin seeding strategy for faster algorithm execution.
/// - `cluster_all` - Whether to assign all points to clusters, including potential noise.
/// - `n_samples_per_center` - Number of samples assigned to each cluster center.
/// - `cluster_centers` - The final cluster centers found by the algorithm.
/// - `labels` - Cluster labels assigned to each input sample.
/// - `n_iter` - The actual number of iterations performed during fitting.
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::meanshift::MeanShift;
/// use ndarray::Array2;
///
/// // Create a 2D dataset
/// let data = Array2::<f64>::from_shape_vec((10, 2),
///     vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1,
///          10.0, 10.0, 10.2, 9.9, 10.1, 10.0, 9.9, 9.8,
///          5.0, 5.0, 5.1, 4.9]).unwrap();
///
/// // Create a MeanShift instance with default parameters
/// let mut ms = MeanShift::default();
///
/// // Fit the model and predict cluster labels
/// let labels = ms.fit_predict(&data).unwrap();
///
/// // Get the cluster centers
/// let centers = ms.get_cluster_centers().clone().unwrap();
/// ```
///
/// # Notes
/// - If unsure about an appropriate bandwidth value, use the `estimate_bandwidth` function.
/// - The bandwidth parameter significantly affects algorithm performance and should be chosen carefully based on data characteristics.
/// - For large datasets, setting `bin_seeding = true` can improve performance.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MeanShift {
    bandwidth: f64,
    max_iter: usize,
    tol: f64,
    bin_seeding: bool,
    cluster_all: bool,
    n_samples_per_center: Option<Array1<usize>>,
    cluster_centers: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    n_iter: Option<usize>,
}

impl Default for MeanShift {
    /// Creates a new MeanShift instance with default parameter values.
    ///
    /// # Default Values
    ///
    /// - `bandwidth` - `1.0`. The kernel bandwidth parameter. A larger value will result in fewer clusters.
    /// - `max_iter` - `300`. Maximum number of iterations to prevent infinite loops.
    /// - `tol` - `1e-3`. Convergence tolerance threshold.
    /// - `bin_seeding` - `false`. Bin seeding is disabled by default.
    /// - `cluster_all` - `true`. All data points will be assigned to clusters by default.
    ///
    /// # Returns
    ///
    /// - `Self` - A new MeanShift instance with default parameters.
    fn default() -> Self {
        Self::new(1.0, None, None, None, None).expect("Default parameters should be valid")
    }
}

impl MeanShift {
    /// Creates a new MeanShift instance with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `bandwidth` - The bandwidth parameter that determines the size of the kernel. Must be positive and finite.
    /// - `max_iter` - The maximum number of iterations for the mean shift algorithm. Must be greater than 0.
    /// - `tol` - The convergence threshold for the algorithm. Must be positive and finite.
    /// - `bin_seeding` - Whether to use bin seeding for initialization.
    /// - `cluster_all` - Whether to assign all points to clusters, even those far from any centroid.
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A Result containing a new MeanShift instance or a ModelError.
    ///
    /// # Errors
    /// - `ModelError::InputValidationError` - If any parameter is invalid (e.g., non-positive bandwidth).
    pub fn new(
        bandwidth: f64,
        max_iter: Option<usize>,
        tol: Option<f64>,
        bin_seeding: Option<bool>,
        cluster_all: Option<bool>,
    ) -> Result<Self, ModelError> {
        // Validate bandwidth
        if bandwidth <= 0.0 || !bandwidth.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "bandwidth must be positive and finite, got {}",
                bandwidth
            )));
        }

        let max_iter_val = max_iter.unwrap_or(300);
        let tol_val = tol.unwrap_or(1e-3);

        // Validate max_iter
        validate_max_iterations(max_iter_val)?;

        // Validate tolerance
        validate_tolerance(tol_val)?;

        Ok(MeanShift {
            bandwidth,
            max_iter: max_iter_val,
            tol: tol_val,
            bin_seeding: bin_seeding.unwrap_or(false),
            cluster_all: cluster_all.unwrap_or(true),
            n_samples_per_center: None,
            cluster_centers: None,
            labels: None,
            n_iter: None,
        })
    }

    // Getters
    get_field!(get_bandwidth, bandwidth, f64);
    get_field_as_ref!(get_cluster_centers, cluster_centers, Option<&Array2<f64>>);
    get_field_as_ref!(get_labels, labels, Option<&Array1<usize>>);
    get_field_as_ref!(
        get_n_samples_per_center,
        n_samples_per_center,
        Option<&Array1<usize>>
    );
    get_field!(get_n_iter, n_iter, Option<usize>);
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_bin_seeding, bin_seeding, bool);
    get_field!(get_cluster_all, cluster_all, bool);

    /// Fits the MeanShift clustering model to the input data.
    ///
    /// # Parameters
    ///
    /// - `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, ModelError>` - A Result containing a mutable reference to the fitted model or a ModelError.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data fails preliminary checks.
    ///
    /// # Performance
    ///
    /// Parallel processing is used when the number of samples exceeds `MEANSHIFT_PARALLEL_THRESHOLD` (1000).
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, ModelError>
    where
        S: Data<Elem = f64> + Send + Sync,
    {
        preliminary_check(x, None)?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Initialize seed points
        let seeds: Vec<usize> = if self.bin_seeding {
            self.get_bin_seeds(x)
        } else {
            // Randomly select points as initial seeds
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = rng();
            indices.shuffle(&mut rng);
            // Limit number of seeds to avoid excessive computation
            let max_seeds = n_samples.min(100);
            indices[..max_seeds].to_vec()
        };

        // Pre-compute gamma for efficiency
        let gamma = 1.0 / (2.0 * self.bandwidth.powi(2));
        let tol_squared = self.tol * self.tol;
        let bandwidth_squared = self.bandwidth * self.bandwidth;

        // Determine whether to use parallel processing
        let use_parallel = n_samples > MEANSHIFT_PARALLEL_THRESHOLD;

        // Helper function for mean shift iteration on a single seed
        let process_seed = |seed_idx: usize| -> Result<(Array1<f64>, usize), ModelError> {
            let mut center = x.row(seed_idx).to_owned();
            let mut completed_iterations = 0;

            loop {
                let mut new_center = Array1::zeros(n_features);
                let mut weight_sum = 0.0;

                // Calculate distances and weights
                let weights: Result<Vec<f64>, ModelError> = if use_parallel {
                    (0..n_samples)
                        .into_par_iter()
                        .map(|i| {
                            let point = x.row(i);
                            let dist = squared_euclidean_distance_row(&center, &point);
                            Ok((-gamma * dist).exp())
                        })
                        .collect()
                } else {
                    (0..n_samples)
                        .map(|i| {
                            let point = x.row(i);
                            let dist = squared_euclidean_distance_row(&center, &point);
                            Ok((-gamma * dist).exp())
                        })
                        .collect()
                };

                let weights = weights?;

                // Calculate weighted average (optimized accumulation)
                for (i, &weight) in weights.iter().enumerate() {
                    if weight > 0.0 {
                        let point = x.row(i);
                        for j in 0..n_features {
                            new_center[j] += point[j] * weight;
                        }
                        weight_sum += weight;
                    }
                }

                // Normalize
                if weight_sum > 0.0 {
                    new_center.mapv_inplace(|x| x / weight_sum);
                }

                // Check convergence using squared distance to avoid sqrt
                let shift_squared = squared_euclidean_distance_row(&center, &new_center);
                center = new_center;

                completed_iterations += 1;

                if shift_squared < tol_squared || completed_iterations >= self.max_iter {
                    break;
                }
            }

            Ok((center, completed_iterations))
        };

        // Create progress bar for seed processing
        let progress_bar = ProgressBar::new(seeds.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} seeds | {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message("Processing seeds...");

        // Process mean shift for each seed point
        let results: Result<Vec<(Array1<f64>, usize)>, ModelError> = if use_parallel {
            let results: Result<Vec<_>, _> = seeds
                .par_iter()
                .map(|&seed_idx| {
                    let result = process_seed(seed_idx);
                    progress_bar.inc(1);
                    result
                })
                .collect();
            results
        } else {
            seeds
                .iter()
                .map(|&seed_idx| {
                    let result = process_seed(seed_idx);
                    progress_bar.inc(1);
                    result
                })
                .collect()
        };

        let results = results?;
        progress_bar.finish_with_message("All seeds processed");

        // Extract centers and calculate actual max iterations
        let centers: Vec<Array1<f64>> = results.iter().map(|(c, _)| c.clone()).collect();
        let max_actual_iter = results.iter().map(|(_, i)| *i).max().unwrap_or(0);
        self.n_iter = Some(max_actual_iter);

        // Merge similar centers with optimized clustering
        let mut unique_centers: Vec<Array1<f64>> = Vec::with_capacity(centers.len());
        let mut center_counts: Vec<usize> = Vec::with_capacity(centers.len());

        for center in centers {
            let mut merged = false;

            // Find closest existing center within bandwidth
            for (i, unique_center) in unique_centers.iter_mut().enumerate() {
                let distance_squared = squared_euclidean_distance_row(&center, &unique_center);

                if distance_squared < bandwidth_squared {
                    // Update existing center using weighted average
                    let count = center_counts[i];
                    let new_count = count + 1;
                    let weight_old = count as f64 / new_count as f64;
                    let weight_new = 1.0 / new_count as f64;

                    // Use parallel update for the center coordinates
                    unique_center.zip_mut_with(&center, |old, &new| {
                        *old = *old * weight_old + new * weight_new;
                    });

                    center_counts[i] = new_count;
                    merged = true;
                    break;
                }
            }

            if !merged {
                unique_centers.push(center);
                center_counts.push(1);
            }
        }

        // Create cluster_centers array
        let n_clusters = unique_centers.len();
        let mut cluster_centers = Array2::zeros((n_clusters, n_features));
        for (i, center) in unique_centers.iter().enumerate() {
            cluster_centers.row_mut(i).assign(center);
        }

        // Helper function for finding nearest cluster label
        let find_label = |i: usize| -> Result<usize, ModelError> {
            let point = x.row(i);
            let mut min_dist_squared = f64::INFINITY;
            let mut label = 0;

            for (j, center) in unique_centers.iter().enumerate() {
                let dist_squared = squared_euclidean_distance_row(&point, &center);
                if dist_squared < min_dist_squared {
                    min_dist_squared = dist_squared;
                    label = j;
                }
            }

            // If not cluster_all and distance is too far, mark as outlier
            if !self.cluster_all && min_dist_squared > bandwidth_squared {
                Ok(n_clusters) // Use n_clusters as outlier label
            } else {
                Ok(label)
            }
        };

        // Assign cluster labels to each data point
        let labels: Result<Vec<usize>, ModelError> = if use_parallel {
            (0..n_samples).into_par_iter().map(find_label).collect()
        } else {
            (0..n_samples).map(find_label).collect()
        };

        let labels = labels?;

        self.cluster_centers = Some(cluster_centers);
        self.labels = Some(Array1::from(labels));
        self.n_samples_per_center = Some(Array1::from(center_counts));

        // Calculate cost using kernel density estimation
        let calculate_cost = |x: &ArrayBase<S, Ix2>,
                              centers: &[Array1<f64>],
                              bandwidth: f64,
                              use_parallel: bool|
         -> Result<f64, ModelError> {
            let n_samples = x.nrows();
            let gamma = 1.0 / (2.0 * bandwidth * bandwidth);

            // Helper function to compute log-likelihood for a single point
            let compute_point_likelihood = |i: usize| -> Result<f64, ModelError> {
                let point = x.row(i);
                // Sum kernel values from all centers
                let kernel_sum: Result<f64, ModelError> = centers
                    .iter()
                    .map(|center| -> Result<f64, ModelError> {
                        let dist_squared = squared_euclidean_distance_row(&point, &center);
                        Ok((-gamma * dist_squared).exp())
                    })
                    .sum();

                let kernel_sum = kernel_sum?;

                // Avoid log(0) by clamping to minimum density
                let density = (kernel_sum / centers.len() as f64).max(1e-15);
                Ok(density.ln())
            };

            // Calculate the negative log-likelihood
            let total_log_likelihood: Result<f64, ModelError> = if use_parallel {
                (0..n_samples)
                    .into_par_iter()
                    .map(compute_point_likelihood)
                    .sum()
            } else {
                (0..n_samples).map(compute_point_likelihood).sum()
            };

            let total_log_likelihood = total_log_likelihood?;

            // Return negative log-likelihood as cost (higher is worse)
            Ok(-total_log_likelihood / n_samples as f64)
        };

        let cost = calculate_cost(x, &unique_centers, self.bandwidth, use_parallel)?;

        // Print training info
        println!(
            "\nMean Shift training completed: {} samples, {} features, {} seeds processed, {} clusters found, max iterations: {}, cost: {:.6}",
            n_samples,
            n_features,
            seeds.len(),
            n_clusters,
            max_actual_iter,
            cost
        );

        Ok(self)
    }

    /// Predicts cluster labels for the input data.
    ///
    /// # Parameters
    ///
    /// - `x` - The input data as a ndarray `ArrayView2<f64>` where each row is a sample.
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, ModelError>` - A Result containing the predicted cluster labels or a ModelError.
    ///
    /// # Errors
    ///
    /// - `ModelError::NotFitted` - If the model has not been fitted yet.
    /// - `ModelError::InputValidationError` - If input data is invalid or dimensions don't match training data.
    ///
    /// # Performance
    ///
    /// Parallel processing is used when the number of samples exceeds `MEANSHIFT_PARALLEL_THRESHOLD` (1000).
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, ModelError> {
        // Check if model has been fitted
        if self.cluster_centers.is_none() {
            return Err(ModelError::NotFitted);
        }

        let centers = self.cluster_centers.as_ref().unwrap();

        // Check for empty input data
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Cannot predict on empty dataset".to_string(),
            ));
        }

        // Check if input feature dimensions match training data
        if x.shape()[1] != centers.shape()[1] {
            return Err(ModelError::InputValidationError(format!(
                "Number of features does not match training data, x columns: {}, training features: {}",
                x.shape()[1],
                centers.shape()[1]
            )));
        }

        // Check for invalid values in input data
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        let n_samples = x.shape()[0];
        let n_clusters = centers.shape()[0];
        let bandwidth_squared = self.bandwidth * self.bandwidth;

        // Determine whether to use parallel processing
        let use_parallel = n_samples > MEANSHIFT_PARALLEL_THRESHOLD;

        // Helper function for finding nearest cluster
        let find_nearest = |i: usize| -> Result<usize, ModelError> {
            let point = x.row(i);
            let mut min_dist_squared = f64::INFINITY;
            let mut label = 0;

            for j in 0..n_clusters {
                let center = centers.row(j);
                let dist_squared = squared_euclidean_distance_row(&point, &center);
                if dist_squared < min_dist_squared {
                    min_dist_squared = dist_squared;
                    label = j;
                }
            }

            // If not cluster_all and distance is too far, mark as outlier
            if !self.cluster_all && min_dist_squared > bandwidth_squared {
                Ok(n_clusters) // Use n_clusters as outlier label
            } else {
                Ok(label)
            }
        };

        // Process all samples with optional parallelization
        let labels: Result<Vec<usize>, ModelError> = if use_parallel {
            (0..n_samples).into_par_iter().map(find_nearest).collect()
        } else {
            (0..n_samples).map(find_nearest).collect()
        };

        Ok(Array1::from(labels?))
    }

    /// Fits the model to the input data and predicts cluster labels.
    ///
    /// # Parameters
    ///
    /// - `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, ModelError>` - A Result containing the predicted cluster labels or a ModelError.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If input data fails preliminary checks.
    /// - `ModelError::NotFitted` - If fitting succeeded but labels were not set correctly.
    pub fn fit_predict<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, ModelError>
    where
        S: Data<Elem = f64> + Sync + Send,
    {
        self.fit(x)?;
        self.labels.clone().ok_or(ModelError::NotFitted)
    }

    /// Estimates the bandwidth to use with the MeanShift algorithm.
    ///
    /// The bandwidth is estimated based on the pairwise distances between a subset of points.
    ///
    /// # Parameters
    ///
    /// - `x` - The input data as a ndarray `ArrayView2<f64>` where each row is a sample.
    /// - `quantile` - The quantile of the pairwise distances to use as the bandwidth.
    /// - `n_samples` - The number of samples to use for the distance calculation.
    /// - `random_state` - Seed for random number generation.
    ///
    /// # Returns
    ///
    /// - `Result<f64, ModelError>` - A Result containing the estimated bandwidth or a ModelError.
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `quantile` is not in range \[0, 1\].
    fn get_bin_seeds<S>(&self, x: &ArrayBase<S, Ix2>) -> Vec<usize>
    where
        S: Data<Elem = f64> + Sync + Send,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Calculate min for each feature in parallel
        let mins: Vec<f64> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let col = x.column(j);
                col.fold(f64::INFINITY, |a, &b| a.min(b))
            })
            .collect();

        // Create grid - this part is harder to parallelize due to shared HashMap
        let bin_size = self.bandwidth;

        // Create thread-safe structures for parallel processing
        let bins_mutex = std::sync::Mutex::new(AHashMap::<Vec<i64>, Vec<usize>>::new());

        // Assign points to bins in parallel
        (0..n_samples).into_par_iter().for_each(|i| {
            let point = x.row(i);
            let mut bin_index = Vec::with_capacity(n_features);

            for j in 0..n_features {
                let idx = ((point[j] - mins[j]) / bin_size).floor() as i64;
                bin_index.push(idx);
            }

            // Lock the HashMap only when updating
            let mut bins = bins_mutex.lock().unwrap();
            bins.entry(bin_index).or_insert_with(Vec::new).push(i);
        });

        // Get the final HashMap
        let bins = bins_mutex.into_inner().unwrap();

        // Select one point from each grid cell as seed
        let mut seeds = Vec::new();
        for (_, indices) in bins {
            if !indices.is_empty() {
                seeds.push(indices[0]);
            }
        }

        seeds
    }

    model_save_and_load_methods!(MeanShift);
}

/// Estimates the bandwidth to use with the MeanShift algorithm.
///
/// The bandwidth is estimated based on the pairwise distances between a subset of points.
///
/// # Parameters
///
/// - `x` - The input data as a ndarray `ArrayView2<f64>` where each row is a sample.
/// - `quantile` - The quantile of the pairwise distances to use as the bandwidth.
/// - `n_samples` - The number of samples to use for the distance calculation.
/// - `random_state` - Seed for random number generation.
///
/// # Returns
///
/// - `Ok(f64)` - The estimated bandwidth
/// - `Err(ModelError::InputValidationError)` - quantile is not in range \[0, 1\]
pub fn estimate_bandwidth<S>(
    x: &ArrayBase<S, Ix2>,
    quantile: Option<f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> Result<f64, ModelError>
where
    S: Data<Elem = f64>,
{
    let quantile = quantile.unwrap_or(0.3);
    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(ModelError::InputValidationError(
            "quantile should be in range [0, 1]".to_string(),
        ));
    }

    let (n_samples_total, _) = x.dim();
    let n_samples = n_samples.unwrap_or(n_samples_total);

    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    // If we have fewer samples than requested, use all samples
    let x_samples = if n_samples >= n_samples_total {
        x.to_owned()
    } else {
        // Random sampling
        let mut indices: Vec<usize> = (0..n_samples_total).collect();
        indices.shuffle(&mut rng);
        let indices = &indices[..n_samples];

        let mut samples = Array2::zeros((n_samples, x.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&x.row(idx));
        }
        samples
    };

    // Create all possible pairs of indices (i,j) where i < j
    let pairs: Vec<(usize, usize)> = (0..n_samples)
        .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
        .collect();

    // Compute distances between all pairs of points in parallel
    let distances: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let point_i = x_samples.row(i);
            let point_j = x_samples.row(j);

            // Euclidean distance
            point_i
                .iter()
                .zip(point_j.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    // Sort distances and select the value at the specified quantile
    let mut distances = distances;
    distances.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let k = (distances.len() as f64 * quantile) as usize;
    Ok(distances.get(k).copied().unwrap_or(0.0))
}

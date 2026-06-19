//! Mean Shift clustering
//!
//! Provides the [`MeanShift`] estimator, which finds clusters by iteratively shifting
//! points toward higher-density regions without requiring the number of clusters up front,
//! plus the [`estimate_bandwidth`] helper for choosing a bandwidth from the data

use crate::error::Error;
use crate::machine_learning::parallel::map_collect;
use crate::machine_learning::validation::{
    preliminary_check, validate_max_iterations, validate_predict_input, validate_tolerance,
};
use crate::math::matmul::{cache_resident, gemm_chunk_rows, gemm_par_auto};
use crate::math::squared_euclidean_distance_row;
use crate::parallel_gates::scan_f64_parallel_min_elems;
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip, s};
use ndarray_rand::rand::seq::SliceRandom;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Mean Shift clustering algorithm
///
/// A centroid-based clustering algorithm that iteratively shifts data points toward areas of
/// higher density. Each point moves in the direction of the mean of points within its current
/// window until convergence. The number of clusters need not be specified in advance
///
/// # Examples
///
/// ```rust
/// use rustyml::machine_learning::MeanShift;
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
///
/// - If unsure about an appropriate bandwidth value, use the `estimate_bandwidth` function
/// - The bandwidth parameter significantly affects algorithm performance and should be chosen carefully based on data characteristics
/// - For large datasets, setting `bin_seeding = true` can improve performance
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MeanShift {
    /// Kernel bandwidth that sets the search radius; larger values lead to fewer clusters
    bandwidth: f64,
    /// Maximum number of iterations to prevent infinite loops
    max_iter: usize,
    /// Convergence tolerance; a point is converged when it moves less than this value
    tol: f64,
    /// Whether to use the bin-seeding strategy for faster execution
    bin_seeding: bool,
    /// Whether to assign all points to clusters, including potential noise
    cluster_all: bool,
    /// Number of samples assigned to each cluster center
    n_samples_per_center: Option<Array1<usize>>,
    /// Final cluster centers found by the algorithm
    cluster_centers: Option<Array2<f64>>,
    /// Cluster labels assigned to each input sample
    labels: Option<Array1<usize>>,
    /// Actual number of iterations performed during fitting
    n_iter: Option<usize>,
}

impl Default for MeanShift {
    /// Creates a new MeanShift instance with default parameter values
    ///
    /// # Default Values
    ///
    /// - `bandwidth` - `1.0`; a larger value results in fewer clusters
    /// - `max_iter` - `300`; maximum number of iterations to prevent infinite loops
    /// - `tol` - `1e-3`; convergence tolerance threshold
    /// - `bin_seeding` - `false`; bin seeding is disabled by default
    /// - `cluster_all` - `true`; all data points are assigned to clusters by default
    ///
    /// # Returns
    ///
    /// - `Self` - A new MeanShift instance with default parameters
    fn default() -> Self {
        Self::new(1.0).expect("Default parameters should be valid")
    }
}

impl MeanShift {
    /// Creates a new MeanShift instance with the specified bandwidth
    ///
    /// `bandwidth` is the dominant hyperparameter. The remaining settings have sensible
    /// defaults and are tuned afterwards through the builder methods listed in the Notes
    ///
    /// # Parameters
    ///
    /// - `bandwidth` - Bandwidth that determines the size of the kernel; must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new MeanShift instance, or an Error
    ///
    /// # Notes
    ///
    /// Lower-priority settings are configured after construction. The convergence-related
    /// setters validate their input and return `Result`. The boolean toggles return `Self`:
    ///
    /// - [`with_max_iter`](Self::with_max_iter) - maximum iterations (default: `300`)
    /// - [`with_tolerance`](Self::with_tolerance) - convergence tolerance (default: `1e-3`)
    /// - [`with_bin_seeding`](Self::with_bin_seeding) - bin-seeding for faster init (default: `false`)
    /// - [`with_cluster_all`](Self::with_cluster_all) - assign all points to clusters (default: `true`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `bandwidth` is non-positive or not finite
    pub fn new(bandwidth: f64) -> Result<Self, Error> {
        if bandwidth <= 0.0 || !bandwidth.is_finite() {
            return Err(Error::invalid_parameter(
                "bandwidth",
                format!("must be positive and finite, got {}", bandwidth),
            ));
        }

        Ok(MeanShift {
            bandwidth,
            max_iter: 300,
            tol: 1e-3,
            bin_seeding: false,
            cluster_all: true,
            n_samples_per_center: None,
            cluster_centers: None,
            labels: None,
            n_iter: None,
        })
    }

    /// Sets the maximum number of iterations (default: `300`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `max_iter` is 0
    pub fn with_max_iter(mut self, max_iter: usize) -> Result<Self, Error> {
        validate_max_iterations(max_iter)?;
        self.max_iter = max_iter;
        Ok(self)
    }

    /// Sets the convergence tolerance (default: `1e-3`)
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `tol` is non-positive or not finite
    pub fn with_tolerance(mut self, tol: f64) -> Result<Self, Error> {
        validate_tolerance(tol)?;
        self.tol = tol;
        Ok(self)
    }

    /// Enables or disables the bin-seeding initialization strategy (default: `false`)
    ///
    /// Setting this to `true` can speed up fitting on large datasets
    pub fn with_bin_seeding(mut self, bin_seeding: bool) -> Self {
        self.bin_seeding = bin_seeding;
        self
    }

    /// Sets whether to assign every point to a cluster, including potential noise
    /// (default: `true`)
    pub fn with_cluster_all(mut self, cluster_all: bool) -> Self {
        self.cluster_all = cluster_all;
        self
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
    get_field!(get_actual_iterations, n_iter, Option<usize>);
    get_field!(get_max_iterations, max_iter, usize);
    get_field!(get_tolerance, tol, f64);
    get_field!(get_bin_seeding, bin_seeding, bool);
    get_field!(get_cluster_all, cluster_all, bool);

    /// Fits the MeanShift clustering model to the input data
    ///
    /// # Parameters
    ///
    /// - `x` - The input data where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<&mut Self, Error>` - A mutable reference to the fitted model, or an Error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If input data fails preliminary checks
    ///
    /// # Performance
    ///
    /// Parallelizes when the total scan work clears the calibrated scan-class gate (see
    /// `crate::parallel_gates`)
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<&mut Self, Error>
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
            (0..n_samples).collect()
        };

        // RBF weighting: w_i = exp(-gamma * ||center - x_i||^2); gamma folds in the bandwidth
        let gamma = 1.0 / (2.0 * self.bandwidth.powi(2));
        let tol_squared = self.tol * self.tol;
        let bandwidth_squared = self.bandwidth * self.bandwidth;
        // Per-sample squared norms, shared across all seeds and iterations
        let x_sq = x.map_axis(Axis(1), |row| row.dot(&row));

        // Each seed task runs O(iterations * n * d) of work
        let use_parallel = seeds
            .len()
            .saturating_mul(n_samples)
            .saturating_mul(n_features)
            >= scan_f64_parallel_min_elems();

        // Mean shift on a single seed
        let process_seed = |seed_idx: usize| -> (Array1<f64>, usize) {
            let mut center = x.row(seed_idx).to_owned();
            let mut completed_iterations = 0;

            loop {
                // RBF weights for every point via the squared-norm identity
                let center_sq = center.dot(&center);
                let projections = x.dot(&center);
                let weights: Array1<f64> =
                    Zip::from(&projections)
                        .and(&x_sq)
                        .map_collect(|&proj, &x_norm_sq| {
                            let dist_sq = (center_sq + x_norm_sq - 2.0 * proj).max(0.0);
                            (-gamma * dist_sq).exp()
                        });
                // Serial sum: runs inside the per-seed parallel loop, where nesting another
                // parallel reduction only adds scheduling overhead
                let weight_sum = weights.sum();

                let new_center = if weight_sum > 0.0 {
                    x.t().dot(&weights) / weight_sum
                } else {
                    Array1::zeros(n_features)
                };

                // Check convergence using squared distance to avoid sqrt
                let shift_squared = squared_euclidean_distance_row(&center, &new_center);
                center = new_center;

                completed_iterations += 1;

                if shift_squared < tol_squared || completed_iterations >= self.max_iter {
                    break;
                }
            }

            (center, completed_iterations)
        };

        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                seeds.len() as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} seeds | {msg}",
            );
            pb.set_message("Processing seeds...");
            pb
        };

        let results: Vec<(Array1<f64>, usize)> = if use_parallel {
            seeds
                .par_iter()
                .map(|&seed_idx| {
                    let result = process_seed(seed_idx);
                    #[cfg(feature = "show_progress")]
                    progress_bar.inc(1);
                    result
                })
                .collect()
        } else {
            seeds
                .iter()
                .map(|&seed_idx| {
                    let result = process_seed(seed_idx);
                    #[cfg(feature = "show_progress")]
                    progress_bar.inc(1);
                    result
                })
                .collect()
        };
        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message("All seeds processed");

        // Extract centers and calculate actual max iterations
        let centers: Vec<Array1<f64>> = results.iter().map(|(c, _)| c.clone()).collect();
        let max_actual_iter = results.iter().map(|(_, i)| *i).max().unwrap_or(0);
        self.n_iter = Some(max_actual_iter);

        // Merge centers that lie within one bandwidth of each other
        let mut unique_centers: Vec<Array1<f64>> = Vec::with_capacity(centers.len());
        let mut center_counts: Vec<usize> = Vec::with_capacity(centers.len());

        for center in centers {
            let mut merged = false;

            // Find closest existing center within bandwidth
            for (i, unique_center) in unique_centers.iter_mut().enumerate() {
                let distance_squared = squared_euclidean_distance_row(&center, unique_center);

                if distance_squared < bandwidth_squared {
                    // Update existing center using weighted average
                    let count = center_counts[i];
                    let new_count = count + 1;
                    let weight_old = count as f64 / new_count as f64;
                    let weight_new = 1.0 / new_count as f64;

                    // Update the center coordinates in place
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

        // Find the nearest cluster label for one sample
        let find_label = |i: usize| -> usize {
            let point = x.row(i);
            let mut min_dist_squared = f64::INFINITY;
            let mut label = 0;

            for (j, center) in unique_centers.iter().enumerate() {
                let dist_squared = squared_euclidean_distance_row(&point, center);
                if dist_squared < min_dist_squared {
                    min_dist_squared = dist_squared;
                    label = j;
                }
            }

            // When cluster_all is off and the point is too far, mark it as an outlier
            if !self.cluster_all && min_dist_squared > bandwidth_squared {
                n_clusters // outlier label
            } else {
                label
            }
        };

        // Assign cluster labels to each data point (scan-class gate: n tasks, each an
        // O(centers * d) distance scan)
        let label_work = n_samples
            .saturating_mul(n_clusters)
            .saturating_mul(x.ncols());
        let labels = map_collect(
            n_samples,
            label_work >= scan_f64_parallel_min_elems(),
            find_label,
        );

        // Count how many samples are assigned to each cluster center
        let mut samples_per_center = vec![0usize; n_clusters];
        for &label in &labels {
            if label < n_clusters {
                samples_per_center[label] += 1;
            }
        }

        self.cluster_centers = Some(cluster_centers);
        self.labels = Some(Array1::from(labels));
        self.n_samples_per_center = Some(Array1::from(samples_per_center));

        Ok(self)
    }

    /// Predicts cluster labels for the input data
    ///
    /// # Parameters
    ///
    /// - `x` - The input data where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, Error>` - The predicted cluster labels, or an Error
    ///
    /// # Errors
    ///
    /// - `Error::NotFitted` - If the model has not been fitted yet
    /// - `Error::InvalidInput` - If input data is invalid or dimensions don't match the training data
    ///
    /// # Performance
    ///
    /// Parallelizes when the total scan work clears the calibrated scan-class gate (see
    /// `crate::parallel_gates`)
    pub fn predict<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, Error>
    where
        S: Data<Elem = f64> + Sync,
    {
        let centers = self
            .cluster_centers
            .as_ref()
            .ok_or_else(|| Error::not_fitted("MeanShift"))?;

        // Validate input against the feature count seen during fitting
        validate_predict_input(x, centers.ncols())?;

        let n_samples = x.nrows();
        let n_clusters = centers.nrows();
        let bandwidth_squared = self.bandwidth * self.bandwidth;

        // Nearest cluster center for one sample, or the outlier label (`n_clusters`) when
        // `cluster_all` is off and the point lies farther than the bandwidth from every center
        let find_nearest = |i: usize| -> usize {
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

            if !self.cluster_all && min_dist_squared > bandwidth_squared {
                n_clusters
            } else {
                label
            }
        };

        // Scan-class gate: n tasks, each an O(centers * d) distance scan
        let label_work = n_samples
            .saturating_mul(n_clusters)
            .saturating_mul(x.ncols());
        let labels = map_collect(
            n_samples,
            label_work >= scan_f64_parallel_min_elems(),
            find_nearest,
        );

        Ok(Array1::from(labels))
    }

    /// Fits the model to the input data and predicts cluster labels
    ///
    /// # Parameters
    ///
    /// - `x` - The input data where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Result<Array1<usize>, Error>` - The predicted cluster labels, or an Error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If input data fails preliminary checks
    pub fn fit_predict<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array1<usize>, Error>
    where
        S: Data<Elem = f64> + Sync + Send,
    {
        self.fit(x)?;
        Ok(self.labels.clone().unwrap())
    }

    /// Computes seed points by binning the feature space onto a grid
    ///
    /// Each sample is mapped to a grid cell of side length `bandwidth`, and one
    /// representative point per non-empty cell is returned. This reduces the number
    /// of seeds the mean-shift iterations have to process on dense datasets
    ///
    /// # Parameters
    ///
    /// - `x` - The input data where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Vec<usize>` - Indices of the selected seed points (one per occupied grid cell)
    fn get_bin_seeds<S>(&self, x: &ArrayBase<S, Ix2>) -> Vec<usize>
    where
        S: Data<Elem = f64> + Sync + Send,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Calculate min for each feature (scan-class gate: d tasks of an O(n) column scan)
        let scan_parallel = n_samples.saturating_mul(n_features) >= scan_f64_parallel_min_elems();
        let col_min = |j: usize| {
            let col = x.column(j);
            col.fold(f64::INFINITY, |a, &b| a.min(b))
        };
        let mins: Vec<f64> = if scan_parallel {
            (0..n_features).into_par_iter().map(col_min).collect()
        } else {
            (0..n_features).map(col_min).collect()
        };

        // The grid is built under a shared HashMap, so this part is harder to parallelize
        let bin_size = self.bandwidth;

        let bins_mutex = std::sync::Mutex::new(AHashMap::<Vec<i64>, Vec<usize>>::new());

        // Assign points to bins (same scan-class gate: n tasks of O(d) quantization each)
        let assign_bin = |i: usize| {
            let point = x.row(i);
            let mut bin_index = Vec::with_capacity(n_features);

            for j in 0..n_features {
                let idx = ((point[j] - mins[j]) / bin_size).floor() as i64;
                bin_index.push(idx);
            }

            // Lock the HashMap only when updating
            let mut bins = bins_mutex.lock().unwrap();
            bins.entry(bin_index).or_default().push(i);
        };
        if scan_parallel {
            (0..n_samples).into_par_iter().for_each(assign_bin);
        } else {
            (0..n_samples).for_each(assign_bin);
        }

        let bins = bins_mutex.into_inner().unwrap();

        // One seed per cell
        let mut seeds = Vec::new();
        for (_, indices) in bins {
            if let Some(&min_idx) = indices.iter().min() {
                seeds.push(min_idx);
            }
        }
        seeds.sort_unstable();

        seeds
    }

    model_save_and_load_methods!(MeanShift);
}

/// Estimates a bandwidth to use with the MeanShift algorithm
///
/// The bandwidth is estimated from the pairwise distances between a subset of points
///
/// # Parameters
///
/// - `x` - The input data where each row is a sample
/// - `quantile` - The quantile of the pairwise distances to use as the bandwidth; defaults to `0.3`
/// - `n_samples` - The number of samples to use for the distance calculation; clamped to the dataset size, defaults to all rows
/// - `random_state` - Seed for random number generation
///
/// # Returns
///
/// - `Result<f64, Error>` - The estimated bandwidth, or an Error
///
/// # Errors
///
/// - `Error::InvalidParameter` - If `quantile` is not in the open range (0, 1)
pub fn estimate_bandwidth<S>(
    x: &ArrayBase<S, Ix2>,
    quantile: Option<f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> Result<f64, Error>
where
    S: Data<Elem = f64>,
{
    let quantile = quantile.unwrap_or(0.3);
    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(Error::invalid_parameter(
            "quantile",
            "must be in the open range (0, 1)",
        ));
    }

    let (n_samples_total, _) = x.dim();
    // Clamp the requested sample count to the dataset size
    let n_samples = n_samples.unwrap_or(n_samples_total).min(n_samples_total);

    let mut rng = crate::random::make_rng(random_state);

    // When the requested count covers every row, use all samples; otherwise sample at random
    let x_samples = if n_samples >= n_samples_total {
        x.to_owned()
    } else {
        let mut indices: Vec<usize> = (0..n_samples_total).collect();
        indices.shuffle(&mut rng);
        let indices = &indices[..n_samples];

        let mut samples = Array2::zeros((n_samples, x.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&x.row(idx));
        }
        samples
    };

    let x_sq = x_samples.map_axis(Axis(1), |row| row.dot(&row));
    let mut distances: Vec<f64> = if cache_resident::<f64>(n_samples, x_samples.ncols()) {
        (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let proj_row = x_samples.dot(&x_samples.row(i));
                ((i + 1)..n_samples)
                    .map(|j| (x_sq[i] + x_sq[j] - 2.0 * proj_row[j]).max(0.0).sqrt())
                    .collect::<Vec<f64>>()
            })
            .collect()
    } else {
        let chunk_rows = gemm_chunk_rows(n_samples);
        let mut distances: Vec<f64> = Vec::new();
        for chunk_start in (0..n_samples).step_by(chunk_rows) {
            let chunk_end = (chunk_start + chunk_rows).min(n_samples);
            let projections = gemm_par_auto(
                &x_samples.slice(s![chunk_start..chunk_end, ..]),
                &x_samples.t(),
            );
            let chunk: Vec<f64> = (chunk_start..chunk_end)
                .into_par_iter()
                .flat_map(|i| {
                    let proj_row = projections.row(i - chunk_start);
                    ((i + 1)..n_samples)
                        .map(|j| (x_sq[i] + x_sq[j] - 2.0 * proj_row[j]).max(0.0).sqrt())
                        .collect::<Vec<f64>>()
                })
                .collect();
            distances.extend(chunk);
        }
        distances
    };

    if distances.is_empty() {
        return Ok(0.0);
    }

    let k = ((distances.len() as f64 * quantile) as usize).min(distances.len() - 1);
    distances.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(distances[k])
}

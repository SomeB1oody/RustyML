use super::*;
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
/// let labels = dbscan.fit_predict(data.view()).unwrap();
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DBSCAN {
    eps: f64,
    min_samples: usize,
    metric: DistanceCalculationMetric,
    labels_: Option<Array1<i32>>,
    core_sample_indices: Option<Array1<usize>>,
}

/// Default parameters for DBSCAN model
///
/// Creates a DBSCAN instance with default parameters:
/// - eps = 0.5
/// - min_samples = 5
/// - metric = Euclidean
impl Default for DBSCAN {
    fn default() -> Self {
        DBSCAN {
            eps: 0.5,
            min_samples: 5,
            metric: DistanceCalculationMetric::Euclidean,
            labels_: None,
            core_sample_indices: None,
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
    /// - `Err(ModelError::InputValidationError)` - If parameters are invalid
    pub fn new(
        eps: f64,
        min_samples: usize,
        metric: DistanceCalculationMetric,
    ) -> Result<Self, ModelError> {
        // Validate eps parameter
        if eps <= 0.0 || !eps.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "eps must be positive and finite, got {}",
                eps
            )));
        }

        // Validate min_samples parameter
        if min_samples == 0 {
            return Err(ModelError::InputValidationError(
                "min_samples must be greater than 0".to_string(),
            ));
        }

        // Validate metric parameter
        match metric {
            DistanceCalculationMetric::Minkowski(p) => {
                if p <= 0.0 || !p.is_finite() {
                    return Err(ModelError::InputValidationError(format!(
                        "Minkowski p must be positive and finite, got {}",
                        p
                    )));
                }
            }
            _ => {} // Euclidean and Manhattan don't need additional validation
        }

        Ok(DBSCAN {
            eps,
            min_samples,
            metric,
            labels_: None,
            core_sample_indices: None,
        })
    }

    // Getters
    get_field!(get_epsilon, eps, f64);
    get_field!(get_min_samples, min_samples, usize);
    get_field!(get_metric, metric, DistanceCalculationMetric);
    get_field_as_ref!(get_labels, labels_, Option<&Array1<i32>>);
    get_field_as_ref!(
        get_core_sample_indices,
        core_sample_indices,
        Option<&Array1<usize>>
    );

    /// Computes distance between two data points using the specified metric
    fn compute_distance(&self, p_row: ArrayView1<f64>, q_row: ArrayView1<f64>) -> f64 {
        match self.metric {
            DistanceCalculationMetric::Euclidean => {
                squared_euclidean_distance_row(p_row, q_row).sqrt()
            }
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(p_row, q_row),
            DistanceCalculationMetric::Minkowski(p) => minkowski_distance_row(p_row, q_row, p),
        }
    }

    /// Find all neighbors of point `p` (points within eps distance)
    ///
    /// Uses parallelization for datasets larger than a threshold to improve performance
    fn region_query(&self, data: ArrayView2<f64>, p: usize) -> Result<Vec<usize>, ModelError> {
        // Bounds check
        if p >= data.nrows() {
            return Err(ModelError::InputValidationError(format!(
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
                .filter_map(|q| {
                    let q_row = data.row(q);
                    let dist = self.compute_distance(p_row, q_row);
                    if dist <= eps { Some(q) } else { None }
                })
                .collect()
        } else {
            // Sequential iteration for smaller datasets
            (0..n_samples)
                .filter_map(|q| {
                    let q_row = data.row(q);
                    let dist = self.compute_distance(p_row, q_row);
                    if dist <= eps { Some(q) } else { None }
                })
                .collect()
        };

        Ok(neighbors)
    }

    /// Performs DBSCAN clustering on the input data
    ///
    /// # Parameters
    ///
    /// * `data` - Input data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The trained instance
    /// - `Err(ModelError::ProcessingError)` - If numerical issues occur during training
    ///
    /// # Notes
    ///
    /// After fitting, cluster labels can be accessed via `get_labels()` method.
    /// Labels of -1 indicate noise points (outliers).
    pub fn fit(&mut self, data: ArrayView2<f64>) -> Result<&mut Self, ModelError> {
        preliminary_check(data, None)?;

        // Check if dataset is empty
        let n_samples = data.nrows();

        // Check for cluster_id overflow early
        if n_samples > i32::MAX as usize {
            return Err(ModelError::InputValidationError(
                "Dataset too large: exceeds maximum number of samples".to_string(),
            ));
        }

        let mut labels = Array1::from(vec![-1; n_samples]); // -1 represents unclassified or noise
        let mut core_samples = AHashSet::with_capacity(n_samples / 4); // Estimate 25% core samples
        let mut cluster_id = 0i32;

        // Initialize progress bar for tracking clustering progress
        let pb = ProgressBar::new(n_samples as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | Clusters: {msg}")
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        pb.set_message("0 | Core points: 0");

        // Main loop processes each point sequentially, the algorithm as a whole remains sequential
        for p in 0..n_samples {
            pb.inc(1);
            if labels[p] != -1 {
                continue;
            }

            let neighbors = self.region_query(data, p).map_err(|e| {
                ModelError::ProcessingError(format!("Region query failed: {:?}", e))
            })?;

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

                let q_neighbors = self.region_query(data, q).map_err(|e| {
                    ModelError::ProcessingError(format!(
                        "Region query failed for point {}: {:?}",
                        q, e
                    ))
                })?;

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
            pb.set_message(format!(
                "{} | Core points: {}",
                cluster_id,
                core_samples.len()
            ));

            // Check for cluster_id overflow
            if cluster_id >= i32::MAX {
                pb.finish_with_message("Error: cluster ID overflow");
                return Err(ModelError::ProcessingError(
                    "Too many clusters: cluster ID overflow".to_string(),
                ));
            }
        }

        // Finish progress bar with final statistics
        pb.finish_with_message(format!(
            "{} | Core points: {} | Noise points: {}",
            cluster_id,
            core_samples.len(),
            labels.iter().filter(|&&x| x == -1).count()
        ));

        self.labels_ = Some(labels);
        // Convert HashSet to sorted Vec for consistent ordering
        let mut core_indices: Vec<usize> = core_samples.into_iter().collect();
        core_indices.sort_unstable();
        self.core_sample_indices = Some(Array1::from(core_indices));

        Ok(self)
    }

    /// Predicts cluster labels for new data points based on trained model
    ///
    /// # Parameters
    ///
    /// - `trained_data` - Original data array that was used for training
    /// - `new_data` - New data points to classify
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<i32>)` - Array of predicted cluster labels
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::InputValidationError)` - If input validation fails
    ///
    /// # Notes
    ///
    /// New points are assigned to the nearest cluster if they are within `eps` distance
    /// of a core point, otherwise they are labeled as noise (-1)
    pub fn predict(
        &self,
        trained_data: ArrayView2<f64>,
        new_data: ArrayView2<f64>,
    ) -> Result<Array1<i32>, ModelError> {
        // Ensure the model has been trained
        let labels = self.labels_.as_ref().ok_or(ModelError::NotFitted)?;
        let core_samples = self
            .core_sample_indices
            .as_ref()
            .ok_or(ModelError::NotFitted)?;

        // Check if trained data is empty
        if trained_data.nrows() == 0 {
            return Err(ModelError::InputValidationError(
                "Trained data is empty".to_string(),
            ));
        }

        // Check if new data is empty
        if new_data.nrows() == 0 {
            return Ok(Array1::from(vec![]));
        }

        // Check dimension matching
        if trained_data.ncols() != new_data.ncols() {
            return Err(ModelError::InputValidationError(format!(
                "Feature dimension mismatch: trained data has {} features, new data has {} features",
                trained_data.ncols(),
                new_data.ncols()
            )));
        }

        if trained_data.nrows() != labels.len() {
            return Err(ModelError::InputValidationError(format!(
                "Trained data rows ({}) don't match labels length ({})",
                trained_data.nrows(),
                labels.len()
            )));
        }

        // Check for invalid values in trained data
        if trained_data.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Trained data contains NaN or infinite values".to_string(),
            ));
        }

        // Check for invalid values in new data
        if new_data.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "New data contains NaN or infinite values".to_string(),
            ));
        }

        // Create a set for faster core sample lookup
        let core_set: AHashSet<usize> = core_samples.iter().copied().collect();

        // Process each row in parallel, collecting into Result<Vec<i32>, ModelError>
        let predictions: Result<Vec<i32>, ModelError> = new_data
            .rows()
            .into_iter()
            .par_bridge() // Convert sequential iterator to parallel iterator
            .map(|row| -> Result<i32, ModelError> {
                let mut min_dist = f64::MAX;
                let mut closest_label = -1;

                // Find the closest classified data point
                for (j, orig_row) in trained_data.rows().into_iter().enumerate() {
                    if labels[j] == -1 {
                        continue; // Skip noise points
                    }

                    let dist = self.compute_distance(row, orig_row);

                    // Check if distance computation is valid
                    if dist.is_nan() || dist.is_infinite() {
                        continue;
                    }

                    // If a core point is found within eps range, assign its label directly
                    if dist <= self.eps && core_set.contains(&j) {
                        return Ok(labels[j]);
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        closest_label = labels[j];
                    }
                }

                // Only assign to closest cluster if within eps distance, otherwise mark as noise
                if min_dist <= self.eps {
                    Ok(closest_label)
                } else {
                    Ok(-1)
                }
            })
            .collect();

        Ok(Array1::from(predictions?))
    }

    /// Performs clustering and returns the labels in one step
    ///
    /// # Parameters
    ///
    /// * `data` - Input data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<i32>)` - Array of cluster labels for each sample
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    ///
    /// # Notes
    ///
    /// This is equivalent to calling `fit()` followed by `get_labels()`,
    /// but more convenient when you don't need to reuse the model.
    pub fn fit_predict(&mut self, data: ArrayView2<f64>) -> Result<Array1<i32>, ModelError> {
        self.fit(data)?;
        Ok(self.labels_.as_ref().unwrap().clone())
    }

    model_save_and_load_methods!(DBSCAN);
}

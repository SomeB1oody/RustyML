use super::*;
use std::collections::VecDeque;

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
/// let mut dbscan = DBSCAN::new(0.5, 2, DistanceCalculationMetric::Euclidean);
/// let labels = dbscan.fit_predict(data.view());
/// ```
#[derive(Debug, Clone)]
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
    /// * `DBSCAN` - A new DBSCAN instance with the specified parameters
    pub fn new(eps: f64, min_samples: usize, metric: DistanceCalculationMetric) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric,
            labels_: None,
            core_sample_indices: None,
        }
    }

    /// Returns the epsilon (neighborhood radius) parameter value
    ///
    /// # Returns
    ///
    /// * `f64` - The current epsilon value
    pub fn get_eps(&self) -> f64 {
        self.eps
    }

    /// Returns the minimum samples parameter value
    ///
    /// # Returns
    ///
    /// * `usize` - The current minimum samples threshold
    pub fn get_min_samples(&self) -> usize {
        self.min_samples
    }

    get_metric!();

    /// Returns the cluster labels assigned to each sample
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<i32>)` - Array of cluster labels if model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_labels(&self) -> Result<&Array1<i32>, ModelError> {
        self.labels_.as_ref().ok_or(ModelError::NotFitted)
    }

    /// Returns the indices of core samples
    ///
    /// Core samples are samples that have at least `min_samples` points within
    /// distance `eps` of themselves.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<usize>)` - Array of indices of core samples if model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_core_sample_indices(&self) -> Result<&Array1<usize>, ModelError> {
        self.core_sample_indices
            .as_ref()
            .ok_or(ModelError::NotFitted)
    }

    /// Computes distance between two data points using the specified metric
    fn compute_distance(&self, p_row: ArrayView1<f64>, q_row: ArrayView1<f64>) -> f64 {
        match self.metric {
            DistanceCalculationMetric::Euclidean => {
                squared_euclidean_distance_row(p_row, q_row).sqrt()
            }
            DistanceCalculationMetric::Manhattan => manhattan_distance_row(p_row, q_row),
            DistanceCalculationMetric::Minkowski => minkowski_distance_row(p_row, q_row, 3.0),
        }
    }

    /// Parallelized version of region_query: find all neighbors of point `p` (points within eps distance)
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

        // Parallel iteration through all rows, calculating distances and filtering points that satisfy the eps condition
        let parallel_neighbors: Vec<usize> = (0..data.nrows())
            .into_par_iter()
            .filter_map(|q| {
                let q_row = data.row(q);
                let dist = self.compute_distance(p_row, q_row);
                if dist <= eps { Some(q) } else { None }
            })
            .collect();

        Ok(parallel_neighbors)
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
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    ///
    /// # Notes
    ///
    /// After fitting, cluster labels can be accessed via `get_labels()` method.
    /// Labels of -1 indicate noise points (outliers).
    pub fn fit(&mut self, data: ArrayView2<f64>) -> Result<&mut Self, ModelError> {
        preliminary_check(data, None)?;

        if self.eps <= 0.0 {
            return Err(ModelError::InputValidationError(
                "eps must be positive".to_string(),
            ));
        }

        if self.min_samples == 0 {
            return Err(ModelError::InputValidationError(
                "min_samples must be greater than 0".to_string(),
            ));
        }

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

        // Main loop processes each point sequentially, the algorithm as a whole remains sequential
        for p in 0..n_samples {
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
                // Skip if q has already been assigned to another cluster
                if labels[q] >= 0 && labels[q] != cluster_id {
                    continue;
                }

                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }

                let q_neighbors = self.region_query(data, q).map_err(|e| {
                    ModelError::ProcessingError(format!(
                        "Region query failed for point {}: {:?}",
                        q, e
                    ))
                })?;

                if q_neighbors.len() >= self.min_samples {
                    core_samples.insert(q);
                    for r in q_neighbors {
                        if labels[r] == -1 {
                            seeds.push_back(r);
                            labels[r] = cluster_id;
                        }
                    }
                }
            }

            cluster_id += 1;

            // Check for cluster_id overflow
            if cluster_id >= i32::MAX {
                return Err(ModelError::ProcessingError(
                    "Too many clusters: cluster ID overflow".to_string(),
                ));
            }
        }

        println!("DBSCAN model computing finished");

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

        // Check if new data is empty
        if new_data.nrows() == 0 {
            return Ok(Array1::from(vec![]));
        }

        let eps_squared = self.eps * self.eps;

        // Create a set for faster core sample lookup
        let core_set: AHashSet<usize> = core_samples.iter().copied().collect();

        // Process each row in parallel, collecting into Vec<i32>
        let predictions: Vec<i32> = new_data
            .rows()
            .into_iter()
            .par_bridge() // Convert sequential iterator to parallel iterator
            .map(|row| {
                let mut min_dist_squared = f64::MAX;
                let mut closest_label = -1;

                // Find the closest classified data point
                for (j, orig_row) in trained_data.rows().into_iter().enumerate() {
                    if labels[j] == -1 {
                        continue; // Skip noise points
                    }

                    let squared_dist = squared_euclidean_distance_row(row, orig_row);

                    // Check if distance computation is valid
                    if squared_dist.is_nan() || squared_dist.is_infinite() {
                        continue;
                    }

                    // If a core point is found within eps range, assign its label directly
                    if squared_dist <= eps_squared && core_set.contains(&j) {
                        return labels[j];
                    }

                    if squared_dist < min_dist_squared {
                        min_dist_squared = squared_dist;
                        closest_label = labels[j];
                    }
                }

                closest_label
            })
            .collect();

        Ok(Array1::from(predictions))
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
}

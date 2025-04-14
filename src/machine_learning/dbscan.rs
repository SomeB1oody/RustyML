use ndarray::{Array1, ArrayView2};
use std::collections::{HashSet, VecDeque};
use crate::ModelError;
use super::DistanceCalculationMetric as Metric;
use rayon::prelude::*;

/// # DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm implementation
///
/// DBSCAN is a popular density-based clustering algorithm that can discover clusters of arbitrary shapes
/// without requiring the number of clusters to be specified beforehand.
///
/// ## Fields
///
/// - `eps` - Neighborhood radius used to find neighbors
/// - `min_samples` - Minimum number of neighbors required to form a core point
/// - `metric` - Distance metric, options: Euclidean, Manhattan, Minkowski(p=3)
///
/// ## Examples
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
    metric: Metric,
    labels_: Option<Array1<i32>>,
    core_sample_indices_: Option<Array1<usize>>,
}

impl Default for DBSCAN {
    /// Creates a DBSCAN instance with default parameters:
    /// - eps = 0.5
    /// - min_samples = 5
    /// - metric = Euclidean
    fn default() -> Self {
        DBSCAN {
            eps: 0.5,
            min_samples: 5,
            metric: Metric::Euclidean,
            labels_: None,
            core_sample_indices_: None,
        }
    }
}

impl DBSCAN {
    /// Creates a new DBSCAN instance with specified parameters
    ///
    /// # Parameters
    ///
    /// * `eps` - Neighborhood radius used to find neighbors
    /// * `min_samples` - Minimum number of neighbors required to form a core point
    /// * `metric` - Distance metric to use (Euclidean, Manhattan, Minkowski)
    ///
    /// # Returns
    ///
    /// * `Self` - A new DBSCAN instance with the specified parameters
    pub fn new(eps: f64, min_samples: usize, metric: Metric) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric,
            labels_: None,
            core_sample_indices_: None,
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

    /// Returns the distance metric being used
    ///
    /// # Returns
    /// 
    /// * `&Metric` - A reference to the Metric enum used by this instance
    pub fn get_metric(&self) -> &Metric {
        &self.metric
    }

    /// Returns the cluster labels assigned to each sample
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<i32>)` - Array of cluster labels if model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_labels(&self) -> Result<&Array1<i32>, ModelError> {
        if let Some(labels) = &self.labels_ {
            Ok(labels)
        } else {
            Err(ModelError::NotFitted)
        }
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
        if let Some(core_sample_indices) = &self.core_sample_indices_ {
            Ok(core_sample_indices)
        } else {
            Err(ModelError::NotFitted)
        }
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
        use super::preliminary_check;
        use crate::math::{squared_euclidean_distance_row, manhattan_distance_row, minkowski_distance_row};

        preliminary_check(data, None)?;

        if self.eps <= 0.0 {
            return Err(ModelError::InputValidationError("eps must be positive".to_string()));
        }

        if self.min_samples <= 0 {
            return Err(ModelError::InputValidationError("min_samples must be greater than 0".to_string()));
        }

        /// Parallelized version of region_query: find all neighbors of point `p` (points within eps distance)
        fn region_query(dbscan: &DBSCAN, data: ArrayView2<f64>, p: usize, metric: Metric) -> Vec<usize> {
            // Pre-compute row p (read-only view) to avoid fetching it repeatedly in each iteration
            let p_row = data.row(p);
            // Parallel iteration through all rows, calculating distances and filtering points that satisfy the eps condition
            (0..data.nrows())
                .into_par_iter()
                .filter_map(|q| {
                    let q_row = data.row(q);
                    let dist = match metric {
                        Metric::Euclidean => squared_euclidean_distance_row(p_row, q_row).sqrt(),
                        Metric::Manhattan => manhattan_distance_row(p_row, q_row),
                        Metric::Minkowski => minkowski_distance_row(p_row, q_row, 3.0), // Default p=3
                    };
                    if dist <= dbscan.eps {
                        Some(q)
                    } else {
                        None
                    }
                })
                .collect()
        }

        let n_samples = data.nrows();
        let mut labels = Array1::from(vec![-1; n_samples]); // -1 represents unclassified or noise
        let mut core_samples = HashSet::new();
        let mut cluster_id = 0;

        // Main loop processes each point sequentially, the algorithm as a whole remains sequential
        for p in 0..n_samples {
            if labels[p] != -1 {
                continue;
            }

            let neighbors = region_query(&self, data, p, self.metric.clone());
            if neighbors.len() < self.min_samples {
                labels[p] = -1; // Mark as noise
                continue;
            }

            // Start a new cluster
            labels[p] = cluster_id;
            core_samples.insert(p);
            let mut seeds = VecDeque::from(neighbors);

            // Expand cluster (the expansion process is still sequential)
            while let Some(q) = seeds.pop_front() {
                // Skip if q has already been assigned to another cluster
                if labels[q] >= 0 && labels[q] != cluster_id {
                    continue;
                }

                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }

                let q_neighbors = region_query(&self, data, q, self.metric.clone());
                if q_neighbors.len() >= self.min_samples {
                    core_samples.insert(q);
                    for &r in &q_neighbors {
                        if labels[r] == -1 || labels[r] == -2 {
                            if labels[r] == -1 {
                                seeds.push_back(r);
                            }
                            labels[r] = cluster_id;
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        println!("DBSCAN model computing finished");

        self.labels_ = Some(labels);
        self.core_sample_indices_ = Some(core_samples.into_iter().collect());

        Ok(self)
    }

    /// Predicts cluster labels for new data points based on trained model
    ///
    /// # Parameters
    ///
    /// - `data` - Original data array that was used for training
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
    pub fn predict(&self, trained_data: ArrayView2<f64>, new_data: ArrayView2<f64>) -> Result<Array1<i32>, ModelError> {
        use crate::math::squared_euclidean_distance_row;

        // Ensure the model has been trained
        let labels = match &self.labels_ {
            Some(l) => l,
            None => return Err(ModelError::NotFitted),
        };

        // Get core sample indices
        let core_samples = self.core_sample_indices_.as_ref().unwrap();

        // Process each row in parallel, collecting into Vec<i32>
        let predictions: Array1<i32> = new_data.rows()
            .into_iter()
            .enumerate()
            .par_bridge() // Convert sequential iterator to parallel iterator
            .map(|(_, row)| {
                let mut min_dist = f64::MAX;
                let mut closest_label = -1;

                // Find the closest classified data point
                for (j, orig_row) in trained_data.rows().into_iter().enumerate() {
                    if labels[j] == -1 {
                        continue; // Skip noise points
                    }

                    let dist = squared_euclidean_distance_row(row.view(), orig_row.view());

                    if dist < min_dist {
                        min_dist = dist;
                        closest_label = labels[j];
                    }

                    // If a core point is found within eps range, assign its label directly
                    if dist <= self.eps && core_samples.iter().any(|&idx| idx == j) {
                        closest_label = labels[j];
                        break;
                    }
                }

                closest_label
            })
            .collect::<Vec<i32>>()
            .into();

        Ok(predictions)
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
use super::*;
use std::ops::AddAssign;

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
/// use rand::random;
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
/// );
///
/// // Fit the model to the data
/// kmeans.fit(data.view()).unwrap();
///
/// // Get cluster labels for all training samples
/// let labels = kmeans.get_labels().as_ref().unwrap();
/// println!("Cluster labels: {:?}", labels);
///
/// // Get the computed centroids
/// let centroids = kmeans.get_centroids().as_ref().unwrap();
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
/// let predicted_labels = kmeans.predict(new_data.view()).unwrap();
/// println!("Predicted labels for new data: {:?}", predicted_labels);
///
/// // Alternative: fit and predict in one step
/// let mut kmeans2 = KMeans::default(); // Uses default parameters
/// let labels = kmeans2.fit_predict(data.view()).unwrap();
/// println!("Labels from fit_predict: {:?}", labels);
/// ```
#[derive(Debug, Clone)]
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
impl Default for KMeans {
    fn default() -> Self {
        KMeans::new(8, 300, 1e-4, None)
    }
}

impl KMeans {
    /// Creates a new KMeans instance with the specified parameters.
    ///
    /// # Parameters
    ///
    /// - `n_clusters` - Number of clusters to form
    /// - `max_iter` - Maximum number of iterations for the algorithm
    /// - `tol` - Convergence tolerance, the algorithm stops when the centroids move less than this value
    /// - `random_seed` - Optional seed for random number generation to ensure reproducibility
    ///
    /// # Returns
    ///
    /// * `KMeans` - A new KMeans instance with the specified configuration
    pub fn new(
        n_clusters: usize,
        max_iterations: usize,
        tolerance: f64,
        random_seed: Option<u64>,
    ) -> Self {
        KMeans {
            n_clusters,
            max_iter: max_iterations,
            tol: tolerance,
            random_seed,
            centroids: None,
            labels: None,
            inertia: None,
            n_iter: None,
        }
    }

    get_field!(get_n_clusters, n_clusters, usize);

    get_field!(get_max_iterations, max_iter, usize);

    get_field!(get_tolerance, tol, f64);

    get_field!(get_random_seed, random_seed, Option<u64>);

    get_field!(get_n_iter, n_iter, Option<usize>);

    get_field_as_ref!(get_labels, labels, &Option<Array1<usize>>);

    get_field!(get_inertia, inertia, Option<f64>);

    get_field!(get_actual_iterations, n_iter, Option<usize>);

    get_field_as_ref!(get_centroids, centroids, &Option<Array2<f64>>);

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
        use crate::math::squared_euclidean_distance_row;

        let sample = x.row(0);
        let centroids = self.centroids.as_ref().unwrap();

        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (i, centroid) in centroids.outer_iter().enumerate() {
            let dist = squared_euclidean_distance_row(sample, centroid)?;
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
    fn init_centroids(&mut self, data: ArrayView2<f64>) -> Result<(), ModelError> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Initialize cluster centers matrix
        let mut centroids = Array2::<f64>::zeros((self.n_clusters, n_features));

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(0),
        };

        // K-means++ initialization method

        // Randomly select the first center point
        let first_center_idx = rng.random_range(0..n_samples);
        centroids.row_mut(0).assign(&data.row(first_center_idx));

        // Select the remaining center points
        for k in 1..self.n_clusters {
            // Calculate the distance from each point to the nearest center
            let distances: Result<Vec<f64>, ModelError> = data
                .outer_iter()
                .into_par_iter()
                .map(|sample| {
                    // Find the closest already selected center point
                    let min_dist = centroids
                        .rows()
                        .into_iter()
                        .take(k)
                        .map(|centroid| squared_euclidean_distance_row(sample, centroid))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .fold(f64::MAX, f64::min);
                    Ok(min_dist)
                })
                .collect();

            let distances = distances?;
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
    /// * `data` - Training data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// - `&mut Self` - A mutable reference to self for method chaining
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn fit(&mut self, data: ArrayView2<f64>) -> Result<&mut Self, ModelError> {
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
        let mut old_inertia = f64::MAX;
        let mut iter_count = 0;

        // Pre-allocate arrays for cluster updates to avoid repeated allocations
        let mut new_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        let mut counts = vec![0usize; self.n_clusters];

        // Main iteration loop
        for i in 0..self.max_iter {
            // Reset for this iteration
            new_centroids.fill(0.0);
            counts.fill(0);

            // Parallel computation: find the closest cluster center and distance for each sample
            let results: Result<Vec<(usize, f64)>, ModelError> = data
                .outer_iter()
                .into_par_iter()
                .enumerate()
                .map(
                    |(idx, sample)| -> Result<(usize, (usize, f64)), ModelError> {
                        let mut min_dist = f64::MAX;
                        let mut min_cluster = 0;

                        // Find closest centroid for this sample
                        for (cluster_idx, centroid) in
                            self.centroids.as_ref().unwrap().outer_iter().enumerate()
                        {
                            let dist = squared_euclidean_distance_row(sample, centroid)?;
                            if dist < min_dist {
                                min_dist = dist;
                                min_cluster = cluster_idx;
                            }
                        }

                        Ok((idx, (min_cluster, min_dist)))
                    },
                )
                .collect::<Result<Vec<_>, _>>()
                .map(|vec| vec.into_iter().map(|(_, result)| result).collect());

            let results = results?;

            // Update labels and compute centroids in a single pass

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

            // Check convergence condition early
            if (old_inertia - inertia).abs() < self.tol * old_inertia.max(self.tol) {
                iter_count = i;
                break;
            }
            old_inertia = inertia;
            iter_count = i;

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

            // Handle empty clusters: for empty clusters, select the point furthest from current centers as new center
            for (cluster_idx, &count) in counts.iter().enumerate() {
                if count == 0 {
                    // Find the sample with maximum distance to its closest centroid
                    let result: Result<(usize, f64), ModelError> = data
                        .outer_iter()
                        .into_par_iter()
                        .enumerate()
                        .map(|(sample_idx, sample)| -> Result<(usize, f64), ModelError> {
                            let distances: Vec<f64> = self
                                .centroids
                                .as_ref()
                                .unwrap()
                                .outer_iter()
                                .enumerate()
                                .filter(|&(idx, _)| idx != cluster_idx) // Exclude the empty cluster
                                .map(|(_, centroid)| {
                                    squared_euclidean_distance_row(sample, centroid)
                                })
                                .collect::<Result<Vec<_>, _>>()?;

                            let min_dist = distances.into_iter().fold(f64::MAX, f64::min);

                            Ok((sample_idx, min_dist))
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map(|vec| {
                            vec.into_iter()
                                .reduce(|acc, curr| if curr.1 > acc.1 { curr } else { acc })
                                .unwrap_or((0, -1.0))
                        });

                    let (farthest_idx, _) = result?;

                    new_centroids
                        .row_mut(cluster_idx)
                        .assign(&data.row(farthest_idx));
                }
            }

            self.centroids = Some(new_centroids.clone());
        }

        self.labels = Some(labels);
        self.inertia = Some(old_inertia);
        self.n_iter = Some(iter_count + 1);

        println!(
            "KMeans model computing finished at iteration {}, cost: {}",
            iter_count + 1,
            old_inertia
        );

        Ok(self)
    }

    /// Predicts the closest cluster for each sample in the input data.
    ///
    /// # Parameters
    ///
    /// * `data` - New data points for which to predict cluster assignments
    ///
    /// # Returns
    ///
    /// - `Array1<usize>` - An array of cluster indices for each input data point
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>, ModelError> {
        if self.centroids.is_none() {
            return Err(ModelError::NotFitted);
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
    /// This is equivalent to calling `fit` followed by `predict`, but more efficient.
    ///
    /// # Parameters
    ///
    /// * `data` - Training data as a 2D array
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<usize>)` - An array of cluster indices for each input data point
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(&mut self, data: ArrayView2<f64>) -> Result<Array1<usize>, ModelError> {
        self.fit(data)?;
        Ok(self.labels.clone().unwrap())
    }
}

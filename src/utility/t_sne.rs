use super::*;
use rand_distr::StandardNormal;

/// A t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation for dimensionality reduction.
///
/// t-SNE is a technique for visualizing high-dimensional data by giving each datapoint
/// a location in a two or three-dimensional map.
///
/// # Fields
///
/// - `perplexity` - Controls the balance between preserving local and global structure. Higher values consider more points as neighbors. Default is 30.0.
/// - `learning_rate` - Step size for gradient descent. Default is 200.0.
/// - `n_iter` - Maximum number of iterations for optimization. Default is 1000.
/// - `dim` - The dimension of the embedded space. Typically 2 or 3 for visualization.
/// - `random_state` - Seed for random number generation to ensure reproducibility. Default is 42.
/// - `early_exaggeration` - Factor to multiply early embeddings to encourage tight cluster formation. Default is 12.0.
/// - `exaggeration_iter` - Number of iterations to use early exaggeration. Default is n_iter/12.
/// - `initial_momentum` - Initial momentum coefficient for gradient updates. Default is 0.5.
/// - `final_momentum` - Final momentum coefficient for gradient updates. Default is 0.8.
/// - `momentum_switch_iter` - Iteration at which momentum changes from initial to final value. Default is n_iter/3.
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use rustyml::utility::t_sne::TSNE;
///
/// let tsne = TSNE::new(None, None, Some(100), 3, None, None, None, None, None, None);
///
/// // Generate some high-dimensional data
/// let data = Array2::<f64>::ones((100, 50));
///
/// // Apply t-SNE dimensionality reduction
/// let embedding = tsne.fit_transform(data.view()).unwrap();
///
/// // `embedding` now contains 100 samples in 3 dimensions
/// assert_eq!(embedding.shape(), &[100, 3]);
/// ```
#[derive(Debug)]
pub struct TSNE {
    perplexity: Option<f64>,
    learning_rate: Option<f64>,
    n_iter: Option<usize>,
    dim: usize,
    random_state: Option<u64>,
    early_exaggeration: Option<f64>,
    exaggeration_iter: Option<usize>,
    initial_momentum: Option<f64>,
    final_momentum: Option<f64>,
    momentum_switch_iter: Option<usize>,
}

impl Default for TSNE {
    fn default() -> Self {
        let default_max_iter = 1000;
        TSNE {
            perplexity: Some(30.0),
            learning_rate: Some(200.0),
            n_iter: Some(default_max_iter),
            dim: 2,
            random_state: Some(42),
            early_exaggeration: Some(12.0),
            exaggeration_iter: Some(default_max_iter / 12),
            initial_momentum: Some(0.5),
            final_momentum: Some(0.8),
            momentum_switch_iter: Some(default_max_iter / 3),
        }
    }
}

impl TSNE {
    /// Creates a new TSNE instance with specified parameters.
    ///
    /// # Parameters
    ///
    /// - `perplexity` - Controls the effective number of neighbors. Higher means more neighbors.
    /// - `learning_rate` - Step size for gradient descent updates.
    /// - `n_iter` - Maximum number of optimization iterations.
    /// - `dim` - Dimensionality of the embedding space.
    /// - `random_state` - Seed for random number generation.
    /// - `early_exaggeration` - Factor to multiply probabilities in early iterations.
    /// - `exaggeration_iter` - Number of iterations to apply early exaggeration.
    /// - `initial_momentum` - Initial momentum coefficient.
    /// - `final_momentum` - Final momentum coefficient.
    /// - `momentum_switch_iter` - Iteration at which momentum switches from initial to final.
    ///
    /// # Returns
    ///
    /// * `TSNE` - A new TSNE instance.
    pub fn new(
        perplexity: Option<f64>,
        learning_rate: Option<f64>,
        n_iter: Option<usize>,
        dim: usize,
        random_state: Option<u64>,
        early_exaggeration: Option<f64>,
        exaggeration_iter: Option<usize>,
        initial_momentum: Option<f64>,
        final_momentum: Option<f64>,
        momentum_switch_iter: Option<usize>,
    ) -> Self {
        TSNE {
            perplexity,
            learning_rate,
            n_iter,
            dim,
            random_state,
            early_exaggeration,
            exaggeration_iter,
            initial_momentum,
            final_momentum,
            momentum_switch_iter,
        }
    }

    /// Returns the perplexity parameter used in t-SNE.
    ///
    /// Perplexity is related to the number of nearest neighbors that
    /// is used in other manifold learning algorithms. Larger datasets
    /// usually require a larger perplexity.
    ///
    /// # Returns
    ///
    /// * `f64` - The perplexity value, defaults to 30.0 if not specified.
    pub fn get_perplexity(&self) -> f64 {
        self.perplexity.unwrap_or(30.0)
    }

    /// Returns the learning rate used in the optimization process.
    ///
    /// The learning rate determines the step size at each iteration
    /// while moving toward the minimum of the cost function.
    ///
    /// # Returns
    ///
    /// *`f64` - The learning rate, defaults to 200.0 if not specified.
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate.unwrap_or(200.0)
    }

    /// Returns the maximum number of iterations for the optimization.
    ///
    /// # Returns
    ///
    /// *`usize` - The maximum number of iterations, defaults to 1000 if not specified.
    pub fn get_n_iter(&self) -> usize {
        self.n_iter.unwrap_or(1000)
    }

    /// Returns the dimensionality of the embedded space.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of dimensions in the embedded space.
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// Returns the random state seed used for reproducibility.
    ///
    /// # Returns
    ///
    /// * `u64` - The random seed value, defaults to 42 if not specified.
    pub fn get_random_state(&self) -> u64 {
        self.random_state.unwrap_or(42)
    }

    /// Returns the early exaggeration factor.
    ///
    /// Early exaggeration increases the attraction between points
    /// in the early phases of optimization to form tighter clusters.
    ///
    /// # Returns
    ///
    /// * `f64` - The early exaggeration factor, defaults to 12.0 if not specified.
    pub fn get_early_exaggeration(&self) -> f64 {
        self.early_exaggeration.unwrap_or(12.0)
    }

    /// Returns the number of iterations for early exaggeration phase.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of iterations for the early exaggeration phase, defaults to 1000/12 if not specified.
    pub fn get_exaggeration_iter(&self) -> usize {
        self.exaggeration_iter.unwrap_or(1000 / 12)
    }

    /// Returns the initial momentum coefficient.
    ///
    /// Momentum accelerates the optimization and helps to escape local minima.
    ///
    /// # Returns
    ///
    /// *`f64` - The initial momentum value, defaults to 0.5 if not specified.
    pub fn get_initial_momentum(&self) -> f64 {
        self.initial_momentum.unwrap_or(0.5)
    }

    /// Returns the final momentum coefficient.
    ///
    /// The momentum is increased from the initial to the final value
    /// during the optimization process.
    ///
    /// # Returns
    ///
    /// * `f64` - The final momentum value, defaults to 0.8 if not specified.
    pub fn get_final_momentum(&self) -> f64 {
        self.final_momentum.unwrap_or(0.8)
    }

    /// Returns the iteration at which momentum value is switched.
    ///
    /// Specifies when to switch from initial momentum to final momentum
    /// during the optimization process.
    ///
    /// # Returns
    ///
    /// * `usize` - The iteration number for momentum switch, defaults to 1000/3 if not specified.
    pub fn get_momentum_switch_iter(&self) -> usize {
        self.momentum_switch_iter.unwrap_or(1000 / 3)
    }

    /// Performs t-SNE dimensionality reduction on input data.
    ///
    /// # Parameters
    ///
    /// * `x` - Input data matrix where each row represents a sample in high-dimensional space.
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - Either a matrix of reduced dimensionality representations where each row corresponds to the original sample
    /// - `Err(ModelError::InputValidationError)` - If input does not match expectation
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        // Optimized parameter validation with reduced closure allocation
        let n_iter = self.validate_positive_param(self.n_iter, 1000, "iterations")?;
        let perplexity = self.validate_positive_f64_param(self.perplexity, 30.0, "perplexity")?;
        let learning_rate =
            self.validate_positive_f64_param(self.learning_rate, 200.0, "learning rate")?;

        let early_exaggeration =
            self.validate_min_f64_param(self.early_exaggeration, 12.0, 1.0, "early exaggeration")?;

        let exaggeration_iter = self.validate_positive_param(
            self.exaggeration_iter,
            n_iter / 12,
            "exaggeration iterations",
        )?;

        let initial_momentum =
            self.validate_range_f64_param(self.initial_momentum, 0.5, "initial momentum")?;

        let final_momentum =
            self.validate_range_f64_param(self.final_momentum, 0.8, "final momentum")?;

        let momentum_switch_iter = self.validate_positive_param(
            self.momentum_switch_iter,
            n_iter / 3,
            "momentum switch iterations",
        )?;

        let random_state = self.random_state.unwrap_or(42);
        let n_samples = x.nrows();

        // Validate dimension parameter
        if self.dim == 0 || self.dim > n_samples {
            return Err(ModelError::InputValidationError(format!(
                "Dimension must be greater than 0 and less than n_samples: {}, got {}",
                n_samples, self.dim
            )));
        }

        // 1. Optimized distance calculation using broadcasting
        let distances = self.compute_pairwise_distances(&x)?;

        // 2. Compute conditional probabilities in parallel
        let p = self.compute_conditional_probabilities(&distances, perplexity)?;

        // 3. Symmetrize and apply early exaggeration
        let p_sym = (&p + &p.t()) / (2.0 * n_samples as f64);
        let mut p_exagg = p_sym.clone() * early_exaggeration;

        // 4. Initialize embedding with better scaling
        let mut y = self.initialize_embedding(n_samples, random_state)?;
        let mut dy = Array2::<f64>::zeros((n_samples, self.dim));

        // 5. Optimized gradient descent loop
        for iter in 0..n_iter {
            // Compute Q matrix more efficiently
            let (q, num) = self.compute_q_matrix(&y)?;

            // Compute gradient with improved numerical stability
            let grad = self.compute_gradient(&y, &p_exagg, &q, &num)?;

            // Update momentum and positions
            let momentum = if iter < momentum_switch_iter {
                initial_momentum
            } else {
                final_momentum
            };

            dy *= momentum;
            dy.scaled_add(-learning_rate, &grad);
            y += &dy;

            // Center the embedding
            let mean_y = y.mean_axis(Axis(0)).unwrap();
            y -= &mean_y;

            // Switch to normal p after early exaggeration phase
            if iter == exaggeration_iter {
                p_exagg = p_sym.clone();
            }
        }

        Ok(y)
    }

    // Helper methods for parameter validation
    fn validate_positive_param(
        &self,
        param: Option<usize>,
        default: usize,
        name: &str,
    ) -> Result<usize, ModelError> {
        match param {
            Some(val) if val > 0 => Ok(val),
            Some(val) => Err(ModelError::InputValidationError(format!(
                "{} must be greater than 0, got {}",
                name, val
            ))),
            None => Ok(default),
        }
    }

    fn validate_positive_f64_param(
        &self,
        param: Option<f64>,
        default: f64,
        name: &str,
    ) -> Result<f64, ModelError> {
        match param {
            Some(val) if val > 0.0 => Ok(val),
            Some(val) => Err(ModelError::InputValidationError(format!(
                "{} must be greater than 0, got {}",
                name, val
            ))),
            None => Ok(default),
        }
    }

    fn validate_min_f64_param(
        &self,
        param: Option<f64>,
        default: f64,
        min_val: f64,
        name: &str,
    ) -> Result<f64, ModelError> {
        match param {
            Some(val) if val > min_val => Ok(val),
            Some(val) => Err(ModelError::InputValidationError(format!(
                "{} must be greater than {}, got {}",
                name, min_val, val
            ))),
            None => Ok(default),
        }
    }

    fn validate_range_f64_param(
        &self,
        param: Option<f64>,
        default: f64,
        name: &str,
    ) -> Result<f64, ModelError> {
        match param {
            Some(val) if (0.0..=1.0).contains(&val) => Ok(val),
            Some(val) => Err(ModelError::InputValidationError(format!(
                "{} must be between 0.0 and 1.0, got {}",
                name, val
            ))),
            None => Ok(default),
        }
    }

    // Optimized distance computation using more efficient approach
    fn compute_pairwise_distances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        let n_samples = x.nrows();
        let mut distances = Array2::<f64>::zeros((n_samples, n_samples));

        // Use parallel computation with better memory access pattern
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let xi = x.row(i);
                for (j, distance) in row.iter_mut().enumerate() {
                    if i != j {
                        let diff = &xi - &x.row(j);
                        *distance = diff.dot(&diff);
                    }
                }
            });

        Ok(distances)
    }

    // Optimized conditional probability computation
    fn compute_conditional_probabilities(
        &self,
        distances: &Array2<f64>,
        perplexity: f64,
    ) -> Result<Array2<f64>, ModelError> {
        let n_samples = distances.nrows();
        let mut p = Array2::<f64>::zeros((n_samples, n_samples));

        // Process each row in parallel
        p.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let (p_i, _) = binary_search_sigma(distances.slice(s![i, ..]), perplexity);
                for (j, &prob) in p_i.iter().enumerate() {
                    if i != j {
                        row[j] = prob;
                    }
                }
            });

        Ok(p)
    }

    // Improved embedding initialization
    fn initialize_embedding(
        &self,
        n_samples: usize,
        random_state: u64,
    ) -> Result<Array2<f64>, ModelError> {
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut y = Array2::<f64>::zeros((n_samples, self.dim));

        // Use better initialization scale based on dimensionality
        let scale = 1e-4 / (self.dim as f64).sqrt();

        // Pre-generate all random numbers
        let total_elements = n_samples * self.dim;
        let random_values: Vec<f64> = (0..total_elements)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * scale)
            .collect();

        // Apply the pre-generated values
        y.indexed_iter_mut()
            .enumerate()
            .for_each(|(idx, (_, elem))| {
                *elem = random_values[idx];
            });

        Ok(y)
    }

    // Optimized Q matrix computation with improved numerical stability
    fn compute_q_matrix(&self, y: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), ModelError> {
        let n_samples = y.nrows();
        let mut num = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute numerator with parallel processing
        num.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let yi = y.row(i);
                for (j, numerator) in row.iter_mut().enumerate() {
                    if i != j {
                        let diff = &yi - &y.row(j);
                        let dist_sq = diff.dot(&diff);
                        *numerator = 1.0 / (1.0 + dist_sq);
                    }
                }
            });

        // Normalize to get Q matrix
        let sum_num = num.sum();
        let q = if sum_num > 0.0 {
            &num / sum_num
        } else {
            return Err(ModelError::ProcessingError(
                "Q matrix normalization failed".to_string(),
            ));
        };

        Ok((q, num))
    }

    // Optimized gradient computation
    fn compute_gradient(
        &self,
        y: &Array2<f64>,
        p: &Array2<f64>,
        q: &Array2<f64>,
        num: &Array2<f64>,
    ) -> Result<Array2<f64>, ModelError> {
        let n_samples = y.nrows();
        let mut grad = Array2::<f64>::zeros((n_samples, self.dim));

        // Parallel gradient computation
        grad.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut grad_i)| {
                let yi = y.row(i);
                for j in 0..n_samples {
                    if i != j {
                        let pq_diff = p[[i, j]] - q[[i, j]];
                        let factor = 4.0 * pq_diff * num[[i, j]];
                        let diff = &yi - &y.row(j);

                        for (d, &diff_val) in diff.iter().enumerate() {
                            grad_i[d] += factor * diff_val;
                        }
                    }
                }
            });

        Ok(grad)
    }
}

/// Finds the appropriate sigma value for a single sample's distances to achieve target perplexity.
///
/// This function uses binary search to find a precision parameter (sigma) that makes the
/// perplexity of the conditional probability distribution match the target value.
///
/// # Parameters
///
/// - `distances` - Vector of squared Euclidean distances from a point to all others.
/// - `target_perplexity` - Desired perplexity value, controlling the effective number of neighbors.
///
/// # Returns
///
/// * `(Array1<f64>, f64)` - A tuple containing:
///   - The normalized probability distribution
///   - The found sigma value that achieves the target perplexity
fn binary_search_sigma(distances: ArrayView1<f64>, target_perplexity: f64) -> (Array1<f64>, f64) {
    let tol = 1e-5;
    let mut sigma_min: f64 = 1e-20;
    let mut sigma_max: f64 = 1e20;
    let mut sigma: f64 = 1.0;
    let n = distances.len();
    let mut p = Array1::<f64>::zeros(n);

    for _ in 0..50 {
        for (j, &d) in distances.iter().enumerate() {
            p[j] = if d == 0.0 {
                0.0
            } else {
                (-d / (2.0 * sigma * sigma)).exp()
            };
        }
        let sum_p = p.sum();
        if sum_p == 0.0 {
            p.fill(1e-10);
        }

        let p_sum = p.sum();
        p.par_mapv_inplace(|v| v / p_sum);

        let h: f64 = p
            .iter()
            .map(|&v| if v > 1e-10 { -v * v.ln() } else { 0.0 })
            .sum();
        let current_perplexity = h.exp();
        let diff = current_perplexity - target_perplexity;
        if diff.abs() < tol {
            break;
        }
        if diff > 0.0 {
            sigma_min = sigma;
            if sigma_max.is_infinite() {
                sigma *= 2.0;
            } else {
                sigma = (sigma + sigma_max) / 2.0;
            }
        } else {
            sigma_max = sigma;
            sigma = (sigma + sigma_min) / 2.0;
        }
    }
    (p, sigma)
}

use crate::ModelError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

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
    /// * `Self` - A new TSNE instance.
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
    /// * `Ok(Array2<f64>)` - Either a matrix of reduced dimensionality representations where each row corresponds to the original sample
    /// - `Err(ModelError::InputValidationError)` - If input does not match expectation
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        use crate::math::squared_euclidean_distance_row;

        fn validate_param<T: PartialOrd + Copy + std::fmt::Display>(
            value: Option<T>,
            default: T,
            check_fn: impl Fn(T) -> bool,
            error_msg: impl Fn(T) -> String,
        ) -> Result<T, ModelError> {
            match value {
                Some(val) => {
                    if !check_fn(val) {
                        Err(ModelError::InputValidationError(error_msg(val)))
                    } else {
                        Ok(val)
                    }
                }
                None => Ok(default),
            }
        }

        let n_iter = validate_param(
            self.n_iter,
            1000,
            |val| val > 0,
            |val| format!("Number of iterations must be greater than 0, got {}", val),
        )?;

        let perplexity = validate_param(
            self.perplexity,
            30.0,
            |val| val > 0.0,
            |val| format!("Perplexity must be greater than 0, got {}", val),
        )?;

        let learning_rate = validate_param(
            self.learning_rate,
            200.0,
            |val| val > 0.0,
            |val| format!("Learning rate must be greater than 0, got {}", val),
        )?;

        let random_state = self.random_state.unwrap_or(42);

        let early_exaggeration = validate_param(
            self.early_exaggeration,
            12.0,
            |val| val > 1.0,
            |val| format!("Early exaggeration must be greater than 1.0, got {}", val),
        )?;

        let exaggeration_iter = validate_param(
            self.exaggeration_iter,
            n_iter / 12,
            |val| val > 0,
            |val| format!("Exaggeration iteration must be greater than 0, got {}", val),
        )?;

        let initial_momentum = validate_param(
            self.initial_momentum,
            0.5,
            |val| val >= 0.0 && val <= 1.0,
            |val| format!("Initial momentum must be between 0.0 and 1.0, got {}", val),
        )?;

        let final_momentum = validate_param(
            self.final_momentum,
            0.8,
            |val| val >= 0.0 && val <= 1.0,
            |val| format!("Final momentum must be between 0.0 and 1.0, got {}", val),
        )?;

        let momentum_switch_iter = validate_param(
            self.momentum_switch_iter,
            n_iter / 3,
            |val| val > 0,
            |val| {
                format!(
                    "Momentum switch iteration must be greater than 0, got {}",
                    val
                )
            },
        )?;

        validate_param(
            Some(self.dim),
            2,
            |val| val > 0 && val <= x.nrows(),
            |val| {
                format!(
                    "Dimension must be greater than 0 and less than n_samples: {}, got {}",
                    x.nrows(),
                    val
                )
            },
        )?;

        let n_samples = x.nrows();

        // 1. Calculate the squared Euclidean distance between all samples in high-dimensional space
        // Fix method: each thread calculates its own row, then merge the results
        let distances = {
            let mut distances = Array2::<f64>::zeros((n_samples, n_samples));

            // Method 1: Use index ranges for parallel computation, each thread is responsible for one row
            let indices: Vec<usize> = (0..n_samples).collect();
            let results: Vec<_> = indices
                .par_iter()
                .map(|&i| {
                    let mut row_dists = Vec::with_capacity(n_samples);
                    for j in 0..n_samples {
                        row_dists.push(squared_euclidean_distance_row(x.row(i), x.row(j)));
                    }
                    (i, row_dists)
                })
                .collect();

            // Collect results
            for (i, row_dists) in results {
                for j in 0..n_samples {
                    distances[[i, j]] = row_dists[j];
                }
            }

            distances
        };

        // 2. Use binary search to calculate conditional probability distribution p_{j|i} for each sample
        let mut p = Array2::<f64>::zeros((n_samples, n_samples));
        let mut rows: Vec<_> = p.axis_iter_mut(Axis(0)).collect();

        rows.par_iter_mut().enumerate().for_each(|(i, row)| {
            let (p_i, _sigma) = binary_search_sigma(distances.slice(s![i, ..]), perplexity);
            for j in 0..n_samples {
                if i != j {
                    row[j] = p_i[j];
                }
            }
        });
        // Symmetrize p: p_sym = (p + p^T) / (2N)
        let p_sym = (&p + &p.t()) / (2.0 * n_samples as f64);
        // Early exaggeration: amplify p in early iterations
        let mut p_exagg = p_sym.clone() * early_exaggeration;

        // 3. Initialize low-dimensional mapping y (using small random values)
        let mut rng = StdRng::seed_from_u64(random_state);
        let mut y = Array2::<f64>::zeros((n_samples, self.dim));
        for mut row in y.axis_iter_mut(Axis(0)) {
            for elem in row.iter_mut() {
                *elem = rng.sample::<f64, _>(StandardNormal) * 1e-4;
            }
        }

        // 4. Initialize gradient momentum matrix dy
        let mut dy = Array2::<f64>::zeros((n_samples, self.dim));

        // 5. Iterate using gradient descent and momentum updates
        for iter in 0..n_iter {
            // Calculate the similarity q between points in low-dimensional space
            let num = {
                let mut num = Array2::<f64>::zeros((n_samples, n_samples));

                // Use the same fix method as before
                let indices: Vec<usize> = (0..n_samples).collect();
                let results: Vec<_> = indices
                    .par_iter()
                    .map(|&i| {
                        let mut row_nums = Vec::with_capacity(n_samples);
                        for j in 0..n_samples {
                            if i != j {
                                let diff = &y.row(i) - &y.row(j);
                                let dist = diff.dot(&diff);
                                row_nums.push((j, 1.0 / (1.0 + dist)));
                            } else {
                                row_nums.push((j, 0.0));
                            }
                        }
                        (i, row_nums)
                    })
                    .collect();

                // Collect results
                for (i, row_nums) in results {
                    for (j, val) in row_nums {
                        num[[i, j]] = val;
                    }
                }

                num
            };

            let sum_num = num.sum();
            let q = &num / sum_num;

            // Calculate gradient dC/dy
            let grad = {
                let mut grad = Array2::<f64>::zeros((n_samples, self.dim));

                // Use the same fix method as before
                let indices: Vec<usize> = (0..n_samples).collect();
                let results: Vec<_> = indices
                    .par_iter()
                    .map(|&i| {
                        let mut grad_i = Array1::<f64>::zeros(self.dim);
                        for j in 0..n_samples {
                            if i != j {
                                let mult = (p_exagg[[i, j]] - q[[i, j]]) * num[[i, j]];
                                let diff = &y.row(i) - &y.row(j);
                                grad_i = grad_i + diff.to_owned() * mult;
                            }
                        }
                        let grad_i = grad_i * 4.0;
                        (i, grad_i)
                    })
                    .collect();

                // Collect results
                for (i, grad_i) in results {
                    for d in 0..self.dim {
                        grad[[i, d]] = grad_i[d];
                    }
                }

                grad
            };

            // Select momentum parameter based on iteration count
            let momentum = if iter < momentum_switch_iter {
                initial_momentum
            } else {
                final_momentum
            };

            // Momentum update formula: dy = momentum * dy - learning_rate * grad
            dy = dy * momentum - &grad * learning_rate;
            y = y + &dy;

            // Keep y zero-mean
            let mean_y = y.mean_axis(Axis(0)).unwrap();
            y = &y - &mean_y;

            // When reaching early exaggeration iteration limit, restore normal p
            if iter == exaggeration_iter {
                p_exagg = p_sym.clone();
            }
        }

        Ok(y)
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

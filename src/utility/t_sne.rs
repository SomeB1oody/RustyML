use super::*;
use crate::math::binary_search_sigma;
use rand_distr::StandardNormal;

/// Threshold for determining whether to use parallel computation in t-SNE.
/// Parallel processing is only beneficial when the number of samples exceeds this threshold,
/// as the overhead of thread creation and management can outweigh benefits for smaller datasets.
const TSNE_PARALLEL_THRESHOLD: usize = 1000;

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
/// // Create a new TSNE instance with custom parameters
/// let tsne = TSNE::new(None, None, Some(100), 3, None, None, None, None, None, None).unwrap();
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    dim: usize,
    random_state: u64,
    early_exaggeration: f64,
    exaggeration_iter: usize,
    initial_momentum: f64,
    final_momentum: f64,
    momentum_switch_iter: usize,
}

/// Default implementation for TSNE
///
/// Creates a new TSNE instance with default values:
/// - `perplexity`: 30.0 (typical range 5-50, controls neighborhood size)
/// - `learning_rate`: 200.0 (typical range 10-1000, controls step size in optimization)
/// - `n_iter`: 1000 (maximum number of optimization iterations)
/// - `dim`: 2 (target dimensionality, commonly 2 for visualization)
/// - `random_state`: 42 (seed for random number generation for reproducibility)
/// - `early_exaggeration`: 12.0 (factor for early exaggeration phase)
/// - `exaggeration_iter`: 83 (approximately n_iter/12, iterations for early exaggeration)
/// - `initial_momentum`: 0.5 (momentum for first phase of optimization)
/// - `final_momentum`: 0.8 (momentum for second phase of optimization)  
/// - `momentum_switch_iter`: 333 (approximately n_iter/3, when to switch momentum values)
impl Default for TSNE {
    fn default() -> Self {
        let default_max_iter = 1000;
        TSNE {
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: default_max_iter,
            dim: 2,
            random_state: 42,
            early_exaggeration: 12.0,
            exaggeration_iter: default_max_iter / 12,
            initial_momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: default_max_iter / 3,
        }
    }
}

impl TSNE {
    /// Creates a new TSNE instance with specified parameters.
    ///
    /// # Parameters
    ///
    /// - `perplexity` - Controls the effective number of neighbors. Higher means more neighbors. Default is 30.0.
    /// - `learning_rate` - Step size for gradient descent updates. Default is 200.0.
    /// - `n_iter` - Maximum number of optimization iterations. Default is 1000.
    /// - `dim` - Dimensionality of the embedding space. Must be greater than 0.
    /// - `random_state` - Seed for random number generation. Default is 42.
    /// - `early_exaggeration` - Factor to multiply probabilities in early iterations. Default is 12.0.
    /// - `exaggeration_iter` - Number of iterations to apply early exaggeration. Default is n_iter/12.
    /// - `initial_momentum` - Initial momentum coefficient. Must be in range [0.0, 1.0]. Default is 0.5.
    /// - `final_momentum` - Final momentum coefficient. Must be in range [0.0, 1.0]. Default is 0.8.
    /// - `momentum_switch_iter` - Iteration at which momentum switches from initial to final. Default is n_iter/3.
    ///
    /// # Returns
    ///
    /// * `Ok(TSNE)` - A new TSNE instance if all parameters are valid
    /// * `Err(ModelError::InputValidationError)` - If any parameter is invalid
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
    ) -> Result<Self, ModelError> {
        let n_iter_val = n_iter.unwrap_or(1000);
        let perplexity_val = perplexity.unwrap_or(30.0);
        let learning_rate_val = learning_rate.unwrap_or(200.0);
        let early_exaggeration_val = early_exaggeration.unwrap_or(12.0);
        let exaggeration_iter_val = exaggeration_iter.unwrap_or(n_iter_val / 12);
        let initial_momentum_val = initial_momentum.unwrap_or(0.5);
        let final_momentum_val = final_momentum.unwrap_or(0.8);
        let momentum_switch_iter_val = momentum_switch_iter.unwrap_or(n_iter_val / 3);

        // Validate parameters
        Self::validate_positive_usize_static(n_iter_val, "n_iter")?;
        Self::validate_positive_f64_static(perplexity_val, "perplexity")?;
        Self::validate_positive_f64_static(learning_rate_val, "learning_rate")?;
        Self::validate_min_f64_static(early_exaggeration_val, 1.0, "early_exaggeration")?;
        Self::validate_positive_usize_static(exaggeration_iter_val, "exaggeration_iter")?;
        Self::validate_range_f64_static(initial_momentum_val, "initial_momentum")?;
        Self::validate_range_f64_static(final_momentum_val, "final_momentum")?;
        Self::validate_positive_usize_static(momentum_switch_iter_val, "momentum_switch_iter")?;
        Self::validate_positive_usize_static(dim, "dim")?;

        Ok(TSNE {
            perplexity: perplexity_val,
            learning_rate: learning_rate_val,
            n_iter: n_iter_val,
            dim,
            random_state: random_state.unwrap_or(42),
            early_exaggeration: early_exaggeration_val,
            exaggeration_iter: exaggeration_iter_val,
            initial_momentum: initial_momentum_val,
            final_momentum: final_momentum_val,
            momentum_switch_iter: momentum_switch_iter_val,
        })
    }

    // Getters
    get_field!(get_perplexity, perplexity, f64);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_actual_iterations, n_iter, usize);
    get_field!(get_dimensions, dim, usize);
    get_field!(get_random_state, random_state, u64);
    get_field!(get_early_exaggeration, early_exaggeration, f64);
    get_field!(get_exaggeration_iterations, exaggeration_iter, usize);
    get_field!(get_initial_momentum, initial_momentum, f64);
    get_field!(get_final_momentum, final_momentum, f64);
    get_field!(get_momentum_switch_iterations, momentum_switch_iter, usize);

    /// Performs t-SNE dimensionality reduction on input data.
    ///
    /// # Parameters
    ///
    /// * `x` - Input data matrix where each row represents a sample in high-dimensional space.
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - A matrix of reduced dimensionality representations where each row corresponds to the original sample
    /// - `Err(ModelError::InputValidationError)` - If input does not match expectation
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        // Validate input data
        if x.is_empty() {
            return Err(ModelError::InputValidationError(
                "Input data is empty".to_string(),
            ));
        }

        if x.iter().any(|&val| !val.is_finite()) {
            return Err(ModelError::InputValidationError(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }

        let n_samples = x.nrows();

        // Validate dimension parameter against actual sample size
        if self.dim > n_samples {
            return Err(ModelError::InputValidationError(format!(
                "dim must be less than or equal to n_samples: {}, got {}",
                n_samples, self.dim
            )));
        }

        // Validate perplexity is appropriate for the dataset size
        if self.perplexity >= n_samples as f64 {
            return Err(ModelError::InputValidationError(format!(
                "perplexity must be less than n_samples: {}, got {}",
                n_samples, self.perplexity
            )));
        }

        // 1. Optimized distance calculation using broadcasting
        let distances = self.compute_pairwise_distances(&x)?;

        // 2. Compute conditional probabilities in parallel
        let p = self.compute_conditional_probabilities(&distances, self.perplexity)?;

        // 3. Symmetrize and apply early exaggeration
        let p_sym = (&p + &p.t()) / 2.0;
        let mut p_exagg = p_sym.clone() * self.early_exaggeration;

        // 4. Initialize embedding with better scaling
        let mut y = self.initialize_embedding(n_samples, self.random_state)?;
        let mut dy = Array2::<f64>::zeros((n_samples, self.dim));

        // Create progress bar for optimization iterations
        let progress_bar = ProgressBar::new(self.n_iter as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} | KL Divergence: {msg}",
                )
                .expect("Failed to set progress bar template")
                .progress_chars("█▓░"),
        );
        progress_bar.set_message(format!("{:.6}", 0.0));

        // 5. Optimized gradient descent loop
        for iter in 0..self.n_iter {
            // Compute Q matrix more efficiently
            let (q, num) = self.compute_q_matrix(&y)?;

            // Compute gradient with improved numerical stability
            let grad = self.compute_gradient(&y, &p_exagg, &q, &num)?;

            // Compute KL divergence for monitoring
            let kl_divergence = self.compute_kl_divergence(&p_exagg, &q);

            // Update progress bar with current KL divergence
            progress_bar.set_message(format!("{:.6}", kl_divergence));
            progress_bar.inc(1);

            // Update momentum and positions
            let momentum = if iter < self.momentum_switch_iter {
                self.initial_momentum
            } else {
                self.final_momentum
            };

            dy *= momentum;
            dy.scaled_add(-self.learning_rate, &grad);
            y += &dy;

            // Center the embedding
            let mean_y = y.mean_axis(Axis(0)).unwrap();
            y -= &mean_y;

            // Switch to normal p after early exaggeration phase
            if iter == self.exaggeration_iter {
                p_exagg = p_sym.clone();
            }
        }

        // Finish progress bar with final KL divergence
        let (final_q, _) = self.compute_q_matrix(&y)?;
        let final_kl = self.compute_kl_divergence(&p_sym, &final_q);
        progress_bar.finish_with_message(format!("{:.6} | Completed", final_kl));

        println!(
            "\nt-SNE dimensionality reduction completed: {} samples, {} -> {} dimensions, {} iterations, final KL divergence: {:.6}",
            n_samples,
            x.ncols(),
            self.dim,
            self.n_iter,
            final_kl
        );

        Ok(y)
    }

    // Helper methods for parameter validation
    fn validate_positive_usize_static(val: usize, name: &str) -> Result<(), ModelError> {
        if val > 0 {
            Ok(())
        } else {
            Err(ModelError::InputValidationError(format!(
                "{} must be greater than 0, got {}",
                name, val
            )))
        }
    }

    fn validate_positive_f64_static(val: f64, name: &str) -> Result<(), ModelError> {
        if val > 0.0 && val.is_finite() {
            Ok(())
        } else {
            Err(ModelError::InputValidationError(format!(
                "{} must be positive and finite, got {}",
                name, val
            )))
        }
    }

    fn validate_min_f64_static(val: f64, min_val: f64, name: &str) -> Result<(), ModelError> {
        if val > min_val && val.is_finite() {
            Ok(())
        } else {
            Err(ModelError::InputValidationError(format!(
                "{} must be greater than {} and finite, got {}",
                name, min_val, val
            )))
        }
    }

    fn validate_range_f64_static(val: f64, name: &str) -> Result<(), ModelError> {
        if (0.0..=1.0).contains(&val) && val.is_finite() {
            Ok(())
        } else {
            Err(ModelError::InputValidationError(format!(
                "{} must be between 0.0 and 1.0, got {}",
                name, val
            )))
        }
    }

    // Optimized distance computation using more efficient approach
    fn compute_pairwise_distances(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ModelError> {
        let n_samples = x.nrows();
        let mut distances = Array2::<f64>::zeros((n_samples, n_samples));

        // Use parallel computation only when sample size exceeds threshold
        if n_samples >= TSNE_PARALLEL_THRESHOLD {
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
        } else {
            // Sequential computation for small datasets
            for (i, mut row) in distances.axis_iter_mut(Axis(0)).enumerate() {
                let xi = x.row(i);
                for (j, distance) in row.iter_mut().enumerate() {
                    if i != j {
                        let diff = &xi - &x.row(j);
                        *distance = diff.dot(&diff);
                    }
                }
            }
        }

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

        // Process each row with conditional parallelization
        if n_samples >= TSNE_PARALLEL_THRESHOLD {
            p.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    let (p_i, _) = binary_search_sigma(&distances.slice(s![i, ..]), perplexity);
                    for (j, &prob) in p_i.iter().enumerate() {
                        if i != j {
                            row[j] = prob;
                        }
                    }
                });
        } else {
            // Sequential computation for small datasets
            for (i, mut row) in p.axis_iter_mut(Axis(0)).enumerate() {
                let (p_i, _) = binary_search_sigma(&distances.slice(s![i, ..]), perplexity);
                for (j, &prob) in p_i.iter().enumerate() {
                    if i != j {
                        row[j] = prob;
                    }
                }
            }
        }

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

        // Use standard t-SNE initialization scale
        let scale = 1e-4;

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

        // Compute numerator with conditional parallelization
        if n_samples >= TSNE_PARALLEL_THRESHOLD {
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
        } else {
            // Sequential computation for small datasets
            for (i, mut row) in num.axis_iter_mut(Axis(0)).enumerate() {
                let yi = y.row(i);
                for (j, numerator) in row.iter_mut().enumerate() {
                    if i != j {
                        let diff = &yi - &y.row(j);
                        let dist_sq = diff.dot(&diff);
                        *numerator = 1.0 / (1.0 + dist_sq);
                    }
                }
            }
        }

        // Normalize to get Q matrix with numerical stability
        let sum_num = num.sum();
        let epsilon = 1e-12;
        let q = if sum_num > epsilon {
            &num / sum_num.max(epsilon)
        } else {
            return Err(ModelError::ProcessingError(
                "Q matrix normalization failed".to_string(),
            ));
        };

        Ok((q, num))
    }

    // Compute KL divergence for monitoring optimization progress
    fn compute_kl_divergence(&self, p: &Array2<f64>, q: &Array2<f64>) -> f64 {
        let epsilon = 1e-12;
        p.iter()
            .zip(q.iter())
            .map(|(&p_val, &q_val)| {
                if p_val > epsilon {
                    p_val * (p_val / q_val.max(epsilon)).ln()
                } else {
                    0.0
                }
            })
            .sum()
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

        // Define gradient computation logic as a closure to avoid code duplication
        let compute_grad_row = |i: usize, mut grad_i: ArrayViewMut1<f64>| {
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
        };

        // Gradient computation with conditional parallelization
        if n_samples >= TSNE_PARALLEL_THRESHOLD {
            grad.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, grad_i)| compute_grad_row(i, grad_i));
        } else {
            // Sequential computation for small datasets
            grad.axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i, grad_i)| compute_grad_row(i, grad_i));
        }

        Ok(grad)
    }

    model_save_and_load_methods!(TSNE);
}

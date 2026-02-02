use crate::error::ModelError;
use crate::math::{binary_search_sigma, squared_euclidean_distance_row};
use crate::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Early exaggeration factor applied to joint probabilities.
const EARLY_EXAGGERATION: f64 = 12.0;
/// Number of iterations to apply early exaggeration.
const EARLY_EXAGGERATION_ITER: usize = 250;
/// Initial momentum used at the start of optimization.
const INITIAL_MOMENTUM: f64 = 0.5;
/// Momentum used after early exaggeration phase.
const FINAL_MOMENTUM: f64 = 0.8;
/// Scale for random initialization of the embedding.
const INIT_SCALE: f64 = 1e-4;
/// Lower bound for q_ij to avoid log(0) in KL divergence.
const MIN_Q: f64 = 1e-12;
/// Threshold for switching to parallel computation in t-SNE.
const TSNE_PRARALLEL_THRESHOLD: usize = 2000;

/// t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction.
///
/// This implementation performs t-SNE optimization with early exaggeration and
/// momentum-based gradient descent.
///
/// # Fields
///
/// - `n_components` - Output embedding dimensions
/// - `perplexity` - Effective neighborhood size
/// - `learning_rate` - Gradient descent learning rate
/// - `n_iter` - Number of optimization iterations
/// - `random_state` - Optional random seed for reproducibility
///
/// # Examples
/// ```rust
/// use rustyml::utility::t_sne::TSNE;
/// use ndarray::array;
///
/// let tsne = TSNE::new(2, 2.0, 200.0, 250, Some(42)).unwrap();
/// let x = array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]];
/// let embedding = tsne.fit_transform(&x).unwrap();
/// assert_eq!(embedding.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    random_state: Option<u64>,
}

impl Default for TSNE {
    /// Creates a TSNE instance with common default parameters.
    ///
    /// # Default Values
    ///
    /// - `n_components` - 2
    /// - `perplexity` - 30.0
    /// - `learning_rate` - 200.0
    /// - `n_iter` - 1000
    /// - `random_state` - None
    fn default() -> Self {
        TSNE::new(2, 30.0, 200.0, 1000, None).expect("Default TSNE parameters should be valid")
    }
}

impl TSNE {
    /// Creates a new TSNE instance with validation.
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of output dimensions (must be greater than 0)
    /// - `perplexity` - Controls effective neighborhood size (must be positive and finite)
    /// - `learning_rate` - Gradient descent learning rate (must be positive and finite)
    /// - `n_iter` - Maximum number of optimization iterations (must be greater than 0)
    /// - `random_state` - Optional random seed for reproducibility
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new TSNE instance or validation error
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If any parameter is invalid
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        random_state: Option<u64>,
    ) -> Result<Self, ModelError> {
        if n_components == 0 {
            return Err(ModelError::InputValidationError(
                "n_components must be greater than 0".to_string(),
            ));
        }

        if perplexity <= 0.0 || !perplexity.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "perplexity must be positive and finite, got {}",
                perplexity
            )));
        }

        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(ModelError::InputValidationError(format!(
                "learning_rate must be positive and finite, got {}",
                learning_rate
            )));
        }

        if n_iter == 0 {
            return Err(ModelError::InputValidationError(
                "n_iter must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            random_state,
        })
    }

    // Getters
    get_field!(get_n_components, n_components, usize);
    get_field!(get_perplexity, perplexity, f64);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_n_iter, n_iter, usize);
    get_field!(get_random_state, random_state, Option<u64>);

    /// Performs t-SNE dimensionality reduction on input data.
    ///
    /// # Parameters
    ///
    /// - `x` - Input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Result<Array2<f64>, ModelError>` - Reduced embedding of shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If the input matrix is empty, too small, or contains non-finite values
    ///
    /// # Performance
    ///
    /// Uses Rayon parallel computation when `x.nrows()` is above `TSNE_PRARALLEL_THRESHOLD` (2000).
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Validate inputs before any heavy computation
        self.validate_input(x)?;

        let x_owned = x.to_owned();
        let n_samples = x_owned.nrows();
        // Decide execution mode from sample count
        let use_parallel = n_samples > TSNE_PRARALLEL_THRESHOLD;

        // Precompute distances and convert them to joint probabilities
        let distances = self.pairwise_squared_distances(&x_owned, use_parallel);
        let p_conditional = self.conditional_probabilities(&distances, use_parallel);
        let p = self.symmetrize_probabilities(&p_conditional);
        let p_exaggerated = p.mapv(|v| v * EARLY_EXAGGERATION);

        // Initialize embedding and momentum buffer
        let mut y = self.init_embedding(n_samples);
        let mut y_incs = Array2::<f64>::zeros((n_samples, self.n_components));

        // Progress bar reports KL divergence each iteration
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

        let exaggeration_iter = EARLY_EXAGGERATION_ITER.min(self.n_iter);
        let mut last_kl = 0.0;

        for iter in 0..self.n_iter {
            // Use early exaggeration for the first phase only
            let p_use = if iter < exaggeration_iter {
                &p_exaggerated
            } else {
                &p
            };

            // Compute Student-t affinities and gradient
            let (num, sum_num) = self.compute_num_matrix(&y, use_parallel);
            let grad = self.compute_gradient(&y, p_use, &num, sum_num, use_parallel);

            // Switch momentum after the exaggeration phase
            let momentum = if iter < exaggeration_iter {
                INITIAL_MOMENTUM
            } else {
                FINAL_MOMENTUM
            };

            // Apply momentum SGD update to the embedding
            for i in 0..n_samples {
                for d in 0..self.n_components {
                    y_incs[[i, d]] = momentum * y_incs[[i, d]] - self.learning_rate * grad[[i, d]];
                    y[[i, d]] += y_incs[[i, d]];
                }
            }

            // Keep embedding centered to avoid drift
            self.center_embedding(&mut y)?;

            // Track KL divergence for reporting only
            last_kl = self.kl_divergence(p_use, &num, sum_num, use_parallel);
            progress_bar.set_message(format!("{:.6}", last_kl));
            progress_bar.inc(1);
        }

        progress_bar.finish_with_message(format!("{:.6}", last_kl));

        Ok(y)
    }

    fn validate_input<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), ModelError>
    where
        S: Data<Elem = f64>,
    {
        // Basic shape and size checks
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(ModelError::InputValidationError(
                "Input data must have at least one row and one column".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(ModelError::InputValidationError(
                "t-SNE requires at least 2 samples".to_string(),
            ));
        }

        if self.perplexity >= x.nrows() as f64 {
            // Perplexity must be less than the number of samples
            return Err(ModelError::InputValidationError(format!(
                "perplexity must be less than number of samples, got perplexity={} with samples={}",
                self.perplexity,
                x.nrows()
            )));
        }

        if let Some(((i, j), _)) = x.indexed_iter().find(|&(_, &val)| !val.is_finite()) {
            // Reject NaN/Inf to avoid blowing up the optimizer
            return Err(ModelError::InputValidationError(format!(
                "Input data contains NaN or infinite value at position [{}, {}]",
                i, j
            )));
        }

        Ok(())
    }

    fn init_embedding(&self, n_samples: usize) -> Array2<f64> {
        // Seeded RNG for reproducibility if provided
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            }
        };

        let mut y = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for d in 0..self.n_components {
                // Small random init avoids large gradients at start
                y[[i, d]] = rng.random_range(-0.5..0.5) * INIT_SCALE;
            }
        }

        y
    }

    fn pairwise_squared_distances(&self, x: &Array2<f64>, parallel: bool) -> Array2<f64> {
        let n_samples = x.nrows();

        if parallel {
            // Compute all rows independently with Rayon
            let rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let row_i = x.row(i);
                    let mut row = vec![0.0; n_samples];
                    for j in 0..n_samples {
                        if i == j {
                            continue;
                        }
                        row[j] = squared_euclidean_distance_row(&row_i, &x.row(j));
                    }
                    row
                })
                .collect();

            let mut distances = Array2::<f64>::zeros((n_samples, n_samples));
            for (i, row) in rows.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    distances[[i, j]] = val;
                }
            }

            distances
        } else {
            let mut distances = Array2::<f64>::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                let row_i = x.row(i);
                for j in (i + 1)..n_samples {
                    // Fill symmetric matrix in the sequential path
                    let dist = squared_euclidean_distance_row(&row_i, &x.row(j));
                    distances[[i, j]] = dist;
                    distances[[j, i]] = dist;
                }
            }
            distances
        }
    }

    fn conditional_probabilities(&self, distances: &Array2<f64>, parallel: bool) -> Array2<f64> {
        let n_samples = distances.nrows();

        // Match perplexity by binary search on sigma per row
        let rows: Vec<Array1<f64>> = if parallel {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let (p_row, _) = binary_search_sigma(&distances.row(i), self.perplexity);
                    p_row
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let (p_row, _) = binary_search_sigma(&distances.row(i), self.perplexity);
                    p_row
                })
                .collect()
        };

        let mut p_conditional = Array2::<f64>::zeros((n_samples, n_samples));
        for (i, row) in rows.into_iter().enumerate() {
            // Assign per-row conditional probabilities
            p_conditional.row_mut(i).assign(&row);
        }

        p_conditional
    }

    fn symmetrize_probabilities(&self, p_conditional: &Array2<f64>) -> Array2<f64> {
        let n_samples = p_conditional.nrows();
        let mut p = Array2::<f64>::zeros((n_samples, n_samples));
        let normalization = 2.0 * n_samples as f64;

        // Average conditional probs to form joint probabilities
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let val = (p_conditional[[i, j]] + p_conditional[[j, i]]) / normalization;
                p[[i, j]] = val;
                p[[j, i]] = val;
            }
        }

        p
    }

    fn compute_num_matrix(&self, y: &Array2<f64>, parallel: bool) -> (Array2<f64>, f64) {
        let n_samples = y.nrows();

        if parallel {
            // Compute 1 / (1 + dist) per pair in parallel
            let rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let row_i = y.row(i);
                    let mut row = vec![0.0; n_samples];
                    for j in 0..n_samples {
                        if i == j {
                            continue;
                        }
                        let dist = squared_euclidean_distance_row(&row_i, &y.row(j));
                        row[j] = 1.0 / (1.0 + dist);
                    }
                    row
                })
                .collect();

            let sum_num: f64 = rows.iter().flat_map(|row| row.iter()).sum();
            let mut num = Array2::<f64>::zeros((n_samples, n_samples));
            for (i, row) in rows.into_iter().enumerate() {
                for (j, val) in row.into_iter().enumerate() {
                    num[[i, j]] = val;
                }
            }

            (num, sum_num)
        } else {
            let mut num = Array2::<f64>::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                let row_i = y.row(i);
                for j in (i + 1)..n_samples {
                    // Fill symmetric Student-t numerator
                    let dist = squared_euclidean_distance_row(&row_i, &y.row(j));
                    let val = 1.0 / (1.0 + dist);
                    num[[i, j]] = val;
                    num[[j, i]] = val;
                }
            }
            let sum_num = num.sum();
            (num, sum_num)
        }
    }

    fn compute_gradient(
        &self,
        y: &Array2<f64>,
        p: &Array2<f64>,
        num: &Array2<f64>,
        sum_num: f64,
        parallel: bool,
    ) -> Array2<f64> {
        let n_samples = y.nrows();
        let n_components = y.ncols();

        if parallel {
            // Compute gradient per row in parallel
            let rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut grad_row = vec![0.0; n_components];
                    for j in 0..n_samples {
                        if i == j {
                            continue;
                        }
                        let q_ij = num[[i, j]] / sum_num;
                        let mult = (p[[i, j]] - q_ij) * num[[i, j]];
                        for d in 0..n_components {
                            grad_row[d] += mult * (y[[i, d]] - y[[j, d]]);
                        }
                    }
                    for d in 0..n_components {
                        // t-SNE gradient scale factor
                        grad_row[d] *= 4.0;
                    }
                    grad_row
                })
                .collect();

            let mut grad = Array2::<f64>::zeros((n_samples, n_components));
            for (i, row) in rows.into_iter().enumerate() {
                for (d, val) in row.into_iter().enumerate() {
                    grad[[i, d]] = val;
                }
            }
            grad
        } else {
            let mut grad = Array2::<f64>::zeros((n_samples, n_components));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i == j {
                        continue;
                    }
                    let q_ij = num[[i, j]] / sum_num;
                    let mult = (p[[i, j]] - q_ij) * num[[i, j]];
                    for d in 0..n_components {
                        grad[[i, d]] += mult * (y[[i, d]] - y[[j, d]]);
                    }
                }
                for d in 0..n_components {
                    // t-SNE gradient scale factor
                    grad[[i, d]] *= 4.0;
                }
            }
            grad
        }
    }

    fn kl_divergence(
        &self,
        p: &Array2<f64>,
        num: &Array2<f64>,
        sum_num: f64,
        parallel: bool,
    ) -> f64 {
        let n_samples = p.nrows();

        if parallel {
            // Sum KL terms per row in parallel
            (0..n_samples)
                .into_par_iter()
                .map(|i| self.kl_divergence_row(p, num, sum_num, i))
                .sum()
        } else {
            (0..n_samples)
                .map(|i| self.kl_divergence_row(p, num, sum_num, i))
                .sum()
        }
    }

    fn kl_divergence_row(&self, p: &Array2<f64>, num: &Array2<f64>, sum_num: f64, i: usize) -> f64 {
        let n_samples = p.nrows();
        let mut kl = 0.0;
        for j in 0..n_samples {
            if i == j {
                continue;
            }
            let p_ij = p[[i, j]];
            if p_ij > 0.0 {
                // Clamp q_ij to avoid log(0)
                let q_ij = (num[[i, j]] / sum_num).max(MIN_Q);
                kl += p_ij * (p_ij / q_ij).ln();
            }
        }
        kl
    }

    fn center_embedding(&self, y: &mut Array2<f64>) -> Result<(), ModelError> {
        // Subtract mean to keep the embedding centered
        let mean = y.mean_axis(Axis(0)).ok_or_else(|| {
            ModelError::ProcessingError("Failed to compute embedding mean".to_string())
        })?;
        for mut row in y.outer_iter_mut() {
            row -= &mean;
        }
        Ok(())
    }

    model_save_and_load_methods!(TSNE);
}

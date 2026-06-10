use crate::error::Error;
use crate::math::squared_euclidean_distance_row;
use crate::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayBase, ArrayViewMut1, Axis, Data, Ix1, Ix2, Zip};
use ndarray_rand::rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Finds the sigma matching a target perplexity for one point's distances, via binary search.
///
/// Returns the resulting probability distribution together with the sigma that achieves the
/// target perplexity. This is t-SNE's per-point precision calibration — an algorithm-specific
/// solver, so it lives with the model rather than in `crate::math`.
fn binary_search_sigma<S>(
    distances: &ArrayBase<S, Ix1>,
    target_perplexity: f64,
) -> (Array1<f64>, f64)
where
    S: Data<Elem = f64>,
{
    let tol = 1e-5;
    // sigma_max starts unbounded: while no finite upper bound is known, the search grows sigma
    // geometrically (the `sigma *= 2.0` path below) until the perplexity overshoots and brackets
    // it, then bisects. sigma_min is a small positive floor (sigma must stay > 0).
    let mut sigma_min: f64 = 1e-20;
    let mut sigma_max: f64 = f64::INFINITY;
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
        let epsilon = 1e-12;

        if sum_p < epsilon {
            // If sum is too small, use uniform distribution
            p.fill(1.0 / n as f64);
        } else {
            p.mapv_inplace(|v| v / sum_p);
        }

        let h: f64 = p
            .iter()
            .map(|&v| if v > 1e-10 { -v * v.ln() } else { 0.0 })
            .sum();
        let current_perplexity = h.exp();
        let diff = current_perplexity - target_perplexity;
        if diff.abs() < tol {
            break;
        }
        // Perplexity increases monotonically with sigma (larger sigma → flatter distribution →
        // higher entropy → higher perplexity). So when the current perplexity is too HIGH we must
        // SHRINK sigma, and when it is too LOW we must GROW it. NOTE: the canonical t-SNE search is
        // written in terms of the precision beta = 1/(2·sigma²), which has the OPPOSITE monotonicity
        // to sigma — these two branches are the mirror image of that reference update rule, because
        // the search variable here is sigma itself. (Swapping them is what broke neighborhood
        // preservation: sigma ran away to its bound and every P(j|i) collapsed to uniform.)
        if diff > 0.0 {
            // Perplexity too high → sigma too large → tighten the upper bound and shrink.
            sigma_max = sigma;
            sigma = (sigma + sigma_min) / 2.0;
        } else {
            // Perplexity too low → sigma too small → raise the lower bound and grow.
            sigma_min = sigma;
            if sigma_max.is_infinite() {
                sigma *= 2.0;
            } else {
                sigma = (sigma + sigma_max) / 2.0;
            }
        }
    }
    (p, sigma)
}

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
/// Lower bound for q_ij to avoid numerical instability.
const MIN_Q: f64 = 1e-12;
/// Threshold for switching to parallel computation in t-SNE.
const TSNE_PARALLEL_THRESHOLD: usize = 2000;

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
/// use rustyml::utils::t_sne::TSNE;
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
    /// - `Result<Self, Error>` - A new TSNE instance or validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any parameter is invalid
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        random_state: Option<u64>,
    ) -> Result<Self, Error> {
        if n_components == 0 {
            return Err(Error::invalid_parameter(
                "n_components",
                "must be greater than 0",
            ));
        }

        if perplexity <= 0.0 || !perplexity.is_finite() {
            return Err(Error::invalid_parameter(
                "perplexity",
                format!("must be positive and finite, got {}", perplexity),
            ));
        }

        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(Error::invalid_parameter(
                "learning_rate",
                format!("must be positive and finite, got {}", learning_rate),
            ));
        }

        if n_iter == 0 {
            return Err(Error::invalid_parameter("n_iter", "must be greater than 0"));
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
    /// - `Result<Array2<f64>, Error>` - Reduced embedding of shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// - `Error::EmptyInput` / `Error::NonFinite` / `Error::InvalidParameter` - If the input matrix is empty, too small, or contains non-finite values
    ///
    /// # Performance
    ///
    /// Uses Rayon parallel computation when `x.nrows()` is above `TSNE_PARALLEL_THRESHOLD` (2000).
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        // Validate inputs before any heavy computation
        self.validate_input(x)?;

        let x_owned = x.to_owned();
        let n_samples = x_owned.nrows();
        // Decide execution mode from sample count
        let use_parallel = n_samples > TSNE_PARALLEL_THRESHOLD;

        // Precompute distances and convert them to joint probabilities
        let distances = self.pairwise_squared_distances(&x_owned, use_parallel);
        let p_conditional = self.conditional_probabilities(&distances, use_parallel);
        let p = self.symmetrize_probabilities(&p_conditional);
        let p_exaggerated = p.mapv(|v| v * EARLY_EXAGGERATION);

        // Initialize embedding and momentum buffer
        let mut y = self.init_embedding(n_samples);
        let mut y_incs = Array2::<f64>::zeros((n_samples, self.n_components));

        // Progress bar reports KL divergence each iteration
        #[cfg(feature = "show_progress")]
        let progress_bar = {
            let pb = crate::create_progress_bar(
                self.n_iter as u64,
                "[{elapsed_precise}] {bar:40} {pos}/{len} | KL Divergence: {msg}",
            );
            pb.set_message(format!("{:.6}", 0.0));
            pb
        };

        let exaggeration_iter = EARLY_EXAGGERATION_ITER.min(self.n_iter);
        #[cfg(feature = "show_progress")]
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

            // Apply momentum SGD update to the embedding:
            // y_incs ← momentum · y_incs − learning_rate · grad, then y ← y + y_incs.
            Zip::from(&mut y_incs).and(&grad).for_each(|inc, &g| {
                *inc = momentum * *inc - self.learning_rate * g;
            });
            y += &y_incs;

            // Keep embedding centered to avoid drift
            self.center_embedding(&mut y)?;

            // Track KL divergence for reporting only
            #[cfg(feature = "show_progress")]
            {
                last_kl = self.kl_divergence(p_use, &num, sum_num, use_parallel);
                progress_bar.set_message(format!("{:.6}", last_kl));
                progress_bar.inc(1);
            }
        }

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message(format!("{:.6}", last_kl));

        Ok(y)
    }

    fn validate_input<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), Error>
    where
        S: Data<Elem = f64>,
    {
        // Shared shape/finiteness checks: non-empty, at least one feature, all finite,
        // and the common minimum-sample guard.
        super::validation::validate_fit_matrix(x)?;
        super::validation::check_min_samples(x, 2, "t-SNE")?;

        // The perplexity bound is t-SNE-specific and stays here.
        if self.perplexity >= x.nrows() as f64 {
            // Perplexity must be less than the number of samples
            return Err(Error::invalid_parameter(
                "perplexity",
                format!(
                    "must be less than number of samples, got perplexity={} with samples={}",
                    self.perplexity,
                    x.nrows()
                ),
            ));
        }

        Ok(())
    }

    fn init_embedding(&self, n_samples: usize) -> Array2<f64> {
        // Seeded RNG for reproducibility if provided
        let mut rng = crate::random::make_rng(self.random_state);

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

        let mut distances = Array2::<f64>::zeros((n_samples, n_samples));
        if parallel {
            // Fill every row in parallel, writing directly into the matrix. The diagonal
            // evaluates to exactly 0.0, matching the symmetric sequential path below.
            Zip::from(distances.outer_iter_mut())
                .and(x.outer_iter())
                .par_for_each(|mut out_row, row_i| {
                    for (j, row_j) in x.outer_iter().enumerate() {
                        out_row[j] = squared_euclidean_distance_row(&row_i, &row_j);
                    }
                });
        } else {
            for i in 0..n_samples {
                let row_i = x.row(i);
                for j in (i + 1)..n_samples {
                    // Exploit symmetry: compute each pair once.
                    let dist = squared_euclidean_distance_row(&row_i, &x.row(j));
                    distances[[i, j]] = dist;
                    distances[[j, i]] = dist;
                }
            }
        }
        distances
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
        let mut num = Array2::<f64>::zeros((n_samples, n_samples));

        if parallel {
            // Fill each row in parallel. The diagonal stays 0 (self-affinity is excluded),
            // so — unlike the distance matrix — the `i == j` term must be skipped explicitly.
            num.outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut out_row)| {
                    let row_i = y.row(i);
                    for (j, row_j) in y.outer_iter().enumerate() {
                        if i != j {
                            let dist = squared_euclidean_distance_row(&row_i, &row_j);
                            out_row[j] = 1.0 / (1.0 + dist);
                        }
                    }
                });
        } else {
            for i in 0..n_samples {
                let row_i = y.row(i);
                for j in (i + 1)..n_samples {
                    // Symmetric Student-t numerator
                    let dist = squared_euclidean_distance_row(&row_i, &y.row(j));
                    let val = 1.0 / (1.0 + dist);
                    num[[i, j]] = val;
                    num[[j, i]] = val;
                }
            }
        }

        // The diagonal is zero in both paths, so this sums only off-diagonal affinities.
        let sum_num = num.sum();
        (num, sum_num)
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
        let mut grad = Array2::<f64>::zeros((n_samples, n_components));

        // Fills gradient row `i` in place from the attractive/repulsive forces between
        // point i and every other point. The t-SNE factor of 4 is folded into `mult`.
        let fill_row = |i: usize, mut row: ArrayViewMut1<f64>| {
            let y_i = y.row(i);
            for j in 0..n_samples {
                if i == j {
                    continue;
                }
                let q_ij = (num[[i, j]] / sum_num).max(MIN_Q);
                let mult = 4.0 * (p[[i, j]] - q_ij) * num[[i, j]];
                let y_j = y.row(j);
                for d in 0..n_components {
                    row[d] += mult * (y_i[d] - y_j[d]);
                }
            }
        };

        if parallel {
            grad.outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, row)| fill_row(i, row));
        } else {
            for (i, row) in grad.outer_iter_mut().enumerate() {
                fill_row(i, row);
            }
        }
        grad
    }

    #[cfg(feature = "show_progress")]
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

    #[cfg(feature = "show_progress")]
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

    fn center_embedding(&self, y: &mut Array2<f64>) -> Result<(), Error> {
        // Subtract mean to keep the embedding centered
        let mean = y
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::computation("Failed to compute embedding mean"))?;
        for mut row in y.outer_iter_mut() {
            row -= &mean;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `binary_search_sigma` on a hand-built distance vector whose first entry is the
    /// self/zero distance.
    ///
    /// Distances = [0.0, 1.0, 1.0, 4.0, 9.0, 16.0] with target perplexity 2.0.
    /// The solver calibrates sigma so the conditional distribution P(j|i) achieves the
    /// target perplexity. Independent of the exact sigma found, the contract is:
    ///   * P sums to 1 (it is normalized each iteration),
    ///   * the achieved perplexity exp(-Σ p ln p) is within the solver's tolerance of
    ///     the target (the loop breaks once |perplexity - target| < 1e-5),
    ///   * the self/zero-distance entry maps to exactly p = 0 (the `d == 0.0` branch).
    #[test]
    fn binary_search_sigma_distribution_and_perplexity() {
        let distances = array![0.0_f64, 1.0, 1.0, 4.0, 9.0, 16.0];
        let target_perplexity = 2.0_f64;

        let (p, _sigma) = binary_search_sigma(&distances, target_perplexity);

        // P is a probability distribution: sums to 1.
        assert_abs_diff_eq!(p.sum(), 1.0_f64, epsilon = 1e-9);

        // Self / zero-distance entry must be exactly 0 (the `d == 0.0` branch).
        assert_abs_diff_eq!(p[0], 0.0_f64, epsilon = 1e-12);

        // Achieved perplexity = exp(-Σ p ln p) (entropy uses the same >1e-10 guard as the
        // solver) must be within tolerance of the target. The loop exits on |diff| < 1e-5;
        // allow a slightly looser bound for accumulated float error.
        let h: f64 = p
            .iter()
            .map(|&v| if v > 1e-10 { -v * v.ln() } else { 0.0 })
            .sum();
        let achieved_perplexity = h.exp();
        assert_abs_diff_eq!(achieved_perplexity, target_perplexity, epsilon = 1e-4);

        // All probabilities are non-negative.
        for &v in p.iter() {
            assert!(v >= 0.0, "probability must be non-negative, got {v}");
        }
    }
}

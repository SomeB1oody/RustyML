//! t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction
//!
//! Provides the [`TSNE`] model with PCA or random embedding initialization, early exaggeration,
//! momentum-based gradient descent, and a per-point sigma solver that calibrates conditional
//! probabilities to a target perplexity

use crate::error::Error;
use crate::math::matmul::{cache_resident, gemm_chunk_rows, gemm_par_auto};
use crate::math::squared_euclidean_distance_row;
use crate::parallel_gates::{cheap_map_f64_parallel_threshold, scan_f64_parallel_min_elems};
use crate::{Deserialize, Serialize};
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, Axis, Data, Ix1, Ix2, Zip, s};
use ndarray_rand::rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Finds the sigma matching a target perplexity for one point's distances, via binary search
///
/// Returns the resulting probability distribution together with the sigma that achieves the
/// target perplexity. This is t-SNE's per-point precision calibration, an algorithm-specific
/// solver, so it lives with the model rather than in `crate::math`
///
/// # Returns
///
/// - `(Array1<f64>, f64)` - the conditional probability distribution and the calibrated sigma
fn binary_search_sigma<S>(
    distances: &ArrayBase<S, Ix1>,
    target_perplexity: f64,
) -> (Array1<f64>, f64)
where
    S: Data<Elem = f64>,
{
    let tol = 1e-5;
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
            // Sum too small: fall back to a uniform distribution
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
        if diff > 0.0 {
            // Perplexity too high, sigma too large: tighten the upper bound and shrink
            sigma_max = sigma;
            sigma = (sigma + sigma_min) / 2.0;
        } else {
            // Perplexity too low, sigma too small: raise the lower bound and grow
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

/// Early exaggeration factor applied to joint probabilities
const EARLY_EXAGGERATION: f64 = 12.0;
/// Number of iterations to apply early exaggeration
const EARLY_EXAGGERATION_ITER: usize = 250;
/// Initial momentum used at the start of optimization
const INITIAL_MOMENTUM: f64 = 0.5;
/// Momentum used after the early exaggeration phase
const FINAL_MOMENTUM: f64 = 0.8;
/// Scale for random initialization of the embedding
const INIT_SCALE: f64 = 1e-4;
/// Additive step for the adaptive gain when the gradient keeps its direction
const GAIN_INCREASE: f64 = 0.2;
/// Multiplicative decay for the adaptive gain when the step oscillates
const GAIN_DECAY: f64 = 0.8;
/// Floor for the per-parameter adaptive gain
const MIN_GAIN: f64 = 0.01;
/// Lower bound for q_ij to avoid numerical instability
const MIN_Q: f64 = 1e-12;
/// Strategy for initializing the low-dimensional embedding before optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Init {
    /// Initialize from the top principal components of the input. Deterministic and gives a
    /// stable, well-spread starting layout, so it is the default and ignores `random_state`
    #[default]
    PCA,
    /// Initialize from small random noise seeded by `random_state`
    Random,
}

/// Gradient computation method for t-SNE optimization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TSNEMethod {
    /// Barnes-Hut approximation. Affinities are kept sparse over each point's nearest neighbors
    /// and the repulsive forces are summarized through a space-partitioning tree, giving roughly
    /// `O(n log n)` per iteration. `angle` (theta) in `[0, 1)` trades accuracy for speed, where
    /// larger values are faster and less accurate. Only supports `n_components <= 3`
    BarnesHut {
        /// Opening angle (theta) in `[0, 1)`; larger is faster and less accurate
        angle: f64,
    },
    /// Exact gradient over all pairs, costing `O(n^2)` per iteration. Supports any `n_components`
    Exact,
}

impl Default for TSNEMethod {
    /// Returns the default method, Barnes-Hut with `angle = 0.5`
    fn default() -> Self {
        TSNEMethod::BarnesHut { angle: 0.5 }
    }
}

/// t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction
///
/// Performs t-SNE optimization with early exaggeration and momentum-based gradient descent
///
/// # Examples
///
/// ```rust
/// use rustyml::utils::t_sne::{TSNE, TSNEMethod};
/// use ndarray::array;
///
/// let tsne = TSNE::new(2, 2.0, 200.0, 250)
///     .unwrap()
///     .with_random_state(42)
///     .with_method(TSNEMethod::Exact)
///     .unwrap();
/// let x = array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]];
/// let embedding = tsne.fit_transform(&x).unwrap();
/// assert_eq!(embedding.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    /// Output embedding dimensions
    n_components: usize,
    /// Effective neighborhood size
    perplexity: f64,
    /// Gradient descent learning rate
    learning_rate: f64,
    /// Number of optimization iterations
    n_iter: usize,
    /// Optional random seed used by the random initialization path
    random_state: Option<u64>,
    /// Embedding initialization strategy
    #[serde(default)]
    init: Init,
    /// Gradient computation method (defaults to [`TSNEMethod::BarnesHut`])
    #[serde(default)]
    method: TSNEMethod,
}

impl Default for TSNE {
    /// Creates a TSNE instance with common default parameters
    ///
    /// # Default Values
    ///
    /// - `n_components` - 2
    /// - `perplexity` - 30.0
    /// - `learning_rate` - 200.0
    /// - `n_iter` - 1000
    /// - `random_state` - None
    /// - `init` - [`Init::PCA`]
    /// - `method` - [`TSNEMethod::BarnesHut`] with `angle = 0.5`
    fn default() -> Self {
        TSNE::new(2, 30.0, 200.0, 1000).expect("Default TSNE parameters should be valid")
    }
}

impl TSNE {
    /// Creates a new TSNE instance with validation
    ///
    /// # Parameters
    ///
    /// - `n_components` - Number of output dimensions (must be greater than 0)
    /// - `perplexity` - Controls effective neighborhood size (must be positive and finite)
    /// - `learning_rate` - Gradient descent learning rate (must be positive and finite)
    /// - `n_iter` - Maximum number of optimization iterations (must be greater than 0)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new TSNE instance or validation error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `n_components` or `n_iter` is 0, or if `perplexity` or
    ///   `learning_rate` is non-positive or not finite
    ///
    /// # Notes
    ///
    /// The gradient method defaults to [`TSNEMethod::BarnesHut`] (with `angle = 0.5`) for
    /// `n_components <= 3` and falls back to [`TSNEMethod::Exact`] otherwise; the embedding is
    /// PCA-initialized and the random path is seeded non-deterministically. Override any of
    /// these with the builder methods (`with_method` returns `Result` because Barnes-Hut
    /// requires a valid angle and a low-dimensional embedding):
    ///
    /// - [`with_random_state`](Self::with_random_state) - fixed seed for the random init path
    /// - [`with_init`](Self::with_init) - embedding initialization ([`Init::PCA`] or [`Init::Random`])
    /// - [`with_method`](Self::with_method) - gradient method ([`TSNEMethod::BarnesHut`] or [`TSNEMethod::Exact`])
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
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

        // Barnes-Hut needs a low-dimensional embedding for the tree, so default to it only when
        // valid; higher-dimensional embeddings fall back to the exact gradient
        let method = if n_components <= 3 {
            TSNEMethod::BarnesHut { angle: 0.5 }
        } else {
            TSNEMethod::Exact
        };

        Ok(Self {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            random_state: None,
            init: Init::PCA,
            method,
        })
    }

    /// Sets a fixed RNG seed for the random initialization path, making it reproducible
    /// (default: `None`; PCA initialization is deterministic regardless)
    ///
    /// # Parameters
    ///
    /// - `seed` - Seed for the random-init RNG
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sets the embedding initialization strategy (default: [`Init::PCA`])
    ///
    /// # Parameters
    ///
    /// - `init` - initialization strategy ([`Init::PCA`] or [`Init::Random`])
    ///
    /// # Returns
    ///
    /// - `Self` - the updated instance, for method chaining
    pub fn with_init(mut self, init: Init) -> Self {
        self.init = init;
        self
    }

    /// Sets the gradient computation method
    /// (default: [`TSNEMethod::BarnesHut`] for `n_components <= 3`, else [`TSNEMethod::Exact`])
    ///
    /// # Parameters
    ///
    /// - `method` - gradient method ([`TSNEMethod::BarnesHut`] or [`TSNEMethod::Exact`])
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - the updated instance, for method chaining
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If a Barnes-Hut `angle` is not in `[0, 1)`, or if Barnes-Hut
    ///   is paired with `n_components > 3`
    pub fn with_method(mut self, method: TSNEMethod) -> Result<Self, Error> {
        // Barnes-Hut needs a valid opening angle and a low-dimensional embedding for the tree
        if let TSNEMethod::BarnesHut { angle } = method {
            if !angle.is_finite() || !(0.0..1.0).contains(&angle) {
                return Err(Error::invalid_parameter(
                    "angle",
                    format!("Barnes-Hut angle must be in [0, 1), got {}", angle),
                ));
            }
            if self.n_components > 3 {
                return Err(Error::invalid_parameter(
                    "n_components",
                    format!(
                        "Barnes-Hut supports at most 3 components, got {}; use TSNEMethod::Exact",
                        self.n_components
                    ),
                ));
            }
        }
        self.method = method;
        Ok(self)
    }

    get_field!(get_n_components, n_components, usize);
    get_field!(get_perplexity, perplexity, f64);
    get_field!(get_learning_rate, learning_rate, f64);
    get_field!(get_n_iter, n_iter, usize);
    get_field!(get_random_state, random_state, Option<u64>);
    get_field!(get_init, init, Init);
    get_field!(get_method, method, TSNEMethod);

    /// Performs t-SNE dimensionality reduction on input data
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
    /// - `Error::EmptyInput` - If the input matrix is empty
    /// - `Error::NonFinite` - If the input contains non-finite values
    /// - `Error::InvalidInput` - If there are fewer than 2 samples
    /// - `Error::InvalidParameter` - If `perplexity` is not less than the number of samples
    ///
    /// # Performance
    ///
    /// Parallelizes when the pairwise work clears the calibrated class gates (see
    /// `crate::parallel_gates`); the GEMMs gate themselves inside `gemm_par_auto`
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>, Error>
    where
        S: Data<Elem = f64>,
    {
        self.validate_input(x)?;

        let x_owned = x.to_owned();

        // Initialize the embedding, then optimize with the configured method
        let y = self.init_embedding(&x_owned);
        match self.method {
            TSNEMethod::Exact => self.optimize_exact(&x_owned, y),
            TSNEMethod::BarnesHut { angle } => self.optimize_barnes_hut(&x_owned, y, angle),
        }
    }

    /// Runs the exact `O(n^2)` optimization over the dense joint-probability matrix
    fn optimize_exact(&self, x: &Array2<f64>, mut y: Array2<f64>) -> Result<Array2<f64>, Error> {
        let n_samples = x.nrows();
        // Cheap-map class gate over the [n, n] matrices this path fills each iteration
        let use_parallel =
            n_samples.saturating_mul(n_samples) >= cheap_map_f64_parallel_threshold();

        // Precompute distances and convert them to joint probabilities
        let distances = self.pairwise_squared_distances(x, use_parallel);
        let p_conditional = self.conditional_probabilities(&distances, use_parallel);
        let p = self.symmetrize_probabilities(&p_conditional);
        let p_exaggerated = p.mapv(|v| v * EARLY_EXAGGERATION);

        let mut y_incs = Array2::<f64>::zeros((n_samples, self.n_components));
        let mut gains = Array2::<f64>::ones((n_samples, self.n_components));

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
            // Early exaggeration for the first phase only
            let p_use = if iter < exaggeration_iter {
                &p_exaggerated
            } else {
                &p
            };

            // Student-t affinities and gradient
            let (num, sum_num) = self.compute_num_matrix(&y, use_parallel);
            let grad = self.compute_gradient(&y, p_use, &num, sum_num, use_parallel);

            let momentum = if iter < exaggeration_iter {
                INITIAL_MOMENTUM
            } else {
                FINAL_MOMENTUM
            };
            self.apply_gradient_step(&mut y, &mut y_incs, &mut gains, &grad, momentum)?;

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

    /// Runs the Barnes-Hut optimization with sparse affinities and tree-summarized repulsion
    fn optimize_barnes_hut(
        &self,
        x: &Array2<f64>,
        mut y: Array2<f64>,
        angle: f64,
    ) -> Result<Array2<f64>, Error> {
        let n_samples = x.nrows();
        // Scan-class gate: the dominant gated work is the one-off neighbor search, n tasks of an
        // O(n) projection-row scan each
        let use_parallel = n_samples.saturating_mul(n_samples) >= scan_f64_parallel_min_elems();

        // Sparse symmetric joint probabilities over each point's nearest neighbors
        let adj = self.neighbor_probabilities(x, use_parallel);

        let mut y_incs = Array2::<f64>::zeros((n_samples, self.n_components));
        let mut gains = Array2::<f64>::ones((n_samples, self.n_components));

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
            let exaggeration = if iter < exaggeration_iter {
                EARLY_EXAGGERATION
            } else {
                1.0
            };

            let (grad, _z) = self.barnes_hut_gradient(&y, &adj, angle, exaggeration, use_parallel);

            let momentum = if iter < exaggeration_iter {
                INITIAL_MOMENTUM
            } else {
                FINAL_MOMENTUM
            };
            self.apply_gradient_step(&mut y, &mut y_incs, &mut gains, &grad, momentum)?;

            #[cfg(feature = "show_progress")]
            {
                last_kl = self.barnes_hut_kl(&y, &adj, _z, exaggeration);
                progress_bar.set_message(format!("{:.6}", last_kl));
                progress_bar.inc(1);
            }
        }

        #[cfg(feature = "show_progress")]
        progress_bar.finish_with_message(format!("{:.6}", last_kl));

        Ok(y)
    }

    /// Applies one momentum SGD step with the adaptive per-parameter gains, then recenters
    ///
    /// Updates `gains` with Jacobs' delta-bar-delta heuristic, accumulates the momentum increment
    /// into `y_incs`, advances `y`, and subtracts the mean to keep the embedding centered
    fn apply_gradient_step(
        &self,
        y: &mut Array2<f64>,
        y_incs: &mut Array2<f64>,
        gains: &mut Array2<f64>,
        grad: &Array2<f64>,
        momentum: f64,
    ) -> Result<(), Error> {
        Zip::from(&mut *gains)
            .and(grad)
            .and(&*y_incs)
            .for_each(|gain, &g, &inc| {
                *gain = if g * inc > 0.0 {
                    *gain * GAIN_DECAY
                } else {
                    *gain + GAIN_INCREASE
                };
                if *gain < MIN_GAIN {
                    *gain = MIN_GAIN;
                }
            });

        // y_incs = momentum*y_incs - learning_rate*(gains*grad), then y += y_incs
        Zip::from(&mut *y_incs)
            .and(&*gains)
            .and(grad)
            .for_each(|inc, &gain, &g| {
                *inc = momentum * *inc - self.learning_rate * gain * g;
            });
        *y += &*y_incs;

        // Keep the embedding centered to avoid drift
        self.center_embedding(y)
    }

    /// Builds the symmetric sparse joint probabilities over each point's nearest neighbors
    ///
    /// For each point the `k = min(n - 1, ceil(3 * perplexity) + 1)` nearest neighbors are found by
    /// a brute-force search, the per-point sigma is calibrated over those neighbors, and the
    /// conditional probabilities are symmetrized into joint probabilities. Returns one neighbor
    /// list per point, each entry holding `(neighbor_index, p_ij)`. The neighbor search is `O(n^2)`
    /// and runs once, while the per-iteration repulsion stays `O(n log n)`
    fn neighbor_probabilities(&self, x: &Array2<f64>, parallel: bool) -> Vec<Vec<(usize, f64)>> {
        let n_samples = x.nrows();
        let k = (((3.0 * self.perplexity).ceil() as usize) + 1)
            .min(n_samples - 1)
            .max(1);

        // Squared distances come from the `||x_i||^2 + ||x_j||^2 - 2 x_i.x_j` identity
        let x_sq = x.map_axis(Axis(1), |row| row.dot(&row));

        // Find the k nearest neighbors of point i with their squared distances, calibrate sigma
        // over them, and return the conditional probabilities
        let conditional_row = |i: usize, proj_row: ArrayView1<f64>| -> (Vec<usize>, Array1<f64>) {
            let mut dists: Vec<(f64, usize)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = (x_sq[i] + x_sq[j] - 2.0 * proj_row[j]).max(0.0);
                    (dist, j)
                })
                .collect();

            // Partial-select the k smallest by (distance, index) for a deterministic neighbor set
            if dists.len() > k {
                dists.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                dists.truncate(k);
            }
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

            let neighbor_dist = Array1::from_iter(dists.iter().map(|&(d, _)| d));
            let (p_row, _) = binary_search_sigma(&neighbor_dist, self.perplexity);
            let neighbor_idx = dists.into_iter().map(|(_, j)| j).collect();
            (neighbor_idx, p_row)
        };

        let conditional: Vec<(Vec<usize>, Array1<f64>)> =
            if cache_resident::<f64>(n_samples, x.ncols()) {
                let swarm_row = |i: usize| {
                    let projections = x.dot(&x.row(i));
                    conditional_row(i, projections.view())
                };
                if parallel {
                    (0..n_samples).into_par_iter().map(swarm_row).collect()
                } else {
                    (0..n_samples).map(swarm_row).collect()
                }
            } else {
                let chunk_rows = gemm_chunk_rows(n_samples);
                let mut conditional = Vec::with_capacity(n_samples);
                for chunk_start in (0..n_samples).step_by(chunk_rows) {
                    let chunk_end = (chunk_start + chunk_rows).min(n_samples);
                    let projections =
                        gemm_par_auto(&x.slice(s![chunk_start..chunk_end, ..]), &x.t());
                    if parallel {
                        let chunk: Vec<(Vec<usize>, Array1<f64>)> = (chunk_start..chunk_end)
                            .into_par_iter()
                            .map(|i| conditional_row(i, projections.row(i - chunk_start)))
                            .collect();
                        conditional.extend(chunk);
                    } else {
                        conditional.extend(
                            (chunk_start..chunk_end)
                                .map(|i| conditional_row(i, projections.row(i - chunk_start))),
                        );
                    }
                }
                conditional
            };

        // Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2n), accumulated from both directed edges
        let norm = 2.0 * n_samples as f64;
        let mut adjacency: Vec<AHashMap<usize, f64>> = vec![AHashMap::new(); n_samples];
        for (i, (neighbor_idx, p_row)) in conditional.iter().enumerate() {
            for (slot, &j) in neighbor_idx.iter().enumerate() {
                let w = p_row[slot] / norm;
                *adjacency[i].entry(j).or_insert(0.0) += w;
                *adjacency[j].entry(i).or_insert(0.0) += w;
            }
        }

        // Sort each neighbor list by index so the downstream force summation order is fixed
        adjacency
            .into_iter()
            .map(|row| {
                let mut neighbors: Vec<(usize, f64)> = row.into_iter().collect();
                neighbors.sort_unstable_by_key(|&(j, _)| j);
                neighbors
            })
            .collect()
    }

    /// Computes the Barnes-Hut gradient and the repulsive normalization term `Z`
    ///
    /// The attractive part sums the sparse affinities over each point's neighbors, while the
    /// repulsive part is summarized through the space-partitioning tree. The gradient matches the
    /// exact path's scale (the factor of 4 is folded in), so the same learning rate fits both
    /// methods. `Z` is summed sequentially so the result is reproducible
    fn barnes_hut_gradient(
        &self,
        y: &Array2<f64>,
        adj: &[Vec<(usize, f64)>],
        angle: f64,
        exaggeration: f64,
        parallel: bool,
    ) -> (Array2<f64>, f64) {
        let n_samples = y.nrows();
        let dim = self.n_components;
        let theta2 = angle * angle;
        let tree = build_sp_tree(y);

        // Per point: attractive force, repulsive force, and the point's normalizer contribution
        let per_point = |i: usize| -> ([f64; 3], [f64; 3], f64) {
            let mut query = [0.0_f64; 3];
            for (d, slot) in query.iter_mut().enumerate().take(dim) {
                *slot = y[[i, d]];
            }

            // Repulsive forces summarized by the tree
            let mut neg_f = [0.0_f64; 3];
            let mut sum_q = 0.0;
            tree.repulsive_force(i, &query[..dim], theta2, &mut neg_f[..dim], &mut sum_q);

            // Attractive forces over the sparse neighbors
            let mut pos_f = [0.0_f64; 3];
            for &(j, p_ij) in &adj[i] {
                let mut d2 = 0.0;
                let mut diff = [0.0_f64; 3];
                for d in 0..dim {
                    diff[d] = query[d] - y[[j, d]];
                    d2 += diff[d] * diff[d];
                }
                let mult = exaggeration * p_ij / (1.0 + d2);
                for d in 0..dim {
                    pos_f[d] += mult * diff[d];
                }
            }
            (pos_f, neg_f, sum_q)
        };

        let results: Vec<([f64; 3], [f64; 3], f64)> = if parallel {
            (0..n_samples).into_par_iter().map(per_point).collect()
        } else {
            (0..n_samples).map(per_point).collect()
        };

        // Sum the normalizer sequentially so Z does not depend on thread scheduling
        let mut z = 0.0;
        for r in &results {
            z += r.2;
        }
        let z = z.max(MIN_Q);

        let mut grad = Array2::<f64>::zeros((n_samples, dim));
        for (i, (pos_f, neg_f, _)) in results.iter().enumerate() {
            for d in 0..dim {
                grad[[i, d]] = 4.0 * (pos_f[d] - neg_f[d] / z);
            }
        }
        (grad, z)
    }

    /// Approximates the KL divergence over the sparse affinity edges for progress reporting
    #[cfg(feature = "show_progress")]
    fn barnes_hut_kl(
        &self,
        y: &Array2<f64>,
        adj: &[Vec<(usize, f64)>],
        z: f64,
        exaggeration: f64,
    ) -> f64 {
        let dim = self.n_components;
        let mut kl = 0.0;
        for (i, neighbors) in adj.iter().enumerate() {
            for &(j, p_ij) in neighbors {
                let p = p_ij * exaggeration;
                if p > 0.0 {
                    let mut d2 = 0.0;
                    for d in 0..dim {
                        let diff = y[[i, d]] - y[[j, d]];
                        d2 += diff * diff;
                    }
                    let q = ((1.0 / (1.0 + d2)) / z).max(MIN_Q);
                    kl += p * (p / q).ln();
                }
            }
        }
        kl
    }

    fn validate_input<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<(), Error>
    where
        S: Data<Elem = f64>,
    {
        // Shared shape/finiteness checks plus the common minimum-sample guard
        super::validation::validate_fit_matrix(x)?;
        super::validation::check_min_samples(x, 2, "t-SNE")?;

        // The perplexity bound is t-SNE-specific
        if self.perplexity >= x.nrows() as f64 {
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

    /// Builds the initial embedding according to the configured [`Init`] strategy
    ///
    /// PCA initialization is deterministic and falls back to random initialization when the
    /// input has fewer features than components or its leading component is degenerate
    fn init_embedding(&self, x: &Array2<f64>) -> Array2<f64> {
        match self.init {
            Init::PCA => self
                .pca_init(x)
                .unwrap_or_else(|| self.random_init(x.nrows())),
            Init::Random => self.random_init(x.nrows()),
        }
    }

    /// Initializes the embedding from the top principal components of `x`
    ///
    /// The result is rescaled so the leading component has standard deviation [`INIT_SCALE`],
    /// which keeps early gradients small. Returns `None` when PCA cannot supply `n_components`
    /// directions or the leading component has zero spread, letting the caller fall back to random
    fn pca_init(&self, x: &Array2<f64>) -> Option<Array2<f64>> {
        if x.ncols() < self.n_components {
            return None;
        }
        let mut pca = crate::utils::pca::PCA::new(self.n_components).ok()?;
        let mut embedding = pca.fit_transform(x).ok()?;

        // Rescale to a small spread on the first component, matching common t-SNE practice
        let col0 = embedding.column(0);
        let count = col0.len() as f64;
        let mean0 = col0.sum() / count;
        let var0 = col0.iter().map(|&v| (v - mean0) * (v - mean0)).sum::<f64>() / count;
        let std0 = var0.sqrt();
        if !std0.is_finite() || std0 <= 0.0 {
            return None;
        }
        embedding.mapv_inplace(|v| v / std0 * INIT_SCALE);
        Some(embedding)
    }

    /// Initializes the embedding from small random noise seeded by `random_state`
    fn random_init(&self, n_samples: usize) -> Array2<f64> {
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
        // D[i, j] = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j
        let x_sq = x.map_axis(Axis(1), |row| row.dot(&row));
        let mut distances = gemm_par_auto(x, &x.t());

        let fill_row = |i: usize, mut row: ArrayViewMut1<f64>| {
            let xi_sq = x_sq[i];
            for (j, v) in row.iter_mut().enumerate() {
                *v = (xi_sq + x_sq[j] - 2.0 * *v).max(0.0);
            }
            row[i] = 0.0;
        };

        if parallel {
            distances
                .outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, row)| fill_row(i, row));
        } else {
            for (i, row) in distances.outer_iter_mut().enumerate() {
                fill_row(i, row);
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
            p_conditional.row_mut(i).assign(&row);
        }

        p_conditional
    }

    fn symmetrize_probabilities(&self, p_conditional: &Array2<f64>) -> Array2<f64> {
        let n_samples = p_conditional.nrows();
        let mut p = Array2::<f64>::zeros((n_samples, n_samples));
        let normalization = 2.0 * n_samples as f64;

        // Average conditional probabilities to form joint probabilities
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
            let partial: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let row_i = y.row(i);
                    ((i + 1)..n_samples)
                        .map(|j| {
                            let dist = squared_euclidean_distance_row(&row_i, &y.row(j));
                            1.0 / (1.0 + dist)
                        })
                        .collect()
                })
                .collect();
            for (i, vals) in partial.into_iter().enumerate() {
                for (offset, val) in vals.into_iter().enumerate() {
                    let j = i + 1 + offset;
                    num[[i, j]] = val;
                    num[[j, i]] = val;
                }
            }
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

        // The diagonal is zero in both paths, so this sums only off-diagonal affinities
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

        // grad_i = 4 * sum_j (p_ij - q_ij) * num_ij * (y_i - y_j) factors into GEMM form
        let mut w = Array2::<f64>::zeros((n_samples, n_samples));
        let fill_w_row = |i: usize, mut w_row: ArrayViewMut1<f64>| {
            for j in 0..n_samples {
                let q_ij = (num[[i, j]] / sum_num).max(MIN_Q);
                w_row[j] = (p[[i, j]] - q_ij) * num[[i, j]];
            }
        };

        if parallel {
            w.outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, row)| fill_w_row(i, row));
        } else {
            for (i, row) in w.outer_iter_mut().enumerate() {
                fill_w_row(i, row);
            }
        }

        let row_sums = w.sum_axis(Axis(1));
        let weighted_y = gemm_par_auto(&w, y);

        // 4 * (s_i * y_i - (W * Y)_i), assembled with broadcast elementwise ops (O(n * d))
        (y * &row_sums.insert_axis(Axis(1)) - weighted_y) * 4.0
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

        // Deterministic blocked fold over the per-row KL terms: a bare rayon `sum` would
        // group by scheduling and make even this displayed value thread-count-dependent
        crate::math::reduction::det_reduce_range(
            n_samples,
            parallel,
            |range| {
                range
                    .map(|i| self.kl_divergence_row(p, num, sum_num, i))
                    .sum::<f64>()
            },
            |a, b| a + b,
            0.0,
        )
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
        // Subtract the mean to keep the embedding centered
        let mean = y
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::computation("Failed to compute embedding mean"))?;
        for mut row in y.outer_iter_mut() {
            row -= &mean;
        }
        Ok(())
    }
}

/// Maximum subdivision depth for the Barnes-Hut tree, bounding recursion on coincident points
const BH_MAX_DEPTH: usize = 50;

/// A node of the space-partitioning tree used for the Barnes-Hut repulsive force approximation
///
/// Each node owns a rectangular cell, the running center of mass and count of the points it
/// contains, and (once subdivided) its `2^dim` child cells. Supports up to 3 embedding dimensions
struct SPNode {
    /// Number of embedding dimensions
    dim: usize,
    /// Geometric center of this cell per dimension
    center: Vec<f64>,
    /// Half-width of this cell per dimension
    half_width: Vec<f64>,
    /// Number of points contained in this cell
    cum_size: usize,
    /// Center of mass of the contained points
    center_of_mass: Vec<f64>,
    /// Index and coordinates of the single point held while this node stays a leaf
    leaf_point: Option<(usize, Vec<f64>)>,
    /// Child cells, empty until the node subdivides, then of length `2^dim`
    children: Vec<SPNode>,
}

impl SPNode {
    /// Creates an empty leaf cell with the given center and half-width
    fn new(center: Vec<f64>, half_width: Vec<f64>, dim: usize) -> Self {
        SPNode {
            dim,
            center,
            half_width,
            cum_size: 0,
            center_of_mass: vec![0.0; dim],
            leaf_point: None,
            children: Vec::new(),
        }
    }

    /// Returns the child cell index whose region contains `p`
    fn child_index(&self, p: &[f64]) -> usize {
        let mut idx = 0;
        for (d, (&pd, &cd)) in p.iter().zip(self.center.iter()).enumerate() {
            if pd > cd {
                idx |= 1 << d;
            }
        }
        idx
    }

    /// Splits this leaf into `2^dim` empty child cells
    fn subdivide(&mut self) {
        let n_children = 1usize << self.dim;
        self.children = Vec::with_capacity(n_children);
        for c in 0..n_children {
            let mut center = vec![0.0; self.dim];
            let mut half_width = vec![0.0; self.dim];
            for d in 0..self.dim {
                half_width[d] = self.half_width[d] / 2.0;
                center[d] = if (c >> d) & 1 == 1 {
                    self.center[d] + half_width[d]
                } else {
                    self.center[d] - half_width[d]
                };
            }
            self.children
                .push(SPNode::new(center, half_width, self.dim));
        }
    }

    /// Inserts a point into the subtree rooted at this node
    fn insert(&mut self, idx: usize, p: &[f64], depth: usize) {
        // Update the running count and center of mass
        self.cum_size += 1;
        let cs = self.cum_size as f64;
        for (com, &pd) in self.center_of_mass.iter_mut().zip(p.iter()) {
            *com = *com * ((cs - 1.0) / cs) + pd / cs;
        }

        if self.children.is_empty() {
            match self.leaf_point.take() {
                None => {
                    // First point lands directly in this empty leaf
                    self.leaf_point = Some((idx, p.to_vec()));
                    return;
                }
                Some((old_idx, old_p)) => {
                    // The depth cap stops coincident points from recursing without end
                    if depth >= BH_MAX_DEPTH {
                        self.leaf_point = Some((old_idx, old_p));
                        return;
                    }
                    self.subdivide();
                    let oc = self.child_index(&old_p);
                    self.children[oc].insert(old_idx, &old_p, depth + 1);
                }
            }
        }

        let ci = self.child_index(p);
        self.children[ci].insert(idx, p, depth + 1);
    }

    /// Accumulates the Barnes-Hut repulsive force and normalization term for `query`
    ///
    /// `target_idx` is the index of the query point so its own leaf is skipped, and `theta2` is the
    /// squared opening angle. Adds the unnormalized repulsive force into `neg_f` and the Student-t
    /// normalizer into `sum_q`
    fn repulsive_force(
        &self,
        target_idx: usize,
        query: &[f64],
        theta2: f64,
        neg_f: &mut [f64],
        sum_q: &mut f64,
    ) {
        if self.cum_size == 0 {
            return;
        }
        // Skip the leaf that holds the query point itself
        if self.children.is_empty()
            && let Some((idx, _)) = &self.leaf_point
            && *idx == target_idx
            && self.cum_size == 1
        {
            return;
        }

        let mut diff = [0.0_f64; 3];
        let mut d2 = 0.0;
        for d in 0..self.dim {
            diff[d] = query[d] - self.center_of_mass[d];
            d2 += diff[d] * diff[d];
        }

        // Largest side length of this cell
        let mut max_width = 0.0_f64;
        for d in 0..self.dim {
            let w = 2.0 * self.half_width[d];
            if w > max_width {
                max_width = w;
            }
        }

        // Treat the cell as one summary when it is a leaf or distant enough (width/dist < theta)
        if self.children.is_empty() || max_width * max_width < theta2 * d2 {
            let inv = 1.0 / (1.0 + d2);
            let mult = self.cum_size as f64 * inv;
            *sum_q += mult;
            let force = mult * inv;
            for d in 0..self.dim {
                neg_f[d] += force * diff[d];
            }
        } else {
            for child in &self.children {
                child.repulsive_force(target_idx, query, theta2, neg_f, sum_q);
            }
        }
    }
}

/// Builds the Barnes-Hut tree over the current embedding `y`
fn build_sp_tree(y: &Array2<f64>) -> SPNode {
    let dim = y.ncols();
    let n = y.nrows();

    // Axis-aligned bounding box of all points
    let mut min = vec![f64::INFINITY; dim];
    let mut max = vec![f64::NEG_INFINITY; dim];
    for row in y.outer_iter() {
        for d in 0..dim {
            let v = row[d];
            if v < min[d] {
                min[d] = v;
            }
            if v > max[d] {
                max[d] = v;
            }
        }
    }

    let mut center = vec![0.0; dim];
    let mut half_width = vec![0.0; dim];
    for d in 0..dim {
        center[d] = (min[d] + max[d]) / 2.0;
        // Pad the half-width so every point lies strictly inside the root cell
        half_width[d] = ((max[d] - min[d]) / 2.0).max(1e-12) + 1e-12;
    }

    let mut root = SPNode::new(center, half_width, dim);
    for i in 0..n {
        let p: Vec<f64> = (0..dim).map(|d| y[[i, d]]).collect();
        root.insert(i, &p, 0);
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `binary_search_sigma` returns a normalized P that hits the target perplexity, with the
    /// zero-distance entry mapped to exactly 0
    #[test]
    fn binary_search_sigma_distribution_and_perplexity() {
        let distances = array![0.0_f64, 1.0, 1.0, 4.0, 9.0, 16.0];
        let target_perplexity = 2.0_f64;

        let (p, _sigma) = binary_search_sigma(&distances, target_perplexity);

        // P is a probability distribution: sums to 1
        assert_abs_diff_eq!(p.sum(), 1.0_f64, epsilon = 1e-9);

        // Self / zero-distance entry must be exactly 0 (the `d == 0.0` branch)
        assert_abs_diff_eq!(p[0], 0.0_f64, epsilon = 1e-12);

        // Achieved perplexity = exp(-sum p ln p) must be within tolerance of the target,
        // looser than the solver's 1e-5 exit bound to absorb accumulated float error
        let h: f64 = p
            .iter()
            .map(|&v| if v > 1e-10 { -v * v.ln() } else { 0.0 })
            .sum();
        let achieved_perplexity = h.exp();
        assert_abs_diff_eq!(achieved_perplexity, target_perplexity, epsilon = 1e-4);

        // All probabilities are non-negative
        for &v in p.iter() {
            assert!(v >= 0.0, "probability must be non-negative, got {v}");
        }
    }

    /// The Barnes-Hut tree root holds every point and its center of mass equals the point mean
    #[test]
    fn sp_tree_root_aggregates_all_points() {
        let y = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let root = build_sp_tree(&y);

        assert_eq!(root.cum_size, 4);
        // Mean of the four corners is (1.0, 1.0)
        assert_abs_diff_eq!(root.center_of_mass[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(root.center_of_mass[1], 1.0, epsilon = 1e-12);
    }

    /// The sparse joint probabilities are symmetric and sum to 1
    #[test]
    fn neighbor_probabilities_are_symmetric_and_normalized() {
        let tsne = TSNE::new(2, 2.0, 200.0, 100)
            .unwrap()
            .with_random_state(0)
            .with_init(Init::PCA)
            .with_method(TSNEMethod::Exact)
            .unwrap();
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0]
        ];
        let adj = tsne.neighbor_probabilities(&x, false);

        // Look up p_ij in point i's neighbor list
        let lookup = |i: usize, j: usize| -> f64 {
            adj[i]
                .iter()
                .find(|&&(k, _)| k == j)
                .map(|&(_, p)| p)
                .unwrap_or(0.0)
        };

        let mut total = 0.0;
        for (i, neighbors) in adj.iter().enumerate() {
            for &(j, p_ij) in neighbors {
                total += p_ij;
                assert_abs_diff_eq!(p_ij, lookup(j, i), epsilon = 1e-12);
            }
        }
        // The joint distribution sums to 1 over all directed pairs
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-9);
    }
}

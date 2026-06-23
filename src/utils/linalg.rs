//! Shared linear-algebra routines for the `utils` transformers
//!
//! Power-iteration and Lanczos solvers that extract the leading eigenpairs of a
//! symmetric matrix, so PCA and Kernel PCA share one implementation instead of
//! carrying near-identical copies

use crate::error::Error;
use crate::math::matmul::gemv_par_auto;
use crate::parallel_gates::cheap_map_f64_parallel_threshold;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Builds a random unit vector of length `n`, falling back to a uniform unit
/// vector if the random draw is numerically zero
///
/// # Parameters
///
/// - `n` - Length of the vector
/// - `rng` - Random number generator for the draw
///
/// # Returns
///
/// - `Array1<f64>` - Unit vector of length `n`
fn random_unit_vector(n: usize, rng: &mut StdRng) -> Array1<f64> {
    let mut v = Array1::<f64>::from_shape_fn(n, |_| rng.random_range(-1.0..1.0));
    let norm = v.dot(&v).sqrt();
    if norm <= f64::EPSILON {
        v.fill(1.0 / (n as f64).sqrt());
    } else {
        v /= norm;
    }
    v
}

/// Computes the dominant eigenpair of a symmetric matrix via power iteration
///
/// # Parameters
///
/// - `matrix` - Symmetric input matrix
/// - `rng` - Random number generator for the starting vector
/// - `max_iter` - Maximum power-iteration steps
/// - `tol` - Convergence tolerance on the eigenvalue estimate
///
/// # Returns
///
/// The unit eigenvector together with its eigenvalue, estimated as the Rayleigh
/// quotient `v^T M v`
///
/// # Errors
///
/// - [`Error::NotConverged`] - If the iteration fails to converge to a
///   finite, non-degenerate eigenpair
/// - [`Error::NonFinite`] - If the iteration produces a non-finite eigenvalue
fn dominant_eigenpair(
    matrix: &Array2<f64>,
    rng: &mut StdRng,
    max_iter: usize,
    tol: f64,
) -> Result<(Array1<f64>, f64), Error> {
    let n = matrix.ncols();
    let mut v = random_unit_vector(n, rng);

    let mut prev_lambda = 0.0;
    for _ in 0..max_iter {
        // One matvec per step
        let w = gemv_par_auto(matrix, &v);
        let lambda = v.dot(&w);
        if !lambda.is_finite() {
            return Err(Error::non_finite("power iteration eigenvalue"));
        }
        let w_norm = w.dot(&w).sqrt();
        if w_norm <= f64::EPSILON || !w_norm.is_finite() {
            return Err(Error::not_converged("Power iteration failed to converge"));
        }
        if (lambda - prev_lambda).abs() < tol {
            return Ok((v, lambda));
        }
        prev_lambda = lambda;
        // Advance toward the dominant eigenvector
        v = &w / w_norm;
    }

    let lambda = v.dot(&gemv_par_auto(matrix, &v));
    if !lambda.is_finite() {
        return Err(Error::non_finite("power iteration eigenvalue"));
    }
    Ok((v, lambda))
}

/// Extracts the top-`k` eigenpairs of a symmetric matrix using power iteration
/// with Hotelling deflation
///
/// The matrix is taken by value because each extracted component is deflated
/// out of it in place. Eigenpairs come back in descending eigenvalue order as
/// parallel vectors: `eigenvectors[i]` is the unit eigenvector for
/// `eigenvalues[i]`. Callers arrange those eigenvectors into rows or columns as
/// their own layout requires
///
/// # Parameters
///
/// - `matrix` - Symmetric input matrix (e.g. a covariance or centered kernel matrix)
/// - `k` - Number of leading eigenpairs to extract
/// - `seed` - Seed for the random initialization, making the result deterministic
/// - `max_iter` - Maximum power-iteration steps per component
/// - `tol` - Convergence tolerance on the eigenvalue estimate
///
/// # Returns
///
/// The eigenvalues and their parallel unit eigenvectors, in descending
/// eigenvalue order
///
/// # Errors
///
/// - [`Error::NotConverged`] - If any component fails to converge
/// - [`Error::NonFinite`] - If a component produces a non-finite eigenvalue
pub(super) fn top_eigenpairs_power_iteration(
    mut matrix: Array2<f64>,
    k: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<Array1<f64>>), Error> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);

    for _ in 0..k {
        let (vector, value) = dominant_eigenpair(&matrix, &mut rng, max_iter, tol)?;
        // Deflate the extracted component so the next iteration surfaces the next one: M = M - lambda v v^T
        deflate_rank_one(&mut matrix, &vector, value);
        eigenvalues.push(value);
        eigenvectors.push(vector);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Subtracts the rank-1 Hotelling term `value * v vᵀ` from `matrix` in place
///
/// Applies the deflation row by row (`row_i -= value * v_i * v`) instead of forming the dense
/// `n x n` outer product first, avoiding that temporary allocation. Rows are updated in parallel
/// once the `n^2` element work clears the cheap-map gate, and serially below it
///
/// # Parameters
///
/// - `matrix` - Symmetric matrix being deflated, modified in place
/// - `v` - Unit eigenvector of the component to remove
/// - `value` - Eigenvalue of the component to remove
fn deflate_rank_one(matrix: &mut Array2<f64>, v: &Array1<f64>, value: f64) {
    let n = matrix.nrows();
    if n.saturating_mul(n) >= cheap_map_f64_parallel_threshold() {
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                row.scaled_add(-value * v[i], v);
            });
    } else {
        for (i, mut row) in matrix.axis_iter_mut(Axis(0)).enumerate() {
            row.scaled_add(-value * v[i], v);
        }
    }
}

/// Extracts the top-`k` eigenpairs of a symmetric matrix using the Lanczos
/// algorithm with full reorthogonalization
///
/// Lanczos builds a Krylov subspace, reduces the problem to a small symmetric
/// tridiagonal eigenproblem, and maps the leading Ritz pairs back. For the
/// dominant eigenpairs of a symmetric matrix (such as a centered kernel matrix)
/// it converges faster and more stably than deflated power iteration. This is a
/// single Lanczos pass without implicit restarts
///
/// Eigenpairs come back in descending eigenvalue order as parallel vectors:
/// `eigenvectors[i]` is the unit eigenvector for `eigenvalues[i]`
///
/// # Parameters
///
/// - `matrix` - Symmetric input matrix
/// - `k` - Number of leading eigenpairs to extract
/// - `seed` - Seed for the random starting vector, making the result deterministic
///
/// # Returns
///
/// The eigenvalues and their parallel unit eigenvectors, in descending
/// eigenvalue order
///
/// # Errors
///
/// - [`Error::NotConverged`] - If the Krylov subspace collapses immediately
pub(super) fn top_eigenpairs_lanczos(
    matrix: &Array2<f64>,
    k: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<Array1<f64>>), Error> {
    let n = matrix.ncols();
    // Krylov dimension: larger than k for accurate leading Ritz pairs, capped at the matrix size
    let m = (2 * k + 20).min(n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut lanczos_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alphas: Vec<f64> = Vec::with_capacity(m);
    let mut betas: Vec<f64> = Vec::with_capacity(m);

    let mut v = random_unit_vector(n, &mut rng);
    let mut v_prev: Option<Array1<f64>> = None;
    let mut beta_prev = 0.0;

    for _ in 0..m {
        // Three-term recurrence: w = A v - alpha v - beta_prev v_prev
        let mut w = gemv_par_auto(matrix, &v);
        let alpha = v.dot(&w);
        w.scaled_add(-alpha, &v);
        if let Some(ref vp) = v_prev {
            w.scaled_add(-beta_prev, vp);
        }

        // Full reorthogonalization against every prior Lanczos vector, repeated once for stability
        for _ in 0..2 {
            for u in lanczos_vectors.iter() {
                let proj = w.dot(u);
                w.scaled_add(-proj, u);
            }
            let proj_v = w.dot(&v);
            w.scaled_add(-proj_v, &v);
        }

        lanczos_vectors.push(v.clone());
        alphas.push(alpha);

        let beta = w.dot(&w).sqrt();
        if beta <= 1e-12 || !beta.is_finite() {
            // Reached an invariant subspace; no further directions to explore
            break;
        }
        betas.push(beta);
        v_prev = Some(v);
        beta_prev = beta;
        v = w / beta;
    }

    let dim = alphas.len();
    if dim == 0 {
        return Err(Error::not_converged(
            "Lanczos iteration produced an empty subspace",
        ));
    }

    // Reduced symmetric tridiagonal problem: diag = alphas, off-diag = betas
    let mut tri = nalgebra::DMatrix::<f64>::zeros(dim, dim);
    for i in 0..dim {
        tri[(i, i)] = alphas[i];
        if i + 1 < dim {
            tri[(i, i + 1)] = betas[i];
            tri[(i + 1, i)] = betas[i];
        }
    }
    let eigen = nalgebra::linalg::SymmetricEigen::new(tri);

    // Order Ritz values descending and keep the leading min(k, dim)
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let take = k.min(dim);
    let mut eigenvalues = Vec::with_capacity(take);
    let mut eigenvectors = Vec::with_capacity(take);
    for &idx in order.iter().take(take) {
        eigenvalues.push(eigen.eigenvalues[idx]);
        // Ritz vector = sum_j T_eigenvector[j, idx] * lanczos_vectors[j]
        let mut ritz = Array1::<f64>::zeros(n);
        for (j, lv) in lanczos_vectors.iter().enumerate() {
            ritz.scaled_add(eigen.eigenvectors[(j, idx)], lv);
        }
        let norm = ritz.dot(&ritz).sqrt();
        if norm > f64::EPSILON {
            ritz /= norm;
        }
        eigenvectors.push(ritz);
    }

    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A fixed symmetric matrix with well-separated eigenvalues
    fn symmetric_test_matrix() -> Array2<f64> {
        array![
            [4.0, 1.0, 0.0, 0.5],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.5, 0.0, 1.0, 1.0],
        ]
    }

    /// Reference eigenvalues (descending) from nalgebra's dense symmetric solver
    fn reference_eigenvalues_desc(a: &Array2<f64>) -> Vec<f64> {
        let n = a.nrows();
        let m = nalgebra::DMatrix::from_row_slice(n, n, a.as_slice().unwrap());
        let eig = nalgebra::linalg::SymmetricEigen::new(m);
        let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        vals.sort_by(|x, y| y.partial_cmp(x).unwrap());
        vals
    }

    /// Asserts that (value, vector) is an eigenpair of `a`: A v ~= lambda v, ||v|| ~= 1
    ///
    /// # Parameters
    ///
    /// - `a` - Symmetric matrix
    /// - `value` - Candidate eigenvalue
    /// - `vector` - Candidate eigenvector
    /// - `tol` - Absolute tolerance for the checks
    fn assert_eigenpair(a: &Array2<f64>, value: f64, vector: &Array1<f64>, tol: f64) {
        let av = a.dot(vector);
        let lv = vector * value;
        for (x, y) in av.iter().zip(lv.iter()) {
            assert!((x - y).abs() < tol, "A·v != λ·v: {} vs {}", x, y);
        }
        assert!(
            (vector.dot(vector).sqrt() - 1.0).abs() < tol,
            "eigenvector is not unit norm"
        );
    }

    #[test]
    fn power_iteration_matches_dense_reference() {
        let a = symmetric_test_matrix();
        let reference = reference_eigenvalues_desc(&a);
        let (vals, vecs) = top_eigenpairs_power_iteration(a.clone(), 2, 0, 2000, 1e-10).unwrap();

        assert_eq!(vals.len(), 2);
        for i in 0..2 {
            assert!(
                (vals[i] - reference[i]).abs() < 1e-4,
                "eigenvalue {} mismatch: {} vs {}",
                i,
                vals[i],
                reference[i]
            );
            assert_eigenpair(&a, vals[i], &vecs[i], 1e-4);
        }
    }

    #[test]
    fn lanczos_matches_dense_reference() {
        let a = symmetric_test_matrix();
        let reference = reference_eigenvalues_desc(&a);
        let (vals, vecs) = top_eigenpairs_lanczos(&a, 3, 0).unwrap();

        assert_eq!(vals.len(), 3);
        for i in 0..3 {
            assert!(
                (vals[i] - reference[i]).abs() < 1e-8,
                "eigenvalue {} mismatch: {} vs {}",
                i,
                vals[i],
                reference[i]
            );
            assert_eigenpair(&a, vals[i], &vecs[i], 1e-8);
        }
    }

    // edge-case tests for top_eigenpairs_power_iteration

    /// k=0 returns two empty vecs without error
    #[test]
    fn power_iteration_k_zero_returns_empty() {
        let a = symmetric_test_matrix();
        let (vals, vecs) = top_eigenpairs_power_iteration(a, 0, 0, 2000, 1e-10).unwrap();
        assert_eq!(vals.len(), 0, "eigenvalues should be empty for k=0");
        assert_eq!(vecs.len(), 0, "eigenvectors should be empty for k=0");
    }

    /// Eigenvalues returned for k>1 are in strictly descending order
    #[test]
    fn power_iteration_eigenvalues_descending_order() {
        let a = symmetric_test_matrix();
        let (vals, _) = top_eigenpairs_power_iteration(a, 3, 0, 2000, 1e-10).unwrap();
        assert_eq!(vals.len(), 3);
        assert!(
            vals[0] > vals[1],
            "λ_0 ({}) must be > λ_1 ({})",
            vals[0],
            vals[1]
        );
        assert!(
            vals[1] > vals[2],
            "λ_1 ({}) must be > λ_2 ({})",
            vals[1],
            vals[2]
        );
    }

    /// Eigenvectors for k>=2 are mutually orthogonal
    #[test]
    fn power_iteration_eigenvectors_mutually_orthogonal() {
        let a = symmetric_test_matrix();
        let (_, vecs) = top_eigenpairs_power_iteration(a, 3, 0, 2000, 1e-10).unwrap();
        assert_eq!(vecs.len(), 3);
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dot = vecs[i].dot(&vecs[j]).abs();
                assert!(dot < 1e-5, "v_{} · v_{} = {} (expected < 1e-5)", i, j, dot);
            }
        }
    }

    // edge-case tests for top_eigenpairs_lanczos

    /// k=0 returns two empty vecs without error
    #[test]
    fn lanczos_k_zero_returns_empty() {
        let a = symmetric_test_matrix();
        let (vals, vecs) = top_eigenpairs_lanczos(&a, 0, 0).unwrap();
        assert_eq!(vals.len(), 0, "eigenvalues should be empty for k=0");
        assert_eq!(vecs.len(), 0, "eigenvectors should be empty for k=0");
    }

    /// On a rank-1 matrix Lanczos reaches an invariant subspace early, so at most 1 non-trivial eigenpair is returned
    #[test]
    fn lanczos_rank_one_invariant_subspace_early_exit() {
        // 3x3 rank-1 matrix: outer product of [1, 0, 0] with itself; only eigenvalue is 1
        let a: Array2<f64> = ndarray::array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],];

        // Request k=5 to confirm fewer pairs come back only because the matrix is rank-1
        let (vals, vecs) = top_eigenpairs_lanczos(&a, 5, 7).unwrap();

        // Invariant subspace reached early (dim <= 2), so the return length is at most 2
        assert!(
            vals.len() <= 2,
            "expected at most 2 eigenpairs from a rank-1 matrix, got {}",
            vals.len()
        );
        assert_eq!(
            vals.len(),
            vecs.len(),
            "eigenvalues and eigenvectors must have equal length"
        );

        // A rank-1 matrix has exactly 1 non-zero eigenvalue, so at most 1 value is >= 1e-3
        let n_large = vals.iter().filter(|&&v| v.abs() >= 1e-3).count();
        assert!(
            n_large <= 1,
            "at most 1 non-trivial eigenvalue expected for rank-1 matrix, found {}",
            n_large
        );
    }
}

//! Small shared linear-algebra routines for the `utility` transformers.
//!
//! Both PCA's and Kernel PCA's iterative solvers extract the leading
//! eigenpairs of a symmetric matrix by power iteration with Hotelling
//! deflation. That routine lives here so the two solvers share one
//! implementation instead of carrying near-identical copies.

use crate::error::ModelError;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};

/// Builds a random unit vector of length `n`, falling back to a uniform unit
/// vector if the random draw happens to be (numerically) zero.
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

/// Computes the dominant eigenpair of a symmetric matrix via power iteration.
///
/// Returns the unit eigenvector together with its eigenvalue, estimated as the
/// Rayleigh quotient `vᵀ M v`.
///
/// # Errors
///
/// - [`ModelError::ProcessingError`] - If the iteration fails to converge to a
///   finite, non-degenerate eigenpair
fn dominant_eigenpair(
    matrix: &Array2<f64>,
    rng: &mut StdRng,
    max_iter: usize,
    tol: f64,
) -> Result<(Array1<f64>, f64), ModelError> {
    let n = matrix.ncols();
    let mut v = random_unit_vector(n, rng);

    let mut prev_lambda = 0.0;
    for _ in 0..max_iter {
        // Iterate toward the dominant eigenvector.
        let w = matrix.dot(&v);
        let w_norm = w.dot(&w).sqrt();
        if w_norm <= f64::EPSILON || !w_norm.is_finite() {
            return Err(ModelError::ProcessingError(
                "Power iteration failed to converge".to_string(),
            ));
        }
        let v_next = &w / w_norm;
        let lambda = v_next.dot(&matrix.dot(&v_next));
        if !lambda.is_finite() {
            return Err(ModelError::ProcessingError(
                "Power iteration produced non-finite eigenvalue".to_string(),
            ));
        }
        if (lambda - prev_lambda).abs() < tol {
            return Ok((v_next, lambda));
        }
        prev_lambda = lambda;
        v = v_next;
    }

    let lambda = v.dot(&matrix.dot(&v));
    if !lambda.is_finite() {
        return Err(ModelError::ProcessingError(
            "Power iteration produced non-finite eigenvalue".to_string(),
        ));
    }
    Ok((v, lambda))
}

/// Extracts the top-`k` eigenpairs of a symmetric matrix using power iteration
/// with Hotelling deflation.
///
/// The matrix is taken by value because each extracted component is deflated
/// out of it in place. Eigenpairs come back in descending eigenvalue order as
/// parallel vectors: `eigenvectors[i]` is the unit eigenvector for
/// `eigenvalues[i]`. Callers arrange those eigenvectors into rows or columns as
/// their own layout requires.
///
/// # Parameters
///
/// - `matrix` - Symmetric input matrix (e.g. a covariance or centered kernel matrix)
/// - `k` - Number of leading eigenpairs to extract
/// - `seed` - Seed for the random initialization, making the result deterministic
/// - `max_iter` - Maximum power-iteration steps per component
/// - `tol` - Convergence tolerance on the eigenvalue estimate
///
/// # Errors
///
/// - [`ModelError::ProcessingError`] - If any component fails to converge
pub(super) fn top_eigenpairs_power_iteration(
    mut matrix: Array2<f64>,
    k: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<Array1<f64>>), ModelError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);

    for _ in 0..k {
        let (vector, value) = dominant_eigenpair(&matrix, &mut rng, max_iter, tol)?;
        // Deflate the extracted component so the next iteration surfaces the next one:
        // M ← M - λ vvᵀ.
        let outer = vector
            .view()
            .insert_axis(Axis(1))
            .dot(&vector.view().insert_axis(Axis(0)));
        matrix.scaled_add(-value, &outer);
        eigenvalues.push(value);
        eigenvectors.push(vector);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Extracts the top-`k` eigenpairs of a symmetric matrix using the Lanczos
/// algorithm with full reorthogonalization.
///
/// Lanczos builds a Krylov subspace, reduces the problem to a small symmetric
/// tridiagonal eigenproblem, and maps the leading Ritz pairs back. For the
/// dominant eigenpairs of a symmetric matrix (such as a centered kernel matrix)
/// it converges far faster and more stably than deflated power iteration — it is
/// the pure-Rust counterpart of the symmetric solver ARPACK is built on (a
/// single Lanczos pass, without implicit restarts).
///
/// Eigenpairs come back in descending eigenvalue order as parallel vectors:
/// `eigenvectors[i]` is the unit eigenvector for `eigenvalues[i]`.
///
/// # Parameters
///
/// - `matrix` - Symmetric input matrix
/// - `k` - Number of leading eigenpairs to extract
/// - `seed` - Seed for the random starting vector, making the result deterministic
///
/// # Errors
///
/// - [`ModelError::ProcessingError`] - If the Krylov subspace collapses immediately
pub(super) fn top_eigenpairs_lanczos(
    matrix: &Array2<f64>,
    k: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<Array1<f64>>), ModelError> {
    let n = matrix.ncols();
    // Krylov dimension: comfortably larger than k for accurate leading Ritz pairs,
    // capped at the matrix size.
    let m = (2 * k + 20).min(n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut lanczos_vectors: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alphas: Vec<f64> = Vec::with_capacity(m);
    let mut betas: Vec<f64> = Vec::with_capacity(m);

    let mut v = random_unit_vector(n, &mut rng);
    let mut v_prev: Option<Array1<f64>> = None;
    let mut beta_prev = 0.0;

    for _ in 0..m {
        // Three-term recurrence: w = A·v - alpha·v - beta_prev·v_prev.
        let mut w = matrix.dot(&v);
        let alpha = v.dot(&w);
        w.scaled_add(-alpha, &v);
        if let Some(ref vp) = v_prev {
            w.scaled_add(-beta_prev, vp);
        }

        // Full reorthogonalization (repeated once for stability) against every
        // Lanczos vector generated so far, keeping the basis numerically orthogonal.
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
            // Reached an invariant subspace; no further directions to explore.
            break;
        }
        betas.push(beta);
        v_prev = Some(v);
        beta_prev = beta;
        v = w / beta;
    }

    let dim = alphas.len();
    if dim == 0 {
        return Err(ModelError::ProcessingError(
            "Lanczos iteration produced an empty subspace".to_string(),
        ));
    }

    // Reduced symmetric tridiagonal problem: diag = alphas, off-diag = betas.
    let mut tri = nalgebra::DMatrix::<f64>::zeros(dim, dim);
    for i in 0..dim {
        tri[(i, i)] = alphas[i];
        if i + 1 < dim {
            tri[(i, i + 1)] = betas[i];
            tri[(i + 1, i)] = betas[i];
        }
    }
    let eigen = nalgebra::linalg::SymmetricEigen::new(tri);

    // Order Ritz values descending and keep the leading min(k, dim).
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
        // Ritz vector = Σ_j T_eigenvector[j, idx] · lanczos_vectors[j].
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

    /// A fixed symmetric matrix with well-separated eigenvalues.
    fn symmetric_test_matrix() -> Array2<f64> {
        array![
            [4.0, 1.0, 0.0, 0.5],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.5, 0.0, 1.0, 1.0],
        ]
    }

    /// Reference eigenvalues (descending) from nalgebra's dense symmetric solver.
    fn reference_eigenvalues_desc(a: &Array2<f64>) -> Vec<f64> {
        let n = a.nrows();
        let m = nalgebra::DMatrix::from_row_slice(n, n, a.as_slice().unwrap());
        let eig = nalgebra::linalg::SymmetricEigen::new(m);
        let mut vals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        vals.sort_by(|x, y| y.partial_cmp(x).unwrap());
        vals
    }

    /// Asserts that (value, vector) is an eigenpair of `a`: A·v ≈ λ·v, ‖v‖ ≈ 1.
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
}

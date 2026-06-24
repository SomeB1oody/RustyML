//! Dense matrix decompositions for `f64` matrices, written in pure Rust on top of `ndarray`
//!
//! These are the three workhorse factorizations the classical-ML and `utils` estimators need -
//! symmetric eigendecomposition, singular value decomposition, and a thin QR orthonormalization -
//! reimplemented directly on [`ndarray`] arrays so the crate no longer carries a second
//! linear-algebra ecosystem (`nalgebra`) just for these leaf operations. Everything here operates
//! on row-major [`Array2<f64>`] / [`Array1<f64>`] and matches the conventions the callers expect:
//!
//! - [`symmetric_eigen`] returns eigenvalues with their eigenvectors stored as the **columns** of
//!   the returned matrix, in ascending eigenvalue order (callers re-sort as they need)
//! - [`svd`] returns singular values in **descending** order with optional `U` and `Vᵀ`, plus
//!   [`Svd::solve`] (minimum-norm least squares) and [`Svd::pseudo_inverse`]
//! - [`qr_q`] returns an orthonormal basis `Q` for the column space of its input
//!
//! ## Algorithms
//!
//! - **Symmetric eigendecomposition** uses Householder tridiagonalization (`tred2`) followed by the
//!   implicit-shift QL iteration with accumulated eigenvectors (`tql2`), the classic
//!   EISPACK/JAMA pair. It is deterministic and accurate to near machine precision for the small to
//!   mid-sized symmetric matrices that arise here (covariance matrices, centered kernels, the
//!   reduced tridiagonal Lanczos problem)
//! - **SVD** uses the one-sided Jacobi method, which orthogonalizes the columns of the (taller
//!   orientation of the) matrix through a sweep of plane rotations. It converges reliably and
//!   computes the small singular values to high relative accuracy
//! - **QR** uses modified Gram-Schmidt with one reorthogonalization pass, matching the stabilized
//!   orthogonalization the Lanczos solver already relies on
//!
//! The module is only compiled when an estimator that needs it is enabled (`machine_learning` or
//! `utils`); the lighter feature tiers never pull it in.

#[cfg(feature = "machine_learning")]
use crate::error::Error;
use ndarray::{Array1, Array2};

/// Eigenvalues and eigenvectors of a real symmetric matrix
///
/// `eigenvectors` stores each unit eigenvector as a **column**: column `j` is the eigenvector for
/// `eigenvalues[j]`. Eigenvalues come back in ascending order
pub(crate) struct SymmetricEigen {
    /// Eigenvalues in ascending order
    pub eigenvalues: Array1<f64>,
    /// Unit eigenvectors stored as columns, aligned with `eigenvalues`
    pub eigenvectors: Array2<f64>,
}

/// Computes the eigenvalues and eigenvectors of a real symmetric matrix
///
/// The input is assumed symmetric; only its lower/upper symmetry is relied on implicitly by the
/// reduction. Uses Householder tridiagonalization (`tred2`) plus the implicit-shift QL algorithm
/// (`tql2`), porting the well-known EISPACK/JAMA routines
///
/// # Parameters
///
/// - `a` - Square symmetric matrix (`n x n`)
///
/// # Returns
///
/// - [`SymmetricEigen`] - Eigenvalues (ascending) and their unit eigenvectors as columns
pub(crate) fn symmetric_eigen(a: &Array2<f64>) -> SymmetricEigen {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols(), "symmetric_eigen requires a square matrix");

    if n == 0 {
        return SymmetricEigen {
            eigenvalues: Array1::zeros(0),
            eigenvectors: Array2::zeros((0, 0)),
        };
    }

    // `v` starts as a copy of `a` (row-major flat buffer) and is overwritten with the eigenvectors
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = a[[i, j]];
        }
    }
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];

    tred2(n, &mut v, &mut d, &mut e);
    tql2(n, &mut v, &mut d, &mut e);

    let eigenvalues = Array1::from_vec(d);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| v[i * n + j]);
    SymmetricEigen {
        eigenvalues,
        eigenvectors,
    }
}

/// Householder reduction of a symmetric matrix to tridiagonal form (EISPACK `tred2`)
///
/// On entry `v` holds the symmetric matrix in row-major order. On exit `v` holds the orthogonal
/// transformation, `d` the diagonal of the tridiagonal matrix, and `e` its subdiagonal
fn tred2(n: usize, v: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
    }

    // Householder reduction to tridiagonal form
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for &dk in d.iter().take(i) {
            scale += dk.abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            // Generate the Householder vector
            for dk in d.iter_mut().take(i) {
                *dk /= scale;
                h += *dk * *dk;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for ej in e.iter_mut().take(i) {
                *ej = 0.0;
            }

            // Apply the similarity transformation to the remaining columns
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + v[j * n + j] * f;
                for k in (j + 1)..=(i - 1) {
                    g += v[k * n + j] * d[k];
                    e[k] += v[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..=(i - 1) {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate the transformations
    for i in 0..(n - 1) {
        v[(n - 1) * n + i] = v[i * n + i];
        v[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k * n + (i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[k * n + (i + 1)] * v[k * n + j];
                }
                for k in 0..=i {
                    v[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k * n + (i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
        v[(n - 1) * n + j] = 0.0;
    }
    v[(n - 1) * n + (n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit-shift QL iteration with accumulated eigenvectors (EISPACK `tql2`)
///
/// On entry `d`/`e` hold the diagonal/subdiagonal of the tridiagonal matrix and `v` the orthogonal
/// transform from [`tred2`]. On exit `d` holds the eigenvalues (sorted ascending) and the columns
/// of `v` the corresponding eigenvectors
#[allow(unused_assignments)] // `c2`/`c3`/`s2` are loop-carried; the final write is intentionally dead
fn tql2(n: usize, v: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    let mut f = 0.0_f64;
    let mut tst1 = 0.0_f64;
    let eps = 2.0_f64.powi(-52);
    for l in 0..n {
        // Find the smallest subdiagonal element to split on
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }

        // If `m == l`, `d[l]` is already an eigenvalue; otherwise iterate
        if m > l {
            loop {
                // Compute the implicit shift
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;

                // Implicit QL transformation
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                let mut i = m;
                while i > l {
                    i -= 1;
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    // Accumulate the eigenvector rotation
                    for k in 0..n {
                        h = v[k * n + (i + 1)];
                        v[k * n + (i + 1)] = s * v[k * n + i] + c * h;
                        v[k * n + i] = c * v[k * n + i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Convergence check
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }

    // Sort eigenvalues (ascending) and reorder the eigenvector columns to match
    for i in 0..(n - 1) {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().take(n).skip(i + 1) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for row in 0..n {
                v.swap(row * n + i, row * n + k);
            }
        }
    }
}

/// Singular value decomposition of a real matrix
///
/// Singular values are non-negative and stored in descending order. When requested, `u` is the
/// `m x r` matrix of left singular vectors and `v_t` the `r x n` matrix of right singular vectors
/// transposed, where `r = min(m, n)`; together they satisfy `A ≈ U Σ Vᵀ`
pub(crate) struct Svd {
    /// Singular values in descending order, length `r = min(m, n)`
    pub singular_values: Array1<f64>,
    /// Left singular vectors as columns (`m x r`), present when `compute_u` was set.
    /// Only the `machine_learning` solve/pseudo-inverse paths read `U`; the `utils` SVD callers
    /// need only the singular values and `Vᵀ`
    #[cfg_attr(not(feature = "machine_learning"), allow(dead_code))]
    pub u: Option<Array2<f64>>,
    /// Right singular vectors transposed (`r x n`), present when `compute_v` was set
    pub v_t: Option<Array2<f64>>,
}

#[cfg(feature = "machine_learning")]
impl Svd {
    /// Solves the least-squares problem `min ||A x - b||` using the decomposition, returning the
    /// minimum-norm solution. Singular values at or below `tol` are treated as zero
    ///
    /// Requires that the decomposition was computed with both `U` and `Vᵀ`
    ///
    /// # Errors
    ///
    /// - [`Error::Computation`] - If `U`/`Vᵀ` were not computed or the right-hand side length does
    ///   not match the row count of `A`
    pub(crate) fn solve(&self, b: &Array1<f64>, tol: f64) -> Result<Array1<f64>, Error> {
        let u = self
            .u
            .as_ref()
            .ok_or_else(|| Error::computation("SVD solve requires the U factor"))?;
        let v_t = self
            .v_t
            .as_ref()
            .ok_or_else(|| Error::computation("SVD solve requires the V^T factor"))?;
        let m = u.nrows();
        let r = self.singular_values.len();
        let n = v_t.ncols();
        if b.len() != m {
            return Err(Error::computation(
                "SVD solve: right-hand side length does not match the number of rows",
            ));
        }

        // c = Σ⁺ Uᵀ b
        let mut c = Array1::<f64>::zeros(r);
        for k in 0..r {
            let sv = self.singular_values[k];
            if sv > tol {
                let mut acc = 0.0;
                for i in 0..m {
                    acc += u[[i, k]] * b[i];
                }
                c[k] = acc / sv;
            }
        }
        // x = V c = (Vᵀ)ᵀ c
        let mut x = Array1::<f64>::zeros(n);
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..r {
                acc += v_t[[k, j]] * c[k];
            }
            x[j] = acc;
        }
        Ok(x)
    }

    /// Builds the Moore-Penrose pseudo-inverse `A⁺ = V Σ⁺ Uᵀ` (`n x m`). Singular values at or
    /// below `tol` are treated as zero
    ///
    /// Requires that the decomposition was computed with both `U` and `Vᵀ`
    ///
    /// # Errors
    ///
    /// - [`Error::Computation`] - If `U`/`Vᵀ` were not computed
    pub(crate) fn pseudo_inverse(&self, tol: f64) -> Result<Array2<f64>, Error> {
        let u = self
            .u
            .as_ref()
            .ok_or_else(|| Error::computation("SVD pseudo_inverse requires the U factor"))?;
        let v_t = self
            .v_t
            .as_ref()
            .ok_or_else(|| Error::computation("SVD pseudo_inverse requires the V^T factor"))?;
        let m = u.nrows();
        let r = self.singular_values.len();
        let n = v_t.ncols();

        let mut sinv = Array1::<f64>::zeros(r);
        for k in 0..r {
            let sv = self.singular_values[k];
            sinv[k] = if sv > tol { 1.0 / sv } else { 0.0 };
        }

        // A⁺[j, i] = Σ_k v_t[k, j] · sinv[k] · u[i, k]
        let mut pinv = Array2::<f64>::zeros((n, m));
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for k in 0..r {
                    acc += v_t[[k, j]] * sinv[k] * u[[i, k]];
                }
                pinv[[j, i]] = acc;
            }
        }
        Ok(pinv)
    }
}

/// Computes the singular value decomposition of a real matrix via one-sided Jacobi
///
/// The Jacobi sweeps run on whichever of `A` / `Aᵀ` is taller, so the working matrix always has at
/// least as many rows as columns; the result is mapped back to `A`'s orientation. Singular values
/// are returned in descending order with optional `U` (`m x r`) and `Vᵀ` (`r x n`)
///
/// # Parameters
///
/// - `a` - Input matrix (`m x n`)
/// - `compute_u` - Whether to return the left singular vectors `U`
/// - `compute_v` - Whether to return the right singular vectors (as `Vᵀ`)
pub(crate) fn svd(a: &Array2<f64>, compute_u: bool, compute_v: bool) -> Svd {
    let m = a.nrows();
    let n = a.ncols();

    // `u_like` is the (rows-of-A x r) left factor, `v_like` the (cols-of-A x r) right factor, both
    // with columns aligned to the (still unsorted) singular values in `s`.
    let (s, u_like, v_like) = if m >= n {
        jacobi_svd_tall(a)
    } else {
        // A = (Aᵀ)ᵀ: if Aᵀ = U₂ Σ V₂ᵀ then A = V₂ Σ U₂ᵀ, so U_A = V₂ and V_A = U₂.
        let at = a.t().to_owned();
        let (s, u2, v2) = jacobi_svd_tall(&at);
        (s, v2, u2)
    };

    let (s_sorted, u_sorted, v_sorted) = sort_svd_descending(s, u_like, v_like);
    Svd {
        singular_values: s_sorted,
        u: if compute_u { Some(u_sorted) } else { None },
        v_t: if compute_v {
            Some(v_sorted.t().to_owned())
        } else {
            None
        },
    }
}

/// One-sided Jacobi SVD for a matrix with at least as many rows as columns (`m >= n`)
///
/// Returns `(singular_values, u, v)` (unsorted) with `A = U diag(s) Vᵀ`, where `u` is `m x n` with
/// orthonormal columns and `v` is `n x n` orthogonal
fn jacobi_svd_tall(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();

    let mut u = a.to_owned(); // m x n, rotated toward orthogonal columns in place
    let mut v = Array2::<f64>::eye(n); // accumulates the right rotations

    let eps = f64::EPSILON;
    let max_sweeps = 60;

    for _sweep in 0..max_sweeps {
        let mut changed = false;
        for p in 0..n {
            for q in (p + 1)..n {
                // 2x2 sub-Gram of columns p, q
                let mut alpha = 0.0; // ||u_p||²
                let mut beta = 0.0; // ||u_q||²
                let mut gamma = 0.0; // u_p · u_q
                for i in 0..m {
                    let up = u[[i, p]];
                    let uq = u[[i, q]];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }

                // Skip if the columns are already orthogonal to working precision
                if gamma == 0.0 || gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                changed = true;

                // Jacobi rotation that diagonalizes [[alpha, gamma], [gamma, beta]]
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // Apply the rotation to the columns of U and V
                for i in 0..m {
                    let up = u[[i, p]];
                    let uq = u[[i, q]];
                    u[[i, p]] = c * up - s * uq;
                    u[[i, q]] = s * up + c * uq;
                }
                for i in 0..n {
                    let vp = v[[i, p]];
                    let vq = v[[i, q]];
                    v[[i, p]] = c * vp - s * vq;
                    v[[i, q]] = s * vp + c * vq;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Singular values are the column norms of the rotated U; normalize to get orthonormal U
    let mut s = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut norm_sq = 0.0;
        for i in 0..m {
            norm_sq += u[[i, j]] * u[[i, j]];
        }
        let norm = norm_sq.sqrt();
        s[j] = norm;
        if norm > 0.0 {
            for i in 0..m {
                u[[i, j]] /= norm;
            }
        }
    }
    (s, u, v)
}

/// Reorders an SVD by descending singular value, permuting the aligned `U`/`V` columns to match
fn sort_svd_descending(
    s: Array1<f64>,
    u: Array2<f64>,
    v: Array2<f64>,
) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let r = s.len();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap_or(std::cmp::Ordering::Equal));

    let s_sorted = Array1::from_shape_fn(r, |k| s[order[k]]);
    let u_sorted = Array2::from_shape_fn((u.nrows(), r), |(i, k)| u[[i, order[k]]]);
    let v_sorted = Array2::from_shape_fn((v.nrows(), r), |(i, k)| v[[i, order[k]]]);
    (s_sorted, u_sorted, v_sorted)
}

/// Returns an orthonormal basis `Q` for the column space of `a` (same shape as `a`)
///
/// Uses modified Gram-Schmidt with one reorthogonalization pass for numerical stability, the same
/// twice-is-enough orthogonalization the Lanczos solver uses. A column that collapses to zero after
/// orthogonalization (a rank-deficient direction) is left as a zero column; downstream consumers
/// treat its contribution as a zero singular value
///
/// # Parameters
///
/// - `a` - Input matrix whose columns span the target subspace (`m x k`, `m >= k`)
#[cfg(feature = "utils")]
pub(crate) fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let mut q = a.to_owned();

    for j in 0..k {
        // Two modified Gram-Schmidt passes against the already-orthonormal columns 0..j
        for _pass in 0..2 {
            for i in 0..j {
                let mut proj = 0.0;
                for row in 0..m {
                    proj += q[[row, i]] * q[[row, j]];
                }
                for row in 0..m {
                    q[[row, j]] -= proj * q[[row, i]];
                }
            }
        }
        // Normalize column j
        let mut norm_sq = 0.0;
        for row in 0..m {
            norm_sq += q[[row, j]] * q[[row, j]];
        }
        let norm = norm_sq.sqrt();
        if norm > f64::EPSILON {
            for row in 0..m {
                q[[row, j]] /= norm;
            }
        } else {
            for row in 0..m {
                q[[row, j]] = 0.0;
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Builds a dense matrix from a nalgebra `DMatrix`-shaped reference for cross-checking
    fn na_dmatrix(a: &Array2<f64>) -> nalgebra::DMatrix<f64> {
        let (r, c) = (a.nrows(), a.ncols());
        nalgebra::DMatrix::from_row_slice(r, c, a.as_slice().unwrap())
    }

    /// Reconstructs `A` from a (possibly thin) SVD and asserts it matches within tolerance
    fn assert_svd_reconstructs(a: &Array2<f64>, tol: f64) {
        let s = svd(a, true, true);
        let u = s.u.as_ref().unwrap();
        let v_t = s.v_t.as_ref().unwrap();
        let r = s.singular_values.len();
        let (m, n) = (a.nrows(), a.ncols());
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..r {
                    acc += u[[i, k]] * s.singular_values[k] * v_t[[k, j]];
                }
                assert_abs_diff_eq!(acc, a[[i, j]], epsilon = tol);
            }
        }
    }

    // symmetric_eigen

    /// Eigenvalues match nalgebra's symmetric solver (both sorted ascending)
    #[test]
    fn symmetric_eigen_eigenvalues_match_nalgebra() {
        let a = array![
            [4.0, 1.0, 0.0, 0.5],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.5, 0.0, 1.0, 1.0],
        ];
        let mine = symmetric_eigen(&a);
        let mut mine_vals: Vec<f64> = mine.eigenvalues.to_vec();
        mine_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let na = nalgebra::linalg::SymmetricEigen::new(na_dmatrix(&a));
        let mut na_vals: Vec<f64> = na.eigenvalues.iter().copied().collect();
        na_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        for (m, n) in mine_vals.iter().zip(na_vals.iter()) {
            assert_abs_diff_eq!(m, n, epsilon = 1e-10);
        }
    }

    /// Each returned pair satisfies `A v = λ v` and the eigenvectors are orthonormal
    #[test]
    fn symmetric_eigen_pairs_are_valid_and_orthonormal() {
        let a = array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        let eig = symmetric_eigen(&a);
        let n = a.nrows();

        for j in 0..n {
            let v = eig.eigenvectors.column(j);
            let av = a.dot(&v);
            for i in 0..n {
                assert_abs_diff_eq!(av[i], eig.eigenvalues[j] * v[i], epsilon = 1e-10);
            }
        }
        // Orthonormal columns: VᵀV == I
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = eig.eigenvectors.column(i).dot(&eig.eigenvectors.column(j));
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(dot, expected, epsilon = 1e-10);
            }
        }
    }

    /// A 1x1 matrix yields its single entry as the eigenvalue with a unit eigenvector
    #[test]
    fn symmetric_eigen_one_by_one() {
        let a = array![[3.5]];
        let eig = symmetric_eigen(&a);
        assert_abs_diff_eq!(eig.eigenvalues[0], 3.5, epsilon = 1e-12);
        assert_abs_diff_eq!(eig.eigenvectors[[0, 0]].abs(), 1.0, epsilon = 1e-12);
    }

    /// A diagonal matrix returns its diagonal as eigenvalues
    #[test]
    fn symmetric_eigen_diagonal() {
        let a = array![[5.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]];
        let eig = symmetric_eigen(&a);
        let mut vals = eig.eigenvalues.to_vec();
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_abs_diff_eq!(vals[0], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(vals[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(vals[2], 5.0, epsilon = 1e-12);
    }

    // svd

    /// Singular values match nalgebra for a tall matrix and reconstruction holds
    #[test]
    fn svd_tall_matches_nalgebra() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let mine = svd(&a, true, true);
        let na = nalgebra::linalg::SVD::new(na_dmatrix(&a), true, true);

        let mut na_sv: Vec<f64> = na.singular_values.iter().copied().collect();
        na_sv.sort_by(|x, y| y.partial_cmp(x).unwrap());
        for (m, n) in mine.singular_values.iter().zip(na_sv.iter()) {
            assert_abs_diff_eq!(m, n, epsilon = 1e-10);
        }
        assert_svd_reconstructs(&a, 1e-10);
    }

    /// SVD reconstructs a wide matrix (m < n triggers the transpose path)
    #[test]
    fn svd_wide_reconstructs() {
        let a = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        assert_svd_reconstructs(&a, 1e-10);
        // r = min(m, n) singular values
        let s = svd(&a, false, true);
        assert_eq!(s.singular_values.len(), 2);
    }

    /// SVD reconstructs a square matrix
    #[test]
    fn svd_square_reconstructs() {
        let a = array![[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]];
        assert_svd_reconstructs(&a, 1e-10);
    }

    /// Least-squares solve matches the normal-equation solution for an overdetermined system
    #[cfg(feature = "machine_learning")]
    #[test]
    fn svd_solve_least_squares() {
        // Overdetermined A x = b; compare against nalgebra's SVD solve
        let a = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let b = array![6.0, 5.0, 7.0, 10.0];
        let mine = svd(&a, true, true).solve(&b, 1e-12).unwrap();

        let na = nalgebra::linalg::SVD::new(na_dmatrix(&a), true, true);
        let nb = nalgebra::DVector::from_row_slice(b.as_slice().unwrap());
        let na_sol = na.solve(&nb, 1e-12).unwrap();
        for i in 0..mine.len() {
            assert_abs_diff_eq!(mine[i], na_sol[i], epsilon = 1e-9);
        }
    }

    /// Pseudo-inverse satisfies the Moore-Penrose identity `A A⁺ A == A`
    #[cfg(feature = "machine_learning")]
    #[test]
    fn svd_pseudo_inverse_identity() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]];
        let pinv = svd(&a, true, true).pseudo_inverse(1e-12).unwrap();
        // A (A⁺ A) == A
        let ap = a.dot(&pinv); // m x m
        let apa = ap.dot(&a); // m x n
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(apa[[i, j]], a[[i, j]], epsilon = 1e-9);
            }
        }
    }

    /// Rank-deficient pseudo-inverse drops the (near-)zero singular value via the tolerance
    #[cfg(feature = "machine_learning")]
    #[test]
    fn svd_pseudo_inverse_rank_deficient() {
        // Columns are identical -> rank 1
        let a = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let s = svd(&a, true, true);
        let tol = 1e-12 * s.singular_values[0].max(1e-12);
        let pinv = s.pseudo_inverse(tol.max(1e-12)).unwrap();
        // A A⁺ A == A still holds for the Moore-Penrose inverse on rank-deficient input
        let ap = a.dot(&pinv);
        let apa = ap.dot(&a);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(apa[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    // qr_q

    /// `qr_q` returns orthonormal columns spanning the same space as the input
    #[cfg(feature = "utils")]
    #[test]
    fn qr_q_orthonormal_and_spanning() {
        let a = array![
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 3.0],
        ];
        let q = qr_q(&a);
        let k = a.ncols();

        // QᵀQ == I
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = q.column(i).dot(&q.column(j));
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(dot, expected, epsilon = 1e-10);
            }
        }
        // The projector Q Qᵀ leaves A's columns unchanged: Q (Qᵀ A) == A
        let qt_a = q.t().dot(&a);
        let proj = q.dot(&qt_a);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(proj[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }
}

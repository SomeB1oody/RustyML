//! Rayon-block-parallel matrix products for `f32`/`f64` ndarray operands
//!
//! Without a BLAS backend, `ndarray`'s `dot` runs on the single-threaded `matrixmultiply`
//! kernels. [`par_matmul`](crate::math::matmul::par_matmul) keeps those kernels for the inner work (cache blocking + SIMD) and
//! adds the missing parallelism at exactly one level: large products are split into row or
//! column blocks across rayon, each block computed by a serial `dot`.
//! [`par_matvec`](crate::math::matmul::par_matvec) is the
//! matrix-vector counterpart. Small products fall through to the plain serial `dot`, gated by
//! the calibrated per-type FLOPs thresholds on [`MatmulElem`](crate::math::matmul::MatmulElem).
//!
//! Block splitting never reorders the per-element accumulation over `k`, so the result is
//! **bitwise identical** to the serial `a.dot(&b)` at any thread count - unlike a k-split
//! reduction, which would change float summation order and break reproducibility.
//!
//! Inside this crate, the neural-network layers use the `f32` instantiation and the
//! classical-ML/utils modules use `f64`. The serial/parallel crossover differs per element type
//! (half the SIMD lanes, twice the bytes per element), so the gate thresholds live on
//! `MatmulElem` as per-type associated constants.

use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, LinalgScalar};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

mod private {
    /// Seals [`MatmulElem`](super::MatmulElem): the splitting strategy and its calibration are
    /// built around `matrixmultiply`'s `f32`/`f64` kernels, so the trait is not implementable
    /// outside this crate
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Minimum rows (or columns) per parallel block.
///
/// Each block's `dot` re-packs the shared operand, so blocks must be tall (or wide) enough to
/// amortize that packing. Calibrated for `f32` on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads),
/// 2026-06-11; see benches/RESULTS.md: 16-64 sit on a plateau (within noise of each other) and
/// 128+ degrades, so 64 is kept; the f64 sweep confirms 64 as the optimum. The packing
/// economics barely depend on the element type, so the constant is shared rather than per-type
const PAR_GEMM_MIN_BLOCK: usize = 64;

/// Minimum rows per parallel block for the matvec split.
///
/// Unlike a GEMM block, a matvec block has no operand re-packing to amortize - the floor only
/// keeps the per-task work above rayon's scheduling overhead - so it can sit far lower than
/// [`PAR_GEMM_MIN_BLOCK`]. That matters for "short, wide" products like `X^T . e` over many
/// samples of few features, where the split axis (the feature count) is small. Calibrated on
/// AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see benches/RESULTS.md: the tall
/// shape plateaus over 8-64 rows and the short-wide shape over 1-16 (degrading from 32 up), so
/// 8 takes the intersection
const PAR_GEMV_MIN_BLOCK: usize = 8;

/// Element types [`par_matmul`] / [`par_matvec`] accept, carrying the per-type serial/parallel
/// crossover thresholds.
///
/// Implemented for `f32` and `f64` only, and sealed: the splitting strategy and its calibration
/// are specific to `matrixmultiply`'s kernels.
///
/// The associated constants are **machine-calibrated defaults**, not semantic contracts: they
/// were measured on one CPU (provenance on each value) and may be retuned in any release. The
/// stable guarantees are the ones [`par_matmul`]/[`par_matvec`] document - results
/// bitwise-equal to the serial `dot` at any thread count, and a standard-layout output
pub trait MatmulElem: LinalgScalar + Send + Sync + private::Sealed {
    /// Minimum estimated FLOPs (`2*m*k*n`) before a matmul is worth splitting across rayon
    const PAR_GEMM_MIN_FLOPS: usize;

    /// Minimum estimated FLOPs (`2*m*k`) before a matvec is worth splitting across rayon.
    ///
    /// Kept separate from the GEMM gate because a matvec is memory-bound (it streams the whole
    /// matrix once for O(1) FLOPs per element), so its crossover is a different cost class
    const PAR_GEMV_MIN_FLOPS: usize;
}

impl MatmulElem for f32 {
    /// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
    /// benches/RESULTS.md: the measured crossover bracket is 2.1M-4.2M FLOPs (square and skinny
    /// shapes agree), so the gate sits at the proven-win end of the bracket
    const PAR_GEMM_MIN_FLOPS: usize = 4_000_000;

    /// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
    /// benches/RESULTS.md: crossover bracket 131K-524K FLOPs (square/tall/short-wide shapes
    /// agree), gate at the proven-win end
    const PAR_GEMV_MIN_FLOPS: usize = 524_288;
}

impl MatmulElem for f64 {
    /// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
    /// benches/RESULTS.md: crossover bracket 524K-1.77M FLOPs (the win at 1.77M is a thin
    /// 1.14x), so the gate sits just above the bracket. Lower than `f32`'s 4M, as expected
    /// from the halved SIMD width
    const PAR_GEMM_MIN_FLOPS: usize = 2_000_000;

    /// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
    /// benches/RESULTS.md: crossover bracket 131K-524K FLOPs, same bracket as `f32` (the
    /// matvec is bandwidth-bound either way), gate at the proven-win end
    const PAR_GEMV_MIN_FLOPS: usize = 524_288;
}

/// `C = A @ B` with block-parallel execution for large products.
///
/// Splits the longer output axis (rows of `A`, or columns of `B`) into per-thread blocks and
/// computes each with the serial `matrixmultiply` kernel; products below the calibrated
/// per-type FLOPs gate ([`MatmulElem::PAR_GEMM_MIN_FLOPS`]) fall through to a plain `a.dot(&b)`.
/// Splitting the `m`/`n` axes leaves every output element's `k`-accumulation order untouched,
/// so the result is **bitwise identical** to the serial product regardless of the thread
/// count - parallelism here never costs reproducibility.
///
/// The returned array is guaranteed to be in standard (row-major) layout. (A bare `a.dot(b)`
/// does not guarantee this: it returns a column-major result when both operands have a row
/// stride of 1, which arrays with a length-1 axis can exhibit.)
///
/// # Parameters
///
/// - `a` - Left operand with shape (m, k); any storage (owned array, view, transpose)
/// - `b` - Right operand with shape (k, n); any storage (owned array, view, transpose)
///
/// # Returns
///
/// - `Array2<T>` - The product with shape (m, n), standard layout
///
/// # Panics
///
/// - If the inner dimensions disagree (same contract as `dot`)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::matmul::par_matmul;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
///
/// // Owned arrays by reference...
/// let c = par_matmul(&a, &b);
/// assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
///
/// // ...or views and transposes
/// let ct = par_matmul(&a.view(), &b.t());
/// assert_eq!(ct, array![[17.0, 23.0], [39.0, 53.0]]);
/// ```
pub fn par_matmul<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (a, b) = (a.view(), b.view());
    let (m, k) = a.dim();
    let n = b.ncols();

    let flops = 2usize.saturating_mul(m).saturating_mul(k).saturating_mul(n);
    if flops < T::PAR_GEMM_MIN_FLOPS {
        return into_standard_layout(a.dot(&b));
    }

    split_matmul(&a, &b, PAR_GEMM_MIN_BLOCK)
}

/// `y = A @ x` with block-parallel execution for large matrices.
///
/// The matrix-vector counterpart of [`par_matmul`]: above the calibrated per-type FLOPs gate
/// ([`MatmulElem::PAR_GEMV_MIN_FLOPS`]) the rows of `A` are split into per-thread blocks, each
/// computed by ndarray's serial matrix-vector `dot`. Splitting the row axis leaves every output
/// element's `k`-accumulation order untouched, so the result is **bitwise identical** to the
/// serial `a.dot(&x)` at any thread count.
///
/// (The blocks deliberately stay on the matrix-vector kernel rather than reusing the
/// matrix-matrix kernel on a `[k, 1]` operand: ndarray dispatches matrix-vector products to a
/// different kernel than matrix-matrix ones, with a different accumulation order, so only the
/// same-kernel split reproduces `a.dot(&x)` bitwise.)
///
/// # Parameters
///
/// - `a` - Matrix with shape (m, k); any storage (owned array, view, transpose)
/// - `x` - Vector with length k; any storage
///
/// # Returns
///
/// - `Array1<T>` - The product with length m
///
/// # Panics
///
/// - If `a`'s column count differs from `x`'s length (same contract as `dot`)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::matmul::par_matvec;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let x = array![10.0_f64, 20.0];
///
/// let y = par_matvec(&a, &x);
/// assert_eq!(y, array![50.0, 110.0]);
///
/// // Transposed views work the same way: A^T . x
/// let yt = par_matvec(&a.t(), &x);
/// assert_eq!(yt, array![70.0, 100.0]);
/// ```
pub fn par_matvec<T, S1, S2>(a: &ArrayBase<S1, Ix2>, x: &ArrayBase<S2, Ix1>) -> Array1<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (a, x) = (a.view(), x.view());
    let (m, k) = a.dim();

    let flops = 2usize.saturating_mul(m).saturating_mul(k);
    if flops < T::PAR_GEMV_MIN_FLOPS {
        return a.dot(&x);
    }

    split_matvec(&a, &x, PAR_GEMV_MIN_BLOCK)
}

/// Element budget for one row-chunk of a tiled product whose full result would be too large
/// to materialize at once (e.g. KNN's `[n_query, n_train]` projections or t-SNE's pairwise
/// blocks): `chunk_rows = GEMM_CHUNK_ELEMS / row_len`, clamped to `[16, 4096]` rows.
///
/// Tiling only pays when the shared operand overflows the cache (see [`cache_resident`]), and
/// there bigger chunks are strictly better - each chunk re-streams the shared operand once, so
/// fewer chunks means less DRAM traffic. Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon
/// threads), 2026-06-11; see benches/RESULTS.md (500k x 64 training set): 8M elements -> 1.23x,
/// 32M -> 1.72x, 64M -> 1.93x over the GEMV swarm. The budget caps the transient chunk buffer
/// at 256 MB of f64 rather than chasing the asymptote
const GEMM_CHUNK_ELEMS: usize = 33_554_432;

/// Rows per chunk when tiling a product with `row_len`-wide output rows under
/// [`GEMM_CHUNK_ELEMS`].
///
/// Crate-internal tiling policy (kept callable by the calibration bench); not part of the
/// public API and carries no stability guarantee
#[doc(hidden)]
pub fn gemm_chunk_rows(row_len: usize) -> usize {
    (GEMM_CHUNK_ELEMS / row_len.max(1)).clamp(16, 4096)
}

/// Matrix size below which repeated row-GEMV sweeps over a shared matrix stay cache-resident,
/// making a per-row GEMV swarm faster than a tiled GEMM.
///
/// When many tasks each compute `X . v` against the same `X`, the whole of `X` is re-read per
/// task - free while `X` fits in the shared L3, but a DRAM re-stream once it overflows, where
/// a tiled GEMM (which streams `X` once per chunk) wins instead. Measured on AMD Ryzen 9 9950X
/// (64 MB L3 in two 32 MB CCDs), 2026-06-11, see benches/RESULTS.md: at 25 MB the swarm wins
/// ~2x; at 256 MB the tiled GEMM wins ~2x. Only those two points bracket the boundary, so the
/// constant sits at the full L3 size; the band between is uncalibrated
const CACHE_RESIDENT_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Whether an `[rows, cols]` matrix of `T` is small enough to treat as cache-resident for
/// repeated row-GEMV sweeps (see [`CACHE_RESIDENT_MAX_BYTES`]).
///
/// Crate-internal strategy policy; not part of the public API and carries no stability
/// guarantee
#[doc(hidden)]
pub fn cache_resident<T>(rows: usize, cols: usize) -> bool {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<T>())
        < CACHE_RESIDENT_MAX_BYTES
}

/// The block-parallel product itself, with no FLOPs gate: always splits (when the block size
/// allows). Kept separate from [`par_matmul`] so the calibration bench can time the split path
/// against the serial `dot` on either side of the gate, and sweep `min_block`.
///
/// Calibration hook only - prefer [`par_matmul`], whose gate keeps small products serial; not
/// part of the public API and carries no stability guarantee
#[doc(hidden)]
pub fn split_matmul<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    min_block: usize,
) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (a, b) = (a.view(), b.view());
    let (m, _) = a.dim();
    let n = b.ncols();
    let threads = rayon::current_num_threads();

    // Split the longer output axis
    if m >= n {
        let chunk = m.div_ceil(threads).max(min_block);
        if chunk >= m {
            return into_standard_layout(a.dot(&b));
        }
        let mut c = Array2::<T>::zeros((m, n));
        c.axis_chunks_iter_mut(Axis(0), chunk)
            .into_par_iter()
            .zip(a.axis_chunks_iter(Axis(0), chunk).into_par_iter())
            .for_each(|(mut c_blk, a_blk)| {
                c_blk.assign(&a_blk.dot(&b));
            });
        c
    } else {
        let chunk = n.div_ceil(threads).max(min_block);
        if chunk >= n {
            return into_standard_layout(a.dot(&b));
        }
        let mut c = Array2::<T>::zeros((m, n));
        c.axis_chunks_iter_mut(Axis(1), chunk)
            .into_par_iter()
            .zip(b.axis_chunks_iter(Axis(1), chunk).into_par_iter())
            .for_each(|(mut c_blk, b_blk)| {
                c_blk.assign(&a.dot(&b_blk));
            });
        c
    }
}

/// The block-parallel matvec itself, with no FLOPs gate: always splits (when the block size
/// allows). Kept separate from [`par_matvec`] so the calibration bench can time the split path
/// against the serial `dot` on either side of the gate, and sweep `min_block`.
///
/// Calibration hook only - prefer [`par_matvec`], whose gate keeps small products serial; not
/// part of the public API and carries no stability guarantee
#[doc(hidden)]
pub fn split_matvec<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    x: &ArrayBase<S2, Ix1>,
    min_block: usize,
) -> Array1<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (a, x) = (a.view(), x.view());
    let (m, _) = a.dim();
    let threads = rayon::current_num_threads();
    let chunk = m.div_ceil(threads).max(min_block);
    if chunk >= m {
        return a.dot(&x);
    }

    let mut y = Array1::<T>::zeros(m);
    y.axis_chunks_iter_mut(Axis(0), chunk)
        .into_par_iter()
        .zip(a.axis_chunks_iter(Axis(0), chunk).into_par_iter())
        .for_each(|(mut y_blk, a_blk)| {
            y_blk.assign(&a_blk.dot(&x));
        });
    y
}

/// Normalizes a `dot` result to standard layout (no-op copy-free in the common row-major case)
fn into_standard_layout<T: LinalgScalar>(m: Array2<T>) -> Array2<T> {
    if m.is_standard_layout() {
        m
    } else {
        m.as_standard_layout().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic pseudo-random f32 matrix (hash-based, no rng dependency so the tests run
    /// under a `math`-only build)
    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let t = (seed as f64) * 0.731 + (i * cols + j) as f64 * 0.618_033_988_7;
            ((t.sin() * 43758.5453).fract() - 0.5) as f32
        })
    }

    /// Deterministic pseudo-random f64 matrix for the f64 instantiation
    fn random_matrix_f64(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let t = (seed as f64) * 0.731 + (i * cols + j) as f64 * 0.618_033_988_7;
            (t.sin() * 43758.5453).fract() - 0.5
        })
    }

    // Bitwise equivalence with the serial product

    /// Row-split path (m >= n, above the FLOP threshold) is bitwise identical to serial dot
    #[test]
    fn par_matmul_row_split_bitwise_equals_serial() {
        // 2*512*64*256 = 16.8M FLOPs >= threshold, m = 512 > n = 256 -> row split
        let a = random_matrix(512, 64, 1);
        let b = random_matrix(64, 256, 2);
        let par = par_matmul(&a, &b);
        let serial = a.dot(&b);
        assert_eq!(par.shape(), serial.shape());
        assert!(
            par.iter().zip(serial.iter()).all(|(x, y)| x == y),
            "row-split parallel product must be bitwise identical to serial"
        );
    }

    /// Column-split path (n > m, above the FLOP threshold) is bitwise identical to serial dot
    #[test]
    fn par_matmul_column_split_bitwise_equals_serial() {
        // 2*64*256*512 = 16.8M FLOPs >= threshold, n = 512 > m = 64 -> column split
        let a = random_matrix(64, 256, 3);
        let b = random_matrix(256, 512, 4);
        let par = par_matmul(&a, &b);
        let serial = a.dot(&b);
        assert!(
            par.iter().zip(serial.iter()).all(|(x, y)| x == y),
            "column-split parallel product must be bitwise identical to serial"
        );
    }

    /// Below the FLOP threshold the serial fallback gives exactly a.dot(b)
    #[test]
    fn par_matmul_small_matches_serial() {
        let a = random_matrix(8, 8, 5);
        let b = random_matrix(8, 8, 6);
        let par = par_matmul(&a, &b);
        let serial = a.dot(&b);
        assert!(par.iter().zip(serial.iter()).all(|(x, y)| x == y));
    }

    /// Transposed operands (the `x.t().dot(dz)` weight-reduction pattern) match serial
    #[test]
    fn par_matmul_transposed_operands_match_serial() {
        // [64, 2048] @ [2048, 64] from transposes: 2*64*2048*64 = 16.8M FLOPs, m == n -> row split
        let x = random_matrix(2048, 64, 7);
        let dz = random_matrix(2048, 64, 8);
        let par = par_matmul(&x.t(), &dz);
        let serial = x.t().dot(&dz);
        assert!(
            par.iter().zip(serial.iter()).all(|(a, b)| a == b),
            "transposed-operand product must match serial"
        );
    }

    /// The f64 instantiation is bitwise identical to serial dot on both split axes
    #[test]
    fn par_matmul_f64_bitwise_equals_serial() {
        // Row split: 2*512*64*256 = 16.8M FLOPs, m > n
        let a = random_matrix_f64(512, 64, 9);
        let b = random_matrix_f64(64, 256, 10);
        let par = par_matmul(&a, &b);
        let serial = a.dot(&b);
        assert!(
            par.iter().zip(serial.iter()).all(|(x, y)| x == y),
            "f64 row-split product must be bitwise identical to serial"
        );

        // Column split: n > m
        let a = random_matrix_f64(64, 256, 11);
        let b = random_matrix_f64(256, 512, 12);
        let par = par_matmul(&a, &b);
        let serial = a.dot(&b);
        assert!(
            par.iter().zip(serial.iter()).all(|(x, y)| x == y),
            "f64 column-split product must be bitwise identical to serial"
        );
    }

    // Layout guarantees

    /// Output is standard layout even when dot would return a column-major result
    /// (both operands with row stride 1, via a length-1 axis)
    #[test]
    fn par_matmul_normalizes_column_major_dot_output() {
        // [4, 1] (strides (1, 1)) @ [1, 3] with strides (1, 1): ndarray's dot returns
        // column-major here; par_matmul must hand back standard layout
        let a = Array2::<f32>::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_owned = Array2::<f32>::from_shape_vec((3, 1), vec![5.0, 6.0, 7.0]).unwrap();
        let b = b_owned.t(); // [1, 3] with strides (1, 1)
        assert_eq!(b.strides(), &[1, 1]);

        let c = par_matmul(&a, &b);
        assert!(c.is_standard_layout(), "output must be standard layout");
        let serial = a.dot(&b);
        assert!(c.iter().zip(serial.iter()).all(|(x, y)| x == y));
    }

    // Matrix-vector product

    /// Above the gate (split path) par_matvec is bitwise identical to serial a.dot(&x)
    #[test]
    fn par_matvec_split_bitwise_equals_serial() {
        // 2*4096*1024 = 8.4M FLOPs >= threshold -> split path
        let a = random_matrix(4096, 1024, 13);
        let x = random_matrix(1024, 1, 14).remove_axis(Axis(1));
        let par = par_matvec(&a, &x);
        let serial = a.dot(&x);
        assert_eq!(par.len(), serial.len());
        assert!(
            par.iter().zip(serial.iter()).all(|(p, s)| p == s),
            "split matvec must be bitwise identical to serial"
        );
    }

    /// Below the gate the serial fallback gives exactly a.dot(&x)
    #[test]
    fn par_matvec_small_matches_serial() {
        let a = random_matrix(16, 8, 15);
        let x = random_matrix(8, 1, 16).remove_axis(Axis(1));
        let par = par_matvec(&a, &x);
        let serial = a.dot(&x);
        assert!(par.iter().zip(serial.iter()).all(|(p, s)| p == s));
    }

    /// The f64 matvec instantiation matches serial on the split path
    #[test]
    fn par_matvec_f64_bitwise_equals_serial() {
        let a = random_matrix_f64(4096, 1024, 17);
        let x = random_matrix_f64(1024, 1, 18).remove_axis(Axis(1));
        let par = par_matvec(&a, &x);
        let serial = a.dot(&x);
        assert!(
            par.iter().zip(serial.iter()).all(|(p, s)| p == s),
            "f64 split matvec must be bitwise identical to serial"
        );
    }

    // Degenerate shapes

    /// Zero-sized axes produce the empty/zero product without panicking
    #[test]
    fn par_matmul_zero_sized_axes() {
        let a = Array2::<f32>::zeros((0, 4));
        let b = Array2::<f32>::zeros((4, 3));
        let c = par_matmul(&a, &b);
        assert_eq!(c.shape(), &[0, 3]);

        let a = Array2::<f32>::zeros((3, 0));
        let b = Array2::<f32>::zeros((0, 4));
        let c = par_matmul(&a, &b);
        assert_eq!(c.shape(), &[3, 4]);
        assert!(c.iter().all(|&x| x == 0.0));
    }

    /// 1x1 product
    #[test]
    fn par_matmul_one_by_one() {
        let a = Array2::<f32>::from_elem((1, 1), 3.0);
        let b = Array2::<f32>::from_elem((1, 1), 4.0);
        let c = par_matmul(&a, &b);
        assert_eq!(c[[0, 0]], 12.0);
    }

    /// Mismatched inner dimensions panic, matching `dot`'s contract
    #[test]
    #[should_panic]
    fn par_matmul_dimension_mismatch_panics() {
        let a = Array2::<f32>::zeros((2, 3));
        let b = Array2::<f32>::zeros((4, 2));
        par_matmul(&a, &b);
    }
}

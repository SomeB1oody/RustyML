//! Matrix products for ndarray operands, backed by the [`gemm`](https://docs.rs/gemm) crate
//!
//! The crate-internal `gemm_par_auto` / `gemv_par_auto` wrappers run every product on the
//! `gemm` crate's kernels - runtime-dispatched to the widest available SIMD (AVX-512 / AVX2+FMA /
//! NEON), packing each operand once and sharing it across threads, with special cases for skinny
//! shapes. They differ in how parallelism is obtained, because the two products have different
//! cost classes:
//!
//! - **GEMM** (`gemm_par_auto`) picks parallelism by shape. Below `MatmulElem::gemm_rayon_min_flops`
//!   it runs 1 serial call (the per-call rayon dispatch would dominate the tiny products in tight
//!   loops - RNN/LSTM timesteps). Above it the split follows the longer axis: when `m >= n` *and*
//!   the columns are too few to feed the pool (fewer than ~16 per thread) it splits the rows itself
//!   (`gemm_rowsplit`), because the `gemm` crate only parallelizes over the columns `n` and that
//!   starves on thin/medium-tall outputs (matvecs, training-loop GEMMs); otherwise (wide `n`, or
//!   many columns) it hands the whole product to the `gemm` crate to parallelize over `n` on the pool
//! - **GEMV** (`gemv_par_auto`) is the `n == 1` case: the `gemm` crate never parallelizes a matvec,
//!   so above `MatmulElem::gemv_rayon_min_flops` it always takes the row split (a matvec is
//!   bandwidth-bound, so the lower gate reflects that extra cores help almost immediately)
//!
//! Because everything runs on the global rayon pool, these calls also compose safely inside a
//! rayon region (the work nests instead of oversubscribing)
//!
//! ## Reproducibility
//!
//! Re-running the same product on the same machine reproduces the result (not necessarily
//! bit-for-bit)
//!
//! `gemm_chunk_rows` and `cache_resident` remain as tiling-strategy helpers for callers that
//! materialize a product in row-chunks (KNN, t-SNE, MeanShift)

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
use ndarray::parallel::prelude::IntoParallelIterator;
#[cfg(any(feature = "machine_learning", feature = "utils"))]
use ndarray::{Array1, Ix1};
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, LinalgScalar};
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

/// Minimum rows per block for the row-split path (GEMV and thin-`n` GEMM)
///
/// A row block has no operand re-packing to amortize, so the floor only keeps the per-task work
/// above rayon's scheduling overhead and can sit low
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
const PAR_ROWSPLIT_MIN_BLOCK: usize = 8;

tunable_gate! {
    /// Columns-per-thread below which the `gemm` crate's column parallelism starves
    ///
    /// The backend parallelizes a GEMM only over its output columns `n`. With fewer than this many
    /// columns per rayon thread the column tasks are too granular to fill the pool, so
    /// `gemm_par_auto` splits the rows itself instead (for `m >= n`). Calibrated by
    /// `benches/gemm_calibrate`: at 32 threads the crossover sits around `n = 512` (16
    /// columns/thread). Override via [`crate::tuning::matmul`]
    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    pub(crate) GEMM_COLPAR_MIN_COLS_PER_THREAD
        => gemm_colpar_min_cols_per_thread / set_gemm_colpar_min_cols_per_thread = 16
}

/// Element types the crate-internal matmul wrappers accept, plus each type's parallelism crossovers
///
/// Bounding `gemm_par_auto` / `gemv_par_auto` on this trait restricts them to the types the
/// `gemm` crate supports here (`f32`, `f64`), so the backend's "unsupported type" panic is
/// unreachable. The `Send + Sync` bound lets `gemv_par_auto` split its rows across rayon
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub(crate) trait MatmulElem: LinalgScalar + Send + Sync {
    /// Estimated-FLOPs (`2*m*k*n`) at or above which `gemm_par_auto` lets the `gemm` kernel run
    /// across the rayon pool; below it the kernel runs serially to skip the per-call dispatch that
    /// dominates the tiny GEMMs in tight loops (RNN/LSTM timesteps, small dense layers)
    ///
    /// Calibrated by `benches/gemm_calibrate`. `f32`'s wider SIMD makes its serial kernel
    /// relatively faster, so its in-loop crossover sits higher than `f64`'s. The value is a
    /// runtime gate (per dtype), overridable via [`crate::tuning::matmul`]
    fn gemm_rayon_min_flops() -> usize;

    /// Estimated-FLOPs (`2*m*k`) at or above which `gemv_par_auto` splits the matvec's rows
    /// across rayon (each block a serial `gemm` call); below it it runs one serial call
    ///
    /// A matvec is memory-bound and the `gemm` crate does not parallelize `n == 1` on its own, so
    /// unlike the GEMM gate this controls a row split the wrapper performs itself. The crossover is
    /// far lower than the GEMM one - extra cores add memory bandwidth, which the bandwidth-bound
    /// matvec can use almost immediately
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fn gemv_rayon_min_flops() -> usize;
}

tunable_gate! {
    /// `f32` GEMM rayon-crossover FLOPs (`<f32 as MatmulElem>::gemm_rayon_min_flops`). Set higher
    /// than the isolated ~4M crossover because the f32 products in this band are RNN/LSTM timestep
    /// GEMMs called in a tight loop
    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    pub(crate) GEMM_RAYON_MIN_FLOPS_F32
        => gemm_rayon_min_flops_f32 / set_gemm_rayon_min_flops_f32 = 8_000_000
}

tunable_gate! {
    /// `f64` GEMM rayon-crossover FLOPs (`<f64 as MatmulElem>::gemm_rayon_min_flops`)
    #[cfg(any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    ))]
    pub(crate) GEMM_RAYON_MIN_FLOPS_F64
        => gemm_rayon_min_flops_f64 / set_gemm_rayon_min_flops_f64 = 1_000_000
}

tunable_gate! {
    /// `f32` GEMV rayon-crossover FLOPs (`<f32 as MatmulElem>::gemv_rayon_min_flops`)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) GEMV_RAYON_MIN_FLOPS_F32
        => gemv_rayon_min_flops_f32 / set_gemv_rayon_min_flops_f32 = 524_288
}

tunable_gate! {
    /// `f64` GEMV rayon-crossover FLOPs (`<f64 as MatmulElem>::gemv_rayon_min_flops`)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    pub(crate) GEMV_RAYON_MIN_FLOPS_F64
        => gemv_rayon_min_flops_f64 / set_gemv_rayon_min_flops_f64 = 524_288
}

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
impl MatmulElem for f32 {
    fn gemm_rayon_min_flops() -> usize {
        gemm_rayon_min_flops_f32()
    }
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fn gemv_rayon_min_flops() -> usize {
        gemv_rayon_min_flops_f32()
    }
}

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
impl MatmulElem for f64 {
    fn gemm_rayon_min_flops() -> usize {
        gemm_rayon_min_flops_f64()
    }
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    fn gemv_rayon_min_flops() -> usize {
        gemv_rayon_min_flops_f64()
    }
}

/// One `gemm`-crate call producing a fresh standard-layout `[m, n]` array: `C = A @ B`
///
/// The operands' strides are passed straight through (the `gemm` crate handles arbitrary,
/// including negative, strides), so no operand is copied or transposed first. `par` selects the
/// kernel's parallelism
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
fn gemm_kernel<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    par: gemm::Parallelism,
) -> Array2<T>
where
    T: LinalgScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    let n = b.ncols();
    let mut out = Array2::<T>::zeros((m, n));
    // An empty product (any zero dimension) is all zeros; `gemm` also requires non-empty inputs
    if m == 0 || n == 0 || k == 0 {
        return out;
    }

    let a_strides = a.strides();
    let b_strides = b.strides();

    // SAFETY: `out` is a freshly allocated `[m, n]` standard-layout array (row stride `n`, column
    // stride `1`), so the destination pointer is valid for `m*n` writes. `a`/`b` are valid
    // `[m, k]`/`[k, n]` arrays whose `as_ptr` + element strides describe in-bounds elements, and
    // the zero-dimension cases are handled above. `gemm` accepts arbitrary (including negative)
    // strides and only reads/writes within the given dimensions. `read_dst = false` overwrites the
    // destination with `beta * a * b`
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            a_strides[1],
            a_strides[0],
            b.as_ptr(),
            b_strides[1],
            b_strides[0],
            T::zero(),
            T::one(),
            false,
            false,
            false,
            par,
        );
    }
    out
}

/// `C = A @ B` parallelized by splitting `A`'s rows across rayon, each block a serial `gemm` call
///
/// The `gemm` crate parallelizes over the output columns (`n`); when `n` is too small to feed the
/// threads (matvecs, and tall-skinny GEMMs like t-SNE's `W @ Y` with a handful of columns) that
/// leaves the cores idle. This splits the long axis - the rows - instead, so the parallelism comes
/// from the dimension the backend leaves serial. The result is standard layout. (Re-running on the
/// same machine reproduces the result, though not necessarily bit-for-bit)
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
fn gemm_rowsplit<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _k) = a.dim();
    let n = b.ncols();
    let threads = rayon::current_num_threads();
    let chunk = m.div_ceil(threads.max(1)).max(PAR_ROWSPLIT_MIN_BLOCK);
    // 1 block covers every row (or an empty axis): just 1 serial call
    if chunk >= m {
        return gemm_kernel(a, b, gemm::Parallelism::None);
    }
    let bv = b.view();
    let mut out = Array2::<T>::zeros((m, n));
    out.axis_chunks_iter_mut(Axis(0), chunk)
        .into_par_iter()
        .zip(a.axis_chunks_iter(Axis(0), chunk).into_par_iter())
        .for_each(|(mut c_blk, a_blk)| {
            c_blk.assign(&gemm_kernel(&a_blk, &bv, gemm::Parallelism::None));
        });
    out
}

/// Shape-aware *parallel* GEMM strategy (no serial/parallel gate): the product is always spread
/// across the pool, splitting whichever axis the backend leaves serial
///
/// - `m >= n` and few columns (`n < gemm_colpar_min_cols_per_thread() * threads`): split the rows
///   via `gemm_rowsplit`, since the backend only parallelizes over `n` and that starves when the
///   columns are too few to feed the pool (matvecs, tall-skinny, training-loop GEMMs)
/// - otherwise (wide `n > m`, or many columns): hand the whole product to the `gemm` crate, which
///   parallelizes over the columns and packs `B` only once
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
fn gemm_par_strategy<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _k) = a.dim();
    let n = b.ncols();
    let threads = rayon::current_num_threads();
    if m >= n && n < threads.saturating_mul(gemm_colpar_min_cols_per_thread()) {
        gemm_rowsplit(a, b)
    } else {
        gemm_kernel(a, b, gemm::Parallelism::Rayon(threads))
    }
}

/// `C = A @ B` with explicit parallelism control, both arms on the `gemm` crate's kernels
///
/// `A` is `(m, k)`, `B` is `(k, n)`; the result is a freshly allocated, standard-layout `(m, n)`
/// array. Any storage works (owned, view, transpose, slice)
///
/// - `parallel == false`: 1 serial `gemm` call (`Parallelism::None`) - still the `gemm` crate's
///   SIMD kernel, just on 1 thread (faster than ndarray's `matrixmultiply` `.dot()`). Use this to
///   force a product serial from inside an already-parallel region, so it does not fork rayon again
///   (nested parallelism / oversubscription)
/// - `parallel == true`: the shape-aware parallel strategy ([`gemm_par_strategy`])
///
/// Callers that want the work-size gate to make the serial-vs-parallel choice should use
/// [`gemm_par_auto`] instead
///
/// # Panics
///
/// - If `A`'s column count differs from `B`'s row count
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub(crate) fn gemm_par_switch<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    parallel: bool,
) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(
        k, kb,
        "gemm_par_switch: inner dimensions disagree (a is {m}x{k}, b is {kb}x{n})",
    );
    if parallel {
        gemm_par_strategy(a, b)
    } else {
        gemm_kernel(a, b, gemm::Parallelism::None)
    }
}

/// `C = A @ B`, parallelized automatically by estimated FLOPs and shape
///
/// Below `MatmulElem::gemm_rayon_min_flops` the product runs serial (the per-call rayon dispatch
/// would dominate the tiny products in tight loops, e.g. RNN/LSTM timesteps); at or above it the
/// shape-aware parallel strategy ([`gemm_par_strategy`]) takes over. This is the entry for callers
/// that have no opinion on parallelism and want the gate to decide; [`gemm_par_switch`] is the
/// entry for callers that must control it explicitly
///
/// # Panics
///
/// - If `A`'s column count differs from `B`'s row count
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utils"
))]
pub(crate) fn gemm_par_auto<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    let n = b.ncols();
    let flops = 2usize.saturating_mul(m).saturating_mul(k).saturating_mul(n);
    gemm_par_switch(a, b, flops >= T::gemm_rayon_min_flops())
}

/// Parallel matvec strategy: split `A`'s rows into per-thread blocks, each computed by a serial
/// `gemm` call ([`gemm_rowsplit`])
///
/// The `gemm` crate never parallelizes a matrix-vector product (`n == 1`) on its own, so this is
/// how a large matvec is given the extra cores' memory bandwidth. No gate - callers route here only
/// once they have decided to parallelize ([`gemv_par_auto`] gates by FLOPs, [`gemv_par_switch`]
/// takes the choice explicitly). The GEMV analogue of [`gemm_par_strategy`]
#[cfg(any(feature = "machine_learning", feature = "utils"))]
fn gemv_par_strategy<T, S1, S2>(a: &ArrayBase<S1, Ix2>, x: &ArrayBase<S2, Ix1>) -> Array1<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    // Treat the vector as a single-column matrix and reuse the row-split GEMM path
    let x_col = x.view().insert_axis(Axis(1)); // [k, 1]
    gemm_rowsplit(a, &x_col).index_axis_move(Axis(1), 0)
}

/// `y = A @ x` with explicit parallelism control (the GEMV analogue of [`gemm_par_switch`])
///
/// `A` is `(m, k)`, `x` has length `k`; the result has length `m`
/// - `parallel == false`: 1 serial `gemm` call (`Parallelism::None`) - still the `gemm` crate's
///   SIMD kernel, just on 1 thread (faster than ndarray's `.dot()`). Use this to force a matvec
///   serial from inside an already-parallel region so it does not fork rayon again (nested
///   parallelism / oversubscription)
/// - `parallel == true`: the row-split strategy ([`gemv_par_strategy`])
///
/// Callers that want the work-size gate to make the choice should use [`gemv_par_auto`] instead
///
/// # Panics
///
/// - If `A`'s column count differs from `x`'s length
#[cfg(any(feature = "machine_learning", feature = "utils"))]
pub(crate) fn gemv_par_switch<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    x: &ArrayBase<S2, Ix1>,
    parallel: bool,
) -> Array1<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    assert_eq!(
        k,
        x.len(),
        "gemv_par_switch: inner dimensions disagree (a is {m}x{k}, x has length {})",
        x.len()
    );
    if parallel {
        gemv_par_strategy(a, x)
    } else {
        // Treat the vector as a single-column matrix and run 1 serial gemm call
        let x_col = x.view().insert_axis(Axis(1)); // [k, 1]
        gemm_kernel(a, &x_col, gemm::Parallelism::None).index_axis_move(Axis(1), 0)
    }
}

/// `y = A @ x`: a matvec, row-split across rayon above `MatmulElem::gemv_rayon_min_flops`
///
/// `A` is `(m, k)`, `x` has length `k`; the result has length `m`. Below the gate the product runs
/// as 1 serial `gemm` call; at or above it [`gemv_par_strategy`] splits `A`'s rows across rayon,
/// giving the bandwidth-bound matvec the extra cores' memory bandwidth. This is the entry for
/// callers that have no opinion on parallelism and want the gate to decide; [`gemv_par_switch`] is
/// the entry for callers that must control it explicitly
///
/// # Panics
///
/// - If `A`'s column count differs from `x`'s length
#[cfg(any(feature = "machine_learning", feature = "utils"))]
pub(crate) fn gemv_par_auto<T, S1, S2>(a: &ArrayBase<S1, Ix2>, x: &ArrayBase<S2, Ix1>) -> Array1<T>
where
    T: MatmulElem,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    let flops = 2usize.saturating_mul(m).saturating_mul(k);
    gemv_par_switch(a, x, flops >= T::gemv_rayon_min_flops())
}

tunable_gate! {
    /// Element budget for one row-chunk of a tiled product whose full result would be too large
    /// to materialize at once (e.g. KNN's `[n_query, n_train]` projections or t-SNE's pairwise
    /// blocks): `chunk_rows = gemm_chunk_elems() / row_len`, clamped to `[16, 4096]` rows
    ///
    /// Tiling only pays when the shared operand overflows the cache (see `cache_resident`); there
    /// bigger chunks are strictly better, so the budget caps the transient chunk buffer at 256 MB
    /// of `f64` rather than chasing the asymptote. Overridable via [`crate::tuning::matmul`]
    pub(crate) GEMM_CHUNK_ELEMS => gemm_chunk_elems / set_gemm_chunk_elems = 33_554_432
}

/// Rows per chunk when tiling a product with `row_len`-wide output rows under `gemm_chunk_elems`
///
/// Crate-internal tiling policy; not part of the public API and carries no stability guarantee
#[doc(hidden)]
pub fn gemm_chunk_rows(row_len: usize) -> usize {
    (gemm_chunk_elems() / row_len.max(1)).clamp(16, 4096)
}

tunable_gate! {
    /// Matrix size (bytes) below which repeated row-GEMV sweeps over a shared matrix stay
    /// cache-resident, making a per-row GEMV swarm faster than a tiled GEMM
    ///
    /// When many tasks each compute `X . v` against the same `X`, the whole of `X` is re-read per
    /// task - free while `X` fits in the shared L3, but a DRAM re-stream once it overflows, where a
    /// tiled GEMM (which streams `X` once per chunk) wins instead. The default sits at a typical
    /// L3 size; the band around it is uncalibrated. Overridable via [`crate::tuning::matmul`] - the
    /// natural knob to match a machine's actual L3
    pub(crate) CACHE_RESIDENT_MAX_BYTES
        => cache_resident_max_bytes / set_cache_resident_max_bytes = 64 * 1024 * 1024
}

/// Whether an `[rows, cols]` matrix of `T` is small enough to treat as cache-resident for
/// repeated row-GEMV sweeps (see `cache_resident_max_bytes`)
///
/// Crate-internal strategy policy; not part of the public API and carries no stability
/// guarantee
#[doc(hidden)]
pub fn cache_resident<T>(rows: usize, cols: usize) -> bool {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<T>())
        < cache_resident_max_bytes()
}

#[cfg(all(
    test,
    any(
        feature = "machine_learning",
        feature = "neural_network",
        feature = "utils"
    )
))]
mod tests {
    use super::*;
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    use ndarray::Array1;
    use ndarray::{Array2, s};

    /// Deterministic pseudo-random matrices (hash-based, no rng dependency)
    fn rand_f32(r: usize, c: usize, seed: u64) -> Array2<f32> {
        Array2::from_shape_fn((r, c), |(i, j)| {
            let t = (seed as f64) * 0.731 + (i * c + j) as f64 * 0.618_033_988_7;
            ((t.sin() * 43758.5453).fract() - 0.5) as f32
        })
    }
    fn rand_f64(r: usize, c: usize, seed: u64) -> Array2<f64> {
        Array2::from_shape_fn((r, c), |(i, j)| {
            let t = (seed as f64) * 0.731 + (i * c + j) as f64 * 0.618_033_988_7;
            (t.sin() * 43758.5453).fract() - 0.5
        })
    }

    /// Independent triple-loop reference product (same element type, fixed k-order)
    fn naive<T: LinalgScalar>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
        let (m, k) = a.dim();
        let n = b.ncols();
        let mut c = Array2::<T>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = T::zero();
                for p in 0..k {
                    acc = acc + a[[i, p]] * b[[p, j]];
                }
                c[[i, j]] = acc;
            }
        }
        c
    }

    fn assert_close_f32(got: &Array2<f32>, want: &Array2<f32>, eps: f32) {
        assert_eq!(got.shape(), want.shape());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= eps, "f32 mismatch: {g} vs {w}");
        }
    }
    fn assert_close_f64(got: &Array2<f64>, want: &Array2<f64>, eps: f64) {
        assert_eq!(got.shape(), want.shape());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= eps, "f64 mismatch: {g} vs {w}");
        }
    }

    // correctness vs an independent reference

    /// gemm_par_auto matches the naive product for f32, on both the serial (< gate) and
    /// rayon (>= gate) shapes (f32 gate is 8M FLOPs)
    #[test]
    fn gemm_par_auto_matches_reference_f32() {
        for &(m, k, n) in &[
            (17usize, 23usize, 19usize),
            (128, 96, 128), /* 3.1M, serial */
        ] {
            let a = rand_f32(m, k, 1);
            let b = rand_f32(k, n, 2);
            assert_close_f32(&gemm_par_auto(&a, &b), &naive(&a, &b), 1e-2);
        }
        // Rayon path (256x300x256 ~ 39M FLOPs >= 8M): cross-check against ndarray's own dot
        let a = rand_f32(256, 300, 3);
        let b = rand_f32(300, 256, 4);
        assert_close_f32(&gemm_par_auto(&a, &b), &a.dot(&b), 1e-2);
    }

    /// gemm_par_auto matches the reference for f64, serial and rayon shapes (f64 gate is 1M)
    #[test]
    fn gemm_par_auto_matches_reference_f64() {
        for &(m, k, n) in &[
            (17usize, 23usize, 19usize),
            (64, 64, 64), /* 524K, serial */
        ] {
            let a = rand_f64(m, k, 1);
            let b = rand_f64(k, n, 2);
            assert_close_f64(&gemm_par_auto(&a, &b), &naive(&a, &b), 1e-9);
        }
        // Rayon path (128x128x128 ~ 4.2M >= 1M)
        let a = rand_f64(128, 128, 3);
        let b = rand_f64(128, 128, 4);
        assert_close_f64(&gemm_par_auto(&a, &b), &a.dot(&b), 1e-9);
    }

    /// Transposed and non-contiguous (sliced) operands feed the right strides to the kernel
    #[test]
    fn gemm_par_auto_strided_operands() {
        // A^T . B (the weight-gradient pattern): a is [k, m], a.t() is [m, k]
        let a = rand_f64(40, 24, 5);
        let b = rand_f64(40, 18, 6);
        let got = gemm_par_auto(&a.t(), &b);
        let want = naive(&a.t().to_owned(), &b);
        assert_close_f64(&got, &want, 1e-9);

        // Row-strided slice of A times a column-strided slice of B
        let a = rand_f64(40, 30, 7);
        let b = rand_f64(30, 40, 8);
        let a_sl = a.slice(s![..;2, ..]); // [20, 30], row stride 60
        let b_sl = b.slice(s![.., ..;2]); // [30, 20], col stride 2
        let got = gemm_par_auto(&a_sl, &b_sl);
        let want = naive(&a_sl.to_owned(), &b_sl.to_owned());
        assert_close_f64(&got, &want, 1e-9);
    }

    /// Thin-output GEMM (`n < threads`, above the gate) takes the row-split path - check it is
    /// correct against an independent product
    #[test]
    fn gemm_par_auto_thin_output_rowsplit() {
        // f64: 4096*64*4*2 = 2.1M >= 1M gate, n=4 thin -> row split
        let a = rand_f64(4096, 64, 61);
        let b = rand_f64(64, 4, 62);
        assert_close_f64(&gemm_par_auto(&a, &b), &a.dot(&b), 1e-9);
        // f32: 16384*64*4*2 = 8.4M >= 8M gate, n=4 thin -> row split
        let a = rand_f32(16384, 64, 63);
        let b = rand_f32(64, 4, 64);
        assert_close_f32(&gemm_par_auto(&a, &b), &a.dot(&b), 1e-2);
    }

    // the reproducibility guard: the parallel kernel matches the serial result

    /// The gemm kernel matches the serial result whether run serially or across any thread count
    /// (this test asserts bit-level equality); if a future `gemm` version split the k-reduction
    /// this test would fail
    #[test]
    fn gemm_kernel_thread_count_independent_f64() {
        // square, dense, and a thin-k shape (the one most likely to trigger a split-k reduction)
        for &(m, k, n) in &[(96usize, 96usize, 96usize), (256, 64, 64), (64, 8192, 64)] {
            let a = rand_f64(m, k, 11);
            let b = rand_f64(k, n, 12);
            let serial = gemm_kernel(&a, &b, gemm::Parallelism::None);
            for threads in [1usize, 2, 4, 8, 16, 32] {
                let par = gemm_kernel(&a, &b, gemm::Parallelism::Rayon(threads));
                assert!(
                    serial
                        .iter()
                        .zip(par.iter())
                        .all(|(s, p)| s.to_bits() == p.to_bits()),
                    "gemm f64 {m}x{k}x{n} differs between serial and Rayon({threads})"
                );
            }
        }
    }

    /// Same serial-matching guarantee for f32 (asserts bit-level equality)
    #[test]
    fn gemm_kernel_thread_count_independent_f32() {
        for &(m, k, n) in &[(96usize, 96usize, 96usize), (64, 8192, 64)] {
            let a = rand_f32(m, k, 13);
            let b = rand_f32(k, n, 14);
            let serial = gemm_kernel(&a, &b, gemm::Parallelism::None);
            for threads in [1usize, 4, 32] {
                let par = gemm_kernel(&a, &b, gemm::Parallelism::Rayon(threads));
                assert!(
                    serial
                        .iter()
                        .zip(par.iter())
                        .all(|(s, p)| s.to_bits() == p.to_bits()),
                    "gemm f32 {m}x{k}x{n} differs between serial and Rayon({threads})"
                );
            }
        }
    }

    /// Running the same product twice on the same machine gives the same result (this test asserts
    /// bit-level equality)
    #[test]
    fn gemm_par_auto_run_to_run_deterministic() {
        let a = rand_f64(200, 200, 21);
        let b = rand_f64(200, 200, 22);
        let c1 = gemm_par_auto(&a, &b);
        let c2 = gemm_par_auto(&a, &b);
        assert!(
            c1.iter()
                .zip(c2.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    // gemv

    /// gemv_par_auto matches the reference for both the serial and the row-split path
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn gemv_par_auto_matches_reference() {
        // small (< 524K FLOPs -> serial) and large (>= 524K -> rayon row split)
        for &(m, k) in &[(40usize, 24usize), (8192, 64) /* 1M, row split */] {
            let a = rand_f64(m, k, 31);
            let x = Array1::from_shape_fn(k, |i| ((i as f64) * 0.37).sin());
            let got = gemv_par_auto(&a, &x);
            let want = a.dot(&x);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() <= 1e-9, "gemv mismatch: {g} vs {w}");
            }
        }
    }

    /// The GEMV row split agrees with a single serial matvec numerically (this test allows a small
    /// rounding-level tolerance rather than exact equality, since each block is a `gemm` call with a
    /// different `m` and the kernel's internal k-blocking can depend on `m`)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn gemv_par_auto_rowsplit_matches_serial_numerically() {
        let m = 8192; // 8192*64*2 ~ 1M >= the 524K gate -> row-split path
        let k = 64;
        let a = rand_f64(m, k, 41);
        let x = Array1::from_shape_fn(k, |i| ((i as f64) * 0.59).sin());
        let split = gemv_par_auto(&a, &x);
        let serial = gemm_kernel(&a, &x.view().insert_axis(Axis(1)), gemm::Parallelism::None)
            .index_axis_move(Axis(1), 0);
        assert_eq!(split.len(), serial.len());
        for (s, p) in split.iter().zip(serial.iter()) {
            assert!(
                (s - p).abs() <= 1e-10,
                "gemv row split vs serial: {s} vs {p}"
            );
        }
    }

    /// GEMV is run-to-run deterministic on the same machine (same chunks -> same result)
    #[cfg(any(feature = "machine_learning", feature = "utils"))]
    #[test]
    fn gemv_par_auto_run_to_run_deterministic() {
        let a = rand_f64(8192, 64, 51);
        let x = Array1::from_shape_fn(64, |i| ((i as f64) * 0.23).sin());
        let y1 = gemv_par_auto(&a, &x);
        let y2 = gemv_par_auto(&a, &x);
        assert!(
            y1.iter()
                .zip(y2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    // degenerate shapes

    #[test]
    fn gemm_par_auto_edge_cases() {
        // zero-sized axes
        let a = Array2::<f64>::zeros((0, 4));
        let b = Array2::<f64>::zeros((4, 3));
        assert_eq!(gemm_par_auto(&a, &b).shape(), &[0, 3]);
        let a = Array2::<f64>::zeros((3, 0));
        let b = Array2::<f64>::zeros((0, 4));
        let c = gemm_par_auto(&a, &b);
        assert_eq!(c.shape(), &[3, 4]);
        assert!(c.iter().all(|&x| x == 0.0));
        // 1x1
        let a = Array2::<f64>::from_elem((1, 1), 3.0);
        let b = Array2::<f64>::from_elem((1, 1), 4.0);
        assert_eq!(gemm_par_auto(&a, &b)[[0, 0]], 12.0);
    }

    #[test]
    #[should_panic]
    fn gemm_par_auto_dimension_mismatch_panics() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array2::<f64>::zeros((4, 2));
        let _ = gemm_par_auto(&a, &b);
    }
}

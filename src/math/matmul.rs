//! The crate's few additions to the [`gemmkit`](https://docs.rs/gemmkit) matrix-product backend
//!
//! Since the backend's own ndarray adapter ([`gemmkit-ndarray`](https://docs.rs/gemmkit-ndarray))
//! is already the right call-site API, the estimators call it directly: `gemmkit_ndarray::dot`
//! for an allocating product on the backend's automatic scheduling, and `gemmkit_ndarray::gemm` /
//! `gemm_fused` where the caller owns the output buffer or fuses an epilogue. This module holds
//! only what the adapter does not provide:
//!
//! - `dot_par`: the allocating product with an explicit [`gemmkit::Parallelism`] (`dot` always
//!   uses the automatic default; callers inside an already-parallel rayon region pass
//!   `Parallelism::Serial` so the product does not fork again)
//! - `matvec`: the matvec with `Array1` operands (the adapter is matrix-shaped; this wraps the
//!   vector as a `[k, 1]` column, which the backend reroutes to its bandwidth-bound GEMV path)
//! - the row-chunk tiling policy (`gemm_chunk_rows`, `cache_resident`) for callers that
//!   materialize a product too large to hold at once (KNN, t-SNE, MeanShift)
//!
//! Scheduling is entirely gemmkit's: `Parallelism::Rayon(0)` gates serial-vs-parallel on the
//! work size, ramps the worker count onto persistent exact-fit pool tiers, parallelizes matvecs
//! by memory bandwidth, and runs on the caller's pool when already inside a rayon region. The
//! knobs live in gemmkit too - `GEMMKIT_*` environment variables (profiles from the
//! [`gemmkit-tune`](https://docs.rs/gemmkit-tune) autotuner), programmatic via
//! [`crate::tuning::matmul::backend`]
//!
//! ## Reproducibility
//!
//! gemmkit's blocking and job order are independent of the worker count, so for a fixed machine
//! and configuration the same product reproduces the same result bit-for-bit, no matter how many
//! threads ran it (and re-running is deterministic). Fused epilogues (bias / activation) are
//! bitwise-identical to the plain product followed by the same scalar map

#[cfg(feature = "machine_learning")]
use ndarray::{Array1, Ix1};
#[cfg(any(feature = "machine_learning", feature = "neural_network"))]
use ndarray::{Array2, ArrayBase, Data, Ix2};

/// `A @ B` into a fresh standard-layout array, with an explicit [`gemmkit::Parallelism`]
///
/// The allocating twin of `gemmkit_ndarray::dot` for callers that must control the parallelism -
/// pass `Parallelism::Serial` from inside an already-parallel rayon region so the product does
/// not fork again, or `Parallelism::Rayon(0)` for the backend's automatic scheduling (which is
/// what plain `dot` always uses). `A` is `(m, k)`, `B` is `(k, n)`; any storage works (owned,
/// view, transpose, slice - strides pass through zero-copy)
///
/// # Panics
///
/// - If `A`'s column count differs from `B`'s row count
#[cfg(any(feature = "machine_learning", feature = "neural_network"))]
pub(crate) fn dot_par<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    par: gemmkit::Parallelism,
) -> Array2<T>
where
    T: gemmkit::GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(
        k, kb,
        "dot_par: inner dimensions disagree (a is {m}x{k}, b is {kb}x{n})",
    );
    // `beta == 0` means the fill value is never read; `from_elem` avoids a `Zero` bound
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemmkit_ndarray::gemm(T::ONE, a, b, T::ZERO, &mut c, par);
    c
}

/// `y = A @ x` for `Array1` operands, with an explicit [`gemmkit::Parallelism`]
///
/// The backend's adapter is matrix-shaped, so this wraps `x` as a `[k, 1]` column and unwraps
/// the `[m, 1]` result; gemmkit detects the `n == 1` shape and runs its bandwidth-bound GEMV
/// path (serial below an LLC-derived byte floor, then split across the memory-bandwidth worker
/// cap - bit-identical at any worker count). `A` is `(m, k)`, `x` has length `k`. Pass
/// `Parallelism::Rayon(0)` for the automatic scheduling, `Parallelism::Serial` from inside an
/// already-parallel region
///
/// # Panics
///
/// - If `A`'s column count differs from `x`'s length
#[cfg(feature = "machine_learning")]
pub(crate) fn matvec<T, S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    x: &ArrayBase<S2, Ix1>,
    par: gemmkit::Parallelism,
) -> Array1<T>
where
    T: gemmkit::GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    use ndarray::Axis;

    let (m, k) = a.dim();
    assert_eq!(
        k,
        x.len(),
        "matvec: inner dimensions disagree (a is {m}x{k}, x has length {})",
        x.len()
    );
    let x_col = x.view().insert_axis(Axis(1)); // [k, 1]
    let mut y = Array1::from_elem(m, T::ZERO);
    let mut y_col = y.view_mut().insert_axis(Axis(1)); // [m, 1]
    gemmkit_ndarray::gemm(T::ONE, a, &x_col, T::ZERO, &mut y_col, par);
    y
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

#[cfg(all(test, any(feature = "machine_learning", feature = "neural_network")))]
mod tests {
    use super::*;
    use gemmkit::Parallelism;
    use gemmkit_ndarray::dot;
    #[cfg(feature = "machine_learning")]
    use ndarray::Array1;
    use ndarray::{Array2, LinalgScalar, s};

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

    /// The backend's `dot` matches the naive product for f32 on both sides of its work gate
    /// (`m*n*k` 589,824 by default)
    #[test]
    fn dot_matches_reference_f32() {
        for &(m, k, n) in &[
            (17usize, 23usize, 19usize), /* tiny, serial */
            (64, 48, 64),                /* below the gate */
        ] {
            let a = rand_f32(m, k, 1);
            let b = rand_f32(k, n, 2);
            assert_close_f32(&dot(&a, &b), &naive(&a, &b), 1e-2);
        }
        // Above the gate (256*300*256 ~ 19.7M): cross-check against ndarray's own dot
        let a = rand_f32(256, 300, 3);
        let b = rand_f32(300, 256, 4);
        assert_close_f32(&dot(&a, &b), &a.dot(&b), 1e-2);
    }

    /// The backend's `dot` matches the reference for f64 on both sides of the work gate
    #[test]
    fn dot_matches_reference_f64() {
        for &(m, k, n) in &[
            (17usize, 23usize, 19usize),
            (64, 64, 64), /* below the gate */
        ] {
            let a = rand_f64(m, k, 1);
            let b = rand_f64(k, n, 2);
            assert_close_f64(&dot(&a, &b), &naive(&a, &b), 1e-9);
        }
        // Above the gate (128^3 ~ 2.1M)
        let a = rand_f64(128, 128, 3);
        let b = rand_f64(128, 128, 4);
        assert_close_f64(&dot(&a, &b), &a.dot(&b), 1e-9);
    }

    /// Transposed and non-contiguous (sliced) operands feed the right strides to the kernel
    #[test]
    fn dot_strided_operands() {
        // A^T . B (the weight-gradient pattern): a is [k, m], a.t() is [m, k]
        let a = rand_f64(40, 24, 5);
        let b = rand_f64(40, 18, 6);
        let got = dot(&a.t(), &b);
        let want = naive(&a.t().to_owned(), &b);
        assert_close_f64(&got, &want, 1e-9);

        // Row-strided slice of A times a column-strided slice of B
        let a = rand_f64(40, 30, 7);
        let b = rand_f64(30, 40, 8);
        let a_sl = a.slice(s![..;2, ..]); // [20, 30], row stride 60
        let b_sl = b.slice(s![.., ..;2]); // [30, 20], col stride 2
        let got = dot(&a_sl, &b_sl);
        let want = naive(&a_sl.to_owned(), &b_sl.to_owned());
        assert_close_f64(&got, &want, 1e-9);
    }

    /// Thin-output GEMM (`n` a handful, above the work gate) - the shape the old hand-rolled
    /// row split existed for - is correct on the backend's own scheduling
    #[test]
    fn dot_thin_output() {
        // f64: 4096*64*4 = 1.05M above the gate, n = 4
        let a = rand_f64(4096, 64, 61);
        let b = rand_f64(64, 4, 62);
        assert_close_f64(&dot(&a, &b), &a.dot(&b), 1e-9);
        // f32: 16384*64*4 = 4.2M above the gate
        let a = rand_f32(16384, 64, 63);
        let b = rand_f32(64, 4, 64);
        assert_close_f32(&dot(&a, &b), &a.dot(&b), 1e-2);
    }

    // the reproducibility guard

    /// The backend's contract: for a fixed config the result is identical no matter how many
    /// workers ran it. Serial and forced-parallel also agree bitwise on the current kernels
    /// (both run the same kernel with thread-count-independent blocking)
    #[test]
    fn dot_par_thread_count_independent_f64() {
        // square, dense, and a thin-k shape (the one most likely to trigger a split-k reduction)
        for &(m, k, n) in &[(96usize, 96usize, 96usize), (256, 64, 64), (64, 8192, 64)] {
            let a = rand_f64(m, k, 11);
            let b = rand_f64(k, n, 12);
            let serial = dot_par(&a, &b, Parallelism::Serial);
            for threads in [2usize, 4, 8, 16, 32] {
                let par = dot_par(&a, &b, Parallelism::Rayon(threads));
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

    /// Same worker-count-independence check for f32
    #[test]
    fn dot_par_thread_count_independent_f32() {
        for &(m, k, n) in &[(96usize, 96usize, 96usize), (64, 8192, 64)] {
            let a = rand_f32(m, k, 13);
            let b = rand_f32(k, n, 14);
            let serial = dot_par(&a, &b, Parallelism::Serial);
            for threads in [2usize, 4, 32] {
                let par = dot_par(&a, &b, Parallelism::Rayon(threads));
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
    fn dot_run_to_run_deterministic() {
        let a = rand_f64(200, 200, 21);
        let b = rand_f64(200, 200, 22);
        let c1 = dot(&a, &b);
        let c2 = dot(&a, &b);
        assert!(
            c1.iter()
                .zip(c2.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    // the epilogue guard: fused bias+activation is bitwise-identical to the unfused pipeline

    /// gemmkit's fused epilogue contract for f32/f64: `gemm_fused` equals plain `gemm` followed
    /// by the same scalar bias-add and activation, bit for bit. The NN layers rely on this to
    /// fuse bias + ReLU without changing numerics
    #[test]
    fn gemm_fused_bias_relu_bitwise_matches_unfused() {
        let a = rand_f32(64, 48, 31);
        let b = rand_f32(48, 40, 32);
        let bias: Vec<f32> = (0..40).map(|j| (j as f32) * 0.05 - 1.0).collect();

        let mut fused = Array2::from_elem((64, 40), 0.0f32);
        gemmkit_ndarray::gemm_fused(
            1.0,
            &a,
            &b,
            0.0,
            &mut fused,
            Some(gemmkit_ndarray::Bias::PerCol(&bias)),
            Some(gemmkit_ndarray::Activation::Relu),
            Parallelism::Rayon(0),
        );

        let mut want = dot(&a, &b);
        for mut row in want.rows_mut() {
            for (v, bj) in row.iter_mut().zip(bias.iter()) {
                *v = (*v + bj).max(0.0);
            }
        }
        assert!(
            fused
                .iter()
                .zip(want.iter())
                .all(|(f, w)| f.to_bits() == w.to_bits()),
            "fused bias+ReLU differs from the unfused pipeline"
        );
    }

    // matvec

    /// matvec matches the reference on both cache-resident and larger shapes
    #[cfg(feature = "machine_learning")]
    #[test]
    fn matvec_matches_reference() {
        for &(m, k) in &[(40usize, 24usize), (8192, 64)] {
            let a = rand_f64(m, k, 31);
            let x = Array1::from_shape_fn(k, |i| ((i as f64) * 0.37).sin());
            let got = matvec(&a, &x, Parallelism::Rayon(0));
            let want = a.dot(&x);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() <= 1e-9, "matvec mismatch: {g} vs {w}");
            }
        }
    }

    /// Forced-serial and auto matvec agree bitwise (the backend reduces each output element over
    /// the full `k` on one worker, so the matvec is bit-identical at any worker count)
    #[cfg(feature = "machine_learning")]
    #[test]
    fn matvec_serial_and_auto_agree_bitwise() {
        let a = rand_f64(8192, 64, 41);
        let x = Array1::from_shape_fn(64, |i| ((i as f64) * 0.59).sin());
        let serial = matvec(&a, &x, Parallelism::Serial);
        let auto = matvec(&a, &x, Parallelism::Rayon(0));
        assert_eq!(serial.len(), auto.len());
        assert!(
            serial
                .iter()
                .zip(auto.iter())
                .all(|(s, p)| s.to_bits() == p.to_bits()),
            "matvec serial vs auto differ"
        );
    }

    /// matvec is run-to-run deterministic on the same machine
    #[cfg(feature = "machine_learning")]
    #[test]
    fn matvec_run_to_run_deterministic() {
        let a = rand_f64(8192, 64, 51);
        let x = Array1::from_shape_fn(64, |i| ((i as f64) * 0.23).sin());
        let y1 = matvec(&a, &x, Parallelism::Rayon(0));
        let y2 = matvec(&a, &x, Parallelism::Rayon(0));
        assert!(
            y1.iter()
                .zip(y2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    // degenerate shapes

    #[test]
    fn dot_edge_cases() {
        // zero-sized axes (handled inside gemmkit: `m == 0`/`n == 0` write nothing, `k == 0`
        // leaves `beta * C`, which is all zeros here)
        let a = Array2::<f64>::zeros((0, 4));
        let b = Array2::<f64>::zeros((4, 3));
        assert_eq!(dot(&a, &b).shape(), &[0, 3]);
        let a = Array2::<f64>::zeros((3, 0));
        let b = Array2::<f64>::zeros((0, 4));
        let c = dot(&a, &b);
        assert_eq!(c.shape(), &[3, 4]);
        assert!(c.iter().all(|&x| x == 0.0));
        // 1x1
        let a = Array2::<f64>::from_elem((1, 1), 3.0);
        let b = Array2::<f64>::from_elem((1, 1), 4.0);
        assert_eq!(dot(&a, &b)[[0, 0]], 12.0);
    }

    #[test]
    #[should_panic]
    fn dot_par_dimension_mismatch_panics() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array2::<f64>::zeros((4, 2));
        let _ = dot_par(&a, &b, Parallelism::Serial);
    }
}

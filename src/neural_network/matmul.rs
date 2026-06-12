//! Shared rayon-parallel matrix multiply for the neural-network layers
//!
//! Without a BLAS backend, `ndarray`'s `dot` runs on the single-threaded `matrixmultiply`
//! kernels. [`par_matmul`] keeps those kernels for the inner work (cache blocking + SIMD) and
//! adds the missing parallelism at exactly one level: large products are split into row or
//! column blocks across rayon, each block computed by a serial `dot`.
//!
//! Block splitting never reorders the per-element accumulation over `k`, so the result is
//! bitwise identical to the serial `a.dot(b)` - unlike a k-split reduction, which would change
//! f32 summation order and break reproducibility.

use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

/// Minimum estimated FLOPs (`2*m*k*n`) before a matmul is worth splitting across rayon.
///
/// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see benches/RESULTS.md: the measured crossover bracket is 2.1M-4.2M FLOPs
/// (square and skinny shapes agree), so the gate sits at the proven-win end of the bracket.
const PAR_GEMM_MIN_FLOPS: usize = 4_000_000;

/// Minimum rows (or columns) per parallel block.
///
/// Each block's `dot` re-packs the shared operand, so blocks must be tall (or wide) enough to
/// amortize that packing. Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see benches/RESULTS.md: 16-64 sit on a
/// plateau (within noise of each other) and 128+ degrades, so 64 is kept.
const PAR_GEMM_MIN_BLOCK: usize = 64;

/// `C = A @ B` with block-parallel execution for large products.
///
/// Splits the longer output axis (rows of `A`, or columns of `B`) into per-thread blocks and
/// computes each with the serial `matrixmultiply` kernel; small products fall through to a plain
/// `a.dot(b)`. Splitting the `m`/`n` axes leaves every output element's `k`-accumulation order
/// untouched, so the result is **bitwise identical** to the serial product regardless of the
/// thread count - parallelism here never costs reproducibility.
///
/// The returned array is guaranteed to be in standard (row-major) layout. (A bare `a.dot(b)`
/// does not guarantee this: it returns a column-major result when both operands have a row
/// stride of 1, which arrays with a length-1 axis can exhibit.)
///
/// # Parameters
///
/// - `a` - Left operand with shape (m, k)
/// - `b` - Right operand with shape (k, n)
///
/// # Returns
///
/// - `Array2<f32>` - The product with shape (m, n), standard layout
///
/// # Panics
///
/// - If the inner dimensions disagree (same contract as `dot`)
pub(crate) fn par_matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let n = b.ncols();

    let flops = 2usize
        .saturating_mul(m)
        .saturating_mul(k)
        .saturating_mul(n);
    if flops < PAR_GEMM_MIN_FLOPS {
        return into_standard_layout(a.dot(&b));
    }

    split_matmul(a, b, PAR_GEMM_MIN_BLOCK)
}

/// The block-parallel product itself, with no FLOPs gate: always splits (when the block size
/// allows). Kept separate from [`par_matmul`] so the calibration bench can time the split path
/// against the serial `dot` on either side of the gate, and sweep `min_block`. Reachable outside
/// the crate only through `bench_internals`
pub fn split_matmul(a: ArrayView2<f32>, b: ArrayView2<f32>, min_block: usize) -> Array2<f32> {
    let (m, _) = a.dim();
    let n = b.ncols();
    let threads = rayon::current_num_threads();

    // Split the longer output axis so skinny shapes (e.g. [in, B*T] @ [B*T, units] weight
    // reductions with small m and n) still get enough blocks
    if m >= n {
        let chunk = m.div_ceil(threads).max(min_block);
        if chunk >= m {
            return into_standard_layout(a.dot(&b));
        }
        let mut c = Array2::<f32>::zeros((m, n));
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
        let mut c = Array2::<f32>::zeros((m, n));
        c.axis_chunks_iter_mut(Axis(1), chunk)
            .into_par_iter()
            .zip(b.axis_chunks_iter(Axis(1), chunk).into_par_iter())
            .for_each(|(mut c_blk, b_blk)| {
                c_blk.assign(&a.dot(&b_blk));
            });
        c
    }
}

/// Normalizes a `dot` result to standard layout (no-op copy-free in the common row-major case)
fn into_standard_layout(m: Array2<f32>) -> Array2<f32> {
    if m.is_standard_layout() {
        m
    } else {
        m.as_standard_layout().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    /// Deterministic random matrix for comparing the parallel and serial paths
    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = crate::random::make_rng(Some(seed));
        Array::random_using((rows, cols), Uniform::new(-1.0, 1.0).unwrap(), &mut rng)
    }

    // Bitwise equivalence with the serial product

    /// Row-split path (m >= n, above the FLOP threshold) is bitwise identical to serial dot
    #[test]
    fn par_matmul_row_split_bitwise_equals_serial() {
        // 2*512*64*256 = 16.8M FLOPs >= threshold, m = 512 > n = 256 -> row split
        let a = random_matrix(512, 64, 1);
        let b = random_matrix(64, 256, 2);
        let par = par_matmul(a.view(), b.view());
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
        let par = par_matmul(a.view(), b.view());
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
        let par = par_matmul(a.view(), b.view());
        let serial = a.dot(&b);
        assert!(par.iter().zip(serial.iter()).all(|(x, y)| x == y));
    }

    /// Transposed operands (the `x.t().dot(dz)` weight-reduction pattern) match serial
    #[test]
    fn par_matmul_transposed_operands_match_serial() {
        // [64, 2048] @ [2048, 64] from transposes: 2*64*2048*64 = 16.8M FLOPs, m == n -> row split
        let x = random_matrix(2048, 64, 7);
        let dz = random_matrix(2048, 64, 8);
        let par = par_matmul(x.t(), dz.view());
        let serial = x.t().dot(&dz);
        assert!(
            par.iter().zip(serial.iter()).all(|(a, b)| a == b),
            "transposed-operand product must match serial"
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

        let c = par_matmul(a.view(), b);
        assert!(c.is_standard_layout(), "output must be standard layout");
        let serial = a.dot(&b);
        assert!(c.iter().zip(serial.iter()).all(|(x, y)| x == y));
    }

    // Degenerate shapes

    /// Zero-sized axes produce the empty/zero product without panicking
    #[test]
    fn par_matmul_zero_sized_axes() {
        let a = Array2::<f32>::zeros((0, 4));
        let b = Array2::<f32>::zeros((4, 3));
        let c = par_matmul(a.view(), b.view());
        assert_eq!(c.shape(), &[0, 3]);

        let a = Array2::<f32>::zeros((3, 0));
        let b = Array2::<f32>::zeros((0, 4));
        let c = par_matmul(a.view(), b.view());
        assert_eq!(c.shape(), &[3, 4]);
        assert!(c.iter().all(|&x| x == 0.0));
    }

    /// 1x1 product
    #[test]
    fn par_matmul_one_by_one() {
        let a = Array2::<f32>::from_elem((1, 1), 3.0);
        let b = Array2::<f32>::from_elem((1, 1), 4.0);
        let c = par_matmul(a.view(), b.view());
        assert_eq!(c[[0, 0]], 12.0);
    }

    /// Mismatched inner dimensions panic, matching `dot`'s contract
    #[test]
    #[should_panic]
    fn par_matmul_dimension_mismatch_panics() {
        let a = Array2::<f32>::zeros((2, 3));
        let b = Array2::<f32>::zeros((4, 2));
        par_matmul(a.view(), b.view());
    }

}

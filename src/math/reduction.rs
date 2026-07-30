//! Deterministic blocked reductions
//!
//! A bare rayon `par_iter().sum::<f64>()` (or `fold().reduce()`) groups its partial sums by
//! work-stealing. The float accumulation order, and therefore the rounded result, then varies
//! from run to run and with the thread count. The helpers here fix the grouping instead. The
//! input is cut into fixed-size blocks. Each block folds serially, and rayon's indexed
//! `collect` gathers the per-block results in block order before they merge sequentially. The
//! grouping depends only on [`DET_REDUCE_BLOCK`](crate::math::reduction::DET_REDUCE_BLOCK),
//! never on scheduling or the `parallel` flag. Re-running on the same machine reproduces the
//! result, though not necessarily bit-for-bit. The flag is a performance hint that only decides
//! whether the blocks run on rayon or in sequence. A caller that pairs these helpers with some
//! other serial kernel below a size threshold makes that switch part of its own reproducibility
//! surface.
//!
//! [`det_reduce`](crate::math::reduction::det_reduce) folds a slice.
//! [`det_reduce_range`](crate::math::reduction::det_reduce_range) folds fixed blocks of an
//! index range instead, for reductions that read several arrays at once or accumulate
//! something richer than a scalar.

use rayon::prelude::{IntoParallelIterator, ParallelIterator, ParallelSlice};
use std::ops::Range;

/// Fixed reduction block size (elements).
///
/// The constant counts elements, not bytes, and is shared by every element type. Changing this
/// value changes the deterministic result grouping, so it is part of the reproducibility
/// surface. Bump it only on purpose.
pub const DET_REDUCE_BLOCK: usize = 16_384;

/// Folds `slice` with a deterministic, scheduling-independent grouping, on rayon or
/// sequentially per the `parallel` flag.
///
/// Both paths fold the same [`DET_REDUCE_BLOCK`]-sized blocks in the same order, so the flag
/// never changes the result. Pass the side of a calibrated size gate, or `false` when in doubt.
/// An input shorter than one block gains nothing from rayon.
///
/// # Parameters
///
/// - `slice` - The values to reduce
/// - `parallel` - Whether the blocks run on rayon (a performance hint)
/// - `fold_block` - Serial fold over one block
/// - `merge` - Combines 2 partial results, applied left-to-right in block order
/// - `identity` - The fold identity (returned for an empty slice)
///
/// # Returns
///
/// - `A` - The merged result, the same on a given machine regardless of the flag value
///
/// # Examples
///
/// ```rust
/// use rustyml::math::reduction::det_reduce;
///
/// let data: Vec<f32> = (0..100_000).map(|i| (i as f32).sin()).collect();
/// // f32 elements, f64 accumulator
/// let sum_sq = det_reduce(
///     &data,
///     true,
///     |block| block.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>(),
///     |a, b| a + b,
///     0.0,
/// );
/// // The flag is a performance hint: serial gives the same bits
/// assert_eq!(
///     sum_sq,
///     det_reduce(
///         &data,
///         false,
///         |b| b.iter().map(|&v| (v as f64) * (v as f64)).sum(),
///         |a, b| a + b,
///         0.0
///     )
/// );
/// ```
pub fn det_reduce<T, A, F, M>(
    slice: &[T],
    parallel: bool,
    fold_block: F,
    merge: M,
    identity: A,
) -> A
where
    T: Sync,
    A: Send,
    F: Fn(&[T]) -> A + Sync + Send,
    M: Fn(A, A) -> A,
{
    if parallel {
        let parts: Vec<A> = slice.par_chunks(DET_REDUCE_BLOCK).map(fold_block).collect();
        parts.into_iter().fold(identity, merge)
    } else {
        slice
            .chunks(DET_REDUCE_BLOCK)
            .map(fold_block)
            .fold(identity, merge)
    }
}

/// Folds the index range `0..n` over fixed [`DET_REDUCE_BLOCK`]-sized blocks, on rayon or
/// sequentially per the `parallel` flag.
///
/// The index-range twin of [`det_reduce`] handles reductions that cannot use a single slice.
/// Examples include zipping several arrays, indexing rows of a matrix, or accumulating a
/// compound state such as per-bucket sums or running moments. `fold_block` receives each
/// block's index sub-range and folds it serially. The per-block results merge left-to-right
/// in block order. Both paths use the same grouping, so the flag never changes the result
/// bits. Pass the side of a calibrated work gate, which may count more than the index range
/// (for example indices times per-index cost).
///
/// # Parameters
///
/// - `n` - The exclusive upper bound of the index range
/// - `parallel` - Whether the blocks run on rayon (a performance hint)
/// - `fold_block` - Serial fold over one block's index range
/// - `merge` - Combines 2 partial results, applied left-to-right in block order
/// - `identity` - The fold identity (returned when `n == 0`)
///
/// # Returns
///
/// - `A` - The merged result, the same on a given machine regardless of the flag value
///
/// # Examples
///
/// ```rust
/// use rustyml::math::reduction::det_reduce_range;
///
/// let xs: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
/// let ys: Vec<f64> = (0..100_000).map(|i| (i as f64).cos()).collect();
/// let dot = det_reduce_range(
///     xs.len(),
///     true,
///     |range| range.map(|i| xs[i] * ys[i]).sum::<f64>(),
///     |a, b| a + b,
///     0.0,
/// );
/// assert!(dot.is_finite());
/// ```
pub fn det_reduce_range<A, F, M>(
    n: usize,
    parallel: bool,
    fold_block: F,
    merge: M,
    identity: A,
) -> A
where
    A: Send,
    F: Fn(Range<usize>) -> A + Sync + Send,
    M: Fn(A, A) -> A,
{
    let n_blocks = n.div_ceil(DET_REDUCE_BLOCK);
    let block_range = |b: usize| {
        let start = b * DET_REDUCE_BLOCK;
        start..(start + DET_REDUCE_BLOCK).min(n)
    };
    if parallel {
        let parts: Vec<A> = (0..n_blocks)
            .into_par_iter()
            .map(|b| fold_block(block_range(b)))
            .collect();
        parts.into_iter().fold(identity, merge)
    } else {
        (0..n_blocks)
            .map(|b| fold_block(block_range(b)))
            .fold(identity, merge)
    }
}

/// Unit tests for the deterministic blocked reductions.
#[cfg(test)]
mod tests {
    use super::*;

    /// The `parallel` flag is a pure performance hint: both paths of both helpers produce
    /// identical results across length edge cases (empty, sub-block, multi-block).
    #[test]
    fn parallel_flag_does_not_change_bits() {
        for len in [0usize, 1, 100, DET_REDUCE_BLOCK, DET_REDUCE_BLOCK * 3 + 17] {
            let data: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();

            let sum = |block: &[f64]| block.iter().sum::<f64>();
            let par = det_reduce(&data, true, sum, |a, b| a + b, 0.0);
            let ser = det_reduce(&data, false, sum, |a, b| a + b, 0.0);
            assert!(
                par == ser,
                "det_reduce: flag changed the bits (len {len}): {par} vs {ser}"
            );

            let sq = |range: Range<usize>| range.map(|i| data[i] * data[i]).sum::<f64>();
            let par = det_reduce_range(len, true, sq, |a, b| a + b, 0.0);
            let ser = det_reduce_range(len, false, sq, |a, b| a + b, 0.0);
            assert!(
                par == ser,
                "det_reduce_range: flag changed the bits (len {len}): {par} vs {ser}"
            );
        }
    }

    /// The blocked fold gives the same result as a hand-rolled serial fold over the same
    /// blocks, for both flag values.
    #[test]
    fn det_reduce_matches_serial_blocked_fold() {
        for len in [0usize, 1, 100, DET_REDUCE_BLOCK, DET_REDUCE_BLOCK * 3 + 17] {
            let data: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();

            let reference = data
                .chunks(DET_REDUCE_BLOCK)
                .map(|block| block.iter().sum::<f64>())
                .fold(0.0, |a, b| a + b);

            for parallel in [false, true] {
                let got = det_reduce(
                    &data,
                    parallel,
                    |block| block.iter().sum::<f64>(),
                    |a, b| a + b,
                    0.0,
                );
                assert!(
                    got == reference,
                    "det_reduce (parallel={parallel}) must match the serial blocked fold \
                     (len {len}): {got} vs {reference}"
                );
            }
        }
    }

    /// The range fold visits every index exactly once, in the same block grouping as the
    /// slice fold, across the same length edge cases.
    #[test]
    fn det_reduce_range_matches_serial_blocked_fold() {
        for len in [0usize, 1, 100, DET_REDUCE_BLOCK, DET_REDUCE_BLOCK * 3 + 17] {
            let data: Vec<f64> = (0..len).map(|i| (i as f64).cos()).collect();

            let reference = data
                .chunks(DET_REDUCE_BLOCK)
                .map(|block| block.iter().map(|v| v * v).sum::<f64>())
                .fold(0.0, |a, b| a + b);

            for parallel in [false, true] {
                let got = det_reduce_range(
                    len,
                    parallel,
                    |range| range.map(|i| data[i] * data[i]).sum::<f64>(),
                    |a, b| a + b,
                    0.0,
                );
                assert!(
                    got == reference,
                    "det_reduce_range (parallel={parallel}) must match the serial blocked \
                     fold (len {len}): {got} vs {reference}"
                );
            }
        }
    }

    /// Compound accumulators work through the range fold: per-bucket sums and counts merge
    /// in block order and match a fully serial scatter.
    #[test]
    fn det_reduce_range_compound_accumulator() {
        let n = DET_REDUCE_BLOCK * 2 + 333;
        let values: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let buckets: Vec<usize> = (0..n).map(|i| i % 4).collect();

        let (par_sums, par_counts) = det_reduce_range(
            n,
            true,
            |range| {
                let mut sums = [0.0f64; 4];
                let mut counts = [0usize; 4];
                for i in range {
                    sums[buckets[i]] += values[i];
                    counts[buckets[i]] += 1;
                }
                (sums, counts)
            },
            |(mut sa, mut ca), (sb, cb)| {
                for k in 0..4 {
                    sa[k] += sb[k];
                    ca[k] += cb[k];
                }
                (sa, ca)
            },
            ([0.0; 4], [0; 4]),
        );

        let mut serial_sums = [0.0f64; 4];
        let mut serial_counts = [0usize; 4];
        for block in (0..n).collect::<Vec<_>>().chunks(DET_REDUCE_BLOCK) {
            let mut sums = [0.0f64; 4];
            for &i in block {
                sums[buckets[i]] += values[i];
                serial_counts[buckets[i]] += 1;
            }
            for k in 0..4 {
                serial_sums[k] += sums[k];
            }
        }

        assert_eq!(par_counts, serial_counts);
        assert!(
            par_sums == serial_sums,
            "compound range fold must match the serial blocked scatter: {par_sums:?} vs {serial_sums:?}"
        );
    }
}

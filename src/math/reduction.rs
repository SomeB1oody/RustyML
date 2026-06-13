//! Deterministic blocked reductions
//!
//! A bare rayon `par_iter().sum::<f64>()` (or `fold().reduce()`) groups its partial sums by
//! work-stealing, so the float accumulation order - and therefore the rounded result - varies
//! from run to run and with the thread count. The helpers here fix the grouping instead: the
//! input is cut into fixed-size blocks, each block is folded serially, the per-block results are
//! collected in block order (rayon's indexed `collect` preserves it), and the blocks are merged
//! sequentially. The grouping depends only on
//! [`DET_REDUCE_BLOCK`](crate::math::reduction::DET_REDUCE_BLOCK), never on scheduling or the
//! `parallel` flag, so the result is **bitwise identical at any thread count, with the flag on
//! or off**: the flag is a pure performance hint, deciding only whether the blocks run on rayon
//! or sequentially. (A caller that instead pairs these helpers with some *other* serial kernel
//! below a size threshold makes that switch part of its own reproducibility surface.)
//!
//! [`det_reduce`](crate::math::reduction::det_reduce) folds a slice;
//! [`det_reduce_range`](crate::math::reduction::det_reduce_range) folds fixed blocks
//! of an index range instead, for reductions that read several arrays at once or accumulate
//! something richer than a scalar. Despite the fixed grouping there is
//! no speed penalty against a bare rayon reduction - measured on the calibration machine the
//! blocked fold is *faster* below ~64 MB working sets (uniform 16 Ki blocks balance better than
//! rayon's adaptive splitting) and identical once memory-bandwidth-bound.

use rayon::prelude::{IntoParallelIterator, ParallelIterator, ParallelSlice};
use std::ops::Range;

/// Fixed reduction block size (elements)
///
/// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11 on `f64` elements;
/// see benches/RESULTS.md: a 4.2M-element sum-of-squares plateaus over 4K-64K-element blocks
/// (17.4-17.9x) with 16K measured fastest, degrading at 256K (too few blocks to balance).
/// The constant counts **elements**, not bytes, and is shared by every element type: 16K
/// `f32` elements (64 KB/block) sit mid-plateau too (validated in the same bench).
/// Changing this value changes the (deterministic) result grouping, so it is part of the
/// reproducibility surface - bump it only deliberately
pub const DET_REDUCE_BLOCK: usize = 16_384;

/// Folds `slice` with a deterministic, scheduling-independent grouping, on rayon or
/// sequentially per the `parallel` flag
///
/// Both paths fold the same [`DET_REDUCE_BLOCK`]-sized blocks in the same order, so the flag
/// never changes the result bits - pass the side of a calibrated size gate (or `false` when in
/// doubt; an input shorter than one block gains nothing from rayon)
///
/// # Parameters
///
/// - `slice` - The values to reduce
/// - `parallel` - Whether the blocks run on rayon (a pure performance hint)
/// - `fold_block` - Serial fold over one block
/// - `merge` - Combines two partial results; applied left-to-right in block order
/// - `identity` - The fold identity (returned for an empty slice)
///
/// # Returns
///
/// - `A` - The merged result, identical at any thread count and either flag value
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
/// // The flag is a pure performance hint: serial gives the same bits
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
/// sequentially per the `parallel` flag
///
/// The index-range twin of [`det_reduce`], for reductions that cannot be expressed over a
/// single slice: zipping several arrays, indexing rows of a matrix, or accumulating a
/// compound state (per-bucket sums, running moments, ...). `fold_block` receives each block's
/// index sub-range and folds it serially; the per-block results are merged left-to-right in
/// block order. Both paths use the same grouping, so the flag never changes the result bits -
/// pass the side of a calibrated work gate (which may count more than the index range, e.g.
/// indices x per-index cost)
///
/// # Parameters
///
/// - `n` - The exclusive upper bound of the index range
/// - `parallel` - Whether the blocks run on rayon (a pure performance hint)
/// - `fold_block` - Serial fold over one block's index range
/// - `merge` - Combines two partial results; applied left-to-right in block order
/// - `identity` - The fold identity (returned when `n == 0`)
///
/// # Returns
///
/// - `A` - The merged result, identical at any thread count and either flag value
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The `parallel` flag is a pure performance hint: both paths of both helpers produce
    /// bitwise-identical results across length edge cases (empty, sub-block, multi-block)
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
    /// blocks, for both flag values
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
    /// slice fold, across the same length edge cases
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

    /// Compound accumulators work through the range fold: per-bucket sums + counts merge in
    /// block order and match a fully serial scatter
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

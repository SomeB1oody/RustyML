//! Deterministic blocked parallel reductions over `f64` data
//!
//! A bare rayon `par_iter().sum::<f64>()` (or `fold().reduce()`) groups its partial sums by
//! work-stealing, so the float accumulation order - and therefore the rounded result - varies
//! from run to run and with the thread count. The helpers here fix the grouping instead: the
//! input is cut into fixed-size blocks, each block is folded serially, the per-block results are
//! collected in block order (rayon's indexed `collect` preserves it), and the blocks are merged
//! sequentially. The grouping depends only on
//! [`DET_REDUCE_BLOCK`](crate::math::reduction::DET_REDUCE_BLOCK), never on scheduling, so the
//! result is **bitwise identical at any thread count** (though not to the unblocked serial
//! fold - the gate below which callers stay serial therefore matters for reproducibility too).
//!
//! [`det_par_fold`](crate::math::reduction::det_par_fold) folds a `&[f64]` slice directly;
//! [`det_par_fold_range`](crate::math::reduction::det_par_fold_range) folds fixed blocks
//! of an index range instead, for reductions that read several arrays at once or accumulate
//! something richer than a scalar (e.g. per-cluster sums). Despite the fixed grouping there is
//! no speed penalty against a bare rayon reduction - measured on the calibration machine the
//! blocked fold is *faster* below ~64 MB working sets (uniform 16 Ki blocks balance better than
//! rayon's adaptive splitting) and identical once memory-bandwidth-bound.

use rayon::prelude::{IntoParallelIterator, ParallelIterator, ParallelSlice};
use std::ops::Range;

/// Fixed reduction block size (elements)
///
/// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
/// benches/RESULTS.md: a 4.2M-element sum-of-squares plateaus over 4K-64K-element blocks
/// (17.4-17.9x) with 16K measured fastest, degrading at 256K (too few blocks to balance).
/// Changing this value changes the (deterministic) result grouping, so it is part of the
/// reproducibility surface - bump it only deliberately
pub const DET_REDUCE_BLOCK: usize = 16_384;

/// Folds `slice` in parallel with a deterministic, scheduling-independent grouping
///
/// # Parameters
///
/// - `slice` - The values to reduce
/// - `fold_block` - Serial fold over one block
/// - `merge` - Combines two partial results; applied left-to-right in block order
/// - `identity` - The fold identity (returned for an empty slice)
///
/// # Returns
///
/// - `A` - The merged result, identical at any thread count
///
/// # Examples
///
/// ```rust
/// use rustyml::math::reduction::det_par_fold;
///
/// let data: Vec<f64> = (0..100_000).map(|i| (i as f64).sin()).collect();
/// let sum_sq = det_par_fold(
///     &data,
///     |block| block.iter().map(|v| v * v).sum::<f64>(),
///     |a, b| a + b,
///     0.0,
/// );
/// // Bitwise-stable: re-running with any rayon thread count gives the same bits
/// assert_eq!(sum_sq, det_par_fold(&data, |b| b.iter().map(|v| v * v).sum(), |a, b| a + b, 0.0));
/// ```
pub fn det_par_fold<A, F, M>(slice: &[f64], fold_block: F, merge: M, identity: A) -> A
where
    A: Send,
    F: Fn(&[f64]) -> A + Sync + Send,
    M: Fn(A, A) -> A,
{
    let parts: Vec<A> = slice
        .par_chunks(DET_REDUCE_BLOCK)
        .map(fold_block)
        .collect();
    parts.into_iter().fold(identity, merge)
}

/// Folds the index range `0..n` in parallel over fixed [`DET_REDUCE_BLOCK`]-sized blocks
///
/// The index-range twin of [`det_par_fold`], for reductions that cannot be expressed over a
/// single `&[f64]` slice: zipping several arrays, indexing rows of a matrix, or accumulating a
/// compound state (sums per bucket, count + mean + M2, ...). `fold_block` receives each block's
/// index sub-range and folds it serially; the per-block results are merged left-to-right in
/// block order, so the grouping - and the float result - is identical at any thread count
///
/// # Parameters
///
/// - `n` - The exclusive upper bound of the index range
/// - `fold_block` - Serial fold over one block's index range
/// - `merge` - Combines two partial results; applied left-to-right in block order
/// - `identity` - The fold identity (returned when `n == 0`)
///
/// # Returns
///
/// - `A` - The merged result, identical at any thread count
///
/// # Examples
///
/// ```rust
/// use rustyml::math::reduction::det_par_fold_range;
///
/// let xs: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
/// let ys: Vec<f64> = (0..100_000).map(|i| (i as f64).cos()).collect();
/// let dot = det_par_fold_range(
///     xs.len(),
///     |range| range.map(|i| xs[i] * ys[i]).sum::<f64>(),
///     |a, b| a + b,
///     0.0,
/// );
/// assert!(dot.is_finite());
/// ```
pub fn det_par_fold_range<A, F, M>(n: usize, fold_block: F, merge: M, identity: A) -> A
where
    A: Send,
    F: Fn(Range<usize>) -> A + Sync + Send,
    M: Fn(A, A) -> A,
{
    let n_blocks = n.div_ceil(DET_REDUCE_BLOCK);
    let parts: Vec<A> = (0..n_blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * DET_REDUCE_BLOCK;
            fold_block(start..(start + DET_REDUCE_BLOCK).min(n))
        })
        .collect();
    parts.into_iter().fold(identity, merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked parallel fold gives the same result as a serial fold over the same blocks,
    /// and is independent of input length edge cases (empty, sub-block, multi-block)
    #[test]
    fn det_par_fold_matches_serial_blocked_fold() {
        for len in [0usize, 1, 100, DET_REDUCE_BLOCK, DET_REDUCE_BLOCK * 3 + 17] {
            let data: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();

            let par = det_par_fold(
                &data,
                |block| block.iter().sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
            let serial = data
                .chunks(DET_REDUCE_BLOCK)
                .map(|block| block.iter().sum::<f64>())
                .fold(0.0, |a, b| a + b);

            assert!(
                par == serial,
                "blocked parallel fold must be bitwise identical to the serial blocked fold \
                 (len {len}): {par} vs {serial}"
            );
        }
    }

    /// The range fold visits every index exactly once, in the same block grouping as the slice
    /// fold, across the same length edge cases
    #[test]
    fn det_par_fold_range_matches_serial_blocked_fold() {
        for len in [0usize, 1, 100, DET_REDUCE_BLOCK, DET_REDUCE_BLOCK * 3 + 17] {
            let data: Vec<f64> = (0..len).map(|i| (i as f64).cos()).collect();

            let par = det_par_fold_range(
                len,
                |range| range.map(|i| data[i] * data[i]).sum::<f64>(),
                |a, b| a + b,
                0.0,
            );
            let serial = data
                .chunks(DET_REDUCE_BLOCK)
                .map(|block| block.iter().map(|v| v * v).sum::<f64>())
                .fold(0.0, |a, b| a + b);

            assert!(
                par == serial,
                "blocked range fold must be bitwise identical to the serial blocked fold \
                 (len {len}): {par} vs {serial}"
            );
        }
    }

    /// Compound accumulators work through the range fold: per-bucket sums + counts merge in
    /// block order and match a fully serial scatter
    #[test]
    fn det_par_fold_range_compound_accumulator() {
        let n = DET_REDUCE_BLOCK * 2 + 333;
        let values: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let buckets: Vec<usize> = (0..n).map(|i| i % 4).collect();

        let (par_sums, par_counts) = det_par_fold_range(
            n,
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

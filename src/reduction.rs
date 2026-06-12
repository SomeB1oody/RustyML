//! Deterministic blocked parallel reduction over `f64` slices
//!
//! A bare rayon `par_iter().sum::<f64>()` (or `fold().reduce()`) groups its partial sums by
//! work-stealing, so the float accumulation order - and therefore the rounded result - varies
//! from run to run and with the thread count. This helper fixes the grouping instead: the slice
//! is cut into fixed-size blocks, each block is folded serially, the per-block results are
//! collected in block order (rayon's indexed `collect` preserves it), and the blocks are merged
//! sequentially. The grouping depends only on [`DET_REDUCE_BLOCK`], never on scheduling, so the
//! result is **bitwise identical at any thread count** (though not to the unblocked serial
//! fold - the gate below which callers stay serial therefore matters for reproducibility too).

use rayon::prelude::{ParallelIterator, ParallelSlice};

/// Fixed reduction block size (elements).
///
/// Calibrated on AMD Ryzen 9 9950X (16C/32T, 32 rayon threads), 2026-06-11; see
/// benches/RESULTS.md: a 4.2M-element sum-of-squares plateaus over 4K-64K-element blocks
/// (17.4-17.9x) with 16K measured fastest, degrading at 256K (too few blocks to balance).
/// Changing this value changes the (deterministic) result grouping, so it is part of the
/// reproducibility surface - bump it only deliberately
pub(crate) const DET_REDUCE_BLOCK: usize = 16_384;

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
pub(crate) fn det_par_fold<A, F, M>(slice: &[f64], fold_block: F, merge: M, identity: A) -> A
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
}

//! Index-mapping helpers that pick parallel or sequential execution based on a size threshold
//!
//! Provides [`map_collect`], the shared "parallelize only for large inputs" dispatch used
//! across the machine-learning models

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Maps `f` over `0..n`, choosing parallel or sequential execution by `threshold`
///
/// Centralizes the "parallelize only for large inputs" dispatch that every model
/// re-implements, so the rule lives in one place instead of being copy-pasted
///
/// # Parameters
///
/// - `n` - Number of items to process (the iterator runs over `0..n`)
/// - `threshold` - Minimum `n` at which parallel execution is used
/// - `f` - Mapping applied to each index
///
/// # Returns
///
/// - `Vec<R>` - The collected results in index order
#[inline]
pub(super) fn map_collect<R, F>(n: usize, threshold: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if n >= threshold {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// map_collect returns identical index-ordered output on both the sequential and parallel branches
    #[test]
    fn map_collect_seq_and_par_match_index_order() {
        let threshold = 4_usize;

        // n below threshold -> sequential branch
        let seq: Vec<usize> = map_collect(3, threshold, |i| i * i);
        assert_eq!(
            seq,
            vec![0, 1, 4],
            "sequential branch must be index-ordered"
        );

        // n at/above threshold -> parallel branch
        let par: Vec<usize> = map_collect(8, threshold, |i| i * i);
        assert_eq!(
            par,
            vec![0, 1, 4, 9, 16, 25, 36, 49],
            "parallel branch must preserve index order"
        );

        // Both branches must agree on the shared prefix
        assert_eq!(
            &par[..3],
            &seq[..],
            "seq and par must agree on shared indices"
        );
    }

    /// `map_collect` on an empty range returns an empty Vec
    #[test]
    fn map_collect_empty() {
        let out: Vec<usize> = map_collect(0, 4, |i| i * 2);
        assert!(out.is_empty(), "empty range must produce empty output");
    }
}

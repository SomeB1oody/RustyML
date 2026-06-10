use crate::error::Error;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Maps `f` over `0..n`, choosing parallel or sequential execution by `threshold`.
///
/// Centralizes the "parallelize only for large inputs" dispatch that every model
/// re-implements, so the rule lives in one place instead of being copy-pasted.
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

/// Fallible counterpart to [`map_collect`]: short-circuits on the first `Err`.
///
/// # Parameters
///
/// - `n` - Number of items to process (the iterator runs over `0..n`)
/// - `threshold` - Minimum `n` at which parallel execution is used
/// - `f` - Fallible mapping applied to each index
///
/// # Returns
///
/// - `Result<Vec<R>, Error>` - The collected results, or the first error encountered
///
/// # Errors
///
/// - Propagates any `Error` returned by `f`
#[inline]
pub(super) fn try_map_collect<R, F>(n: usize, threshold: usize, f: F) -> Result<Vec<R>, Error>
where
    R: Send,
    F: Fn(usize) -> Result<R, Error> + Sync + Send,
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

    /// `map_collect` must return identical, index-ordered output regardless of whether
    /// the sequential (n < threshold) or parallel (n >= threshold) branch runs.
    ///
    /// With threshold = 4: n=3 takes the sequential branch, n=8 the parallel branch.
    /// Mapping i -> i*i yields the squares in index order in both cases.
    #[test]
    fn map_collect_seq_and_par_match_index_order() {
        let threshold = 4_usize;

        // n below threshold -> sequential branch.
        let seq: Vec<usize> = map_collect(3, threshold, |i| i * i);
        assert_eq!(
            seq,
            vec![0, 1, 4],
            "sequential branch must be index-ordered"
        );

        // n at/above threshold -> parallel branch; results still in index order.
        let par: Vec<usize> = map_collect(8, threshold, |i| i * i);
        assert_eq!(
            par,
            vec![0, 1, 4, 9, 16, 25, 36, 49],
            "parallel branch must preserve index order"
        );

        // The common prefix produced by both branches must agree element-for-element.
        assert_eq!(
            &par[..3],
            &seq[..],
            "seq and par must agree on shared indices"
        );
    }

    /// `map_collect` on an empty range returns an empty Vec (sequential branch, n=0).
    #[test]
    fn map_collect_empty() {
        let out: Vec<usize> = map_collect(0, 4, |i| i * 2);
        assert!(out.is_empty(), "empty range must produce empty output");
    }

    /// `try_map_collect` short-circuits on the FIRST `Err` (in index order) when collecting
    /// into `Result`. Using the sequential branch (n < threshold) makes "first" deterministic.
    ///
    /// Indices 2 and 4 both fail; collect into `Result` must surface index 2's error message
    /// (the earlier failure), and the success branch returns all values in index order.
    #[test]
    fn try_map_collect_propagates_first_err() {
        let threshold = 100_usize; // keep n=5 on the sequential, deterministic-order branch

        let result: Result<Vec<usize>, Error> = try_map_collect(5, threshold, |i| {
            if i == 2 || i == 4 {
                Err(Error::invalid_input(format!("fail at {i}")))
            } else {
                Ok(i * 10)
            }
        });

        match result {
            Err(Error::InvalidInput(msg)) => {
                assert_eq!(
                    msg, "fail at 2",
                    "the first (lowest-index) error must propagate"
                );
            }
            other => panic!("expected Err(InvalidInput), got {other:?}"),
        }
    }

    /// `try_map_collect` returns `Ok` with all values in index order when every mapping
    /// succeeds.
    #[test]
    fn try_map_collect_all_ok_index_order() {
        let result: Result<Vec<usize>, Error> =
            try_map_collect(4, 100, |i| Ok::<usize, Error>(i + 1));
        assert_eq!(result.unwrap(), vec![1, 2, 3, 4]);
    }
}

use crate::error::ModelError;
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
/// - `Result<Vec<R>, ModelError>` - The collected results, or the first error encountered
///
/// # Errors
///
/// - Propagates any `ModelError` returned by `f`
#[inline]
pub(super) fn try_map_collect<R, F>(n: usize, threshold: usize, f: F) -> Result<Vec<R>, ModelError>
where
    R: Send,
    F: Fn(usize) -> Result<R, ModelError> + Sync + Send,
{
    if n >= threshold {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

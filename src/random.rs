//! Thread-local seeds for random model and utility operations.
//!
//! The helpers supply generators for model initialization, random training steps, and dataset splits.
//! An explicit local seed leaves the global stream unchanged.
//! Without a local seed, consumers draw a seed from the global stream or use system entropy.
//!
//! # Seed resolution
//!
//! `make_rng(Some(seed))` creates a generator from that seed.
//! `make_rng(None)` draws a sub-seed from the current thread's global stream when that stream exists.
//! Otherwise, it uses system entropy.
//! `make_rng_opt` follows the same rules but returns `None` instead of an entropy-seeded generator.
//! Its callers can keep deterministic behavior until a seed requests random choices.
//!
//! Reset the global seed before repeating the same sequence of randomized operations.
//! Consumers without local seeds share the stream in call order.
//! A different call order can assign different seeds to those consumers.
//! Consumers with local seeds do not advance the global stream.
//!
//! # Thread scope
//!
//! The global stream belongs to the thread that calls [`set_global_seed`].
//! Other threads retain their own state.
//! Set the seed on the thread where a component creates its generator.
//! Existing generators keep their state after the global seed changes or clears.
//! Clear the global seed after tests that share a thread to prevent state from reaching later tests.
//!
//! # Independent seeds
//!
//! The iterative PCA and KernelPCA eigensolvers use fixed starting seeds outside this module.
//! Randomized SVD uses the seed in its `SVDSolver::Randomized` option.
//! The global controls do not change those solver seeds.

use ndarray_rand::rand::{RngCore, SeedableRng, rng, rngs::StdRng};
use std::cell::RefCell;

thread_local! {
    /// Per-thread global seed stream. `None` until `set_global_seed` is called on this thread
    static GLOBAL_SEED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Sets the thread-local global seed
///
/// Consumers on this thread derive sub-seeds from this stream when they have no local seed.
/// Reset this seed before repeating the same sequence of generator requests.
///
/// # Parameters
///
/// - `seed` - The seed for the thread-local global RNG stream
pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = Some(StdRng::seed_from_u64(seed)));
}

/// Clears the thread-local global seed, restoring entropy-based behavior for unseeded components
///
/// Useful to isolate tests that may share a thread (for example under `--test-threads=1`)
pub fn clear_global_seed() {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = None);
}

/// Creates a generator only when a local or global seed exists.
///
/// # Parameters
///
/// - `random_state` - Local seed, or `None` to use the global seed stream
///
/// # Returns
///
/// - `Option<StdRng>` - Seeded generator, or `None` when neither seed source is set
pub(crate) fn make_rng_opt(random_state: Option<u64>) -> Option<StdRng> {
    match random_state {
        // Explicit local seed: independent, and does not touch the global stream
        Some(seed) => Some(StdRng::seed_from_u64(seed)),
        // No local seed: derive from the global stream if a global seed is set, else signal "no seed"
        None => GLOBAL_SEED_RNG.with(|cell| {
            cell.borrow_mut()
                .as_mut()
                .map(|global| StdRng::seed_from_u64(global.next_u64()))
        }),
    }
}

/// Resolves a `random_state` into a concrete RNG (see the [module docs](self) for the rules)
///
/// `Some` uses the given seed.
/// `None` consults the thread-local global, deriving a sub-seed from it if a global seed is set, or falls
/// back to entropy
///
/// # Parameters
///
/// - `random_state` - The per-consumer seed, or `None` to defer to the global/entropy
///
/// # Returns
///
/// - `StdRng` - A freshly seeded RNG for the caller to own and advance
pub(crate) fn make_rng(random_state: Option<u64>) -> StdRng {
    // Falls back to entropy when no local or global seed exists
    make_rng_opt(random_state).unwrap_or_else(|| StdRng::from_rng(&mut rng()))
}

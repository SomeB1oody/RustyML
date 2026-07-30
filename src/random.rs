//! Crate-wide control of pseudo-random number generation for reproducibility
//!
//! Most randomized components in the crate draw their RNG through `make_rng` (or its sibling
//! `make_rng_opt`, for callers that stay deterministic unless explicitly seeded). A single
//! [`set_global_seed`] call makes them reproducible together. This routes randomness through one
//! entry point for:
//!
//! - neural-network components: weight initialization, dropout/noise masks, and the
//!   [`Sequential`](crate::neural_network::sequential::Sequential) minibatch shuffle
//! - machine-learning estimators: k-means, SVC/LinearSVC, MeanShift, Isolation Forest, and others
//! - utilities: `train_test_split`, t-SNE
//!
//! # Seed resolution
//!
//! `make_rng` resolves a per-consumer `random_state: Option<u64>` against the process-global
//! (thread-local) seed as follows:
//!
//! - `Some(seed)`: use that seed. The global stays untouched
//! - `None`, with a global seed set: derive an independent sub-seed from the global stream
//! - `None`, with no global seed: seed from entropy (not reproducible)
//!
//! Because an explicit `Some` seed never consumes the global stream, adding or removing a seeded
//! component does not change the seeds the unseeded ones get. Unseeded components, by contrast,
//! draw from the shared stream in construction order, so their reproducibility is order-sensitive
//! (this matches Keras' global-seed behavior)
//!
//! # Threading
//!
//! The global seed is **thread-local**: [`set_global_seed`] only affects the thread that calls
//! it, so set the seed on the same thread that constructs your models. This is lock-free, and
//! because the default test harness spawns a fresh thread per test, each test starts unseeded.
//! Under `--test-threads=1`, however, all tests share one thread. A test that sets a global seed
//! should call [`clear_global_seed`] afterwards, ideally with a drop guard so it runs even on
//! panic. This avoids leaking the seed into a later test that expects unseeded behavior
//!
//! # Intentional exclusions
//!
//! Not every pseudo-random draw in the crate goes through this module. A draw is worth routing
//! here only when it has a real, lasting effect on the result. The `utils` dimensionality
//! reducers (`pca`, `kernel_pca`) are left out, for 2 reasons:
//!
//! - Their iterative eigensolvers (PCA's `PowerIteration`, and KernelPCA's `Lanczos` and
//!   `PowerIteration`) seed a starting vector with a fixed constant. These methods converge to the
//!   same eigenvectors regardless of the starting vector, so the seed only pins an arbitrary
//!   eigenvector sign. It has no effect on reproducibility worth routing through the global seed
//! - Randomized SVD (`SVDSolver::Randomized(u64)`) takes its seed as a public argument, so the
//!   caller always supplies it. There is no unseeded path for the global to fill
//!
//! General rule: route a draw through this module only when an unseeded call would make a
//! pseudo-random choice that changes the result

use ndarray_rand::rand::{RngCore, SeedableRng, rng, rngs::StdRng};
use std::cell::RefCell;

thread_local! {
    /// Per-thread global seed stream. `None` until `set_global_seed` is called on this thread
    static GLOBAL_SEED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Sets the thread-local global seed
///
/// After this call, every component constructed **on this thread** with `random_state == None`
/// becomes reproducible (it derives its RNG from the global stream). Call this before
/// constructing the models/estimators whose randomness you want to fix
///
/// # Parameters
///
/// - `seed` - The seed for the thread-local global RNG stream
pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = Some(StdRng::seed_from_u64(seed)));
}

/// Clears the thread-local global seed, restoring entropy-based behavior for unseeded components
///
/// Useful to isolate tests that may share a thread (e.g. under `--test-threads=1`)
pub fn clear_global_seed() {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = None);
}

/// Resolves a `random_state` into an RNG only when a seed is in effect. Returns `None` when
/// there is none (`random_state` is `None` and no global seed is set)
///
/// This is for callers that should stay deterministic unless randomness is explicitly requested,
/// e.g. a decision tree that breaks split ties randomly only when seeded. `Some(seed)` uses that
/// seed and ignores the global. `None` derives a sub-seed from the thread-local global if one is
/// set, or returns `None` otherwise (the signal: no randomization requested)
///
/// # Parameters
///
/// - `random_state` - The per-consumer seed, or `None` to defer to the global
///
/// # Returns
///
/// - `Option<StdRng>` - A seeded RNG if a local or global seed is active, else `None`
pub(crate) fn make_rng_opt(random_state: Option<u64>) -> Option<StdRng> {
    match random_state {
        // Explicit local seed: independent, and does not touch the global stream
        Some(seed) => Some(StdRng::seed_from_u64(seed)),
        // No local seed: derive from the global stream if one is set, else signal "no seed"
        None => GLOBAL_SEED_RNG.with(|cell| {
            cell.borrow_mut()
                .as_mut()
                .map(|global| StdRng::seed_from_u64(global.next_u64()))
        }),
    }
}

/// Resolves a `random_state` into a concrete RNG (see the [module docs](self) for the rules)
///
/// This is the single entry point for all randomness in the crate. `Some` uses the given seed.
/// `None` consults the thread-local global, deriving a sub-seed from it if one is set, or falls
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

//! Crate-wide control of pseudo-random number generation for reproducibility.
//!
//! Every randomized component in the crate draws its RNG through [`make_rng`], so a single
//! [`set_global_seed`] call can make the whole library reproducible. This routes the
//! neural-network components (weight initialization, dropout/noise masks, and the
//! [`Sequential`](crate::neural_network::sequential::Sequential) minibatch shuffle), the
//! machine-learning estimators (k-means, SVC/LinearSVC, MeanShift, Isolation Forest, …), and the
//! utilities (`train_test_split`, t-SNE) — all through this one entry point.
//!
//! # Seed resolution
//!
//! [`make_rng`] resolves a per-consumer `random_state: Option<u64>` against the process-global
//! (thread-local) seed as follows:
//!
//! - `Some(seed)` — use that seed; the global is **ignored and left untouched**.
//! - `None` + a global seed is set — derive an independent sub-seed from the global stream.
//! - `None` + no global seed — seed from entropy (non-reproducible).
//!
//! Because an explicit `Some` seed never consumes the global stream, adding or removing an
//! explicitly-seeded component does not perturb the seeds handed to the unseeded ones. Unseeded
//! components, by contrast, draw from the shared stream in construction order, so their
//! reproducibility is order-sensitive (this matches Keras' global-seed behavior).
//!
//! # Threading
//!
//! The global seed is **thread-local**: [`set_global_seed`] only affects the thread that calls
//! it, so set the seed on the same thread that constructs your models. This is lock-free, and
//! because the default test harness spawns a fresh thread per test, each test starts unseeded.
//! Under `--test-threads=1`, however, all tests share one thread, so a test that sets a global
//! seed should [`clear_global_seed`] afterwards (ideally panic-safely, e.g. via a drop guard)
//! to avoid leaking it into a later test that relies on unseeded behavior.

use ndarray_rand::rand::{RngCore, SeedableRng, rng, rngs::StdRng};
use std::cell::RefCell;

thread_local! {
    /// Per-thread global seed stream. `None` until `set_global_seed` is called on this thread.
    static GLOBAL_SEED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Sets the thread-local global seed.
///
/// After this call, every component constructed **on this thread** with `random_state == None`
/// becomes reproducible (it derives its RNG from the global stream). Call this before
/// constructing the models/estimators whose randomness you want to fix.
///
/// # Parameters
///
/// - `seed` - The seed for the thread-local global RNG stream
pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = Some(StdRng::seed_from_u64(seed)));
}

/// Clears the thread-local global seed, restoring entropy-based behavior for unseeded components.
///
/// Primarily useful to isolate tests that may share a thread (e.g. under `--test-threads=1`).
pub fn clear_global_seed() {
    GLOBAL_SEED_RNG.with(|cell| *cell.borrow_mut() = None);
}

/// Resolves a `random_state` into an RNG **only when a seed is in effect**, returning `None`
/// when there is none (`random_state` is `None` AND no global seed is set).
///
/// This is for callers that should stay deterministic unless randomness is explicitly requested —
/// e.g. a decision tree that breaks split ties randomly only when seeded. `Some(seed)` uses that
/// seed (ignoring the global); `None` derives a sub-seed from the thread-local global if one is set,
/// otherwise returns `None` (the signal: "no randomization requested").
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
        // Explicit local seed: independent, and does not touch the global stream.
        Some(seed) => Some(StdRng::seed_from_u64(seed)),
        // No local seed: derive from the global stream if one is set, else signal "no seed".
        None => GLOBAL_SEED_RNG.with(|cell| {
            cell.borrow_mut()
                .as_mut()
                .map(|global| StdRng::seed_from_u64(global.next_u64()))
        }),
    }
}

/// Resolves a `random_state` into a concrete RNG (see the [module docs](self) for the rules).
///
/// This is the single entry point for all randomness in the crate: `Some` uses the given seed,
/// `None` consults the thread-local global (deriving a sub-seed from it if set, else entropy).
///
/// # Parameters
///
/// - `random_state` - The per-consumer seed, or `None` to defer to the global/entropy
///
/// # Returns
///
/// - `StdRng` - A freshly seeded RNG for the caller to own and advance
pub(crate) fn make_rng(random_state: Option<u64>) -> StdRng {
    // Reuse the optional resolver; the only difference is the entropy fallback when no seed exists.
    make_rng_opt(random_state).unwrap_or_else(|| StdRng::from_rng(&mut rng()))
}

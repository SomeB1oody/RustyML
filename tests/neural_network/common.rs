//! Shared helpers for the `neural_network` integration tests

#![allow(dead_code)]

use ndarray::{ArrayBase, ArrayViewD, Data, Dimension};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use rustyml::neural_network::traits::Layer;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A deterministically seeded RNG, for reproducible tests
///
/// Always seed test RNGs. Never seed the thread RNG. This keeps failures reproducible.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// RAII guard for the crate-wide thread-local global seed
///
/// [`GlobalSeedGuard::set`] installs the global seed (see [`rustyml::set_global_seed`]). The
/// guard clears the seed on drop, even during a panic. This keeps a failing global-seed test
/// from leaking its seed into a sibling test on the same thread.
///
/// The default test harness runs each test on its own thread. Under `--test-threads=1`, every
/// test shares 1 thread. The panic-safe clear on drop is what keeps the tests isolated there.
///
/// Always bind the guard to a variable. An unbound guard drops immediately and clears the seed
/// before the test runs. This is why the type carries `#[must_use]`.
#[must_use = "bind the guard to a variable; an unbound guard clears the seed immediately"]
pub struct GlobalSeedGuard;

impl GlobalSeedGuard {
    /// Installs `seed` as the thread-local global seed. Clears it when the returned guard drops
    pub fn set(seed: u64) -> Self {
        rustyml::set_global_seed(seed);
        GlobalSeedGuard
    }
}

impl Drop for GlobalSeedGuard {
    fn drop(&mut self) {
        rustyml::clear_global_seed();
    }
}

// The process-global tuning gates

/// Reads 1 tuning gate
pub type GateGetter = fn() -> usize;

/// Writes 1 tuning gate
pub type GateSetter = fn(usize);

/// Every tuning gate that a neural-network layer reads, as a name, a getter, and a setter
///
/// The names and the functions come from `rustyml::tuning`, the public entry point of the
/// gates. The f64 gates and the tree gates are absent, because no layer reads them. The
/// matrix-product gates are absent for the same reason. They shape the caller-side tiling of
/// the classical-ML estimators, and a layer reaches the backend's own scheduling instead.
pub const NEURAL_NETWORK_GATES: &[(&str, GateGetter, GateSetter)] = &[
    (
        "elementwise.cheap_map_f32",
        rustyml::tuning::elementwise::get_cheap_map_f32,
        rustyml::tuning::elementwise::set_cheap_map_f32,
    ),
    (
        "elementwise.exp_map_f32",
        rustyml::tuning::elementwise::get_exp_map_f32,
        rustyml::tuning::elementwise::set_exp_map_f32,
    ),
    (
        "elementwise.spatial_dropout_scale",
        rustyml::tuning::elementwise::get_spatial_dropout_scale,
        rustyml::tuning::elementwise::set_spatial_dropout_scale,
    ),
    (
        "elementwise.fused_slice",
        rustyml::tuning::elementwise::get_fused_slice,
        rustyml::tuning::elementwise::set_fused_slice,
    ),
    (
        "reduction.sq_sum_f32",
        rustyml::tuning::reduction::get_sq_sum_f32,
        rustyml::tuning::reduction::set_sq_sum_f32,
    ),
    (
        "conv.parallel_min_flops",
        rustyml::tuning::conv::get_parallel_min_flops,
        rustyml::tuning::conv::set_parallel_min_flops,
    ),
    (
        "conv.naive_parallel_min_flops",
        rustyml::tuning::conv::get_naive_parallel_min_flops,
        rustyml::tuning::conv::set_naive_parallel_min_flops,
    ),
    (
        "pool.parallel_min_ops",
        rustyml::tuning::pool::get_parallel_min_ops,
        rustyml::tuning::pool::set_parallel_min_ops,
    ),
    (
        "upsampling.parallel_min_ops",
        rustyml::tuning::upsampling::get_parallel_min_ops,
        rustyml::tuning::upsampling::set_parallel_min_ops,
    ),
    (
        "norm.batch_norm",
        rustyml::tuning::norm::get_batch_norm,
        rustyml::tuning::norm::set_batch_norm,
    ),
    (
        "norm.col_fold",
        rustyml::tuning::norm::get_col_fold,
        rustyml::tuning::norm::set_col_fold,
    ),
    (
        "norm.row_pass",
        rustyml::tuning::norm::get_row_pass,
        rustyml::tuning::norm::set_row_pass,
    ),
];

/// Every task-size cap that a neural-network layer reads, as a name, a getter, and a setter
///
/// A cap is process-global, exactly like a tuning gate, so [`GateGuard`] saves and restores it
/// under the same lock. The table comes from `rustyml::bench_internals`, which owns the names,
/// documents what each cap counts, and lists the drivers that must never take one.
///
/// A gate picks the parallel branch of a kernel. A cap decides how many tasks that branch then
/// builds. The 2 are separate: a small test tensor clears no calibrated task-size rule, so the
/// parallel branch of such a kernel runs with exactly 1 task until a cap splits it. Neither a
/// gate nor a cap changes a result.
pub const NEURAL_NETWORK_SPLIT_CAPS: &[(&str, GateGetter, GateSetter)] =
    rustyml::bench_internals::SPLIT_CAPS;

/// The 1 lock over the process-global tuning gates
///
/// A gate is a process-global atomic, and the default test harness runs the tests of 1 binary
/// at the same time. A test that moves a gate therefore races every test that reads one. The
/// lock removes the race without `--test-threads=1`, which would make the result depend on how
/// the suite is invoked.
static GATE_LOCK: RwLock<()> = RwLock::new(());

/// Guard for a test that depends on the gates as the crate configured them
///
/// [`read_gates`] builds it. Bind it for as long as the test reads a gate or runs a kernel
/// whose path a gate selects. It holds the shared side of the lock, so any number of such tests
/// run together, and none of them runs while a [`GateGuard`] holds the exclusive side.
#[must_use = "bind the guard to a variable; an unbound guard releases the lock immediately"]
pub struct GateReadGuard(RwLockReadGuard<'static, ()>);

/// Takes the shared side of the tuning-gate lock. See [`GateReadGuard`]
///
/// Never call this while a [`GateGuard`] of the same test is alive. The lock takes no upgrade,
/// so the test would stop for ever.
pub fn read_gates() -> GateReadGuard {
    // A test that panics under the lock poisons it. The gate values are still correct, because
    // every writer restores them on its unwind path, so the poison carries no information here
    GateReadGuard(GATE_LOCK.read().unwrap_or_else(PoisonError::into_inner))
}

/// RAII guard for a test that moves a tuning gate or a task-size cap
///
/// [`GateGuard::acquire`] takes the exclusive side of the lock and saves every gate of
/// [`NEURAL_NETWORK_GATES`] and every cap of [`NEURAL_NETWORK_SPLIT_CAPS`]. Move any gate
/// through the `rustyml::tuning` setters while the guard lives. [`GateGuard::set_all`] puts 1
/// value in every gate for you, and [`GateGuard::with_split_cap`] puts 1 value in every cap.
///
/// The guard restores every saved value when it drops, which includes the path where the test
/// panics. This is the same shape as [`GlobalSeedGuard`], which guards the global seed.
///
/// Always bind the guard to a variable. An unbound guard drops at once, and restores the gates
/// before the test runs. This is why the type carries `#[must_use]`.
#[must_use = "bind the guard to a variable; an unbound guard restores the gates immediately"]
pub struct GateGuard {
    /// Holds the exclusive lock for as long as the gates stay moved
    _lock: RwLockWriteGuard<'static, ()>,
    /// The value of every gate of [`NEURAL_NETWORK_GATES`], in that order, before the guard
    saved: Vec<usize>,
    /// The value of every cap of [`NEURAL_NETWORK_SPLIT_CAPS`], in that order, before the guard
    saved_caps: Vec<usize>,
}

impl GateGuard {
    /// Takes the exclusive side of the lock, and saves every gate and every cap
    pub fn acquire() -> Self {
        let lock = GATE_LOCK.write().unwrap_or_else(PoisonError::into_inner);
        let saved: Vec<usize> = NEURAL_NETWORK_GATES
            .iter()
            .map(|(_, get, _)| get())
            .collect();
        let saved_caps: Vec<usize> = NEURAL_NETWORK_SPLIT_CAPS
            .iter()
            .map(|(_, get, _)| get())
            .collect();
        Self {
            _lock: lock,
            saved,
            saved_caps,
        }
    }

    /// Takes the exclusive side of the lock, and then puts `value` in every gate
    ///
    /// A `value` of 0 sends every gated kernel down its parallel branch, because each gate
    /// compares its work estimate with `>=`. A `value` of `usize::MAX` holds every kernel on
    /// its serial branch.
    ///
    /// This moves no cap. A parallel branch that a gate opens still builds 1 task for a small
    /// input. Add [`GateGuard::with_split_cap`] to split it.
    pub fn set_all(value: usize) -> Self {
        let guard = Self::acquire();
        for (_, _, set) in NEURAL_NETWORK_GATES {
            set(value);
        }
        guard
    }

    /// Puts `value` in every task-size cap of [`NEURAL_NETWORK_SPLIT_CAPS`], and gives the guard
    /// back
    ///
    /// A `value` of 1 or more holds each task of a capped driver at that many units or fewer,
    /// so a small input builds more than 1 task. The unit is the driver's own: output positions
    /// for the convolution and the windowed pooling forward passes, channels for the pooling
    /// backward pass, and destination rows for the resize and the embedding gather. A `value`
    /// of 0 is the production value, and it restores the calibrated task size.
    pub fn with_split_cap(self, value: usize) -> Self {
        for (_, _, set) in NEURAL_NETWORK_SPLIT_CAPS {
            set(value);
        }
        self
    }

    /// Puts 0 back in the 1 cap that `name` selects, and gives the guard back
    ///
    /// Use this after [`GateGuard::with_split_cap`] for a driver that is not invariant to its
    /// task size. The value 0 is the production value, so that driver keeps its calibrated task
    /// size while every other driver stays capped. `rustyml::bench_internals` names the 1 driver
    /// that needs the exception today, and a caller that reaches that driver must take it.
    ///
    /// # Panics
    ///
    /// - If no cap of [`NEURAL_NETWORK_SPLIT_CAPS`] carries the name. A renamed cap therefore
    ///   fails loudly instead of losing the exception without a word
    pub fn without_split_cap(self, name: &str) -> Self {
        let found = NEURAL_NETWORK_SPLIT_CAPS
            .iter()
            .find(|(cap, _, _)| *cap == name)
            .unwrap_or_else(|| panic!("no task-size cap is named {name:?}"));
        (found.2)(0);
        self
    }
}

impl Drop for GateGuard {
    fn drop(&mut self) {
        for ((_, _, set), &value) in NEURAL_NETWORK_GATES.iter().zip(self.saved.iter()) {
            set(value);
        }
        for ((_, _, set), &value) in NEURAL_NETWORK_SPLIT_CAPS.iter().zip(self.saved_caps.iter()) {
            set(value);
        }
    }
}

/// Asserts 2 arrays or tensors are element-wise equal within `eps` (absolute difference)
///
/// For single scalars, use the `assert_abs_diff_eq!` or `assert_relative_eq!` macro from
/// `approx` directly.
pub fn assert_allclose<A, S1, S2, D>(actual: &ArrayBase<S1, D>, expected: &ArrayBase<S2, D>, eps: A)
where
    A: approx::AbsDiffEq<Epsilon = A> + Copy + std::fmt::Debug,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: actual {:?} vs expected {:?}",
        actual.shape(),
        expected.shape()
    );
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            a.abs_diff_eq(e, eps),
            "element mismatch: actual {a:?} vs expected {e:?} (eps {eps:?})"
        );
    }
}

/// 1 named array of a layer, or a panic that names what the layer holds instead
///
/// The checkpoint format addresses every array by name, so a test that reads a weight reads it
/// by the same name. A wrong name is a test defect, and the panic lists the names the layer
/// gives, so the report says what to write instead.
pub fn named<'a>(layer: &'a dyn Layer, name: &str) -> ArrayViewD<'a, f32> {
    layer.weight(name).unwrap_or_else(|| {
        let held: Vec<&str> = layer.weights().iter().map(|entry| entry.name).collect();
        panic!(
            "the layer `{}` holds no weight named `{name}`; it holds {held:?}",
            layer.layer_type()
        )
    })
}

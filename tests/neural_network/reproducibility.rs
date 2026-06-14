//! Integration tests for the reproducibility / RNG API
//!
//! Exercises the public seeding surface: per-layer seeding via `with_random_state` on
//! layers, the thread-local global seed (`set_global_seed` / `clear_global_seed`),
//! and the `Sequential` fit-time shuffle seed (`new_with_seed` / `set_seed`)
//!
//! Equal seeds yield byte-identical output, so the "same" assertions use zero epsilon; the
//! "different" assertions require the max absolute difference to clear a small threshold

use ndarray::Array2;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::Activation;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;

use super::common::{GlobalSeedGuard, assert_allclose};

// helpers

/// Build a 2-D Tensor from row-major data
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// A fixed 4-feature input row reused across the layer-level tests
fn fixed_input_4() -> Tensor {
    t2(1, 4, vec![0.5, -1.0, 2.0, 0.25])
}

/// Maximum element-wise absolute difference between two equally-shaped tensors
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch in max_abs_diff");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build a tiny `Dense(4 -> 3, ReLU)`, applying `seed` via `with_random_state` when `Some`
fn dense_4_3(seed: Option<u64>) -> Dense {
    let dense = Dense::new(4, 3, Activation::ReLU).expect("Dense::new(4,3) must succeed");
    match seed {
        Some(s) => dense.with_random_state(s),
        None => dense,
    }
}

// 1. same_seed_same_init

/// Two `Dense` layers with the same explicit seed produce byte-identical `predict()`
/// output on the same input
#[test]
fn same_seed_same_init() {
    let a = dense_4_3(Some(7));
    let b = dense_4_3(Some(7));

    let x = fixed_input_4();
    let pa = a.predict(&x).unwrap();
    let pb = b.predict(&x).unwrap();

    // Identical seed => identical weights => identical predict output, zero epsilon
    assert_allclose(&pa, &pb, 0.0_f32);
}

// 2. different_seed_differs

/// `Some(1)` vs `Some(2)` initialize different weights and produce a different `predict()`
/// output (max absolute difference clears a small threshold)
#[test]
fn different_seed_differs() {
    let a = dense_4_3(Some(1));
    let b = dense_4_3(Some(2));

    let x = fixed_input_4();
    let pa = a.predict(&x).unwrap();
    let pb = b.predict(&x).unwrap();

    let diff = max_abs_diff(&pa, &pb);
    assert!(
        diff > 1e-4_f32,
        "expected different seeds to give different predict output, max abs diff = {diff}"
    );
}

// 3. global_seed_reproducible

/// With the same global seed set, an unseeded model (`random_state == None`) rebuilds to the
/// same weights and produces identical `predict()` output
#[test]
fn global_seed_reproducible() {
    let x = fixed_input_4();

    let _seed = GlobalSeedGuard::set(123);
    let first = dense_4_3(None); // unseeded: draws from the global stream
    let p_first = first.predict(&x).unwrap();

    // Reset the global to the same value and rebuild the identical unseeded model
    rustyml::set_global_seed(123);
    let second = dense_4_3(None);
    let p_second = second.predict(&x).unwrap();

    // `_seed` clears the global on drop, even if this assertion panics
    assert_allclose(&p_first, &p_second, 0.0_f32);
}

// 4. local_overrides_global

/// An explicit local seed ignores the global seed: a `Some(5)` layer built with a global set
/// equals one built with no global, so the local seed alone determines the weights
#[test]
fn local_overrides_global() {
    let x = fixed_input_4();

    // Built with a global seed active; the guard clears it at the end of this block, even on panic
    let p_with_global = {
        let _seed = GlobalSeedGuard::set(999);
        dense_4_3(Some(5)).predict(&x).unwrap()
    };

    // Built with no global seed active
    let p_without_global = dense_4_3(Some(5)).predict(&x).unwrap();

    // Local Some(5) ignores the global => identical weights => identical output, zero epsilon
    assert_allclose(&p_with_global, &p_without_global, 0.0_f32);
}

// 5. training_reproducible

/// Two `Sequential` models with identical architecture, layer seeds, and shuffle seed reach
/// an identical `predict()` output after batched training (batch_size 2 exercises the shuffle)
#[test]
fn training_reproducible() {
    // Tiny 4-sample dataset for a 4 -> 3 -> 1 regression model
    #[rustfmt::skip]
    let x = t2(4, 4, vec![
        0.5, -1.0, 2.0, 0.25,
        1.0,  0.0, -0.5, 1.5,
        -2.0, 0.5, 1.0, -1.0,
        0.25, 2.0, -1.5, 0.0,
    ]);
    let y = t2(4, 1, vec![1.0, 0.0, -1.0, 0.5]);

    // Build two identical, fully-seeded models and train them identically
    let build_and_train = || -> Sequential {
        let mut model = Sequential::new_with_seed(42);
        model
            .add(dense_4_3(Some(7)))
            .add(
                Dense::new(3, 1, Activation::Linear)
                    .unwrap()
                    .with_random_state(11),
            )
            .compile(
                SGD::new(0.05, 0.0, false, 0.0).unwrap(),
                MeanSquaredError::new(),
            );
        // batch_size < n_samples => the seeded per-epoch shuffle is actually used
        model.fit_with_batches(&x, &y, 5, 2).unwrap();
        model
    };

    let model_a = build_and_train();
    let model_b = build_and_train();

    let x_test = t2(1, 4, vec![0.5, -1.0, 2.0, 0.25]);
    let pa = model_a.predict(&x_test).unwrap();
    let pb = model_b.predict(&x_test).unwrap();

    // Reproducible init + reproducible shuffle => identical trained weights => identical output
    assert_allclose(&pa, &pb, 0.0_f32);
}

// 6. global_seed_advances_between_unseeded_draws

/// Under a single global seed, two consecutively-built unseeded layers get different
/// initializations, since each draw advances the global stream (`Linear` readout)
#[test]
fn global_seed_advances_between_unseeded_draws() {
    let x = fixed_input_4();

    let (p_first, p_second) = {
        let _seed = GlobalSeedGuard::set(2024);
        let first = Dense::new(4, 3, Activation::Linear).unwrap(); // sub-seed #1 from the global
        let second = Dense::new(4, 3, Activation::Linear).unwrap(); // sub-seed #2, must differ
        (first.predict(&x).unwrap(), second.predict(&x).unwrap())
        // `_seed` clears the global here, before the assertion below
    };

    let diff = max_abs_diff(&p_first, &p_second);
    assert!(
        diff > 1e-4_f32,
        "two unseeded layers under one global seed must differ (global stream must advance), max abs diff = {diff}"
    );
}

// 7. cleared_global_seed_unseeded_layers_differ (entropy fallback)

/// Inverse of `global_seed_reproducible`: with no global seed installed, two unseeded `Dense`
/// layers fall back to OS entropy and produce different `predict()` output
#[test]
fn cleared_global_seed_unseeded_layers_differ() {
    let x = fixed_input_4();

    // Set a global, build one unseeded layer from it (matches the reproducibility setup)
    let guard = GlobalSeedGuard::set(777);
    let _seeded_from_global = Dense::new(4, 3, Activation::Linear).unwrap();
    // Clear the global so subsequent unseeded layers fall back to entropy. Dropping the guard
    // already calls clear_global_seed; the explicit call below is an idempotent reaffirmation
    drop(guard);
    rustyml::clear_global_seed(); // explicit: no global seed is installed

    let a = Dense::new(4, 3, Activation::Linear).unwrap();
    let b = Dense::new(4, 3, Activation::Linear).unwrap();
    let pa = a.predict(&x).unwrap();
    let pb = b.predict(&x).unwrap();

    let diff = max_abs_diff(&pa, &pb);
    assert!(
        diff > 1e-4_f32,
        "with no global seed, two unseeded layers must differ (entropy fallback), max abs diff = {diff}"
    );
}

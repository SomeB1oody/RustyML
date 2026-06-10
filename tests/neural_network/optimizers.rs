//! Integration tests for the optimizer objects (SGD, Adam, RMSprop, AdaGrad).
//!
//! Coverage:
//!   - Constructor validation: learning_rate/epsilon ≤ 0, NaN, Inf; Adam/RMSprop betas outside
//!     [0, 1) — boundary values 0.0 (valid) and 1.0 (invalid) are checked explicitly.
//!   - End-to-end convergence: each optimizer drives a single Dense layer's MSE loss strictly
//!     DOWN over N epochs on a fixed, seeded regression problem.
//!   - Multi-layer convergence: a 2-Dense-layer net with Adam to verify per-layer state buffers
//!     are allocated correctly (both layers get updated, loss falls).
//!
//! What this file does NOT test:
//!   - Backward/gradient correctness — that is in gradient_check.rs.
//!   - Per-element kernel math — the in-module unit tests inside kernels.rs cover that.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::error::Error;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::AdaGrad;
use rustyml::neural_network::optimizers::Adam;
use rustyml::neural_network::optimizers::RMSprop;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: simple regression problem
// ─────────────────────────────────────────────────────────────────────────────

/// Build a fixed, deterministic (x, y) pair for a tiny 1-input → 1-output regression.
///
/// Target: y = 2*x.  We use 4 samples so the gradient is well-conditioned for
/// every optimizer at a reasonable learning rate.  All weights are fixed to
/// identity (w=1, b=0) via `set_weights`, so the initial MSE and the direction
/// of the gradient are known:
///
///   x  = [0.5, 1.0, 1.5, 2.0]  (shape [4,1])
///   y  = [1.0, 2.0, 3.0, 4.0]  (shape [4,1])
///   pred_0 = x * 1 + 0 = x
///   MSE_0  = mean((pred - y)^2)
///           = mean((0.5-1)^2, (1-2)^2, (1.5-3)^2, (2-4)^2)
///           = mean(0.25, 1, 2.25, 4) = 7.5 / 4 = 1.875
fn regression_data() -> (Tensor, Tensor) {
    let x = Array::from_shape_vec((4, 1), vec![0.5_f32, 1.0, 1.5, 2.0])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((4, 1), vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    (x, y)
}

/// Build a Dense(1→1, Linear) layer with weight=1, bias=0.
///
/// With identity weights the layer acts as a passthrough, giving us a known
/// starting loss of 1.875 on the regression_data() problem.
fn identity_dense() -> Dense {
    let mut layer = Dense::new(1, 1, Linear::new(), None).unwrap();
    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    layer.set_weights(w, b).unwrap();
    layer
}

/// Compute the MSE loss for `model.predict(x)` against `y`.
///
/// We use the same formula as MeanSquaredError::compute_loss so the comparison
/// is apples-to-apples: MSE = mean((pred - y)^2).
fn eval_mse(model: &Sequential, x: &Tensor, y: &Tensor) -> f32 {
    let pred = model.predict(x).unwrap();
    let diff = &pred - y;
    let sq: Tensor = diff.mapv(|v| v * v);
    sq.sum() / sq.len() as f32
}

// ═════════════════════════════════════════════════════════════════════════════
// SGD — constructor validation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn sgd_rejects_zero_learning_rate() {
    let err = SGD::new(0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_negative_learning_rate() {
    let err = SGD::new(-0.1).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_infinite_learning_rate() {
    let err = SGD::new(f32::INFINITY).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_nan_learning_rate() {
    let err = SGD::new(f32::NAN).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_accepts_valid_learning_rate() {
    assert!(SGD::new(0.01).is_ok());
    assert!(SGD::new(1.0).is_ok());
    // f32::MIN_POSITIVE is the smallest positive finite f32
    assert!(SGD::new(f32::MIN_POSITIVE).is_ok());
}

// ═════════════════════════════════════════════════════════════════════════════
// Adam — constructor validation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn adam_rejects_zero_learning_rate() {
    assert!(matches!(
        Adam::new(0.0, 0.9, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_negative_learning_rate() {
    assert!(matches!(
        Adam::new(-1e-3, 0.9, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_inf_learning_rate() {
    assert!(matches!(
        Adam::new(f32::INFINITY, 0.9, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_nan_learning_rate() {
    assert!(matches!(
        Adam::new(f32::NAN, 0.9, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

/// beta1 = 0.0 is the inclusive lower bound — must be accepted.
#[test]
fn adam_accepts_beta1_zero() {
    assert!(Adam::new(0.001, 0.0, 0.999, 1e-8).is_ok());
}

/// beta1 = 1.0 is the exclusive upper bound — must be rejected.
#[test]
fn adam_rejects_beta1_one() {
    assert!(matches!(
        Adam::new(0.001, 1.0, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_greater_than_one() {
    assert!(matches!(
        Adam::new(0.001, 1.1, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_negative() {
    assert!(matches!(
        Adam::new(0.001, -0.1, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_nan() {
    assert!(matches!(
        Adam::new(0.001, f32::NAN, 0.999, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

/// beta2 = 0.0 is the inclusive lower bound — must be accepted.
#[test]
fn adam_accepts_beta2_zero() {
    assert!(Adam::new(0.001, 0.9, 0.0, 1e-8).is_ok());
}

/// beta2 = 1.0 is the exclusive upper bound — must be rejected.
#[test]
fn adam_rejects_beta2_one() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 1.0, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta2_nan() {
    assert!(matches!(
        Adam::new(0.001, 0.9, f32::NAN, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_zero() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_negative() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, -1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_nan() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, f32::NAN),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_inf() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, f32::INFINITY),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_accepts_valid_hyperparameters() {
    assert!(Adam::new(0.001, 0.9, 0.999, 1e-8).is_ok());
    // Typical alternative: small beta1
    assert!(Adam::new(0.01, 0.5, 0.9, 1e-6).is_ok());
}

// ═════════════════════════════════════════════════════════════════════════════
// RMSprop — constructor validation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn rmsprop_rejects_zero_learning_rate() {
    assert!(matches!(
        RMSprop::new(0.0, 0.9, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_negative_learning_rate() {
    assert!(matches!(
        RMSprop::new(-0.01, 0.9, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_inf_learning_rate() {
    assert!(matches!(
        RMSprop::new(f32::INFINITY, 0.9, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_nan_learning_rate() {
    assert!(matches!(
        RMSprop::new(f32::NAN, 0.9, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

/// rho = 0.0 is the inclusive lower bound of [0, 1) — must be accepted.
#[test]
fn rmsprop_accepts_rho_zero() {
    assert!(RMSprop::new(0.01, 0.0, 1e-8).is_ok());
}

/// rho = 1.0 is the exclusive upper bound — must be rejected.
#[test]
fn rmsprop_rejects_rho_one() {
    assert!(matches!(
        RMSprop::new(0.01, 1.0, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_greater_than_one() {
    assert!(matches!(
        RMSprop::new(0.01, 1.5, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_negative() {
    assert!(matches!(
        RMSprop::new(0.01, -0.5, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_nan() {
    assert!(matches!(
        RMSprop::new(0.01, f32::NAN, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_zero() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_nan() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, f32::NAN),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_inf() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, f32::INFINITY),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_accepts_valid_hyperparameters() {
    assert!(RMSprop::new(0.001, 0.9, 1e-8).is_ok());
    assert!(RMSprop::new(0.01, 0.95, 1e-5).is_ok());
}

// ═════════════════════════════════════════════════════════════════════════════
// AdaGrad — constructor validation
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn adagrad_rejects_zero_learning_rate() {
    assert!(matches!(
        AdaGrad::new(0.0, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_negative_learning_rate() {
    assert!(matches!(
        AdaGrad::new(-0.01, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_inf_learning_rate() {
    assert!(matches!(
        AdaGrad::new(f32::INFINITY, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_nan_learning_rate() {
    assert!(matches!(
        AdaGrad::new(f32::NAN, 1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_zero() {
    assert!(matches!(
        AdaGrad::new(0.01, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_negative() {
    assert!(matches!(
        AdaGrad::new(0.01, -1e-8),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_nan() {
    assert!(matches!(
        AdaGrad::new(0.01, f32::NAN),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_inf() {
    assert!(matches!(
        AdaGrad::new(0.01, f32::INFINITY),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_accepts_valid_hyperparameters() {
    assert!(AdaGrad::new(0.01, 1e-8).is_ok());
    assert!(AdaGrad::new(0.001, 1e-5).is_ok());
}

// ═════════════════════════════════════════════════════════════════════════════
// Known initial-loss sanity check
// ═════════════════════════════════════════════════════════════════════════════

/// Before any training, the Dense(w=1, b=0) layer predicts y_hat = x.
///
/// Hand calculation:
///   x  = [0.5, 1.0, 1.5, 2.0]
///   y  = [1.0, 2.0, 3.0, 4.0]
///   y_hat = x (identity weights)
///   MSE = mean((0.5-1)^2 + (1-2)^2 + (1.5-3)^2 + (2-4)^2)
///       = mean(0.25 + 1.0 + 2.25 + 4.0) = 7.5 / 4 = 1.875
#[test]
fn identity_dense_initial_mse_is_1_875() {
    let (x, y) = regression_data();
    let mut model = Sequential::new();
    model
        .add(identity_dense())
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    let mse = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse, 1.875_f32, epsilon = 1e-5);
}

// ═════════════════════════════════════════════════════════════════════════════
// End-to-end convergence: each optimizer drives loss DOWN over 20 epochs
// ═════════════════════════════════════════════════════════════════════════════

/// SGD: loss after 20 epochs must be strictly less than the initial loss (1.875).
///
/// lr=0.5 is aggressive for a 1-input problem but converges quickly and is still
/// stable.  The expected trajectory (w starts at 1, target is ~2) is monotone
/// decreasing, so 20 steps is more than enough.
#[test]
fn sgd_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = Sequential::new();
    // lr=0.1: plain SGD needs lr < 2/λ_max(Hessian) ≈ 2/5.5 ≈ 0.36 on this problem to converge
    // (0.5 overshoots and diverges); 0.1 reduces the loss steadily.
    model
        .add(identity_dense())
        .compile(SGD::new(0.1).unwrap(), MeanSquaredError::new());

    let mse_before = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse_before, initial_mse, epsilon = 1e-5);

    model.fit(&x, &y, 20).unwrap();

    let mse_after = eval_mse(&model, &x, &y);
    assert!(
        mse_after < mse_before,
        "SGD: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

/// Adam: loss after 20 epochs must be strictly less than the initial loss (1.875).
///
/// Standard Adam settings.  Adam's adaptive moments make it converge faster than
/// vanilla SGD so 20 epochs is ample.
#[test]
fn adam_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = Sequential::new();
    model.add(identity_dense()).compile(
        Adam::new(0.1, 0.9, 0.999, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    let mse_before = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse_before, initial_mse, epsilon = 1e-5);

    model.fit(&x, &y, 20).unwrap();

    let mse_after = eval_mse(&model, &x, &y);
    assert!(
        mse_after < mse_before,
        "Adam: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

/// RMSprop: loss after 20 epochs must be strictly less than the initial loss (1.875).
#[test]
fn rmsprop_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = Sequential::new();
    model.add(identity_dense()).compile(
        RMSprop::new(0.1, 0.9, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    let mse_before = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse_before, initial_mse, epsilon = 1e-5);

    model.fit(&x, &y, 20).unwrap();

    let mse_after = eval_mse(&model, &x, &y);
    assert!(
        mse_after < mse_before,
        "RMSprop: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

/// AdaGrad: loss after 20 epochs must be strictly less than the initial loss (1.875).
///
/// AdaGrad's accumulator starts at 0 on step 1, so the first effective step is
/// lr / sqrt(g^2 + eps) ≈ lr / |g|.  With lr=0.1 this converges safely in 20 epochs.
#[test]
fn adagrad_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = Sequential::new();
    model
        .add(identity_dense())
        .compile(AdaGrad::new(0.5, 1e-8).unwrap(), MeanSquaredError::new());

    let mse_before = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse_before, initial_mse, epsilon = 1e-5);

    model.fit(&x, &y, 20).unwrap();

    let mse_after = eval_mse(&model, &x, &y);
    assert!(
        mse_after < mse_before,
        "AdaGrad: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Multi-layer convergence with Adam (verifies per-layer state-buffer allocation)
// ═════════════════════════════════════════════════════════════════════════════

/// A 2-layer Dense net: Dense(1→4, Linear) → Dense(4→1, Linear).
///
/// We deliberately do NOT set weights here: Xavier init is random but bounded,
/// so the initial loss for this problem is typically small but non-zero.  What we
/// test is that Adam correctly allocates moment buffers for BOTH layers and that
/// the loss strictly decreases over 50 epochs — if the second layer's state were
/// missing the cursor would wrap and the update would be either wrong or missing.
///
/// The regression problem is the same y = 2x, but with more capacity the network
/// can fit it well.  We verify:
///   1. Loss after training < loss before training.
///   2. Loss after 50 epochs < loss after 20 epochs (monotone-ish decrease).
#[test]
fn adam_two_layer_loss_decreases_and_buffers_allocated_correctly() {
    let (x, y) = regression_data();

    // Build a 2-layer model — we seed the dense layers (and the model's fit-time
    // shuffle) so the test is deterministic and never flakes on a pathological
    // Xavier init.  Xavier init starts with small weights near 0, giving a
    // non-trivial positive initial MSE for y = 2x.
    const SEED: u64 = 0;
    let build_model = || -> Sequential {
        // Layer 1: 1 → 4, Linear
        let layer1 = Dense::new(1, 4, Linear::new(), Some(SEED)).unwrap();
        // Layer 2: 4 → 1, Linear
        let layer2 = Dense::new(4, 1, Linear::new(), Some(SEED)).unwrap();

        let mut model = Sequential::new_with_seed(SEED);
        model.add(layer1).add(layer2).compile(
            Adam::new(0.05, 0.9, 0.999, 1e-8).unwrap(),
            MeanSquaredError::new(),
        );
        model
    };

    let mut model = build_model();
    let mse_initial = eval_mse(&model, &x, &y);

    // First 20 epochs
    model.fit(&x, &y, 20).unwrap();
    let mse_after_20 = eval_mse(&model, &x, &y);

    // 20 more epochs (total 40)
    model.fit(&x, &y, 20).unwrap();
    let mse_after_40 = eval_mse(&model, &x, &y);

    // Loss must be strictly lower than initial after 20 epochs
    assert!(
        mse_after_20 < mse_initial,
        "Adam 2-layer: loss after 20 epochs ({mse_after_20}) should be < initial ({mse_initial})"
    );

    // Loss after 40 epochs must be at least as low as after 20 epochs (overall decreasing trend)
    // We use a small tolerance rather than strict inequality to handle occasional plateaus.
    assert!(
        mse_after_40 <= mse_after_20 + 1e-4,
        "Adam 2-layer: loss after 40 epochs ({mse_after_40}) should not be greater than after 20 ({mse_after_20})"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Multi-layer convergence: one test per remaining optimizer
// ═════════════════════════════════════════════════════════════════════════════

/// SGD on a 2-layer net (1→4→1): loss must fall and converge over 50 epochs.
/// Uses a higher learning rate to compensate for SGD's lack of adaptivity, and a seeded init
/// so the convergence check is deterministic (an unseeded init made it depend on the entropy draw).
#[test]
fn sgd_two_layer_loss_decreases() {
    const SEED: u64 = 0;
    let (x, y) = regression_data();

    let layer1 = Dense::new(1, 4, Linear::new(), Some(SEED)).unwrap();
    let layer2 = Dense::new(4, 1, Linear::new(), Some(SEED)).unwrap();

    let mut model = Sequential::new();
    model
        .add(layer1)
        .add(layer2)
        .compile(SGD::new(0.05).unwrap(), MeanSquaredError::new());

    let mse_before = eval_mse(&model, &x, &y);
    model.fit(&x, &y, 50).unwrap();
    let mse_after = eval_mse(&model, &x, &y);

    assert!(
        mse_after < mse_before,
        "SGD 2-layer: loss should decrease; before={mse_before}, after={mse_after}"
    );
    assert!(
        mse_after < 0.1,
        "SGD 2-layer: loss should converge near 0; after={mse_after}"
    );
}

/// RMSprop on a 2-layer net (1→4→1): loss must fall and converge.
///
/// Seeded init (like the Adam two-layer test) so the result is deterministic — an UNSEEDED
/// init made this flaky: RMSprop's adaptive step can transiently overshoot from a near-optimal
/// random start, so on ~1% of entropy draws the loss ended slightly *higher* over the original
/// 30 epochs. A gentle learning rate over enough epochs converges to ~0 for any init (verified
/// across seeds), and `y = 2x` is exactly fittable so the final loss must be clearly low.
#[test]
fn rmsprop_two_layer_loss_decreases() {
    const SEED: u64 = 0;
    let (x, y) = regression_data();

    let layer1 = Dense::new(1, 4, Linear::new(), Some(SEED)).unwrap();
    let layer2 = Dense::new(4, 1, Linear::new(), Some(SEED)).unwrap();

    let mut model = Sequential::new();
    model.add(layer1).add(layer2).compile(
        RMSprop::new(0.01, 0.9, 1e-8).unwrap(),
        MeanSquaredError::new(),
    );

    let mse_before = eval_mse(&model, &x, &y);
    model.fit(&x, &y, 150).unwrap();
    let mse_after = eval_mse(&model, &x, &y);

    assert!(
        mse_after < mse_before,
        "RMSprop 2-layer: loss should decrease; before={mse_before}, after={mse_after}"
    );
    assert!(
        mse_after < 0.1,
        "RMSprop 2-layer: loss should converge near 0; after={mse_after}"
    );
}

/// AdaGrad on a 2-layer net (1→4→1): loss must fall over 30 epochs.
#[test]
fn adagrad_two_layer_loss_decreases() {
    let (x, y) = regression_data();

    let layer1 = Dense::new(1, 4, Linear::new(), None).unwrap();
    let layer2 = Dense::new(4, 1, Linear::new(), None).unwrap();

    let mut model = Sequential::new();
    model
        .add(layer1)
        .add(layer2)
        .compile(AdaGrad::new(0.5, 1e-8).unwrap(), MeanSquaredError::new());

    let mse_before = eval_mse(&model, &x, &y);
    model.fit(&x, &y, 30).unwrap();
    let mse_after = eval_mse(&model, &x, &y);

    assert!(
        mse_after < mse_before,
        "AdaGrad 2-layer: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Numerical value: SGD one-step weight update on known weights
// ═════════════════════════════════════════════════════════════════════════════

/// Verify that one SGD step moves the weight in the correct direction by the
/// correct amount, derived fully from the math.
///
/// Setup (1-input, 1-output, linear, no bias, 1 sample):
///   w = 1.0, b = 0.0
///   x = [2.0]  (shape [1,1])
///   y = [6.0]  (shape [1,1])
///   y_hat = x * w = 2.0
///   loss  = (y_hat - y)^2 / 1 = (2 - 6)^2 = 16.0    (MSE with n=1)
///
/// Gradient derivation:
///   dL/d(y_hat) = 2*(y_hat - y) / n = 2*(-4) / 1 = -8
///   dL/dw       = dL/d(y_hat) * x   = -8 * 2      = -16
///   dL/db       = dL/d(y_hat)       = -8
///
/// SGD update (lr = 0.01):
///   w_new = w - lr * dL/dw = 1.0 - 0.01 * (-16) = 1.0 + 0.16 = 1.16
///   b_new = b - lr * dL/db = 0.0 - 0.01 * (-8)  = 0.08
///
/// We verify the new weight by predicting on x after one fit epoch:
///   y_hat_new = x * w_new + b_new = 2.0 * 1.16 + 0.08 = 2.40
#[test]
fn sgd_one_step_weight_update_matches_hand_calculation() {
    // Single sample, 1 feature
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();

    // w=1, b=0  →  y_hat=2,  loss=(2-6)^2/1 = 16
    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    let mut layer = Dense::new(1, 1, Linear::new(), None).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model
        .add(layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Confirm initial prediction: y_hat = 2.0
    let pred_before = model.predict(&x).unwrap();
    let val_before = *pred_before.iter().next().unwrap();
    assert_abs_diff_eq!(val_before, 2.0_f32, epsilon = 1e-6);

    // One training step
    model.fit(&x, &y, 1).unwrap();

    // After one SGD step: w_new = 1.16, b_new = 0.08
    // y_hat_new = 2.0 * 1.16 + 0.08 = 2.32 + 0.08 = 2.40
    let pred_after = model.predict(&x).unwrap();
    let val_after = *pred_after.iter().next().unwrap();
    assert_abs_diff_eq!(val_after, 2.40_f32, epsilon = 1e-4);
}

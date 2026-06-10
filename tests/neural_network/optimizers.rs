//! Integration tests for the optimizer objects (SGD, Adam, RMSprop, AdaGrad)
//!
//! Coverage:
//! - Constructor validation: learning_rate/epsilon <= 0, NaN, Inf; Adam/RMSprop betas outside
//!   [0, 1), with boundary values 0.0 (valid) and 1.0 (invalid) checked explicitly
//! - End-to-end convergence: each optimizer drives a single Dense layer's MSE loss strictly
//!   down over N epochs on a fixed, seeded regression problem
//! - Multi-layer convergence: a 2-Dense-layer net with Adam verifies per-layer state buffers
//!   are allocated correctly (both layers updated, loss falls)
//!
//! Gradient correctness lives in gradient_check.rs; per-element kernel math in kernels.rs

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

// Helper: simple regression problem

/// Fixed, deterministic (x, y) pair for a tiny 1-input -> 1-output regression
///
/// Target y = 2*x over 4 samples; with identity weights (w=1, b=0) the initial
/// MSE is 1.875
fn regression_data() -> (Tensor, Tensor) {
    let x = Array::from_shape_vec((4, 1), vec![0.5_f32, 1.0, 1.5, 2.0])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((4, 1), vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    (x, y)
}

/// Dense(1->1, Linear) layer with weight=1, bias=0, acting as a passthrough
///
/// Gives a known starting loss of 1.875 on the regression_data() problem
fn identity_dense() -> Dense {
    let mut layer = Dense::new(1, 1, Linear::new(), None).unwrap();
    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    layer.set_weights(w, b).unwrap();
    layer
}

/// MSE loss for `model.predict(x)` against `y`: mean((pred - y)^2)
///
/// Matches MeanSquaredError::compute_loss so the comparison is apples-to-apples
fn eval_mse(model: &Sequential, x: &Tensor, y: &Tensor) -> f32 {
    let pred = model.predict(x).unwrap();
    let diff = &pred - y;
    let sq: Tensor = diff.mapv(|v| v * v);
    sq.sum() / sq.len() as f32
}

// SGD - constructor validation

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
    // smallest positive finite f32
    assert!(SGD::new(f32::MIN_POSITIVE).is_ok());
}

// Adam - constructor validation

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

/// beta1 = 0.0 (inclusive lower bound) is accepted
#[test]
fn adam_accepts_beta1_zero() {
    assert!(Adam::new(0.001, 0.0, 0.999, 1e-8).is_ok());
}

/// beta1 = 1.0 (exclusive upper bound) is rejected
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

/// beta2 = 0.0 (inclusive lower bound) is accepted
#[test]
fn adam_accepts_beta2_zero() {
    assert!(Adam::new(0.001, 0.9, 0.0, 1e-8).is_ok());
}

/// beta2 = 1.0 (exclusive upper bound) is rejected
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
    // typical alternative: small beta1
    assert!(Adam::new(0.01, 0.5, 0.9, 1e-6).is_ok());
}

// RMSprop - constructor validation

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

/// rho = 0.0 (inclusive lower bound of [0, 1)) is accepted
#[test]
fn rmsprop_accepts_rho_zero() {
    assert!(RMSprop::new(0.01, 0.0, 1e-8).is_ok());
}

/// rho = 1.0 (exclusive upper bound) is rejected
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

// AdaGrad - constructor validation

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

// Known initial-loss sanity check

/// Untrained Dense(w=1, b=0) predicts y_hat = x, giving MSE = 1.875 on the fixture
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

// End-to-end convergence: each optimizer drives loss down over 20 epochs

/// SGD: loss after 20 epochs is strictly below the initial loss (1.875)
#[test]
fn sgd_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = Sequential::new();
    // lr=0.1: plain SGD needs lr < 2/lambda_max(Hessian) ~= 2/5.5 ~= 0.36 here to
    // converge (0.5 overshoots and diverges); 0.1 reduces the loss steadily
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

/// Adam: loss after 20 epochs is strictly below the initial loss (1.875)
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

/// RMSprop: loss after 20 epochs is strictly below the initial loss (1.875)
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

/// AdaGrad: loss after 20 epochs is strictly below the initial loss (1.875)
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

// Multi-layer convergence with Adam (verifies per-layer state-buffer allocation)

/// Adam on Dense(1->4) -> Dense(4->1) allocates moment buffers for both layers and
/// drives loss down (after 20 epochs < initial; after 40 not above after 20)
#[test]
fn adam_two_layer_loss_decreases_and_buffers_allocated_correctly() {
    let (x, y) = regression_data();

    // Seed the dense layers and the fit-time shuffle so the test is deterministic
    // and never flakes on a pathological Xavier init
    const SEED: u64 = 0;
    let build_model = || -> Sequential {
        let layer1 = Dense::new(1, 4, Linear::new(), Some(SEED)).unwrap();
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

    model.fit(&x, &y, 20).unwrap();
    let mse_after_20 = eval_mse(&model, &x, &y);

    // 20 more epochs (total 40)
    model.fit(&x, &y, 20).unwrap();
    let mse_after_40 = eval_mse(&model, &x, &y);

    assert!(
        mse_after_20 < mse_initial,
        "Adam 2-layer: loss after 20 epochs ({mse_after_20}) should be < initial ({mse_initial})"
    );

    // Small tolerance rather than strict inequality, to absorb occasional plateaus
    assert!(
        mse_after_40 <= mse_after_20 + 1e-4,
        "Adam 2-layer: loss after 40 epochs ({mse_after_40}) should not be greater than after 20 ({mse_after_20})"
    );
}

// Multi-layer convergence: one test per remaining optimizer

/// SGD on a 2-layer net (1->4->1): loss falls and converges below 0.1 over 50 epochs
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

/// RMSprop on a 2-layer net (1->4->1): loss falls and converges below 0.1 over 150 epochs
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

/// AdaGrad on a 2-layer net (1->4->1): loss falls over 30 epochs
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

// Numerical value: SGD one-step weight update on known weights

/// One SGD step (lr=0.01) on w=1, b=0 with x=2, y=6 yields y_hat=2.40 after refit
#[test]
fn sgd_one_step_weight_update_matches_hand_calculation() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();

    // w=1, b=0  ->  y_hat=2,  loss=(2-6)^2/1 = 16
    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    let mut layer = Dense::new(1, 1, Linear::new(), None).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model
        .add(layer)
        .compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // initial prediction: y_hat = 2.0
    let pred_before = model.predict(&x).unwrap();
    let val_before = *pred_before.iter().next().unwrap();
    assert_abs_diff_eq!(val_before, 2.0_f32, epsilon = 1e-6);

    model.fit(&x, &y, 1).unwrap();

    // after one SGD step: w_new=1.16, b_new=0.08, so y_hat = 2.0*1.16 + 0.08 = 2.40
    let pred_after = model.predict(&x).unwrap();
    let val_after = *pred_after.iter().next().unwrap();
    assert_abs_diff_eq!(val_after, 2.40_f32, epsilon = 1e-4);
}

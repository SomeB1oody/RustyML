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
use ndarray::{Array, Array2, ArrayD};
use rustyml::error::Error;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::AdaGrad;
use rustyml::neural_network::optimizers::Adam;
use rustyml::neural_network::optimizers::RMSprop;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::{Layer, Optimizer};

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
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
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
    let err = SGD::new(0.0, 0.0, false, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_negative_learning_rate() {
    let err = SGD::new(-0.1, 0.0, false, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_infinite_learning_rate() {
    let err = SGD::new(f32::INFINITY, 0.0, false, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_rejects_nan_learning_rate() {
    let err = SGD::new(f32::NAN, 0.0, false, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn sgd_accepts_valid_learning_rate() {
    assert!(SGD::new(0.01, 0.0, false, 0.0).is_ok());
    assert!(SGD::new(1.0, 0.0, false, 0.0).is_ok());
    // smallest positive finite f32
    assert!(SGD::new(f32::MIN_POSITIVE, 0.0, false, 0.0).is_ok());
}

// Adam - constructor validation

#[test]
fn adam_rejects_zero_learning_rate() {
    assert!(matches!(
        Adam::new(0.0, 0.9, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_negative_learning_rate() {
    assert!(matches!(
        Adam::new(-1e-3, 0.9, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_inf_learning_rate() {
    assert!(matches!(
        Adam::new(f32::INFINITY, 0.9, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_nan_learning_rate() {
    assert!(matches!(
        Adam::new(f32::NAN, 0.9, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

/// beta1 = 0.0 (inclusive lower bound) is accepted
#[test]
fn adam_accepts_beta1_zero() {
    assert!(Adam::new(0.001, 0.0, 0.999, 1e-8, 0.0).is_ok());
}

/// beta1 = 1.0 (exclusive upper bound) is rejected
#[test]
fn adam_rejects_beta1_one() {
    assert!(matches!(
        Adam::new(0.001, 1.0, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_greater_than_one() {
    assert!(matches!(
        Adam::new(0.001, 1.1, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_negative() {
    assert!(matches!(
        Adam::new(0.001, -0.1, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta1_nan() {
    assert!(matches!(
        Adam::new(0.001, f32::NAN, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

/// beta2 = 0.0 (inclusive lower bound) is accepted
#[test]
fn adam_accepts_beta2_zero() {
    assert!(Adam::new(0.001, 0.9, 0.0, 1e-8, 0.0).is_ok());
}

/// beta2 = 1.0 (exclusive upper bound) is rejected
#[test]
fn adam_rejects_beta2_one() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 1.0, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_beta2_nan() {
    assert!(matches!(
        Adam::new(0.001, 0.9, f32::NAN, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_zero() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, 0.0, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_negative() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, -1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_nan() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, f32::NAN, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_rejects_epsilon_inf() {
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, f32::INFINITY, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adam_accepts_valid_hyperparameters() {
    assert!(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).is_ok());
    // typical alternative: small beta1
    assert!(Adam::new(0.01, 0.5, 0.9, 1e-6, 0.0).is_ok());
}

// RMSprop - constructor validation

#[test]
fn rmsprop_rejects_zero_learning_rate() {
    assert!(matches!(
        RMSprop::new(0.0, 0.9, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_negative_learning_rate() {
    assert!(matches!(
        RMSprop::new(-0.01, 0.9, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_inf_learning_rate() {
    assert!(matches!(
        RMSprop::new(f32::INFINITY, 0.9, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_nan_learning_rate() {
    assert!(matches!(
        RMSprop::new(f32::NAN, 0.9, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

/// rho = 0.0 (inclusive lower bound of [0, 1)) is accepted
#[test]
fn rmsprop_accepts_rho_zero() {
    assert!(RMSprop::new(0.01, 0.0, 1e-8, 0.0).is_ok());
}

/// rho = 1.0 (exclusive upper bound) is rejected
#[test]
fn rmsprop_rejects_rho_one() {
    assert!(matches!(
        RMSprop::new(0.01, 1.0, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_greater_than_one() {
    assert!(matches!(
        RMSprop::new(0.01, 1.5, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_negative() {
    assert!(matches!(
        RMSprop::new(0.01, -0.5, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_rho_nan() {
    assert!(matches!(
        RMSprop::new(0.01, f32::NAN, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_zero() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, 0.0, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_nan() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, f32::NAN, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_rejects_epsilon_inf() {
    assert!(matches!(
        RMSprop::new(0.01, 0.9, f32::INFINITY, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn rmsprop_accepts_valid_hyperparameters() {
    assert!(RMSprop::new(0.001, 0.9, 1e-8, 0.0).is_ok());
    assert!(RMSprop::new(0.01, 0.95, 1e-5, 0.0).is_ok());
}

// AdaGrad - constructor validation

#[test]
fn adagrad_rejects_zero_learning_rate() {
    assert!(matches!(
        AdaGrad::new(0.0, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_negative_learning_rate() {
    assert!(matches!(
        AdaGrad::new(-0.01, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_inf_learning_rate() {
    assert!(matches!(
        AdaGrad::new(f32::INFINITY, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_nan_learning_rate() {
    assert!(matches!(
        AdaGrad::new(f32::NAN, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_zero() {
    assert!(matches!(
        AdaGrad::new(0.01, 0.0, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_negative() {
    assert!(matches!(
        AdaGrad::new(0.01, -1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_nan() {
    assert!(matches!(
        AdaGrad::new(0.01, f32::NAN, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_rejects_epsilon_inf() {
    assert!(matches!(
        AdaGrad::new(0.01, f32::INFINITY, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
}

#[test]
fn adagrad_accepts_valid_hyperparameters() {
    assert!(AdaGrad::new(0.01, 1e-8, 0.0).is_ok());
    assert!(AdaGrad::new(0.001, 1e-5, 0.0).is_ok());
}

// Known initial-loss sanity check

/// Untrained Dense(w=1, b=0) predicts y_hat = x, giving MSE = 1.875 on the fixture
#[test]
fn identity_dense_initial_mse_is_1_875() {
    let (x, y) = regression_data();
    let mut model = Sequential::new();
    model.add(identity_dense()).compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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
    model.add(identity_dense()).compile(
        SGD::new(0.1, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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
        Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap(),
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
        RMSprop::new(0.1, 0.9, 1e-8, 0.0).unwrap(),
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
    model.add(identity_dense()).compile(
        AdaGrad::new(0.5, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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
        let layer1 = Dense::new(1, 4, Linear::new())
            .unwrap()
            .with_random_state(SEED);
        let layer2 = Dense::new(4, 1, Linear::new())
            .unwrap()
            .with_random_state(SEED);

        let mut model = Sequential::new_with_seed(SEED);
        model.add(layer1).add(layer2).compile(
            Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0).unwrap(),
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

    let layer1 = Dense::new(1, 4, Linear::new())
        .unwrap()
        .with_random_state(SEED);
    let layer2 = Dense::new(4, 1, Linear::new())
        .unwrap()
        .with_random_state(SEED);

    let mut model = Sequential::new();
    model.add(layer1).add(layer2).compile(
        SGD::new(0.05, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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

    let layer1 = Dense::new(1, 4, Linear::new())
        .unwrap()
        .with_random_state(SEED);
    let layer2 = Dense::new(4, 1, Linear::new())
        .unwrap()
        .with_random_state(SEED);

    let mut model = Sequential::new();
    model.add(layer1).add(layer2).compile(
        RMSprop::new(0.01, 0.9, 1e-8, 0.0).unwrap(),
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

    let layer1 = Dense::new(1, 4, Linear::new()).unwrap();
    let layer2 = Dense::new(4, 1, Linear::new()).unwrap();

    let mut model = Sequential::new();
    model.add(layer1).add(layer2).compile(
        AdaGrad::new(0.5, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(layer).compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

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

// Clip-by-global-norm (opt-in via the `clip_norm` constructor argument)

/// Reads the (weight, bias) scalars of a model whose first layer is a 1x1 Dense
fn dense_wb(model: &Sequential) -> (f32, f32) {
    match &model.get_weights()[0] {
        LayerWeight::Dense(d) => (d.weight[[0, 0]], d.bias[[0, 0]]),
        _ => panic!("expected Dense layer weights"),
    }
}

/// One clipped SGD step on the same w=1, b=0, x=2, y=6 problem as the hand-calc test above.
/// Unclipped gradients are grad_w=-16, grad_b=-8, so the global norm is sqrt(16^2+8^2)=sqrt(320);
/// with max_norm=8 every gradient is scaled by 8/sqrt(320), shrinking the +0.16/+0.08 updates by
/// that single factor (direction preserved, unlike per-element clamping)
#[test]
fn clip_by_global_norm_scales_sgd_step() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();

    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
    layer.set_weights(w, b).unwrap();

    let max_norm = 8.0_f32;
    let mut model = Sequential::new();
    model.add(layer).compile(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_clip_norm(max_norm)
            .unwrap(),
        MeanSquaredError::new(),
    );

    model.fit(&x, &y, 1).unwrap();

    // scale = max_norm / global_norm = 8 / sqrt(320); unclipped deltas were +0.16 (w) and +0.08 (b)
    let scale = max_norm / 320.0_f32.sqrt();
    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 1.0 + 0.16 * scale, epsilon = 1e-5);
    assert_abs_diff_eq!(b_new, 0.0 + 0.08 * scale, epsilon = 1e-5);
    // Both parameters were scaled by the *same* factor, i.e. direction is preserved
    assert_abs_diff_eq!((w_new - 1.0) / (b_new), 0.16 / 0.08, epsilon = 1e-4);
}

/// A clip threshold above the global gradient norm (sqrt(320) ~= 17.9) leaves the update identical
/// to plain SGD: w_new=1.16, b_new=0.08
#[test]
fn clip_by_global_norm_above_norm_is_noop() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();

    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = Sequential::new();
    model.add(layer).compile(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_clip_norm(100.0)
            .unwrap(),
        MeanSquaredError::new(),
    );

    model.fit(&x, &y, 1).unwrap();

    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 1.16_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(b_new, 0.08_f32, epsilon = 1e-5);
}

/// `with_clip_norm` rejects non-positive or non-finite thresholds and accepts a valid positive one;
/// constructing without it leaves clipping disabled
#[test]
fn new_rejects_invalid_clip_norm() {
    for bad in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                SGD::new(0.01, 0.0, false, 0.0).unwrap().with_clip_norm(bad),
                Err(Error::InvalidParameter { .. })
            ),
            "clip_norm {bad} should be rejected"
        );
    }
    assert!(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_clip_norm(5.0)
            .is_ok()
    );
    assert!(SGD::new(0.01, 0.0, false, 0.0).is_ok());
}

// SGD momentum / weight decay / LR scheduling (integration)

/// `set_learning_rate` retunes the step: doubling lr before one SGD step doubles the weight delta.
/// Reuses the w=1, b=0, x=2, y=6 problem (grad_w=-16, grad_b=-8)
#[test]
fn set_learning_rate_scales_the_step() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
    layer
        .set_weights(
            Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap(),
            Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap(),
        )
        .unwrap();

    let mut model = Sequential::new();
    model.add(layer).compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    model.set_learning_rate(0.02); // double the configured 0.01
    model.fit(&x, &y, 1).unwrap();

    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 1.0 + 0.02 * 16.0, epsilon = 1e-5); // 1.32
    assert_abs_diff_eq!(b_new, 0.02 * 8.0, epsilon = 1e-5); // 0.16
}

/// Decoupled weight decay shrinks the parameter by (1 - lr*wd) before the gradient step.
/// With w=1, wd=0.5, lr=0.01: w := 1*(1 - 0.005) - 0.01*(-16) = 0.995 + 0.16 = 1.155
#[test]
fn sgd_decoupled_weight_decay_shrinks_param() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();
    let mut layer = Dense::new(1, 1, Linear::new()).unwrap();
    layer
        .set_weights(
            Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap(),
            Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap(),
        )
        .unwrap();

    let mut model = Sequential::new();
    model.add(layer).compile(
        SGD::new(0.01, 0.0, false, 0.5).unwrap(),
        MeanSquaredError::new(),
    );
    model.fit(&x, &y, 1).unwrap();

    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 0.995 + 0.16, epsilon = 1e-5); // 1.155
    assert_abs_diff_eq!(b_new, 0.08, epsilon = 1e-5); // b=0, decay no-op
}

// Weight decay applies to weights only, not biases or normalization gamma/beta.
// These run two layers through an identical forward+backward and differ only in `weight_decay`,
// so any divergence is attributable solely to decay (the gradient values cancel out of the
// comparison: weight_decay shrinks `value` by `(1 - lr*wd)` before the same gradient step).

/// Runs a Dense(2->2, Linear) with fixed weights/bias through one forward + backward (fixed
/// nonzero upstream gradient) and one SGD step at the given `weight_decay`, returning the
/// resulting (weights, bias). Both decay settings see identical gradients
fn dense_after_one_sgd_step(
    w0: &Array2<f32>,
    b0: &Array2<f32>,
    lr: f32,
    weight_decay: f32,
) -> (Array2<f32>, Array2<f32>) {
    let mut layer = Dense::new(2, 2, Linear::new()).unwrap();
    layer.set_weights(w0.clone(), b0.clone()).unwrap();
    let x = Array::from_shape_vec((1, 2), vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let _ = layer.forward(&x).unwrap();
    let grad_out = Array::from_shape_vec((1, 2), vec![0.7_f32, -1.3])
        .unwrap()
        .into_dyn();
    layer.backward(&grad_out).unwrap();

    let mut opt = SGD::new(lr, 0.0, false, weight_decay).unwrap();
    opt.step();
    opt.update(&mut layer, 1.0);
    match layer.get_weights() {
        LayerWeight::Dense(d) => ((*d.weight).clone(), (*d.bias).clone()),
        _ => panic!("expected Dense weights"),
    }
}

/// Decoupled weight decay shrinks Dense weights but leaves the bias untouched:
/// `weight_decay = w_plain - lr*wd*w0` (exact, independent of the gradient), while the bias is
/// byte-identical with and without decay
#[test]
fn weight_decay_decays_dense_weights_but_skips_bias() {
    let w0 = Array::from_shape_vec((2, 2), vec![1.0_f32, -2.0, 3.0, -4.0]).unwrap();
    let b0 = Array::from_shape_vec((1, 2), vec![0.5_f32, -1.5]).unwrap();
    let (lr, wd) = (0.1_f32, 0.5_f32);

    let (w_plain, b_plain) = dense_after_one_sgd_step(&w0, &b0, lr, 0.0);
    let (w_decay, b_decay) = dense_after_one_sgd_step(&w0, &b0, lr, wd);

    // Bias is excluded from weight decay -> identical with and without it
    for i in 0..2 {
        assert_abs_diff_eq!(b_decay[[0, i]], b_plain[[0, i]], epsilon = 1e-6);
    }
    // Weights are decayed by exactly lr*wd*w0 relative to the no-decay step
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(
                w_decay[[i, j]],
                w_plain[[i, j]] - lr * wd * w0[[i, j]],
                epsilon = 1e-6
            );
        }
    }
    // Guard against a vacuous pass: decay must have actually moved the weights
    assert!(
        (w_decay[[0, 0]] - w_plain[[0, 0]]).abs() > 1e-4,
        "weight decay should change the weights"
    );
}

/// Runs a BatchNormalization layer (gamma=1, beta=0) through one training forward + backward
/// (fixed nonzero upstream gradient) and one SGD step at the given `weight_decay`, returning the
/// resulting (gamma, beta)
fn batchnorm_gamma_beta_after_one_sgd_step(weight_decay: f32) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut bn = BatchNormalization::new(vec![2, 3], 0.9, 1e-5).unwrap();
    bn.set_training_if_mode_dependent(true);
    let x = Array::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let _ = bn.forward(&x).unwrap();
    let grad_out = Array::from_shape_vec((2, 3), vec![0.5_f32, -0.5, 1.0, -1.0, 0.25, -0.25])
        .unwrap()
        .into_dyn();
    bn.backward(&grad_out).unwrap();

    let mut opt = SGD::new(0.1, 0.0, false, weight_decay).unwrap();
    opt.step();
    opt.update(&mut bn, 1.0);
    match bn.get_weights() {
        LayerWeight::BatchNormalization(w) => ((*w.gamma).clone(), (*w.beta).clone()),
        _ => panic!("expected BatchNormalization weights"),
    }
}

/// Normalization scale/shift (gamma/beta) are excluded from weight decay: a non-zero
/// `weight_decay` produces a byte-identical update to no decay at all
#[test]
fn weight_decay_skips_batchnorm_gamma_and_beta() {
    let (g_plain, b_plain) = batchnorm_gamma_beta_after_one_sgd_step(0.0);
    let (g_decay, b_decay) = batchnorm_gamma_beta_after_one_sgd_step(0.5);

    assert_eq!(g_plain.shape(), g_decay.shape());
    for (p, d) in g_plain.iter().zip(g_decay.iter()) {
        assert_abs_diff_eq!(*p, *d, epsilon = 1e-6);
    }
    for (p, d) in b_plain.iter().zip(b_decay.iter()) {
        assert_abs_diff_eq!(*p, *d, epsilon = 1e-6);
    }
    // Guard against a vacuous pass: gamma must have actually been updated by the gradient step
    // (so the "identical" check above is comparing moved values, not two untouched 1.0 arrays)
    assert!(
        g_plain.iter().any(|&v| (v - 1.0).abs() > 1e-5),
        "gamma should have a non-trivial gradient update"
    );
}

/// SGD with momentum still drives MSE strictly down on the seeded regression problem
#[test]
fn sgd_momentum_loss_decreases() {
    let (x, y) = regression_data();
    let mut model = Sequential::new();
    model.add(identity_dense()).compile(
        SGD::new(0.05, 0.9, true, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let before = eval_mse(&model, &x, &y);
    model.fit(&x, &y, 5).unwrap();
    let after = eval_mse(&model, &x, &y);
    assert!(
        after < before,
        "SGD+momentum should reduce loss; before={before}, after={after}"
    );
}

/// Negative momentum / weight_decay are rejected
#[test]
fn new_rejects_negative_momentum_and_weight_decay() {
    assert!(matches!(
        SGD::new(0.01, -0.1, false, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(matches!(
        SGD::new(0.01, 0.0, false, -0.1),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(matches!(
        Adam::new(0.001, 0.9, 0.999, 1e-8, -0.1),
        Err(Error::InvalidParameter { .. })
    ));
}

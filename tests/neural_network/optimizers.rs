//! Integration tests for the optimizer objects: SGD, Adam, AdamW, RMSprop, and AdaGrad.
//!
//! Coverage:
//! - Constructor validation: learning_rate or epsilon at or below 0, NaN, or Inf. Adam and
//!   RMSprop betas outside [0, 1), with boundary values 0.0 (valid) and 1.0 (invalid) checked
//!   explicitly.
//! - End-to-end convergence: each optimizer drives a single Dense layer's MSE loss strictly
//!   down over a fixed number of epochs on a fixed, seeded regression problem.
//! - Multi-layer convergence: a 2-Dense-layer net with Adam verifies per-layer state buffers
//!   are allocated correctly (both layers updated, loss falls).
//!
//! Gradient correctness lives in `gradient_check.rs`.

use crate::common::named;
use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, ArrayD};
use rustyml::error::Error;
use rustyml::neural_network::layers::ParamCounts;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::AdaGrad;
use rustyml::neural_network::optimizers::Adam;
use rustyml::neural_network::optimizers::AdamW;
use rustyml::neural_network::optimizers::RMSprop;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::{
    LayerBase, Optimizer, ParamRef, UnaryLayer, WeightMut, WeightRef,
};
use rustyml::neural_network::{Ctx, Shape, Tensor};

// Helper: simple regression problem

/// Fixed, deterministic (x, y) pair for a tiny 1-input -> 1-output regression
///
/// Target y = 2*x over 4 samples. With identity weights (w=1, b=0), the initial MSE is 1.875
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
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 1])).unwrap();
    let w = Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
    let b = Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap();
    layer.set_weights(w, b).unwrap();
    layer
}

/// MSE loss for `model.predict(x)` against `y`: mean((pred - y)^2)
///
/// Matches MeanSquaredError::compute_loss so the comparison is like-for-like
fn eval_mse(model: &Sequential, x: &Tensor, y: &Tensor) -> f32 {
    let pred = model.predict(x).unwrap();
    let diff = &pred - y;
    let sq: Tensor = diff.mapv(|v| v * v);
    sq.sum() / sq.len() as f32
}

// SGD: constructor validation

#[test]
fn sgd_rejects_invalid_learning_rate() {
    for lr in [0.0_f32, -0.1, f32::INFINITY, f32::NAN] {
        assert!(
            matches!(
                SGD::new(lr, 0.0, false, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for learning_rate={lr:?}"
        );
    }
}

#[test]
fn sgd_accepts_valid_learning_rate() {
    assert!(SGD::new(0.01, 0.0, false, 0.0).is_ok());
    assert!(SGD::new(1.0, 0.0, false, 0.0).is_ok());
    // smallest positive finite f32
    assert!(SGD::new(f32::MIN_POSITIVE, 0.0, false, 0.0).is_ok());
}

// Adam: constructor validation

#[test]
fn adam_rejects_invalid_learning_rate() {
    for lr in [0.0_f32, -1e-3, f32::INFINITY, f32::NAN] {
        assert!(
            matches!(
                Adam::new(lr, 0.9, 0.999, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for learning_rate={lr:?}"
        );
    }
}

/// beta1 must lie in [0, 1): 0.0 (inclusive lower) is accepted. 1.0 (exclusive upper),
/// out-of-range values, and NaN are rejected
#[test]
fn adam_beta1_bounds() {
    assert!(
        Adam::new(0.001, 0.0, 0.999, 1e-8, 0.0).is_ok(),
        "beta1=0.0 (inclusive lower bound) is accepted"
    );
    for beta1 in [1.0_f32, 1.1, -0.1, f32::NAN] {
        assert!(
            matches!(
                Adam::new(0.001, beta1, 0.999, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for beta1={beta1:?}"
        );
    }
}

/// beta2 must lie in [0, 1): 0.0 (inclusive lower) is accepted. 1.0 (exclusive upper)
/// and NaN are rejected
#[test]
fn adam_beta2_bounds() {
    assert!(
        Adam::new(0.001, 0.9, 0.0, 1e-8, 0.0).is_ok(),
        "beta2=0.0 (inclusive lower bound) is accepted"
    );
    for beta2 in [1.0_f32, f32::NAN] {
        assert!(
            matches!(
                Adam::new(0.001, 0.9, beta2, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for beta2={beta2:?}"
        );
    }
}

#[test]
fn adam_rejects_invalid_epsilon() {
    for eps in [0.0_f32, -1e-8, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                Adam::new(0.001, 0.9, 0.999, eps, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for epsilon={eps:?}"
        );
    }
}

#[test]
fn adam_accepts_valid_hyperparameters() {
    assert!(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).is_ok());
    // typical alternative: small beta1
    assert!(Adam::new(0.01, 0.5, 0.9, 1e-6, 0.0).is_ok());
}

// RMSprop: constructor validation

#[test]
fn rmsprop_rejects_invalid_learning_rate() {
    for lr in [0.0_f32, -0.01, f32::INFINITY, f32::NAN] {
        assert!(
            matches!(
                RMSprop::new(lr, 0.9, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for learning_rate={lr:?}"
        );
    }
}

/// rho must lie in [0, 1): 0.0 (inclusive lower) is accepted. 1.0 (exclusive upper),
/// out-of-range values, and NaN are rejected
#[test]
fn rmsprop_rho_bounds() {
    assert!(
        RMSprop::new(0.01, 0.0, 1e-8, 0.0).is_ok(),
        "rho=0.0 (inclusive lower bound) is accepted"
    );
    for rho in [1.0_f32, 1.5, -0.5, f32::NAN] {
        assert!(
            matches!(
                RMSprop::new(0.01, rho, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for rho={rho:?}"
        );
    }
}

#[test]
fn rmsprop_rejects_invalid_epsilon() {
    for eps in [0.0_f32, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                RMSprop::new(0.01, 0.9, eps, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for epsilon={eps:?}"
        );
    }
}

#[test]
fn rmsprop_accepts_valid_hyperparameters() {
    assert!(RMSprop::new(0.001, 0.9, 1e-8, 0.0).is_ok());
    assert!(RMSprop::new(0.01, 0.95, 1e-5, 0.0).is_ok());
}

// AdaGrad: constructor validation

#[test]
fn adagrad_rejects_invalid_learning_rate() {
    for lr in [0.0_f32, -0.01, f32::INFINITY, f32::NAN] {
        assert!(
            matches!(
                AdaGrad::new(lr, 1e-8, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for learning_rate={lr:?}"
        );
    }
}

#[test]
fn adagrad_rejects_invalid_epsilon() {
    for eps in [0.0_f32, -1e-8, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                AdaGrad::new(0.01, eps, 0.0),
                Err(Error::InvalidParameter { .. })
            ),
            "expected InvalidParameter for epsilon={eps:?}"
        );
    }
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
    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let mse = eval_mse(&model, &x, &y);
    assert_abs_diff_eq!(mse, 1.875_f32, epsilon = 1e-5);
}

// End-to-end convergence: each optimizer drives loss down over 20 epochs

#[test]
fn sgd_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    // Plain SGD converges only below lr = 2 / lambda_max(Hessian), about 2 / 5.5, or about 0.36
    // here. A rate of 0.5 diverges. This test uses 0.1, which lowers the loss steadily.
    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

#[test]
fn adam_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

#[test]
fn rmsprop_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

#[test]
fn adagrad_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();
    let initial_mse = 1.875_f32;

    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

/// Adam on Dense(1->4) -> Dense(4->1) allocates moment buffers for both layers. Loss falls:
/// below the initial value after 20 epochs, and not above the 20-epoch value after 40
#[test]
fn adam_two_layer_loss_decreases_and_buffers_allocated_correctly() {
    let (x, y) = regression_data();

    // Seed the dense layers and the fit-time shuffle so the test is deterministic
    // and never flakes on a pathological Xavier init
    const SEED: u64 = 0;
    let build_model = || -> Sequential {
        let layer1 = Dense::new(4, Linear::new())
            .unwrap()
            .with_random_state(SEED);
        let layer2 = Dense::new(1, Linear::new())
            .unwrap()
            .with_random_state(SEED);

        let mut model = SequentialBuilder::new_with_seed(SEED)
            .add(layer1)
            .add(layer2)
            .build(&Shape::known(x.shape()))
            .unwrap();
        model.compile(
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

// Multi-layer convergence: 1 test per remaining optimizer

/// SGD on a 2-layer net (1->4->1): loss falls and converges below 0.1 over 50 epochs
#[test]
fn sgd_two_layer_loss_decreases() {
    const SEED: u64 = 0;
    let (x, y) = regression_data();

    let layer1 = Dense::new(4, Linear::new())
        .unwrap()
        .with_random_state(SEED);
    let layer2 = Dense::new(1, Linear::new())
        .unwrap()
        .with_random_state(SEED);

    let mut model = SequentialBuilder::new()
        .add(layer1)
        .add(layer2)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

    let layer1 = Dense::new(4, Linear::new())
        .unwrap()
        .with_random_state(SEED);
    let layer2 = Dense::new(1, Linear::new())
        .unwrap()
        .with_random_state(SEED);

    let mut model = SequentialBuilder::new()
        .add(layer1)
        .add(layer2)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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
    const SEED: u64 = 0;
    let (x, y) = regression_data();

    let layer1 = Dense::new(4, Linear::new())
        .unwrap()
        .with_random_state(SEED);
    let layer2 = Dense::new(1, Linear::new())
        .unwrap()
        .with_random_state(SEED);

    let mut model = SequentialBuilder::new()
        .add(layer1)
        .add(layer2)
        .build(&Shape::known(&[1, 1]))
        .unwrap();
    model.compile(
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

// Numerical value: SGD 1-step weight update on known weights

/// 1 SGD step (lr=0.01) on w=1, b=0 with x=2, y=6 yields y_hat=2.40 after refit
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
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(x.shape())).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let pred_before = model.predict(&x).unwrap();
    let val_before = *pred_before.iter().next().unwrap();
    assert_abs_diff_eq!(val_before, 2.0_f32, epsilon = 1e-6);

    model.fit(&x, &y, 1).unwrap();

    // after 1 SGD step: w_new=1.16, b_new=0.08, so y_hat = 2.0*1.16 + 0.08 = 2.40
    let pred_after = model.predict(&x).unwrap();
    let val_after = *pred_after.iter().next().unwrap();
    assert_abs_diff_eq!(val_after, 2.40_f32, epsilon = 1e-4);
}

// Clip-by-global-norm (opt-in via the `with_global_clipnorm` builder)

/// Reads the (weight, bias) scalars of a model whose first layer is a 1x1 Dense
fn dense_wb(model: &Sequential) -> (f32, f32) {
    (
        model.weight("0.kernel").expect("layer 0 must be Dense")[[0, 0]],
        model.weight("0.bias").expect("layer 0 must be Dense")[[0, 0]],
    )
}

/// 1 clipped SGD step on the same w=1, b=0, x=2, y=6 problem as the hand-calc test above.
/// Clipping scales every gradient by max_norm / global_norm, preserving direction
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
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(x.shape())).unwrap();
    layer.set_weights(w, b).unwrap();

    let max_norm = 8.0_f32;
    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_global_clipnorm(max_norm)
            .unwrap(),
        MeanSquaredError::new(),
    );

    model.fit(&x, &y, 1).unwrap();

    // grad_w=-16, grad_b=-8, so global_norm = sqrt(16^2 + 8^2), which is sqrt(320).
    // scale = max_norm / global_norm = 8 / sqrt(320).
    // Unclipped deltas were +0.16 (w) and +0.08 (b).
    let scale = max_norm / 320.0_f32.sqrt();
    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 1.0 + 0.16 * scale, epsilon = 1e-5);
    assert_abs_diff_eq!(b_new, 0.0 + 0.08 * scale, epsilon = 1e-5);
    // Clipping scales both parameters by the same factor, so direction is preserved
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
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(x.shape())).unwrap();
    layer.set_weights(w, b).unwrap();

    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_global_clipnorm(100.0)
            .unwrap(),
        MeanSquaredError::new(),
    );

    model.fit(&x, &y, 1).unwrap();

    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 1.16_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(b_new, 0.08_f32, epsilon = 1e-5);
}

/// `with_global_clipnorm` rejects non-positive or non-finite thresholds, and accepts a valid
/// positive one. Constructing without it leaves clipping disabled
#[test]
fn new_rejects_invalid_global_clipnorm() {
    for bad in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            matches!(
                SGD::new(0.01, 0.0, false, 0.0)
                    .unwrap()
                    .with_global_clipnorm(bad),
                Err(Error::InvalidParameter { .. })
            ),
            "global_clipnorm {bad} should be rejected"
        );
    }
    assert!(
        SGD::new(0.01, 0.0, false, 0.0)
            .unwrap()
            .with_global_clipnorm(5.0)
            .is_ok()
    );
    assert!(SGD::new(0.01, 0.0, false, 0.0).is_ok());
}

// SGD: momentum, weight decay, and LR scheduling (integration)

/// `set_learning_rate` retunes the step: doubling lr before 1 SGD step doubles the weight delta.
/// Reuses the w=1, b=0, x=2, y=6 problem (grad_w=-16, grad_b=-8)
#[test]
fn set_learning_rate_scales_the_step() {
    let x = Array::from_shape_vec((1, 1), vec![2.0_f32])
        .unwrap()
        .into_dyn();
    let y = Array::from_shape_vec((1, 1), vec![6.0_f32])
        .unwrap()
        .into_dyn();
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 1])).unwrap();
    layer
        .set_weights(
            Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap(),
            Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap(),
        )
        .unwrap();

    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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
    let mut layer = Dense::new(1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 1])).unwrap();
    layer
        .set_weights(
            Array::from_shape_vec((1, 1), vec![1.0_f32]).unwrap(),
            Array::from_shape_vec((1, 1), vec![0.0_f32]).unwrap(),
        )
        .unwrap();

    let mut model = SequentialBuilder::new()
        .add(layer)
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        SGD::new(0.01, 0.0, false, 0.5).unwrap(),
        MeanSquaredError::new(),
    );
    model.fit(&x, &y, 1).unwrap();

    let (w_new, b_new) = dense_wb(&model);
    assert_abs_diff_eq!(w_new, 0.995 + 0.16, epsilon = 1e-5); // 1.155
    assert_abs_diff_eq!(b_new, 0.08, epsilon = 1e-5); // b=0, decay no-op
}

// Weight decay applies to weights only, not to biases or to normalization gamma and beta.

/// Runs a Dense(2->2, Linear) with fixed weights and bias through 1 forward and backward pass
/// and 1 SGD step at the given `weight_decay`. The upstream gradient is fixed and nonzero.
/// Returns the resulting (weights, bias). Both decay settings see identical gradients.
fn dense_after_one_sgd_step(
    w0: &Array2<f32>,
    b0: &Array2<f32>,
    lr: f32,
    weight_decay: f32,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    layer.set_weights(w0.clone(), b0.clone()).unwrap();
    let x = Array::from_shape_vec((1, 2), vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let _ = layer.forward(&x, &mut ctx).unwrap();
    let grad_out = Array::from_shape_vec((1, 2), vec![0.7_f32, -1.3])
        .unwrap()
        .into_dyn();
    layer.backward(&grad_out, &mut ctx).unwrap();

    let mut opt = SGD::new(lr, 0.0, false, weight_decay).unwrap();
    opt.step();
    opt.update(0, &mut layer, ctx.grads(), 1.0);
    (
        named(&layer, "kernel").to_owned(),
        named(&layer, "bias").to_owned(),
    )
}

/// Decoupled weight decay shrinks Dense weights by exactly `lr*wd*w0`, but leaves the bias
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

/// Runs a BatchNormalization layer (gamma=1, beta=0) through 1 training forward and backward
/// pass and 1 SGD step at the given `weight_decay`. The upstream gradient is fixed and nonzero.
/// Returns the resulting (gamma, beta).
fn batchnorm_gamma_beta_after_one_sgd_step(weight_decay: f32) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
    let x = Array::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let _ = bn.forward_mut(&x, &mut ctx).unwrap();
    let grad_out = Array::from_shape_vec((2, 3), vec![0.5_f32, -0.5, 1.0, -1.0, 0.25, -0.25])
        .unwrap()
        .into_dyn();
    bn.backward(&grad_out, &mut ctx).unwrap();

    let mut opt = SGD::new(0.1, 0.0, false, weight_decay).unwrap();
    opt.step();
    opt.update(0, &mut bn, ctx.grads(), 1.0);
    (
        named(&bn, "gamma").to_owned(),
        named(&bn, "beta").to_owned(),
    )
}

/// Normalization scale and shift (gamma and beta) are excluded from weight decay: a non-zero
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
    // Guard against a vacuous pass. The gradient step must actually update gamma.
    // The "identical" check above then compares moved values, not 2 untouched 1.0 arrays.
    assert!(
        g_plain.iter().any(|&v| (v - 1.0).abs() > 1e-5),
        "gamma should have a non-trivial gradient update"
    );
}

// Adam (classic coupled L2 weight decay) vs AdamW (decoupled weight decay)

/// Runs a Dense(2->2, Linear) with fixed weights and bias through 1 forward and backward pass
/// and 1 step of the given optimizer. The upstream gradient is fixed and nonzero. Returns the
/// resulting (weights, bias). Generic over the optimizer, so Adam and AdamW share the harness.
fn dense_weights_after_one_step<O: Optimizer>(
    mut opt: O,
    w0: &Array2<f32>,
    b0: &Array2<f32>,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    layer.set_weights(w0.clone(), b0.clone()).unwrap();
    let x = Array::from_shape_vec((1, 2), vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let _ = layer.forward(&x, &mut ctx).unwrap();
    let grad_out = Array::from_shape_vec((1, 2), vec![0.7_f32, -1.3])
        .unwrap()
        .into_dyn();
    layer.backward(&grad_out, &mut ctx).unwrap();

    opt.step();
    opt.update(0, &mut layer, ctx.grads(), 1.0);
    (
        named(&layer, "kernel").to_owned(),
        named(&layer, "bias").to_owned(),
    )
}

/// With `weight_decay == 0.0`, Adam and AdamW are the same algorithm: identical weights and bias
#[test]
fn adam_equals_adamw_without_weight_decay() {
    let w0 = Array::from_shape_vec((2, 2), vec![1.0_f32, -2.0, 3.0, -4.0]).unwrap();
    let b0 = Array::from_shape_vec((1, 2), vec![0.5_f32, -1.5]).unwrap();

    let (w_adam, b_adam) =
        dense_weights_after_one_step(Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap(), &w0, &b0);
    let (w_adamw, b_adamw) =
        dense_weights_after_one_step(AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap(), &w0, &b0);

    for (a, w) in w_adam.iter().zip(w_adamw.iter()) {
        assert_abs_diff_eq!(*a, *w, epsilon = 1e-7);
    }
    for (a, w) in b_adam.iter().zip(b_adamw.iter()) {
        assert_abs_diff_eq!(*a, *w, epsilon = 1e-7);
    }
}

/// Coupled L2 (Adam) flows through the moments and the adaptive denominator. Decoupled decay
/// (AdamW) does not, so the 2 schemes diverge on the weights when weight_decay is non-zero
#[test]
fn adam_l2_and_adamw_decoupled_differ_with_weight_decay() {
    let w0 = Array::from_shape_vec((2, 2), vec![1.0_f32, -2.0, 3.0, -4.0]).unwrap();
    let b0 = Array::from_shape_vec((1, 2), vec![0.5_f32, -1.5]).unwrap();
    let wd = 0.5_f32;

    let (w_adam, b_adam) =
        dense_weights_after_one_step(Adam::new(0.1, 0.9, 0.999, 1e-8, wd).unwrap(), &w0, &b0);
    let (w_adamw, b_adamw) =
        dense_weights_after_one_step(AdamW::new(0.1, 0.9, 0.999, 1e-8, wd).unwrap(), &w0, &b0);

    assert!(
        w_adam
            .iter()
            .zip(w_adamw.iter())
            .any(|(a, w)| (a - w).abs() > 1e-5),
        "coupled (Adam) and decoupled (AdamW) weight decay should produce different weights"
    );
    // Bias is excluded from weight decay in both, so it updates identically
    for (a, w) in b_adam.iter().zip(b_adamw.iter()) {
        assert_abs_diff_eq!(*a, *w, epsilon = 1e-7);
    }
}

#[test]
fn adamw_single_layer_loss_decreases_over_20_epochs() {
    let (x, y) = regression_data();

    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
        AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01).unwrap(),
        MeanSquaredError::new(),
    );

    let mse_before = eval_mse(&model, &x, &y);
    model.fit(&x, &y, 20).unwrap();
    let mse_after = eval_mse(&model, &x, &y);
    assert!(
        mse_after < mse_before,
        "AdamW: loss should decrease; before={mse_before}, after={mse_after}"
    );
}

/// AdamW routes through the same validators as Adam: rejects out-of-range betas or negative
/// weight_decay, accepts valid hyperparameters
#[test]
fn adamw_validates_hyperparameters() {
    assert!(matches!(
        AdamW::new(0.001, 1.0, 0.999, 1e-8, 0.0),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(matches!(
        AdamW::new(0.001, 0.9, 0.999, 1e-8, -0.1),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01).is_ok());
    assert!(
        AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_global_clipnorm(1.0)
            .is_ok()
    );
}

#[test]
fn sgd_momentum_loss_decreases() {
    let (x, y) = regression_data();
    let mut model = SequentialBuilder::new()
        .add(identity_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    model.compile(
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

// Learning rate: readable as well as writable

/// Every optimizer round-trips learning_rate(): it reports the value it was built with, and the
/// value it was last set to. This lets a schedule read the current rate instead of a stale copy
#[test]
fn every_optimizer_reports_its_current_learning_rate() {
    fn round_trip(mut opt: impl Optimizer, configured: f32) {
        assert_abs_diff_eq!(opt.learning_rate(), configured, epsilon = 0.0_f32);

        // The decay a scheduler writes: derived from the current value, not from a copy
        let decayed = opt.learning_rate() * 0.1;
        opt.set_learning_rate(decayed);
        assert_abs_diff_eq!(opt.learning_rate(), configured * 0.1, epsilon = 1e-9);
    }

    round_trip(SGD::new(0.05, 0.9, false, 0.0).unwrap(), 0.05);
    round_trip(Adam::new(0.003, 0.9, 0.999, 1e-8, 0.0).unwrap(), 0.003);
    round_trip(AdamW::new(0.003, 0.9, 0.999, 1e-8, 0.01).unwrap(), 0.003);
    round_trip(RMSprop::new(0.002, 0.9, 1e-8, 0.0).unwrap(), 0.002);
    round_trip(AdaGrad::new(0.02, 1e-8, 0.0).unwrap(), 0.02);
}

// Parameter identity: per-parameter optimizer state must follow the parameter, not a position

/// Builds a Dense(2 -> 2, Linear) that holds the identity map, so 2 runs of the same schedule
/// stay comparable element by element
fn pass_through_dense() -> Dense {
    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let weights = Array::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let bias = Array::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    layer.set_weights(weights, bias).unwrap();
    layer
}

/// Runs 1 forward and 1 backward pass over a fixed input and a fixed upstream gradient, and
/// gives the context back, so the caller reads the gradients of the pass out of its store
///
/// `scope` is the position the caller gives the layer. The context addresses every gradient by
/// that position, so the matching `update` call finds them
fn one_pass(layer: &mut Dense, scope: usize) -> Ctx {
    let mut ctx = Ctx::training();
    ctx.set_owner(scope);
    let x = Array::from_shape_vec((1, 2), vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let _ = layer.forward(&x, &mut ctx).unwrap();
    let grad_out = Array::from_shape_vec((1, 2), vec![0.7_f32, -1.3])
        .unwrap()
        .into_dyn();
    layer.backward(&grad_out, &mut ctx).unwrap();
    ctx
}

/// The live kernel of a Dense layer
fn dense_kernel(layer: &Dense) -> ArrayD<f32> {
    named(layer, "kernel").to_owned()
}

/// Asserts that 2 kernels agree element by element
fn assert_same_kernel(got: &ArrayD<f32>, want: &ArrayD<f32>, message: &str) {
    assert_eq!(got.shape(), want.shape());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((*g - *w).abs() <= 1e-6, "{message}: {g} != {w}");
    }
}

/// A layer whose gradient store holds nothing on 1 step and 2 gradients on the next must not
/// move any other layer's per-parameter optimizer state
///
/// `quiet` runs no pass before step 1, so its store holds nothing there. `tracked` runs a pass
/// on both steps, so its store holds 2 gradients each time. The momentum buffer of every
/// `tracked` tensor must therefore be the same buffer on both steps, and `quiet` must start
/// from a zero buffer of its own
#[test]
fn a_changing_parameter_count_must_not_move_another_layer_state() {
    let mut quiet = pass_through_dense();
    let mut tracked = pass_through_dense();
    let mut opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();

    // Step 1: only `tracked` holds gradients
    let quiet_ctx = Ctx::training();
    let tracked_ctx = one_pass(&mut tracked, 1);
    assert!(
        quiet_ctx.grads().is_empty(),
        "a layer that ran no backward pass must hold no gradient"
    );
    assert_eq!(tracked_ctx.grads().len(), 2);
    opt.step();
    opt.update(0, &mut quiet, quiet_ctx.grads(), 1.0);
    opt.update(1, &mut tracked, tracked_ctx.grads(), 1.0);

    // Step 2: both hold gradients
    let quiet_ctx = one_pass(&mut quiet, 0);
    let tracked_ctx = one_pass(&mut tracked, 1);
    assert_eq!(quiet_ctx.grads().len(), 2);
    opt.step();
    opt.update(0, &mut quiet, quiet_ctx.grads(), 1.0);
    opt.update(1, &mut tracked, tracked_ctx.grads(), 1.0);

    // Control 1: `tracked` alone, on the same 2-step schedule and its own optimizer
    let mut tracked_control = pass_through_dense();
    let mut tracked_control_opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();
    for _ in 0..2 {
        let ctx = one_pass(&mut tracked_control, 1);
        tracked_control_opt.step();
        tracked_control_opt.update(1, &mut tracked_control, ctx.grads(), 1.0);
    }

    // Control 2: `quiet` alone, on the 1 step it takes part in, and its own optimizer
    let mut quiet_control = pass_through_dense();
    let mut quiet_control_opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();
    let quiet_control_ctx = one_pass(&mut quiet_control, 0);
    quiet_control_opt.step();
    quiet_control_opt.update(0, &mut quiet_control, quiet_control_ctx.grads(), 1.0);

    assert_same_kernel(
        &dense_kernel(&tracked),
        &dense_kernel(&tracked_control),
        "the momentum of `tracked` must not move when another layer starts yielding parameters",
    );
    assert_same_kernel(
        &dense_kernel(&quiet),
        &dense_kernel(&quiet_control),
        "`quiet` must start from its own zero momentum, not inherit another layer's",
    );
}

/// Builds a Dense(2 -> 2, Linear) that holds the identity map and a bias of `[1, 0]`
///
/// The bias moves the output away from the input, so a layer that stands after this one sees
/// something other than `x`. The 2 layers then have 2 different kernel gradients, which is
/// what lets the data of the test below hold 1 of them at 0 and the other away from it
fn shifted_dense() -> Dense {
    let mut layer = Dense::new(2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let weights = Array::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let bias = Array::from_shape_vec((1, 2), vec![1.0_f32, 0.0]).unwrap();
    layer.set_weights(weights, bias).unwrap();
    layer
}

/// Builds a bias-free Dense(2 -> 2, Linear) that holds the identity map
///
/// A bias-free layer holds the kernel alone. A step that leaves the kernel where it is
/// therefore leaves the whole layer where it is, and the layer goes on passing its input and
/// its gradient through unchanged
fn pass_through_dense_no_bias() -> Dense {
    let mut layer = Dense::new(2, Linear::new()).unwrap().with_use_bias(false);
    layer.build(&Shape::known(&[1, 2])).unwrap();
    let weights = Array::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    layer.set_weights(weights, None::<Array2<f32>>).unwrap();
    layer
}

/// A second layer in the stack moves no per-parameter optimizer state of the first one
///
/// The optimizer keys its state on [`ParamId`], which is the position of the layer counted
/// from the input plus the name the layer gives the tensor. The first layer therefore holds
/// `{scope 0, "kernel"}` in both models below, and the second layer of the stacked model
/// reaches a key of its own. A key that dropped the scope would give both layers 1 momentum
/// buffer, and the first layer would carry the gradient of the second one into batch 2
///
/// The 2 models must agree element by element, so the second layer must pass the forward
/// output and the backward gradient through unchanged for every batch that the comparison
/// covers. It holds the identity map at the start, and the data below holds it there. The
/// first layer adds a bias of `[1, 0]`, so what reaches the second layer is `h = x + [1, 0]`.
/// The kernel gradient of the second layer is `h^T (pred - y)`, and `y` is picked so that
/// `h` has 2 proportional rows that both stand orthogonal to `pred - y`. That gradient is
/// therefore 0, and a bias-free layer has no other tensor to move. The kernel gradient of the
/// first layer is `x^T (pred - y)`, and the bias shift keeps that 1 away from 0, so a shared
/// momentum buffer would carry a real value from the first layer into the second one
#[test]
fn a_second_layer_moves_no_state_of_the_first() {
    // The first layer adds `[1, 0]`, so it gives the second layer `h = [[1, 2], [2, 4]]`, whose
    // 2 rows are proportional
    let x = Array::from_shape_vec((2, 2), vec![0.0_f32, 2.0, 1.0, 4.0])
        .unwrap()
        .into_dyn();
    // The prediction is `h`, so `pred - y` is `[[2, 2], [-1, -1]]`. Every column of it stands
    // orthogonal to both columns of `h`, and to neither column of `x`
    let y = Array::from_shape_vec((2, 2), vec![-1.0_f32, 0.0, 3.0, 5.0])
        .unwrap()
        .into_dyn();

    let mut plain = SequentialBuilder::new()
        .add(shifted_dense())
        .build(&Shape::known(x.shape()))
        .unwrap();
    plain.compile(
        SGD::new(0.1, 0.9, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    plain.train_batch(&x, &y).unwrap();
    plain.train_batch(&x, &y).unwrap();

    let mut stacked = SequentialBuilder::new()
        .add(shifted_dense())
        .add(pass_through_dense_no_bias())
        .build(&Shape::known(x.shape()))
        .unwrap();
    stacked.compile(
        SGD::new(0.1, 0.9, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    stacked.train_batch(&x, &y).unwrap();
    stacked.train_batch(&x, &y).unwrap();

    let kernel_of = |model: &Sequential, path: &str| {
        model
            .weight(path)
            .unwrap_or_else(|| panic!("{path} must name a kernel"))
            .to_owned()
    };

    // The comparison holds because batch 1 gives the second layer a gradient of 0, and its
    // batch-2 step lands after every backward pass of that batch, so the first layer's
    // gradient always passes through the second layer as the identity.
    let mut probe = SequentialBuilder::new()
        .add(shifted_dense())
        .add(pass_through_dense_no_bias())
        .build(&Shape::known(x.shape()))
        .unwrap();
    probe.compile(
        SGD::new(0.1, 0.9, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    probe.train_batch(&x, &y).unwrap();
    assert_same_kernel(
        &kernel_of(&probe, "1.kernel"),
        &Array::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
        "the second layer must take no step in batch 1",
    );

    assert_same_kernel(
        &kernel_of(&stacked, "0.kernel"),
        &kernel_of(&plain, "0.kernel"),
        "the first layer must keep its own momentum when a second layer stands after it",
    );
}

/// A stand-in for a layer whose gradient roster changes between steps
///
/// It holds 2 tensors of equal length, and the store of a pass can carry the gradient of the
/// second one alone. No layer in the crate has that shape, but the optimizer contract must
/// hold for it too
struct RosterLayer {
    /// The tensor whose gradient the store can leave out
    gamma: Vec<f32>,
    /// The tensor whose gradient the store always holds
    beta: Vec<f32>,
}

impl RosterLayer {
    /// A layer whose 2 tensors both start at 0
    fn new() -> Self {
        Self {
            gamma: vec![0.0; 4],
            beta: vec![0.0; 4],
        }
    }
}

/// The fixed gradient of every tensor of a `RosterLayer`, on every step
const ROSTER_GRAD: [f32; 4] = [0.5, -0.25, 1.0, -2.0];

impl LayerBase for RosterLayer {
    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.gamma.len() + self.beta.len())
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        vec![
            ParamRef::no_decay("gamma", &mut self.gamma),
            ParamRef::no_decay("beta", &mut self.beta),
        ]
    }

    fn weights(&self) -> Vec<WeightRef<'_>> {
        Vec::new()
    }

    fn weights_mut(&mut self) -> Vec<WeightMut<'_>> {
        Vec::new()
    }
}

/// The gradient store of 1 step of a `RosterLayer`, at layer position 0
///
/// `beta` always holds a gradient. `gamma` holds one only when `with_gamma` is set, which is
/// how the test makes a tensor join the roster on a later step
fn roster_grads(with_gamma: bool) -> Ctx {
    let mut ctx = Ctx::training();
    if with_gamma {
        ctx.add_grad("gamma", Array::from_vec(ROSTER_GRAD.to_vec()).into_dyn())
            .unwrap();
    }
    ctx.add_grad("beta", Array::from_vec(ROSTER_GRAD.to_vec()).into_dyn())
        .unwrap();
    ctx
}

/// A tensor that joins the parameter roster on a later step must not take the optimizer state
/// of a tensor that was there first
///
/// `gamma` is the first entry of the roster and it holds no gradient on step 1. A positional
/// key gives its slot to `beta` on step 1, and hands that slot back to `gamma` on step 2. The 2
/// tensors have equal length, so a length check reports nothing
#[test]
fn a_tensor_joining_the_roster_must_not_take_another_tensor_state() {
    // Run under test: gamma joins on step 2
    let mut layer = RosterLayer::new();
    let mut opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();
    opt.step();
    opt.update(0, &mut layer, roster_grads(false).grads(), 1.0);
    opt.step();
    opt.update(0, &mut layer, roster_grads(true).grads(), 1.0);

    // Control for beta: the same 2 steps, and gamma never joins
    let mut beta_control = RosterLayer::new();
    let mut beta_control_opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();
    for _ in 0..2 {
        beta_control_opt.step();
        beta_control_opt.update(0, &mut beta_control, roster_grads(false).grads(), 1.0);
    }

    // Control for gamma: 1 step from a zero momentum buffer of its own
    let mut gamma_control = RosterLayer::new();
    let mut gamma_control_opt = SGD::new(0.1, 0.9, false, 0.0).unwrap();
    gamma_control_opt.step();
    gamma_control_opt.update(0, &mut gamma_control, roster_grads(true).grads(), 1.0);

    for i in 0..4 {
        assert!(
            (layer.beta[i] - beta_control.beta[i]).abs() <= 1e-6,
            "beta must keep its own momentum when gamma joins: {} != {}",
            layer.beta[i],
            beta_control.beta[i]
        );
        assert!(
            (layer.gamma[i] - gamma_control.gamma[i]).abs() <= 1e-6,
            "gamma must start from its own zero momentum: {} != {}",
            layer.gamma[i],
            gamma_control.gamma[i]
        );
    }
    // Guard against a vacuous pass: both tensors must have actually moved
    assert!(layer.beta[0].abs() > 1e-4 && layer.gamma[0].abs() > 1e-4);
}

// The layer half of the parameter address: state must never cross between 2 layers

/// The upstream gradient of the layer at scope 0
const FIRST_LAYER_GRAD: [f32; 2] = [0.7, -1.3];

/// The upstream gradient of the layer at scope 1
///
/// The 2 gradients must differ. State that crosses between the layers then changes the result,
/// and the test can see it
const SECOND_LAYER_GRAD: [f32; 2] = [-0.4, 2.1];

/// Runs 1 forward and 1 backward pass over a fixed input and the given upstream gradient, and
/// gives the context back
///
/// The activation is Linear and the input is fixed, so the parameter gradients are the same on
/// every call. The schedules below are therefore deterministic. `scope` is the position the
/// caller gives the layer, so every gradient reaches the address that `update` reads
fn one_pass_with(layer: &mut Dense, scope: usize, grad_out: [f32; 2]) -> Ctx {
    let mut ctx = Ctx::training();
    ctx.set_owner(scope);
    let x = Array::from_shape_vec((1, 2), vec![1.0_f32, 2.0])
        .unwrap()
        .into_dyn();
    let _ = layer.forward(&x, &mut ctx).unwrap();
    let grad = Array::from_shape_vec((1, 2), grad_out.to_vec())
        .unwrap()
        .into_dyn();
    layer.backward(&grad, &mut ctx).unwrap();
    ctx
}

/// Asserts that an optimizer keys its per-parameter state on the layer index as well as on the
/// name of the tensor
///
/// 2 Dense layers of the same shape run on 1 optimizer, at scope 0 and at scope 1. Both layers
/// name their tensors `kernel` and `bias`, and the 2 tensors of a name have equal length. A key
/// that drops the scope therefore collides across the 2 layers, and the length guard inside the
/// optimizer cannot see the collision. The gradients of the 2 layers differ, so a collision
/// moves both layers away from the control, which gives each layer an optimizer of its own
fn scope_must_separate_2_same_shape_layers<O: Optimizer>(make: impl Fn() -> O, optimizer: &str) {
    // 1 step is not enough. The state that step 1 writes is the state that step 2 reads
    const STEPS: usize = 3;

    // Run under test: 1 optimizer, 2 layers, 2 scopes
    let mut first = pass_through_dense();
    let mut second = pass_through_dense();
    let mut opt = make();
    for _ in 0..STEPS {
        let first_ctx = one_pass_with(&mut first, 0, FIRST_LAYER_GRAD);
        let second_ctx = one_pass_with(&mut second, 1, SECOND_LAYER_GRAD);
        opt.step();
        opt.update(0, &mut first, first_ctx.grads(), 1.0);
        opt.update(1, &mut second, second_ctx.grads(), 1.0);
    }

    // Control: the same schedule, with 1 optimizer per layer, so no state can cross
    let mut first_control = pass_through_dense();
    let mut second_control = pass_through_dense();
    let mut first_opt = make();
    let mut second_opt = make();
    for _ in 0..STEPS {
        let first_ctx = one_pass_with(&mut first_control, 0, FIRST_LAYER_GRAD);
        let second_ctx = one_pass_with(&mut second_control, 1, SECOND_LAYER_GRAD);
        first_opt.step();
        second_opt.step();
        first_opt.update(0, &mut first_control, first_ctx.grads(), 1.0);
        second_opt.update(1, &mut second_control, second_ctx.grads(), 1.0);
    }

    let first_kernel = dense_kernel(&first);
    let second_kernel = dense_kernel(&second);
    assert_same_kernel(
        &first_kernel,
        &dense_kernel(&first_control),
        &format!(
            "{optimizer}: the layer at scope 0 must not read the state of the layer at scope 1"
        ),
    );
    assert_same_kernel(
        &second_kernel,
        &dense_kernel(&second_control),
        &format!(
            "{optimizer}: the layer at scope 1 must not read the state of the layer at scope 0"
        ),
    );

    // Guard against a vacuous pass. Both layers must have moved away from the start, and the 2
    // layers must reach different weights. A shared key is invisible if either fails
    let start = dense_kernel(&pass_through_dense());
    assert!(
        first_kernel
            .iter()
            .zip(start.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "{optimizer}: the layer at scope 0 must have moved"
    );
    assert!(
        second_kernel
            .iter()
            .zip(start.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "{optimizer}: the layer at scope 1 must have moved"
    );
    assert!(
        first_kernel
            .iter()
            .zip(second_kernel.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "{optimizer}: the 2 layers must reach different weights"
    );
}

/// SGD momentum buffers must belong to 1 layer each
#[test]
fn sgd_scope_must_separate_2_same_shape_layers() {
    scope_must_separate_2_same_shape_layers(|| SGD::new(0.1, 0.9, false, 0.0).unwrap(), "SGD");
}

/// Adam moment buffers must belong to 1 layer each
#[test]
fn adam_scope_must_separate_2_same_shape_layers() {
    scope_must_separate_2_same_shape_layers(
        || Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        "Adam",
    );
}

/// AdamW moment buffers must belong to 1 layer each
#[test]
fn adamw_scope_must_separate_2_same_shape_layers() {
    scope_must_separate_2_same_shape_layers(
        || AdamW::new(0.05, 0.9, 0.999, 1e-8, 0.01).unwrap(),
        "AdamW",
    );
}

/// RMSprop caches must belong to 1 layer each
#[test]
fn rmsprop_scope_must_separate_2_same_shape_layers() {
    scope_must_separate_2_same_shape_layers(
        || RMSprop::new(0.05, 0.9, 1e-8, 0.0).unwrap(),
        "RMSprop",
    );
}

/// AdaGrad accumulators must belong to 1 layer each
#[test]
fn adagrad_scope_must_separate_2_same_shape_layers() {
    scope_must_separate_2_same_shape_layers(|| AdaGrad::new(0.1, 1e-8, 0.0).unwrap(), "AdaGrad");
}

// Momentum state must reach the parameter, and not only the right parameter

/// The SGD momentum buffer must survive between 2 calls to `update`
///
/// Every identity test above is control relative. It sees state that reaches the wrong tensor,
/// and never state that reaches no tensor at all. This test closes that hole for SGD.
///
/// The gradient is the same on every step, so the trajectory has a closed form. With
/// `v = momentum * v + g` and `p -= lr * v`, 3 steps at momentum 0.9 move the parameter by
/// `(1 + 1.9 + 2.71) = 5.61` times the plain SGD step. A buffer that starts again on every
/// call gives 3 times the plain step, which is plain SGD. The Nesterov arm uses the look-ahead
/// step `g + momentum * v`, where the same 3 steps sum to 8.049, against 5.7 for a buffer that
/// is thrown away
#[test]
fn sgd_momentum_velocity_must_survive_between_steps() {
    // 1 plain SGD step gives the displacement `lr * g` of every element
    let mut plain = pass_through_dense();
    let mut plain_opt = SGD::new(0.1, 0.0, false, 0.0).unwrap();
    let plain_ctx = one_pass(&mut plain, 0);
    plain_opt.step();
    plain_opt.update(0, &mut plain, plain_ctx.grads(), 1.0);
    let start = dense_kernel(&pass_through_dense());
    let unit = &start - &dense_kernel(&plain);
    assert!(
        unit.iter().any(|d| d.abs() > 1e-4),
        "the plain step must move the kernel, or every comparison below is vacuous"
    );

    for (nesterov, factor) in [(false, 5.61_f32), (true, 8.049_f32)] {
        let mut layer = pass_through_dense();
        let mut opt = SGD::new(0.1, 0.9, nesterov, 0.0).unwrap();
        for _ in 0..3 {
            let ctx = one_pass(&mut layer, 0);
            opt.step();
            opt.update(0, &mut layer, ctx.grads(), 1.0);
        }
        let got = dense_kernel(&layer);
        for ((g, s), u) in got.iter().zip(start.iter()).zip(unit.iter()) {
            let want = *s - factor * *u;
            assert!(
                (*g - want).abs() <= 1e-4,
                "3 momentum steps with nesterov={nesterov} must sum to {factor} plain steps: \
                 {g} != {want}"
            );
        }
    }
}

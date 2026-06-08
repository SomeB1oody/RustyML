#![cfg(feature = "neural_network")]

use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layer::activation_layer::Activation;
use rustyml::neural_network::layer::dense::Dense;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::neural_network_trait::Optimizer;
use rustyml::neural_network::optimizer::kernels::{
    adagrad_step, adam_step, rmsprop_step, sgd_step,
};
use rustyml::neural_network::optimizer::{AdaGrad, Adam, RMSprop, SGD};
use rustyml::neural_network::sequential::Sequential;

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

#[test]
fn test_adam_new() {
    assert!(Adam::new(0.001, 0.9, 0.999, 1e-8).is_ok());
    assert!(Adam::new(-0.001, 0.9, 0.999, 1e-8).is_err()); // negative lr
    assert!(Adam::new(0.0, 0.9, 0.999, 1e-8).is_err()); // zero lr
    assert!(Adam::new(0.001, -0.1, 0.999, 1e-8).is_err()); // beta1 < 0
    assert!(Adam::new(0.001, 1.1, 0.999, 1e-8).is_err()); // beta1 > 1
    assert!(Adam::new(0.001, 0.9, -0.999, 1e-8).is_err()); // beta2 < 0
    assert!(Adam::new(0.001, 0.9, 1.1, 1e-8).is_err()); // beta2 > 1
    assert!(Adam::new(0.001, 0.9, 0.999, -1e-8).is_err()); // negative epsilon
    assert!(Adam::new(0.001, 0.9, 0.999, 0.0).is_err()); // zero epsilon
    assert!(Adam::new(f32::INFINITY, 0.9, 0.999, 1e-8).is_err());
    assert!(Adam::new(0.001, 0.9, 0.999, f32::INFINITY).is_err());
}

#[test]
fn test_rmsprop_new() {
    assert!(RMSprop::new(0.001, 0.9, 1e-8).is_ok());
    assert!(RMSprop::new(-0.001, 0.9, 1e-8).is_err());
    assert!(RMSprop::new(0.0, 0.9, 1e-8).is_err());
    assert!(RMSprop::new(0.001, -0.1, 1e-8).is_err());
    assert!(RMSprop::new(0.001, 1.1, 1e-8).is_err());
    assert!(RMSprop::new(0.001, 0.9, -1e-8).is_err());
    assert!(RMSprop::new(0.001, 0.9, 0.0).is_err());
    assert!(RMSprop::new(f32::INFINITY, 0.9, 1e-8).is_err());
    assert!(RMSprop::new(0.001, 0.9, f32::INFINITY).is_err());
}

#[test]
fn test_sgd_new() {
    assert!(SGD::new(0.01).is_ok());
    assert!(SGD::new(-0.01).is_err());
    assert!(SGD::new(0.0).is_err());
    assert!(SGD::new(f32::INFINITY).is_err());
}

#[test]
fn test_ada_grad_new() {
    assert!(AdaGrad::new(0.01, 1e-8).is_ok());
    assert!(AdaGrad::new(-0.01, 1e-8).is_err());
    assert!(AdaGrad::new(0.0, 1e-8).is_err());
    assert!(AdaGrad::new(0.01, -1e-8).is_err());
    assert!(AdaGrad::new(0.01, 0.0).is_err());
    assert!(AdaGrad::new(f32::INFINITY, 1e-8).is_err());
    assert!(AdaGrad::new(0.01, f32::INFINITY).is_err());
}

// ---------------------------------------------------------------------------
// Flat-slice update kernels (the single source of truth for every optimizer)
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_step() {
    // param = param - lr * grad
    let mut weights = vec![1.0, 2.0, 3.0, 4.0];
    let grads = vec![0.1, 0.2, 0.3, 0.4];
    sgd_step(&mut weights, &grads, 0.1);
    assert!((weights[0] - 0.99).abs() < 1e-6);
    assert!((weights[1] - 1.98).abs() < 1e-6);
    assert!((weights[2] - 2.97).abs() < 1e-6);
    assert!((weights[3] - 3.96).abs() < 1e-6);
}

#[test]
fn test_adam_step() {
    let mut param = vec![0.0, 0.0];
    let grad = vec![0.1, 0.1];
    let mut m = vec![0.0, 0.0];
    let mut v = vec![0.0, 0.0];

    adam_step(
        &mut param, &grad, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 1,
    );

    // m = 0.9*0 + 0.1*0.1 = 0.01 ; v = 0.999*0 + 0.001*0.01 = 1e-5
    assert!((m[0] - 0.01).abs() < 1e-6);
    assert!((v[0] - 0.00001).abs() < 1e-6);

    // m_hat = 0.01/(1-0.9) = 0.1 ; v_hat = 1e-5/(1-0.999) = 0.01
    // update = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 0.1 / 0.1 = 0.001
    let expected = -(0.001 * 0.1 / (0.01_f32.sqrt() + 1e-8));
    assert!((param[0] - expected).abs() < 1e-6);
}

#[test]
fn test_adam_step_bias_correction_changes_with_timestep() {
    // Two updates with identical gradients differ because bias correction depends on `t`.
    let grad = vec![0.1];
    let mut p1 = vec![0.0];
    let (mut m1, mut v1) = (vec![0.0], vec![0.0]);
    adam_step(&mut p1, &grad, &mut m1, &mut v1, 0.001, 0.9, 0.999, 1e-8, 1);
    let delta1 = -p1[0];

    let mut p2 = p1.clone();
    adam_step(&mut p2, &grad, &mut m1, &mut v1, 0.001, 0.9, 0.999, 1e-8, 2);
    let delta2 = p1[0] - p2[0];

    assert!((delta1 - delta2).abs() > 1e-10);
}

#[test]
fn test_rmsprop_step() {
    let mut param = vec![1.0, 2.0];
    let grad = vec![0.1, 0.2];
    let mut cache = vec![0.0, 0.0];

    rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.001, 1e-8);

    // cache = 0.9*0 + 0.1*grad^2
    assert!((cache[0] - 0.001).abs() < 1e-6);
    assert!((cache[1] - 0.004).abs() < 1e-6);

    let expected0 = 1.0 - 0.001 * 0.1 / (0.001_f32.sqrt() + 1e-8);
    let expected1 = 2.0 - 0.001 * 0.2 / (0.004_f32.sqrt() + 1e-8);
    assert!((param[0] - expected0).abs() < 1e-6);
    assert!((param[1] - expected1).abs() < 1e-6);
}

#[test]
fn test_rmsprop_step_accumulates() {
    let mut param = vec![1.0, 2.0];
    let grad = vec![0.1, 0.2];
    let mut cache = vec![0.0, 0.0];
    rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.001, 1e-8);
    let cache_after_first = cache.clone();
    rmsprop_step(&mut param, &grad, &mut cache, 0.9, 0.001, 1e-8);
    // cache_new = rho*cache_old + (1-rho)*grad^2
    let expected0 = 0.9 * cache_after_first[0] + 0.1 * 0.01;
    let expected1 = 0.9 * cache_after_first[1] + 0.1 * 0.04;
    assert!((cache[0] - expected0).abs() < 1e-6);
    assert!((cache[1] - expected1).abs() < 1e-6);
}

#[test]
fn test_adagrad_step() {
    let mut param = vec![0.0, 0.0];
    let grad = vec![0.1, 0.1];
    let mut acc = vec![0.0, 0.0];

    adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);

    // acc = grad^2 = 0.01
    assert!((acc[0] - 0.01).abs() < 1e-6);
    // update = lr * grad / (sqrt(acc) + eps)
    let expected = -(0.01 * 0.1 / (0.01_f32.sqrt() + 1e-8));
    assert!((param[0] - expected).abs() < 1e-6);
}

#[test]
fn test_adagrad_step_accumulates() {
    let mut param = vec![0.0, 0.0];
    let grad = vec![0.1, 0.2];
    let mut acc = vec![0.0, 0.0];
    adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);
    let acc_after_first = acc.clone();
    adagrad_step(&mut param, &grad, &mut acc, 0.01, 1e-8);
    // AdaGrad accumulates (never decays): acc_new = acc_old + grad^2
    assert!((acc[0] - (acc_after_first[0] + 0.01)).abs() < 1e-6);
    assert!((acc[1] - (acc_after_first[1] + 0.04)).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// End-to-end: every optimizer drives a layer through `Optimizer::{step, update}`
// (exercises the optimizer-owned per-parameter state + cursor mapping).
// ---------------------------------------------------------------------------

fn mse(pred: &Tensor, target: &Tensor) -> f32 {
    (pred - target).mapv(|v| v * v).sum() / pred.len() as f32
}

/// Trains a single Dense layer to fit a linear target and asserts the loss decreases.
fn assert_optimizer_reduces_loss<O: 'static + Optimizer>(optimizer: O) {
    // Linear target y = x @ w_true, learnable exactly by a Dense(Linear) layer.
    let x =
        Array::from_shape_fn((8, 3), |(i, j)| 0.1 * (i as f32) + 0.2 * (j as f32) - 0.5).into_dyn();
    let y = Array::from_shape_fn((8, 1), |(i, _)| 0.3 * (i as f32) - 0.1).into_dyn();

    let mut model = Sequential::new();
    model
        .add(Dense::new(3, 1, Activation::Linear).unwrap())
        .compile(optimizer, MeanSquaredError::new());

    let loss_before = mse(&model.predict(&x).unwrap(), &y);
    model.fit(&x, &y, 60).unwrap();
    let loss_after = mse(&model.predict(&x).unwrap(), &y);

    assert!(
        loss_after < loss_before,
        "optimizer did not reduce loss: {loss_before} -> {loss_after}"
    );
}

#[test]
fn test_sgd_end_to_end_reduces_loss() {
    assert_optimizer_reduces_loss(SGD::new(0.05).unwrap());
}

#[test]
fn test_adam_end_to_end_reduces_loss() {
    assert_optimizer_reduces_loss(Adam::new(0.05, 0.9, 0.999, 1e-8).unwrap());
}

#[test]
fn test_rmsprop_end_to_end_reduces_loss() {
    assert_optimizer_reduces_loss(RMSprop::new(0.01, 0.9, 1e-8).unwrap());
}

#[test]
fn test_ada_grad_end_to_end_reduces_loss() {
    assert_optimizer_reduces_loss(AdaGrad::new(0.1, 1e-8).unwrap());
}

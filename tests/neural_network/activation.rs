//! Integration tests for the activation layers and the `Activation` enum.
//!
//! Expected values come from the mathematical definitions. `gradient_check.rs` covers gradient
//! correctness. This file does not duplicate those backward-value checks.
//!
//! Coverage:
//!   - forward values, predict() == forward()
//!   - backward before forward -> NnError::ForwardPassNotRun
//!   - non-finite input propagates (pure math, no rejection)
//!   - empty input -> Error::EmptyInput
//!   - Activation enum forward delegation, and the Linear layer
//!   - parameter validation: LeakyReLU and ELU reject an unusable parameter, in the layer
//!     constructor and in a trainable layer that embeds the activation
//!   - Default() equals the layer's own default value, and a standalone layer equals the
//!     matching Activation variant

// The pinned float32 values in this file are reference values. Keep them as written, so a
// reader can compare them to the reference output character by character.
#![allow(clippy::excessive_precision)]

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::elu::ELU;
use rustyml::neural_network::layers::activation::exponential::Exponential;
use rustyml::neural_network::layers::activation::hard_sigmoid::HardSigmoid;
use rustyml::neural_network::layers::activation::leaky_relu::LeakyReLU;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::selu::SELU;
use rustyml::neural_network::layers::activation::sigmoid::Sigmoid;
use rustyml::neural_network::layers::activation::softmax::Softmax;
use rustyml::neural_network::layers::activation::softplus::Softplus;
use rustyml::neural_network::layers::activation::softsign::Softsign;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use crate::common::assert_allclose;

/// Build a 2-D tensor from row-major `data` with shape `(rows, cols)`
fn tensor2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .expect("shape/data mismatch in tensor2")
        .into_dyn()
}

// ReLU layer

/// ReLU(x) = max(0, x) on known values
#[test]
fn relu_forward_known_values() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 3, vec![-2.0, 0.0, 3.0]);
    let output = layer.forward(&input).expect("ReLU forward failed");

    let expected = tensor2(1, 3, vec![0.0, 0.0, 3.0]);
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// All-negative input maps to zeros
#[test]
fn relu_forward_all_negative() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 2, vec![-5.0, -1.0]);
    let output = layer
        .forward(&input)
        .expect("ReLU forward all-negative failed");

    let expected = tensor2(1, 2, vec![0.0, 0.0]);
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// All-positive input passes through unchanged
#[test]
fn relu_forward_all_positive() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 3, vec![0.5, 1.0, 100.0]);
    let output = layer
        .forward(&input)
        .expect("ReLU forward all-positive failed");
    assert_allclose(&output, &input, 1e-6_f32);
}

/// predict() equals forward()
#[test]
fn relu_predict_equals_forward() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 3, vec![-2.0, 0.0, 3.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn relu_backward_before_forward_is_error() {
    let mut layer = ReLU::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("ReLU")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Non-finite input is not rejected: ReLU is pure math. NaN propagates,
/// +inf passes through, and -inf maps to 0
#[test]
fn relu_non_finite_input_propagates() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 3, vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    let out = layer
        .forward(&input)
        .expect("forward must not reject non-finite input");
    let v = out.as_slice().expect("contiguous");
    assert!(v[0].is_nan(), "NaN propagates through ReLU");
    assert!(v[1].is_infinite() && v[1] > 0.0, "+inf passes through");
    assert_eq!(v[2], 0.0, "-inf maps to 0");
}

// Sigmoid layer

/// sigmoid(x) = 1 / (1 + exp(-x)) on known values
#[test]
fn sigmoid_forward_known_values() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 3, vec![0.0, 2.0, -2.0]);
    let output = layer.forward(&input).expect("Sigmoid forward failed");

    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.5_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 0.880797_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(vals[2], 0.119203_f32, epsilon = 1e-5);
}

/// Outputs lie in [0, 1] (f32 saturates to exactly 0.0/1.0 at extreme inputs)
#[test]
fn sigmoid_forward_outputs_bounded() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 5, vec![-100.0, -1.0, 0.0, 1.0, 100.0]);
    let output = layer.forward(&input).expect("Sigmoid forward failed");
    for &v in output.iter() {
        assert!((0.0..=1.0).contains(&v), "sigmoid output {v} outside [0,1]");
    }
    // For moderate inputs the output is strictly interior
    let moderate = tensor2(1, 3, vec![-1.0, 0.0, 1.0]);
    for &v in layer.forward(&moderate).unwrap().iter() {
        assert!(
            v > 0.0 && v < 1.0,
            "sigmoid({v}) should be strictly in (0,1)"
        );
    }
}

/// sigmoid(x) + sigmoid(-x) == 1 (anti-symmetry identity)
#[test]
fn sigmoid_forward_antisymmetry() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 3, vec![0.5, 1.0, 3.0]);
    let out_pos = layer.forward(&input).expect("pos forward");
    let neg_input = tensor2(1, 3, vec![-0.5, -1.0, -3.0]);
    let out_neg = layer.forward(&neg_input).expect("neg forward");
    for (&p, &n) in out_pos.iter().zip(out_neg.iter()) {
        assert_abs_diff_eq!(p + n, 1.0_f32, epsilon = 1e-6);
    }
}

/// predict() equals forward()
#[test]
fn sigmoid_predict_equals_forward() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 3, vec![0.0, 2.0, -2.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn sigmoid_backward_before_forward_is_error() {
    let mut layer = Sigmoid::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Sigmoid")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// NaN input is not rejected: it propagates through (sigmoid(NaN) = NaN)
#[test]
fn sigmoid_nan_input_propagates() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 2, vec![f32::NAN, 1.0]);
    let out = layer
        .forward(&input)
        .expect("forward must not reject non-finite input");
    let v = out.as_slice().expect("contiguous");
    assert!(v[0].is_nan(), "NaN propagates through sigmoid");
    assert!(
        (v[1] - 0.7310586).abs() < 1e-6,
        "finite values are unaffected"
    );
}

// Tanh layer

/// tanh(x) = (e^x - e^-x) / (e^x + e^-x) on known values
#[test]
fn tanh_forward_known_values() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 4, vec![0.0, 1.0, -1.0, 2.0]);
    let output = layer.forward(&input).expect("Tanh forward failed");

    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 0.761594_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(vals[2], -0.761594_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(vals[3], 0.964028_f32, epsilon = 1e-5);
}

/// tanh is odd: tanh(-x) == -tanh(x)
#[test]
fn tanh_forward_odd_symmetry() {
    let mut layer = Tanh::new();
    let xs = vec![0.3, 1.5, 2.7];
    let pos_input = tensor2(1, 3, xs.clone());
    let neg_input = tensor2(1, 3, xs.iter().map(|&v| -v).collect());
    let out_pos = layer.forward(&pos_input).expect("pos forward");
    let out_neg = layer.forward(&neg_input).expect("neg forward");
    for (&p, &n) in out_pos.iter().zip(out_neg.iter()) {
        assert_abs_diff_eq!(p, -n, epsilon = 1e-6);
    }
}

/// Outputs lie in [-1, 1] (f32 saturates to exactly +/-1.0 at extreme inputs)
#[test]
fn tanh_forward_outputs_bounded() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 4, vec![-100.0, -1.0, 1.0, 100.0]);
    let output = layer.forward(&input).expect("Tanh forward failed");
    for &v in output.iter() {
        assert!((-1.0..=1.0).contains(&v), "tanh output {v} outside [-1, 1]");
    }
    // For moderate inputs the output is strictly interior
    let moderate = tensor2(1, 3, vec![-1.0, 0.0, 1.0]);
    for &v in layer.forward(&moderate).unwrap().iter() {
        assert!(
            v > -1.0 && v < 1.0,
            "tanh({v}) should be strictly in (-1,1)"
        );
    }
}

/// predict() equals forward()
#[test]
fn tanh_predict_equals_forward() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 4, vec![0.0, 1.0, -1.0, 2.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn tanh_backward_before_forward_is_error() {
    let mut layer = Tanh::new();
    let grad = tensor2(1, 4, vec![1.0; 4]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Tanh")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Non-finite input is not rejected: NaN propagates and tanh saturates +/-inf to +/-1
#[test]
fn tanh_non_finite_input_propagates() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 3, vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY]);
    let out = layer
        .forward(&input)
        .expect("forward must not reject non-finite input");
    let v = out.as_slice().expect("contiguous");
    assert!(v[0].is_nan(), "NaN propagates through tanh");
    assert_eq!(v[1], -1.0, "tanh(-inf) saturates to -1");
    assert_eq!(v[2], 1.0, "tanh(+inf) saturates to +1");
}

// Softmax layer

/// softmax([0,1,2]) ~= [0.09003, 0.24473, 0.66524]
#[test]
fn softmax_forward_known_values_1x3() {
    let mut layer = Softmax::new();
    let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
    let output = layer.forward(&input).expect("Softmax forward failed");

    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
}

/// Each row sums to 1.0 (probability simplex constraint)
#[test]
fn softmax_forward_rows_sum_to_one() {
    let mut layer = Softmax::new();
    let input = tensor2(2, 3, vec![0.0, 1.0, 2.0, -1.0, 0.5, 3.0]);
    let output = layer.forward(&input).expect("Softmax forward failed");
    let flat: Vec<f32> = output.iter().cloned().collect();
    let row0_sum = flat[0] + flat[1] + flat[2];
    let row1_sum = flat[3] + flat[4] + flat[5];
    assert_abs_diff_eq!(row0_sum, 1.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(row1_sum, 1.0_f32, epsilon = 1e-6);
}

/// Shift-invariance: adding a constant to every logit leaves the output unchanged
#[test]
fn softmax_forward_shift_invariant() {
    let mut layer = Softmax::new();
    let base_input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
    let shifted_input = tensor2(1, 3, vec![100.0, 101.0, 102.0]);

    let base_out = layer.forward(&base_input).expect("base forward");
    let shifted_out = layer.forward(&shifted_input).expect("shifted forward");

    assert_allclose(&shifted_out, &base_out, 1e-5_f32);
}

/// Equal logits give uniform probability: softmax([5,5,5]) = [1/3, 1/3, 1/3]
#[test]
fn softmax_forward_equal_logits_uniform() {
    let mut layer = Softmax::new();
    let input = tensor2(1, 3, vec![5.0, 5.0, 5.0]);
    let output = layer.forward(&input).expect("Softmax forward failed");
    let third = 1.0_f32 / 3.0;
    for &v in output.iter() {
        assert_abs_diff_eq!(v, third, epsilon = 1e-6);
    }
}

/// Rows with the same logit difference produce the same probabilities
#[test]
fn softmax_forward_two_rows_same_difference() {
    let mut layer = Softmax::new();
    let input = tensor2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let output = layer.forward(&input).expect("Softmax forward failed");

    let flat: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(flat[0], 0.26894_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(flat[1], 0.73106_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(flat[2], 0.26894_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(flat[3], 0.73106_f32, epsilon = 1e-4);
}

/// predict() equals forward()
#[test]
fn softmax_predict_equals_forward() {
    let mut layer = Softmax::new();
    let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn softmax_backward_before_forward_is_error() {
    let mut layer = Softmax::new();
    let grad = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Softmax")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// 1-D input -> InvalidInput error (Softmax requires at least 2 dimensions)
#[test]
fn softmax_1d_input_is_error() {
    let mut layer = Softmax::new();
    let input = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}

/// NaN input is not rejected: it contaminates the row's normalizer, so the whole row is NaN
#[test]
fn softmax_nan_input_propagates() {
    let mut layer = Softmax::new();
    let input = tensor2(1, 3, vec![1.0, f32::NAN, 2.0]);
    let out = layer
        .forward(&input)
        .expect("forward must not reject non-finite input");
    assert!(
        out.iter().any(|v| v.is_nan()),
        "a NaN in the row propagates into the softmax output"
    );
}

// Linear layer

/// Linear is the identity: output == input exactly
#[test]
fn linear_forward_is_identity() {
    let mut layer = Linear::new();
    let data = vec![-3.0_f32, -1.5, 0.0, 1.0, 4.7, 100.0];
    let input = tensor2(2, 3, data.clone());
    let output = layer.forward(&input).expect("Linear forward failed");
    let expected = tensor2(2, 3, data);
    assert_allclose(&output, &expected, 0.0_f32);
}

/// Linear preserves arbitrary shapes
#[test]
fn linear_forward_preserves_3d_shape() {
    let mut layer = Linear::new();
    let data: Vec<f32> = (0..24).map(|v| v as f32 * 0.5 - 6.0).collect();
    let input = Array::from_shape_vec((2, 3, 4), data.clone())
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).expect("Linear 3D forward failed");
    assert_eq!(output.shape(), input.shape(), "shape should be preserved");
    let expected = Array::from_shape_vec((2, 3, 4), data).unwrap().into_dyn();
    assert_allclose(&output, &expected, 0.0_f32);
}

/// predict() equals forward() (both return the input unchanged)
#[test]
fn linear_predict_equals_forward() {
    let mut layer = Linear::new();
    let input = tensor2(2, 3, vec![-3.0, -1.5, 0.0, 1.0, 4.7, 100.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 0.0_f32);
}

/// Linear backward passes the upstream gradient through unchanged (derivative is 1)
#[test]
fn linear_backward_passes_gradient_through() {
    let mut layer = Linear::new();
    let input = tensor2(1, 3, vec![1.0, 2.0, 3.0]);
    layer.forward(&input).expect("forward");

    let grad = tensor2(1, 3, vec![0.1, -0.2, 0.5]);
    let grad_in = layer.backward(&grad).expect("backward");
    assert_allclose(&grad_in, &grad, 0.0_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn linear_backward_before_forward_is_error() {
    let mut layer = Linear::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Linear")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// Non-finite input is not rejected: Linear is the identity, so NaN and Inf pass straight through
#[test]
fn linear_non_finite_input_propagates() {
    let mut layer = Linear::new();
    let input = tensor2(1, 2, vec![f32::NAN, f32::NEG_INFINITY]);
    let out = layer
        .forward(&input)
        .expect("forward must not reject non-finite input");
    let v = out.as_slice().expect("contiguous");
    assert!(
        v[0].is_nan() && v[1] == f32::NEG_INFINITY,
        "identity passes values through"
    );
}

/// predict() likewise passes non-finite values through unchanged
#[test]
fn linear_predict_non_finite_propagates() {
    let layer = Linear::new();
    let input = tensor2(1, 2, vec![f32::NAN, 1.0]);
    let out = layer
        .predict(&input)
        .expect("predict must not reject non-finite input");
    assert!(out.as_slice().expect("contiguous")[0].is_nan());
}

// LeakyReLU layer.
// The pinned forward values of the 7 sections below are reference values in float32.

/// LeakyReLU(x) = x for x >= 0, and negative_slope * x below 0, on known values
#[test]
fn leaky_relu_forward_known_values() {
    let mut layer = LeakyReLU::new(0.3).expect("slope 0.3 is valid");
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("LeakyReLU forward failed");

    let expected = tensor2(2, 3, vec![-0.30000001, 2.0, -0.90000004, 4.0, -1.5, 6.0]);
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// predict() equals forward()
#[test]
fn leaky_relu_predict_equals_forward() {
    let mut layer = LeakyReLU::new(0.3).expect("slope 0.3 is valid");
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn leaky_relu_backward_before_forward_is_error() {
    let mut layer = LeakyReLU::new(0.3).expect("slope 0.3 is valid");
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(
                "LeakyReLU"
            )))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// LeakyReLU takes the positive branch at exactly 0, so its derivative there is 1, not the slope
#[test]
fn leaky_relu_backward_derivative_at_zero_is_one() {
    let mut layer = LeakyReLU::new(0.3).expect("slope 0.3 is valid");
    let input = tensor2(1, 3, vec![-2.0, 0.0, 2.0]);
    layer.forward(&input).expect("LeakyReLU forward");

    let grad_output = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("LeakyReLU backward");

    let expected = tensor2(1, 3, vec![0.3, 1.0, 1.0]);
    assert_allclose(&grad_in, &expected, 1e-6_f32);
}

// ELU layer

/// ELU(x) = x for x > 0, and alpha * (e^x - 1) at 0 and below, on known values
#[test]
fn elu_forward_known_values() {
    let mut layer = ELU::new(1.0).expect("alpha 1.0 is valid");
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("ELU forward failed");

    let expected = tensor2(
        2,
        3,
        vec![-0.63212055, 2.0, -0.95021296, 4.0, -0.99326205, 6.0],
    );
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// predict() equals forward()
#[test]
fn elu_predict_equals_forward() {
    let mut layer = ELU::new(1.0).expect("alpha 1.0 is valid");
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn elu_backward_before_forward_is_error() {
    let mut layer = ELU::new(1.0).expect("alpha 1.0 is valid");
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("ELU")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// ELU takes the negative branch at exactly 0, so its derivative there is alpha, not 1
#[test]
fn elu_backward_derivative_at_zero_is_alpha() {
    let mut layer = ELU::new(0.5).expect("alpha 0.5 is valid");
    let input = tensor2(1, 2, vec![0.0, 3.0]);
    layer.forward(&input).expect("ELU forward");

    let grad_output = tensor2(1, 2, vec![1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("ELU backward");

    let expected = tensor2(1, 2, vec![0.5, 1.0]);
    assert_allclose(&grad_in, &expected, 1e-6_f32);
}

// SELU layer.
// The 2 constants are fixed at alpha = 1.6732632 and scale = 1.0507010, so their product is
// 1.7580993.

/// SELU(x) = scale * x for x > 0, and scale * alpha * (e^x - 1) at 0 and below, on known values
#[test]
fn selu_forward_known_values() {
    let mut layer = SELU::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("SELU forward failed");

    let expected = tensor2(
        2,
        3,
        vec![
            -1.1113307, 2.101402, -1.6705688, 4.2028041, -1.7462534, 6.3042059,
        ],
    );
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// predict() equals forward()
#[test]
fn selu_predict_equals_forward() {
    let mut layer = SELU::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn selu_backward_before_forward_is_error() {
    let mut layer = SELU::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("SELU")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// SELU takes the negative branch at exactly 0, so its derivative there is scale * alpha
#[test]
fn selu_backward_derivative_at_zero_is_scale_times_alpha() {
    let mut layer = SELU::new();
    let input = tensor2(1, 2, vec![0.0, 3.0]);
    layer.forward(&input).expect("SELU forward");

    let grad_output = tensor2(1, 2, vec![1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("SELU backward");

    let expected = tensor2(1, 2, vec![1.7580993, 1.0507010]);
    assert_allclose(&grad_in, &expected, 1e-6_f32);
}

// Softplus layer

/// Softplus(x) = ln(1 + e^x) on known values
#[test]
fn softplus_forward_known_values() {
    let mut layer = Softplus::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("Softplus forward failed");

    let expected = tensor2(
        2,
        3,
        vec![
            0.31326169,
            2.1269281,
            0.048587352,
            4.0181499,
            0.0067153485,
            6.0024757,
        ],
    );
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// Outputs stay strictly positive, even far down the negative tail
#[test]
fn softplus_forward_outputs_positive() {
    let mut layer = Softplus::new();
    let input = tensor2(1, 4, vec![-30.0, -5.0, 0.0, 5.0]);
    let output = layer.forward(&input).expect("Softplus forward failed");
    for &v in output.iter() {
        assert!(v > 0.0, "softplus output {v} should be strictly positive");
    }
}

/// predict() equals forward()
#[test]
fn softplus_predict_equals_forward() {
    let mut layer = Softplus::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn softplus_backward_before_forward_is_error() {
    let mut layer = Softplus::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Softplus")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

// Softsign layer

/// Softsign(x) = x / (1 + |x|) on known values
#[test]
fn softsign_forward_known_values() {
    let mut layer = Softsign::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("Softsign forward failed");

    let expected = tensor2(
        2,
        3,
        vec![-0.5, 0.66666669, -0.75, 0.80000001, -0.83333331, 0.85714287],
    );
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// Outputs stay strictly inside (-1, 1), because the saturation is polynomial
#[test]
fn softsign_forward_outputs_bounded() {
    let mut layer = Softsign::new();
    let input = tensor2(1, 5, vec![-1000.0, -1.0, 0.0, 1.0, 1000.0]);
    let output = layer.forward(&input).expect("Softsign forward failed");
    for &v in output.iter() {
        assert!(
            v > -1.0 && v < 1.0,
            "softsign output {v} should be strictly in (-1, 1)"
        );
    }
}

/// predict() equals forward()
#[test]
fn softsign_predict_equals_forward() {
    let mut layer = Softsign::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn softsign_backward_before_forward_is_error() {
    let mut layer = Softsign::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun("Softsign")))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

// HardSigmoid layer

/// HardSigmoid(x) = clip(x/6 + 0.5, 0, 1) on known values
#[test]
fn hard_sigmoid_forward_known_values() {
    let mut layer = HardSigmoid::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("HardSigmoid forward failed");

    let expected = tensor2(2, 3, vec![0.33333334, 0.83333337, 0.0, 1.0, 0.0, 1.0]);
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// predict() equals forward()
#[test]
fn hard_sigmoid_predict_equals_forward() {
    let mut layer = HardSigmoid::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-7_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn hard_sigmoid_backward_before_forward_is_error() {
    let mut layer = HardSigmoid::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(
                "HardSigmoid"
            )))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

// Exponential layer.
// The pinned values reach 403, where 1 float32 unit in the last place is already about 3e-5,
// so this section uses a tolerance of 1e-3.

/// Exponential(x) = e^x on known values
#[test]
fn exponential_forward_known_values() {
    let mut layer = Exponential::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let output = layer.forward(&input).expect("Exponential forward failed");

    let expected = tensor2(
        2,
        3,
        vec![
            0.36787945,
            7.3890562,
            0.049787067,
            54.598148,
            0.006737947,
            403.4288,
        ],
    );
    assert_allclose(&output, &expected, 1e-3_f32);
}

/// predict() equals forward()
#[test]
fn exponential_predict_equals_forward() {
    let mut layer = Exponential::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let fwd = layer.forward(&input).expect("forward");
    let pred = layer.predict(&input).expect("predict");
    assert_allclose(&pred, &fwd, 1e-3_f32);
}

/// backward before forward returns NnError::ForwardPassNotRun
#[test]
fn exponential_backward_before_forward_is_error() {
    let mut layer = Exponential::new();
    let grad = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(
                "Exponential"
            )))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// The exponential is its own derivative, so backward with an all-ones upstream gradient
/// returns the forward output
#[test]
fn exponential_backward_derivative_equals_output() {
    let mut layer = Exponential::new();
    let input = tensor2(1, 3, vec![-1.0, 0.0, 2.0]);
    let output = layer.forward(&input).expect("Exponential forward");

    let grad_output = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("Exponential backward");

    assert_allclose(&grad_in, &output, 1e-6_f32);
}

// Activation enum: forward() delegates to the standalone layers, so its output
// must match theirs

/// Activation::Linear.forward is identity
#[test]
fn activation_enum_linear_is_identity() {
    let input = tensor2(1, 3, vec![-1.0, 0.0, 2.0]);
    let output = Activation::Linear
        .forward(&input)
        .expect("enum linear forward");
    assert_allclose(&output, &input, 0.0_f32);
}

/// Activation::ReLU.forward: ReLU(-2,0,3) = (0,0,3)
#[test]
fn activation_enum_relu_known_values() {
    let input = tensor2(1, 3, vec![-2.0, 0.0, 3.0]);
    let output = Activation::ReLU.forward(&input).expect("enum relu forward");
    let expected = tensor2(1, 3, vec![0.0, 0.0, 3.0]);
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// Activation::Sigmoid.forward: sigmoid(0)=0.5, sigmoid(2)~=0.880797
#[test]
fn activation_enum_sigmoid_known_values() {
    let input = tensor2(1, 2, vec![0.0, 2.0]);
    let output = Activation::Sigmoid
        .forward(&input)
        .expect("enum sigmoid forward");
    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.5_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 0.880797_f32, epsilon = 1e-5);
}

/// Activation::Tanh.forward: tanh(0)=0, tanh(1)~=0.761594
#[test]
fn activation_enum_tanh_known_values() {
    let input = tensor2(1, 2, vec![0.0, 1.0]);
    let output = Activation::Tanh.forward(&input).expect("enum tanh forward");
    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 0.761594_f32, epsilon = 1e-5);
}

/// Activation::Softmax.forward: softmax([0,1,2]) ~= [0.0900, 0.2447, 0.6652], row sums to 1
#[test]
fn activation_enum_softmax_known_values() {
    let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
    let output = Activation::Softmax
        .forward(&input)
        .expect("enum softmax forward");
    let vals: Vec<f32> = output.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
    let sum: f32 = vals.iter().sum();
    assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
}

/// Activation::Softmax.forward rejects 1-D input (ndim < 2)
#[test]
fn activation_enum_softmax_rejects_1d() {
    let input = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
    let result = Activation::Softmax.forward(&input);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        result
    );
}

// From<Layer> -> Activation conversions (confirms the enum round-trips)

/// From<Linear> for Activation yields Activation::Linear
#[test]
fn from_linear_yields_activation_linear() {
    let act: Activation = Linear::new().into();
    assert_eq!(act, Activation::Linear);
}

/// From<ReLU> for Activation yields Activation::ReLU
#[test]
fn from_relu_yields_activation_relu() {
    let act: Activation = ReLU::new().into();
    assert_eq!(act, Activation::ReLU);
}

/// From<Sigmoid> for Activation yields Activation::Sigmoid
#[test]
fn from_sigmoid_yields_activation_sigmoid() {
    let act: Activation = Sigmoid::new().into();
    assert_eq!(act, Activation::Sigmoid);
}

/// From<Tanh> for Activation yields Activation::Tanh
#[test]
fn from_tanh_yields_activation_tanh() {
    let act: Activation = Tanh::new().into();
    assert_eq!(act, Activation::Tanh);
}

/// From<Softmax> for Activation yields Activation::Softmax
#[test]
fn from_softmax_yields_activation_softmax() {
    let act: Activation = Softmax::new().into();
    assert_eq!(act, Activation::Softmax);
}

// All 5 activation layers reject 0-length input in forward() and predict().
// The empty check runs first, so a 0-element tensor yields Error::EmptyInput.

/// Fresh boxed instances of all 5 activation layers, in a stable order
fn all_activation_layers() -> Vec<(&'static str, Box<dyn Layer>)> {
    vec![
        ("ReLU", Box::new(ReLU::new())),
        ("Sigmoid", Box::new(Sigmoid::new())),
        ("Tanh", Box::new(Tanh::new())),
        ("Softmax", Box::new(Softmax::new())),
        ("Linear", Box::new(Linear::new())),
    ]
}

/// forward() on a 0-length input gives Error::EmptyInput for every layer
#[test]
fn all_layers_empty_input_forward_is_error() {
    // 2-D tensor with 0 rows has 0 elements, so it is empty
    let empty = Array2::<f32>::zeros((0, 3)).into_dyn();
    for (name, mut layer) in all_activation_layers() {
        let result = layer.forward(&empty);
        assert!(
            matches!(result, Err(Error::EmptyInput(_))),
            "{name}: expected EmptyInput from forward, got {:?}",
            result
        );
    }
}

/// predict() on a 0-length input gives Error::EmptyInput for every layer
#[test]
fn all_layers_empty_input_predict_is_error() {
    let empty = Array2::<f32>::zeros((0, 3)).into_dyn();
    for (name, layer) in all_activation_layers() {
        let result = layer.predict(&empty);
        assert!(
            matches!(result, Err(Error::EmptyInput(_))),
            "{name}: expected EmptyInput from predict, got {:?}",
            result
        );
    }
}

// All 5 activation layers check only the structural shape of grad_output in backward().
// A wrong shape gives ShapeMismatch. They do not sanitize values: a non-finite grad_output
// flows through instead, surfacing downstream in the next forward pass or as a NaN loss.

/// Wrong-shaped grad_output after a valid forward gives Error::ShapeMismatch
#[test]
fn all_layers_backward_wrong_shape_is_shape_mismatch() {
    let input = tensor2(1, 3, vec![1.0, 2.0, 3.0]); // valid for all 5 layers (2D)
    // grad with a different shape (1x2 instead of the cached 1x3)
    let bad_grad = tensor2(1, 2, vec![1.0, 1.0]);
    for (name, mut layer) in all_activation_layers() {
        layer.forward(&input).expect("valid forward should succeed");
        let result = layer.backward(&bad_grad);
        assert!(
            matches!(result, Err(Error::ShapeMismatch { .. })),
            "{name}: expected ShapeMismatch from backward, got {:?}",
            result
        );
    }
}

/// Non-finite grad_output (correct shape) is not rejected: backward is pure math, so it returns Ok.
/// The result carries the non-finite value forward, instead of masking it or raising an error
#[test]
fn all_layers_backward_propagates_non_finite_grad() {
    let input = tensor2(1, 3, vec![1.0, 2.0, 3.0]);
    let nan_grad = tensor2(1, 3, vec![1.0, f32::NAN, 1.0]);
    for (name, mut layer) in all_activation_layers() {
        layer.forward(&input).expect("valid forward should succeed");
        let grad_in = layer.backward(&nan_grad).unwrap_or_else(|e| {
            panic!("{name}: backward should propagate, not error, got {:?}", e)
        });
        assert!(
            grad_in.iter().any(|&v| !v.is_finite()),
            "{name}: expected a non-finite value to propagate through backward, got {:?}",
            grad_in
        );
    }
}

// ReLU derivative: ReLU'(x) = 1 for x > 0 and 0 for x <= 0.
// backward passes grad_output through where the input was positive, and zeroes it elsewhere.

/// ReLU backward passes the gradient through positive inputs and blocks the rest
#[test]
fn relu_backward_derivative_from_definition() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 4, vec![2.0, -3.0, 0.5, -1.0]);
    layer.forward(&input).expect("ReLU forward");

    let grad_output = tensor2(1, 4, vec![0.1, 0.2, 0.3, 0.4]);
    let grad_in = layer.backward(&grad_output).expect("ReLU backward");

    let expected = tensor2(1, 4, vec![0.1, 0.0, 0.3, 0.0]);
    assert_allclose(&grad_in, &expected, 1e-6_f32);
}

// Sigmoid derivative: sigmoid'(x) = a * (1 - a) where a = sigmoid(x).
// backward multiplies grad_output element-wise by a*(1-a).

/// Sigmoid backward with all-ones grad_output equals a*(1-a) element-wise
#[test]
fn sigmoid_backward_derivative_from_definition() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 3, vec![0.0, 2.0, -2.0]);
    layer.forward(&input).expect("Sigmoid forward");

    let grad_output = tensor2(1, 3, vec![1.0, 1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("Sigmoid backward");

    let vals: Vec<f32> = grad_in.iter().cloned().collect();
    assert_abs_diff_eq!(vals[0], 0.25_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(vals[1], 0.1049936_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(vals[2], 0.1049936_f32, epsilon = 1e-5);
}

// Parameterized activations: bounds, defaults, and agreement with the enum.
// The lower bound on `negative_slope` and on `alpha` is strict. The backward pass separates
// the 2 branches by the sign of the activated output. A value of 0 collapses the whole
// negative side onto a = 0, which erases the branch. A negative value reads the wrong branch.
// Use Activation::ReLU for a slope of 0.

/// LeakyReLU::new rejects a slope of 0 and a negative slope
#[test]
fn leaky_relu_new_rejects_unusable_slope() {
    for slope in [0.0_f32, -0.1] {
        let result = LeakyReLU::new(slope);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "slope {slope}: expected InvalidParameter, got {:?}",
            result
        );
    }
}

/// ELU::new rejects an alpha of 0 and a negative alpha
#[test]
fn elu_new_rejects_unusable_alpha() {
    for alpha in [0.0_f32, -1.0] {
        let result = ELU::new(alpha);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "alpha {alpha}: expected InvalidParameter, got {:?}",
            result
        );
    }
}

/// LeakyReLU::default() uses the default slope of 0.3, so it matches LeakyReLU::new(0.3)
#[test]
fn leaky_relu_default_matches_explicit_slope() {
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let default_out = LeakyReLU::default()
        .forward(&input)
        .expect("default forward");
    let explicit_out = LeakyReLU::new(0.3)
        .expect("slope 0.3 is valid")
        .forward(&input)
        .expect("explicit forward");
    assert_allclose(&default_out, &explicit_out, 0.0_f32);
}

/// ELU::default() uses the default alpha of 1.0, so it matches ELU::new(1.0)
#[test]
fn elu_default_matches_explicit_alpha() {
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let default_out = ELU::default().forward(&input).expect("default forward");
    let explicit_out = ELU::new(1.0)
        .expect("alpha 1.0 is valid")
        .forward(&input)
        .expect("explicit forward");
    assert_allclose(&default_out, &explicit_out, 0.0_f32);
}

/// A trainable layer rejects an unusable LeakyReLU slope where the model is built
#[test]
fn dense_rejects_unusable_leaky_relu_slope() {
    let result = Dense::new(
        2,
        2,
        Activation::LeakyReLU {
            negative_slope: -0.1,
        },
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// A trainable layer rejects an unusable ELU alpha where the model is built
#[test]
fn dense_rejects_unusable_elu_alpha() {
    let result = Dense::new(2, 2, Activation::ELU { alpha: 0.0 });
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// HardSigmoid saturates to exactly 0 and 1, and its derivative on both saturated ends is 0
#[test]
fn hard_sigmoid_saturates_exactly() {
    let mut layer = HardSigmoid::new();
    let input = tensor2(1, 4, vec![-3.0, 3.0, -10.0, 10.0]);
    let output = layer.forward(&input).expect("HardSigmoid forward");

    let expected = tensor2(1, 4, vec![0.0, 1.0, 0.0, 1.0]);
    assert_allclose(&output, &expected, 0.0_f32);

    let grad_output = tensor2(1, 4, vec![1.0, 1.0, 1.0, 1.0]);
    let grad_in = layer.backward(&grad_output).expect("HardSigmoid backward");

    let expected_grad = tensor2(1, 4, vec![0.0, 0.0, 0.0, 0.0]);
    assert_allclose(&grad_in, &expected_grad, 0.0_f32);
}

/// The SELU layer is a thin wrapper, so its forward equals Activation::SELU.forward exactly
#[test]
fn selu_layer_matches_activation_enum() {
    let mut layer = SELU::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let layer_out = layer.forward(&input).expect("SELU layer forward");
    let enum_out = Activation::SELU.forward(&input).expect("enum SELU forward");
    assert_allclose(&layer_out, &enum_out, 0.0_f32);
}

/// The Softplus layer is a thin wrapper, so its forward equals Activation::Softplus.forward
#[test]
fn softplus_layer_matches_activation_enum() {
    let mut layer = Softplus::new();
    let input = tensor2(2, 3, vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
    let layer_out = layer.forward(&input).expect("Softplus layer forward");
    let enum_out = Activation::Softplus
        .forward(&input)
        .expect("enum Softplus forward");
    assert_allclose(&layer_out, &enum_out, 0.0_f32);
}

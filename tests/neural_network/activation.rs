//! Integration tests for the activation layers and the `Activation` enum
//!
//! Expected values are derived from the mathematical definitions. Gradient
//! correctness is covered by tests/neural_network/gradient_check.rs, so this
//! file does not duplicate those backward-value checks. Coverage:
//!   - forward values, predict() == forward()
//!   - backward before forward -> NnError::ForwardPassNotRun
//!   - NaN/Inf input -> NonFinite, empty input -> EmptyInput
//!   - Activation enum forward delegation, and the Linear layer

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::sigmoid::Sigmoid;
use rustyml::neural_network::layers::activation::softmax::Softmax;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::traits::Layer;

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

/// NaN in input -> NonFinite error
#[test]
fn relu_nan_input_is_error() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 2, vec![1.0, f32::NAN]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// Inf in input -> NonFinite error
#[test]
fn relu_inf_input_is_error() {
    let mut layer = ReLU::new();
    let input = tensor2(1, 2, vec![1.0, f32::INFINITY]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
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

/// NaN in input -> NonFinite
#[test]
fn sigmoid_nan_input_is_error() {
    let mut layer = Sigmoid::new();
    let input = tensor2(1, 2, vec![f32::NAN, 1.0]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
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

/// NaN in input -> NonFinite
#[test]
fn tanh_nan_input_is_error() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 2, vec![0.5, f32::NAN]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// -Inf in input -> NonFinite
#[test]
fn tanh_neg_inf_input_is_error() {
    let mut layer = Tanh::new();
    let input = tensor2(1, 2, vec![f32::NEG_INFINITY, 0.0]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
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

/// NaN in input -> NonFinite
#[test]
fn softmax_nan_input_is_error() {
    let mut layer = Softmax::new();
    let input = tensor2(1, 3, vec![1.0, f32::NAN, 2.0]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
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

/// NaN in input -> NonFinite
#[test]
fn linear_nan_input_is_error() {
    let mut layer = Linear::new();
    let input = tensor2(1, 2, vec![f32::NAN, 1.0]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// Inf in input -> NonFinite
#[test]
fn linear_inf_input_is_error() {
    let mut layer = Linear::new();
    let input = tensor2(1, 2, vec![1.0, f32::NEG_INFINITY]);
    let result = layer.forward(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite, got {:?}",
        result
    );
}

/// predict() also rejects NaN
#[test]
fn linear_predict_nan_is_error() {
    let layer = Linear::new();
    let input = tensor2(1, 2, vec![f32::NAN, 1.0]);
    let result = layer.predict(&input);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "expected NonFinite in predict, got {:?}",
        result
    );
}

// Activation enum - forward delegates to the standalone layers, so the enum's
// forward() must match the same values

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

// From<Layer> -> Activation conversions (ensures the enum round-trips)

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

// All five activation layers reject 0-length input in forward and predict; the
// empty check runs first, so a zero-element tensor yields Error::EmptyInput

/// Fresh boxed instances of all five activation layers, in a stable order
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

// All five activation layers validate only the structural shape of grad_output in backward
// (wrong shape gives ShapeMismatch); they do NOT sanitize values. A non-finite grad_output is
// pure-math propagated, surfacing downstream (next forward / NaN loss) rather than being rejected here

/// Wrong-shaped grad_output after a valid forward gives Error::ShapeMismatch
#[test]
fn all_layers_backward_wrong_shape_is_shape_mismatch() {
    let input = tensor2(1, 3, vec![1.0, 2.0, 3.0]); // valid for all five (2-D)
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

/// Non-finite grad_output (correct shape) is propagated, not rejected: backward is pure math, so it
/// returns Ok and the non-finite value flows into the result instead of being masked or erroring
#[test]
fn all_layers_backward_propagates_non_finite_grad() {
    let input = tensor2(1, 3, vec![1.0, 2.0, 3.0]);
    let nan_grad = tensor2(1, 3, vec![1.0, f32::NAN, 1.0]);
    for (name, mut layer) in all_activation_layers() {
        layer.forward(&input).expect("valid forward should succeed");
        let grad_in = layer
            .backward(&nan_grad)
            .unwrap_or_else(|e| panic!("{name}: backward should propagate, not error, got {:?}", e));
        assert!(
            grad_in.iter().any(|&v| !v.is_finite()),
            "{name}: expected a non-finite value to propagate through backward, got {:?}",
            grad_in
        );
    }
}

// ReLU derivative: ReLU'(x) = 1 for x > 0 and 0 for x <= 0, so backward passes
// grad_output through where input was > 0 and zeroes it elsewhere

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

// Sigmoid derivative: sigmoid'(x) = a * (1 - a) where a = sigmoid(x), so backward
// multiplies grad_output element-wise by a*(1-a)

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

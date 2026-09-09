//! Integration tests for the recurrent layers: SimpleRNN, LSTM, GRU
//!
//! Expected values are hand-computed from the mathematical definitions. Backward/gradient values
//! are covered by tests/neural_network/gradient_check.rs and are not duplicated here, except for
//! the `return_sequences` and `go_backwards` gradients at the end of this file. Those 2 flags
//! change the backward pass itself. The finite-difference harness of `gradient_check.rs` drives
//! every layer with a constant upstream gradient, which hides 2 of their failure modes

use crate::common::assert_allclose;
use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2};
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::traits::{Layer, LayerBase, ParamId, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

/// The gradient of 1 named parameter of the driven layer, as a flat slice
///
/// A test drives 1 layer and sets no owner, so every gradient of the pass lands at
/// position 0 of the gradient store.
fn grad_of<'a>(ctx: &'a Ctx, name: &'static str) -> &'a [f32] {
    ctx.grads()
        .get(ParamId::new(0, name))
        .unwrap_or_else(|| panic!("the backward pass must give `{name}` a gradient"))
        .as_slice()
        .expect("a gradient of the store is in the standard memory order")
}

// SimpleRNN

/// SimpleRNN forward over 1 timestep, 1 unit with Tanh yields tanh(0.5)
#[test]
fn simple_rnn_forward_1step_1unit_tanh() {
    let mut rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[1, 1, 1])).unwrap();

    let kernel = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::from_elem((1, 1), 0.5_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

#[test]
fn simple_rnn_forward_2step_tanh_state_threading() {
    let mut rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[1, 2, 1])).unwrap();

    let kernel = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::from_elem((1, 1), 1.0_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_shape_vec((1, 2, 1), vec![0.3_f32, 0.7_f32])
        .unwrap()
        .into_dyn();
    let out = rnn.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.75792147_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN with ReLU passes a positive pre-activation through unchanged
#[test]
fn simple_rnn_forward_relu_positive() {
    let mut rnn = SimpleRNN::new(1, ReLU::new()).unwrap();
    rnn.build(&Shape::known(&[1, 1, 1])).unwrap();

    let kernel = Array2::from_elem((1, 1), 0.5_f32);
    let rk = Array2::from_elem((1, 1), 0.5_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = rnn.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);
    let expected = Array::from_elem((1, 1), 0.5_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN with ReLU clips a negative pre-activation to exactly 0
#[test]
fn simple_rnn_relu_negative_preactivation_is_zero() {
    let mut rnn = SimpleRNN::new(1, ReLU::new()).unwrap();
    rnn.build(&Shape::known(&[1, 1, 1])).unwrap();

    let kernel = Array2::from_elem((1, 1), -1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.8_f32).into_dyn();
    let out = rnn.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);
    let expected = Array::zeros((1, 1)).into_dyn();
    assert_allclose(&out, &expected, 1e-7);
}

/// SimpleRNN output shape is (batch, units) regardless of timestep count
#[test]
fn simple_rnn_output_shape_batch3_units2() {
    let mut rnn = SimpleRNN::new(2, Tanh::new()).unwrap();
    // 3 batches, 5 timesteps, 4 input features
    let x = Array::zeros((3, 5, 4)).into_dyn();
    let out = rnn.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
}

/// SimpleRNN gives the same values in an inference pass as in a training pass
#[test]
fn simple_rnn_predict_equals_forward() {
    let mut rnn = SimpleRNN::new(3, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[2, 3, 2])).unwrap();

    let kernel = Array2::from_shape_vec((2, 3), vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).unwrap();
    let rk = Array2::from_shape_vec((3, 3), vec![0.1, 0.0, -0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.1])
        .unwrap();
    let bias = Array2::from_shape_vec((1, 3), vec![0.1, -0.1, 0.2]).unwrap();
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_shape_vec(
        (2, 3, 2),
        vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
        ],
    )
    .unwrap()
    .into_dyn();

    let out_training = rnn.forward(&x, &mut Ctx::training()).unwrap();
    let out_inference = rnn.forward(&x, &mut Ctx::inference()).unwrap();

    assert_eq!(out_training.shape(), out_inference.shape());
    assert_allclose(&out_training, &out_inference, 1e-6);
}

/// SimpleRNN constructor rejects a zero dimension, whichever argument carries it
#[test]
fn simple_rnn_new_rejects_zero_dimension() {
    let err = SimpleRNN::new(0, Tanh::new()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero units, got: {err:?}"
    );
}

#[test]
fn simple_rnn_build_rejects_zero_input_dim() {
    let mut rnn = SimpleRNN::new(3, Tanh::new()).unwrap();
    let err = rnn.build(&Shape::known(&[1, 1, 0])).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero input_dim, got: {err:?}"
    );
}

/// SimpleRNN forward rejects a non-3D input (2D tensor)
#[test]
fn simple_rnn_forward_rejects_2d_input() {
    let mut rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn(); // 2D, not 3D
    let err = rnn.forward_mut(&x, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

#[test]
fn simple_rnn_forward_rejects_1d_input() {
    let mut rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    let x = Array::zeros(4).into_dyn(); // 1D
    let err = rnn.forward_mut(&x, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// SimpleRNN backward before forward returns ForwardPassNotRun
#[test]
fn simple_rnn_backward_before_forward_errors() {
    let rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = rnn.backward(&grad, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SimpleRNN"))
        ),
        "expected ForwardPassNotRun(\"SimpleRNN\"), got: {err:?}"
    );
}

/// SimpleRNN backward with units == 1 and input_dim > 1 must not panic on the grad_x reshape
///
/// `dot` can return a column-major grad_x here, and the reshape must tolerate it.
#[test]
fn simple_rnn_backward_units_one_multi_feature_reshapes() {
    let mut rnn = SimpleRNN::new(1, Tanh::new()).unwrap();
    let x = Array::from_elem((1, 2, 2), 0.5_f32).into_dyn();
    let mut ctx = Ctx::training();
    rnn.forward_mut(&x, &mut ctx).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let grad_x = rnn.backward(&grad, &mut ctx).unwrap();
    assert_eq!(grad_x.shape(), &[1, 2, 2]);
}

#[test]
fn simple_rnn_set_weights_wrong_kernel_shape_errors() {
    let mut rnn = SimpleRNN::new(3, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[1, 1, 3])).unwrap();
    // kernel should be (2,3). (3,2) is passed instead.
    let bad_kernel = Array2::zeros((3, 2));
    let rk = Array2::zeros((3, 3));
    let bias = Array2::zeros((1, 3));
    let err = rnn.set_weights(bad_kernel, rk, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got: {err:?}"
    );
}

// LSTM

/// LSTM forward over 1 timestep, 1 unit with Tanh and forget bias = 1.0
#[test]
fn lstm_forward_1step_1unit_tanh() {
    let mut lstm = LSTM::new(1, Tanh::new()).unwrap();
    lstm.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32); // forget bias = 1.0

    lstm.set_gate_weights(
        k.clone(),
        rk.clone(),
        b_zero.clone(), // input gate
        k.clone(),
        rk.clone(),
        b_one.clone(), // forget gate
        k.clone(),
        rk.clone(),
        b_zero.clone(), // cell gate
        k.clone(),
        rk.clone(),
        b_zero.clone(), // output gate
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = lstm.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.17426972_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// Forget-gate bias of 1.0 yields a different final state than 0.0 across timesteps
#[test]
fn lstm_forget_bias_is_one_not_zero() {
    // 2 LSTMs: 1 with forget_bias=1.0 (the default), 1 with 0.0
    let mut lstm_correct = LSTM::new(1, Tanh::new()).unwrap();
    lstm_correct.build(&Shape::known(&[1, 1, 1])).unwrap();
    let mut lstm_zero_forget = LSTM::new(1, Tanh::new()).unwrap();
    lstm_zero_forget.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm_correct
        .set_gate_weights(
            k.clone(),
            rk.clone(),
            b_zero.clone(),
            k.clone(),
            rk.clone(),
            b_one.clone(), // forget bias = 1.0
            k.clone(),
            rk.clone(),
            b_zero.clone(),
            k.clone(),
            rk.clone(),
            b_zero.clone(),
        )
        .unwrap();

    lstm_zero_forget
        .set_gate_weights(
            k.clone(),
            rk.clone(),
            b_zero.clone(),
            k.clone(),
            rk.clone(),
            b_zero.clone(), // forget bias = 0.0
            k.clone(),
            rk.clone(),
            b_zero.clone(),
            k.clone(),
            rk.clone(),
            b_zero.clone(),
        )
        .unwrap();

    // 2 timesteps: at step 1, c_0 = 0, so f*c_prev = 0 and the forget bias has no effect.
    // Only from step 2 onward (c_prev != 0) does forget_bias=1.0 diverge from 0.0.
    let x = Array::from_elem((1, 2, 1), 0.5_f32).into_dyn();
    let h_correct = lstm_correct.forward(&x, &mut Ctx::training()).unwrap()[[0, 0]];
    let h_zero = lstm_zero_forget.forward(&x, &mut Ctx::training()).unwrap()[[0, 0]];
    assert!(
        (h_correct - h_zero).abs() > 1e-4,
        "forget_bias=1.0 should produce a measurably different final state than forget_bias=0.0 \
         across timesteps, but h_correct={h_correct:.8}, h_zero={h_zero:.8}"
    );
}

/// LSTM over 2 timesteps threads cell state from t=0 into t=1
#[test]
fn lstm_forward_2step_cell_state_threads_through() {
    let mut lstm = LSTM::new(1, Tanh::new()).unwrap();
    lstm.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm.set_gate_weights(
        k.clone(),
        rk.clone(),
        b_zero.clone(),
        k.clone(),
        rk.clone(),
        b_one.clone(),
        k.clone(),
        rk.clone(),
        b_zero.clone(),
        k.clone(),
        rk.clone(),
        b_zero.clone(),
    )
    .unwrap();

    let x = Array::from_shape_vec((1, 2, 1), vec![0.3_f32, 0.7_f32])
        .unwrap()
        .into_dyn();
    let out = lstm.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.33209477_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

#[test]
fn lstm_output_shape_batch2_units3() {
    let mut lstm = LSTM::new(3, Tanh::new()).unwrap();
    let x = Array::zeros((2, 5, 4)).into_dyn();
    let out = lstm.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
}

/// LSTM gives the same values in an inference pass as in a training pass
#[test]
fn lstm_predict_equals_forward() {
    let mut lstm = LSTM::new(2, Tanh::new()).unwrap();
    lstm.build(&Shape::known(&[2, 3, 2])).unwrap();

    // Non-trivial weights so both paths exercise the same computation
    let kernel = Array2::from_shape_vec((2, 2), vec![0.3, -0.3, 0.2, -0.2]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    let b_zero = Array2::zeros((1, 2));
    let b_one = Array2::from_elem((1, 2), 1.0_f32);

    lstm.set_gate_weights(
        kernel.clone(),
        rk.clone(),
        b_zero.clone(),
        kernel.clone(),
        rk.clone(),
        b_one.clone(),
        kernel.clone(),
        rk.clone(),
        b_zero.clone(),
        kernel.clone(),
        rk.clone(),
        b_zero.clone(),
    )
    .unwrap();

    let x = Array::from_shape_vec(
        (2, 3, 2),
        vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
        ],
    )
    .unwrap()
    .into_dyn();

    let out_training = lstm.forward(&x, &mut Ctx::training()).unwrap();
    let out_inference = lstm.forward(&x, &mut Ctx::inference()).unwrap();

    assert_eq!(out_training.shape(), out_inference.shape());
    assert_allclose(&out_training, &out_inference, 1e-6);
}

/// LSTM constructor rejects a zero dimension, whichever argument carries it
#[test]
fn lstm_new_rejects_zero_dimension() {
    let err = LSTM::new(0, Tanh::new()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero units, got: {err:?}"
    );
}

#[test]
fn lstm_build_rejects_zero_input_dim() {
    let mut lstm = LSTM::new(3, Tanh::new()).unwrap();
    let err = lstm.build(&Shape::known(&[1, 1, 0])).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero input_dim, got: {err:?}"
    );
}

/// LSTM forward rejects a non-3D input (2D tensor)
#[test]
fn lstm_forward_rejects_2d_input() {
    let mut lstm = LSTM::new(1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = lstm.forward_mut(&x, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// LSTM backward before forward returns ForwardPassNotRun
#[test]
fn lstm_backward_before_forward_errors() {
    let lstm = LSTM::new(1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = lstm.backward(&grad, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("LSTM"))
        ),
        "expected ForwardPassNotRun(\"LSTM\"), got: {err:?}"
    );
}

/// LSTM set_weights with wrong kernel shape returns NnError::WeightShape
#[test]
fn lstm_set_weights_wrong_shape_errors() {
    let mut lstm = LSTM::new(3, Tanh::new()).unwrap();

    let good_k = Array2::zeros((2, 3));
    let good_rk = Array2::zeros((3, 3));
    let good_b = Array2::zeros((1, 3));

    // Wrong shape for cell_kernel: should be (2,3), given (3,2)
    let bad_k = Array2::zeros((3, 2));
    let err = lstm
        .set_gate_weights(
            good_k.clone(),
            good_rk.clone(),
            good_b.clone(),
            good_k.clone(),
            good_rk.clone(),
            good_b.clone(),
            bad_k,
            good_rk.clone(),
            good_b.clone(), // bad cell kernel
            good_k.clone(),
            good_rk.clone(),
            good_b.clone(),
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got: {err:?}"
    );
}

// GRU

/// GRU forward over 1 timestep, 1 unit with Tanh
#[test]
fn gru_forward_1step_1unit_tanh() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_gate_weights(
        k.clone(),
        rk.clone(),
        bias.clone(), // reset gate
        k.clone(),
        rk.clone(),
        bias.clone(), // update gate
        k.clone(),
        rk.clone(),
        bias.clone(), // candidate gate
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    // z = sigmoid(0.5) = 0.62245933, candidate = tanh(0.5) = 0.46211716, h_prev = 0
    // h = z*h_prev + (1-z)*candidate = 0.37754067 * 0.46211716 = 0.17446802
    let expected = Array::from_elem((1, 1), 0.174_468_02_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GRU over 2 timesteps blends previous hidden state through the update gate
#[test]
fn gru_forward_2step_hidden_state_blending() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_gate_weights(
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
    )
    .unwrap();

    let x = Array::from_shape_vec((1, 2, 1), vec![0.3_f32, 0.7_f32])
        .unwrap()
        .into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    // t0: z = sigmoid(0.3) = 0.57444252, cand = tanh(0.3) = 0.29131261, h_prev = 0
    //     h = 0.42555748 * 0.29131261 = 0.12397026
    // t1: z = sigmoid(0.7) = 0.66818777, cand = tanh(0.7) = 0.60436778
    //     h = 0.66818777 * 0.12397026 + 0.33181223 * 0.60436778 = 0.28337203
    let expected = Array::from_elem((1, 1), 0.283_372_03_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// GRU update gate z~=0 hands the candidate straight through, since Keras' `z` weights the
/// previous state. An open update gate takes the candidate, whose kernel here gives tanh(0).
#[test]
fn gru_update_gate_zero_takes_the_candidate() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k_zero = Array2::zeros((1, 1));
    let k_neg = Array2::from_elem((1, 1), -20.0_f32); // drives z_t ~= 0
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_gate_weights(
        k_zero.clone(),
        rk.clone(),
        bias.clone(), // reset
        k_neg.clone(),
        rk.clone(),
        bias.clone(), // update (near-zero z_t)
        k_zero.clone(),
        rk.clone(),
        bias.clone(), // candidate
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    // h_prev=0 and the candidate kernel is zero, so h_t = 1 * tanh(0) = 0
    assert_eq!(out.shape(), &[1, 1]);
    assert!(
        out[[0, 0]].abs() < 1e-4,
        "expected h_t about 0 when z about 0, got: {}",
        out[[0, 0]]
    );
}

/// GRU update gate z~=1 keeps the previous hidden state, whatever the candidate says. The
/// candidate here is a clearly non-zero tanh(1), so a flipped `z` would show 0.76, not 0
#[test]
fn gru_update_gate_one_keeps_previous_hidden() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k_one = Array2::from_elem((1, 1), 1.0_f32);
    let k_large = Array2::from_elem((1, 1), 20.0_f32); // drives z_t ~= 1
    let k_zero = Array2::zeros((1, 1));
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_gate_weights(
        k_zero.clone(),
        rk.clone(),
        bias.clone(), // reset (r~=0.5, irrelevant since h_prev=0)
        k_large.clone(),
        rk.clone(),
        bias.clone(), // update (z~=1)
        k_one.clone(),
        rk.clone(),
        bias.clone(), // candidate kernel=1.0
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    // h_t ~= 1*h_prev + 0*tanh(1.0) ~= 0, even though the candidate is tanh(1) = 0.76159416
    assert_eq!(out.shape(), &[1, 1]);
    assert!(
        out[[0, 0]].abs() < 1e-4,
        "expected h_t about 0 when z about 1 (the previous state is kept), got: {}",
        out[[0, 0]]
    );
}

/// The fused kernel's column blocks are ordered [z | r | h] (Keras' update, reset, candidate).
/// This test writes the fused tensors directly instead of through `set_gate_weights`.
#[test]
fn gru_fused_kernel_first_block_is_the_update_gate() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    // [z | r | h] over 1 unit: z driven to 0, r left at 0, candidate kernel 1
    let kernel = Array2::from_shape_vec((1, 3), vec![-20.0_f32, 0.0, 1.0]).unwrap();
    let recurrent_kernel = Array2::zeros((1, 3));
    let bias = Array2::zeros((1, 3));
    gru.set_weights(kernel, recurrent_kernel, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    // z = sigmoid(-20) ~= 0, h_prev = 0, so h = (1 - z) * tanh(1) ~= tanh(1)
    let expected: f32 = 1.0_f32.tanh();
    assert!(
        (out[[0, 0]] - expected).abs() < 1e-4,
        "expected h_t about tanh(1)={expected:.8}, got {} - the fused blocks are not [z | r | h]",
        out[[0, 0]]
    );
}

#[test]
fn gru_output_shape_batch2_units4() {
    let mut gru = GRU::new(4, Tanh::new()).unwrap();
    let x = Array::zeros((2, 5, 3)).into_dyn();
    let out = gru.forward_mut(&x, &mut Ctx::training()).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
}

/// GRU gives the same values in an inference pass as in a training pass
#[test]
fn gru_predict_equals_forward() {
    let mut gru = GRU::new(3, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[2, 3, 2])).unwrap();

    let k = Array2::from_shape_vec((2, 3), vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).unwrap();
    let rk = Array2::from_shape_vec((3, 3), vec![0.1, 0.0, -0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.1])
        .unwrap();
    let bias = Array2::from_shape_vec((1, 3), vec![0.1, -0.1, 0.2]).unwrap();

    gru.set_gate_weights(
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
    )
    .unwrap();

    let x = Array::from_shape_vec(
        (2, 3, 2),
        vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
        ],
    )
    .unwrap()
    .into_dyn();

    let out_training = gru.forward(&x, &mut Ctx::training()).unwrap();
    let out_inference = gru.forward(&x, &mut Ctx::inference()).unwrap();

    assert_eq!(out_training.shape(), out_inference.shape());
    assert_allclose(&out_training, &out_inference, 1e-6);
}

/// GRU constructor rejects a zero dimension, whichever argument carries it
#[test]
fn gru_new_rejects_zero_dimension() {
    let err = GRU::new(0, Tanh::new()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero units, got: {err:?}"
    );
}

#[test]
fn gru_build_rejects_zero_input_dim() {
    let mut gru = GRU::new(3, Tanh::new()).unwrap();
    let err = gru.build(&Shape::known(&[1, 1, 0])).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for zero input_dim, got: {err:?}"
    );
}

/// GRU forward rejects a non-3D input (2D tensor)
#[test]
fn gru_forward_rejects_2d_input() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = gru.forward_mut(&x, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

#[test]
fn gru_forward_rejects_4d_input() {
    let mut gru = GRU::new(1, Tanh::new()).unwrap();
    let x = Array::zeros((1, 2, 3, 4)).into_dyn();
    let err = gru.forward_mut(&x, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// GRU backward before forward returns ForwardPassNotRun
#[test]
fn gru_backward_before_forward_errors() {
    let gru = GRU::new(1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = gru.backward(&grad, &mut Ctx::training()).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun("GRU"))),
        "expected ForwardPassNotRun(\"GRU\"), got: {err:?}"
    );
}

/// GRU set_weights with wrong recurrent_kernel shape returns NnError::WeightShape
#[test]
fn gru_set_weights_wrong_shape_errors() {
    let mut gru = GRU::new(3, Tanh::new()).unwrap();

    let good_k = Array2::zeros((2, 3));
    let good_rk = Array2::zeros((3, 3));
    let good_b = Array2::zeros((1, 3));

    // Wrong shape for update_recurrent_kernel: (2,3) instead of (3,3)
    let bad_rk = Array2::zeros((2, 3));
    let err = gru
        .set_gate_weights(
            good_k.clone(),
            good_rk.clone(),
            good_b.clone(),
            good_k.clone(),
            bad_rk.clone(),
            good_b.clone(),
            good_k.clone(),
            good_rk.clone(),
            good_b.clone(),
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got: {err:?}"
    );
}

// Cross-layer: Activation enum interface vs concrete activation types

/// SimpleRNN accepts Activation enum values, not just concrete activation structs
#[test]
fn simple_rnn_accepts_activation_enum_tanh() {
    let mut rnn = SimpleRNN::new(1, Activation::Tanh).unwrap();
    rnn.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(k, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x, &mut Ctx::training()).unwrap();

    // tanh(0.5) ~= 0.46211716, same as the concrete-Tanh test above
    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

#[test]
fn lstm_accepts_activation_enum_tanh() {
    let mut lstm = LSTM::new(1, Activation::Tanh).unwrap();
    lstm.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm.set_gate_weights(
        k.clone(),
        rk.clone(),
        b_zero.clone(),
        k.clone(),
        rk.clone(),
        b_one.clone(),
        k.clone(),
        rk.clone(),
        b_zero.clone(),
        k.clone(),
        rk.clone(),
        b_zero.clone(),
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = lstm.forward(&x, &mut Ctx::training()).unwrap();

    let expected = Array::from_elem((1, 1), 0.17426972_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

#[test]
fn gru_accepts_activation_enum_tanh() {
    let mut gru = GRU::new(1, Activation::Tanh).unwrap();
    gru.build(&Shape::known(&[1, 1, 1])).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_gate_weights(
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = gru.forward(&x, &mut Ctx::training()).unwrap();

    // z = sigmoid(0.5) = 0.62245933, candidate = tanh(0.5) = 0.46211716, h_prev = 0
    // h = z*h_prev + (1-z)*candidate = 0.37754067 * 0.46211716 = 0.17446802
    let expected = Array::from_elem((1, 1), 0.174_468_02_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// Determinism: same weights + same input => identical outputs

/// SimpleRNN: 2 forward passes with the same weights and input are bit-identical
#[test]
fn simple_rnn_forward_is_deterministic() {
    let mut rnn = SimpleRNN::new(2, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[1, 3, 2])).unwrap();

    let k = Array2::from_shape_vec((2, 2), vec![0.3, -0.1, 0.2, -0.4]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.2]).unwrap();
    let bias = Array2::from_shape_vec((1, 2), vec![0.05, -0.05]).unwrap();
    rnn.set_weights(k, rk, bias).unwrap();

    let x = Array::from_shape_vec((1, 3, 2), vec![0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
        .unwrap()
        .into_dyn();

    let out1 = rnn.forward(&x, &mut Ctx::training()).unwrap();
    let out2 = rnn.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out1, out2, "forward passes must be bit-identical");
}

/// GRU: 2 forward passes with the same weights are bit-identical
#[test]
fn gru_forward_is_deterministic() {
    let mut gru = GRU::new(2, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 3, 2])).unwrap();

    let k = Array2::from_shape_vec((2, 2), vec![0.3, -0.1, 0.2, -0.4]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.2]).unwrap();
    let bias = Array2::from_shape_vec((1, 2), vec![0.05, -0.05]).unwrap();

    gru.set_gate_weights(
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
        k.clone(),
        rk.clone(),
        bias.clone(),
    )
    .unwrap();

    let x = Array::from_shape_vec((1, 3, 2), vec![0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
        .unwrap()
        .into_dyn();

    let out1 = gru.forward(&x, &mut Ctx::training()).unwrap();
    let out2 = gru.forward(&x, &mut Ctx::training()).unwrap();

    assert_eq!(out1, out2, "GRU forward passes must be bit-identical");
}
// param_count(): trainable parameter formula per recurrent layer

/// SimpleRNN param_count = input_dim*units + units*units + units (1 gate)
#[test]
fn simple_rnn_param_count_formula() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut rnn = SimpleRNN::new(2, Tanh::new()).unwrap();
    rnn.build(&Shape::known(&[1, 1, 3])).unwrap();
    // 3*2 + 2*2 + 2 = 6 + 4 + 2 = 12
    assert_eq!(rnn.param_count(), ParamCounts::trainable(12));
}

/// GRU param_count = 3 * (input_dim*units + units*units + units) (3 gates)
#[test]
fn gru_param_count_formula() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut gru = GRU::new(2, Tanh::new()).unwrap();
    gru.build(&Shape::known(&[1, 1, 3])).unwrap();
    // 3 * (3*2 + 2*2 + 2) = 3 * 12 = 36
    assert_eq!(gru.param_count(), ParamCounts::trainable(36));
}

/// LSTM param_count = 4 * (input_dim*units + units*units + units) (4 gates)
#[test]
fn lstm_param_count_formula() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut lstm = LSTM::new(2, Tanh::new()).unwrap();
    lstm.build(&Shape::known(&[1, 1, 3])).unwrap();
    // 4 * (3*2 + 2*2 + 2) = 4 * 12 = 48
    assert_eq!(lstm.param_count(), ParamCounts::trainable(48));
}

// return_sequences and go_backwards
// The 3 layers share 1 set of fixed weights and 1 input. The values are exact multiples of 1/8,
// which f32 holds without rounding, so the same numbers pin all 4 flag combinations.

/// The fixed rank-3 input of the flag tests, with shape (batch 2, timesteps 3, features 2)
///
/// Every timestep differs, and the 2 batch rows differ, so a wrong time index or a wrong batch
/// row cannot cancel out.
fn flag_input() -> Tensor {
    Array::from_shape_vec(
        (2, 3, 2),
        vec![
            1.0_f32, 0.5, -2.0, 0.25, 3.0, -1.5, 0.25, -1.0, 1.5, -0.5, -0.75, 2.0,
        ],
    )
    .unwrap()
    .into_dyn()
}

/// The fixed weights of a `units = 2` layer with `n_gates` gate blocks, as
/// (kernel, recurrent kernel, bias)
fn flag_weights(n_gates: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    // Multiples of 1/8, cycling with a different phase per matrix
    let value = |index: usize, seed: usize| ((((index * 5 + seed * 3) % 9) as f32) - 4.0) / 8.0;
    let width = 2 * n_gates;
    let kernel = Array2::from_shape_fn((2, width), |(r, c)| value(r * width + c, 1));
    let recurrent = Array2::from_shape_fn((2, width), |(r, c)| value(r * width + c, 2));
    let bias = Array2::from_shape_fn((1, width), |(_, c)| value(c, 3));
    (kernel, recurrent, bias)
}

/// Builds the flag-test SimpleRNN with fixed weights
fn flag_simple_rnn(return_sequences: bool, go_backwards: bool) -> SimpleRNN {
    let mut layer = SimpleRNN::new(2, Tanh::new())
        .unwrap()
        .with_return_sequences(return_sequences)
        .with_go_backwards(go_backwards);
    layer.build(&Shape::known(&[2, 3, 2])).unwrap();
    let (kernel, recurrent, bias) = flag_weights(1);
    layer.set_weights(kernel, recurrent, bias).unwrap();
    layer
}

/// Builds the flag-test LSTM with fixed weights
fn flag_lstm(return_sequences: bool, go_backwards: bool) -> LSTM {
    let mut layer = LSTM::new(2, Tanh::new())
        .unwrap()
        .with_return_sequences(return_sequences)
        .with_go_backwards(go_backwards);
    layer.build(&Shape::known(&[2, 3, 2])).unwrap();
    let (kernel, recurrent, bias) = flag_weights(4);
    layer.set_weights(kernel, recurrent, bias).unwrap();
    layer
}

/// Builds the flag-test GRU with fixed weights
fn flag_gru(return_sequences: bool, go_backwards: bool) -> GRU {
    let mut layer = GRU::new(2, Tanh::new())
        .unwrap()
        .with_return_sequences(return_sequences)
        .with_go_backwards(go_backwards);
    layer.build(&Shape::known(&[2, 3, 2])).unwrap();
    let (kernel, recurrent, bias) = flag_weights(3);
    layer.set_weights(kernel, recurrent, bias).unwrap();
    layer
}

/// The upstream gradient of the flag tests, shaped like `output`
///
/// No 2 entries are equal, and the values change along the time axis. A constant upstream
/// gradient hides a backward pass that reads the time axis in the wrong direction, because
/// every slice is then the same. The Keras runs that produced the pinned values below used
/// exactly these numbers.
fn flag_upstream(output: &Tensor) -> Tensor {
    let shape = output.shape().to_vec();
    if shape.len() == 3 {
        Tensor::from_shape_fn(shape, |index| {
            0.5 * (1 + index[1]) as f32 + 0.25 * index[0] as f32 - 0.125 * index[2] as f32
        })
    } else {
        Tensor::from_shape_fn(shape, |index| {
            1.0 + 0.25 * index[0] as f32 - 0.125 * index[1] as f32
        })
    }
}

/// Compares a flat gradient slice against pinned values, 1 element at a time
fn assert_close_flat(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{what}[{index}]: got {got}, expected {want}"
        );
    }
}

/// Compares the analytic input gradient against a central finite difference of
/// L = sum(upstream * output), with a time-varying upstream gradient
fn check_flag_input_gradient(layer: &mut dyn UnaryLayer, x: &Tensor, tol: f32) {
    const EPS: f32 = 1e-3;
    let mut ctx = Ctx::training();
    let output = layer.forward(x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let analytic = layer.backward(&upstream, &mut ctx).unwrap();
    assert_eq!(
        analytic.shape(),
        x.shape(),
        "input-gradient shape must match the input shape"
    );

    let analytic_flat: Vec<f32> = analytic.iter().cloned().collect();
    let mut x_flat: Vec<f32> = x.iter().cloned().collect();
    for i in 0..x_flat.len() {
        let original = x_flat[i];

        x_flat[i] = original + EPS;
        let plus = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_plus: f32 = (&layer.forward(&plus, &mut Ctx::inference()).unwrap() * &upstream).sum();

        x_flat[i] = original - EPS;
        let minus = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_minus: f32 =
            (&layer.forward(&minus, &mut Ctx::inference()).unwrap() * &upstream).sum();

        x_flat[i] = original;

        let numeric = (l_plus - l_minus) / (2.0 * EPS);
        assert_abs_diff_eq!(analytic_flat[i], numeric, epsilon = tol);
    }
}

/// Compares every analytic weight gradient against a central finite difference of
/// L = sum(upstream * output), with a time-varying upstream gradient
fn check_flag_weight_gradients(layer: &mut dyn UnaryLayer, x: &Tensor, tol: f32) {
    const EPS: f32 = 1e-3;
    let mut ctx = Ctx::training();
    let output = layer.forward(x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    layer.backward(&upstream, &mut ctx).unwrap();

    let params: Vec<(Vec<f32>, Vec<f32>)> = layer
        .parameters_mut()
        .into_iter()
        .map(|param| (param.value.to_vec(), grad_of(&ctx, param.name).to_vec()))
        .collect();
    assert_eq!(
        params.len(),
        3,
        "expected kernel, recurrent kernel, and bias"
    );

    for (p_idx, (values, grads)) in params.iter().enumerate() {
        for i in 0..values.len() {
            let original = values[i];

            layer.parameters_mut()[p_idx].value[i] = original + EPS;
            let l_plus: f32 = (&layer.forward(x, &mut Ctx::inference()).unwrap() * &upstream).sum();

            layer.parameters_mut()[p_idx].value[i] = original - EPS;
            let l_minus: f32 =
                (&layer.forward(x, &mut Ctx::inference()).unwrap() * &upstream).sum();

            layer.parameters_mut()[p_idx].value[i] = original;

            let numeric = (l_plus - l_minus) / (2.0 * EPS);
            assert_abs_diff_eq!(grads[i], numeric, epsilon = tol);
        }
    }
}

/// return_sequences turns the SimpleRNN output into (batch, timesteps, units), in both the
/// training forward pass and the inference pass
#[test]
fn simple_rnn_return_sequences_output_is_rank_3() {
    let x = flag_input();
    let layer = flag_simple_rnn(true, false);
    assert_eq!(
        layer.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 3, 2]
    );
    assert_eq!(
        layer.forward(&x, &mut Ctx::inference()).unwrap().shape(),
        &[2, 3, 2]
    );

    let last_only = flag_simple_rnn(false, false);
    assert_eq!(
        last_only.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 2]
    );
    assert_eq!(
        last_only
            .forward(&x, &mut Ctx::inference())
            .unwrap()
            .shape(),
        &[2, 2]
    );
}

/// return_sequences turns the LSTM output into (batch, timesteps, units)
#[test]
fn lstm_return_sequences_output_is_rank_3() {
    let x = flag_input();
    let layer = flag_lstm(true, false);
    assert_eq!(
        layer.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 3, 2]
    );
    assert_eq!(
        layer.forward(&x, &mut Ctx::inference()).unwrap().shape(),
        &[2, 3, 2]
    );

    let last_only = flag_lstm(false, false);
    assert_eq!(
        last_only.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 2]
    );
}

/// return_sequences turns the GRU output into (batch, timesteps, units)
#[test]
fn gru_return_sequences_output_is_rank_3() {
    let x = flag_input();
    let layer = flag_gru(true, false);
    assert_eq!(
        layer.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 3, 2]
    );
    assert_eq!(
        layer.forward(&x, &mut Ctx::inference()).unwrap().shape(),
        &[2, 3, 2]
    );

    let last_only = flag_gru(false, false);
    assert_eq!(
        last_only.forward(&x, &mut Ctx::training()).unwrap().shape(),
        &[2, 2]
    );
}

/// go_backwards changes no output shape, with or without return_sequences
#[test]
fn go_backwards_keeps_the_output_shape() {
    let x = flag_input();
    for return_sequences in [false, true] {
        let expected: &[usize] = if return_sequences {
            &[2, 3, 2]
        } else {
            &[2, 2]
        };
        assert_eq!(
            flag_simple_rnn(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap()
                .shape(),
            expected
        );
        assert_eq!(
            flag_lstm(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap()
                .shape(),
            expected
        );
        assert_eq!(
            flag_gru(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap()
                .shape(),
            expected
        );
    }
}

/// go_backwards emits the states in processing order and never flips them back to input order
///
/// The weights make the cell an identity, h_t = x_t, so the returned sequence is the input
/// sequence in processing order. With go_backwards the input [1, 2, 3, 4, 5] must come back as
/// [5, 4, 3, 2, 1]. Any re-reversal of the output belongs to a later bidirectional wrapper, not
/// to this flag.
#[test]
fn simple_rnn_go_backwards_emits_states_in_processing_order() {
    let x = Array::from_shape_vec((1, 5, 1), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let identity_cell = |go_backwards: bool| {
        let mut layer = SimpleRNN::new(1, Activation::Linear)
            .unwrap()
            .with_return_sequences(true)
            .with_go_backwards(go_backwards);
        layer.build(&Shape::known(&[1, 1, 1])).unwrap();
        layer
            .set_weights(
                Array2::from_elem((1, 1), 1.0),
                Array2::zeros((1, 1)),
                Array2::zeros((1, 1)),
            )
            .unwrap();
        layer.forward(&x, &mut Ctx::training()).unwrap()
    };

    let forwards = identity_cell(false);
    let backwards = identity_cell(true);

    let expected_forwards = Array::from_shape_vec((1, 5, 1), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let expected_backwards = Array::from_shape_vec((1, 5, 1), vec![5.0_f32, 4.0, 3.0, 2.0, 1.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&forwards, &expected_forwards, 1e-6);
    assert_allclose(&backwards, &expected_backwards, 1e-6);
}

/// The last slot of a returned sequence is bit-identical to the output of the same layer with
/// return_sequences off, which is also the final state that the recurrence carries out
///
/// This holds for both settings of go_backwards, because the last processing step ends the
/// recurrence in both directions.
#[test]
fn return_sequences_last_slot_equals_the_final_state() {
    let x = flag_input();
    for go_backwards in [false, true] {
        let cases: Vec<(&str, Tensor, Tensor, Tensor)> = vec![
            (
                "SimpleRNN",
                flag_simple_rnn(true, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_simple_rnn(false, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_simple_rnn(false, go_backwards)
                    .forward(&x, &mut Ctx::inference())
                    .unwrap(),
            ),
            (
                "LSTM",
                flag_lstm(true, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_lstm(false, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_lstm(false, go_backwards)
                    .forward(&x, &mut Ctx::inference())
                    .unwrap(),
            ),
            (
                "GRU",
                flag_gru(true, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_gru(false, go_backwards)
                    .forward(&x, &mut Ctx::training())
                    .unwrap(),
                flag_gru(false, go_backwards)
                    .forward(&x, &mut Ctx::inference())
                    .unwrap(),
            ),
        ];
        for (name, sequence, last_state, inference_state) in cases {
            let sequence3 = sequence.into_dimensionality::<ndarray::Ix3>().unwrap();
            let last_slot = sequence3.index_axis(ndarray::Axis(1), 2);
            let last_state2 = last_state.into_dimensionality::<ndarray::Ix2>().unwrap();
            let inference2 = inference_state
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            assert_eq!(
                last_slot.to_owned(),
                last_state2,
                "{name}: last slot must equal the return_sequences=false output \
                 (go_backwards={go_backwards})"
            );
            assert_eq!(
                last_slot.to_owned(),
                inference2,
                "{name}: last slot must equal the final state of the inference pass \
                 (go_backwards={go_backwards})"
            );
        }
    }
}

/// go_backwards on the given input equals the same layer without the flag on the time-reversed
/// input, for all 3 layers
///
/// This pins the forward index map on its own: only the order in which the timesteps enter the
/// recurrence changes.
#[test]
fn go_backwards_equals_the_forward_layer_on_reversed_input() {
    let x = flag_input();
    let mut reversed = Tensor::zeros(x.raw_dim());
    for t in 0..3 {
        reversed
            .index_axis_mut(ndarray::Axis(1), t)
            .assign(&x.index_axis(ndarray::Axis(1), 2 - t));
    }

    for return_sequences in [false, true] {
        assert_eq!(
            flag_simple_rnn(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap(),
            flag_simple_rnn(return_sequences, false)
                .forward(&reversed, &mut Ctx::training())
                .unwrap(),
            "SimpleRNN go_backwards must equal the forward layer on the reversed input"
        );
        assert_eq!(
            flag_lstm(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap(),
            flag_lstm(return_sequences, false)
                .forward(&reversed, &mut Ctx::training())
                .unwrap(),
            "LSTM go_backwards must equal the forward layer on the reversed input"
        );
        assert_eq!(
            flag_gru(return_sequences, true)
                .forward(&x, &mut Ctx::training())
                .unwrap(),
            flag_gru(return_sequences, false)
                .forward(&reversed, &mut Ctx::training())
                .unwrap(),
            "GRU go_backwards must equal the forward layer on the reversed input"
        );
    }
}

/// Every SimpleRNN gradient matches a finite difference, for all 4 flag combinations
///
/// The upstream gradient varies along the time axis. A constant one would hide 2 defects. The
/// first is a backward pass that reads the upstream time axis in the reverse direction. The
/// second overwrites the carried gradient of a step instead of adding the direct contribution to
/// it.
#[test]
fn simple_rnn_gradients_match_finite_difference_for_every_flag_combination() {
    let x = flag_input();
    for return_sequences in [false, true] {
        for go_backwards in [false, true] {
            check_flag_input_gradient(
                &mut flag_simple_rnn(return_sequences, go_backwards),
                &x,
                2e-2,
            );
            check_flag_weight_gradients(
                &mut flag_simple_rnn(return_sequences, go_backwards),
                &x,
                2e-2,
            );
        }
    }
}

/// Every LSTM gradient matches a finite difference, for all 4 flag combinations
#[test]
fn lstm_gradients_match_finite_difference_for_every_flag_combination() {
    let x = flag_input();
    for return_sequences in [false, true] {
        for go_backwards in [false, true] {
            check_flag_input_gradient(&mut flag_lstm(return_sequences, go_backwards), &x, 3e-2);
            check_flag_weight_gradients(&mut flag_lstm(return_sequences, go_backwards), &x, 3e-2);
        }
    }
}

/// Every GRU gradient matches a finite difference, for all 4 flag combinations
#[test]
fn gru_gradients_match_finite_difference_for_every_flag_combination() {
    let x = flag_input();
    for return_sequences in [false, true] {
        for go_backwards in [false, true] {
            check_flag_input_gradient(&mut flag_gru(return_sequences, go_backwards), &x, 3e-2);
            check_flag_weight_gradients(&mut flag_gru(return_sequences, go_backwards), &x, 3e-2);
        }
    }
}

/// With return_sequences on, the backward pass rejects a rank-2 gradient
#[test]
fn return_sequences_backward_rejects_a_rank_2_gradient() {
    let x = flag_input();
    let layer = flag_simple_rnn(true, false);
    let mut ctx = Ctx::training();
    layer.forward(&x, &mut ctx).unwrap();
    let err = layer
        .backward(&Tensor::ones(vec![2, 2]), &mut ctx)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// With return_sequences off, the backward pass rejects a rank-3 gradient
#[test]
fn backward_rejects_a_rank_3_gradient_without_return_sequences() {
    let x = flag_input();
    let layer = flag_lstm(false, false);
    let mut ctx = Ctx::training();
    layer.forward(&x, &mut ctx).unwrap();
    let err = layer
        .backward(&Tensor::ones(vec![2, 3, 2]), &mut ctx)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// With return_sequences on, the backward pass rejects a gradient with the wrong timestep count
#[test]
fn return_sequences_backward_rejects_a_wrong_timestep_count() {
    let x = flag_input();
    let layer = flag_gru(true, false);
    let mut ctx = Ctx::training();
    layer.forward(&x, &mut ctx).unwrap();
    let err = layer
        .backward(&Tensor::ones(vec![2, 4, 2]), &mut ctx)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// `output_shape` prints the time axis only when return_sequences is on
#[test]
fn output_shape_shows_the_time_axis_when_return_sequences() {
    assert_eq!(flag_simple_rnn(false, false).output_shape(), "(None, 2)");
    assert_eq!(
        flag_simple_rnn(true, false).output_shape(),
        "(None, None, 2)"
    );
    assert_eq!(flag_lstm(true, false).output_shape(), "(None, None, 2)");
    assert_eq!(flag_gru(true, false).output_shape(), "(None, None, 2)");
}

/// SimpleRNN with return_sequences on and go_backwards off matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn simple_rnn_return_sequences_matches_keras() {
    let x = flag_input();
    let layer = flag_simple_rnn(true, false);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.5545997,
            0.3583574,
            -0.24892446,
            -0.7192768,
            -0.83576536,
            0.9873171,
            -0.48633605,
            0.635149,
            -0.51602226,
            0.82336944,
            -0.22269693,
            -0.8408054,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            0.25054154,
            -0.30537015,
            0.12154034,
            -0.25103593,
            -0.03920213,
            -0.017328419,
            0.20742314,
            -0.29309633,
            0.1301502,
            -0.2720108,
            0.030198768,
            -0.23810008,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[0.3496985, 0.31599754, 1.873532, 0.4730997],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[-2.0973096, -0.79738307, 2.1362145, 0.89261675],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[5.4103193, 2.7538834],
        1e-5,
        "grad_bias",
    );
}

/// SimpleRNN with return_sequences on and go_backwards on matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn simple_rnn_return_sequences_go_backwards_matches_keras() {
    let x = flag_input();
    let layer = flag_simple_rnn(true, true);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.70390564,
            0.98284495,
            -0.057346564,
            -0.7383669,
            -0.7241064,
            0.44755006,
            -0.38528392,
            -0.84828365,
            -0.8011903,
            0.86860174,
            -0.38491935,
            0.68672025,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            0.4606049,
            -0.549793,
            0.11600702,
            -0.22857058,
            -0.027373,
            -0.011149498,
            0.24299803,
            -0.4293374,
            0.13053736,
            -0.19356237,
            0.024405725,
            -0.10735464,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[0.46831328, 0.88652456, -0.29602283, 0.007811159],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[-2.0633967, -1.2219566, 1.2253647, 0.05485581],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[4.5807, 3.039535],
        1e-5,
        "grad_bias",
    );
}

/// LSTM with return_sequences on and go_backwards off matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn lstm_return_sequences_matches_keras() {
    let x = flag_input();
    let layer = flag_lstm(true, false);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.06765706,
            0.023553586,
            -0.10493167,
            0.196208,
            0.033383537,
            -0.15733401,
            0.045450613,
            0.050081395,
            0.05193903,
            -0.061121304,
            -0.10756472,
            0.13856108,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.17409706,
            -0.031293392,
            0.03688872,
            -0.09638176,
            -0.04067032,
            -0.0666401,
            -0.19899246,
            -0.022938691,
            -0.35949486,
            -0.009079859,
            0.00010004453,
            0.01727938,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[
            0.5904733,
            -0.5597786,
            -0.13735434,
            0.111462876,
            0.61537766,
            2.2943888,
            0.30915204,
            -0.6266583,
            -0.4314919,
            0.46769798,
            0.09959567,
            -0.10815649,
            -0.30539474,
            -0.6152131,
            -0.35726708,
            0.33183357,
        ],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[
            -0.008106956,
            0.0015690634,
            0.009512457,
            -0.002524186,
            -0.008245606,
            0.034126412,
            -0.002365034,
            0.012811491,
            0.026728157,
            -0.01752933,
            -0.013027558,
            0.0070465943,
            0.037620932,
            0.06654514,
            0.010438064,
            -0.032600705,
        ],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[
            0.000060293823,
            0.2935255,
            -0.061339106,
            0.017778756,
            1.0380362,
            2.5953631,
            -0.1780461,
            -0.055545352,
        ],
        1e-5,
        "grad_bias",
    );
}

/// LSTM with return_sequences on and go_backwards on matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn lstm_return_sequences_go_backwards_matches_keras() {
    let x = flag_input();
    let layer = flag_lstm(true, true);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            0.10290291,
            -0.19481526,
            -0.05656884,
            -0.19899441,
            -0.11248118,
            -0.08286833,
            -0.12080535,
            0.20313686,
            -0.08614126,
            0.00007648483,
            0.015377403,
            0.050328977,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.1439949,
            -0.074894644,
            0.15502214,
            -0.14898567,
            -0.17439052,
            0.034595758,
            -0.15369596,
            -0.06909001,
            -0.32124698,
            -0.08991817,
            0.011020429,
            -0.038134776,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[
            0.49334764,
            -0.817438,
            -0.27636185,
            0.3176492,
            0.44359598,
            2.6033297,
            0.040531687,
            -0.13643096,
            -0.46952128,
            0.54946846,
            0.083258525,
            -0.09569697,
            -0.2009471,
            -0.7782507,
            -0.22264296,
            0.110534534,
        ],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[
            -0.012289774,
            0.019935433,
            0.02210893,
            -0.019172555,
            -0.04920122,
            -0.15587437,
            0.0047810026,
            -0.006230273,
            0.03337007,
            -0.040072322,
            -0.02416394,
            0.050993584,
            -0.035234574,
            0.032153495,
            0.016554007,
            0.019151004,
        ],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[
            -0.10728342,
            0.2833891,
            -0.16577987,
            -0.11529134,
            1.2285926,
            2.5431266,
            -0.22867866,
            -0.06892677,
        ],
        1e-5,
        "grad_bias",
    );
}

/// GRU with return_sequences on and go_backwards off matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn gru_return_sequences_matches_keras() {
    let x = flag_input();
    let layer = flag_gru(true, false);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            0.07740468,
            0.0,
            -0.15070978,
            0.5847126,
            -0.5187029,
            0.4571597,
            -0.4226371,
            0.103469394,
            -0.38575846,
            0.036150984,
            0.059671525,
            0.39422223,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.010166224,
            0.40921775,
            -0.29145205,
            0.43735614,
            0.14094763,
            0.38503677,
            -0.27779615,
            0.4107467,
            0.056554534,
            0.64688504,
            -0.23370856,
            0.27479464,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[
            0.27078903,
            1.389612,
            -0.11682667,
            -0.06690118,
            3.7433739,
            -0.06377387,
            -1.0905398,
            -0.63310885,
            -0.013828568,
            0.033715293,
            -0.6488985,
            0.6108237,
        ],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[
            0.15495442,
            -0.0131989345,
            0.041505873,
            0.0045489334,
            -0.337397,
            -0.19915271,
            0.038828164,
            0.09434194,
            -0.0140583655,
            -0.012043089,
            0.21039428,
            0.051282093,
        ],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[
            -0.011169866,
            -0.40618157,
            -0.10069622,
            -0.023467647,
            5.000223,
            2.8018348,
        ],
        1e-5,
        "grad_bias",
    );
}

/// GRU with return_sequences on and go_backwards on matches Keras 3.15.1
///
/// Every expected value comes from Keras 3.15.1 with the same weights, the same input, and the
/// same time-varying upstream gradient. The gradients pin the 2 rules that a plausible but wrong
/// backward pass breaks. The upstream slice of a step must add to the carried gradient. The gate
/// gradient of a step must go to the input timestep of that step.
#[test]
fn gru_return_sequences_go_backwards_matches_keras() {
    let x = flag_input();
    let layer = flag_gru(true, true);
    let mut ctx = Ctx::training();
    let output = layer.forward(&x, &mut ctx).unwrap();
    let upstream = flag_upstream(&output);
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();

    let expected_output = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            -0.4310903,
            -0.075711645,
            -0.42406064,
            0.5368707,
            -0.13581459,
            0.38729942,
            0.27444428,
            0.3911135,
            -0.1349066,
            0.24974419,
            -0.5050205,
            0.27652773,
        ],
    )
    .unwrap();
    assert_allclose(&output, &expected_output, 1e-5);

    let expected_grad_input = Tensor::from_shape_vec(
        vec![2, 3, 2],
        vec![
            0.04979286,
            0.44526592,
            -0.25146422,
            0.43066517,
            0.082815774,
            0.39240703,
            -0.14389306,
            0.41218275,
            0.03063862,
            0.639113,
            -0.37738752,
            0.25003532,
        ],
    )
    .unwrap();
    assert_allclose(&grad_input, &expected_grad_input, 1e-5);

    assert_close_flat(
        grad_of(&ctx, "kernel"),
        &[
            0.81879985,
            1.4973927,
            0.078411855,
            -0.0266757,
            3.5445094,
            0.022810549,
            -0.9924164,
            -0.6868999,
            -0.044752475,
            -0.0022787661,
            -0.7942069,
            1.2556272,
        ],
        1e-5,
        "grad_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "recurrent_kernel"),
        &[
            0.09131509,
            0.1512349,
            0.050219238,
            0.0023983782,
            -0.19612712,
            -0.11685265,
            0.062770486,
            0.17555079,
            -0.012414873,
            -0.010123244,
            0.52781,
            0.2543729,
        ],
        1e-5,
        "grad_recurrent_kernel",
    );
    assert_close_flat(
        grad_of(&ctx, "bias"),
        &[
            0.18320782,
            -0.20841561,
            -0.060462885,
            -0.020826302,
            4.9836674,
            2.9753606,
        ],
        1e-5,
        "grad_bias",
    );
}

// The 2 guards that the cell protocol made uniform

/// All 3 layers accept a weight matrix in a non-standard memory order, and store it in the
/// standard order
///
/// An optimizer reads every array of a layer as a flat slice, so an array in column-major order
/// panics rather than returning an error.
#[test]
fn set_weights_normalizes_a_column_major_array() {
    /// Builds a kernel of the right shape whose memory order is column-major
    fn column_major(rows: usize, columns: usize) -> Array2<f32> {
        let values: Vec<f32> = (0..rows * columns).map(|i| i as f32).collect();
        // `reversed_axes` gives an owned array of shape (rows, columns) that is not contiguous
        // in the standard order
        let array = Array2::from_shape_vec((columns, rows), values).expect("the shape holds");
        let flipped = array.reversed_axes();
        assert!(!flipped.is_standard_layout());
        flipped
    }

    let shape = Shape::known(&[2, 4, 3]);

    let mut simple = SimpleRNN::new(2, Activation::Tanh).unwrap();
    simple.build(&shape).unwrap();
    simple
        .set_weights(
            column_major(3, 2),
            column_major(2, 2),
            Array2::zeros((1, 2)),
        )
        .expect("a column-major array of the right shape is accepted");
    assert_eq!(simple.parameters_mut().len(), 3);

    let mut lstm = LSTM::new(2, Activation::Tanh).unwrap();
    lstm.build(&shape).unwrap();
    lstm.set_weights(
        column_major(3, 8),
        column_major(2, 8),
        Array2::zeros((1, 8)),
    )
    .expect("a column-major array of the right shape is accepted");
    assert_eq!(lstm.parameters_mut().len(), 3);

    let mut gru = GRU::new(2, Activation::Tanh).unwrap();
    gru.build(&shape).unwrap();
    gru.set_weights(
        column_major(3, 6),
        column_major(2, 6),
        Array2::zeros((1, 6)),
    )
    .expect("a column-major array of the right shape is accepted");
    assert_eq!(gru.parameters_mut().len(), 3);
}

/// A built recurrent layer refuses a forward pass over a tensor that is not rank 3
#[test]
fn a_built_layer_refuses_an_input_that_is_not_rank_3() {
    let shape = Shape::known(&[2, 4, 3]);
    let flat: Tensor = Array::zeros((2, 3)).into_dyn();

    let mut simple = SimpleRNN::new(2, Activation::Tanh).unwrap();
    simple.build(&shape).unwrap();
    let mut ctx = Ctx::inference();
    let error = simple
        .forward(&flat, &mut ctx)
        .expect_err("a rank-2 tensor is not a sequence");
    assert!(
        matches!(error, Error::InvalidInput(_)),
        "expected InvalidInput, got {error:?}"
    );

    let mut lstm = LSTM::new(2, Activation::Tanh).unwrap();
    lstm.build(&shape).unwrap();
    assert!(lstm.forward(&flat, &mut ctx).is_err());

    let mut gru = GRU::new(2, Activation::Tanh).unwrap();
    gru.build(&shape).unwrap();
    assert!(gru.forward(&flat, &mut ctx).is_err());
}
/// The 3 public layers print under their own names
#[test]
fn debug_prints_the_layer_name() {
    let text = format!("{:?}", SimpleRNN::new(2, Activation::Tanh).unwrap());
    assert!(text.starts_with("SimpleRNN {"), "got {text}");
    let text = format!("{:?}", LSTM::new(2, Activation::Tanh).unwrap());
    assert!(text.starts_with("LSTM {"), "got {text}");
    let text = format!("{:?}", GRU::new(2, Activation::Tanh).unwrap());
    assert!(text.starts_with("GRU {"), "got {text}");
}

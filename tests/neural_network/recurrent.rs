//! Integration tests for the recurrent layers: SimpleRNN, LSTM, GRU
//!
//! Expected values are hand-computed from the mathematical definitions. Backward/gradient values
//! are covered by tests/neural_network/gradient_check.rs and are not duplicated here

use crate::common::assert_allclose;
use ndarray::{Array, Array2};
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

// SimpleRNN

/// SimpleRNN forward over 1 timestep, 1 unit with Tanh yields tanh(0.5)
#[test]
fn simple_rnn_forward_1step_1unit_tanh() {
    let mut rnn = SimpleRNN::new(1, 1, Tanh::new()).unwrap();

    let kernel = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::from_elem((1, 1), 0.5_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN over 2 timesteps threads hidden state between steps (Tanh)
#[test]
fn simple_rnn_forward_2step_tanh_state_threading() {
    let mut rnn = SimpleRNN::new(1, 1, Tanh::new()).unwrap();

    let kernel = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::from_elem((1, 1), 1.0_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_shape_vec((1, 2, 1), vec![0.3_f32, 0.7_f32])
        .unwrap()
        .into_dyn();
    let out = rnn.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.75792147_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN with ReLU passes a positive pre-activation through unchanged
#[test]
fn simple_rnn_forward_relu_positive() {
    let mut rnn = SimpleRNN::new(1, 1, ReLU::new()).unwrap();

    let kernel = Array2::from_elem((1, 1), 0.5_f32);
    let rk = Array2::from_elem((1, 1), 0.5_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);
    let expected = Array::from_elem((1, 1), 0.5_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN with ReLU clips a negative pre-activation to exactly 0
#[test]
fn simple_rnn_relu_negative_preactivation_is_zero() {
    let mut rnn = SimpleRNN::new(1, 1, ReLU::new()).unwrap();

    let kernel = Array2::from_elem((1, 1), -1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.8_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);
    let expected = Array::zeros((1, 1)).into_dyn();
    assert_allclose(&out, &expected, 1e-7);
}

/// SimpleRNN output shape is (batch, units) regardless of timestep count
#[test]
fn simple_rnn_output_shape_batch3_units2() {
    let mut rnn = SimpleRNN::new(4, 2, Tanh::new()).unwrap();
    // 3 batches, 5 timesteps, 4 input features
    let x = Array::zeros((3, 5, 4)).into_dyn();
    let out = rnn.forward(&x).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
}

/// SimpleRNN predict() matches forward() element-wise in eval mode
#[test]
fn simple_rnn_predict_equals_forward() {
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new()).unwrap();

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

    let out_forward = rnn.forward(&x).unwrap();
    let out_predict = rnn.predict(&x).unwrap();

    assert_eq!(out_forward.shape(), out_predict.shape());
    assert_allclose(&out_forward, &out_predict, 1e-6);
}

/// SimpleRNN constructor rejects a zero dimension, whichever argument carries it
#[test]
fn simple_rnn_new_rejects_zero_dimension() {
    // (input_dim, units, name of the bad argument), 1 row per zero-valued argument
    let cases = [(0, 3, "input_dim"), (2, 0, "units")];
    for (input_dim, units, bad_arg) in cases {
        let err = SimpleRNN::new(input_dim, units, Tanh::new()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for zero {bad_arg} (input_dim={input_dim}, units={units}), got: {err:?}"
        );
    }
}

/// SimpleRNN forward rejects a non-3D input (2D tensor)
#[test]
fn simple_rnn_forward_rejects_2d_input() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn(); // 2D, not 3D
    let err = rnn.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// SimpleRNN forward rejects a 1D input
#[test]
fn simple_rnn_forward_rejects_1d_input() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new()).unwrap();
    let x = Array::zeros(4).into_dyn(); // 1D
    let err = rnn.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// SimpleRNN backward before forward returns ForwardPassNotRun
#[test]
fn simple_rnn_backward_before_forward_errors() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = rnn.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("SimpleRNN"))
        ),
        "expected ForwardPassNotRun(\"SimpleRNN\"), got: {err:?}"
    );
}

/// SimpleRNN backward with units == 1 and input_dim > 1 must not panic on the grad_x reshape.
/// Regression test: `dot` can return a column-major grad_x here, and the reshape must tolerate it.
#[test]
fn simple_rnn_backward_units_one_multi_feature_reshapes() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new()).unwrap();
    let x = Array::from_elem((1, 2, 2), 0.5_f32).into_dyn();
    rnn.forward(&x).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let grad_x = rnn.backward(&grad).unwrap();
    assert_eq!(grad_x.shape(), &[1, 2, 2]);
}

/// SimpleRNN set_weights rejects a kernel with wrong shape
#[test]
fn simple_rnn_set_weights_wrong_kernel_shape_errors() {
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new()).unwrap();
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
    let mut lstm = LSTM::new(1, 1, Tanh::new()).unwrap();

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
    let out = lstm.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.17426972_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// Forget-gate bias of 1.0 yields a different final state than 0.0 across timesteps
#[test]
fn lstm_forget_bias_is_one_not_zero() {
    // 2 LSTMs: one with forget_bias=1.0 (the default), one with 0.0
    let mut lstm_correct = LSTM::new(1, 1, Tanh::new()).unwrap();
    let mut lstm_zero_forget = LSTM::new(1, 1, Tanh::new()).unwrap();

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
    let h_correct = lstm_correct.forward(&x).unwrap()[[0, 0]];
    let h_zero = lstm_zero_forget.forward(&x).unwrap()[[0, 0]];
    assert!(
        (h_correct - h_zero).abs() > 1e-4,
        "forget_bias=1.0 should produce a measurably different final state than forget_bias=0.0 \
         across timesteps, but h_correct={h_correct:.8}, h_zero={h_zero:.8}"
    );
}

/// LSTM over 2 timesteps threads cell state from t=0 into t=1
#[test]
fn lstm_forward_2step_cell_state_threads_through() {
    let mut lstm = LSTM::new(1, 1, Tanh::new()).unwrap();

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
    let out = lstm.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.33209477_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// LSTM output shape is (batch=2, units=3)
#[test]
fn lstm_output_shape_batch2_units3() {
    let mut lstm = LSTM::new(4, 3, Tanh::new()).unwrap();
    let x = Array::zeros((2, 5, 4)).into_dyn();
    let out = lstm.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
}

/// LSTM predict() matches forward() in eval mode
#[test]
fn lstm_predict_equals_forward() {
    let mut lstm = LSTM::new(2, 2, Tanh::new()).unwrap();

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

    let out_forward = lstm.forward(&x).unwrap();
    let out_predict = lstm.predict(&x).unwrap();

    assert_eq!(out_forward.shape(), out_predict.shape());
    assert_allclose(&out_forward, &out_predict, 1e-6);
}

/// LSTM constructor rejects a zero dimension, whichever argument carries it
#[test]
fn lstm_new_rejects_zero_dimension() {
    // (input_dim, units, name of the bad argument), 1 row per zero-valued argument
    let cases = [(0, 3, "input_dim"), (2, 0, "units")];
    for (input_dim, units, bad_arg) in cases {
        let err = LSTM::new(input_dim, units, Tanh::new()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for zero {bad_arg} (input_dim={input_dim}, units={units}), got: {err:?}"
        );
    }
}

/// LSTM forward rejects a non-3D input (2D tensor)
#[test]
fn lstm_forward_rejects_2d_input() {
    let mut lstm = LSTM::new(2, 1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = lstm.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// LSTM backward before forward returns ForwardPassNotRun
#[test]
fn lstm_backward_before_forward_errors() {
    let mut lstm = LSTM::new(2, 1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = lstm.backward(&grad).unwrap_err();
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
    let mut lstm = LSTM::new(2, 3, Tanh::new()).unwrap();

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
    let mut gru = GRU::new(1, 1, Tanh::new()).unwrap();

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
    let out = gru.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1]);

    // z = sigmoid(0.5) = 0.62245933, candidate = tanh(0.5) = 0.46211716, h_prev = 0
    // h = z*h_prev + (1-z)*candidate = 0.37754067 * 0.46211716 = 0.17446802
    let expected = Array::from_elem((1, 1), 0.174_468_02_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GRU over 2 timesteps blends previous hidden state through the update gate
#[test]
fn gru_forward_2step_hidden_state_blending() {
    let mut gru = GRU::new(1, 1, Tanh::new()).unwrap();

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
    let out = gru.forward(&x).unwrap();

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
    let mut gru = GRU::new(1, 1, Tanh::new()).unwrap();

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
    let out = gru.forward(&x).unwrap();

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
    let mut gru = GRU::new(1, 1, Tanh::new()).unwrap();

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
    let out = gru.forward(&x).unwrap();

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
    let mut gru = GRU::new(1, 1, Tanh::new()).unwrap();

    // [z | r | h] over 1 unit: z driven to 0, r left at 0, candidate kernel 1
    let kernel = Array2::from_shape_vec((1, 3), vec![-20.0_f32, 0.0, 1.0]).unwrap();
    let recurrent_kernel = Array2::zeros((1, 3));
    let bias = Array2::zeros((1, 3));
    gru.set_weights(kernel, recurrent_kernel, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = gru.forward(&x).unwrap();

    // z = sigmoid(-20) ~= 0, h_prev = 0, so h = (1 - z) * tanh(1) ~= tanh(1)
    let expected: f32 = 1.0_f32.tanh();
    assert!(
        (out[[0, 0]] - expected).abs() < 1e-4,
        "expected h_t about tanh(1)={expected:.8}, got {} - the fused blocks are not [z | r | h]",
        out[[0, 0]]
    );
}

/// GRU output shape is (batch=2, units=4)
#[test]
fn gru_output_shape_batch2_units4() {
    let mut gru = GRU::new(3, 4, Tanh::new()).unwrap();
    let x = Array::zeros((2, 5, 3)).into_dyn();
    let out = gru.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
}

/// GRU predict() matches forward() element-wise
#[test]
fn gru_predict_equals_forward() {
    let mut gru = GRU::new(2, 3, Tanh::new()).unwrap();

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

    let out_forward = gru.forward(&x).unwrap();
    let out_predict = gru.predict(&x).unwrap();

    assert_eq!(out_forward.shape(), out_predict.shape());
    assert_allclose(&out_forward, &out_predict, 1e-6);
}

/// GRU constructor rejects a zero dimension, whichever argument carries it
#[test]
fn gru_new_rejects_zero_dimension() {
    // (input_dim, units, name of the bad argument), 1 row per zero-valued argument
    let cases = [(0, 3, "input_dim"), (2, 0, "units")];
    for (input_dim, units, bad_arg) in cases {
        let err = GRU::new(input_dim, units, Tanh::new()).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for zero {bad_arg} (input_dim={input_dim}, units={units}), got: {err:?}"
        );
    }
}

/// GRU forward rejects a non-3D input (2D tensor)
#[test]
fn gru_forward_rejects_2d_input() {
    let mut gru = GRU::new(2, 1, Tanh::new()).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = gru.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// GRU forward rejects a 4D input
#[test]
fn gru_forward_rejects_4d_input() {
    let mut gru = GRU::new(2, 1, Tanh::new()).unwrap();
    let x = Array::zeros((1, 2, 3, 4)).into_dyn();
    let err = gru.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// GRU backward before forward returns ForwardPassNotRun
#[test]
fn gru_backward_before_forward_errors() {
    let mut gru = GRU::new(2, 1, Tanh::new()).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = gru.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun("GRU"))),
        "expected ForwardPassNotRun(\"GRU\"), got: {err:?}"
    );
}

/// GRU set_weights with wrong recurrent_kernel shape returns NnError::WeightShape
#[test]
fn gru_set_weights_wrong_shape_errors() {
    let mut gru = GRU::new(2, 3, Tanh::new()).unwrap();

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
    let mut rnn = SimpleRNN::new(1, 1, Activation::Tanh).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(k, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    // tanh(0.5) ~= 0.46211716, same as the concrete-Tanh test above
    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// LSTM accepts Activation enum
#[test]
fn lstm_accepts_activation_enum_tanh() {
    let mut lstm = LSTM::new(1, 1, Activation::Tanh).unwrap();

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
    let out = lstm.forward(&x).unwrap();

    let expected = Array::from_elem((1, 1), 0.17426972_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// GRU accepts Activation enum
#[test]
fn gru_accepts_activation_enum_tanh() {
    let mut gru = GRU::new(1, 1, Activation::Tanh).unwrap();

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
    let out = gru.forward(&x).unwrap();

    // z = sigmoid(0.5) = 0.62245933, candidate = tanh(0.5) = 0.46211716, h_prev = 0
    // h = z*h_prev + (1-z)*candidate = 0.37754067 * 0.46211716 = 0.17446802
    let expected = Array::from_elem((1, 1), 0.174_468_02_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// Determinism: same weights + same input => identical outputs

/// SimpleRNN: 2 forward passes with the same weights and input are bit-identical
#[test]
fn simple_rnn_forward_is_deterministic() {
    let mut rnn = SimpleRNN::new(2, 2, Tanh::new()).unwrap();

    let k = Array2::from_shape_vec((2, 2), vec![0.3, -0.1, 0.2, -0.4]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.2]).unwrap();
    let bias = Array2::from_shape_vec((1, 2), vec![0.05, -0.05]).unwrap();
    rnn.set_weights(k, rk, bias).unwrap();

    let x = Array::from_shape_vec((1, 3, 2), vec![0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
        .unwrap()
        .into_dyn();

    let out1 = rnn.forward(&x).unwrap();
    let out2 = rnn.forward(&x).unwrap();

    assert_eq!(out1, out2, "forward passes must be bit-identical");
}

/// GRU: 2 forward passes with the same weights are bit-identical
#[test]
fn gru_forward_is_deterministic() {
    let mut gru = GRU::new(2, 2, Tanh::new()).unwrap();

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

    let out1 = gru.forward(&x).unwrap();
    let out2 = gru.forward(&x).unwrap();

    assert_eq!(out1, out2, "GRU forward passes must be bit-identical");
}
// param_count(): trainable parameter formula per recurrent layer

/// SimpleRNN param_count = input_dim*units + units*units + units (1 gate)
#[test]
fn simple_rnn_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let rnn = SimpleRNN::new(3, 2, Tanh::new()).unwrap();
    // 3*2 + 2*2 + 2 = 6 + 4 + 2 = 12
    assert_eq!(rnn.param_count(), TrainingParameters::Trainable(12));
}

/// GRU param_count = 3 * (input_dim*units + units*units + units) (3 gates)
#[test]
fn gru_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let gru = GRU::new(3, 2, Tanh::new()).unwrap();
    // 3 * (3*2 + 2*2 + 2) = 3 * 12 = 36
    assert_eq!(gru.param_count(), TrainingParameters::Trainable(36));
}

/// LSTM param_count = 4 * (input_dim*units + units*units + units) (4 gates)
#[test]
fn lstm_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let lstm = LSTM::new(3, 2, Tanh::new()).unwrap();
    // 4 * (3*2 + 2*2 + 2) = 4 * 12 = 48
    assert_eq!(lstm.param_count(), TrainingParameters::Trainable(48));
}

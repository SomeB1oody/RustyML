//! Integration tests for the recurrent layers: SimpleRNN, LSTM, GRU.
//!
//! Every expected value is derived from the mathematical definition and hand-computed; no expected
//! value was obtained by running the layer and recording its output.  Backward/gradient values are
//! covered by tests/neural_network/gradient_check.rs and are intentionally not duplicated here.

use crate::common::assert_allclose;
use ndarray::{Array, Array2};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::traits::Layer;

// ══════════════════════════════════════════════════════════════════════════════════
// SimpleRNN
// ══════════════════════════════════════════════════════════════════════════════════

/// SimpleRNN with 1 timestep, 1 input feature, 1 unit, Tanh activation.
///
/// Weights: kernel=[[1.0]], recurrent_kernel=[[0.5]], bias=[[0.0]]
/// Input (1, 1, 1): x = [[[ 0.5 ]]]
///
/// Hand calculation:
///   h_prev = 0.0
///   z = x * kernel + h_prev * rk + bias = 0.5*1.0 + 0.0*0.5 + 0.0 = 0.5
///   h_t = tanh(0.5) ≈ 0.46211716
#[test]
fn simple_rnn_forward_1step_1unit_tanh() {
    let mut rnn = SimpleRNN::new(1, 1, Tanh::new(), None).unwrap();

    let kernel = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::from_elem((1, 1), 0.5_f32);
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(kernel, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    // Output shape must be (batch=1, units=1)
    assert_eq!(out.shape(), &[1, 1]);

    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// SimpleRNN with 2 timesteps, 1 input feature, 1 unit, Tanh activation.
///
/// Weights: kernel=[[1.0]], recurrent_kernel=[[1.0]], bias=[[0.0]]
/// Input (1, 2, 1): x[0]=0.3, x[1]=0.7
///
/// Hand calculation:
///   t=0: h_prev=0, z=0.3*1+0*1+0=0.3, h_0=tanh(0.3)≈0.29131261
///   t=1: h_prev=0.29131261, z=0.7+0.29131261=0.99131261, h_1=tanh(0.99131261)≈0.75792147
#[test]
fn simple_rnn_forward_2step_tanh_state_threading() {
    let mut rnn = SimpleRNN::new(1, 1, Tanh::new(), None).unwrap();

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

/// SimpleRNN with ReLU activation and a non-negative pre-activation value.
///
/// Weights: kernel=[[0.5]], recurrent_kernel=[[0.5]], bias=[[0.0]]
/// Input (1, 1, 1): x=1.0
///
/// Hand calculation:
///   z = 1.0*0.5 + 0.0*0.5 + 0.0 = 0.5
///   h_t = ReLU(0.5) = 0.5  (identity for positive input)
#[test]
fn simple_rnn_forward_relu_positive() {
    let mut rnn = SimpleRNN::new(1, 1, ReLU::new(), None).unwrap();

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

/// SimpleRNN: ReLU clips a negative pre-activation to exactly 0.
///
/// Weights: kernel=[[-1.0]], recurrent_kernel=[[0.0]], bias=[[0.0]]
/// Input (1, 1, 1): x=0.8
///
/// Hand calculation:
///   z = 0.8*(-1.0) + 0.0 + 0.0 = -0.8
///   h_t = ReLU(-0.8) = 0.0
#[test]
fn simple_rnn_relu_negative_preactivation_is_zero() {
    let mut rnn = SimpleRNN::new(1, 1, ReLU::new(), None).unwrap();

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

/// SimpleRNN output shape with batch=3, units=2.
///
/// Checks the output shape contract (batch, units) without asserting specific values.
#[test]
fn simple_rnn_output_shape_batch3_units2() {
    let mut rnn = SimpleRNN::new(4, 2, Tanh::new(), None).unwrap();
    // 3 batches, 5 timesteps, 4 input features
    let x = Array::zeros((3, 5, 4)).into_dyn();
    let out = rnn.forward(&x).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
}

/// SimpleRNN: predict() produces the same output as forward() (eval-mode equivalence).
///
/// Both must agree element-wise within floating-point tolerance.
#[test]
fn simple_rnn_predict_equals_forward() {
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new(), None).unwrap();

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

/// SimpleRNN constructor rejects input_dim=0.
#[test]
fn simple_rnn_new_rejects_zero_input_dim() {
    let err = SimpleRNN::new(0, 3, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// SimpleRNN constructor rejects units=0.
#[test]
fn simple_rnn_new_rejects_zero_units() {
    let err = SimpleRNN::new(2, 0, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// SimpleRNN forward rejects a non-3D input (2D tensor).
#[test]
fn simple_rnn_forward_rejects_2d_input() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new(), None).unwrap();
    let x = Array::zeros((4, 2)).into_dyn(); // 2D, not 3D
    let err = rnn.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// SimpleRNN forward rejects a 1D input.
#[test]
fn simple_rnn_forward_rejects_1d_input() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new(), None).unwrap();
    let x = Array::zeros(4).into_dyn(); // 1D
    let err = rnn.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// SimpleRNN backward before forward returns ForwardPassNotRun error.
#[test]
fn simple_rnn_backward_before_forward_errors() {
    let mut rnn = SimpleRNN::new(2, 1, Tanh::new(), None).unwrap();
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

/// SimpleRNN set_weights rejects kernel with wrong shape.
#[test]
fn simple_rnn_set_weights_wrong_kernel_shape_errors() {
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new(), None).unwrap();
    // kernel should be (2,3), we pass (3,2)
    let bad_kernel = Array2::zeros((3, 2));
    let rk = Array2::zeros((3, 3));
    let bias = Array2::zeros((1, 3));
    let err = rnn.set_weights(bad_kernel, rk, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got: {err:?}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════════
// LSTM
// ══════════════════════════════════════════════════════════════════════════════════

/// LSTM with 1 timestep, 1 input feature, 1 unit, Tanh activation.
///
/// Weights: all kernels=[[1.0]], all recurrent_kernels=[[0.0]], all biases=[[0.0]]
///          except forget_bias=[[1.0]] (the default initialization).
/// Input (1, 1, 1): x=0.5
///
/// Hand calculation (h_prev=0, c_prev=0):
///   i_raw = 0.5*1+0+0 = 0.5,  i_t = sigmoid(0.5) ≈ 0.62245933
///   f_raw = 0.5*1+0+1.0 = 1.5, f_t = sigmoid(1.5) ≈ 0.81757448
///   g_raw = 0.5*1+0+0 = 0.5,  g_t = tanh(0.5) ≈ 0.46211716
///   o_raw = 0.5*1+0+0 = 0.5,  o_t = sigmoid(0.5) ≈ 0.62245933
///   c_t = f_t*c_prev + i_t*g_t = 0 + 0.62245933*0.46211716 ≈ 0.28764914
///   c_t_activated = tanh(0.28764914) ≈ 0.27996965
///   h_t = o_t * c_t_activated = 0.62245933*0.27996965 ≈ 0.17426972
#[test]
fn lstm_forward_1step_1unit_tanh() {
    let mut lstm = LSTM::new(1, 1, Tanh::new(), None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32); // forget bias = 1.0

    lstm.set_weights(
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

    // Expected: h_t ≈ 0.17426972
    let expected = Array::from_elem((1, 1), 0.17426972_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// LSTM: forget gate bias defaults to 1.0 — validates with a purely structural test.
///
/// The forget gate's bias materially affects the output: an LSTM whose forget-gate bias is 1.0
/// (the library's default, see `Gate::new(.., .., 1.0)` for the forget gate) must produce a
/// different result than one whose forget-gate bias is 0.0. The difference is only observable from
/// the SECOND timestep onward, because at step 1 the previous cell state is 0 and `f·c_prev = 0`.
#[test]
fn lstm_forget_bias_is_one_not_zero() {
    // Two LSTMs: one with forget_bias=1.0 (correct default path), one with 0.0.
    let mut lstm_correct = LSTM::new(1, 1, Tanh::new(), None).unwrap();
    let mut lstm_zero_forget = LSTM::new(1, 1, Tanh::new(), None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm_correct
        .set_weights(
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
        .set_weights(
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

    // TWO timesteps: on step 1 the cell state c_0 = 0, so the forget gate (f·c_prev) has NO effect
    // regardless of its bias; only from step 2 onward, where c_prev != 0, does the forget bias
    // change the output. So forget_bias=1.0 must yield a different FINAL hidden state than 0.0.
    // (A 1-timestep input would give identical outputs — the original bug in this test.)
    let x = Array::from_elem((1, 2, 1), 0.5_f32).into_dyn();
    let h_correct = lstm_correct.forward(&x).unwrap()[[0, 0]];
    let h_zero = lstm_zero_forget.forward(&x).unwrap()[[0, 0]];
    assert!(
        (h_correct - h_zero).abs() > 1e-4,
        "forget_bias=1.0 should produce a measurably different final state than forget_bias=0.0 \
         across timesteps, but h_correct={h_correct:.8}, h_zero={h_zero:.8}"
    );
}

/// LSTM 2-step, 1-unit: verifies that cell state from t=0 propagates to t=1.
///
/// Weights: all kernels=[[1.0]], all recurrent_kernels=[[0.0]],
///          biases: i=0, f=1.0, g=0, o=0.
/// Input (1, 2, 1): x[0]=0.3, x[1]=0.7
///
/// Hand calculation (all h_prev=0, c_prev=0 at start):
///   t=0: i_t=sigmoid(0.3)≈0.57444252, f_t=sigmoid(1.3)≈0.78583498,
///        g_t=tanh(0.3)≈0.29131261, o_t=sigmoid(0.3)≈0.57444252
///        c_t=0+0.57444252*0.29131261≈0.16734235
///        c_act=tanh(0.16734235)≈0.16579760, h_t=0.57444252*0.16579760≈0.09524119
///   t=1: h_prev=0.09524119, c_prev=0.16734235
///        i_raw=0.7, i_t=sigmoid(0.7)≈0.66818777
///        f_raw=0.7+1.0=1.7, f_t=sigmoid(1.7)≈0.84553473
///        g_raw=0.7, g_t=tanh(0.7)≈0.60436778
///        o_raw=0.7, o_t=sigmoid(0.7)≈0.66818777
///        c_t=0.84553473*0.16734235+0.66818777*0.60436778≈0.14152+0.40381≈0.54532
///        c_act=tanh(0.54532)≈0.49700815
///        h_t=0.66818777*0.49700815≈0.33209477
#[test]
fn lstm_forward_2step_cell_state_threads_through() {
    let mut lstm = LSTM::new(1, 1, Tanh::new(), None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm.set_weights(
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

/// LSTM output shape with batch=2, units=3.
#[test]
fn lstm_output_shape_batch2_units3() {
    let mut lstm = LSTM::new(4, 3, Tanh::new(), None).unwrap();
    let x = Array::zeros((2, 5, 4)).into_dyn();
    let out = lstm.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
}

/// LSTM: predict() matches forward() in eval mode.
#[test]
fn lstm_predict_equals_forward() {
    let mut lstm = LSTM::new(2, 2, Tanh::new(), None).unwrap();

    // Use non-trivial weights to ensure both paths exercise the same computation.
    let kernel = Array2::from_shape_vec((2, 2), vec![0.3, -0.3, 0.2, -0.2]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    let b_zero = Array2::zeros((1, 2));
    let b_one = Array2::from_elem((1, 2), 1.0_f32);

    lstm.set_weights(
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

/// LSTM constructor rejects input_dim=0.
#[test]
fn lstm_new_rejects_zero_input_dim() {
    let err = LSTM::new(0, 3, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// LSTM constructor rejects units=0.
#[test]
fn lstm_new_rejects_zero_units() {
    let err = LSTM::new(2, 0, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// LSTM forward rejects a non-3D input (2D tensor).
#[test]
fn lstm_forward_rejects_2d_input() {
    let mut lstm = LSTM::new(2, 1, Tanh::new(), None).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = lstm.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// LSTM backward before forward returns ForwardPassNotRun.
#[test]
fn lstm_backward_before_forward_errors() {
    let mut lstm = LSTM::new(2, 1, Tanh::new(), None).unwrap();
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

/// LSTM set_weights with wrong kernel shape returns NnError::WeightShape.
#[test]
fn lstm_set_weights_wrong_shape_errors() {
    let mut lstm = LSTM::new(2, 3, Tanh::new(), None).unwrap();

    let good_k = Array2::zeros((2, 3));
    let good_rk = Array2::zeros((3, 3));
    let good_b = Array2::zeros((1, 3));

    // Pass wrong shape for cell_kernel (should be (2,3), we give (3,2))
    let bad_k = Array2::zeros((3, 2));
    let err = lstm
        .set_weights(
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

// ══════════════════════════════════════════════════════════════════════════════════
// GRU
// ══════════════════════════════════════════════════════════════════════════════════

/// GRU with 1 timestep, 1 input feature, 1 unit, Tanh activation.
///
/// Weights: all kernels=[[1.0]], all recurrent_kernels=[[0.0]], all biases=[[0.0]].
/// Input (1, 1, 1): x=0.5, h_prev=0.
///
/// Hand calculation:
///   r_raw = 0.5*1+0*0+0 = 0.5, r_t = sigmoid(0.5) ≈ 0.62245933
///   z_raw = 0.5*1+0*0+0 = 0.5, z_t = sigmoid(0.5) ≈ 0.62245933
///   r_h = r_t * h_prev = 0
///   h_cand_raw = 0.5*1 + 0*1 + 0 = 0.5
///   h_cand = tanh(0.5) ≈ 0.46211716
///   h_t = (1-0.62245933)*0 + 0.62245933*0.46211716 ≈ 0.28764914
#[test]
fn gru_forward_1step_1unit_tanh() {
    let mut gru = GRU::new(1, 1, Tanh::new(), None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_weights(
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

    // Expected h_t ≈ 0.28764914
    let expected = Array::from_elem((1, 1), 0.28764914_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// GRU 2-step: verifies that previous hidden state feeds into reset and update gates.
///
/// Weights: all kernels=[[1.0]], all recurrent_kernels=[[0.0]], all biases=[[0.0]].
/// Input (1, 2, 1): x[0]=0.3, x[1]=0.7
///
/// Hand calculation:
///   t=0: h_prev=0
///     r_t=sigmoid(0.3)≈0.57444252, z_t=sigmoid(0.3)≈0.57444252
///     r_h=0, h_cand_raw=0.3, h_cand=tanh(0.3)≈0.29131261
///     h_t=(1-0.57444252)*0 + 0.57444252*0.29131261 ≈ 0.16734235
///   t=1: h_prev=0.16734235 (recurrent kernel=0, so only input matters for gates)
///     r_t=sigmoid(0.7)≈0.66818777, z_t=sigmoid(0.7)≈0.66818777
///     r_h = 0.66818777 * 0.16734235 ≈ 0.11183 (but rk=0 → h_cand_raw ignores r_h)
///     h_cand_raw = 0.7*1 + r_h*0 + 0 = 0.7
///     h_cand = tanh(0.7) ≈ 0.60436778
///     h_t=(1-0.66818777)*0.16734235 + 0.66818777*0.60436778 ≈ 0.05552+0.40384 ≈ 0.45936
///
/// NOTE: with recurrent_kernels=0 the r_h term contributes 0 to h_cand_raw, so
///   h_cand_raw = x_t * candidate_kernel = 0.7.  The update-gate blending still uses h_prev.
///
/// Precise: (1-0.66818777)*0.16734235=0.33181223*0.16734235≈0.055517
///          + 0.66818777*tanh(0.7)=0.66818777*0.60436778≈0.404040
///          = 0.459557 ≈ 0.45936 (matches Python: 0.45935740)
#[test]
fn gru_forward_2step_hidden_state_blending() {
    let mut gru = GRU::new(1, 1, Tanh::new(), None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_weights(
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

    // Hand calculation (precise):
    //   (1-0.66818777)*0.16734235 + 0.66818777*tanh(0.7)
    //   = 0.33181223*0.16734235 + 0.66818777*0.60436778
    //   ≈ 0.055517 + 0.404040 = 0.45935740
    let expected = Array::from_elem((1, 1), 0.459_357_4_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-5);
}

/// GRU: update gate z=0 → hidden state unchanged from h_prev.
///
/// When z_t≈0, h_t ≈ (1-0)*h_prev + 0*h_cand = h_prev.
/// We achieve z_t≈0 by setting update_kernel to a large negative value and
/// using zero recurrent weights and zero bias, with x=1.0.
///
/// z_raw = 1.0 * (-20.0) + 0 + 0 = -20 => z_t = sigmoid(-20) ≈ 2e-9 ≈ 0
/// h_t ≈ (1-0)*h_prev + 0*h_cand = h_prev = 0
///
/// With h_prev=0, h_t must be ≈ 0 regardless of the candidate value.
#[test]
fn gru_update_gate_zero_leaves_hidden_unchanged() {
    let mut gru = GRU::new(1, 1, Tanh::new(), None).unwrap();

    let k_zero = Array2::zeros((1, 1));
    let k_neg = Array2::from_elem((1, 1), -20.0_f32); // drives z_t ≈ 0
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_weights(
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

    // h_prev=0 and z≈0 means h_t ≈ 0
    assert_eq!(out.shape(), &[1, 1]);
    assert!(
        out[[0, 0]].abs() < 1e-4,
        "expected h_t≈0 when z≈0, got: {}",
        out[[0, 0]]
    );
}

/// GRU: update gate z=1 → hidden state = h_candidate.
///
/// When z_t≈1, h_t ≈ 0*h_prev + 1*h_cand = h_cand.
/// Achieved by setting update_kernel to a large positive value: x=1.0,
/// z_raw=20.0 => z_t=sigmoid(20)≈1.
///
/// candidate_kernel=1.0, so h_cand_raw=1.0, h_cand=tanh(1.0)≈0.76159416.
/// h_t ≈ 0*0 + 1*0.76159416 ≈ 0.76159416
#[test]
fn gru_update_gate_one_replaces_hidden_with_candidate() {
    let mut gru = GRU::new(1, 1, Tanh::new(), None).unwrap();

    let k_one = Array2::from_elem((1, 1), 1.0_f32);
    let k_large = Array2::from_elem((1, 1), 20.0_f32); // drives z_t ≈ 1
    let k_zero = Array2::zeros((1, 1));
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_weights(
        k_zero.clone(),
        rk.clone(),
        bias.clone(), // reset (r≈0.5, doesn't matter since h_prev=0)
        k_large.clone(),
        rk.clone(),
        bias.clone(), // update (z≈1)
        k_one.clone(),
        rk.clone(),
        bias.clone(), // candidate kernel=1.0
    )
    .unwrap();

    let x = Array::from_elem((1, 1, 1), 1.0_f32).into_dyn();
    let out = gru.forward(&x).unwrap();

    // h_cand = tanh(1.0) ≈ 0.76159416
    // h_t ≈ 0*h_prev + 1*h_cand ≈ 0.76159416
    let expected_h_cand: f32 = 1.0_f32.tanh();
    assert_eq!(out.shape(), &[1, 1]);
    assert!(
        (out[[0, 0]] - expected_h_cand).abs() < 1e-4,
        "expected h_t≈tanh(1)={expected_h_cand:.8}, got: {}",
        out[[0, 0]]
    );
}

/// GRU output shape with batch=2, units=4.
#[test]
fn gru_output_shape_batch2_units4() {
    let mut gru = GRU::new(3, 4, Tanh::new(), None).unwrap();
    let x = Array::zeros((2, 5, 3)).into_dyn();
    let out = gru.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
}

/// GRU: predict() produces the same output as forward().
#[test]
fn gru_predict_equals_forward() {
    let mut gru = GRU::new(2, 3, Tanh::new(), None).unwrap();

    let k = Array2::from_shape_vec((2, 3), vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).unwrap();
    let rk = Array2::from_shape_vec((3, 3), vec![0.1, 0.0, -0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.1])
        .unwrap();
    let bias = Array2::from_shape_vec((1, 3), vec![0.1, -0.1, 0.2]).unwrap();

    gru.set_weights(
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

/// GRU constructor rejects input_dim=0.
#[test]
fn gru_new_rejects_zero_input_dim() {
    let err = GRU::new(0, 3, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// GRU constructor rejects units=0.
#[test]
fn gru_new_rejects_zero_units() {
    let err = GRU::new(2, 0, Tanh::new(), None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got: {err:?}"
    );
}

/// GRU forward rejects a non-3D input (2D tensor).
#[test]
fn gru_forward_rejects_2d_input() {
    let mut gru = GRU::new(2, 1, Tanh::new(), None).unwrap();
    let x = Array::zeros((4, 2)).into_dyn();
    let err = gru.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// GRU forward rejects a 4D input.
#[test]
fn gru_forward_rejects_4d_input() {
    let mut gru = GRU::new(2, 1, Tanh::new(), None).unwrap();
    let x = Array::zeros((1, 2, 3, 4)).into_dyn();
    let err = gru.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

/// GRU backward before forward returns ForwardPassNotRun.
#[test]
fn gru_backward_before_forward_errors() {
    let mut gru = GRU::new(2, 1, Tanh::new(), None).unwrap();
    let grad = Array::ones((1, 1)).into_dyn();
    let err = gru.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun("GRU"))),
        "expected ForwardPassNotRun(\"GRU\"), got: {err:?}"
    );
}

/// GRU set_weights with wrong recurrent_kernel shape returns NnError::WeightShape.
#[test]
fn gru_set_weights_wrong_shape_errors() {
    let mut gru = GRU::new(2, 3, Tanh::new(), None).unwrap();

    let good_k = Array2::zeros((2, 3));
    let good_rk = Array2::zeros((3, 3));
    let good_b = Array2::zeros((1, 3));

    // Wrong shape for update_recurrent_kernel: (2,3) instead of (3,3)
    let bad_rk = Array2::zeros((2, 3));
    let err = gru
        .set_weights(
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

// ══════════════════════════════════════════════════════════════════════════════════
// Cross-layer: Activation enum interface vs concrete activation types
// ══════════════════════════════════════════════════════════════════════════════════

/// SimpleRNN accepts Activation enum values (not just concrete activation structs).
#[test]
fn simple_rnn_accepts_activation_enum_tanh() {
    let mut rnn = SimpleRNN::new(1, 1, Activation::Tanh, None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));
    rnn.set_weights(k, rk, bias).unwrap();

    let x = Array::from_elem((1, 1, 1), 0.5_f32).into_dyn();
    let out = rnn.forward(&x).unwrap();

    // tanh(0.5) ≈ 0.46211716 — same as test above
    let expected = Array::from_elem((1, 1), 0.46211716_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

/// LSTM accepts Activation enum.
#[test]
fn lstm_accepts_activation_enum_tanh() {
    let mut lstm = LSTM::new(1, 1, Activation::Tanh, None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let b_zero = Array2::zeros((1, 1));
    let b_one = Array2::from_elem((1, 1), 1.0_f32);

    lstm.set_weights(
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

/// GRU accepts Activation enum.
#[test]
fn gru_accepts_activation_enum_tanh() {
    let mut gru = GRU::new(1, 1, Activation::Tanh, None).unwrap();

    let k = Array2::from_elem((1, 1), 1.0_f32);
    let rk = Array2::zeros((1, 1));
    let bias = Array2::zeros((1, 1));

    gru.set_weights(
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

    let expected = Array::from_elem((1, 1), 0.28764914_f32).into_dyn();
    assert_allclose(&out, &expected, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════════
// Determinism: same weights + same input → identical outputs
// ══════════════════════════════════════════════════════════════════════════════════

/// Two forward passes with the same weights and input produce bit-identical outputs.
#[test]
fn simple_rnn_forward_is_deterministic() {
    let mut rnn = SimpleRNN::new(2, 2, Tanh::new(), None).unwrap();

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

/// Two forward passes of GRU with the same weights are bit-identical.
#[test]
fn gru_forward_is_deterministic() {
    let mut gru = GRU::new(2, 2, Tanh::new(), None).unwrap();

    let k = Array2::from_shape_vec((2, 2), vec![0.3, -0.1, 0.2, -0.4]).unwrap();
    let rk = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.2]).unwrap();
    let bias = Array2::from_shape_vec((1, 2), vec![0.05, -0.05]).unwrap();

    gru.set_weights(
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
// ══════════════════════════════════════════════════════════════════════════════════
// param_count(): trainable parameter formula per recurrent layer
// ══════════════════════════════════════════════════════════════════════════════════

/// SimpleRNN::param_count() = input_dim*units + units*units + units (×1 gate).
///
/// Derivation (input_dim=3, units=2):
///   kernel:           input_dim*units = 3*2 = 6
///   recurrent_kernel: units*units     = 2*2 = 4
///   bias:             units           =       2
///   total = 6 + 4 + 2 = 12
#[test]
fn simple_rnn_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let rnn = SimpleRNN::new(3, 2, Tanh::new(), None).unwrap();
    // 3*2 + 2*2 + 2 = 6 + 4 + 2 = 12
    assert_eq!(rnn.param_count(), TrainingParameters::Trainable(12));
}

/// GRU::param_count() = 3 * (input_dim*units + units*units + units) (3 gates).
///
/// Derivation (input_dim=3, units=2):
///   per-gate = 3*2 + 2*2 + 2 = 12
///   GRU has 3 gates (reset, update, candidate): 3 * 12 = 36
#[test]
fn gru_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let gru = GRU::new(3, 2, Tanh::new(), None).unwrap();
    // 3 * (3*2 + 2*2 + 2) = 3 * 12 = 36
    assert_eq!(gru.param_count(), TrainingParameters::Trainable(36));
}

/// LSTM::param_count() = 4 * (input_dim*units + units*units + units) (4 gates).
///
/// Derivation (input_dim=3, units=2):
///   per-gate = 3*2 + 2*2 + 2 = 12
///   LSTM has 4 gates (input, forget, cell, output): 4 * 12 = 48
#[test]
fn lstm_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let lstm = LSTM::new(3, 2, Tanh::new(), None).unwrap();
    // 4 * (3*2 + 2*2 + 2) = 4 * 12 = 48
    assert_eq!(lstm.param_count(), TrainingParameters::Trainable(48));
}

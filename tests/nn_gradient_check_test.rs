#![cfg(feature = "neural_network")]

//! Numerical (finite-difference) gradient checks for layer backward passes.
//!
//! For each layer we set `L = sum(output)`, so `dL/dx = backward(ones)`. We then compare that
//! analytic input gradient against a central finite-difference estimate of `dL/dx`. This catches
//! gradient bugs that shape-only or "loss decreased" convergence tests miss — in particular it
//! guards the SimpleRNN BPTT per-timestep activation-derivative fix.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::traits::Layer;

/// Compares `layer.backward(ones)` against a central finite-difference estimate of `d sum(output)/dx`.
fn check_input_gradient(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    // Analytic input gradient: with L = sum(output), dL/dx = backward(ones).
    let out = layer.forward(x).unwrap();
    let upstream = Tensor::ones(out.raw_dim());
    let analytic = layer.backward(&upstream).unwrap();
    assert_eq!(
        analytic.shape(),
        x.shape(),
        "input-gradient shape must match input shape"
    );

    let analytic_flat: Vec<f32> = analytic.iter().cloned().collect();
    let mut x_flat: Vec<f32> = x.iter().cloned().collect();

    for i in 0..x_flat.len() {
        let orig = x_flat[i];

        x_flat[i] = orig + eps;
        let xp = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_plus: f32 = layer.forward(&xp).unwrap().sum();

        x_flat[i] = orig - eps;
        let xm = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_minus: f32 = layer.forward(&xm).unwrap().sum();

        x_flat[i] = orig;

        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert_abs_diff_eq!(analytic_flat[i], numeric, epsilon = tol);
    }
}

#[test]
fn dense_input_gradient_matches_finite_difference() {
    // Linear activation keeps the layer smooth (no ReLU kink at 0).
    let mut dense = Dense::new(3, 2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((4, 3), (0..12).map(|v| 0.1 * v as f32 - 0.5).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut dense, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_input_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(
        2,
        (2, 2),
        vec![1, 1, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 4, 4),
        (0..16).map(|v| 0.1 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_input_gradient_matches_finite_difference() {
    // Linear activation makes the convolution exactly linear in its input, so the central
    // finite-difference estimate matches the analytic gradient tightly.
    let mut conv = Conv1D::new(2, 2, vec![1, 1, 5], 1, PaddingType::Valid, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 1, 5), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_input_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 3, 3, 3),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn separable_conv2d_input_gradient_matches_finite_difference() {
    let mut conv = SeparableConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec((1, 2, 4, 4), (0..32).map(|v| 0.05 * v as f32 - 0.7).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_input_gradient_matches_finite_difference() {
    // `new` Xavier-initializes the weights, so the layer is a genuine (non-constant) map.
    let mut conv =
        DepthwiseConv2D::new(2, (2, 2), vec![1, 2, 4, 4], (1, 1), PaddingType::Valid, Linear::new())
            .unwrap();
    let x = Array::from_shape_vec((1, 2, 4, 4), (0..32).map(|v| 0.05 * v as f32 - 0.7).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn simple_rnn_input_gradient_matches_finite_difference() {
    // Multi-timestep + Tanh: this fails if BPTT reuses a single stale activation cache, so it
    // directly guards the per-timestep activation-derivative fix.
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut rnn, &x, 1e-3, 2e-2);
}

#[test]
fn lstm_input_gradient_matches_finite_difference() {
    // Design A: the configurable activation is the per-timestep candidate AND cell-state
    // nonlinearity. This finite-difference check guards the BPTT through both activation
    // derivatives (a stale or wrong derivative here would fail).
    let mut lstm = LSTM::new(2, 3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut lstm, &x, 1e-3, 3e-2);
}

#[test]
fn gru_input_gradient_matches_finite_difference() {
    // Design A: the configurable activation is the per-timestep candidate nonlinearity.
    let mut gru = GRU::new(2, 3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut gru, &x, 1e-3, 3e-2);
}

#[test]
fn batch_normalization_input_gradient_matches_finite_difference() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    let x = Array::from_shape_vec(
        (4, 3),
        vec![
            0.5, -1.0, 2.0, 1.5, 0.2, -0.7, -1.2, 0.8, 1.1, 0.3, -0.4, 0.9,
        ],
    )
    .unwrap()
    .into_dyn();
    // Batch-norm gradients are more sensitive, so use a slightly looser tolerance.
    check_input_gradient(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn conv1d_same_padding_output_length_is_ceil_of_input() {
    // `Same` padding output length must be ceil(input_len / stride), computed from the ORIGINAL
    // (unpadded) length — not from the padded buffer.
    let cases = [
        // (input_len, kernel, stride, expected_out_len)
        (10usize, 3usize, 1usize, 10usize),
        (10, 3, 2, 5),
        (8, 5, 1, 8),
        (7, 3, 2, 4),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut conv = Conv1D::new(
            2,
            kernel,
            vec![1, 1, len],
            stride,
            PaddingType::Same,
            Linear::new(),
        )
        .unwrap();
        let x = Array::ones((1, 1, len)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, 2, expected],
            "Conv1D Same: input_len={}, kernel={}, stride={}",
            len,
            kernel,
            stride
        );
    }
}

/// Compares the analytic parameter gradients (from `layer.parameters()` after `backward(ones)`)
/// against a central finite-difference estimate of `d sum(output)/d param`.
///
/// This is generic over any trainable layer: `parameters()` exposes each weight/bias tensor as a
/// flat `(value, grad)` slice pair, so we perturb each value in place and re-run the forward pass.
/// It complements [`check_input_gradient`] by guarding the *weight* gradient paths (e.g. the
/// parallelized SeparableConv2D backward).
fn check_weight_gradient(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    let out = layer.forward(x).unwrap();
    let upstream = Tensor::ones(out.raw_dim());
    layer.backward(&upstream).unwrap();

    // Snapshot each parameter tensor's current values and analytic gradients.
    let params: Vec<(Vec<f32>, Vec<f32>)> = layer
        .parameters()
        .into_iter()
        .map(|pg| (pg.value.to_vec(), pg.grad.to_vec()))
        .collect();
    assert!(!params.is_empty(), "layer exposes no parameters to check");

    for (p_idx, (values, grads)) in params.iter().enumerate() {
        for i in 0..values.len() {
            let orig = values[i];

            // `parameters()[p_idx].value` is a mutable view into the live weight array, so writing
            // through it perturbs the actual parameter; the temporary Vec is only a handle.
            layer.parameters()[p_idx].value[i] = orig + eps;
            let l_plus: f32 = layer.forward(x).unwrap().sum();

            layer.parameters()[p_idx].value[i] = orig - eps;
            let l_minus: f32 = layer.forward(x).unwrap().sum();

            layer.parameters()[p_idx].value[i] = orig;

            let numeric = (l_plus - l_minus) / (2.0 * eps);
            assert_abs_diff_eq!(grads[i], numeric, epsilon = tol);
        }
    }
}

#[test]
fn dense_weight_gradient_matches_finite_difference() {
    let mut dense = Dense::new(3, 2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((4, 3), (0..12).map(|v| 0.1 * v as f32 - 0.5).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut dense, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(2, 2, vec![1, 1, 5], 1, PaddingType::Valid, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 1, 5), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(
        2,
        (2, 2),
        vec![1, 1, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec((1, 1, 4, 4), (0..16).map(|v| 0.1 * v as f32 - 0.7).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 3, 3, 3),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn separable_conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = SeparableConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
    )
    .unwrap();
    let x = Array::from_shape_vec((1, 2, 4, 4), (0..32).map(|v| 0.05 * v as f32 - 0.7).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_weight_gradient_matches_finite_difference() {
    let mut conv =
        DepthwiseConv2D::new(2, (2, 2), vec![1, 2, 4, 4], (1, 1), PaddingType::Valid, Linear::new())
            .unwrap();
    let x = Array::from_shape_vec((1, 2, 4, 4), (0..32).map(|v| 0.05 * v as f32 - 0.7).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

//! Numerical (finite-difference) gradient checks for layer backward passes
//!
//! For each layer, L = sum(output), so dL/dx = backward(ones). The analytic input gradient is
//! compared against a central finite-difference estimate of dL/dx, catching gradient bugs that
//! shape-only or "loss decreased" convergence tests miss - in particular the SimpleRNN BPTT
//! per-timestep activation-derivative fix

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::softmax::Softmax;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D;
use rustyml::neural_network::layers::pooling::average_pooling_2d::AveragePooling2D;
use rustyml::neural_network::layers::pooling::average_pooling_3d::AveragePooling3D;
use rustyml::neural_network::layers::pooling::global_average_pooling_1d::GlobalAveragePooling1D;
use rustyml::neural_network::layers::pooling::global_average_pooling_2d::GlobalAveragePooling2D;
use rustyml::neural_network::layers::pooling::global_average_pooling_3d::GlobalAveragePooling3D;
use rustyml::neural_network::layers::pooling::global_max_pooling_1d::GlobalMaxPooling1D;
use rustyml::neural_network::layers::pooling::global_max_pooling_2d::GlobalMaxPooling2D;
use rustyml::neural_network::layers::pooling::global_max_pooling_3d::GlobalMaxPooling3D;
use rustyml::neural_network::layers::pooling::max_pooling_1d::MaxPooling1D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::pooling::max_pooling_3d::MaxPooling3D;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::{
    LayerNormalization, LayerNormalizationAxis,
};
use rustyml::neural_network::traits::Layer;

/// Compares `layer.backward(ones)` against a central finite-difference estimate of d sum(output)/dx
fn check_input_gradient(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    // Analytic input gradient: with L = sum(output), dL/dx = backward(ones)
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
    // Linear activation keeps the layer smooth (no ReLU kink at 0)
    let mut dense = Dense::new(3, 2, Linear::new(), None).unwrap();
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
        None,
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
    // finite-difference estimate matches the analytic gradient tightly
    let mut conv = Conv1D::new(
        2,
        2,
        vec![1, 1, 5],
        1,
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
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
        None,
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
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn separable_conv2d_same_padding_input_gradient_matches_finite_difference() {
    // A 3x3 kernel under `Same` adds leading padding; guards that backward accumulates the input
    // gradient in padded coordinates and strips the padding, consistent with the padded forward pass
    let mut conv = SeparableConv2D::new(
        2,
        (3, 3),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Same,
        1,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_input_gradient_matches_finite_difference() {
    // `new` Xavier-initializes the weights, so the layer is a genuine (non-constant) map
    let mut conv = DepthwiseConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn simple_rnn_input_gradient_matches_finite_difference() {
    // Multi-timestep + Tanh: fails if BPTT reuses a single stale activation cache, so it guards
    // the per-timestep activation-derivative fix
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new(), None).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut rnn, &x, 1e-3, 2e-2);
}

#[test]
fn lstm_input_gradient_matches_finite_difference() {
    // Design A: the configurable activation is the per-timestep candidate AND cell-state
    // nonlinearity; guards BPTT through both activation derivatives
    let mut lstm = LSTM::new(2, 3, Tanh::new(), None).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut lstm, &x, 1e-3, 3e-2);
}

#[test]
fn gru_input_gradient_matches_finite_difference() {
    // Design A: the configurable activation is the per-timestep candidate nonlinearity
    let mut gru = GRU::new(2, 3, Tanh::new(), None).unwrap();
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
    // Batch-norm gradients are more sensitive, so use a slightly looser tolerance
    check_input_gradient(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn conv1d_same_padding_output_length_is_ceil_of_input() {
    // `Same` padding output length must be ceil(input_len / stride), computed from the ORIGINAL
    // (unpadded) length, not from the padded buffer
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
            None,
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

/// Compares analytic parameter gradients (from `layer.parameters()` after `backward(ones)`) against
/// a central finite-difference estimate of d sum(output)/d param
///
/// Generic over any trainable layer: `parameters()` exposes each weight/bias tensor as a flat
/// `(value, grad)` slice pair, so each value is perturbed in place and the forward pass re-run
/// Complements [`check_input_gradient`] by guarding the weight gradient paths (e.g. the
/// parallelized SeparableConv2D backward)
fn check_weight_gradient(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    let out = layer.forward(x).unwrap();
    let upstream = Tensor::ones(out.raw_dim());
    layer.backward(&upstream).unwrap();

    // Snapshot each parameter tensor's current values and analytic gradients
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
            // through it perturbs the actual parameter; the temporary Vec is only a handle
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
    let mut dense = Dense::new(3, 2, Linear::new(), None).unwrap();
    let x = Array::from_shape_vec((4, 3), (0..12).map(|v| 0.1 * v as f32 - 0.5).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut dense, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(
        2,
        2,
        vec![1, 1, 5],
        1,
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
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
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 4, 4),
        (0..16).map(|v| 0.1 * v as f32 - 0.7).collect(),
    )
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
        None,
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
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn separable_conv2d_same_padding_weight_gradient_matches_finite_difference() {
    // Companion to the `Same` input-gradient check: a 3x3 kernel with leading padding guards that
    // the depthwise weight gradient is accumulated against the padded input
    let mut conv = SeparableConv2D::new(
        2,
        (3, 3),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Same,
        1,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = DepthwiseConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

// Weighted-loss gradient checks: L = sum(output) is degenerate where the output-sum is constant in
// the input (softmax, zero-mean normalizers), so a weighted loss L = sum(W * output) is used instead

/// A fixed, non-uniform weight tensor shaped like `like`, with all entries in [0.7, 1.3]
fn loss_weights(like: &Tensor) -> Tensor {
    let n = like.len();
    let flat: Vec<f32> = (0..n).map(|k| 1.0 + 0.1 * ((k % 7) as f32 - 3.0)).collect();
    Tensor::from_shape_vec(like.raw_dim(), flat).unwrap()
}

/// A tensor of distinct, well-separated values (gap 0.5 >> eps), so max-pooling argmax positions are
/// unambiguous and never flip under a finite-difference perturbation
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|v| 0.5 * v as f32 - 0.25 * n as f32).collect();
    Array::from_shape_vec(shape.to_vec(), data).unwrap()
}

/// Like [`check_input_gradient`] but with a weighted loss L = sum(W * output), so it is
/// non-degenerate for softmax and zero-mean normalization layers
fn check_input_gradient_weighted(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    let out = layer.forward(x).unwrap();
    let w = loss_weights(&out);
    let analytic = layer.backward(&w).unwrap();
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
        let l_plus: f32 = (&layer.forward(&xp).unwrap() * &w).sum();

        x_flat[i] = orig - eps;
        let xm = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_minus: f32 = (&layer.forward(&xm).unwrap() * &w).sum();

        x_flat[i] = orig;

        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert_abs_diff_eq!(analytic_flat[i], numeric, epsilon = tol);
    }
}

// Softmax: the Jacobian-vector backward. L = sum(output) is degenerate (rows sum to 1), so this
// must use the weighted loss - it actually exercises a[i]*(g[i] - sum(a * g))
#[test]
fn softmax_input_gradient_matches_finite_difference() {
    let mut softmax = Softmax::new();
    let x = Array::from_shape_vec((2, 3), vec![0.2, -0.5, 1.0, 0.7, 0.1, -0.3])
        .unwrap()
        .into_dyn();
    check_input_gradient_weighted(&mut softmax, &x, 1e-3, 2e-2);
}

// Pooling layers (no trainable parameters -> input gradient only)

#[test]
fn max_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling1D::new(2, vec![1, 2, 6], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 2, 6]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn max_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling2D::new((2, 2), vec![1, 2, 4, 4], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 2, 4, 4]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn max_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 1, 4, 4, 4]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling1D::new(2, vec![1, 2, 6], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 2, 6]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling2D::new((2, 2), vec![1, 2, 4, 4], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 2, 4, 4]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling3D::new((2, 2, 2), vec![1, 1, 4, 4, 4], None, PaddingType::Valid).unwrap();
    let x = ramp(&[1, 1, 4, 4, 4]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_max_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalMaxPooling1D::new();
    let x = ramp(&[1, 2, 5]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_max_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalMaxPooling2D::new();
    let x = ramp(&[1, 2, 3, 3]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_max_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalMaxPooling3D::new();
    let x = ramp(&[1, 2, 2, 2, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_average_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalAveragePooling1D::new();
    let x = ramp(&[1, 2, 5]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_average_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalAveragePooling2D::new();
    let x = ramp(&[1, 2, 3, 3]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_average_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalAveragePooling3D::new();
    let x = ramp(&[1, 2, 2, 2, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

// Convolution with `Same` padding: a different backward code path than `Valid` (existing conv checks
// all use `Valid`). Ones-based helpers suffice (conv output sum is non-constant); odd kernels keep padding symmetric

#[test]
fn conv1d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(
        2,
        3,
        vec![1, 1, 6],
        1,
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec((1, 1, 6), (0..6).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(
        2,
        3,
        vec![1, 1, 6],
        1,
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec((1, 1, 6), (0..6).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(
        2,
        (3, 3),
        vec![1, 1, 5, 5],
        (1, 1),
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 5, 5),
        (0..25).map(|v| 0.05 * v as f32 - 0.6).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(
        2,
        (3, 3),
        vec![1, 1, 5, 5],
        (1, 1),
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 5, 5),
        (0..25).map(|v| 0.05 * v as f32 - 0.6).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(
        2,
        (3, 3, 3),
        vec![1, 1, 4, 4, 4],
        (1, 1, 1),
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 4, 4, 4),
        (0..64).map(|v| 0.03 * v as f32 - 0.9).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn conv3d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(
        2,
        (3, 3, 3),
        vec![1, 1, 4, 4, 4],
        (1, 1, 1),
        PaddingType::Same,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::from_shape_vec(
        (1, 1, 4, 4, 4),
        (0..64).map(|v| 0.03 * v as f32 - 0.9).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

// SeparableConv2D `Same` padding, depth_multiplier=2, 3x3 (symmetric) kernel. With the 2x2
// (asymmetric-padding) Same checks above, guards the depthwise zero-padding fix (the stage previously clamped at the border)
#[test]
fn separable_conv2d_same_padding_3x3_dm2_gradients_match_finite_difference() {
    // 3x3 (symmetric padding) + depth_multiplier=2: input and weight gradients together
    let make = || {
        SeparableConv2D::new(
            2,
            (3, 3),
            vec![1, 2, 5, 5],
            (1, 1),
            PaddingType::Same,
            2,
            Linear::new(),
            None,
        )
        .unwrap()
    };
    let x = Array::from_shape_vec(
        (1, 2, 5, 5),
        (0..50).map(|v| 0.04 * v as f32 - 1.0).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut make(), &x, 1e-3, 3e-2);
    check_weight_gradient(&mut make(), &x, 1e-3, 3e-2);
}

// Recurrent weight gradients (existing recurrent checks cover only the INPUT gradient). BPTT
// accumulates weight gradients across timesteps, so a per-timestep bug shows up here

#[test]
fn simple_rnn_weight_gradient_matches_finite_difference() {
    let mut rnn = SimpleRNN::new(2, 3, Tanh::new(), None).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut rnn, &x, 1e-3, 3e-2);
}

#[test]
fn lstm_weight_gradient_matches_finite_difference() {
    let mut lstm = LSTM::new(2, 3, Tanh::new(), None).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut lstm, &x, 1e-3, 3e-2);
}

#[test]
fn gru_weight_gradient_matches_finite_difference() {
    let mut gru = GRU::new(2, 3, Tanh::new(), None).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut gru, &x, 1e-3, 3e-2);
}

/// Like [`check_weight_gradient`] but with the weighted loss L = sum(W * output). Needed for
/// normalization layers: with gamma=1, beta=0 the all-ones upstream gives a ~0 gamma-gradient (sum
/// over the normalization group of the zero-mean normalized values), so the plain check is degenerate
fn check_weight_gradient_weighted(layer: &mut dyn Layer, x: &Tensor, eps: f32, tol: f32) {
    let out = layer.forward(x).unwrap();
    let w = loss_weights(&out);
    layer.backward(&w).unwrap();

    let params: Vec<(Vec<f32>, Vec<f32>)> = layer
        .parameters()
        .into_iter()
        .map(|pg| (pg.value.to_vec(), pg.grad.to_vec()))
        .collect();
    assert!(!params.is_empty(), "layer exposes no parameters to check");

    for (p_idx, (values, grads)) in params.iter().enumerate() {
        for i in 0..values.len() {
            let orig = values[i];

            layer.parameters()[p_idx].value[i] = orig + eps;
            let l_plus: f32 = (&layer.forward(x).unwrap() * &w).sum();

            layer.parameters()[p_idx].value[i] = orig - eps;
            let l_minus: f32 = (&layer.forward(x).unwrap() * &w).sum();

            layer.parameters()[p_idx].value[i] = orig;

            let numeric = (l_plus - l_minus) / (2.0 * eps);
            assert_abs_diff_eq!(grads[i], numeric, epsilon = tol);
        }
    }
}

// Normalization layers in TRAINING mode (the grad_mean / grad_var path), using the WEIGHTED loss
// since a zero-mean normalizer's output sums to ~0; checks both input and gamma/beta gradients

#[test]
fn layer_normalization_default_input_gradient_matches_finite_difference() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    ln.set_training_if_mode_dependent(true);
    let x = ramp(&[2, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_default_weight_gradient_matches_finite_difference() {
    let mut ln =
        LayerNormalization::new(vec![2, 4], LayerNormalizationAxis::Default, 1e-5).unwrap();
    ln.set_training_if_mode_dependent(true);
    let x = ramp(&[2, 4]);
    check_weight_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_custom_axis_input_gradient_matches_finite_difference() {
    let mut ln =
        LayerNormalization::new(vec![3, 4], LayerNormalizationAxis::Custom(0), 1e-5).unwrap();
    ln.set_training_if_mode_dependent(true);
    let x = ramp(&[3, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn group_normalization_input_gradient_matches_finite_difference() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    gn.set_training_if_mode_dependent(true);
    let x = ramp(&[1, 4, 4]);
    check_input_gradient_weighted(&mut gn, &x, 1e-3, 5e-2);
}

#[test]
fn group_normalization_weight_gradient_matches_finite_difference() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    gn.set_training_if_mode_dependent(true);
    let x = ramp(&[1, 4, 4]);
    check_weight_gradient_weighted(&mut gn, &x, 1e-3, 5e-2);
}

#[test]
fn instance_normalization_input_gradient_matches_finite_difference() {
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();
    inn.set_training_if_mode_dependent(true);
    let x = ramp(&[1, 3, 4]);
    check_input_gradient_weighted(&mut inn, &x, 1e-3, 5e-2);
}

#[test]
fn instance_normalization_weight_gradient_matches_finite_difference() {
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();
    inn.set_training_if_mode_dependent(true);
    let x = ramp(&[1, 3, 4]);
    check_weight_gradient_weighted(&mut inn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_input_gradient_weighted_matches_finite_difference() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    bn.set_training_if_mode_dependent(true);
    let x = ramp(&[4, 3]);
    check_input_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_weight_gradient_matches_finite_difference() {
    let mut bn = BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap();
    bn.set_training_if_mode_dependent(true);
    let x = ramp(&[4, 3]);
    check_weight_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

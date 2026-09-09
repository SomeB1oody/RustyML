//! Numerical (finite-difference) gradient checks for layer backward passes.
//!
//! For each layer, L = sum(output), so dL/dx = backward(ones). This compares the analytic input
//! gradient against a central finite-difference estimate of dL/dx. It catches gradient bugs that
//! a shape-only check or a "loss decreased" convergence test would miss.

use approx::assert_abs_diff_eq;
use ndarray::Array;
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::elu::ELU;
use rustyml::neural_network::layers::activation::exponential::Exponential;
use rustyml::neural_network::layers::activation::hard_sigmoid::HardSigmoid;
use rustyml::neural_network::layers::activation::leaky_relu::LeakyReLU;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::p_relu::PReLU;
use rustyml::neural_network::layers::activation::selu::SELU;
use rustyml::neural_network::layers::activation::softmax::Softmax;
use rustyml::neural_network::layers::activation::softplus::Softplus;
use rustyml::neural_network::layers::activation::softsign::Softsign;
use rustyml::neural_network::layers::activation::tanh::Tanh;
use rustyml::neural_network::layers::border::cropping_1d::Cropping1D;
use rustyml::neural_network::layers::border::cropping_2d::Cropping2D;
use rustyml::neural_network::layers::border::cropping_3d::Cropping3D;
use rustyml::neural_network::layers::border::zero_padding_1d::ZeroPadding1D;
use rustyml::neural_network::layers::border::zero_padding_2d::ZeroPadding2D;
use rustyml::neural_network::layers::border::zero_padding_3d::ZeroPadding3D;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::embedding::Embedding;
use rustyml::neural_network::layers::identity::Identity;
use rustyml::neural_network::layers::merge::{
    Add, Average, Concatenate, Maximum, Minimum, Multiply, Subtract,
};
use rustyml::neural_network::layers::permute::Permute;
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
use rustyml::neural_network::layers::regularization::normalization::unit_normalization::{
    UnitNormalization, UnitNormalizationAxis,
};
use rustyml::neural_network::layers::repeat_vector::RepeatVector;
use rustyml::neural_network::layers::reshape::Reshape;
use rustyml::neural_network::layers::upsampling::{
    Interpolation, UpSampling1D, UpSampling2D, UpSampling3D,
};
use rustyml::neural_network::traits::{Layer, ParamId, UnaryLayer};

/// Compares `layer.backward(ones)` against a central finite-difference estimate of
/// d sum(output)/dx.
///
/// Every pass runs in a training context. The backward pass needs the cache that only a
/// training forward pass writes, and a mode-dependent layer must take the same branch in the
/// analytic pass and in each finite-difference probe.
fn check_input_gradient(layer: &mut dyn UnaryLayer, x: &Tensor, eps: f32, tol: f32) {
    // With L = sum(output), the analytic input gradient is backward(ones)
    let mut ctx = Ctx::training();
    let out = layer.forward_mut(x, &mut ctx).unwrap();
    let upstream = Tensor::ones(out.raw_dim());
    let analytic = layer.backward(&upstream, &mut ctx).unwrap();
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
        let l_plus: f32 = layer.forward(&xp, &mut Ctx::training()).unwrap().sum();

        x_flat[i] = orig - eps;
        let xm = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_minus: f32 = layer.forward(&xm, &mut Ctx::training()).unwrap().sum();

        x_flat[i] = orig;

        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert_abs_diff_eq!(analytic_flat[i], numeric, epsilon = tol);
    }
}

#[test]
fn dense_input_gradient_matches_finite_difference() {
    // Linear activation keeps the layer smooth (no ReLU kink at 0), so finite differences
    // are exact.
    let mut dense = Dense::new(2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((4, 3), (0..12).map(|v| 0.1 * v as f32 - 0.5).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut dense, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_input_gradient_matches_finite_difference() {
    // Channels-last: [batch, height, width, channels]
    let mut conv = Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 1),
        (0..16).map(|v| 0.1 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_input_gradient_matches_finite_difference() {
    // Linear activation makes the convolution linear in its input, so finite differences match
    // tightly. Channels-last: [batch, length, channels]
    let mut conv = Conv1D::new(2, 2, 1, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 5, 1), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_input_gradient_matches_finite_difference() {
    // Channels-last: [batch, depth, height, width, channels]
    let mut conv = Conv3D::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 3, 3, 3, 1),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn separable_conv2d_input_gradient_matches_finite_difference() {
    let mut conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn separable_conv2d_same_padding_input_gradient_matches_finite_difference() {
    // A 3x3 kernel under `Same` needs padding at the border. This guards that the depthwise
    // stage skips each out-of-range tap instead of reading past the input.
    let mut conv = SeparableConv2D::new(2, (3, 3), (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_input_gradient_matches_finite_difference() {
    // `new` Xavier-initializes the weights, so the layer is a genuine (non-constant) map
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_same_padding_input_gradient_matches_finite_difference() {
    // A 3x3 `Same` kernel needs padding at the border. This guards that DepthwiseConv2D skips
    // each out-of-range tap instead of reading past the input.
    let mut conv = DepthwiseConv2D::new((3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = DepthwiseConv2D::new((3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

/// A `[1, length, channels]` ramp fixture for the 1D depthwise and separable checks
fn seq_1d(length: usize, channels: usize) -> Tensor {
    Array::from_shape_vec(
        (1, length, channels),
        (0..length * channels)
            .map(|v| 0.05 * v as f32 - 0.7)
            .collect(),
    )
    .unwrap()
    .into_dyn()
}

#[test]
fn depthwise_conv1d_input_gradient_matches_finite_difference() {
    // `new` Xavier-initializes the weights, so the layer is a genuine (non-constant) map
    let mut conv = DepthwiseConv1D::new(3, 1, Linear::new()).unwrap();
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn depthwise_conv1d_weight_gradient_matches_finite_difference() {
    let mut conv = DepthwiseConv1D::new(3, 1, Linear::new()).unwrap();
    check_weight_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn depthwise_conv1d_same_padding_input_gradient_matches_finite_difference() {
    // A width-3 `Same` kernel needs padding at both ends. This guards that the layer skips each
    // out-of-range tap instead of reading past the input.
    let mut conv = DepthwiseConv1D::new(3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn depthwise_conv1d_same_padding_weight_gradient_matches_finite_difference() {
    // Companion to the check above. The taps that lie in the padding must not enter the
    // weight-gradient sum.
    let mut conv = DepthwiseConv1D::new(3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    check_weight_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

/// At `depth_multiplier = 2` the shared kernel leaves its `dm == 1` fast path. Both gradients
/// must still route each multiplier back to its own input channel.
#[test]
fn depthwise_conv1d_depth_multiplier_2_gradients_match_finite_difference() {
    let mut conv = DepthwiseConv1D::new(3, 2, Linear::new())
        .unwrap()
        .with_depth_multiplier(2)
        .unwrap()
        .with_padding(PaddingType::Same);
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);

    let mut twin = DepthwiseConv1D::new(3, 2, Linear::new())
        .unwrap()
        .with_depth_multiplier(2)
        .unwrap()
        .with_padding(PaddingType::Same);
    check_weight_gradient(&mut twin, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn separable_conv1d_input_gradient_matches_finite_difference() {
    let mut conv = SeparableConv1D::new(3, 3, 1, 1, Linear::new()).unwrap();
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn separable_conv1d_weight_gradient_matches_finite_difference() {
    let mut conv = SeparableConv1D::new(3, 3, 1, 1, Linear::new()).unwrap();
    check_weight_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn separable_conv1d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = SeparableConv1D::new(3, 3, 1, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);
}

/// The `Same` and `depth_multiplier = 2` case at once. It covers all 3 parameter tensors while
/// the depthwise stage pads at both ends.
#[test]
fn separable_conv1d_same_padding_depth_multiplier_2_gradients_match_finite_difference() {
    let mut conv = SeparableConv1D::new(3, 3, 2, 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    check_input_gradient(&mut conv, &seq_1d(8, 2), 1e-3, 2e-2);

    let mut twin = SeparableConv1D::new(3, 3, 2, 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    check_weight_gradient(&mut twin, &seq_1d(8, 2), 1e-3, 2e-2);
}

#[test]
fn simple_rnn_input_gradient_matches_finite_difference() {
    // Multi-timestep with Tanh. This fails if BPTT reuses a single stale activation derivative
    // across timesteps instead of 1 per timestep.
    let mut rnn = SimpleRNN::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut rnn, &x, 1e-3, 2e-2);
}

#[test]
fn lstm_input_gradient_matches_finite_difference() {
    // The configurable activation is the per-timestep candidate and cell-state nonlinearity,
    // so this guards BPTT through both activation derivatives.
    let mut lstm = LSTM::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut lstm, &x, 1e-3, 3e-2);
}

#[test]
fn gru_input_gradient_matches_finite_difference() {
    // The configurable activation is the per-timestep candidate nonlinearity.
    let mut gru = GRU::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut gru, &x, 1e-3, 3e-2);
}

#[test]
fn batch_normalization_input_gradient_matches_finite_difference() {
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
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
    // `Same` padding output length must be ceil(input_len / stride), computed from the original,
    // unpadded length, not the padded buffer.
    let cases = [
        // (input_len, kernel, stride, expected_out_len)
        (10usize, 3usize, 1usize, 10usize),
        (10, 3, 2, 5),
        (8, 5, 1, 8),
        (7, 3, 2, 4),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut conv = Conv1D::new(2, kernel, stride, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same);
        let x = Array::ones((1, len, 1)).into_dyn();
        let out = conv.forward_mut(&x, &mut Ctx::inference()).unwrap();
        // Channels-last output: [batch, out_len, filters]
        assert_eq!(
            out.shape(),
            &[1, expected, 2],
            "Conv1D Same: input_len={}, kernel={}, stride={}",
            len,
            kernel,
            stride
        );
    }
}

/// Snapshots the current values of every parameter of the layer, next to the analytic gradient
/// that the backward pass put in the store.
///
/// The layer drives no model, so every gradient sits at layer position 0.
fn param_snapshots(layer: &mut dyn UnaryLayer, ctx: &Ctx) -> Vec<(Vec<f32>, Vec<f32>)> {
    let names: Vec<&'static str> = layer
        .parameters_mut()
        .iter()
        .map(|param| param.name)
        .collect();
    let params: Vec<(Vec<f32>, Vec<f32>)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let value = layer.parameters_mut()[index].value.to_vec();
            let grad = ctx
                .grads()
                .get(ParamId::new(0, name))
                .unwrap_or_else(|| panic!("the backward pass gave parameter `{name}` no gradient"))
                .iter()
                .cloned()
                .collect();
            (value, grad)
        })
        .collect();
    assert!(!params.is_empty(), "layer exposes no parameters to check");
    params
}

/// Compares the analytic gradients in `ctx.grads()` against a central finite-difference estimate
/// of d sum(output)/d param, perturbing each value in place.
fn check_weight_gradient(layer: &mut dyn UnaryLayer, x: &Tensor, eps: f32, tol: f32) {
    let mut ctx = Ctx::training();
    let out = layer.forward_mut(x, &mut ctx).unwrap();
    let upstream = Tensor::ones(out.raw_dim());
    layer.backward(&upstream, &mut ctx).unwrap();

    let params = param_snapshots(layer, &ctx);

    for (p_idx, (values, grads)) in params.iter().enumerate() {
        for i in 0..values.len() {
            let orig = values[i];

            // `parameters_mut()[p_idx].value` is a mutable view into the live weight array, so
            // writing through it perturbs the actual parameter
            layer.parameters_mut()[p_idx].value[i] = orig + eps;
            let l_plus: f32 = layer.forward(x, &mut Ctx::training()).unwrap().sum();

            layer.parameters_mut()[p_idx].value[i] = orig - eps;
            let l_minus: f32 = layer.forward(x, &mut Ctx::training()).unwrap().sum();

            layer.parameters_mut()[p_idx].value[i] = orig;

            let numeric = (l_plus - l_minus) / (2.0 * eps);
            assert_abs_diff_eq!(grads[i], numeric, epsilon = tol);
        }
    }
}

#[test]
fn dense_weight_gradient_matches_finite_difference() {
    let mut dense = Dense::new(2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((4, 3), (0..12).map(|v| 0.1 * v as f32 - 0.5).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut dense, &x, 1e-3, 1e-2);
}

/// The slope gradient must sum over the batch axis only, and keep 1 slope per feature
#[test]
fn p_relu_weight_gradient_matches_finite_difference() {
    let mut p_relu = PReLU::new(0.0).unwrap();
    p_relu.build(&Shape::known(&[3, 4])).unwrap();
    p_relu
        .set_weights(Array::from_vec(vec![0.2f32, -0.35, 0.6, 0.45]).into_dyn())
        .unwrap();
    // Every probe stays clear of the kink at x = 0, and each column mixes both branches
    let x = Array::from_shape_vec(
        (3, 4),
        vec![
            0.8, -0.5, 1.2, -1.4, -0.3, 0.9, -1.6, 0.2, 1.1, -0.7, 0.4, -0.6,
        ],
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut p_relu, &x, 1e-3, 1e-2);
}

/// The same check where each slope also collects the 2 shared spatial axes
#[test]
fn p_relu_shared_axes_weight_gradient_matches_finite_difference() {
    let mut p_relu = PReLU::new(0.0)
        .unwrap()
        .with_shared_axes(vec![1, 2])
        .unwrap();
    p_relu.build(&Shape::known(&[2, 3, 3, 2])).unwrap();
    p_relu
        .set_weights(
            Array::from_shape_vec((1, 1, 2), vec![0.3f32, -0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
    let x = Array::from_shape_vec(
        (2, 3, 3, 2),
        (0..36).map(|v| 0.1 * v as f32 - 1.85).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut p_relu, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(2, 2, 1, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 5, 1), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 1),
        (0..16).map(|v| 0.1 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 3, 3, 3, 1),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn separable_conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn separable_conv2d_same_padding_weight_gradient_matches_finite_difference() {
    // Companion to the `Same` input-gradient check. The out-of-range taps that the depthwise
    // stage skips must not enter its weight-gradient sum.
    let mut conv = SeparableConv2D::new(2, (3, 3), (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn depthwise_conv2d_weight_gradient_matches_finite_difference() {
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

// Weighted-loss checks: L = sum(output) is degenerate for a layer whose output sum stays
// constant in the input (softmax, zero-mean normalizers). These use L = sum(W * output) instead.

/// A fixed, non-uniform weight tensor shaped like `like`, with all entries in [0.7, 1.3]
fn loss_weights(like: &Tensor) -> Tensor {
    let n = like.len();
    let flat: Vec<f32> = (0..n).map(|k| 1.0 + 0.1 * ((k % 7) as f32 - 3.0)).collect();
    Tensor::from_shape_vec(like.raw_dim(), flat).unwrap()
}

/// A tensor of distinct, well-separated values (gap 0.5 >> eps), so max-pooling argmax
/// positions stay unambiguous and never flip under a finite-difference perturbation.
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|v| 0.5 * v as f32 - 0.25 * n as f32).collect();
    Array::from_shape_vec(shape.to_vec(), data).unwrap()
}

/// Like [`check_input_gradient`] but with a weighted loss L = sum(W * output), so it is
/// non-degenerate for softmax and zero-mean normalization layers
fn check_input_gradient_weighted(layer: &mut dyn UnaryLayer, x: &Tensor, eps: f32, tol: f32) {
    let mut ctx = Ctx::training();
    let out = layer.forward_mut(x, &mut ctx).unwrap();
    let w = loss_weights(&out);
    let analytic = layer.backward(&w, &mut ctx).unwrap();
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
        let l_plus: f32 = (&layer.forward(&xp, &mut Ctx::training()).unwrap() * &w).sum();

        x_flat[i] = orig - eps;
        let xm = Tensor::from_shape_vec(x.raw_dim(), x_flat.clone()).unwrap();
        let l_minus: f32 = (&layer.forward(&xm, &mut Ctx::training()).unwrap() * &w).sum();

        x_flat[i] = orig;

        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert_abs_diff_eq!(analytic_flat[i], numeric, epsilon = tol);
    }
}

// Softmax: the Jacobian-vector backward. L = sum(output) is degenerate, since rows sum to 1,
// so this uses the weighted loss instead. It exercises a[i] * (g[i] - sum(a * g)).
#[test]
fn softmax_input_gradient_matches_finite_difference() {
    let mut softmax = Softmax::new();
    let x = Array::from_shape_vec((2, 3), vec![0.2, -0.5, 1.0, 0.7, 0.1, -0.3])
        .unwrap()
        .into_dyn();
    check_input_gradient_weighted(&mut softmax, &x, 1e-3, 2e-2);
}

// The remaining activation layers map elementwise, so the ones-based helper stays non-degenerate.
// A central difference needs differentiability, so every probe stays clear of a kink by more
// than `eps`.

/// LeakyReLU with a non-default slope, so a backward pass that assumes the 0.3 default fails
#[test]
fn leaky_relu_input_gradient_matches_finite_difference() {
    // The kink sits at x = 0, so every probe stays at least 0.2 away from it
    let mut leaky_relu = LeakyReLU::new(0.2).unwrap();
    let x = Array::from_shape_vec((2, 4), vec![0.8, -0.5, 1.2, -1.4, 0.3, -0.9, 1.6, -0.2])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut leaky_relu, &x, 1e-3, 1e-2);
}

/// PReLU with a slope per feature, so a backward pass that reads the wrong slope fails
#[test]
fn p_relu_input_gradient_matches_finite_difference() {
    // The kink sits at x = 0, so every probe stays at least 0.2 away from it
    let mut p_relu = PReLU::new(0.0).unwrap();
    p_relu.build(&Shape::known(&[2, 4])).unwrap();
    p_relu
        .set_weights(Array::from_vec(vec![0.2f32, -0.35, 0.6, 0.45]).into_dyn())
        .unwrap();
    let x = Array::from_shape_vec((2, 4), vec![0.8, -0.5, 1.2, -1.4, 0.3, -0.9, 1.6, -0.2])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut p_relu, &x, 1e-3, 1e-2);
}

/// PReLU with the 2 spatial axes shared, where 1 slope covers a whole channel plane
#[test]
fn p_relu_shared_axes_input_gradient_matches_finite_difference() {
    let mut p_relu = PReLU::new(0.0)
        .unwrap()
        .with_shared_axes(vec![1, 2])
        .unwrap();
    p_relu.build(&Shape::known(&[2, 3, 3, 2])).unwrap();
    p_relu
        .set_weights(
            Array::from_shape_vec((1, 1, 2), vec![0.3f32, -0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
    // The kink sits at x = 0, so the ramp skips it
    let x = Array::from_shape_vec(
        (2, 3, 3, 2),
        (0..36).map(|v| 0.1 * v as f32 - 1.85).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut p_relu, &x, 1e-3, 1e-2);
}

/// ELU with a non-default alpha, which scales both the negative branch and its derivative
#[test]
fn elu_input_gradient_matches_finite_difference() {
    // The kink sits at x = 0, so every probe stays at least 0.25 away from it
    let mut elu = ELU::new(1.5).unwrap();
    let x = Array::from_shape_vec((2, 4), vec![0.7, -0.4, 1.1, -1.3, 0.3, -0.8, 1.6, -0.25])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut elu, &x, 1e-3, 1e-2);
}

/// SELU, whose 2 fixed constants must appear on both branches of the derivative
#[test]
fn selu_input_gradient_matches_finite_difference() {
    // The kink sits at x = 0, so every probe stays at least 0.3 away from it
    let mut selu = SELU::new();
    let x = Array::from_shape_vec((2, 4), vec![0.6, -0.3, 1.0, -1.2, 0.4, -0.7, 1.8, -0.9])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut selu, &x, 1e-3, 1e-2);
}

/// Softplus, which is smooth everywhere, so any moderate probe is valid
#[test]
fn softplus_input_gradient_matches_finite_difference() {
    let mut softplus = Softplus::new();
    let x = Array::from_shape_vec((2, 4), vec![0.6, -0.9, 1.4, -1.8, 0.2, -0.4, 2.2, -1.2])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut softplus, &x, 1e-3, 1e-2);
}

/// Softsign, which is smooth everywhere, with probes far enough out to reach the flat tail
#[test]
fn softsign_input_gradient_matches_finite_difference() {
    let mut softsign = Softsign::new();
    let x = Array::from_shape_vec((2, 4), vec![0.5, -0.7, 1.5, -2.0, 0.9, -0.3, 2.5, -1.1])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut softsign, &x, 1e-3, 1e-2);
}

/// HardSigmoid on the linear segment only, where the derivative is the constant 1/6
#[test]
fn hard_sigmoid_input_gradient_matches_finite_difference() {
    // Kinks sit at x = -3 and x = 3. Every probe here stays at least 0.5 away from both, since
    // a finite difference cannot cross a breakpoint.
    let mut hard_sigmoid = HardSigmoid::new();
    let x = Array::from_shape_vec((2, 4), vec![0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.9, -2.0])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut hard_sigmoid, &x, 1e-3, 1e-2);
}

/// Exponential, whose derivative is its own output
#[test]
fn exponential_input_gradient_matches_finite_difference() {
    // Probes stay inside [-1.0, 1.5], because e^x and its curvature both grow fast. The
    // truncation error of a central difference is proportional to the third derivative
    let mut exponential = Exponential::new();
    let x = Array::from_shape_vec((2, 4), vec![0.4, -0.6, 1.2, -1.0, 0.8, -0.2, 1.5, 0.1])
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut exponential, &x, 1e-3, 1e-2);
}

// Pooling layers (no trainable parameters -> input gradient only)

#[test]
fn max_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling1D::new(2);
    let x = ramp(&[1, 6, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn max_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling2D::new((2, 2));
    let x = ramp(&[1, 4, 4, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn max_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = MaxPooling3D::new((2, 2, 2));
    let x = ramp(&[1, 4, 4, 4, 1]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling1D::new(2);
    let x = ramp(&[1, 6, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling2D::new((2, 2));
    let x = ramp(&[1, 4, 4, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn average_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = AveragePooling3D::new((2, 2, 2));
    let x = ramp(&[1, 4, 4, 4, 1]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_max_pooling_1d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalMaxPooling1D::new();
    let x = ramp(&[1, 5, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_max_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalMaxPooling2D::new();
    let x = ramp(&[1, 3, 3, 2]);
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
    let x = ramp(&[1, 5, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_average_pooling_2d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalAveragePooling2D::new();
    let x = ramp(&[1, 3, 3, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

#[test]
fn global_average_pooling_3d_input_gradient_matches_finite_difference() {
    let mut pool = GlobalAveragePooling3D::new();
    let x = ramp(&[1, 2, 2, 2, 2]);
    check_input_gradient_weighted(&mut pool, &x, 1e-3, 1e-2);
}

// Reshape (no trainable parameters, input gradient only). A reshape is a pure index remap, so
// each input gradient equals exactly 1, and a shape-only check would miss a wrong target or a
// permuted element.

#[test]
fn reshape_split_input_gradient_matches_finite_difference() {
    // Rank 2 -> rank 3: the -1 axis resolves to 3, so [2, 6] becomes [2, 3, 2]
    let mut reshape = Reshape::new(vec![-1, 2]).unwrap();
    let x = Array::from_shape_vec((2, 6), (0..12).map(|v| 0.1 * v as f32 - 0.6).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut reshape, &x, 1e-3, 1e-2);
}

#[test]
fn reshape_merge_input_gradient_matches_finite_difference() {
    // Rank 4 -> rank 2: the -1 axis merges the 3 trailing axes, so [2, 2, 3, 2] becomes [2, 12]
    let mut reshape = Reshape::new(vec![-1]).unwrap();
    let x = Array::from_shape_vec(
        (2, 2, 3, 2),
        (0..24).map(|v| 0.05 * v as f32 - 0.6).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut reshape, &x, 1e-3, 1e-2);
}

// Border layers (no trainable parameters, input gradient only). Zero-padding drops the gradient
// of every padded position, and cropping drops it for every removed position, leaving the rest
// unchanged. The weighted loss catches a border that lands 1 position off, which a symmetric
// border would hide. All 6 layers are exactly linear, so the tolerance stays tight.

#[test]
fn zero_padding_1d_input_gradient_matches_finite_difference() {
    let mut pad = ZeroPadding1D::new((2, 1));
    let x = ramp(&[2, 4, 3]);
    check_input_gradient_weighted(&mut pad, &x, 1e-3, 1e-2);
}

#[test]
fn zero_padding_2d_input_gradient_matches_finite_difference() {
    let mut pad = ZeroPadding2D::new(((1, 2), (0, 1)));
    let x = ramp(&[2, 3, 4, 2]);
    check_input_gradient_weighted(&mut pad, &x, 1e-3, 1e-2);
}

#[test]
fn zero_padding_3d_input_gradient_matches_finite_difference() {
    let mut pad = ZeroPadding3D::new(((1, 0), (0, 2), (1, 1)));
    let x = ramp(&[1, 2, 3, 2, 2]);
    check_input_gradient_weighted(&mut pad, &x, 1e-3, 1e-2);
}

#[test]
fn cropping_1d_input_gradient_matches_finite_difference() {
    let mut crop = Cropping1D::new((1, 2));
    let x = ramp(&[2, 6, 3]);
    check_input_gradient_weighted(&mut crop, &x, 1e-3, 1e-2);
}

#[test]
fn cropping_2d_input_gradient_matches_finite_difference() {
    let mut crop = Cropping2D::new(((2, 1), (0, 1)));
    let x = ramp(&[2, 5, 4, 2]);
    check_input_gradient_weighted(&mut crop, &x, 1e-3, 1e-2);
}

#[test]
fn cropping_3d_input_gradient_matches_finite_difference() {
    let mut crop = Cropping3D::new(((1, 0), (0, 1), (1, 1)));
    let x = ramp(&[1, 3, 3, 4, 2]);
    check_input_gradient_weighted(&mut crop, &x, 1e-3, 1e-2);
}

// Permute and RepeatVector (no trainable parameters, input gradient only). A permute routes each
// output gradient to exactly 1 input, so the weighted loss catches a wrong inverse order. A
// repeat sums its n step weights into each input gradient, which the ones-based helper would
// also hide.

#[test]
fn permute_swap_input_gradient_matches_finite_difference() {
    let mut permute = Permute::new(vec![2, 1]).unwrap();
    let x = ramp(&[2, 3, 4]);
    check_input_gradient_weighted(&mut permute, &x, 1e-3, 1e-2);
}

#[test]
fn permute_rotate_input_gradient_matches_finite_difference() {
    // A rotation moves every axis, so a backward pass that inverts only 1 pair fails here
    let mut permute = Permute::new(vec![3, 1, 2]).unwrap();
    let x = ramp(&[2, 2, 3, 2]);
    check_input_gradient_weighted(&mut permute, &x, 1e-3, 1e-2);
}

#[test]
fn repeat_vector_input_gradient_matches_finite_difference() {
    let mut repeat = RepeatVector::new(3).unwrap();
    let x = ramp(&[3, 4]);
    check_input_gradient_weighted(&mut repeat, &x, 1e-3, 1e-2);
}

// Upsampling layers (no trainable parameters, input gradient only). Each output position is a
// weighted sum of input positions, so the weighted loss exposes a misplaced tap that the
// ones-based helper would hide, since an output position's weights always sum to 1. Every
// factor below is uneven per axis, since an equal factor would hide the same bug.
//
// The step here is 1e-1 rather than the usual 1e-3: these layers are exactly linear, so a
// central difference has no truncation error, leaving only float32 rounding of the 2 loss
// values. A wider step divides that rounding by more.

#[test]
fn up_sampling_1d_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling1D::new(3).unwrap();
    let x = ramp(&[2, 4, 3]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_2d_nearest_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling2D::new((2, 3), Interpolation::Nearest).unwrap();
    let x = ramp(&[2, 3, 4, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_2d_bilinear_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling2D::new((3, 2), Interpolation::Bilinear).unwrap();
    let x = ramp(&[2, 3, 4, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_2d_bicubic_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling2D::new((2, 3), Interpolation::Bicubic).unwrap();
    let x = ramp(&[1, 4, 5, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_2d_lanczos3_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling2D::new((1, 2), Interpolation::Lanczos3).unwrap();
    let x = ramp(&[1, 4, 6, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_2d_lanczos5_input_gradient_matches_finite_difference() {
    // 12 positions on the wide axis, so the kernel of 11 taps is not clipped everywhere
    let mut layer = UpSampling2D::new((2, 1), Interpolation::Lanczos5).unwrap();
    let x = ramp(&[1, 12, 3, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

#[test]
fn up_sampling_3d_input_gradient_matches_finite_difference() {
    let mut layer = UpSampling3D::new((2, 1, 3)).unwrap();
    let x = ramp(&[1, 2, 3, 2, 2]);
    check_input_gradient_weighted(&mut layer, &x, 1e-1, 1e-2);
}

// An identity layer routes every output gradient back to the same input, so each entry must be
// exactly the upstream value. The ones-based helper is enough.

#[test]
fn identity_input_gradient_matches_finite_difference() {
    let mut identity = Identity::new();
    let x = ramp(&[3, 4]);
    check_input_gradient(&mut identity, &x, 1e-3, 1e-2);
}

// UnitNormalization discards the length of each group, so sum(output) barely moves with the
// input and the ones-based helper is degenerate. The weighted loss is not. The 3 checks below
// cover the trailing-axis row path, a middle axis, and 2 axes that are not next to each other.

#[test]
fn unit_normalization_input_gradient_weighted_matches_finite_difference() {
    let mut unit = UnitNormalization::new(UnitNormalizationAxis::Default).unwrap();
    let x = ramp(&[4, 5]);
    check_input_gradient_weighted(&mut unit, &x, 1e-3, 5e-2);
}

#[test]
fn unit_normalization_middle_axis_input_gradient_matches_finite_difference() {
    let mut unit = UnitNormalization::new(UnitNormalizationAxis::Custom(1)).unwrap();
    let x = ramp(&[2, 4, 3]);
    check_input_gradient_weighted(&mut unit, &x, 1e-3, 5e-2);
}

#[test]
fn unit_normalization_separated_axes_input_gradient_matches_finite_difference() {
    let mut unit = UnitNormalization::new(UnitNormalizationAxis::Multiple(vec![0, 2])).unwrap();
    let x = ramp(&[2, 4, 3]);
    check_input_gradient_weighted(&mut unit, &x, 1e-3, 5e-2);
}

// `Same` padding takes a different backward path than `Valid`, which every earlier conv check
// here uses. The ones-based helper still works, since a convolution's output sum is not
// constant in the input. Odd kernels below keep the padding symmetric.

#[test]
fn conv1d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 6, 1), (0..6).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1D::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 6, 1), (0..6).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(2, (3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 5, 5, 1),
        (0..25).map(|v| 0.05 * v as f32 - 0.6).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2D::new(2, (3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 5, 5, 1),
        (0..25).map(|v| 0.05 * v as f32 - 0.6).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(2, (3, 3, 3), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 4, 1),
        (0..64).map(|v| 0.03 * v as f32 - 0.9).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn conv3d_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3D::new(2, (3, 3, 3), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 4, 4, 4, 1),
        (0..64).map(|v| 0.03 * v as f32 - 0.9).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

// SeparableConv2D `Same` padding, depth_multiplier=2, 3x3 (symmetric) kernel. Every other `Same`
// check in this file uses depth_multiplier=1. Only this one guards the depthwise stage's
// zero-padding at the border while each input channel fans out to several output channels.
#[test]
fn separable_conv2d_same_padding_3x3_dm2_gradients_match_finite_difference() {
    // 3x3 (symmetric padding) + depth_multiplier=2: input and weight gradients together
    let make = || {
        SeparableConv2D::new(2, (3, 3), (1, 1), 2, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same)
    };
    let x = Array::from_shape_vec(
        (1, 5, 5, 2),
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
    let mut rnn = SimpleRNN::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut rnn, &x, 1e-3, 3e-2);
}

#[test]
fn lstm_weight_gradient_matches_finite_difference() {
    let mut lstm = LSTM::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut lstm, &x, 1e-3, 3e-2);
}

#[test]
fn gru_weight_gradient_matches_finite_difference() {
    let mut gru = GRU::new(3, Tanh::new()).unwrap();
    let x = Array::from_shape_vec((1, 3, 2), vec![0.3, -0.6, 0.9, -0.2, 0.5, -0.8])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut gru, &x, 1e-3, 3e-2);
}

// Embedding gradients test only the weight direction. The input holds indices, so
// perturbing it selects a different row instead of moving along a derivative

#[test]
fn embedding_weight_gradient_matches_finite_difference() {
    let mut embedding = Embedding::new(5, 3).unwrap().with_random_state(4);
    let x = Array::from_shape_vec((2, 3), vec![0.0f32, 4.0, 2.0, 1.0, 3.0, 0.0])
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut embedding, &x, 1e-3, 1e-2);
}

/// An index that appears several times accumulates, so its row gets the sum of the positions
#[test]
fn embedding_repeated_index_weight_gradient_matches_finite_difference() {
    let mut embedding = Embedding::new(4, 3).unwrap().with_random_state(5);
    // Row 1 appears 4 times, row 3 twice, and rows 0 and 2 once each
    let x = Array::from_shape_vec((2, 4), vec![1.0f32, 1.0, 0.0, 3.0, 3.0, 1.0, 2.0, 1.0])
        .unwrap()
        .into_dyn();
    check_weight_gradient_weighted(&mut embedding, &x, 1e-3, 1e-2);
}

/// Like [`check_weight_gradient`] but with the weighted loss L = sum(W * output). This avoids
/// the near-zero gamma gradient that an all-ones upstream gives normalization layers.
fn check_weight_gradient_weighted(layer: &mut dyn UnaryLayer, x: &Tensor, eps: f32, tol: f32) {
    let mut ctx = Ctx::training();
    let out = layer.forward_mut(x, &mut ctx).unwrap();
    let w = loss_weights(&out);
    layer.backward(&w, &mut ctx).unwrap();

    let params = param_snapshots(layer, &ctx);

    for (p_idx, (values, grads)) in params.iter().enumerate() {
        for i in 0..values.len() {
            let orig = values[i];

            layer.parameters_mut()[p_idx].value[i] = orig + eps;
            let l_plus: f32 = (&layer.forward(x, &mut Ctx::training()).unwrap() * &w).sum();

            layer.parameters_mut()[p_idx].value[i] = orig - eps;
            let l_minus: f32 = (&layer.forward(x, &mut Ctx::training()).unwrap() * &w).sum();

            layer.parameters_mut()[p_idx].value[i] = orig;

            let numeric = (l_plus - l_minus) / (2.0 * eps);
            assert_abs_diff_eq!(grads[i], numeric, epsilon = tol);
        }
    }
}

// Normalization layers in training mode (the grad_mean / grad_var path), using the weighted
// loss, since a zero-mean normalizer's output sums to ~0. Checks both the input and the
// gamma/beta gradients.

#[test]
fn layer_normalization_default_input_gradient_matches_finite_difference() {
    let mut ln = LayerNormalization::new(1e-5).unwrap();
    let x = ramp(&[2, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_default_weight_gradient_matches_finite_difference() {
    let mut ln = LayerNormalization::new(1e-5).unwrap();
    let x = ramp(&[2, 4]);
    check_weight_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_custom_axis_input_gradient_matches_finite_difference() {
    let mut ln = LayerNormalization::new(1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(0))
        .unwrap();
    let x = ramp(&[3, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_rank3_default_input_gradient_matches_finite_difference() {
    // Rank-3 Default exercises the fused row path with several rows per leading index
    let mut ln = LayerNormalization::new(1e-5).unwrap();
    let x = ramp(&[2, 3, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_trailing_custom_weight_gradient_matches_finite_difference() {
    // Custom on the trailing axis routes to the row path (unlike Custom(0) above)
    let mut ln = LayerNormalization::new(1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Custom(1))
        .unwrap();
    let x = ramp(&[3, 4]);
    check_weight_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_multiple_trailing_input_gradient_matches_finite_difference() {
    // Trailing in-order Multiple axes resolve to the zero-copy row path
    let mut ln = LayerNormalization::new(1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![1, 2]))
        .unwrap();
    let x = ramp(&[2, 3, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn layer_normalization_multiple_permuted_input_gradient_matches_finite_difference() {
    // Axes [0, 2] need a genuine merge permutation: the transpose-in / transpose-out bracket
    // around the row path
    let mut ln = LayerNormalization::new(1e-5)
        .unwrap()
        .with_normalized_axis(LayerNormalizationAxis::Multiple(vec![0, 2]))
        .unwrap();
    let x = ramp(&[2, 3, 4]);
    check_input_gradient_weighted(&mut ln, &x, 1e-3, 5e-2);
}

#[test]
fn group_normalization_input_gradient_matches_finite_difference() {
    let mut gn = GroupNormalization::new(2, 1e-5).unwrap();
    let x = ramp(&[1, 4, 4]);
    check_input_gradient_weighted(&mut gn, &x, 1e-3, 5e-2);
}

#[test]
fn group_normalization_weight_gradient_matches_finite_difference() {
    let mut gn = GroupNormalization::new(2, 1e-5).unwrap();
    let x = ramp(&[1, 4, 4]);
    check_weight_gradient_weighted(&mut gn, &x, 1e-3, 5e-2);
}

#[test]
fn group_normalization_batched_input_gradient_matches_finite_difference() {
    // batch > 1: every sample gets its own per-group statistics, so a fold that leaked across
    // the batch axis shows up here
    let mut gn = GroupNormalization::new(2, 1e-5).unwrap();
    let x = ramp(&[2, 4, 4]);
    check_input_gradient_weighted(&mut gn, &x, 1e-3, 5e-2);
}

#[test]
fn instance_normalization_input_gradient_matches_finite_difference() {
    let mut inn = InstanceNormalization::new(1e-5).unwrap();
    let x = ramp(&[1, 3, 4]);
    check_input_gradient_weighted(&mut inn, &x, 1e-3, 5e-2);
}

#[test]
fn instance_normalization_weight_gradient_matches_finite_difference() {
    let mut inn = InstanceNormalization::new(1e-5).unwrap();
    let x = ramp(&[1, 3, 4]);
    check_weight_gradient_weighted(&mut inn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_input_gradient_weighted_matches_finite_difference() {
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
    let x = ramp(&[4, 3]);
    check_input_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_weight_gradient_matches_finite_difference() {
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
    let x = ramp(&[4, 3]);
    check_weight_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_spatial_input_gradient_matches_finite_difference() {
    // Rank-4 [batch, height, width, channels]: BatchNormalization reduces the statistics over
    // the batch and both spatial axes. It normalizes each of the 3 channels over
    // 2*2*2 = 8 elements rather than 2.
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
    let x = ramp(&[2, 2, 2, 3]);
    check_input_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

#[test]
fn batch_normalization_spatial_weight_gradient_matches_finite_difference() {
    let mut bn = BatchNormalization::new(0.9, 1e-5).unwrap();
    let x = ramp(&[2, 2, 2, 3]);
    check_weight_gradient_weighted(&mut bn, &x, 1e-3, 5e-2);
}

// A transposed convolution runs a plain one's 2 halves in the other order: its forward pass is
// the plain backward's col2im scatter, and its backward pass is the plain forward's im2col
// gather. This exercises code no plain check reaches: the scatter's crop and the gather's pad.
// Both need their own case with `Same` padding and a stride above 1, since the crop and the pad
// are 0 under `Valid` at stride 1.

#[test]
fn conv1d_transpose_input_gradient_matches_finite_difference() {
    // Channels-last: [batch, length, channels]
    let mut conv = Conv1DTranspose::new(2, 3, 2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 4, 1), (0..4).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_transpose_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1DTranspose::new(2, 3, 2, Linear::new()).unwrap();
    let x = Array::from_shape_vec((1, 4, 1), (0..4).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_transpose_same_padding_input_gradient_matches_finite_difference() {
    // `Same` is the mode that crops the scatter buffer on the forward pass and pads the output
    // gradient back up on the backward pass
    let mut conv = Conv1DTranspose::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 5, 1), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv1d_transpose_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv1DTranspose::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec((1, 5, 1), (0..5).map(|v| 0.1 * v as f32 - 0.3).collect())
        .unwrap()
        .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_transpose_input_gradient_matches_finite_difference() {
    // Channels-last: [batch, height, width, channels]
    let mut conv = Conv2DTranspose::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 3, 3, 2),
        (0..18).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_transpose_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2DTranspose::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 3, 3, 2),
        (0..18).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_transpose_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv2DTranspose::new(2, (3, 3), (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 3, 3, 1),
        (0..9).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv2d_transpose_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv2DTranspose::new(2, (3, 3), (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 3, 3, 1),
        (0..9).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_transpose_input_gradient_matches_finite_difference() {
    // Channels-last: [batch, depth, height, width, channels]
    let mut conv = Conv3DTranspose::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        (0..8).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_transpose_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3DTranspose::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        (0..8).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 1e-2);
}

#[test]
fn conv3d_transpose_same_padding_input_gradient_matches_finite_difference() {
    let mut conv = Conv3DTranspose::new(2, (3, 3, 3), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        (0..8).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_input_gradient(&mut conv, &x, 1e-3, 2e-2);
}

#[test]
fn conv3d_transpose_same_padding_weight_gradient_matches_finite_difference() {
    let mut conv = Conv3DTranspose::new(2, (3, 3, 3), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        (0..8).map(|v| 0.05 * v as f32 - 0.2).collect(),
    )
    .unwrap()
    .into_dyn();
    check_weight_gradient(&mut conv, &x, 1e-3, 2e-2);
}

// Merge layers: the same finite-difference rule, over a layer that takes several inputs

/// Compares the analytic gradient of every input of a merge layer against a central difference
///
/// The rule is the one every check above uses: `L = sum(output)`, so `dL/dx` is what the
/// backward pass gives for an upstream of ones. A merge layer gives 1 gradient per input, and
/// each is checked against its own input
///
/// # Parameters
///
/// - `layer` - The merge layer under test, already built or built by this call
/// - `inputs` - 1 tensor per input of the layer
/// - `h` - Step of the central difference
/// - `tolerance` - How far the 2 gradients may sit apart
fn check_merge_input_gradients<L: Layer>(layer: &mut L, inputs: &[Tensor], h: f32, tolerance: f32) {
    let shapes: Vec<Shape> = inputs
        .iter()
        .map(|tensor| Shape::known(tensor.shape()))
        .collect();
    layer.build_many(&shapes).unwrap();

    let refs: Vec<&Tensor> = inputs.iter().collect();
    let mut ctx = Ctx::training();
    let output = layer.forward_many(&refs, &mut ctx).unwrap();
    let analytic = layer
        .backward_many(&Tensor::ones(output.raw_dim()), &mut ctx)
        .unwrap();
    assert_eq!(
        analytic.len(),
        inputs.len(),
        "a merge layer gives 1 gradient per input"
    );

    // A forward pass of an inference context writes nothing, so the sum below can run as many
    // times as the difference needs
    let sum_of = |tensors: &[Tensor]| -> f32 {
        let refs: Vec<&Tensor> = tensors.iter().collect();
        layer
            .forward_many(&refs, &mut Ctx::inference())
            .unwrap()
            .sum()
    };

    for (position, gradient) in analytic.iter().enumerate() {
        assert_eq!(
            gradient.shape(),
            inputs[position].shape(),
            "the gradient of input {position} must hold the shape of that input"
        );
        for index in 0..inputs[position].len() {
            let mut raised = inputs.to_vec();
            let mut lowered = inputs.to_vec();
            raised[position].as_slice_mut().unwrap()[index] += h;
            lowered[position].as_slice_mut().unwrap()[index] -= h;
            let numeric = (sum_of(&raised) - sum_of(&lowered)) / (2.0 * h);
            assert_abs_diff_eq!(
                gradient.as_slice().unwrap()[index],
                numeric,
                epsilon = tolerance
            );
        }
    }
}

/// A tensor of distinct values, so no position of a maximum or a minimum ties
fn distinct(shape: &[usize], offset: f32) -> Tensor {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| offset + (i as f32) * 0.37 - 1.0)
        .collect();
    Array::from_shape_vec(ndarray::IxDyn(shape), values).unwrap()
}

/// Add gives every input the whole gradient
#[test]
fn merge_add_input_gradients() {
    let mut layer = Add::new();
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 3], 1.5)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// An input that broadcast reduces its gradient back to its own shape
#[test]
fn merge_add_reduces_a_broadcast_gradient() {
    let mut layer = Add::new();
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 1], 1.5)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// Rank alignment inserts an axis after the batch axis, and the gradient sums it away
#[test]
fn merge_add_reduces_a_rank_aligned_gradient() {
    let mut layer = Add::new();
    let inputs = vec![distinct(&[2, 3, 4], 0.0), distinct(&[2, 4], 1.5)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 2e-2);
}

/// Subtract negates the gradient of its second input
#[test]
fn merge_subtract_input_gradients() {
    let mut layer = Subtract::new();
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 3], 2.0)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// Multiply gives each input the product of the others, which is the only non-linear merge
#[test]
fn merge_multiply_input_gradients() {
    let mut layer = Multiply::new();
    let inputs = vec![
        distinct(&[2, 3], 0.5),
        distinct(&[2, 3], 1.5),
        distinct(&[2, 3], 2.5),
    ];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 3e-2);
}

/// Multiply reduces a broadcast gradient back, over a product of 2 inputs
#[test]
fn merge_multiply_reduces_a_broadcast_gradient() {
    let mut layer = Multiply::new();
    let inputs = vec![distinct(&[2, 3], 0.5), distinct(&[2, 1], 1.5)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 3e-2);
}

/// Average divides by the input count, so every gradient carries that factor
#[test]
fn merge_average_input_gradients() {
    let mut layer = Average::new();
    let inputs = vec![
        distinct(&[2, 3], 0.0),
        distinct(&[2, 3], 1.0),
        distinct(&[2, 3], 2.0),
    ];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// Maximum routes the gradient of a position to the input that wins it
///
/// The values are distinct, so no position ties and the function is differentiable at every
/// point the difference reads
#[test]
fn merge_maximum_input_gradients() {
    let mut layer = Maximum::new();
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 3], 0.13)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-3, 1e-2);
}

/// Minimum routes the gradient of a position to the input that wins it
#[test]
fn merge_minimum_input_gradients() {
    let mut layer = Minimum::new();
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 3], 0.13)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-3, 1e-2);
}

/// Concatenate slices the gradient along its axis, in input order
#[test]
fn merge_concatenate_input_gradients() {
    let mut layer = Concatenate::new(-1);
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[2, 2], 1.0)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// Concatenate joins along a leading axis as well, because its axis counts from the full rank
#[test]
fn merge_concatenate_on_the_batch_axis_input_gradients() {
    let mut layer = Concatenate::new(0);
    let inputs = vec![distinct(&[2, 3], 0.0), distinct(&[4, 3], 1.0)];
    check_merge_input_gradients(&mut layer, &inputs, 1e-2, 1e-2);
}

/// The reversal is its own inverse, and its input gradient is the reversed upstream gradient
#[test]
fn reverse_input_gradient_matches_finite_difference() {
    use rustyml::neural_network::layers::Reverse;

    let values: Vec<f32> = (0..24).map(|v| v as f32 * 0.1 - 0.7).collect();
    let x = Array::from_shape_vec((2, 4, 3), values).unwrap().into_dyn();
    for axis in [1_i32, 2, -1, -2] {
        let mut layer = Reverse::new(axis);
        check_input_gradient(&mut layer, &x, 1e-3, 2e-3);
    }
}

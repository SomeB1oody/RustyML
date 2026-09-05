//! Integration tests for Conv1D and Conv2D forward values, shapes, error paths,
//! param counts, set_weights, and predict==forward equivalence.
//!
//! Layout is Keras channels-last: Conv1D sees \[batch, length, channels\] and Conv2D sees
//! \[batch, height, width, channels\]. Kernels are \[k, Cin, F\] / \[kh, kw, Cin, F\] and every
//! bias is rank-1 of length `filters`.
//!
//! Expected values come from the cross-correlation definition with known weights,
//! not from recorded implementation output. Gradient checks live in gradient_check.rs.
//!
//! The dilation and causal-padding cases at the end also cover DepthwiseConv1D and
//! SeparableConv1D, because those 2 layers share the dilated kernel with the depthwise stage.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array3, Array4, array};
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::{Conv1D, ConvPadding};
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::traits::{LayerBase, ParamId, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

use super::common::{assert_allclose, named};

// Conv1D - forward with known weights

/// All-ones kernel=3, stride=1, Valid, Linear: each output is a windowed sum
#[test]
fn conv1d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // Windowed sums of [1,2,3,4,5]: 1+2+3, 2+3+4, 3+4+5
    let expected = Array::from_shape_vec((1, 3, 1), vec![6.0f32, 9.0, 12.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// All-ones kernel, stride=2, Valid: selects every other window
#[test]
fn conv1d_stride2_windowed_sums() {
    let mut layer = Conv1D::new(1, 3, 2, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 6, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 1]);
    // Windows start at 0 and 2: 1+2+3=6, 3+4+5=12
    let expected = Array::from_shape_vec((1, 2, 1), vec![6.0f32, 12.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Asymmetric kernel weight=[2, 0, 1], bias=0: output respects tap positions
#[test]
fn conv1d_asymmetric_kernel_values() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    // weight shape [kernel=3, channels=1, filters=1]. Taps in order are 2, 0, 1.
    let weights = Array3::from_shape_vec((3, 1, 1), vec![2.0f32, 0.0, 1.0]).unwrap();
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // Cross-correlation, no kernel flip: out[i] = 2*x[i] + 0*x[i+1] + 1*x[i+2]
    // 2*1+3=5, 2*2+4=8, 2*3+5=11
    let expected = Array::from_shape_vec((1, 3, 1), vec![5.0f32, 8.0, 11.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Bias adds a constant to every output element
#[test]
fn conv1d_bias_offset_adds_to_every_output() {
    let mut layer = Conv1D::new(1, 2, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    // taps [1, 0]: the layer copies x[i], then bias adds 5
    let weights = Array3::from_shape_vec((2, 1, 1), vec![1.0f32, 0.0]).unwrap();
    let bias = Array1::from_elem(1, 5.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 4, 1]);
    let expected = Array::from_shape_vec((1, 4, 1), vec![6.0f32, 7.0, 8.0, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// 2 filters with different tap patterns and biases produce independent outputs
#[test]
fn conv1d_two_filters_independent_outputs() {
    let mut layer = Conv1D::new(2, 2, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    // weight [kernel=2, channels=1, filters=2], flat order (k0,f0), (k0,f1), (k1,f0), (k1,f1).
    // Filter 0 taps = [1, 1] and filter 1 taps = [0, 1].
    let weights = Array3::from_shape_vec((2, 1, 2), vec![1.0f32, 0.0, 1.0, 1.0]).unwrap();
    let bias = array![0.0f32, 10.0];
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 4, 2]);

    // Filter 0, taps [1,1], bias 0: windowed sums of width 2 over [1,2,3,4,5]
    for (i, expected_val) in [3.0f32, 5.0, 7.0, 9.0].iter().enumerate() {
        assert_abs_diff_eq!(output[[0, i, 0]], *expected_val, epsilon = 1e-6f32);
    }
    // Filter 1, taps [0,1], bias 10: out[i] = x[i+1] + 10
    for (i, expected_val) in [12.0f32, 13.0, 14.0, 15.0].iter().enumerate() {
        assert_abs_diff_eq!(output[[0, i, 1]], *expected_val, epsilon = 1e-6f32);
    }
}

/// ReLU activation clips negative pre-activations to zero
#[test]
fn conv1d_relu_activation_clips_negatives() {
    let mut layer = Conv1D::new(1, 3, 1, ReLU::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![-1.0f32, -2.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // Pre-activation windowed sums are -2, 1, 6. ReLU clips the first value to 0.
    let expected = Array::from_shape_vec((1, 3, 1), vec![0.0f32, 1.0, 6.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Same padding, stride=1: output length stays at input length with symmetric zero padding
#[test]
fn conv1d_same_padding_forward_values() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 5, 1]);
    // pad_total = (5-1)*1 + 3 - 5 = 2, so pad_before = pad_after = 1: [0,1,2,3,4,5,0]
    // Windowed sums: 0+1+2, 1+2+3, 2+3+4, 3+4+5, 4+5+0
    let expected = Array::from_shape_vec((1, 5, 1), vec![3.0f32, 6.0, 9.0, 12.0, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Same padding, stride=2: output length is ceil(input_len / stride) with trailing-only pad
#[test]
fn conv1d_same_padding_stride2_output_length_and_values() {
    let mut layer = Conv1D::new(1, 3, 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    layer.build(&Shape::known(&[1, 6, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // out_len = ceil(6/2) = 3. pad_total = (3-1)*2 + 3 - 6 = 1, all on the trailing edge.
    // Padded input is [1,2,3,4,5,6,0]. Windows start at 0, 2, 4: 1+2+3, 3+4+5, 5+6+0
    let expected = Array::from_shape_vec((1, 3, 1), vec![6.0f32, 12.0, 11.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Batch size > 1: the layer processes identical samples independently and yields identical outputs
#[test]
fn conv1d_batch_forward_independent_samples() {
    let mut layer = Conv1D::new(1, 2, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[2, 4, 1])).unwrap();
    let weights = Array3::from_elem((2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // 2 identical samples [1,2,3,4]. Each yields windowed sums [3,5,7].
    let input = Array::from_shape_vec((2, 4, 1), vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[2, 3, 1]);
    // Both samples produce the same output
    for b in 0..2 {
        assert_abs_diff_eq!(output[[b, 0, 0]], 3.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 1, 0]], 5.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 2, 0]], 7.0f32, epsilon = 1e-6f32);
    }
}

/// input_length == kernel_size produces exactly 1 output element
#[test]
fn conv1d_input_equals_kernel_produces_single_output() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 3, 1), vec![10.0f32, 20.0, 30.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 60.0f32, epsilon = 1e-6f32);
}

/// 2 input channels: the filter sums contributions across both channels
#[test]
fn conv1d_two_input_channels_cross_channel_sum() {
    let mut layer = Conv1D::new(1, 2, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 2])).unwrap();
    // weight [kernel=2, channels=2, filters=1]: channel 0 taps = [1,1], channel 1 taps = [2,2].
    // Laid out kernel-major, the flat order is (k0,c0), (k0,c1), (k1,c0), (k1,c1)
    let weights = Array3::from_shape_vec((2, 2, 1), vec![1.0f32, 2.0, 1.0, 2.0]).unwrap();
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // input [batch=1, length=3, channels=2]: ch0=[1,2,3], ch1=[10,20,30], interleaved per
    // position. The 2 channels differ, so a swapped channel axis would change the result
    let input = Array::from_shape_vec((1, 3, 2), vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 1]);
    // out[0] = 1*(1+2) + 2*(10+20) = 3 + 60 = 63
    assert_abs_diff_eq!(output[[0, 0, 0]], 63.0f32, epsilon = 1e-6f32);
    // out[1] = 1*(2+3) + 2*(20+30) = 5 + 100 = 105
    assert_abs_diff_eq!(output[[0, 1, 0]], 105.0f32, epsilon = 1e-6f32);
}

// Conv1D - param_count

/// param_count = filters * channels * kernel_size + filters
#[test]
fn conv1d_param_count_formula() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut layer = Conv1D::new(4, 2, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 8, 3])).unwrap();
    assert_eq!(layer.param_count(), ParamCounts::trainable(28));
}

/// param_count for filters=2, channels=1, kernel=3 is 2*1*3 + 2 = 8
#[test]
fn conv1d_param_count_single_channel() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut layer = Conv1D::new(2, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    assert_eq!(layer.param_count(), ParamCounts::trainable(8));
}

// Conv1D - set_weights

/// set_weights with the correct shape succeeds and the forward output reflects it
#[test]
fn conv1d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();

    // All-twos weights: output is 2x the windowed sum
    let weights = Array3::from_elem((3, 1, 1), 2.0f32);
    let bias = Array1::zeros(1);
    assert!(layer.set_weights(weights, bias).is_ok());

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    let expected = Array::from_shape_vec((1, 3, 1), vec![12.0f32, 18.0, 24.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// set_weights with a mismatched weight shape returns a WeightShape error
#[test]
fn conv1d_set_weights_mismatched_weight_shape_errors() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();

    // Wrong weight shape [3, 1, 2] instead of [3, 1, 1]
    let wrong_weights = Array3::zeros((3, 1, 2));
    let bias = Array1::zeros(1);
    let result = layer.set_weights(wrong_weights, bias);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

/// set_weights with a mismatched bias shape returns a WeightShape error
#[test]
fn conv1d_set_weights_mismatched_bias_shape_errors() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();

    let weights = Array3::zeros((3, 1, 1));
    // Wrong bias [2] instead of [1]
    let wrong_bias = Array1::zeros(2);
    let result = layer.set_weights(weights, wrong_bias);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

// Conv1D - constructor error paths

/// Each invalid scalar hyperparameter (filters=0, kernel_size=0, stride=0)
/// independently makes the constructor return InvalidParameter
#[test]
fn conv1d_invalid_scalar_param_errors() {
    // (filters, kernel_size, stride, label)
    let cases = [
        (0usize, 3usize, 1usize, "filters=0"),
        (1, 0, 1, "kernel_size=0"),
        (1, 3, 0, "stride=0"),
    ];
    for (filters, kernel_size, stride, label) in cases {
        let result = Conv1D::new(filters, kernel_size, stride, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label}, got {:?}",
            result
        );
    }
}

/// Each invalid input_shape (wrong ndim or zero channels) independently makes the build step
/// return InvalidInput
///
/// A length below the kernel size is not on this list. `Same` and `Causal` padding make that
/// geometry legal. The padding mode is not final at construction, so the forward pass carries
/// the rule instead.
#[test]
fn conv1d_build_rejects_invalid_input_shape() {
    // (input_shape, label)
    let cases = [
        (vec![1, 5], "2D input_shape"),       // input_shape must be 3D
        (vec![1, 5, 5, 1], "4D input_shape"), // input_shape must be 3D
        (vec![1, 5, 0], "channels=0"),        // channels is the last axis
    ];
    for (input_shape, label) in cases {
        let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
        let result = layer.build(&Shape::known(&input_shape));
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput for {label}, got {:?}",
            result
        );
    }
}

// Conv1D - forward error paths

/// Passing a 2D tensor to forward() returns InvalidInput
#[test]
fn conv1d_forward_wrong_ndim_errors() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    let bad_input = Array::zeros((1_usize, 5_usize)).into_dyn();
    let result = layer.forward_mut(&bad_input, &mut Ctx::training());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D tensor, got {:?}",
        result
    );
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn conv1d_backward_before_forward_errors() {
    let layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3_usize, 1_usize)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun error, got {:?}",
        result
    );
}

// Conv1D - predict == forward in eval mode (no randomness, layer is stateless)

/// predict() returns the same values as forward() for a deterministic layer
#[test]
fn conv1d_predict_equals_forward() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input, &mut Ctx::training()).unwrap();
    let predict_output = layer.forward(&input, &mut Ctx::inference()).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv1d_predict_deterministic() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![2.0f32, 4.0, 6.0, 8.0, 10.0])
        .unwrap()
        .into_dyn();

    let out1 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    let out2 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv1D - named weight shapes

/// The Conv1D kernel has shape [kernel, channels, filters], and the bias [filters]
#[test]
fn conv1d_weights_correct_shapes() {
    let mut layer = Conv1D::new(3, 5, 1, Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 10, 2])).unwrap();
    assert_eq!(named(&layer, "kernel").shape(), &[5, 2, 3]);
    assert_eq!(named(&layer, "bias").shape(), &[3]);
}

// Conv2D - forward with known weights

/// All-ones 2x2 kernel, 1 channel, Valid, stride=(1,1): each output is a 2x2 windowed sum
#[test]
fn conv2d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // Rows of the 4x4 plane: [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]
    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 3, 1]);

    // 2x2 sums, e.g. (0,0) = 1+2+5+6 = 14, (1,1) = 6+7+10+11 = 34, (2,2) = 11+12+15+16 = 54
    let expected_flat = vec![14.0f32, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
    for (i, &expected_val) in expected_flat.iter().enumerate() {
        let h = i / 3;
        let w = i % 3;
        assert_abs_diff_eq!(output[[0, h, w, 0]], expected_val, epsilon = 1e-5f32);
    }
}

/// 1x1 kernel acts as a per-element scalar multiply
#[test]
fn conv2d_1x1_kernel_scalar_multiply() {
    let mut layer = Conv2D::new(1, (1, 1), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 2.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = input[[0, i, j, 0]] * 2.0;
            assert_abs_diff_eq!(output[[0, i, j, 0]], expected, epsilon = 1e-6f32);
        }
    }
}

/// Bias adds a constant to every output element
#[test]
fn conv2d_bias_shifts_all_outputs() {
    let mut layer = Conv2D::new(1, (1, 1), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2, 2, 1])).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 1.0f32);
    let bias = Array1::from_elem(1, 3.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 2, 2, 1), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    let expected = Array::from_shape_vec((1, 2, 2, 1), vec![4.0f32, 5.0, 6.0, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// stride=(2,2) with Valid padding: output shape is floor((H-k)/s)+1 with strided windows
#[test]
fn conv2d_stride2_valid_output_shape_and_values() {
    let mut layer = Conv2D::new(1, (2, 2), (2, 2), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    // Windows anchored at rows/cols {0, 2}
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 14.0f32, epsilon = 1e-5f32); // 1+2+5+6
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 22.0f32, epsilon = 1e-5f32); // 3+4+7+8
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 46.0f32, epsilon = 1e-5f32); // 9+10+13+14
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 54.0f32, epsilon = 1e-5f32); // 11+12+15+16
}

/// 2 filters with independent weight patterns: all-ones kernel vs top-left-only kernel
#[test]
fn conv2d_two_filters_independent_outputs() {
    let mut layer = Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();
    // weight [kh=2, kw=2, channels=1, filters=2]: filter0 = [[1,1],[1,1]], filter1 = [[1,0],[0,0]].
    // Flat order is (kh,kw)-major with the filter axis last, so each pair is (f0, f1) at 1 tap
    let weights = Array4::from_shape_vec(
        (2, 2, 1, 2),
        vec![1.0f32, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let bias = Array1::zeros(2);
    layer.set_weights(weights, bias).unwrap();

    // Rows of the 3x3 plane: [1,2,3], [4,5,6], [7,8,9]
    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    // Filter 0: 2x2 sums
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 12.0f32, epsilon = 1e-5f32); // 1+2+4+5
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 16.0f32, epsilon = 1e-5f32); // 2+3+5+6
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 24.0f32, epsilon = 1e-5f32); // 4+5+7+8
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 28.0f32, epsilon = 1e-5f32); // 5+6+8+9

    // Filter 1: top-left position only
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 1.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 2.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 0, 1]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 1, 1]], 5.0f32, epsilon = 1e-5f32);
}

/// 2 input channels: the filter sums contributions across the (last) channel axis
#[test]
fn conv2d_two_input_channels_cross_channel_sum() {
    let mut layer = Conv2D::new(1, (1, 1), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 2, 2, 2])).unwrap();
    // weight [kh=1, kw=1, channels=2, filters=1]: channel 0 weight 1, channel 1 weight 10
    let weights = Array4::from_shape_vec((1, 1, 2, 1), vec![1.0f32, 10.0]).unwrap();
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // input [batch=1, height=2, width=2, channels=2], channels innermost:
    // (0,0) -> [1,2], (0,1) -> [3,4], (1,0) -> [5,6], (1,1) -> [7,8]
    let input = Array::from_shape_vec(
        (1, 2, 2, 2),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap()
    .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    // out[h,w] = 1*x[h,w,0] + 10*x[h,w,1]. The per-channel weights and the per-channel data
    // both differ, so reading the channel axis anywhere but last changes every value.
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 21.0f32, epsilon = 1e-5f32); // 1 + 10*2
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 43.0f32, epsilon = 1e-5f32); // 3 + 10*4
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 65.0f32, epsilon = 1e-5f32); // 5 + 10*6
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 87.0f32, epsilon = 1e-5f32); // 7 + 10*8
}

/// ReLU clips negative pre-activations to zero
#[test]
fn conv2d_relu_clips_negatives() {
    let mut layer = Conv2D::new(1, (2, 1), (1, 1), ReLU::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 1, 1])).unwrap();
    let weights = Array4::from_elem((2, 1, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 3, 1, 1), vec![-3.0f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 2, 1, 1]);
    // Pre-activations: -3+1 = -2 (clipped) and 1+2 = 3
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 0.0f32, epsilon = 1e-6f32);
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 3.0f32, epsilon = 1e-6f32);
}

/// Same padding output shape is ceil(H/stride) x ceil(W/stride)
#[test]
fn conv2d_same_padding_output_shapes() {
    // Case 1: stride=(1,1)
    let mut layer1 = Conv2D::new(1, (3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x1 = Array::ones((1, 5, 5, 1)).into_dyn();
    let out1 = layer1.forward_mut(&x1, &mut Ctx::training()).unwrap();
    assert_eq!(out1.shape(), &[1, 5, 5, 1]);

    // Case 2: stride=(2,2): ceil(4/2)=2
    let mut layer2 = Conv2D::new(1, (3, 3), (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x2 = Array::ones((1, 4, 4, 1)).into_dyn();
    let out2 = layer2.forward_mut(&x2, &mut Ctx::training()).unwrap();
    assert_eq!(out2.shape(), &[1, 2, 2, 1]);
}

/// Same padding with all-ones kernel and input: corners sum 4, edges 6, center 9 ones
#[test]
fn conv2d_same_padding_all_ones_values() {
    let mut layer = Conv2D::new(1, (3, 3), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();
    let weights = Array4::from_elem((3, 3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((1, 3, 3, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    // pad_total = (3-1)*1 + 3 - 3 = 2 per axis, 1 row/column of zeros on each side
    // Corners: 2x2 = 4 overlapping ones
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 2, 0, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 2, 2, 0]], 4.0f32, epsilon = 1e-5f32);
    // Edge midpoints: 2x3 = 6 overlapping ones
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 2, 0]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 2, 1, 0]], 6.0f32, epsilon = 1e-5f32);
    // Center: 3x3 = 9 overlapping ones
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 9.0f32, epsilon = 1e-5f32);
}

// Conv2D - param_count

/// param_count = filters * channels * kh * kw + filters
#[test]
fn conv2d_param_count_formula() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut layer = Conv2D::new(2, (3, 3), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 5, 5, 3])).unwrap();
    assert_eq!(layer.param_count(), ParamCounts::trainable(56));
}

/// param_count for filters=1, channels=1, kernel=(2,2) is 1*1*2*2 + 1 = 5
#[test]
fn conv2d_param_count_single_filter_single_channel() {
    use rustyml::neural_network::layers::ParamCounts;
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    assert_eq!(layer.param_count(), ParamCounts::trainable(5));
}

// Conv2D - set_weights

/// set_weights with the correct shape succeeds and new weights change the output
#[test]
fn conv2d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv2D::new(1, (1, 1), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();
    // weight=3.0 with a 1x1 kernel: output is 3 * input
    let weights = Array4::from_elem((1, 1, 1, 1), 3.0f32);
    let bias = Array1::zeros(1);
    assert!(layer.set_weights(weights, bias).is_ok());

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(
                output[[0, i, j, 0]],
                input[[0, i, j, 0]] * 3.0,
                epsilon = 1e-6f32
            );
        }
    }
}

/// set_weights with a mismatched weight shape returns a WeightShape error
#[test]
fn conv2d_set_weights_mismatched_weight_shape_errors() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    // Wrong weight shape [2,2,1,2] instead of [2,2,1,1]
    let wrong_weights = Array4::zeros((2, 2, 1, 2));
    let bias = Array1::zeros(1);
    let result = layer.set_weights(wrong_weights, bias);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

/// set_weights with a mismatched bias shape returns a WeightShape error
#[test]
fn conv2d_set_weights_mismatched_bias_shape_errors() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    let weights = Array4::zeros((2, 2, 1, 1));
    // Wrong bias [3] instead of [1]
    let wrong_bias = Array1::zeros(3);
    let result = layer.set_weights(weights, wrong_bias);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

// Conv2D - constructor error paths

/// Each invalid scalar hyperparameter (filters=0, kernel height/width=0,
/// stride height/width=0) independently makes the constructor return InvalidParameter
#[test]
fn conv2d_invalid_scalar_param_errors() {
    // (filters, kernel, stride, label)
    let cases = [
        (0usize, (3usize, 3usize), (1usize, 1usize), "filters=0"),
        (1, (0, 3), (1, 1), "kernel height=0"),
        (1, (3, 0), (1, 1), "kernel width=0"),
        (1, (3, 3), (0, 1), "stride height=0"),
        (1, (3, 3), (1, 0), "stride width=0"),
    ];
    for (filters, kernel, stride, label) in cases {
        let result = Conv2D::new(filters, kernel, stride, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label}, got {:?}",
            result
        );
    }
}

/// Each invalid input_shape (wrong ndim or zero channels) independently makes the build step
/// return InvalidInput
///
/// A spatial axis below the kernel size is not on this list. `Same` padding makes that geometry
/// legal, so the forward pass carries the rule instead.
#[test]
fn conv2d_build_rejects_invalid_input_shape() {
    // (input_shape, label)
    let cases = [
        (vec![1, 5, 1], "3D input_shape"),       // input_shape must be 4D
        (vec![1, 5, 5, 5, 1], "5D input_shape"), // input_shape must be 4D
        (vec![1, 5, 5, 0], "channels=0"),        // channels is the last axis
    ];
    for (input_shape, label) in cases {
        let mut layer = Conv2D::new(1, (3, 3), (1, 1), Linear::new()).unwrap();
        let result = layer.build(&Shape::known(&input_shape));
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput for {label}, got {:?}",
            result
        );
    }
}

// Conv2D - forward error paths

/// Passing a 3D tensor to forward() returns InvalidInput (4D expected)
#[test]
fn conv2d_forward_wrong_ndim_errors() {
    let mut layer = Conv2D::new(1, (3, 3), (1, 1), Linear::new()).unwrap();
    let bad_input = Array::zeros((1_usize, 5_usize, 5_usize)).into_dyn();
    let result = layer.forward_mut(&bad_input, &mut Ctx::training());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D tensor, got {:?}",
        result
    );
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn conv2d_backward_before_forward_errors() {
    let layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3_usize, 3_usize, 1_usize)).into_dyn();
    let result = layer.backward(&grad, &mut Ctx::training());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun error, got {:?}",
        result
    );
}

// Conv2D - predict == forward

/// predict() equals forward() for a deterministic layer
#[test]
fn conv2d_predict_equals_forward() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input, &mut Ctx::training()).unwrap();
    let predict_output = layer.forward(&input, &mut Ctx::inference()).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv2d_predict_deterministic() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 1])).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();

    let out1 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    let out2 = layer.forward(&input, &mut Ctx::inference()).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv2D - named weight shapes

/// The Conv2D kernel has shape [kh, kw, channels, filters], and the bias [filters]
#[test]
fn conv2d_weights_correct_shapes() {
    let mut layer = Conv2D::new(4, (3, 3), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[1, 8, 8, 2])).unwrap();
    assert_eq!(named(&layer, "kernel").shape(), &[3, 3, 2, 4]);
    assert_eq!(named(&layer, "bias").shape(), &[4]);
}

// Conv1D - Valid output length formula for diverse (len, kernel, stride) cases

/// Valid output length is (input_len - kernel) / stride + 1 across parameterized cases
#[test]
fn conv1d_valid_output_length_cases() {
    // (input_len, kernel, stride, expected_out_len)
    let cases = [
        (10usize, 3usize, 1usize, 8usize),
        (10, 3, 2, 4),
        (8, 4, 2, 3),
        (6, 2, 3, 2),
        (5, 5, 1, 1),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut conv = Conv1D::new(1, kernel, stride, Linear::new()).unwrap();
        let x = Array::ones((1, len, 1)).into_dyn();
        let mut ctx = Ctx::training();
        let out = conv.forward_mut(&x, &mut ctx).unwrap();
        assert_eq!(
            out.shape(),
            &[1, expected, 1],
            "Conv1D Valid: input_len={len}, kernel={kernel}, stride={stride}"
        );
    }
}

// Conv1D - Same output length = ceil(input_len / stride) for diverse cases

/// Same output length is ceil(input_len / stride) across parameterized cases
#[test]
fn conv1d_same_output_length_cases() {
    let cases = [
        (10usize, 3usize, 1usize, 10usize),
        (10, 3, 2, 5),
        (8, 5, 1, 8),
        (7, 3, 2, 4),
        (9, 4, 3, 3),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut conv = Conv1D::new(1, kernel, stride, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same);
        let x = Array::ones((1, len, 1)).into_dyn();
        let mut ctx = Ctx::training();
        let out = conv.forward_mut(&x, &mut ctx).unwrap();
        assert_eq!(
            out.shape(),
            &[1, expected, 1],
            "Conv1D Same: input_len={len}, kernel={kernel}, stride={stride}"
        );
    }
}

// Conv2D - Valid output shape for diverse (H, W, kh, kw, sh, sw) cases

/// Valid output shape is (dim - kernel) / stride + 1 on each spatial axis, across parameterized
/// cases
#[test]
fn conv2d_valid_output_shape_cases() {
    // (H, W, kh, kw, sh, sw, out_h, out_w)
    let cases = [
        (
            6usize, 6usize, 3usize, 3usize, 1usize, 1usize, 4usize, 4usize,
        ),
        (6, 8, 3, 3, 1, 1, 4, 6),
        (8, 8, 3, 3, 2, 2, 3, 3),
        (5, 5, 2, 2, 1, 1, 4, 4),
        (5, 7, 2, 3, 2, 1, 2, 5),
    ];
    for (h, w, kh, kw, sh, sw, out_h, out_w) in cases {
        let mut conv = Conv2D::new(1, (kh, kw), (sh, sw), Linear::new()).unwrap();
        let x = Array::ones((1, h, w, 1)).into_dyn();
        let mut ctx = Ctx::training();
        let out = conv.forward_mut(&x, &mut ctx).unwrap();
        assert_eq!(
            out.shape(),
            &[1, out_h, out_w, 1],
            "Conv2D Valid: H={h}, W={w}, k=({kh},{kw}), s=({sh},{sw})"
        );
    }
}
// Conv2D - convolution_engine parallel forward / weight-grad backward branches

// The engine runs in parallel once the estimated GEMM FLOPs clear the gate in
// `crate::tuning::conv`. Each test below sizes its tensors to clear that gate. Each test also
// checks the crossing against the live gate value first. A retuned threshold then fails the
// assertion instead of silently demoting the test to the serial path.

/// Parallel forward branch: windowed sums on a tensor whose estimated FLOPs clear the gate
#[test]
fn conv2d_parallel_forward_windowed_sums() {
    // The gate values are process-global. This guard holds the shared side of the lock
    // in `common`, so no test that moves a gate runs while this test reads one
    let _gates = crate::common::read_gates();
    // 2 * batch(2) * filters(2) * out_plane(354*354) * Cin*k(1*2*2) = 4_010_112
    let gemm_flops = 2 * 2 * 2 * (354 * 354) * 4;
    assert!(
        gemm_flops >= rustyml::tuning::conv::get_parallel_min_flops(),
        "fixture no longer clears the engine's parallel gate: {gemm_flops} FLOPs"
    );

    let mut layer = Conv2D::new(2, (2, 2), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[2, 355, 355, 1])).unwrap();
    // Both filters use an all-ones 2x2 kernel. Bias is filter0=0, filter1=100.
    let weights = Array4::from_elem((2, 2, 1, 2), 1.0f32);
    let bias = array![0.0f32, 100.0];
    layer.set_weights(weights, bias).unwrap();

    // x[b,i,j,0] = i + j, identical across the 2 batch samples
    let mut input = Array4::<f32>::zeros((2, 355, 355, 1));
    for b in 0..2 {
        for i in 0..355 {
            for j in 0..355 {
                input[[b, i, j, 0]] = (i + j) as f32;
            }
        }
    }
    let mut ctx = Ctx::training();
    let output = layer.forward(&input.into_dyn(), &mut ctx).unwrap();

    assert_eq!(output.shape(), &[2, 354, 354, 2]);

    // The 2x2 window at (oh, ow) sums (oh+ow) + (oh+ow+1) + (oh+ow+1) + (oh+ow+2), which is
    // 4*oh + 4*ow + 4. out[b,oh,ow,1] = that + 100, checked over both batches
    let f0 = |oh: usize, ow: usize| (4 * oh + 4 * ow + 4) as f32;
    for b in 0..2 {
        // corners / edges of filter 0
        assert_abs_diff_eq!(output[[b, 0, 0, 0]], f0(0, 0), epsilon = 1e-4f32); // 4
        assert_abs_diff_eq!(output[[b, 0, 1, 0]], f0(0, 1), epsilon = 1e-4f32); // 8
        assert_abs_diff_eq!(output[[b, 1, 0, 0]], f0(1, 0), epsilon = 1e-4f32); // 8
        assert_abs_diff_eq!(output[[b, 353, 353, 0]], f0(353, 353), epsilon = 1e-4f32); // 2828
        // filter 1 = filter 0 + bias 100
        assert_abs_diff_eq!(output[[b, 0, 0, 1]], f0(0, 0) + 100.0, epsilon = 1e-4f32); // 104
        assert_abs_diff_eq!(
            output[[b, 353, 353, 1]],
            f0(353, 353) + 100.0,
            epsilon = 1e-4f32
        ); // 2928
    }
}

/// Parallel weight-gradient backward branch: every kernel tap accumulates the same count
#[test]
fn conv2d_parallel_weight_grad_constant_count() {
    // The gate values are process-global. This guard holds the shared side of the lock
    // in `common`, so no test that moves a gate runs while this test reads one
    let _gates = crate::common::read_gates();
    // 4 * batch(4) * filters(1) * out_plane(118*118) * Cin*k(2*3*3) = 4_010_112
    let gemm_flops = 4 * 4 * (118 * 118) * 18;
    assert!(
        gemm_flops >= rustyml::tuning::conv::get_parallel_min_flops(),
        "fixture no longer clears the engine's parallel gate: {gemm_flops} FLOPs"
    );

    let mut layer = Conv2D::new(1, (3, 3), (1, 1), Linear::new()).unwrap();
    layer.build(&Shape::known(&[4, 120, 120, 2])).unwrap();
    // Weight values do not affect the weight gradient here. This test uses all-ones weights
    // and bias 0, for determinism.
    let weights = Array4::from_elem((3, 3, 2, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((4, 120, 120, 2)).into_dyn();
    let mut ctx = Ctx::training();
    let _ = layer.forward(&input, &mut ctx).unwrap();

    // gradient of the conv output: all ones, shape [batch, out_h, out_w, filters]
    let grad = Array::ones((4, 118, 118, 1)).into_dyn();
    let _input_grad = layer.backward(&grad, &mut ctx).unwrap();

    // The gradient store holds the weight gradient under its parameter name.
    let weight_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    // 3*3 taps * 2 input channels * 1 filter
    assert_eq!(weight_grad.len(), 18, "expected 18 weight-grad entries");
    // dL/dw[kh,kw,c,f] = sum_{b,oh,ow} grad * x = 4 batches * 118*118 positions * 1 * 1 = 55_696
    for &g in weight_grad.iter() {
        assert_abs_diff_eq!(g, 55_696.0f32, epsilon = 1e-2f32);
    }
}

// Dilation

/// A dilated kernel skips input positions between its taps
///
/// 3 taps spaced 2 apart span 5 cells, so a length of 9 holds 5 windows and output `o` sums
/// `x[o]`, `x[o + 2]`, and `x[o + 4]`. An undilated kernel would sum 3 adjacent cells instead.
#[test]
fn conv1d_dilation_spaces_the_taps_out() {
    let mut layer = Conv1D::new(1, 3, 1, Linear::new())
        .unwrap()
        .with_dilation_rate(2)
        .unwrap();
    layer.build(&Shape::known(&[1, 9, 1])).unwrap();
    layer
        .set_weights(Array3::from_elem((3, 1, 1), 1.0f32), Array1::zeros(1))
        .unwrap();

    let input = Array::from_shape_vec((1, 9, 1), (1..=9).map(|v| v as f32).collect::<Vec<f32>>())
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 5, 1], "keff = 5 leaves 5 windows");
    let expected = Array::from_shape_vec((1, 5, 1), vec![9.0f32, 12.0, 15.0, 18.0, 21.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// `Same` padding keeps the input length at every dilation
///
/// The padding rule pads by the dilated extent, not by the kernel size, so the output length
/// cannot change with the dilation. An implementation that pads by `(k - 1) / 2` shortens the
/// output as soon as the dilation grows.
#[test]
fn conv1d_dilated_same_padding_keeps_the_length() {
    for dilation in [1usize, 2, 3] {
        let mut layer = Conv1D::new(1, 3, 1, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same)
            .with_dilation_rate(dilation)
            .unwrap();
        layer.build(&Shape::known(&[1, 8, 1])).unwrap();
        layer
            .set_weights(Array3::from_elem((3, 1, 1), 1.0f32), Array1::zeros(1))
            .unwrap();

        let input = Array::ones((1, 8, 1)).into_dyn();
        let mut ctx = Ctx::training();
        let output = layer.forward(&input, &mut ctx).unwrap();
        assert_eq!(
            output.shape(),
            &[1, 8, 1],
            "dilation {dilation} must keep the length"
        );
    }
}

/// A dilated kernel longer than the input is rejected under `Valid` and accepted under the
/// other 2 padding modes
///
/// A kernel of 3 fits a length of 6, but the 7 cells that same kernel spans at dilation 3 do
/// not. Under `Valid` no complete window fits, so the forward pass rejects the input. `Same`
/// and `Causal` supply the missing cells, and Keras 3.15.1 returns the input length there.
#[test]
fn a_dilated_kernel_longer_than_the_input_fails_only_under_valid() {
    let build = |padding: ConvPadding| {
        Conv1D::new(1, 3, 1, Linear::new())
            .unwrap()
            .with_padding(padding)
            .with_dilation_rate(3)
            .unwrap()
    };
    let input = Array::ones((1, 6, 1)).into_dyn();

    let err = build(ConvPadding::Valid)
        .forward_mut(&input, &mut Ctx::training())
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput under Valid, got {err:?}"
    );

    for padding in [ConvPadding::Same, ConvPadding::Causal] {
        let output = build(padding)
            .forward_mut(&input, &mut Ctx::training())
            .unwrap();
        assert_eq!(
            output.shape(),
            &[1, 6, 1],
            "{padding:?} must return the input length"
        );
    }

    // The same kernel at dilation 2 spans 5 cells, which fits under Valid as well
    let mut fits = build_dilation_2();
    assert_eq!(
        fits.forward_mut(&input, &mut Ctx::training())
            .unwrap()
            .shape(),
        &[1, 2, 1]
    );
}

/// A Conv1D at a dilation of 2, used by the padding-mode case above
fn build_dilation_2() -> Conv1D {
    Conv1D::new(1, 3, 1, Linear::new())
        .unwrap()
        .with_dilation_rate(2)
        .unwrap()
}

/// A plain convolution rejects a stride above 1 together with a dilation above 1
///
/// The rule reads the maximum over the axes, not 1 axis at a time. A stride above 1 on 1 axis
/// and a dilation above 1 on another axis is rejected as well.
#[test]
fn conv_rejects_a_stride_and_a_dilation_above_one_together() {
    let strided = Conv1D::new(1, 3, 2, Linear::new())
        .unwrap()
        .with_dilation_rate(2);
    assert!(
        matches!(strided, Err(Error::InvalidParameter { .. })),
        "Conv1D must reject stride 2 with dilation 2"
    );

    // Different axes, so no single axis carries both. The rule still fires
    let mixed = Conv2D::new(1, (2, 2), (2, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((1, 2));
    assert!(
        matches!(mixed, Err(Error::InvalidParameter { .. })),
        "Conv2D must reject a stride of 2 on 1 axis with a dilation of 2 on another"
    );

    // A depthwise convolution accepts the same pair, so the rule must not sit in a shared
    // validator
    assert!(
        DepthwiseConv1D::new(2, 2, Linear::new())
            .unwrap()
            .with_dilation_rate(3)
            .is_ok(),
        "DepthwiseConv1D must accept stride 2 with dilation 3"
    );
}

/// Each spatial axis carries its own dilation
///
/// A `(1, 3)` dilation leaves the height taps adjacent and spreads the width taps 3 apart, so
/// output `(oh, ow)` sums the 4 cells `(oh, ow)`, `(oh, ow + 3)`, `(oh + 1, ow)`, and
/// `(oh + 1, ow + 3)`. Each input cell holds `h * 10 + w`, so a swapped axis is visible at once.
#[test]
fn conv2d_dilation_is_per_axis() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((1, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 4, 7, 1])).unwrap();
    layer
        .set_weights(Array4::from_elem((2, 2, 1, 1), 1.0f32), Array1::zeros(1))
        .unwrap();

    let values: Vec<f32> = (0..4)
        .flat_map(|h| (0..7).map(move |w| (h * 10 + w) as f32))
        .collect();
    let input = Array::from_shape_vec((1, 4, 7, 1), values)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 4, 1], "keff is 2 by 4");
    let expected = Array::from_shape_vec(
        (1, 3, 4, 1),
        vec![
            26.0f32, 30.0, 34.0, 38.0, 66.0, 70.0, 74.0, 78.0, 106.0, 110.0, 114.0, 118.0,
        ],
    )
    .unwrap()
    .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// The window advances by the stride and the taps sit `dilation` apart. The 2 factors are
/// independent
///
/// Output `o` of a depthwise pass reads `o * stride + tap * dilation`. The wrong form
/// `(o * stride + tap) * dilation` agrees only when the stride equals the dilation, which the
/// depthwise layers do not forbid. A stride of 2 with a dilation of 3 tells the 2 apart: the
/// windows here read `(0, 3)`, `(2, 5)`, and `(4, 7)`, and the wrong form would read `(0, 3)`,
/// `(6, 9)`, and `(12, 15)`.
#[test]
fn depthwise_conv1d_keeps_the_stride_and_the_dilation_independent() {
    let mut layer = DepthwiseConv1D::new(2, 2, Linear::new())
        .unwrap()
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 8, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((2, 1, 1), vec![1.0f32, 10.0]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 8, 1), (1..=8).map(|v| v as f32).collect::<Vec<f32>>())
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // x[0] + 10 * x[3], x[2] + 10 * x[5], x[4] + 10 * x[7]
    let expected = Array::from_shape_vec((1, 3, 1), vec![41.0f32, 63.0, 85.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// The dilation reaches the depthwise stage of a separable convolution
///
/// The layer runs the same stride 2 and dilation 3 as the depthwise test above, behind a 1-tap
/// pointwise stage of weight 1. The values must therefore agree. The pointwise stage reads 1
/// tap, so no dilation can change it.
#[test]
fn separable_conv1d_dilates_the_depthwise_stage() {
    let mut layer = SeparableConv1D::new(1, 2, 2, 1, Linear::new())
        .unwrap()
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 8, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((2, 1, 1), vec![1.0f32, 10.0]).unwrap(),
            Array3::from_elem((1, 1, 1), 1.0f32),
            Array1::zeros(1),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 8, 1), (1..=8).map(|v| v as f32).collect::<Vec<f32>>())
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    let expected = Array::from_shape_vec((1, 3, 1), vec![41.0f32, 63.0, 85.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

// Causal padding

/// Causal padding puts all `(k - 1) * dilation` pad cells on the leading edge
///
/// The kernel is `[10, 1]` at dilation 3, so the leading pad is 3 cells and output `o` is
/// `10 * xpad[o] + 1 * xpad[o + 3]`. The first 3 outputs therefore hold only the second tap,
/// which reads the current position. A leading pad of `k - 1` (1 cell) or of `keff` (4 cells)
/// gives different values, and a split pad leaks a later input position into an earlier output.
#[test]
fn conv1d_causal_padding_is_all_on_the_leading_edge() {
    let mut layer = Conv1D::new(1, 2, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Causal)
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 6, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((2, 1, 1), vec![10.0f32, 1.0]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 6, 1), (1..=6).map(|v| v as f32).collect::<Vec<f32>>())
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 6, 1], "causal keeps the length");
    let expected = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 14.0, 25.0, 36.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// No causal output position depends on any later input position
///
/// This perturbs 1 input position at a time and compares the whole output against the
/// unperturbed run. Every output before that position must stay bit-identical, which is the
/// property a shape test cannot check. The output at the perturbed position must change, which
/// proves the last tap reads the current position rather than an earlier one.
#[test]
fn conv1d_causal_output_never_reads_a_later_input() {
    let (length, channels, filters, kernel, dilation) = (8usize, 2usize, 2usize, 3usize, 2usize);
    let mut layer = Conv1D::new(filters, kernel, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Causal)
        .with_dilation_rate(dilation)
        .unwrap();
    layer.build(&Shape::known(&[1, length, channels])).unwrap();
    let weights: Vec<f32> = (0..kernel * channels * filters)
        .map(|i| (i as f32) * 0.25 - 1.0)
        .collect();
    layer
        .set_weights(
            Array3::from_shape_vec((kernel, channels, filters), weights).unwrap(),
            Array1::from_vec(vec![0.5f32, -0.25]),
        )
        .unwrap();

    let base_values: Vec<f32> = (0..length * channels)
        .map(|i| (i as f32) * 0.5 - 2.0)
        .collect();
    let base_input = Array::from_shape_vec((1, length, channels), base_values.clone())
        .unwrap()
        .into_dyn();
    let base_output = layer.forward(&base_input, &mut Ctx::inference()).unwrap();

    for position in 0..length {
        let mut values = base_values.clone();
        for c in 0..channels {
            values[position * channels + c] += 3.0;
        }
        let input = Array::from_shape_vec((1, length, channels), values)
            .unwrap()
            .into_dyn();
        let output = layer.forward(&input, &mut Ctx::inference()).unwrap();

        for earlier in 0..position {
            for f in 0..filters {
                assert_eq!(
                    output[[0, earlier, f]],
                    base_output[[0, earlier, f]],
                    "output {earlier} moved when input {position} changed, so the pass reads \
                     the future"
                );
            }
        }
        let changed =
            (0..filters).any(|f| output[[0, position, f]] != base_output[[0, position, f]]);
        assert!(
            changed,
            "output {position} ignored its own input position, so the last tap is misplaced"
        );
    }
}

/// Causal and `Same` padding give the same output length and different values
///
/// A shape-only test cannot tell the 2 apart. This pins both halves: the lengths agree, and the
/// values do not.
#[test]
fn conv1d_causal_matches_same_in_length_only() {
    let build = |padding: ConvPadding| {
        let mut layer = Conv1D::new(1, 3, 1, Linear::new())
            .unwrap()
            .with_padding(padding);
        layer.build(&Shape::known(&[1, 6, 1])).unwrap();
        layer
            .set_weights(
                Array3::from_shape_vec((3, 1, 1), vec![1.0f32, 2.0, 4.0]).unwrap(),
                Array1::zeros(1),
            )
            .unwrap();
        layer
    };
    let input = Array::from_shape_vec((1, 6, 1), (1..=6).map(|v| v as f32).collect::<Vec<f32>>())
        .unwrap()
        .into_dyn();

    let causal = build(ConvPadding::Causal)
        .forward(&input, &mut Ctx::inference())
        .unwrap();
    let same = build(ConvPadding::Same)
        .forward(&input, &mut Ctx::inference())
        .unwrap();

    assert_eq!(causal.shape(), same.shape(), "both keep the input length");
    // Causal reads xpad = [0, 0, 1..6], so output 0 is 4 * 1 = 4. Same reads [0, 1..6, 0], so
    // its output 0 is 2 * 1 + 4 * 2 = 10
    assert_abs_diff_eq!(causal[[0, 0, 0]], 4.0f32, epsilon = 1e-6f32);
    assert_abs_diff_eq!(same[[0, 0, 0]], 10.0f32, epsilon = 1e-6f32);
}

/// A strided causal pass keeps `ceil(length / stride)` positions
#[test]
fn conv1d_causal_strided_output_length() {
    let mut layer = Conv1D::new(1, 3, 2, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Causal);
    layer.build(&Shape::known(&[1, 7, 1])).unwrap();
    let input = Array::ones((1, 7, 1)).into_dyn();
    let output = layer.forward(&input, &mut Ctx::inference()).unwrap();
    assert_eq!(output.shape(), &[1, 4, 1]);
}

/// The analytic gradients of a dilated causal pass match a central difference
///
/// The forward tests above fix the values. This checks that the backward pass transposes the
/// same geometry, for the input, the kernel, and the bias.
#[test]
fn conv1d_causal_dilated_gradients_match_a_finite_difference() {
    let (length, kernel, dilation) = (7usize, 3usize, 2usize);
    let weights = Array3::from_shape_vec((kernel, 1, 1), vec![0.5f32, -1.25, 2.0]).unwrap();
    let bias = Array1::from_vec(vec![0.25f32]);
    let build = |weights: &Array3<f32>, bias: &Array1<f32>| {
        let mut layer = Conv1D::new(1, kernel, 1, Linear::new())
            .unwrap()
            .with_padding(ConvPadding::Causal)
            .with_dilation_rate(dilation)
            .unwrap();
        layer.build(&Shape::known(&[1, length, 1])).unwrap();
        layer.set_weights(weights.clone(), bias.clone()).unwrap();
        layer
    };

    let x_values: Vec<f32> = (0..length).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let input = Array::from_shape_vec((1, length, 1), x_values.clone())
        .unwrap()
        .into_dyn();
    let upstream_values: Vec<f32> = (0..length).map(|i| 1.0 + (i as f32) * 0.25).collect();
    let upstream = Array::from_shape_vec((1, length, 1), upstream_values.clone())
        .unwrap()
        .into_dyn();

    let layer = build(&weights, &bias);
    let mut ctx = Ctx::training();
    layer.forward(&input, &mut ctx).unwrap();
    let input_grad = layer.backward(&upstream, &mut ctx).unwrap();
    let weight_grad = flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap());
    let bias_grad = flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap());

    // Loss is `sum(output * upstream)`, whose gradient of the output is `upstream`
    let loss = |layer: &Conv1D, input: &ndarray::ArrayD<f32>| -> f32 {
        layer
            .forward(input, &mut Ctx::inference())
            .unwrap()
            .iter()
            .zip(upstream.iter())
            .map(|(y, g)| y * g)
            .sum()
    };
    let step = 1e-2f32;
    let reference = build(&weights, &bias);

    for position in 0..length {
        let mut plus = input.clone();
        plus[[0, position, 0]] += step;
        let mut minus = input.clone();
        minus[[0, position, 0]] -= step;
        let numeric = (loss(&reference, &plus) - loss(&reference, &minus)) / (2.0 * step);
        assert_abs_diff_eq!(input_grad[[0, position, 0]], numeric, epsilon = 1e-3f32);
    }

    for tap in 0..kernel {
        let mut plus = weights.clone();
        plus[[tap, 0, 0]] += step;
        let mut minus = weights.clone();
        minus[[tap, 0, 0]] -= step;
        let numeric = (loss(&build(&plus, &bias), &input) - loss(&build(&minus, &bias), &input))
            / (2.0 * step);
        assert_abs_diff_eq!(weight_grad[tap], numeric, epsilon = 1e-3f32);
    }

    let mut plus = bias.clone();
    plus[0] += step;
    let mut minus = bias.clone();
    minus[0] -= step;
    let numeric = (loss(&build(&weights, &plus), &input) - loss(&build(&weights, &minus), &input))
        / (2.0 * step);
    assert_abs_diff_eq!(bias_grad[0], numeric, epsilon = 1e-3f32);
}

// ---------------------------------------------------------------------------------------
// Effective kernels longer than the input, and dilated gradients, pinned to Keras 3.15.1
// ---------------------------------------------------------------------------------------
//
// Keras 3.15.1 on the jax backend produced every expected value in this section. Each case
// gives the layer the weights, the input, and the upstream gradient that the ramps below
// build. A rerun of the probe therefore reproduces the numbers. Every ramp value is exact in
// f32, so no rounding enters the comparison.
//
// A kernel whose effective extent is longer than the input axis is legal under `Same` and
// `Causal` padding, and Keras accepts it. Only `Valid` rejects it, because no complete window
// fits there.

/// Kernel ramp: element `i` holds `((i % 7) - 3) * 0.25`
fn ramp_kernel(count: usize) -> Vec<f32> {
    (0..count).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect()
}

/// Pointwise-kernel ramp: element `i` holds `((i % 5) - 2) * 0.5`
fn ramp_pointwise(count: usize) -> Vec<f32> {
    (0..count).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect()
}

/// Bias ramp: filter `i` holds `i * 0.5 - 0.25`
fn ramp_bias(count: usize) -> Vec<f32> {
    (0..count).map(|i| i as f32 * 0.5 - 0.25).collect()
}

/// Input ramp: element `i` holds `((i % 9) - 4) * 0.5`
fn ramp_input(count: usize) -> Vec<f32> {
    (0..count).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect()
}

/// Upstream-gradient ramp: element `i` holds `((i % 5) - 2) * 0.25 + 0.125`
fn ramp_upstream(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.25 + 0.125)
        .collect()
}

/// The elements of a tensor, in row-major order
fn flat(tensor: &ndarray::ArrayD<f32>) -> Vec<f32> {
    tensor.iter().copied().collect()
}

/// Asserts that 2 flat value lists agree to 1e-5, and names the first element that does not
fn assert_flat_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "{label}[{index}]: got {got}, want {want}"
        );
    }
}

/// A Conv1D whose dilated kernel is longer than the input matches Keras under `Same`
///
/// The 3 taps span 7 cells at a dilation of 3, and the input holds 5. Keras 3.15.1 pads the
/// missing cells and returns the input length. The forward values and all 3 gradients follow.
#[test]
fn conv1d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = Conv1D::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Same)
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((3, 1, 2), ramp_kernel(6)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 1), ramp_input(5))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 2]);
    assert_flat_close(
        &flat(&output),
        &[0.125, 0.0, 0.125, 0.25, 0.0, 0.25, 1.375, 1.25, 0.875, 1.0],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 2), ramp_upstream(10))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[0.125, -0.625, -0.15625, -0.125, 0.125],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[-0.3125, -1.1875, 0.0, 0.0, 0.1875, 0.0625],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.625],
        "grad_bias",
    );
}

/// A Conv1D whose dilated kernel is longer than the input matches Keras under `Causal`
///
/// Causal padding puts all 6 pad cells on the leading edge, so output position 0 reads only
/// pad. Keras 3.15.1 accepts the same configuration and returns the same values.
#[test]
fn conv1d_dilated_kernel_longer_than_the_input_matches_keras_under_causal() {
    let mut layer = Conv1D::new(2, 3, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Causal)
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((3, 1, 2), ramp_kernel(6)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 1), ramp_input(5))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            -0.75, -0.75, -0.625, -0.5, -0.5, -0.25, 0.125, 0.0, 0.125, 0.25,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 2), ramp_upstream(10))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[-0.125, 0.125, -0.03125, 0.03125, 0.40625],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[0.0, 0.0, -0.3125, -1.1875, 0.0, 0.0],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.625],
        "grad_bias",
    );
}

/// A Conv1D whose solid kernel is longer than the input matches Keras under `Same`
///
/// This is the rule that comes before dilation. A kernel of 7 on an input of 5 needs no
/// dilation to overrun the axis. Keras 3.15.1 accepts it under `Same` in the same way.
#[test]
fn conv1d_plain_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = Conv1D::new(2, 7, 1, Linear::new())
        .unwrap()
        .with_padding(ConvPadding::Same);
    layer.build(&Shape::known(&[1, 5, 1])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((7, 1, 2), ramp_kernel(14)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 1), ramp_input(5))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            -1.25, 1.5, -1.375, 0.5, -0.625, 0.375, 1.0, 1.125, 1.0, 0.75,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 2), ramp_upstream(10))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[-0.09375, -0.5625, 0.5, -0.1875, 0.09375],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[
            -0.3125, -1.1875, -1.4375, -0.0625, -1.25, -0.625, 0.0, 0.0, 0.125, 0.0, 0.3125,
            -0.0625, 0.1875, 0.0625,
        ],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.625],
        "grad_bias",
    );
}

/// A Conv2D whose dilated kernel is longer than both input axes matches Keras under `Same`
#[test]
fn conv2d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = Conv2D::new(1, (2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 1])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 1, 1), ramp_kernel(4)).unwrap(),
            Array1::from_vec(ramp_bias(1)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 3, 3, 1), ramp_input(9))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    assert_flat_close(
        &flat(&output),
        &[-0.25, -0.5, -0.625, 0.25, 1.25, 0.875, -0.5, 0.125, -0.25],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 3, 1), ramp_upstream(9))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.46875, 0.28125, -0.1875, -0.09375, -0.28125, 0.0625, 0.03125, -0.03125, 0.0,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[-0.75, -0.4375, 0.0625, -0.75],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625],
        "grad_bias",
    );
}

/// A DepthwiseConv1D whose dilated kernel is longer than the input matches Keras under `Same`
#[test]
fn depthwise_conv1d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = DepthwiseConv1D::new(3, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 2])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((3, 2, 1), ramp_kernel(6)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 2), ramp_input(10))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 2]);
    assert_flat_close(
        &flat(&output),
        &[0.5, 1.0, 0.5, -0.75, -0.25, 0.25, 1.0, 1.0, 0.0, 0.5],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 2), ramp_upstream(10))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            0.1875, -0.0625, -0.3125, -0.3125, -0.15625, 0.0, -0.0625, -0.0625, -0.0625, 0.1875,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[-0.125, -0.5, 1.25, -1.25, -0.125, -0.9375],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.625],
        "grad_bias",
    );
}

/// A SeparableConv1D whose dilated depthwise kernel is longer than the input matches Keras
/// under `Same`
#[test]
fn separable_conv1d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = SeparableConv1D::new(2, 3, 1, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 2])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((3, 2, 1), ramp_kernel(6)).unwrap(),
            Array3::from_shape_vec((1, 2, 2), ramp_pointwise(4)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 2), ramp_input(10))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 2]);
    assert_flat_close(
        &flat(&output),
        &[-1.0, 0.25, -1.0, -0.625, -0.25, 0.25, -1.5, 0.0, -0.5, 0.25],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 2), ramp_upstream(10))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.15625, -0.03125, 0.59375, -0.15625, 0.109375, 0.0, 0.09375, -0.03125, 0.09375,
            0.09375,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(
            ctx.grads()
                .get(ParamId::new(0, "depthwise_kernel"))
                .unwrap(),
        ),
        &[0.5625, -0.25, -1.875, -0.625, -0.1875, -0.46875],
        "grad_depthwise_weights",
    );
    assert_flat_close(
        &flat(
            ctx.grads()
                .get(ParamId::new(0, "pointwise_kernel"))
                .unwrap(),
        ),
        &[-0.25, 0.5, -0.40625, -0.21875],
        "grad_pointwise_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.625],
        "grad_bias",
    );
}

/// Conv2D dilated gradients under `Valid` match Keras 3.15.1
///
/// The dilation is (2, 3) and the stride is 1, so a backward pass that drops the dilation lands
/// on the wrong input cells. The 2 rates differ from each other and from the stride, so no
/// single wrong constant reproduces this result.
#[test]
fn conv2d_dilated_gradients_match_keras_under_valid() {
    let mut layer = Conv2D::new(2, (2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((2, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 6, 1])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 1, 2), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 6, 1), ramp_input(30))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            2.0, 0.25, 2.0, -0.125, 2.0, -0.5, -1.375, -0.875, -1.375, -1.25, -1.375, -1.625,
            -1.375, 2.5, -1.375, 2.125, -1.375, 1.75,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 3, 2), ramp_upstream(18))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            0.34375, -0.28125, -0.28125, 0.09375, -0.03125, -0.15625, 0.03125, -0.59375, 0.34375,
            0.03125, -0.09375, 0.09375, -0.4375, -0.0625, 0.0, -0.21875, -0.34375, 0.78125,
            0.03125, 0.40625, -0.15625, -0.1875, -0.1875, -0.1875, 0.21875, -0.03125, 0.03125,
            -0.1875, 0.75, -0.1875,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[-0.5, 0.75, 1.375, -1.125, 1.375, -1.125, -0.125, -0.75],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.875, 0.625],
        "grad_bias",
    );
}

/// Conv2D dilated gradients under `Same` match Keras 3.15.1
///
/// `Same` padding and dilation both shift the cell a tap reads, so the backward pass must apply
/// the 2 together.
#[test]
fn conv2d_dilated_gradients_match_keras_under_same() {
    let mut layer = Conv2D::new(2, (2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((2, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 6, 1])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 1, 2), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 6, 1), ramp_input(30))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 5, 6, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            1.25, -1.25, -1.5, 2.25, -1.0, 2.125, -0.5, 2.0, -0.75, -0.75, -0.625, -0.5, 0.375,
            -0.125, 2.0, 0.25, 2.0, -0.125, 2.0, -0.5, 0.375, 1.0, 0.125, 1.0, -1.5, 1.0, -1.375,
            -0.875, -1.375, -1.25, -1.375, -1.625, 1.125, 1.0, 0.875, 1.0, 1.125, -1.25, -1.375,
            2.5, -1.375, 2.125, -1.375, 1.75, -1.5, -1.25, -1.75, -1.25, 0.0, 0.25, 1.375, 1.25,
            0.875, 1.0, 0.375, 0.75, 0.125, 0.5, -0.25, 0.25,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 5, 6, 2), ramp_upstream(60))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.28125, 0.03125, -0.625, 0.1875, -0.25, -0.09375, 0.25, -0.625, 0.03125, -0.03125,
            0.21875, -0.09375, -0.625, 0.375, -0.03125, 0.21875, 0.15625, -0.21875, 0.375, 0.125,
            0.21875, 0.15625, -0.84375, -0.34375, 0.40625, -0.15625, 0.03125, -0.21875, -0.15625,
            -0.1875,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[
            -2.4375, 0.3125, 1.1875, -1.1875, -0.6875, 0.4375, -1.875, -0.25,
        ],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[3.75, 3.75],
        "grad_bias",
    );
}

/// DepthwiseConv1D gradients at a stride of 2 and a dilation of 3 match Keras 3.15.1
///
/// The stride and the dilation differ, and the depth multiplier is 2, so a backward pass that
/// confuses the 2 rates or the output channel order fails here.
#[test]
fn depthwise_conv1d_stride_and_dilation_gradients_match_keras() {
    let mut layer = DepthwiseConv1D::new(2, 2, Linear::new())
        .unwrap()
        .with_depth_multiplier(2)
        .unwrap()
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 9, 2])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((2, 2, 2), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(4)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 9, 2), ramp_input(18))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 4]);
    assert_flat_close(
        &flat(&output),
        &[
            1.5, 1.75, 2.25, 0.125, -0.625, -0.5, -0.125, 2.0, -1.625, -0.5, 2.0, 0.5,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 4), ramp_upstream(12))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            0.34375, -0.03125, 0.0, 0.0, -0.28125, 0.03125, -0.15625, -0.1875, -0.59375, 0.09375,
            -0.03125, -0.1875, 0.0, 0.0, 0.40625, -0.1875, 0.0, 0.0,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "kernel")).unwrap()),
        &[1.5, 1.5, 0.5, -0.25, -1.125, 0.75, -0.0625, 0.3125],
        "grad_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.625, 0.125, -0.375, 0.375],
        "grad_bias",
    );
}

/// SeparableConv1D gradients at a stride of 2 and a dilation of 3 match Keras 3.15.1
#[test]
fn separable_conv1d_stride_and_dilation_gradients_match_keras() {
    let mut layer = SeparableConv1D::new(2, 2, 2, 1, Linear::new())
        .unwrap()
        .with_dilation_rate(3)
        .unwrap();
    layer.build(&Shape::known(&[1, 9, 2])).unwrap();
    layer
        .set_weights(
            Array3::from_shape_vec((2, 2, 1), ramp_kernel(4)).unwrap(),
            Array3::from_shape_vec((1, 2, 2), ramp_pointwise(4)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 9, 2), ramp_input(18))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 2]);
    assert_flat_close(
        &flat(&output),
        &[-1.5, 0.0, -0.625, -0.0625, 1.375, 1.5625],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 2), ramp_upstream(6))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.328125, 0.03125, 0.0, 0.0, 0.234375, -0.09375, -0.109375, 0.0, 0.328125, 0.09375,
            0.078125, 0.0, 0.0, 0.0, 0.109375, 0.0, 0.0, 0.0,
        ],
        "grad_input",
    );
    assert_flat_close(
        &flat(
            ctx.grads()
                .get(ParamId::new(0, "depthwise_kernel"))
                .unwrap(),
        ),
        &[-1.75, 0.5625, 0.6875, -0.46875],
        "grad_depthwise_weights",
    );
    assert_flat_close(
        &flat(
            ctx.grads()
                .get(ParamId::new(0, "pointwise_kernel"))
                .unwrap(),
        ),
        &[-1.4375, 0.59375, 0.3125, -0.5625],
        "grad_pointwise_weights",
    );
    assert_flat_close(
        &flat(ctx.grads().get(ParamId::new(0, "bias")).unwrap()),
        &[0.375, -0.125],
        "grad_bias",
    );
}

//! Integration tests for Conv1D and Conv2D forward values, shapes, error paths,
//! param counts, set_weights, and predict==forward equivalence
//!
//! Layout is Keras channels-last: Conv1D sees \[batch, length, channels\] and Conv2D sees
//! \[batch, height, width, channels\]. Kernels are \[k, Cin, F\] / \[kh, kw, Cin, F\] and every
//! bias is rank-1 of length `filters`
//!
//! Expected values come from the cross-correlation definition with known weights,
//! not from recorded implementation output. Gradient checks live in gradient_check.rs

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array3, Array4, array};
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::assert_allclose;

// Conv1D - forward with known weights

/// All-ones kernel=3, stride=1, Valid, Linear: each output is a windowed sum
#[test]
fn conv1d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv1D::new(1, 3, vec![1, 6, 1], 2, Linear::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    // weight shape [kernel=3, channels=1, filters=1]; taps in order are 2, 0, 1
    let weights = Array3::from_shape_vec((3, 1, 1), vec![2.0f32, 0.0, 1.0]).unwrap();
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv1D::new(1, 2, vec![1, 5, 1], 1, Linear::new()).unwrap();
    // taps [1, 0]: the layer copies x[i], then bias adds 5
    let weights = Array3::from_shape_vec((2, 1, 1), vec![1.0f32, 0.0]).unwrap();
    let bias = Array1::from_elem(1, 5.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 4, 1]);
    let expected = Array::from_shape_vec((1, 4, 1), vec![6.0f32, 7.0, 8.0, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Two filters with different tap patterns and biases produce independent outputs
#[test]
fn conv1d_two_filters_independent_outputs() {
    let mut layer = Conv1D::new(2, 2, vec![1, 5, 1], 1, Linear::new()).unwrap();
    // weight [kernel=2, channels=1, filters=2], flat order (k0,f0), (k0,f1), (k1,f0), (k1,f1):
    // filter 0 taps = [1, 1], filter 1 taps = [0, 1]. The 2x2 (kernel, filter) block is
    // deliberately asymmetric, so reading it with the kernel and filter axes swapped would give
    // filter 0 = [1, 0] and filter 1 = [1, 1] and change every expected value below
    let weights = Array3::from_shape_vec((2, 1, 2), vec![1.0f32, 0.0, 1.0, 1.0]).unwrap();
    let bias = array![0.0f32, 10.0];
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, ReLU::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![-1.0f32, -2.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // Pre-activation windowed sums: -2, 1, 6; ReLU clips the first to 0
    let expected = Array::from_shape_vec((1, 3, 1), vec![0.0f32, 1.0, 6.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Same padding, stride=1: output length stays at input length with symmetric zero padding
#[test]
fn conv1d_same_padding_forward_values() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv1D::new(1, 3, vec![1, 6, 1], 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 3, 1]);
    // out_len = ceil(6/2) = 3; pad_total = (3-1)*2 + 3 - 6 = 1, all on the trailing edge:
    // [1,2,3,4,5,6,0]. Windows start at 0, 2, 4: 1+2+3, 3+4+5, 5+6+0
    let expected = Array::from_shape_vec((1, 3, 1), vec![6.0f32, 12.0, 11.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Batch size > 1: identical samples are processed independently and yield identical outputs
#[test]
fn conv1d_batch_forward_independent_samples() {
    let mut layer = Conv1D::new(1, 2, vec![2, 4, 1], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // Two identical samples [1,2,3,4]; each yields windowed sums [3,5,7]
    let input = Array::from_shape_vec((2, 4, 1), vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3, 1]);
    // Both samples produce the same output
    for b in 0..2 {
        assert_abs_diff_eq!(output[[b, 0, 0]], 3.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 1, 0]], 5.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 2, 0]], 7.0f32, epsilon = 1e-6f32);
    }
}

/// input_length == kernel_size produces exactly one output element
#[test]
fn conv1d_input_equals_kernel_produces_single_output() {
    let mut layer = Conv1D::new(1, 3, vec![1, 3, 1], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 3, 1), vec![10.0f32, 20.0, 30.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 60.0f32, epsilon = 1e-6f32);
}

/// Two input channels: the filter sums contributions across both channels
#[test]
fn conv1d_two_input_channels_cross_channel_sum() {
    let mut layer = Conv1D::new(1, 2, vec![1, 3, 2], 1, Linear::new()).unwrap();
    // weight [kernel=2, channels=2, filters=1]: channel 0 taps = [1,1], channel 1 taps = [2,2].
    // Laid out kernel-major, the flat order is (k0,c0), (k0,c1), (k1,c0), (k1,c1)
    let weights = Array3::from_shape_vec((2, 2, 1), vec![1.0f32, 2.0, 1.0, 2.0]).unwrap();
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // input [batch=1, length=3, channels=2]: ch0=[1,2,3], ch1=[10,20,30], interleaved per
    // position. The two channels differ, so a swapped channel axis would change the result
    let input = Array::from_shape_vec((1, 3, 2), vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv1D::new(4, 2, vec![1, 8, 3], 1, Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(28));
}

/// param_count for filters=2, channels=1, kernel=3 is 2*1*3 + 2 = 8
#[test]
fn conv1d_param_count_single_channel() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv1D::new(2, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(8));
}

// Conv1D - set_weights

/// set_weights with the correct shape succeeds and the forward output reflects it
#[test]
fn conv1d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();

    // All-twos weights: output is 2x the windowed sum
    let weights = Array3::from_elem((3, 1, 1), 2.0f32);
    let bias = Array1::zeros(1);
    assert!(layer.set_weights(weights, bias).is_ok());

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    let expected = Array::from_shape_vec((1, 3, 1), vec![12.0f32, 18.0, 24.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// set_weights with a mismatched weight shape returns a WeightShape error
#[test]
fn conv1d_set_weights_mismatched_weight_shape_errors() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();

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
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();

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
        let result = Conv1D::new(filters, kernel_size, vec![1, 5, 1], stride, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label}, got {:?}",
            result
        );
    }
}

/// Each invalid input_shape (wrong ndim, zero channels, or length < kernel)
/// independently makes the constructor return InvalidInput
#[test]
fn conv1d_invalid_input_shape_errors() {
    // (input_shape, label)
    let cases = [
        (vec![1, 5], "2D input_shape"),           // input_shape must be 3D
        (vec![1, 5, 5, 1], "4D input_shape"),     // input_shape must be 3D
        (vec![1, 5, 0], "channels=0"),            // channels is the last axis
        (vec![1, 2, 1], "input length < kernel"), // length=2 < kernel=3
    ];
    for (input_shape, label) in cases {
        let result = Conv1D::new(1, 3, input_shape, 1, Linear::new());
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
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    let bad_input = Array::zeros((1_usize, 5_usize)).into_dyn();
    let result = layer.forward(&bad_input);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D tensor, got {:?}",
        result
    );
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn conv1d_backward_before_forward_errors() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3_usize, 1_usize)).into_dyn();
    let result = layer.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun error, got {:?}",
        result
    );
}

// Conv1D - predict == forward in eval mode (no randomness; layer is stateless)

/// predict() returns the same values as forward() for a deterministic layer
#[test]
fn conv1d_predict_equals_forward() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input).unwrap();
    let predict_output = layer.predict(&input).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv1d_predict_deterministic() {
    let mut layer = Conv1D::new(1, 3, vec![1, 5, 1], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 5, 1), vec![2.0f32, 4.0, 6.0, 8.0, 10.0])
        .unwrap()
        .into_dyn();

    let out1 = layer.predict(&input).unwrap();
    let out2 = layer.predict(&input).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv1D - get_weights shape

/// get_weights returns Conv1D weights with shape [kernel, channels, filters] and bias [filters]
#[test]
fn conv1d_get_weights_correct_shapes() {
    let layer = Conv1D::new(3, 5, vec![1, 10, 2], 1, Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv1D(w) => {
            assert_eq!(w.weight.shape(), &[5, 2, 3]);
            assert_eq!(w.bias.shape(), &[3]);
        }
        _other => panic!("expected LayerWeight::Conv1D variant"),
    }
}

// Conv2D - forward with known weights

/// All-ones 2x2 kernel, 1 channel, Valid, stride=(1,1): each output is a 2x2 windowed sum
#[test]
fn conv2d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    // Rows of the 4x4 plane: [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]
    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 2.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 2, 2, 1], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 1.0f32);
    let bias = Array1::from_elem(1, 3.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 2, 2, 1), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    let expected = Array::from_shape_vec((1, 2, 2, 1), vec![4.0f32, 5.0, 6.0, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// stride=(2,2) with Valid padding: output shape is floor((H-k)/s)+1 with strided windows
#[test]
fn conv2d_stride2_valid_output_shape_and_values() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (2, 2), Linear::new()).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    // Windows anchored at rows/cols {0, 2}
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 14.0f32, epsilon = 1e-5f32); // 1+2+5+6
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 22.0f32, epsilon = 1e-5f32); // 3+4+7+8
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 46.0f32, epsilon = 1e-5f32); // 9+10+13+14
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 54.0f32, epsilon = 1e-5f32); // 11+12+15+16
}

/// Two filters with independent weight patterns: all-ones kernel vs top-left-only kernel
#[test]
fn conv2d_two_filters_independent_outputs() {
    let mut layer = Conv2D::new(2, (2, 2), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();
    // weight [kh=2, kw=2, channels=1, filters=2]: filter0 = [[1,1],[1,1]], filter1 = [[1,0],[0,0]].
    // Flat order is (kh,kw)-major with the filter axis last, so each pair is (f0, f1) at one tap
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
    let output = layer.forward(&input).unwrap();

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

/// Two input channels: the filter sums contributions across the (last) channel axis
#[test]
fn conv2d_two_input_channels_cross_channel_sum() {
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 2, 2, 2], (1, 1), Linear::new()).unwrap();
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
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 1]);
    // out[h,w] = 1*x[h,w,0] + 10*x[h,w,1]; the per-channel weights and the per-channel data both
    // differ, so reading the channel axis anywhere but last changes every value
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 21.0f32, epsilon = 1e-5f32); // 1 + 10*2
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 43.0f32, epsilon = 1e-5f32); // 3 + 10*4
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 65.0f32, epsilon = 1e-5f32); // 5 + 10*6
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 87.0f32, epsilon = 1e-5f32); // 7 + 10*8
}

/// ReLU clips negative pre-activations to zero
#[test]
fn conv2d_relu_clips_negatives() {
    let mut layer = Conv2D::new(1, (2, 1), vec![1, 3, 1, 1], (1, 1), ReLU::new()).unwrap();
    let weights = Array4::from_elem((2, 1, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 3, 1, 1), vec![-3.0f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 1, 1]);
    // Pre-activations: -3+1 = -2 (clipped) and 1+2 = 3
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 0.0f32, epsilon = 1e-6f32);
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 3.0f32, epsilon = 1e-6f32);
}

/// Same padding output shape is ceil(H/stride) x ceil(W/stride)
#[test]
fn conv2d_same_padding_output_shapes() {
    // Case 1: stride=(1,1)
    let mut layer1 = Conv2D::new(1, (3, 3), vec![1, 5, 5, 1], (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x1 = Array::ones((1, 5, 5, 1)).into_dyn();
    let out1 = layer1.forward(&x1).unwrap();
    assert_eq!(out1.shape(), &[1, 5, 5, 1]);

    // Case 2: stride=(2,2): ceil(4/2)=2
    let mut layer2 = Conv2D::new(1, (3, 3), vec![1, 4, 4, 1], (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x2 = Array::ones((1, 4, 4, 1)).into_dyn();
    let out2 = layer2.forward(&x2).unwrap();
    assert_eq!(out2.shape(), &[1, 2, 2, 1]);
}

/// Same padding with all-ones kernel and input: corners sum 4, edges 6, center 9 ones
#[test]
fn conv2d_same_padding_all_ones_values() {
    let mut layer = Conv2D::new(1, (3, 3), vec![1, 3, 3, 1], (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array4::from_elem((3, 3, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((1, 3, 3, 1)).into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    // pad_total = (3-1)*1 + 3 - 3 = 2 per axis, one row/column of zeros on each side
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
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv2D::new(2, (3, 3), vec![1, 5, 5, 3], (1, 1), Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(56));
}

/// param_count for filters=1, channels=1, kernel=(2,2) is 1*1*2*2 + 1 = 5
#[test]
fn conv2d_param_count_single_filter_single_channel() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(5));
}

// Conv2D - set_weights

/// set_weights with the correct shape succeeds and new weights change the output
#[test]
fn conv2d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();
    // weight=3.0 with a 1x1 kernel: output is 3 * input
    let weights = Array4::from_elem((1, 1, 1, 1), 3.0f32);
    let bias = Array1::zeros(1);
    assert!(layer.set_weights(weights, bias).is_ok());

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
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
        let result = Conv2D::new(filters, kernel, vec![1, 5, 5, 1], stride, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label}, got {:?}",
            result
        );
    }
}

/// Each invalid input_shape (wrong ndim, or a spatial dim < kernel)
/// independently makes the constructor return InvalidInput
#[test]
fn conv2d_invalid_input_shape_errors() {
    // (input_shape, label)
    let cases = [
        (vec![1, 5, 1], "3D input_shape"),       // input_shape must be 4D
        (vec![1, 5, 5, 5, 1], "5D input_shape"), // input_shape must be 4D
        (vec![1, 2, 5, 1], "height < kernel"),   // height=2 < kernel_h=3
        (vec![1, 5, 2, 1], "width < kernel"),    // width=2 < kernel_w=3
    ];
    for (input_shape, label) in cases {
        let result = Conv2D::new(1, (3, 3), input_shape, (1, 1), Linear::new());
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
    let mut layer = Conv2D::new(1, (3, 3), vec![1, 5, 5, 1], (1, 1), Linear::new()).unwrap();
    let bad_input = Array::zeros((1_usize, 5_usize, 5_usize)).into_dyn();
    let result = layer.forward(&bad_input);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D tensor, got {:?}",
        result
    );
}

/// backward() before forward() returns ForwardPassNotRun
#[test]
fn conv2d_backward_before_forward_errors() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3_usize, 3_usize, 1_usize)).into_dyn();
    let result = layer.backward(&grad);
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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input).unwrap();
    let predict_output = layer.predict(&input).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv2d_predict_deterministic() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((2, 2, 1, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 1), input_data)
        .unwrap()
        .into_dyn();

    let out1 = layer.predict(&input).unwrap();
    let out2 = layer.predict(&input).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv2D - get_weights shape

/// get_weights returns Conv2D weights with shape [kh, kw, channels, filters] and bias [filters]
#[test]
fn conv2d_get_weights_correct_shapes() {
    let layer = Conv2D::new(4, (3, 3), vec![1, 8, 8, 2], (1, 1), Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv2D(w) => {
            assert_eq!(w.weight.shape(), &[3, 3, 2, 4]);
            assert_eq!(w.bias.shape(), &[4]);
        }
        _other => panic!("expected LayerWeight::Conv2D variant"),
    }
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
        let mut conv = Conv1D::new(1, kernel, vec![1, len, 1], stride, Linear::new()).unwrap();
        let x = Array::ones((1, len, 1)).into_dyn();
        let out = conv.forward(&x).unwrap();
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
        let mut conv = Conv1D::new(1, kernel, vec![1, len, 1], stride, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same);
        let x = Array::ones((1, len, 1)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, expected, 1],
            "Conv1D Same: input_len={len}, kernel={kernel}, stride={stride}"
        );
    }
}

// Conv2D - Valid output shape for diverse (H, W, kh, kw, sh, sw) cases

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
        let mut conv = Conv2D::new(1, (kh, kw), vec![1, h, w, 1], (sh, sw), Linear::new()).unwrap();
        let x = Array::ones((1, h, w, 1)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, out_h, out_w, 1],
            "Conv2D Valid: H={h}, W={w}, k=({kh},{kw}), s=({sh},{sw})"
        );
    }
}
// Conv2D - convolution_engine parallel forward / weight-grad backward branches

// The engine gates on estimated GEMM FLOPs (`CONV_PARALLEL_MIN_FLOPS`, default 4_000_000):
// `2 * batch * filters * out_plane * Cin*k` on the forward pass and `4 * ...` on the backward
// pass. These two tests size their tensors to just clear that gate and pin results to the
// cross-correlation definition. Each asserts the crossing against the live gate value first, so a
// retuned threshold fails loudly instead of silently demoting the test to the serial path

/// Parallel forward branch: windowed sums on a tensor whose estimated FLOPs clear the gate
#[test]
fn conv2d_parallel_forward_windowed_sums() {
    // 2 * batch(2) * filters(2) * out_plane(354*354) * Cin*k(1*2*2) = 4_010_112
    let gemm_flops = 2 * 2 * 2 * (354 * 354) * 4;
    assert!(
        gemm_flops >= rustyml::tuning::conv::get_parallel_min_flops(),
        "fixture no longer clears the engine's parallel gate: {gemm_flops} FLOPs"
    );

    let mut layer = Conv2D::new(2, (2, 2), vec![2, 355, 355, 1], (1, 1), Linear::new()).unwrap();
    // Both filters use an all-ones 2x2 kernel; bias filter0=0, filter1=100
    let weights = Array4::from_elem((2, 2, 1, 2), 1.0f32);
    let bias = array![0.0f32, 100.0];
    layer.set_weights(weights, bias).unwrap();

    // x[b,i,j,0] = i + j, identical across the two batch samples
    let mut input = Array4::<f32>::zeros((2, 355, 355, 1));
    for b in 0..2 {
        for i in 0..355 {
            for j in 0..355 {
                input[[b, i, j, 0]] = (i + j) as f32;
            }
        }
    }
    let output = layer.forward(&input.into_dyn()).unwrap();

    assert_eq!(output.shape(), &[2, 354, 354, 2]);

    // The 2x2 window at (oh, ow) sums (oh+ow) + (oh+ow+1) + (oh+ow+1) + (oh+ow+2) = 4*oh + 4*ow + 4
    // out[b,oh,ow,1] = that + 100, checked over both batches
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
    // 4 * batch(4) * filters(1) * out_plane(118*118) * Cin*k(2*3*3) = 4_010_112
    let gemm_flops = 4 * 4 * (118 * 118) * 18;
    assert!(
        gemm_flops >= rustyml::tuning::conv::get_parallel_min_flops(),
        "fixture no longer clears the engine's parallel gate: {gemm_flops} FLOPs"
    );

    let mut layer = Conv2D::new(1, (3, 3), vec![4, 120, 120, 2], (1, 1), Linear::new()).unwrap();
    // Weights do not affect the weight gradient; all-ones, bias 0, for determinism
    let weights = Array4::from_elem((3, 3, 2, 1), 1.0f32);
    let bias = Array1::zeros(1);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((4, 120, 120, 2)).into_dyn();
    let _ = layer.forward(&input).unwrap();

    // grad w.r.t. the conv output: all ones, shape [batch, out_h, out_w, filters]
    let grad = Array::ones((4, 118, 118, 1)).into_dyn();
    let _input_grad = layer.backward(&grad).unwrap();

    // parameters() returns weights first; its `.grad` is the flat weight gradient
    let params = layer.parameters();
    let weight_grad = params[0].grad;
    // 3*3 taps * 2 input channels * 1 filter
    assert_eq!(weight_grad.len(), 18, "expected 18 weight-grad entries");
    // dL/dw[kh,kw,c,f] = sum_{b,oh,ow} grad * x = 4 batches * 118*118 positions * 1 * 1 = 55_696
    for &g in weight_grad.iter() {
        assert_abs_diff_eq!(g, 55_696.0f32, epsilon = 1e-2f32);
    }
}

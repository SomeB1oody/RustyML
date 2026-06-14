//! Integration tests for Conv1D and Conv2D forward values, shapes, error paths,
//! param counts, set_weights, and predict==forward equivalence
//!
//! Expected values come from the cross-correlation definition with known weights,
//! not from recorded implementation output. Gradient checks live in gradient_check.rs

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, Array3, Array4};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::traits::Layer;

use super::common::assert_allclose;

// Conv1D - forward with known weights

/// All-ones kernel=3, stride=1, Valid, Linear: each output is a windowed sum
#[test]
fn conv1d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3]);
    let expected = Array::from_shape_vec((1, 1, 3), vec![6.0f32, 9.0, 12.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// All-ones kernel, stride=2, Valid: selects every other window
#[test]
fn conv1d_stride2_windowed_sums() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 6], 2, Linear::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 6), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2]);
    let expected = Array::from_shape_vec((1, 1, 2), vec![6.0f32, 12.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Asymmetric kernel weight=[2, 0, 1], bias=0: output respects tap positions
#[test]
fn conv1d_asymmetric_kernel_values() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    // weight shape [filters=1, channels=1, kernel=3]
    let weights = Array3::from_shape_vec((1, 1, 3), vec![2.0f32, 0.0, 1.0]).unwrap();
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3]);
    let expected = Array::from_shape_vec((1, 1, 3), vec![5.0f32, 8.0, 11.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Bias adds a constant to every output element
#[test]
fn conv1d_bias_offset_adds_to_every_output() {
    let mut layer = Conv1D::new(1, 2, vec![1, 1, 5], 1, Linear::new()).unwrap();
    let weights = Array3::from_shape_vec((1, 1, 2), vec![1.0f32, 0.0]).unwrap();
    let bias = Array2::from_elem((1, 1), 5.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 4]);
    let expected = Array::from_shape_vec((1, 1, 4), vec![6.0f32, 7.0, 8.0, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Two filters share an all-ones kernel but differ in bias, producing independent outputs
#[test]
fn conv1d_two_filters_independent_outputs() {
    let mut layer = Conv1D::new(2, 2, vec![1, 1, 5], 1, Linear::new()).unwrap();
    // weight [2, 1, 2]: filter0=[1,1], filter1=[1,1]; bias [1, 2]: [0, 10]
    let weights = Array3::from_shape_vec((2, 1, 2), vec![1.0f32, 1.0, 1.0, 1.0]).unwrap();
    let bias = Array2::from_shape_vec((1, 2), vec![0.0f32, 10.0]).unwrap();
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 4]);

    // Filter 0: windowed sums
    for (i, expected_val) in [3.0f32, 5.0, 7.0, 9.0].iter().enumerate() {
        assert_abs_diff_eq!(output[[0, 0, i]], *expected_val, epsilon = 1e-6f32);
    }
    // Filter 1: windowed sums + 10
    for (i, expected_val) in [13.0f32, 15.0, 17.0, 19.0].iter().enumerate() {
        assert_abs_diff_eq!(output[[0, 1, i]], *expected_val, epsilon = 1e-6f32);
    }
}

/// ReLU activation clips negative pre-activations to zero
#[test]
fn conv1d_relu_activation_clips_negatives() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, ReLU::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![-1.0f32, -2.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3]);
    let expected = Array::from_shape_vec((1, 1, 3), vec![0.0f32, 1.0, 6.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Same padding, stride=1: output length stays at input length with symmetric zero padding
#[test]
fn conv1d_same_padding_forward_values() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 5]);
    let expected = Array::from_shape_vec((1, 1, 5), vec![3.0f32, 6.0, 9.0, 12.0, 9.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Same padding, stride=2: output length is ceil(input_len / stride) with trailing-only pad
#[test]
fn conv1d_same_padding_stride2_output_length_and_values() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 6], 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 6), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3]);
    let expected = Array::from_shape_vec((1, 1, 3), vec![6.0f32, 12.0, 11.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Batch size > 1: identical samples are processed independently and yield identical outputs
#[test]
fn conv1d_batch_forward_independent_samples() {
    let mut layer = Conv1D::new(1, 2, vec![2, 1, 4], 1, Linear::new()).unwrap();
    let weights = Array3::from_shape_vec((1, 1, 2), vec![1.0f32, 1.0]).unwrap();
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    // Two identical samples [1,2,3,4]; each yields windowed sums [3,5,7]
    let input = Array::from_shape_vec((2, 1, 4), vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1, 3]);
    // Both samples produce the same output
    for b in 0..2 {
        assert_abs_diff_eq!(output[[b, 0, 0]], 3.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 0, 1]], 5.0f32, epsilon = 1e-6f32);
        assert_abs_diff_eq!(output[[b, 0, 2]], 7.0f32, epsilon = 1e-6f32);
    }
}

/// input_length == kernel_size produces exactly one output element
#[test]
fn conv1d_input_equals_kernel_produces_single_output() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 3], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 3), vec![10.0f32, 20.0, 30.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 60.0f32, epsilon = 1e-6f32);
}

/// Two input channels: the filter sums contributions across both channels
#[test]
fn conv1d_two_input_channels_cross_channel_sum() {
    let mut layer = Conv1D::new(1, 2, vec![1, 2, 3], 1, Linear::new()).unwrap();
    // weight [1, 2, 2]: filter 0 channel 0 = [1,1], channel 1 = [2,2]
    let weights = Array3::from_shape_vec((1, 2, 2), vec![1.0f32, 1.0, 2.0, 2.0]).unwrap();
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    // input [1, 2, 3]: ch0=[1,2,3], ch1=[1,2,3]
    let input = Array::from_shape_vec((1, 2, 3), vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 9.0f32, epsilon = 1e-6f32);
    assert_abs_diff_eq!(output[[0, 0, 1]], 15.0f32, epsilon = 1e-6f32);
}

// Conv1D - param_count

/// param_count = filters * channels * kernel_size + filters
#[test]
fn conv1d_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv1D::new(4, 2, vec![1, 3, 8], 1, Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(28));
}

/// param_count for filters=2, channels=1, kernel=3 is 2*1*3 + 2 = 8
#[test]
fn conv1d_param_count_single_channel() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv1D::new(2, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(8));
}

// Conv1D - set_weights

/// set_weights with the correct shape succeeds and the forward output reflects it
#[test]
fn conv1d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();

    // All-twos weights: output is 2x the windowed sum
    let weights = Array3::from_elem((1, 1, 3), 2.0f32);
    let bias = Array2::zeros((1, 1));
    assert!(layer.set_weights(weights, bias).is_ok());

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    let expected = Array::from_shape_vec((1, 1, 3), vec![12.0f32, 18.0, 24.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// set_weights with a mismatched weight shape returns a WeightShape error
#[test]
fn conv1d_set_weights_mismatched_weight_shape_errors() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();

    // Wrong weight shape [2, 1, 3] instead of [1, 1, 3]
    let wrong_weights = Array3::zeros((2, 1, 3));
    let bias = Array2::zeros((1, 1));
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
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();

    let weights = Array3::zeros((1, 1, 3));
    // Wrong bias [1, 2] instead of [1, 1]
    let wrong_bias = Array2::zeros((1, 2));
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

#[test]
fn conv1d_filters_zero_errors() {
    let result = Conv1D::new(0, 3, vec![1, 1, 5], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for filters=0, got {:?}",
        result
    );
}

#[test]
fn conv1d_kernel_size_zero_errors() {
    let result = Conv1D::new(1, 0, vec![1, 1, 5], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for kernel_size=0, got {:?}",
        result
    );
}

#[test]
fn conv1d_stride_zero_errors() {
    let result = Conv1D::new(1, 3, vec![1, 1, 5], 0, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride=0, got {:?}",
        result
    );
}

/// input_shape must be 3D; a 2D shape returns InvalidInput
#[test]
fn conv1d_wrong_input_ndim_2d_errors() {
    let result = Conv1D::new(1, 3, vec![1, 5], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input_shape, got {:?}",
        result
    );
}

/// input_shape must be 3D; a 4D shape returns InvalidInput
#[test]
fn conv1d_wrong_input_ndim_4d_errors() {
    let result = Conv1D::new(1, 3, vec![1, 1, 5, 5], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 4D input_shape, got {:?}",
        result
    );
}

/// input_shape with channels=0 returns InvalidInput
#[test]
fn conv1d_zero_input_channels_errors() {
    let result = Conv1D::new(1, 3, vec![1, 0, 5], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for channels=0, got {:?}",
        result
    );
}

/// input length < kernel_size returns InvalidInput
#[test]
fn conv1d_input_smaller_than_kernel_errors() {
    // length=2 < kernel=3
    let result = Conv1D::new(1, 3, vec![1, 1, 2], 1, Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput when input length < kernel, got {:?}",
        result
    );
}

// Conv1D - forward error paths

/// Passing a 2D tensor to forward() returns InvalidInput
#[test]
fn conv1d_forward_wrong_ndim_errors() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
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
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 1_usize, 3_usize)).into_dyn();
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
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input).unwrap();
    let predict_output = layer.predict(&input).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv1d_predict_deterministic() {
    let mut layer = Conv1D::new(1, 3, vec![1, 1, 5], 1, Linear::new()).unwrap();
    let weights = Array3::from_elem((1, 1, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 5), vec![2.0f32, 4.0, 6.0, 8.0, 10.0])
        .unwrap()
        .into_dyn();

    let out1 = layer.predict(&input).unwrap();
    let out2 = layer.predict(&input).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv1D - get_weights shape

/// get_weights returns Conv1D weights with shape [filters, channels, kernel] and bias [1, filters]
#[test]
fn conv1d_get_weights_correct_shapes() {
    let layer = Conv1D::new(3, 5, vec![1, 2, 10], 1, Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv1D(w) => {
            assert_eq!(w.weight.shape(), &[3, 2, 5]);
            assert_eq!(w.bias.shape(), &[1, 3]);
        }
        _other => panic!("expected LayerWeight::Conv1D variant"),
    }
}

// Conv2D - forward with known weights

/// All-ones 2x2 kernel, 1 channel, Valid, stride=(1,1): each output is a 2x2 windowed sum
#[test]
fn conv2d_all_ones_kernel_windowed_sums() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 2, 2), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 4, 4), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3, 3]);

    let expected_flat = vec![14.0f32, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
    for (i, &expected_val) in expected_flat.iter().enumerate() {
        let h = i / 3;
        let w = i % 3;
        assert_abs_diff_eq!(output[[0, 0, h, w]], expected_val, epsilon = 1e-5f32);
    }
}

/// 1x1 kernel acts as a per-element scalar multiply
#[test]
fn conv2d_1x1_kernel_scalar_multiply() {
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 1, 3, 3], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 2.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 3, 3), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = input[[0, 0, i, j]] * 2.0;
            assert_abs_diff_eq!(output[[0, 0, i, j]], expected, epsilon = 1e-6f32);
        }
    }
}

/// Bias adds a constant to every output element
#[test]
fn conv2d_bias_shifts_all_outputs() {
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 1, 2, 2], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 1, 1), 1.0f32);
    let bias = Array2::from_elem((1, 1), 3.0f32);
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    let expected = Array::from_shape_vec((1, 1, 2, 2), vec![4.0f32, 5.0, 6.0, 7.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// stride=(2,2) with Valid padding: output shape is floor((H-k)/s)+1 with strided windows
#[test]
fn conv2d_stride2_valid_output_shape_and_values() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (2, 2), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 2, 2), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 4, 4), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 14.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 22.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 46.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 54.0f32, epsilon = 1e-5f32);
}

/// Two filters with independent weight patterns: all-ones kernel vs top-left-only kernel
#[test]
fn conv2d_two_filters_independent_outputs() {
    let mut layer = Conv2D::new(2, (2, 2), vec![1, 1, 3, 3], (1, 1), Linear::new()).unwrap();
    // weight [2, 1, 2, 2]: filter0 = [[1,1],[1,1]], filter1 = [[1,0],[0,0]]
    let weights = Array4::from_shape_vec(
        (2, 1, 2, 2),
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let bias = Array2::zeros((1, 2));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 3, 3), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    // Filter 0
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 12.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 16.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 24.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 28.0f32, epsilon = 1e-5f32);

    // Filter 1: top-left position only
    assert_abs_diff_eq!(output[[0, 1, 0, 0]], 1.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 0, 1]], 2.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 1, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 1, 1, 1]], 5.0f32, epsilon = 1e-5f32);
}

/// ReLU clips negative pre-activations to zero
#[test]
fn conv2d_relu_clips_negatives() {
    let mut layer = Conv2D::new(1, (2, 1), vec![1, 1, 3, 1], (1, 1), ReLU::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 2, 1), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::from_shape_vec((1, 1, 3, 1), vec![-3.0f32, 1.0, 2.0])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 0.0f32, epsilon = 1e-6f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 3.0f32, epsilon = 1e-6f32);
}

/// Same padding output shape is ceil(H/stride) x ceil(W/stride)
#[test]
fn conv2d_same_padding_output_shapes() {
    // Case 1: stride=(1,1)
    let mut layer1 = Conv2D::new(1, (3, 3), vec![1, 1, 5, 5], (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x1 = Array::ones((1, 1, 5, 5)).into_dyn();
    let out1 = layer1.forward(&x1).unwrap();
    assert_eq!(out1.shape(), &[1, 1, 5, 5]);

    // Case 2: stride=(2,2): ceil(4/2)=2
    let mut layer2 = Conv2D::new(1, (3, 3), vec![1, 1, 4, 4], (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x2 = Array::ones((1, 1, 4, 4)).into_dyn();
    let out2 = layer2.forward(&x2).unwrap();
    assert_eq!(out2.shape(), &[1, 1, 2, 2]);
}

/// Same padding with all-ones kernel and input: corners sum 4, edges 6, center 9 ones
#[test]
fn conv2d_same_padding_all_ones_values() {
    let mut layer = Conv2D::new(1, (3, 3), vec![1, 1, 3, 3], (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array4::from_elem((1, 1, 3, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((1, 1, 3, 3)).into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    // Corners: 4 overlapping ones
    assert_abs_diff_eq!(output[[0, 0, 0, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 0, 2]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 2, 0]], 4.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 2, 2]], 4.0f32, epsilon = 1e-5f32);
    // Edge midpoints: 6 overlapping ones
    assert_abs_diff_eq!(output[[0, 0, 0, 1]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 0]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 1, 2]], 6.0f32, epsilon = 1e-5f32);
    assert_abs_diff_eq!(output[[0, 0, 2, 1]], 6.0f32, epsilon = 1e-5f32);
    // Center: 9 overlapping ones
    assert_abs_diff_eq!(output[[0, 0, 1, 1]], 9.0f32, epsilon = 1e-5f32);
}

// Conv2D - param_count

/// param_count = filters * channels * kh * kw + filters
#[test]
fn conv2d_param_count_formula() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv2D::new(2, (3, 3), vec![1, 3, 5, 5], (1, 1), Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(56));
}

/// param_count for filters=1, channels=1, kernel=(2,2) is 1*1*2*2 + 1 = 5
#[test]
fn conv2d_param_count_single_filter_single_channel() {
    use rustyml::neural_network::layers::TrainingParameters;
    let layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(5));
}

// Conv2D - set_weights

/// set_weights with the correct shape succeeds and new weights change the output
#[test]
fn conv2d_set_weights_correct_shape_succeeds() {
    let mut layer = Conv2D::new(1, (1, 1), vec![1, 1, 3, 3], (1, 1), Linear::new()).unwrap();
    // weight=3.0 with a 1x1 kernel: output is 3 * input
    let weights = Array4::from_elem((1, 1, 1, 1), 3.0f32);
    let bias = Array2::zeros((1, 1));
    assert!(layer.set_weights(weights, bias).is_ok());

    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 3, 3), input_data)
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(
                output[[0, 0, i, j]],
                input[[0, 0, i, j]] * 3.0,
                epsilon = 1e-6f32
            );
        }
    }
}

/// set_weights with a mismatched weight shape returns a WeightShape error
#[test]
fn conv2d_set_weights_mismatched_weight_shape_errors() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    // Wrong weight shape [2,1,2,2] instead of [1,1,2,2]
    let wrong_weights = Array4::zeros((2, 1, 2, 2));
    let bias = Array2::zeros((1, 1));
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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    let weights = Array4::zeros((1, 1, 2, 2));
    // Wrong bias [1,3] instead of [1,1]
    let wrong_bias = Array2::zeros((1, 3));
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

#[test]
fn conv2d_filters_zero_errors() {
    let result = Conv2D::new(0, (3, 3), vec![1, 1, 5, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for filters=0, got {:?}",
        result
    );
}

#[test]
fn conv2d_kernel_height_zero_errors() {
    let result = Conv2D::new(1, (0, 3), vec![1, 1, 5, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for kernel height=0, got {:?}",
        result
    );
}

#[test]
fn conv2d_kernel_width_zero_errors() {
    let result = Conv2D::new(1, (3, 0), vec![1, 1, 5, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for kernel width=0, got {:?}",
        result
    );
}

#[test]
fn conv2d_stride_height_zero_errors() {
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 5, 5], (0, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride height=0, got {:?}",
        result
    );
}

#[test]
fn conv2d_stride_width_zero_errors() {
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 5, 5], (1, 0), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride width=0, got {:?}",
        result
    );
}

/// input_shape must be 4D; a 3D shape returns InvalidInput
#[test]
fn conv2d_wrong_input_ndim_3d_errors() {
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 3D input_shape, got {:?}",
        result
    );
}

/// input_shape must be 4D; a 5D shape returns InvalidInput
#[test]
fn conv2d_wrong_input_ndim_5d_errors() {
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 5, 5, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 5D input_shape, got {:?}",
        result
    );
}

/// input height < kernel height returns InvalidInput
#[test]
fn conv2d_input_height_smaller_than_kernel_errors() {
    // height=2 < kernel_h=3
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 2, 5], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput when height < kernel, got {:?}",
        result
    );
}

/// input width < kernel width returns InvalidInput
#[test]
fn conv2d_input_width_smaller_than_kernel_errors() {
    // width=2 < kernel_w=3
    let result = Conv2D::new(1, (3, 3), vec![1, 1, 5, 2], (1, 1), Linear::new());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput when width < kernel, got {:?}",
        result
    );
}

// Conv2D - forward error paths

/// Passing a 3D tensor to forward() returns InvalidInput (4D expected)
#[test]
fn conv2d_forward_wrong_ndim_errors() {
    let mut layer = Conv2D::new(1, (3, 3), vec![1, 1, 5, 5], (1, 1), Linear::new()).unwrap();
    let bad_input = Array::zeros((1_usize, 1_usize, 5_usize)).into_dyn();
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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 1_usize, 3_usize, 3_usize)).into_dyn();
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
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 2, 2), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 4, 4), input_data)
        .unwrap()
        .into_dyn();

    let forward_output = layer.forward(&input).unwrap();
    let predict_output = layer.predict(&input).unwrap();

    assert_allclose(&predict_output, &forward_output, 1e-7f32);
}

/// predict() returns the same result across repeated calls
#[test]
fn conv2d_predict_deterministic() {
    let mut layer = Conv2D::new(1, (2, 2), vec![1, 1, 4, 4], (1, 1), Linear::new()).unwrap();
    let weights = Array4::from_elem((1, 1, 2, 2), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Array::from_shape_vec((1, 1, 4, 4), input_data)
        .unwrap()
        .into_dyn();

    let out1 = layer.predict(&input).unwrap();
    let out2 = layer.predict(&input).unwrap();
    assert_allclose(&out1, &out2, 0.0f32);
}

// Conv2D - get_weights shape

/// get_weights returns Conv2D weights with shape [filters, channels, kh, kw] and bias [1, filters]
#[test]
fn conv2d_get_weights_correct_shapes() {
    let layer = Conv2D::new(4, (3, 3), vec![1, 2, 8, 8], (1, 1), Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv2D(w) => {
            assert_eq!(w.weight.shape(), &[4, 2, 3, 3]);
            assert_eq!(w.bias.shape(), &[1, 4]);
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
        let mut conv = Conv1D::new(1, kernel, vec![1, 1, len], stride, Linear::new()).unwrap();
        let x = Array::ones((1, 1, len)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, 1, expected],
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
        let mut conv = Conv1D::new(1, kernel, vec![1, 1, len], stride, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same);
        let x = Array::ones((1, 1, len)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, 1, expected],
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
        let mut conv = Conv2D::new(1, (kh, kw), vec![1, 1, h, w], (sh, sw), Linear::new()).unwrap();
        let x = Array::ones((1, 1, h, w)).into_dyn();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, 1, out_h, out_w],
            "Conv2D Valid: H={h}, W={w}, k=({kh},{kw}), s=({sh},{sw})"
        );
    }
}
// Conv2D - convolution_engine parallel forward / weight-grad backward branches

// The engine switches to rayon at CONV_PARALLEL_THRESHOLD (10_000); these two tests cross
// it with the smallest qualifying tensors and pin results to the cross-correlation definition

/// Parallel forward branch (workload batch * filters * out_plane >= 10_000): windowed sums
#[test]
fn conv2d_parallel_forward_windowed_sums() {
    let mut layer = Conv2D::new(2, (2, 2), vec![2, 1, 51, 51], (1, 1), Linear::new()).unwrap();
    // Both filters use an all-ones 2x2 kernel; bias filter0=0, filter1=100
    let weights = Array4::from_elem((2, 1, 2, 2), 1.0f32);
    let bias = Array2::from_shape_vec((1, 2), vec![0.0f32, 100.0]).unwrap();
    layer.set_weights(weights, bias).unwrap();

    // x[b,0,i,j] = i + j, identical across the two batch samples
    let mut input = Array4::<f32>::zeros((2, 1, 51, 51));
    for b in 0..2 {
        for i in 0..51 {
            for j in 0..51 {
                input[[b, 0, i, j]] = (i + j) as f32;
            }
        }
    }
    let output = layer.forward(&input.into_dyn()).unwrap();

    assert_eq!(output.shape(), &[2, 2, 50, 50]);

    // out[b,0,oh,ow] = 4*oh + 4*ow + 4; out[b,1,..] = that + 100, checked over both batches
    let f0 = |oh: usize, ow: usize| (4 * oh + 4 * ow + 4) as f32;
    for b in 0..2 {
        // corners / edges of filter 0
        assert_abs_diff_eq!(output[[b, 0, 0, 0]], f0(0, 0), epsilon = 1e-4f32); // 4
        assert_abs_diff_eq!(output[[b, 0, 0, 1]], f0(0, 1), epsilon = 1e-4f32); // 8
        assert_abs_diff_eq!(output[[b, 0, 1, 0]], f0(1, 0), epsilon = 1e-4f32); // 8
        assert_abs_diff_eq!(output[[b, 0, 49, 49]], f0(49, 49), epsilon = 1e-4f32); // 396
        // filter 1 = filter 0 + bias 100
        assert_abs_diff_eq!(output[[b, 1, 0, 0]], f0(0, 0) + 100.0, epsilon = 1e-4f32); // 104
        assert_abs_diff_eq!(
            output[[b, 1, 49, 49]],
            f0(49, 49) + 100.0,
            epsilon = 1e-4f32
        ); // 496
    }
}

/// Parallel weight-gradient backward branch (workload >= 10_000): every kernel tap gets 576
#[test]
fn conv2d_parallel_weight_grad_constant_count() {
    let mut layer = Conv2D::new(1, (3, 3), vec![4, 2, 14, 14], (1, 1), Linear::new()).unwrap();
    // Weights do not affect the weight gradient; all-ones, bias 0, for determinism
    let weights = Array4::from_elem((1, 2, 3, 3), 1.0f32);
    let bias = Array2::zeros((1, 1));
    layer.set_weights(weights, bias).unwrap();

    let input = Array::ones((4, 2, 14, 14)).into_dyn();
    let _ = layer.forward(&input).unwrap();

    // grad w.r.t. the conv output: all ones, shape [batch, filters, out_h, out_w]
    let grad = Array::ones((4, 1, 12, 12)).into_dyn();
    let _input_grad = layer.backward(&grad).unwrap();

    // parameters() returns weights first; its `.grad` is the flat weight gradient
    let params = layer.parameters();
    let weight_grad = params[0].grad;
    assert_eq!(weight_grad.len(), 18, "expected 18 weight-grad entries");
    for &g in weight_grad.iter() {
        assert_abs_diff_eq!(g, 576.0f32, epsilon = 1e-3f32);
    }
}

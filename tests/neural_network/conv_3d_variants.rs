//! Integration tests for Conv3D, DepthwiseConv2D, and SeparableConv2D
//!
//! Expected values come from the mathematical definition or hand calculation, not from
//! recording layer output. Gradient checks live in tests/neural_network/gradient_check.rs

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2, Array4, Array5};
use rustyml::error::{Error, NnError};
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::traits::Layer;

use crate::common::assert_allclose;

// Conv3D - constructor validation

/// filters=0 must be rejected with InvalidParameter
#[test]
fn conv3d_new_rejects_zero_filters() {
    let err = Conv3D::new(
        0,
        (2, 2, 2),
        vec![1, 1, 4, 4, 4],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A zero in the kernel tuple must be rejected with InvalidParameter
#[test]
fn conv3d_new_rejects_zero_kernel_dimension() {
    // Second kernel dimension is 0
    let err = Conv3D::new(
        2,
        (2, 0, 2),
        vec![1, 1, 4, 4, 4],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A zero in the stride tuple must be rejected with InvalidParameter
#[test]
fn conv3d_new_rejects_zero_stride() {
    let err = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 1, 4, 4, 4],
        (1, 0, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A 4D input_shape (not 5D) must be rejected with InvalidInput
#[test]
fn conv3d_new_rejects_non_5d_input_shape() {
    let err = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 1, 4, 4], // only 4 dims
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// An input_shape containing a 0 dimension must be rejected with InvalidInput
#[test]
fn conv3d_new_rejects_zero_input_dimension() {
    let err = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 0, 4, 4, 4], // channels = 0
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// Conv3D - forward output shape

/// Valid padding forward output shape is [1, 2, 3, 3, 3] for input [1,1,4,4,4], kernel (2,2,2)
#[test]
fn conv3d_forward_output_shape_valid_padding() {
    let mut conv = Conv3D::new(
        2,
        (2, 2, 2),
        vec![1, 1, 4, 4, 4],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::ones((1_usize, 1, 4, 4, 4)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 3, 3, 3],
        "Conv3D Valid output shape mismatch"
    );
}

/// Same padding forward output shape equals ceil(input / stride): [1, 2, 4, 4, 4] here
#[test]
fn conv3d_forward_output_shape_same_padding() {
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
    let x = Array::ones((1_usize, 1, 4, 4, 4)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 4, 4, 4],
        "Conv3D Same output shape mismatch"
    );
}

/// Stride-2 Valid forward output shape is [2, 1, 2, 2, 2] for input [2,1,5,5,5], kernel (3,3,3)
#[test]
fn conv3d_forward_output_shape_stride2_valid() {
    let mut conv = Conv3D::new(
        1,
        (3, 3, 3),
        vec![2, 1, 5, 5, 5],
        (2, 2, 2),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::ones((2_usize, 1, 5, 5, 5)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[2, 1, 2, 2, 2],
        "Conv3D stride-2 Valid output shape mismatch"
    );
}

// Conv3D - known-weight forward value

/// All-ones 2x2x2 kernel sums the 8-element window: forward output matches the hand calculation
#[test]
fn conv3d_known_weight_forward_values() {
    let mut conv = Conv3D::new(
        1,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();

    // All weights 1, bias 0
    let w = Array5::ones((1_usize, 1, 2, 2, 2));
    let b = Array2::zeros((1_usize, 1));
    conv.set_weights(w, b).unwrap();

    // Input: x[i] = 0.05*i - 0.40 for i = 0..27
    let input_data: Vec<f32> = (0..27).map(|i| 0.05 * i as f32 - 0.40).collect();
    let x = Array::from_shape_vec((1_usize, 1, 3, 3, 3), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);

    // window sum = 52 -> 0.05*52 - 3.20 = -0.60
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], -0.60_f32, epsilon = 1e-5);
    // window sum = 60 -> 0.05*60 - 3.20 = -0.20
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 1]], -0.20_f32, epsilon = 1e-5);
    // window sum = 76 -> 0.05*76 - 3.20 = 0.60
    assert_abs_diff_eq!(out[[0, 0, 0, 1, 0]], 0.60_f32, epsilon = 1e-5);
    // window sum = 156 -> 0.05*156 - 3.20 = 4.60
    assert_abs_diff_eq!(out[[0, 0, 1, 1, 1]], 4.60_f32, epsilon = 1e-5);
}

// Conv3D - predict() == forward() in eval mode

/// predict() returns identical values to forward() (Conv3D has no train/eval difference)
#[test]
fn conv3d_predict_equals_forward() {
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
        (1_usize, 1, 3, 3, 3),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();

    let fwd = conv.forward(&x).unwrap();
    let pred = conv.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// Conv3D - error paths

/// Calling backward before forward must return NeuralNetwork(ForwardPassNotRun)
#[test]
fn conv3d_backward_before_forward_errors() {
    let mut conv = Conv3D::new(
        1,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let grad = Array::ones((1_usize, 1, 2, 2, 2)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// A non-5D input to forward() must return InvalidInput
#[test]
fn conv3d_forward_rejects_non_5d_input() {
    let mut conv = Conv3D::new(
        1,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    let x_4d = Array::ones((1_usize, 1, 3, 3)).into_dyn(); // 4D
    let err = conv.forward(&x_4d).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// set_weights with wrong shape must return NeuralNetwork(WeightShape)
#[test]
fn conv3d_set_weights_shape_mismatch_errors() {
    let mut conv = Conv3D::new(
        1,
        (2, 2, 2),
        vec![1, 1, 3, 3, 3],
        (1, 1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    // Layer expects weights [1,1,2,2,2]; supply [2,1,2,2,2]
    let wrong_w = Array5::zeros((2_usize, 1, 2, 2, 2));
    let b = Array2::zeros((1_usize, 1));
    let err = conv.set_weights(wrong_w, b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// DepthwiseConv2D - constructor validation

/// filters != channels must be rejected with InvalidParameter
#[test]
fn depthwise_conv2d_new_rejects_filters_not_equal_channels() {
    // input_shape has 2 channels, but 3 filters are requested
    let err = DepthwiseConv2D::new(
        3,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for filters != channels, got {err:?}"
    );
}

/// filters=0 must be rejected
#[test]
fn depthwise_conv2d_new_rejects_zero_filters() {
    let err = DepthwiseConv2D::new(
        0,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for filters=0, got {err:?}"
    );
}

/// A zero kernel dimension must be rejected
#[test]
fn depthwise_conv2d_new_rejects_zero_kernel_dimension() {
    let err = DepthwiseConv2D::new(
        2,
        (0, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for kernel_size.0=0, got {err:?}"
    );
}

// DepthwiseConv2D - channel independence (the core correctness invariant)

/// Each channel convolves only its own input: all-ones kernel gives 4.0, zero kernel gives 0.0
#[test]
fn depthwise_conv2d_channel_independence() {
    let mut conv = DepthwiseConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 3, 3],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();

    // Channel 0 kernel = all 1s; channel 1 kernel stays zero
    let mut w = Array4::<f32>::zeros((2, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 0, 0, 1]] = 1.0;
    w[[0, 0, 1, 0]] = 1.0;
    w[[0, 0, 1, 1]] = 1.0;
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input: channel 0 = 1s, channel 1 = 2s
    let mut input = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            input[[0, 0, h, ww]] = 1.0_f32;
            input[[0, 1, h, ww]] = 2.0_f32;
        }
    }
    let x = input.into_dyn();

    let out = conv.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Channel 0: every 2x2 window of all-1 input summed by all-1 kernel = 4.0
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, 0, oh, ow]], 4.0_f32, epsilon = 1e-6,);
        }
    }

    // Channel 1: zero kernel -> 0.0 everywhere, regardless of non-zero input
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, 1, oh, ow]], 0.0_f32, epsilon = 1e-6,);
        }
    }
}

/// Channel 0 output does not change when channel 1 input changes (no cross-channel bleed)
#[test]
fn depthwise_conv2d_cross_channel_no_bleed() {
    let mut conv = DepthwiseConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 3, 3],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();

    // Channel 0 kernel = [[1,0],[0,0]] (only top-left); channel 1 kernel = all zeros
    let mut w = Array4::<f32>::zeros((2, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input A: channel 0 = 1s everywhere, channel 1 = 0s
    let mut inp_a = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            inp_a[[0, 0, h, ww]] = 1.0_f32;
        }
    }
    let out_a = conv.forward(&inp_a.into_dyn()).unwrap();

    // Input B: channel 0 = 1s everywhere, channel 1 = 999s
    let mut inp_b = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            inp_b[[0, 0, h, ww]] = 1.0_f32;
            inp_b[[0, 1, h, ww]] = 999.0_f32;
        }
    }
    let out_b = conv.forward(&inp_b.into_dyn()).unwrap();

    // Channel 0 output must be identical in both cases
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_a[[0, 0, oh, ow]], out_b[[0, 0, oh, ow]], epsilon = 1e-6,);
        }
    }

    // Channel 0: top-left kernel picks up x[oh,ow] = 1.0 for all positions
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, 0, oh, ow]], 1.0_f32, epsilon = 1e-6);
        }
    }

    // Channel 1: zero kernel -> 0.0 everywhere even though input was 999
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, 1, oh, ow]], 0.0_f32, epsilon = 1e-6);
        }
    }
}

// DepthwiseConv2D - known-weight forward values

/// Single-channel 2x2 kernel [[1,2],[3,4]] with bias 0.5 over input 1..=9 matches hand calculation
#[test]
fn depthwise_conv2d_known_weight_single_channel() {
    let mut conv = DepthwiseConv2D::new(
        1,
        (2, 2),
        vec![1, 1, 3, 3],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();

    // Kernel: [[1,2],[3,4]], bias 0.5
    let mut w = Array4::<f32>::zeros((1, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 0, 0, 1]] = 2.0;
    w[[0, 0, 1, 0]] = 3.0;
    w[[0, 0, 1, 1]] = 4.0;
    let bias = Array1::from_vec(vec![0.5_f32]);
    conv.set_weights(w, bias).unwrap();

    // Input: 1..=9 in row-major order
    let x = Array::from_shape_vec(
        (1_usize, 1, 3, 3),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 37.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 47.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 67.5_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 77.5_f32, epsilon = 1e-5);
}

// DepthwiseConv2D - predict() == forward() in eval mode

#[test]
fn depthwise_conv2d_predict_equals_forward() {
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
        (1_usize, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();

    let fwd = conv.forward(&x).unwrap();
    let pred = conv.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// DepthwiseConv2D - error paths

/// backward before forward must return ForwardPassNotRun
#[test]
fn depthwise_conv2d_backward_before_forward_errors() {
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
    let grad = Array::ones((1_usize, 2, 3, 3)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong weight shape must return NeuralNetwork(WeightShape)
#[test]
fn depthwise_conv2d_set_weights_shape_mismatch_errors() {
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
    // Layer expects weights [2,1,2,2]; supply [3,1,2,2]
    let wrong_w = Array4::<f32>::zeros((3, 1, 2, 2));
    let bias = Array1::<f32>::zeros(2);
    let err = conv.set_weights(wrong_w, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// SeparableConv2D - constructor validation

/// filters=0 must be rejected
#[test]
fn separable_conv2d_new_rejects_zero_filters() {
    let err = SeparableConv2D::new(
        0,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for filters=0, got {err:?}"
    );
}

/// depth_multiplier=0 must be rejected
#[test]
fn separable_conv2d_new_rejects_zero_depth_multiplier() {
    let err = SeparableConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        0,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for depth_multiplier=0, got {err:?}"
    );
}

/// Zero kernel dimension must be rejected
#[test]
fn separable_conv2d_new_rejects_zero_kernel() {
    let err = SeparableConv2D::new(
        2,
        (2, 0),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for kernel_size.1=0, got {err:?}"
    );
}

/// Non-4D input_shape must be rejected
#[test]
fn separable_conv2d_new_rejects_non_4d_input_shape() {
    let err = SeparableConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4], // only 3 dims
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for 3D input_shape, got {err:?}"
    );
}

// SeparableConv2D - depth_multiplier behavior and output shape

/// depth_multiplier=1, filters=1, 1 input channel: forward output shape is [1, 1, 2, 2]
#[test]
fn separable_conv2d_output_shape_dm1() {
    let mut conv = SeparableConv2D::new(
        1,
        (2, 2),
        vec![1, 1, 3, 3],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::ones((1_usize, 1, 3, 3)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 1, 2, 2],
        "SeparableConv2D dm=1 shape mismatch"
    );
}

/// depth_multiplier=2, 2 input channels, 4 output filters: forward output shape is [1, 4, 2, 2]
#[test]
fn separable_conv2d_output_shape_dm2() {
    let mut conv = SeparableConv2D::new(
        4,
        (2, 2),
        vec![1, 2, 3, 3],
        (1, 1),
        PaddingType::Valid,
        2,
        Linear::new(),
        None,
    )
    .unwrap();
    let x = Array::ones((1_usize, 2, 3, 3)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 4, 2, 2],
        "SeparableConv2D dm=2 output shape mismatch"
    );
}

// SeparableConv2D - identity depthwise + identity pointwise reproduces input

/// Identity 1x1 depthwise and pointwise kernels with zero bias reproduce the input exactly
#[test]
fn separable_conv2d_identity_reproduces_input() {
    let mut conv = SeparableConv2D::new(
        1,
        (1, 1),
        vec![1, 1, 3, 3],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap();

    // depthwise_weights: [dm=1, channels=1, 1, 1] -> 1.0
    let dw = Array4::<f32>::ones((1, 1, 1, 1));
    // pointwise_weights: [filters=1, channels*dm=1, 1, 1] -> 1.0
    let pw = Array4::<f32>::ones((1, 1, 1, 1));
    let bias = Array2::<f32>::zeros((1, 1));
    conv.set_weights(dw, pw, bias).unwrap();

    let input_data: Vec<f32> = (0..9).map(|i| i as f32 * 0.5 - 2.0).collect();
    let x = Array::from_shape_vec((1_usize, 1, 3, 3), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    // Every output value must equal the corresponding input value
    for h in 0..3 {
        for w in 0..3 {
            assert_abs_diff_eq!(out[[0, 0, h, w]], x[[0, 0, h, w]], epsilon = 1e-6);
        }
    }
}

// SeparableConv2D - known-weight forward value

/// Depthwise [[1,0],[0,1]] then pointwise scale-by-2 plus bias 1 over input 1..=9 matches by hand
#[test]
fn separable_conv2d_known_weight_forward_values() {
    let mut conv = SeparableConv2D::new(
        1,
        (2, 2),
        vec![1, 1, 3, 3],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap();

    // depthwise: [[1,0],[0,1]] picks up (i,j) and (i+1,j+1)
    let mut dw = Array4::<f32>::zeros((1, 1, 2, 2));
    dw[[0, 0, 0, 0]] = 1.0;
    dw[[0, 0, 1, 1]] = 1.0;

    // pointwise: scale by 2, bias 1.0
    let mut pw = Array4::<f32>::zeros((1, 1, 1, 1));
    pw[[0, 0, 0, 0]] = 2.0;

    let bias = Array2::from_elem((1, 1), 1.0_f32);
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 1, 3, 3),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 13.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], 17.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 25.0_f32, epsilon = 1e-5);
    assert_abs_diff_eq!(out[[0, 0, 1, 1]], 29.0_f32, epsilon = 1e-5);
}

/// `Same` padding zero-pads the depthwise stage: output is the zero-padded 3x3 box filter over 1..=9
/// (regression - the old boundary-clipping code returned the unpadded 3x3 sum 45 at [0,0])
#[test]
fn separable_conv2d_same_padding_zero_pads_depthwise() {
    let mut conv = SeparableConv2D::new(
        1,
        (3, 3),
        vec![1, 1, 3, 3],
        (1, 1),
        PaddingType::Same,
        1,
        Linear::new(),
        None,
    )
    .unwrap();

    // depthwise: 3x3 all-ones box filter; pointwise: identity (scale by 1); bias 0
    let dw = Array4::<f32>::from_elem((1, 1, 3, 3), 1.0);
    let pw = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0);
    let bias = Array2::<f32>::zeros((1, 1));
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 1, 3, 3),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    // `Same` keeps the spatial shape
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    let expected = [
        [12.0_f32, 21.0, 16.0],
        [27.0, 45.0, 33.0],
        [24.0, 39.0, 28.0],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert_abs_diff_eq!(out[[0, 0, i, j]], want, epsilon = 1e-4);
        }
    }
}

// SeparableConv2D - predict() == forward() in eval mode

#[test]
fn separable_conv2d_predict_equals_forward() {
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
        (1_usize, 2, 4, 4),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();

    let fwd = conv.forward(&x).unwrap();
    let pred = conv.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// SeparableConv2D - error paths

/// backward before forward must return ForwardPassNotRun
#[test]
fn separable_conv2d_backward_before_forward_errors() {
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
    let grad = Array::ones((1_usize, 2, 3, 3)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong depthwise shape must return NeuralNetwork(WeightShape)
#[test]
fn separable_conv2d_set_weights_shape_mismatch_errors() {
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
    // depthwise_weights should be [dm=1, channels=2, 2, 2]; supply wrong dm=3
    let bad_dw = Array4::<f32>::zeros((3, 2, 2, 2));
    let ok_pw = Array4::<f32>::zeros((2, 2, 1, 1));
    let ok_b = Array2::<f32>::zeros((1, 2));
    let err = conv.set_weights(bad_dw, ok_pw, ok_b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// SeparableConv2D - depth_multiplier correctly expands intermediate channels

/// depth_multiplier=2 yields two scaled copies; filter 0 reproduces input, filter 1 is 2x input
#[test]
fn separable_conv2d_depth_multiplier_2_forward_values() {
    let mut conv = SeparableConv2D::new(
        2,
        (1, 1),
        vec![1, 1, 2, 2],
        (1, 1),
        PaddingType::Valid,
        2,
        Linear::new(),
        None,
    )
    .unwrap();

    // depthwise_weights shape [dm=2, channels=1, 1, 1]
    let mut dw = Array4::<f32>::zeros((2, 1, 1, 1));
    dw[[0, 0, 0, 0]] = 1.0; // dm=0: scale by 1
    dw[[1, 0, 0, 0]] = 2.0; // dm=1: scale by 2

    // pointwise_weights [filters=2, channels*dm=2, 1, 1]: filter 0 selects channel 0, filter 1 channel 1
    let mut pw = Array4::<f32>::zeros((2, 2, 1, 1));
    pw[[0, 0, 0, 0]] = 1.0;
    pw[[1, 1, 0, 0]] = 1.0;

    let bias = Array2::<f32>::zeros((1, 2));
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec((1_usize, 1, 2, 2), vec![1.0_f32, 2., 3., 4.])
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Filter 0: should reproduce input (dm=0, scale=1, pw=1)
    let expected_f0 = [[1.0_f32, 2.], [3., 4.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, 0, h, w]], expected_f0[h][w], epsilon = 1e-5);
        }
    }

    // Filter 1: should be 2x input (dm=1, scale=2, pw=1)
    let expected_f1 = [[2.0_f32, 4.], [6., 8.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, 1, h, w]], expected_f1[h][w], epsilon = 1e-5);
        }
    }
}
// DepthwiseConv2D - PARALLEL branch (workload >= 1500)

/// All-ones 2x2 kernel over a [1,1,40,40] ramp crosses the parallel threshold and sums correctly
#[test]
fn depthwise_conv2d_parallel_windowed_sums() {
    let mut conv = DepthwiseConv2D::new(
        1,
        (2, 2),
        vec![1, 1, 40, 40],
        (1, 1),
        PaddingType::Valid,
        Linear::new(),
        None,
    )
    .unwrap();
    // All-ones 2x2 kernel, zero bias
    let w = Array4::<f32>::from_elem((1, 1, 2, 2), 1.0);
    let bias = Array1::<f32>::zeros(1);
    conv.set_weights(w, bias).unwrap();

    // x[0,0,i,j] = i + j
    let mut input = Array::<f32, _>::zeros((1, 1, 40, 40));
    for i in 0..40 {
        for j in 0..40 {
            input[[0, 0, i, j]] = (i + j) as f32;
        }
    }
    let out = conv.forward(&input.into_dyn()).unwrap();

    assert_eq!(out.shape(), &[1, 1, 39, 39]);

    let expect = |oh: usize, ow: usize| (4 * oh + 4 * ow + 4) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 4
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], expect(0, 1), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(1, 0), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 0, 38, 38]], expect(38, 38), epsilon = 1e-4); // 308
}

// SeparableConv2D - PARALLEL branch (depthwise & pointwise workloads >= 5000)

/// A [1,1,72,72] ramp crosses both parallel thresholds; final out = 8*oh + 8*ow + 9
#[test]
fn separable_conv2d_parallel_windowed_sums() {
    let mut conv = SeparableConv2D::new(
        1,
        (2, 2),
        vec![1, 1, 72, 72],
        (1, 1),
        PaddingType::Valid,
        1,
        Linear::new(),
        None,
    )
    .unwrap();

    // depthwise: all-ones 2x2; pointwise: scale by 2; bias 1.0
    let dw = Array4::<f32>::from_elem((1, 1, 2, 2), 1.0);
    let pw = Array4::<f32>::from_elem((1, 1, 1, 1), 2.0);
    let bias = Array2::<f32>::from_elem((1, 1), 1.0);
    conv.set_weights(dw, pw, bias).unwrap();

    // x[0,0,i,j] = i + j
    let mut input = Array::<f32, _>::zeros((1, 1, 72, 72));
    for i in 0..72 {
        for j in 0..72 {
            input[[0, 0, i, j]] = (i + j) as f32;
        }
    }
    let out = conv.forward(&input.into_dyn()).unwrap();

    assert_eq!(out.shape(), &[1, 1, 71, 71]);

    // final out[0,0,oh,ow] = 8*oh + 8*ow + 9
    let expect = |oh: usize, ow: usize| (8 * oh + 8 * ow + 9) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 9
    assert_abs_diff_eq!(out[[0, 0, 0, 1]], expect(0, 1), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(1, 0), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 0, 70, 70]], expect(70, 70), epsilon = 1e-4); // 1129
}

//! Integration tests for Conv3D, DepthwiseConv2D, and SeparableConv2D.
//!
//! Every expected value is derived from the mathematical definition or a hand calculation,
//! not by recording the layer's output. Gradient checks are already covered by
//! tests/neural_network/gradient_check.rs and are NOT duplicated here.

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

// ════════════════════════════════════════════════════════════════════════════════════════
// Conv3D — constructor validation
// ════════════════════════════════════════════════════════════════════════════════════════

/// filters=0 must be rejected with InvalidParameter.
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

/// A zero in the kernel tuple must be rejected with InvalidParameter.
#[test]
fn conv3d_new_rejects_zero_kernel_dimension() {
    // Second kernel dimension is 0.
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

/// A zero in the stride tuple must be rejected with InvalidParameter.
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

/// A 4D input_shape (not 5D) must be rejected with InvalidInput.
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

/// An input_shape containing a 0 dimension must be rejected with InvalidInput.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// Conv3D — forward output shape
// ════════════════════════════════════════════════════════════════════════════════════════

/// Valid padding output shape formula:
///   out_d = (D - Kd) / sd + 1,  similarly for h, w.
/// With input [1,1,4,4,4], kernel (2,2,2), stride (1,1,1):
///   out_d = (4-2)/1+1 = 3, same for h, w → [1, 2, 3, 3, 3].
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

/// Same padding: output shape must equal ceil(input / stride) in each spatial dim.
/// With input [1,1,4,4,4], kernel (3,3,3), stride (1,1,1):
///   out = ceil(4/1) = 4 in each dim → [1, 2, 4, 4, 4].
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

/// Stride-2 Valid output:
/// input [2,1,5,5,5], kernel (3,3,3), stride (2,2,2):
///   out_d = (5-3)/2+1 = 2, same for h, w → [2, 1, 2, 2, 2].
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

// ════════════════════════════════════════════════════════════════════════════════════════
// Conv3D — known-weight forward value
// ════════════════════════════════════════════════════════════════════════════════════════

/// Known-weight forward value test.
///
/// Layer: 1 filter, kernel (2,2,2), 1 input channel, stride (1,1,1), Valid, Linear activation.
/// Weights: all 1s (shape [1,1,2,2,2]), bias: 0.
/// Input: x[i] = 0.05*i - 0.40 for i in 0..27, shaped [1,1,3,3,3].
///
/// The 3D convolution at each output position is the sum of the 8 input elements in the
/// corresponding [2,2,2] window.
///
/// Hand calculation (row-major order, d×h×w indexing, flat index = d*9 + h*3 + w):
///   out[0,0,0,0,0] = Σ x[d,h,w] for d in 0..2, h in 0..2, w in 0..2
///     flat indices: 0,1,2,3,4,5,9,10,11,12,13,14 ... wait, 2×2×2 window:
///     d=0,h=0: w=0→idx0, w=1→idx1
///     d=0,h=1: w=0→idx3, w=1→idx4
///     d=1,h=0: w=0→idx9, w=1→idx10
///     d=1,h=1: w=0→idx12, w=1→idx13
///     sum of indices = 0+1+3+4+9+10+12+13 = 52
///     sum = 0.05*52 - 8*0.40 = 2.60 - 3.20 = -0.60
///
///   out[0,0,1,1,1] = window starting at (d=1,h=1,w=1):
///     flat indices: 1*9+1*3+1=13, 13+1=14, 1*9+2*3+1=16, 16+1=17,
///                   2*9+1*3+1=22, 22+1=23, 2*9+2*3+1=25, 25+1=26
///     sum = 0.05*(13+14+16+17+22+23+25+26) - 8*0.40 = 0.05*156 - 3.20 = 7.80 - 3.20 = 4.60
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

    // Set all weights to 1, bias to 0.
    let w = Array5::ones((1_usize, 1, 2, 2, 2));
    let b = Array2::zeros((1_usize, 1));
    conv.set_weights(w, b).unwrap();

    // Input: x[i] = 0.05*i - 0.40 for i = 0..27.
    let input_data: Vec<f32> = (0..27).map(|i| 0.05 * i as f32 - 0.40).collect();
    let x = Array::from_shape_vec((1_usize, 1, 3, 3, 3), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();

    // Output shape: [1,1,2,2,2]
    assert_eq!(out.shape(), &[1, 1, 2, 2, 2]);

    // out[0,0,0,0,0]: window d in 0..2, h in 0..2, w in 0..2
    // flat indices: 0,1,3,4,9,10,12,13 → sum = 52 → 0.05*52 - 3.20 = -0.60
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], -0.60_f32, epsilon = 1e-5);

    // out[0,0,0,0,1]: window d in 0..2, h in 0..2, w in 1..3
    // flat indices: 1,2,4,5,10,11,13,14 → sum = 60 → 0.05*60 - 3.20 = -0.20
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 1]], -0.20_f32, epsilon = 1e-5);

    // out[0,0,0,1,0]: window d in 0..2, h in 1..3, w in 0..2
    // flat indices: 3,4,6,7,12,13,15,16 → sum = 76 → 0.05*76 - 3.20 = 0.60
    assert_abs_diff_eq!(out[[0, 0, 0, 1, 0]], 0.60_f32, epsilon = 1e-5);

    // out[0,0,1,1,1]: window d in 1..3, h in 1..3, w in 1..3
    // flat indices: 13,14,16,17,22,23,25,26 → sum = 156 → 0.05*156 - 3.20 = 4.60
    assert_abs_diff_eq!(out[[0, 0, 1, 1, 1]], 4.60_f32, epsilon = 1e-5);
}

// ════════════════════════════════════════════════════════════════════════════════════════
// Conv3D — predict() == forward() in eval mode
// ════════════════════════════════════════════════════════════════════════════════════════

/// predict() must return identical values to forward() (Conv3D has no train/eval difference).
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

// ════════════════════════════════════════════════════════════════════════════════════════
// Conv3D — error paths
// ════════════════════════════════════════════════════════════════════════════════════════

/// Calling backward before forward must return NeuralNetwork(ForwardPassNotRun).
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

/// A non-5D input to forward() must return InvalidInput.
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

/// set_weights with wrong shape must return NeuralNetwork(WeightShape).
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
    // Layer expects weights [1,1,2,2,2]; supply [2,1,2,2,2].
    let wrong_w = Array5::zeros((2_usize, 1, 2, 2, 2));
    let b = Array2::zeros((1_usize, 1));
    let err = conv.set_weights(wrong_w, b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — constructor validation
// ════════════════════════════════════════════════════════════════════════════════════════

/// filters != channels must be rejected with InvalidParameter.
#[test]
fn depthwise_conv2d_new_rejects_filters_not_equal_channels() {
    // input_shape has 2 channels, but we specify 3 filters.
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

/// filters=0 must be rejected.
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

/// A zero kernel dimension must be rejected.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — channel independence (the core correctness invariant)
// ════════════════════════════════════════════════════════════════════════════════════════

/// Channel independence test.
///
/// Layer: 2 channels (hence 2 filters), 2×2 kernel, stride (1,1), Valid, Linear activation.
/// Weight layout: shape [filters, 1, kH, kW] = [2, 1, 2, 2].
///   Channel 0 kernel: [[1,1],[1,1]]  (all ones)
///   Channel 1 kernel: [[0,0],[0,0]]  (all zeros)
///   Biases: [0.0, 0.0]
///
/// Input [1, 2, 3, 3]:
///   Channel 0: all 1s → output is 2×2 with every value = 4.0
///     (sum of 4 ones-weighted input elements)
///   Channel 1: all 2s → output is 2×2 with every value = 0.0
///     (kernel is zero so all contributions cancel)
///
/// This test verifies that:
///   1. Channel 0 output depends ONLY on channel 0 weights and channel 0 input.
///   2. Channel 1 output is identically zero (the zero kernel kills all input).
///   3. Channel 1 input values (which are non-zero) do NOT leak into channel 0 output.
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

    // Channel 0 kernel = all 1s; Channel 1 kernel = all 0s.
    let mut w = Array4::<f32>::zeros((2, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 0, 0, 1]] = 1.0;
    w[[0, 0, 1, 0]] = 1.0;
    w[[0, 0, 1, 1]] = 1.0;
    // channel 1 stays zero
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input: channel 0 = 1s, channel 1 = 2s.
    let mut input = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            input[[0, 0, h, ww]] = 1.0_f32;
            input[[0, 1, h, ww]] = 2.0_f32;
        }
    }
    let x = input.into_dyn();

    let out = conv.forward(&x).unwrap();

    // Output shape: [1, 2, 2, 2]
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Channel 0: every 2×2 window of all-1 input summed by all-1 kernel = 4.0
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, 0, oh, ow]], 4.0_f32, epsilon = 1e-6,);
        }
    }

    // Channel 1: zero kernel → output must be 0.0 everywhere,
    // regardless of non-zero input values.
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, 1, oh, ow]], 0.0_f32, epsilon = 1e-6,);
        }
    }
}

/// Verify that channel 0's output does NOT change when channel 1 input changes.
/// This is the dual of channel_independence: non-zero channel-1 input must NOT
/// bleed into channel-0 output.
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

    // Channel 0 kernel = [[1,0],[0,0]] (only top-left), channel 1 kernel = all zeros.
    let mut w = Array4::<f32>::zeros((2, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input A: channel 0 = 1s everywhere, channel 1 = 0s.
    let mut inp_a = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            inp_a[[0, 0, h, ww]] = 1.0_f32;
        }
    }
    let out_a = conv.forward(&inp_a.into_dyn()).unwrap();

    // Input B: channel 0 = 1s everywhere, channel 1 = 999s.
    let mut inp_b = Array::zeros((1_usize, 2, 3, 3));
    for h in 0..3 {
        for ww in 0..3 {
            inp_b[[0, 0, h, ww]] = 1.0_f32;
            inp_b[[0, 1, h, ww]] = 999.0_f32;
        }
    }
    let out_b = conv.forward(&inp_b.into_dyn()).unwrap();

    // Channel 0 output must be identical in both cases.
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_a[[0, 0, oh, ow]], out_b[[0, 0, oh, ow]], epsilon = 1e-6,);
        }
    }

    // Channel 0: top-left kernel picks up x[oh,ow] = 1.0 for all positions.
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, 0, oh, ow]], 1.0_f32, epsilon = 1e-6);
        }
    }

    // Channel 1: zero kernel → 0.0 everywhere even though input was 999.
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, 1, oh, ow]], 0.0_f32, epsilon = 1e-6);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — known-weight forward values
// ════════════════════════════════════════════════════════════════════════════════════════

/// Single-channel, 2×2 kernel with non-trivial weights.
/// Input [1,1,3,3]: row-major values 1..=9.
/// Kernel: [[1,2],[3,4]], bias: 0.5.
///
/// Output [1,1,2,2]:
///   out[0,0,0,0] = 1*1 + 2*2 + 3*4 + 4*7 + 0.5 = 1+4+12+28+0.5 = ... let me redo:
///     input window (h=0,w=0): [input[0,0]=1, input[0,1]=2, input[1,0]=4, input[1,1]=5]
///     kernel: k[0,0]=1, k[0,1]=2, k[1,0]=3, k[1,1]=4
///     sum = 1*1 + 2*2 + 4*3 + 5*4 = 1 + 4 + 12 + 20 = 37, + bias 0.5 = 37.5
///   out[0,0,0,1]: window (h=0,w=1): input values [2,3,5,6]
///     sum = 2*1 + 3*2 + 5*3 + 6*4 = 2+6+15+24 = 47 + 0.5 = 47.5
///   out[0,0,1,0]: window (h=1,w=0): input values [4,5,7,8]
///     sum = 4*1 + 5*2 + 7*3 + 8*4 = 4+10+21+32 = 67 + 0.5 = 67.5
///   out[0,0,1,1]: window (h=1,w=1): input values [5,6,8,9]
///     sum = 5*1 + 6*2 + 8*3 + 9*4 = 5+12+24+36 = 77 + 0.5 = 77.5
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

    // Kernel: [[1,2],[3,4]], bias: 0.5
    let mut w = Array4::<f32>::zeros((1, 1, 2, 2));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 0, 0, 1]] = 2.0;
    w[[0, 0, 1, 0]] = 3.0;
    w[[0, 0, 1, 1]] = 4.0;
    let bias = Array1::from_vec(vec![0.5_f32]);
    conv.set_weights(w, bias).unwrap();

    // Input: 1..=9 in row-major order.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — predict() == forward() in eval mode
// ════════════════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — error paths
// ════════════════════════════════════════════════════════════════════════════════════════

/// backward before forward must return ForwardPassNotRun.
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

/// set_weights with wrong weight shape must return NeuralNetwork(WeightShape).
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
    // Layer expects weights [2,1,2,2]; supply [3,1,2,2].
    let wrong_w = Array4::<f32>::zeros((3, 1, 2, 2));
    let bias = Array1::<f32>::zeros(2);
    let err = conv.set_weights(wrong_w, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — constructor validation
// ════════════════════════════════════════════════════════════════════════════════════════

/// filters=0 must be rejected.
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

/// depth_multiplier=0 must be rejected.
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

/// Zero kernel dimension must be rejected.
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

/// Non-4D input_shape must be rejected.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — depth_multiplier behavior and output shape
// ════════════════════════════════════════════════════════════════════════════════════════

/// depth_multiplier=1, filters=1, 1 input channel:
/// depthwise output: [batch, channels*dm, oh, ow] = [1, 1*1, 2, 2]
/// pointwise output: [1, 1, 2, 2]
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

/// depth_multiplier=2 with 2 input channels and 4 output filters:
///   depthwise channels = 2*2 = 4
///   pointwise: input channels=4, output=4
///   spatial: (3-2)/1+1 = 2 in each dim
///   final output: [1, 4, 2, 2]
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

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — identity depthwise + identity pointwise reproduces input
// ════════════════════════════════════════════════════════════════════════════════════════

/// Identity test.
///
/// Setup: 1 input channel, depth_multiplier=1, 1 output filter,
///        1×1 kernel (all operations are pointwise), Valid padding, stride (1,1), Linear.
///
/// With a 1×1 kernel:
///   depthwise_weights shape = [1, 1, 1, 1]; set to [[[[1.0]]]].
///   pointwise_weights shape = [1, 1*1, 1, 1] = [1, 1, 1, 1]; set to [[[[1.0]]]].
///   bias = 0.
///
/// Valid-padded 1×1 convolution with stride 1 leaves spatial dimensions unchanged:
///   output_h = (H - 1)/1 + 1 = H, likewise W.
///
/// Mathematical result:
///   depthwise out[b,0,i,j] = input[b,0,i,j] * 1.0 = input[b,0,i,j]
///   pointwise out[b,0,i,j] = depthwise_out[b,0,i,j] * 1.0 + 0.0 = input[b,0,i,j]
/// So the output must equal the input exactly.
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

    // depthwise_weights: [dm=1, channels=1, 1, 1] → 1.0
    let dw = Array4::<f32>::ones((1, 1, 1, 1));
    // pointwise_weights: [filters=1, channels*dm=1, 1, 1] → 1.0
    let pw = Array4::<f32>::ones((1, 1, 1, 1));
    let bias = Array2::<f32>::zeros((1, 1));
    conv.set_weights(dw, pw, bias).unwrap();

    let input_data: Vec<f32> = (0..9).map(|i| i as f32 * 0.5 - 2.0).collect();
    let x = Array::from_shape_vec((1_usize, 1, 3, 3), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    // Every output value must equal the corresponding input value.
    for h in 0..3 {
        for w in 0..3 {
            assert_abs_diff_eq!(out[[0, 0, h, w]], x[[0, 0, h, w]], epsilon = 1e-6);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — known-weight forward value
// ════════════════════════════════════════════════════════════════════════════════════════

/// Manually calculated forward pass.
///
/// Setup: 1 channel, depth_multiplier=1, 1 output filter, 2×2 kernel, Valid, stride (1,1).
///   depthwise_weights [1, 1, 2, 2]: [[1, 0], [0, 1]]  (identity-like kernel)
///   pointwise_weights [1, 1, 1, 1]: [[[[2.0]]]]
///   bias [1, 1]: 1.0
///
/// Input [1, 1, 3, 3]: 1..=9 in row-major order.
///
/// Step 1 – depthwise (Valid, kernel [[1,0],[0,1]]):
///   out_dw[i,j] = 1*x[i,j] + 0*x[i,j+1] + 0*x[i+1,j] + 1*x[i+1,j+1]
///              = x[i,j] + x[i+1,j+1]
///   dw[0,0] = 1+5=6, dw[0,1] = 2+6=8, dw[1,0] = 4+8=12, dw[1,1] = 5+9=14
///
/// Step 2 – pointwise (multiply by 2, add bias 1):
///   out[0,0] = 6*2 + 1 = 13, out[0,1] = 8*2 + 1 = 17
///   out[1,0] = 12*2 + 1 = 25, out[1,1] = 14*2 + 1 = 29
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

    // depthwise: [[1,0],[0,1]] — picks up (i,j) and (i+1,j+1)
    let mut dw = Array4::<f32>::zeros((1, 1, 2, 2));
    dw[[0, 0, 0, 0]] = 1.0;
    dw[[0, 0, 1, 1]] = 1.0;

    // pointwise: scale by 2
    let mut pw = Array4::<f32>::zeros((1, 1, 1, 1));
    pw[[0, 0, 0, 0]] = 2.0;

    // bias: 1.0
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

/// SeparableConv2D with `Same` padding must zero-pad the depthwise stage, not boundary-clip.
///
/// Regression test: the depthwise stage used to align every kernel window at `i*stride` and clip
/// at the input border, silently computing a `Valid`-style, top-left-aligned result even under
/// `Same`. With a 3×3 kernel, correct `Same` padding adds a *leading* zero row/column
/// (`pad_top = pad_left = 1`), which the buggy code never produced.
///
/// Setup: 1 channel, depth_multiplier=1, 1 filter, 3×3 all-ones depthwise kernel, identity
/// pointwise (weight 1, bias 0), Linear, stride (1,1), `Same`. The output equals the zero-padded
/// 3×3 box filter over `1..=9`:
///
/// ```text
///   padded (pad 1 every edge):        output[i,j] = sum of centered 3×3 window
///     0 0 0 0 0                            12 21 16
///     0 1 2 3 0          ==>               27 45 33
///     0 4 5 6 0                            24 39 28
///     0 7 8 9 0
///     0 0 0 0 0
/// ```
///
/// The old boundary-clipping code instead produced 45 at `[0,0]` (the full unpadded 3×3 sum).
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

    // depthwise: 3×3 all-ones box filter; pointwise: identity (scale by 1); bias: 0.
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
    // `Same` keeps the spatial shape.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — predict() == forward() in eval mode
// ════════════════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — error paths
// ════════════════════════════════════════════════════════════════════════════════════════

/// backward before forward must return ForwardPassNotRun.
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

/// set_weights with wrong depthwise shape must return NeuralNetwork(WeightShape).
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
    // depthwise_weights should be [dm=1, channels=2, 2, 2]; supply wrong dm=3.
    let bad_dw = Array4::<f32>::zeros((3, 2, 2, 2));
    let ok_pw = Array4::<f32>::zeros((2, 2, 1, 1));
    let ok_b = Array2::<f32>::zeros((1, 2));
    let err = conv.set_weights(bad_dw, ok_pw, ok_b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — depth_multiplier correctly expands intermediate channels
// ════════════════════════════════════════════════════════════════════════════════════════

/// With depth_multiplier=2, 1 input channel, 2 output filters, 1×1 kernel, Valid:
///   depthwise_weights [2, 1, 1, 1]: dm0=1.0, dm1=2.0 → two scaled copies of input
///   pointwise_weights [2, 2, 1, 1]: pw[0,0]=1.0,pw[0,1]=0.0; pw[1,0]=0.0,pw[1,1]=1.0
///   bias [1, 2]: 0.0
///
/// Input [1, 1, 2, 2]: [[1,2],[3,4]]
///
/// Step 1 – depthwise (1×1 kernel, no spatial reduction):
///   Intermediate channel 0 (dm=0): input * 1.0 → [[1,2],[3,4]]
///   Intermediate channel 1 (dm=1): input * 2.0 → [[2,4],[6,8]]
///
/// Step 2 – pointwise:
///   Output filter 0: picks intermediate channel 0 → [[1,2],[3,4]]
///   Output filter 1: picks intermediate channel 1 → [[2,4],[6,8]]
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

    // pointwise_weights shape [filters=2, channels*dm=2, 1, 1]
    // filter 0 selects intermediate channel 0 only; filter 1 selects channel 1 only.
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

    // Filter 0: should reproduce input (dm=0, scale=1, pw=1).
    let expected_f0 = [[1.0_f32, 2.], [3., 4.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, 0, h, w]], expected_f0[h][w], epsilon = 1e-5);
        }
    }

    // Filter 1: should be 2× input (dm=1, scale=2, pw=1).
    let expected_f1 = [[2.0_f32, 4.], [6., 8.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, 1, h, w]], expected_f1[h][w], epsilon = 1e-5);
        }
    }
}
// ════════════════════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D — PARALLEL branch (workload >= 1500)
// ════════════════════════════════════════════════════════════════════════════════════════

/// DepthwiseConv2D parallel forward branch.
///
/// The layer runs in parallel once `batch*channels*out_h*out_w >= 1500`
/// (DEPTHWISE_CONV_2D_PARALLEL_THRESHOLD). All other depthwise tests here are tiny and only
/// hit the sequential path; this is the smallest tensor that crosses the threshold.
///
/// batch=1, channels=1 (=filters), input [1,1,40,40], all-ones 2x2 kernel, stride 1, Valid:
///   out = (40-2)/1+1 = 39  →  workload = 1*1*39*39 = 1521  (>= 1500 → parallel)
///
/// Input ramp x[0,0,i,j] = i + j; an all-ones 2x2 kernel sums its receptive field:
///   out[0,0,oh,ow] = (oh+ow)+(oh+ow+1)+(oh+1+ow)+(oh+1+ow+1) = 4*oh + 4*ow + 4
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
    // All-ones 2x2 kernel, zero bias.
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

// ════════════════════════════════════════════════════════════════════════════════════════
// SeparableConv2D — PARALLEL branch (depthwise & pointwise workloads >= 5000)
// ════════════════════════════════════════════════════════════════════════════════════════

/// SeparableConv2D parallel depthwise + pointwise branches.
///
/// Both stages switch to parallel at SEPARABLE_CONV_2D_PARALLEL_THRESHOLD (5000):
///   depthwise: batch*channels*depth_multiplier*out_h*out_w >= 5000
///   pointwise: batch*filters*out_h*out_w >= 5000
///
/// batch=1, channels=1, dm=1, filters=1, input [1,1,72,72], kernel (2,2), stride 1, Valid:
///   out = (72-2)/1+1 = 71  →  depthwise = 1*1*1*71*71 = 5041 (>=5000 → parallel)
///                            pointwise = 1*1*71*71 = 5041 (>=5000 → parallel)
///
/// depthwise = all-ones 2x2 box filter; pointwise weight = 2.0; bias = 1.0.
/// With input ramp x[0,0,i,j] = i + j:
///   depthwise out[0,0,oh,ow] = 4*oh + 4*ow + 4   (single channel, dm=1)
///   final     out[0,0,oh,ow] = depthwise * 2.0 + 1.0 = 8*oh + 8*ow + 9
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

    // depthwise: all-ones 2x2; pointwise: scale by 2; bias: 1.0.
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

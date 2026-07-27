//! Integration tests for Conv3D, DepthwiseConv2D, and SeparableConv2D
//!
//! Tensors are channels-last (Keras): Conv3D takes \[batch, depth, height, width, channels\] and
//! the 2D layers take \[batch, height, width, channels\]. Kernels are \[k..., Cin, F\] and every
//! bias is a rank-1 \[F\] vector
//!
//! Expected values come from the mathematical definition or hand calculation, not from
//! recording layer output. Gradient checks live in tests/neural_network/gradient_check.rs

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array4, Array5};
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use crate::common::assert_allclose;

// Conv3D - constructor validation

/// Each invalid constructor argument must be rejected with the matching error variant.
/// Rows fold the former per-argument tests:
/// - filters=0                -> InvalidParameter (was conv3d_new_rejects_zero_filters)
/// - zero in kernel tuple     -> InvalidParameter (was conv3d_new_rejects_zero_kernel_dimension)
/// - zero in stride tuple     -> InvalidParameter (was conv3d_new_rejects_zero_stride)
/// - 4D (non-5D) input_shape  -> InvalidInput     (was conv3d_new_rejects_non_5d_input_shape)
/// - zero input dimension     -> InvalidInput     (was conv3d_new_rejects_zero_input_dimension)
#[test]
fn conv3d_new_rejects_invalid_args() {
    enum Want {
        Param,
        Input,
    }
    // (label, filters, kernel, input_shape, stride, expected error)
    type Case = (
        &'static str,
        usize,
        (usize, usize, usize),
        Vec<usize>,
        (usize, usize, usize),
        Want,
    );
    let cases: [Case; 5] = [
        (
            "zero filters",
            0,
            (2, 2, 2),
            vec![1, 4, 4, 4, 1],
            (1, 1, 1),
            Want::Param,
        ),
        (
            "zero kernel dimension",
            2,
            (2, 0, 2),
            vec![1, 4, 4, 4, 1],
            (1, 1, 1),
            Want::Param,
        ),
        (
            "zero stride",
            2,
            (2, 2, 2),
            vec![1, 4, 4, 4, 1],
            (1, 0, 1),
            Want::Param,
        ),
        (
            "non-5D input_shape",
            2,
            (2, 2, 2),
            vec![1, 4, 4, 4],
            (1, 1, 1),
            Want::Input,
        ),
        (
            "zero input dimension",
            2,
            (2, 2, 2),
            vec![1, 4, 4, 4, 0],
            (1, 1, 1),
            Want::Input,
        ),
    ];
    for (label, filters, kernel, input_shape, stride, want) in cases {
        let err = Conv3D::new(filters, kernel, input_shape, stride, Linear::new()).unwrap_err();
        match want {
            Want::Param => assert!(
                matches!(err, Error::InvalidParameter { .. }),
                "[{label}] expected InvalidParameter, got {err:?}"
            ),
            Want::Input => assert!(
                matches!(err, Error::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            ),
        }
    }
}

// Conv3D - forward output shape

/// Valid padding forward output shape is [1, 3, 3, 3, 2] for input [1,4,4,4,1], kernel (2,2,2)
#[test]
fn conv3d_forward_output_shape_valid_padding() {
    let mut conv =
        Conv3D::new(2, (2, 2, 2), vec![1, 4, 4, 4, 1], (1, 1, 1), Linear::new()).unwrap();
    let x = Array::ones((1_usize, 4, 4, 4, 1)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 3, 3, 3, 2],
        "Conv3D Valid output shape mismatch"
    );
}

/// Same padding forward output shape equals ceil(input / stride): [1, 4, 4, 4, 2] here
#[test]
fn conv3d_forward_output_shape_same_padding() {
    let mut conv = Conv3D::new(2, (3, 3, 3), vec![1, 4, 4, 4, 1], (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::ones((1_usize, 4, 4, 4, 1)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 4, 4, 4, 2],
        "Conv3D Same output shape mismatch"
    );
}

/// Stride-2 Valid forward output shape is [2, 2, 2, 2, 1] for input [2,5,5,5,1], kernel (3,3,3)
#[test]
fn conv3d_forward_output_shape_stride2_valid() {
    let mut conv =
        Conv3D::new(1, (3, 3, 3), vec![2, 5, 5, 5, 1], (2, 2, 2), Linear::new()).unwrap();
    let x = Array::ones((2_usize, 5, 5, 5, 1)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[2, 2, 2, 2, 1],
        "Conv3D stride-2 Valid output shape mismatch"
    );
}

// Conv3D - known-weight forward value

/// All-ones 2x2x2 kernel sums the 8-element window: forward output matches the hand calculation
///
/// With one channel, `x[0, d, h, w, 0] = 0.05 * (9d + 3h + w) - 0.40`. The window at output
/// `(od, oh, ow)` covers `d in {od, od+1}`, `h in {oh, oh+1}`, `w in {ow, ow+1}`, so the sum of
/// its flat indices is `72*od + 24*oh + 8*ow + 52` and the output is `0.05 * S - 8 * 0.40`
#[test]
fn conv3d_known_weight_forward_values() {
    let mut conv =
        Conv3D::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();

    // All weights 1, bias 0. Weight layout is [kd, kh, kw, Cin, F]
    let w = Array5::ones((2_usize, 2, 2, 1, 1));
    let b = Array1::zeros(1);
    conv.set_weights(w, b).unwrap();

    // Input: x[i] = 0.05*i - 0.40 for i = 0..27
    let input_data: Vec<f32> = (0..27).map(|i| 0.05 * i as f32 - 0.40).collect();
    let x = Array::from_shape_vec((1_usize, 3, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2, 1]);

    // (od,oh,ow) = (0,0,0): S = 52 -> 0.05*52 - 3.20 = -0.60
    assert_abs_diff_eq!(out[[0, 0, 0, 0, 0]], -0.60_f32, epsilon = 1e-5);
    // (0,0,1): S = 60 -> 0.05*60 - 3.20 = -0.20
    assert_abs_diff_eq!(out[[0, 0, 0, 1, 0]], -0.20_f32, epsilon = 1e-5);
    // (0,1,0): S = 76 -> 0.05*76 - 3.20 = 0.60
    assert_abs_diff_eq!(out[[0, 0, 1, 0, 0]], 0.60_f32, epsilon = 1e-5);
    // (1,1,1): S = 156 -> 0.05*156 - 3.20 = 4.60
    assert_abs_diff_eq!(out[[0, 1, 1, 1, 0]], 4.60_f32, epsilon = 1e-5);
}

// Conv3D - predict() == forward() in eval mode

/// predict() returns identical values to forward() (Conv3D has no train/eval difference)
#[test]
fn conv3d_predict_equals_forward() {
    let mut conv =
        Conv3D::new(2, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 3, 3, 3, 1),
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
    let mut conv =
        Conv3D::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 2, 2, 2, 1)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// A non-5D input to forward() must return InvalidInput
#[test]
fn conv3d_forward_rejects_non_5d_input() {
    let mut conv =
        Conv3D::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();
    let x_4d = Array::ones((1_usize, 3, 3, 1)).into_dyn(); // 4D
    let err = conv.forward(&x_4d).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// Constructing with a declared input spatial dim smaller than the kernel is rejected (parity
/// with Conv1D/Conv2D, which already checked this)
#[test]
fn conv3d_new_rejects_input_smaller_than_kernel() {
    let err = Conv3D::new(
        1,
        (3, 3, 3),
        vec![1, 2, 4, 4, 1], // depth = 2 < kernel depth 3
        (1, 1, 1),
        Linear::new(),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// A runtime input smaller than the kernel under Valid padding returns an error instead of
/// panicking on a usize underflow in the geometry
#[test]
fn conv3d_forward_rejects_input_smaller_than_kernel() {
    let mut conv = Conv3D::new(
        1,
        (3, 3, 3),
        vec![1, 5, 5, 5, 1], // declared shape is valid (>= kernel)
        (1, 1, 1),
        Linear::new(),
    )
    .unwrap();
    // Feed a genuinely smaller tensor at runtime: depth 2 < kernel depth 3
    let x_small = Array::ones((1_usize, 2, 5, 5, 1)).into_dyn();
    let err = conv.forward(&x_small).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// set_weights with wrong shape must return NeuralNetwork(WeightShape)
#[test]
fn conv3d_set_weights_shape_mismatch_errors() {
    let mut conv =
        Conv3D::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();
    // Layer expects weights [2,2,2,1,1] ([kd,kh,kw,Cin,F]); supply 2 filters instead of 1
    let wrong_w = Array5::zeros((2_usize, 2, 2, 1, 2));
    let b = Array1::zeros(1);
    let err = conv.set_weights(wrong_w, b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// DepthwiseConv2D - constructor validation

/// Each invalid constructor argument must be rejected with the matching error variant.
///
/// `DepthwiseConv2D::new` no longer takes a `filters` argument (the output channel count is
/// `channels * depth_multiplier`), so the former filters rows are gone; what remains is the
/// kernel/stride/input-shape validation.
#[test]
fn depthwise_conv2d_new_rejects_invalid_args() {
    enum Want {
        Param,
        Input,
    }
    // (label, kernel_size, input_shape, strides, expected error)
    type Case = (
        &'static str,
        (usize, usize),
        Vec<usize>,
        (usize, usize),
        Want,
    );
    let cases: [Case; 5] = [
        (
            "kernel_size.0=0",
            (0, 2),
            vec![1, 4, 4, 2],
            (1, 1),
            Want::Param,
        ),
        ("strides.1=0", (2, 2), vec![1, 4, 4, 2], (1, 0), Want::Param),
        (
            "non-4D input_shape",
            (2, 2),
            vec![1, 4, 4],
            (1, 1),
            Want::Input,
        ),
        (
            "zero channels",
            (2, 2),
            vec![1, 4, 4, 0],
            (1, 1),
            Want::Input,
        ),
        (
            "input smaller than kernel",
            (3, 3),
            vec![1, 2, 4, 2],
            (1, 1),
            Want::Input,
        ),
    ];
    for (label, kernel_size, input_shape, strides, want) in cases {
        let err =
            DepthwiseConv2D::new(kernel_size, input_shape, strides, Linear::new()).unwrap_err();
        match want {
            Want::Param => assert!(
                matches!(err, Error::InvalidParameter { .. }),
                "[{label}] expected InvalidParameter, got {err:?}"
            ),
            Want::Input => assert!(
                matches!(err, Error::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            ),
        }
    }
}

/// `with_depth_multiplier(0)` is rejected with InvalidParameter
#[test]
fn depthwise_conv2d_with_depth_multiplier_rejects_zero() {
    let err = DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Linear::new())
        .unwrap()
        .with_depth_multiplier(0)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A runtime input whose channel count differs from the declared one returns DimensionMismatch
/// (previously an `assert_eq!` panic)
#[test]
fn depthwise_conv2d_forward_rejects_wrong_channels() {
    let mut conv = DepthwiseConv2D::new(
        (2, 2),
        vec![1, 4, 4, 2], // declared with 2 channels
        (1, 1),
        Linear::new(),
    )
    .unwrap();
    // Feed a tensor with 3 channels instead of 2
    let x = Array::ones((1_usize, 4, 4, 3)).into_dyn();
    let err = conv.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

// DepthwiseConv2D - channel independence (the core correctness invariant)

/// Each channel convolves only its own input: all-ones kernel gives 4.0, zero kernel gives 0.0
#[test]
fn depthwise_conv2d_channel_independence() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 3, 3, 2], (1, 1), Linear::new()).unwrap();

    // Weight layout is [kh, kw, channels, depth_multiplier], dm = 1 here
    // Channel 0 kernel = all 1s; channel 1 kernel stays zero
    let mut w = Array4::<f32>::zeros((2, 2, 2, 1));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 1, 0, 0]] = 1.0;
    w[[1, 0, 0, 0]] = 1.0;
    w[[1, 1, 0, 0]] = 1.0;
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input: channel 0 = 1s, channel 1 = 2s
    let mut input = Array::zeros((1_usize, 3, 3, 2));
    for h in 0..3 {
        for ww in 0..3 {
            input[[0, h, ww, 0]] = 1.0_f32;
            input[[0, h, ww, 1]] = 2.0_f32;
        }
    }
    let x = input.into_dyn();

    let out = conv.forward(&x).unwrap();

    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Channel 0: every 2x2 window of all-1 input summed by all-1 kernel = 4.0
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, oh, ow, 0]], 4.0_f32, epsilon = 1e-6,);
        }
    }

    // Channel 1: zero kernel -> 0.0 everywhere, regardless of non-zero input
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out[[0, oh, ow, 1]], 0.0_f32, epsilon = 1e-6,);
        }
    }
}

/// Channel 0 output does not change when channel 1 input changes (no cross-channel bleed)
#[test]
fn depthwise_conv2d_cross_channel_no_bleed() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 3, 3, 2], (1, 1), Linear::new()).unwrap();

    // Channel 0 kernel = [[1,0],[0,0]] (only top-left); channel 1 kernel = all zeros
    let mut w = Array4::<f32>::zeros((2, 2, 2, 1));
    w[[0, 0, 0, 0]] = 1.0;
    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(w, bias).unwrap();

    // Input A: channel 0 = 1s everywhere, channel 1 = 0s
    let mut inp_a = Array::zeros((1_usize, 3, 3, 2));
    for h in 0..3 {
        for ww in 0..3 {
            inp_a[[0, h, ww, 0]] = 1.0_f32;
        }
    }
    let out_a = conv.forward(&inp_a.into_dyn()).unwrap();

    // Input B: channel 0 = 1s everywhere, channel 1 = 999s
    let mut inp_b = Array::zeros((1_usize, 3, 3, 2));
    for h in 0..3 {
        for ww in 0..3 {
            inp_b[[0, h, ww, 0]] = 1.0_f32;
            inp_b[[0, h, ww, 1]] = 999.0_f32;
        }
    }
    let out_b = conv.forward(&inp_b.into_dyn()).unwrap();

    // Channel 0 output must be identical in both cases
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_a[[0, oh, ow, 0]], out_b[[0, oh, ow, 0]], epsilon = 1e-6,);
        }
    }

    // Channel 0: top-left kernel picks up x[oh,ow] = 1.0 for all positions
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, oh, ow, 0]], 1.0_f32, epsilon = 1e-6);
        }
    }

    // Channel 1: zero kernel -> 0.0 everywhere even though input was 999
    for oh in 0..2 {
        for ow in 0..2 {
            assert_abs_diff_eq!(out_b[[0, oh, ow, 1]], 0.0_f32, epsilon = 1e-6);
        }
    }
}

// DepthwiseConv2D - known-weight forward values

/// Single-channel 2x2 kernel [[1,2],[3,4]] with bias 0.5 over input 1..=9 matches hand calculation
#[test]
fn depthwise_conv2d_known_weight_single_channel() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();

    // Kernel: [[1,2],[3,4]] laid out as [kh, kw, channels=1, dm=1]; bias 0.5
    let mut w = Array4::<f32>::zeros((2, 2, 1, 1));
    w[[0, 0, 0, 0]] = 1.0;
    w[[0, 1, 0, 0]] = 2.0;
    w[[1, 0, 0, 0]] = 3.0;
    w[[1, 1, 0, 0]] = 4.0;
    let bias = Array1::from_vec(vec![0.5_f32]);
    conv.set_weights(w, bias).unwrap();

    // Input: 1..=9 in row-major order
    let x = Array::from_shape_vec(
        (1_usize, 3, 3, 1),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);

    // 1*1 + 2*2 + 4*3 + 5*4 = 37, + 0.5
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 37.5_f32, epsilon = 1e-5);
    // 2*1 + 3*2 + 5*3 + 6*4 = 47, + 0.5
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 47.5_f32, epsilon = 1e-5);
    // 4*1 + 5*2 + 7*3 + 8*4 = 67, + 0.5
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 67.5_f32, epsilon = 1e-5);
    // 5*1 + 6*2 + 8*3 + 9*4 = 77, + 0.5
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 77.5_f32, epsilon = 1e-5);
}

// DepthwiseConv2D - predict() == forward() in eval mode

#[test]
fn depthwise_conv2d_predict_equals_forward() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Linear::new()).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 4, 4, 2),
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
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong weight shape must return NeuralNetwork(WeightShape)
#[test]
fn depthwise_conv2d_set_weights_shape_mismatch_errors() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 4, 4, 2], (1, 1), Linear::new()).unwrap();
    // Layer expects weights [2,2,2,1] ([kh,kw,channels,dm]); supply 3 channels
    let wrong_w = Array4::<f32>::zeros((2, 2, 3, 1));
    let bias = Array1::<f32>::zeros(2);
    let err = conv.set_weights(wrong_w, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// SeparableConv2D - constructor validation

/// Each invalid constructor argument must be rejected with the matching error variant.
/// Rows fold the former per-argument tests:
/// - filters=0            -> InvalidParameter (was separable_conv2d_new_rejects_zero_filters)
/// - depth_multiplier=0   -> InvalidParameter (was separable_conv2d_new_rejects_zero_depth_multiplier)
/// - zero kernel dim      -> InvalidParameter (was separable_conv2d_new_rejects_zero_kernel)
/// - non-4D input_shape   -> InvalidInput     (was separable_conv2d_new_rejects_non_4d_input_shape)
#[test]
fn separable_conv2d_new_rejects_invalid_args() {
    enum Want {
        Param,
        Input,
    }
    // (label, filters, kernel, input_shape, depth_multiplier, expected error)
    type Case = (&'static str, usize, (usize, usize), Vec<usize>, usize, Want);
    let cases: [Case; 4] = [
        ("filters=0", 0, (2, 2), vec![1, 4, 4, 2], 1, Want::Param),
        (
            "depth_multiplier=0",
            2,
            (2, 2),
            vec![1, 4, 4, 2],
            0,
            Want::Param,
        ),
        (
            "kernel_size.1=0",
            2,
            (2, 0),
            vec![1, 4, 4, 2],
            1,
            Want::Param,
        ),
        ("3D input_shape", 2, (2, 2), vec![1, 4, 4], 1, Want::Input),
    ];
    for (label, filters, kernel, input_shape, depth_multiplier, want) in cases {
        let err = SeparableConv2D::new(
            filters,
            kernel,
            input_shape,
            (1, 1),
            depth_multiplier,
            Linear::new(),
        )
        .unwrap_err();
        match want {
            Want::Param => assert!(
                matches!(err, Error::InvalidParameter { .. }),
                "[{label}] expected InvalidParameter, got {err:?}"
            ),
            Want::Input => assert!(
                matches!(err, Error::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            ),
        }
    }
}

// SeparableConv2D - depth_multiplier behavior and output shape

/// depth_multiplier=1, filters=1, 1 input channel: forward output shape is [1, 2, 2, 1]
#[test]
fn separable_conv2d_output_shape_dm1() {
    let mut conv =
        SeparableConv2D::new(1, (2, 2), vec![1, 3, 3, 1], (1, 1), 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 3, 3, 1)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 2, 1],
        "SeparableConv2D dm=1 shape mismatch"
    );
}

/// depth_multiplier=2, 2 input channels, 4 output filters: forward output shape is [1, 2, 2, 4]
#[test]
fn separable_conv2d_output_shape_dm2() {
    let mut conv =
        SeparableConv2D::new(4, (2, 2), vec![1, 3, 3, 2], (1, 1), 2, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 2, 4],
        "SeparableConv2D dm=2 output shape mismatch"
    );
}

// SeparableConv2D - identity depthwise + identity pointwise reproduces input

/// Identity 1x1 depthwise and pointwise kernels with zero bias reproduce the input exactly
#[test]
fn separable_conv2d_identity_reproduces_input() {
    let mut conv =
        SeparableConv2D::new(1, (1, 1), vec![1, 3, 3, 1], (1, 1), 1, Linear::new()).unwrap();

    // depthwise_weights: [kh=1, kw=1, channels=1, dm=1] -> 1.0
    let dw = Array4::<f32>::ones((1, 1, 1, 1));
    // pointwise_weights: [1, 1, channels*dm=1, filters=1] -> 1.0
    let pw = Array4::<f32>::ones((1, 1, 1, 1));
    let bias = Array1::<f32>::zeros(1);
    conv.set_weights(dw, pw, bias).unwrap();

    let input_data: Vec<f32> = (0..9).map(|i| i as f32 * 0.5 - 2.0).collect();
    let x = Array::from_shape_vec((1_usize, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 1]);

    // Every output value must equal the corresponding input value
    for h in 0..3 {
        for w in 0..3 {
            assert_abs_diff_eq!(out[[0, h, w, 0]], x[[0, h, w, 0]], epsilon = 1e-6);
        }
    }
}

// SeparableConv2D - known-weight forward value

/// Depthwise [[1,0],[0,1]] then pointwise scale-by-2 plus bias 1 over input 1..=9 matches by hand
#[test]
fn separable_conv2d_known_weight_forward_values() {
    let mut conv =
        SeparableConv2D::new(1, (2, 2), vec![1, 3, 3, 1], (1, 1), 1, Linear::new()).unwrap();

    // depthwise [kh, kw, C, dm]: [[1,0],[0,1]] picks up (i,j) and (i+1,j+1)
    let mut dw = Array4::<f32>::zeros((2, 2, 1, 1));
    dw[[0, 0, 0, 0]] = 1.0;
    dw[[1, 1, 0, 0]] = 1.0;

    // pointwise [1, 1, C*dm, F]: scale by 2, bias 1.0
    let mut pw = Array4::<f32>::zeros((1, 1, 1, 1));
    pw[[0, 0, 0, 0]] = 2.0;

    let bias = Array1::from_elem(1, 1.0_f32);
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 3, 3, 1),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 1]);

    // depthwise (0,0) = 1 + 5 = 6 -> 2*6 + 1
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], 13.0_f32, epsilon = 1e-5);
    // depthwise (0,1) = 2 + 6 = 8 -> 2*8 + 1
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], 17.0_f32, epsilon = 1e-5);
    // depthwise (1,0) = 4 + 8 = 12 -> 2*12 + 1
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], 25.0_f32, epsilon = 1e-5);
    // depthwise (1,1) = 5 + 9 = 14 -> 2*14 + 1
    assert_abs_diff_eq!(out[[0, 1, 1, 0]], 29.0_f32, epsilon = 1e-5);
}

/// `Same` padding zero-pads the depthwise stage: output is the zero-padded 3x3 box filter over 1..=9
/// (regression - the old boundary-clipping code returned the unpadded 3x3 sum 45 at [0,0])
#[test]
fn separable_conv2d_same_padding_zero_pads_depthwise() {
    let mut conv = SeparableConv2D::new(1, (3, 3), vec![1, 3, 3, 1], (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);

    // depthwise: 3x3 all-ones box filter; pointwise: identity (scale by 1); bias 0
    let dw = Array4::<f32>::from_elem((3, 3, 1, 1), 1.0);
    let pw = Array4::<f32>::from_elem((1, 1, 1, 1), 1.0);
    let bias = Array1::<f32>::zeros(1);
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 3, 3, 1),
        vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap()
    .into_dyn();

    let out = conv.forward(&x).unwrap();
    // `Same` keeps the spatial shape
    assert_eq!(out.shape(), &[1, 3, 3, 1]);

    // pad_total = (3-1)*1 + 3 - 3 = 2, so one zero row/column on each edge; each entry is the
    // sum of the in-bounds members of its 3x3 neighbourhood over [[1,2,3],[4,5,6],[7,8,9]]
    let expected = [
        [12.0_f32, 21.0, 16.0],
        [27.0, 45.0, 33.0],
        [24.0, 39.0, 28.0],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert_abs_diff_eq!(out[[0, i, j, 0]], want, epsilon = 1e-4);
        }
    }
}

// SeparableConv2D - predict() == forward() in eval mode

#[test]
fn separable_conv2d_predict_equals_forward() {
    let mut conv =
        SeparableConv2D::new(2, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Linear::new()).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 4, 4, 2),
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
    let mut conv =
        SeparableConv2D::new(2, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let err = conv.backward(&grad).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong depthwise shape must return NeuralNetwork(WeightShape)
#[test]
fn separable_conv2d_set_weights_shape_mismatch_errors() {
    let mut conv =
        SeparableConv2D::new(2, (2, 2), vec![1, 4, 4, 2], (1, 1), 1, Linear::new()).unwrap();
    // depthwise_weights should be [kh=2, kw=2, channels=2, dm=1]; supply wrong dm=3
    let bad_dw = Array4::<f32>::zeros((2, 2, 2, 3));
    let ok_pw = Array4::<f32>::zeros((1, 1, 2, 2));
    let ok_b = Array1::<f32>::zeros(2);
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
    let mut conv =
        SeparableConv2D::new(2, (1, 1), vec![1, 2, 2, 1], (1, 1), 2, Linear::new()).unwrap();

    // depthwise_weights shape [kh=1, kw=1, channels=1, dm=2]
    let mut dw = Array4::<f32>::zeros((1, 1, 1, 2));
    dw[[0, 0, 0, 0]] = 1.0; // dm=0: scale by 1
    dw[[0, 0, 0, 1]] = 2.0; // dm=1: scale by 2

    // pointwise_weights [1, 1, channels*dm=2, filters=2]: filter 0 selects intermediate channel 0,
    // filter 1 selects intermediate channel 1
    let mut pw = Array4::<f32>::zeros((1, 1, 2, 2));
    pw[[0, 0, 0, 0]] = 1.0;
    pw[[0, 0, 1, 1]] = 1.0;

    let bias = Array1::<f32>::zeros(2);
    conv.set_weights(dw, pw, bias).unwrap();

    let x = Array::from_shape_vec((1_usize, 2, 2, 1), vec![1.0_f32, 2., 3., 4.])
        .unwrap()
        .into_dyn();

    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 2, 2]);

    // Filter 0: should reproduce input (dm=0, scale=1, pw=1)
    let expected_f0 = [[1.0_f32, 2.], [3., 4.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, h, w, 0]], expected_f0[h][w], epsilon = 1e-5);
        }
    }

    // Filter 1: should be 2x input (dm=1, scale=2, pw=1)
    let expected_f1 = [[2.0_f32, 4.], [6., 8.]];
    for h in 0..2 {
        for w in 0..2 {
            assert_abs_diff_eq!(out[[0, h, w, 1]], expected_f1[h][w], epsilon = 1e-5);
        }
    }
}

// DepthwiseConv2D - large-input windowed sums

/// All-ones 2x2 kernel over a [1,40,40,1] ramp: every window sums to 4*oh + 4*ow + 4
#[test]
fn depthwise_conv2d_large_input_windowed_sums() {
    let mut conv = DepthwiseConv2D::new((2, 2), vec![1, 40, 40, 1], (1, 1), Linear::new()).unwrap();
    // All-ones 2x2 kernel ([kh, kw, channels, dm]), zero bias
    let w = Array4::<f32>::from_elem((2, 2, 1, 1), 1.0);
    let bias = Array1::<f32>::zeros(1);
    conv.set_weights(w, bias).unwrap();

    // x[0,i,j,0] = i + j
    let mut input = Array::<f32, _>::zeros((1, 40, 40, 1));
    for i in 0..40 {
        for j in 0..40 {
            input[[0, i, j, 0]] = (i + j) as f32;
        }
    }
    let out = conv.forward(&input.into_dyn()).unwrap();

    assert_eq!(out.shape(), &[1, 39, 39, 1]);

    // (i+j) + (i+j+1) + (i+1+j) + (i+1+j+1) = 4*(i+j) + 4
    let expect = |oh: usize, ow: usize| (4 * oh + 4 * ow + 4) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 4
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(0, 1), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], expect(1, 0), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 38, 38, 0]], expect(38, 38), epsilon = 1e-4); // 308
}

// SeparableConv2D - large-input windowed sums

/// A [1,72,72,1] ramp through an all-ones 2x2 depthwise then a scale-by-2 pointwise plus bias 1:
/// final out = 2*(4*oh + 4*ow + 4) + 1 = 8*oh + 8*ow + 9
#[test]
fn separable_conv2d_large_input_windowed_sums() {
    let mut conv =
        SeparableConv2D::new(1, (2, 2), vec![1, 72, 72, 1], (1, 1), 1, Linear::new()).unwrap();

    // depthwise: all-ones 2x2; pointwise: scale by 2; bias 1.0
    let dw = Array4::<f32>::from_elem((2, 2, 1, 1), 1.0);
    let pw = Array4::<f32>::from_elem((1, 1, 1, 1), 2.0);
    let bias = Array1::<f32>::from_elem(1, 1.0);
    conv.set_weights(dw, pw, bias).unwrap();

    // x[0,i,j,0] = i + j
    let mut input = Array::<f32, _>::zeros((1, 72, 72, 1));
    for i in 0..72 {
        for j in 0..72 {
            input[[0, i, j, 0]] = (i + j) as f32;
        }
    }
    let out = conv.forward(&input.into_dyn()).unwrap();

    assert_eq!(out.shape(), &[1, 71, 71, 1]);

    // final out[0,oh,ow,0] = 8*oh + 8*ow + 9
    let expect = |oh: usize, ow: usize| (8 * oh + 8 * ow + 9) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 9
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(0, 1), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], expect(1, 0), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 70, 70, 0]], expect(70, 70), epsilon = 1e-4); // 1129
}

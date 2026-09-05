//! Integration tests for Conv3D, DepthwiseConv2D, and SeparableConv2D.
//!
//! Tensors are channels-last (Keras): Conv3D takes \[batch, depth, height, width, channels\] and
//! the 2D layers take \[batch, height, width, channels\]. Kernels are \[k..., Cin, F\] and every
//! bias is a rank-1 \[F\] vector.
//!
//! Expected values come from the mathematical definition or hand calculation, not from
//! recording layer output. Gradient checks live in tests/neural_network/gradient_check.rs.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array4, Array5};
use rustyml::neural_network::Ctx;
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::traits::{Layer, ParamId, UnaryLayer};
use rustyml::{error::Error, neural_network::NnError};

use crate::common::assert_allclose;

// Conv3D - constructor validation

/// Each invalid constructor argument returns the matching error: InvalidParameter for a zero
/// filter count, kernel dimension, or stride, and InvalidInput for a bad input_shape.
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
        let err = match want {
            Want::Param => Conv3D::new(filters, kernel, stride, Linear::new()).unwrap_err(),
            Want::Input => {
                let mut layer = Conv3D::new(filters, kernel, stride, Linear::new()).unwrap();
                layer.build(&Shape::known(&input_shape)).unwrap_err()
            }
        };
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
    let mut conv = Conv3D::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x = Array::ones((1_usize, 4, 4, 4, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 3, 3, 3, 2],
        "Conv3D Valid output shape mismatch"
    );
}

/// Same padding forward output shape equals ceil(input / stride): [1, 4, 4, 4, 2] here
#[test]
fn conv3d_forward_output_shape_same_padding() {
    let mut conv = Conv3D::new(2, (3, 3, 3), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let x = Array::ones((1_usize, 4, 4, 4, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 4, 4, 4, 2],
        "Conv3D Same output shape mismatch"
    );
}

/// Stride-2 Valid forward output shape is [2, 2, 2, 2, 1] for input [2,5,5,5,1], kernel (3,3,3)
#[test]
fn conv3d_forward_output_shape_stride2_valid() {
    let mut conv = Conv3D::new(1, (3, 3, 3), (2, 2, 2), Linear::new()).unwrap();
    let x = Array::ones((2_usize, 5, 5, 5, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(
        out.shape(),
        &[2, 2, 2, 2, 1],
        "Conv3D stride-2 Valid output shape mismatch"
    );
}

// Conv3D - known-weight forward value

/// All-ones 2x2x2 kernel sums the 8-element window. Forward output matches a hand calculation
/// for `x[0, d, h, w, 0] = 0.05 * (9d + 3h + w) - 0.40`.
#[test]
fn conv3d_known_weight_forward_values() {
    let mut conv = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 3, 1])).unwrap();

    // All weights 1, bias 0. Weight layout is [kd, kh, kw, Cin, F]
    let w = Array5::ones((2_usize, 2, 2, 1, 1));
    let b = Array1::zeros(1);
    conv.set_weights(w, b).unwrap();

    // Input: x[i] = 0.05*i - 0.40 for i = 0..27
    let input_data: Vec<f32> = (0..27).map(|i| 0.05 * i as f32 - 0.40).collect();
    let x = Array::from_shape_vec((1_usize, 3, 3, 3, 1), input_data)
        .unwrap()
        .into_dyn();

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();

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

// Conv3D - a forward pass with an inference context equals one with a training context

/// A forward pass with an inference context returns the same values as one with a training
/// context (Conv3D has no train/eval difference)
#[test]
fn conv3d_predict_equals_forward() {
    let mut conv = Conv3D::new(2, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 3, 1])).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 3, 3, 3, 1),
        (0..27).map(|v| 0.05 * v as f32 - 0.4).collect(),
    )
    .unwrap()
    .into_dyn();

    let mut train_ctx = Ctx::training();
    let fwd = conv.forward(&x, &mut train_ctx).unwrap();
    let mut infer_ctx = Ctx::inference();
    let pred = conv.forward(&x, &mut infer_ctx).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// Conv3D - error paths

/// Calling backward before forward must return NeuralNetwork(ForwardPassNotRun)
#[test]
fn conv3d_backward_before_forward_errors() {
    let conv = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 2, 2, 2, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.backward(&grad, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// A non-5D input to forward() must return InvalidInput
#[test]
fn conv3d_forward_rejects_non_5d_input() {
    let mut conv = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    let x_4d = Array::ones((1_usize, 3, 3, 1)).into_dyn(); // 4D
    let mut ctx = Ctx::training();
    let err = conv.forward_mut(&x_4d, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// A declared input spatial dim smaller than the kernel builds, and only a `Valid` forward pass
/// rejects it, matching Conv1D and Conv2D
///
/// `Same` padding supplies the missing cells, so Keras 3.15.1 accepts the same layer and returns
/// the input spatial size.
#[test]
fn conv3d_input_smaller_than_kernel_fails_only_under_valid() {
    // depth = 2 < kernel depth 3
    let layer = || Conv3D::new(1, (3, 3, 3), (1, 1, 1), Linear::new()).unwrap();
    let input = Array::ones((1_usize, 2, 4, 4, 1)).into_dyn();

    let mut ctx = Ctx::training();
    let err = layer().forward_mut(&input, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput under Valid, got {err:?}"
    );

    let mut same_ctx = Ctx::training();
    let output = layer()
        .with_padding(PaddingType::Same)
        .forward_mut(&input, &mut same_ctx)
        .unwrap();
    assert_eq!(output.shape(), &[1, 2, 4, 4, 1]);
}

/// A runtime input smaller than the kernel under Valid padding returns an error instead of
/// panicking on a usize underflow in the geometry
#[test]
fn conv3d_forward_rejects_input_smaller_than_kernel() {
    let mut conv = Conv3D::new(
        1,
        (3, 3, 3), // declared shape is valid (>= kernel)
        (1, 1, 1),
        Linear::new(),
    )
    .unwrap();
    // Feed a genuinely smaller tensor at runtime: depth 2 < kernel depth 3
    let x_small = Array::ones((1_usize, 2, 5, 5, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.forward_mut(&x_small, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// set_weights with wrong shape must return NeuralNetwork(WeightShape)
#[test]
fn conv3d_set_weights_shape_mismatch_errors() {
    let mut conv = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 3, 3, 3, 1])).unwrap();
    // Layer expects weights [2,2,2,1,1] ([kd,kh,kw,Cin,F]). This test supplies 2 filters
    // instead of 1.
    let wrong_w = Array5::zeros((2_usize, 2, 2, 1, 2));
    let b = Array1::zeros(1);
    let err = conv.set_weights(wrong_w, b).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// DepthwiseConv2D - constructor validation

/// Each invalid constructor argument returns the matching error: InvalidParameter for a zero
/// kernel or stride, and InvalidInput for a bad input_shape.
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
    let cases: [Case; 4] = [
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
    ];
    for (label, kernel_size, input_shape, strides, want) in cases {
        let err = match want {
            Want::Param => DepthwiseConv2D::new(kernel_size, strides, Linear::new()).unwrap_err(),
            Want::Input => {
                let mut layer = DepthwiseConv2D::new(kernel_size, strides, Linear::new()).unwrap();
                layer.build(&Shape::known(&input_shape)).unwrap_err()
            }
        };
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

/// `with_depth_multiplier(0)` returns InvalidParameter
#[test]
fn depthwise_conv2d_with_depth_multiplier_rejects_zero() {
    let err = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_depth_multiplier(0)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// `output_shape` reports `channels * depth_multiplier`, which is what forward emits
///
/// `output_shape` feeds `summary()`. The shared 2D calculator preserves the input channel count,
/// which suits a plain convolution but not a depthwise one.
#[test]
fn depthwise_conv2d_output_shape_reports_the_multiplied_channel_count() {
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_depth_multiplier(2)
        .unwrap();
    let x = Array::ones((1_usize, 4, 4, 3)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(out.shape(), &[1, 3, 3, 6]);
    assert_eq!(conv.output_shape(), "(None, 3, 3, 6)");
}

/// A runtime input whose channel count differs from the built shape returns InvalidInput,
/// instead of a panic.
///
/// The build step fixes the channel count, so a later mismatch is a build-shape disagreement
/// (`validate_built_input`) rather than the layer's own runtime check.
#[test]
fn depthwise_conv2d_forward_rejects_wrong_channels() {
    let mut conv = DepthwiseConv2D::new(
        (2, 2), // declared with 2 channels
        (1, 1),
        Linear::new(),
    )
    .unwrap();
    conv.build(&Shape::known(&[1, 4, 4, 2])).unwrap();
    // Feed a tensor with 3 channels instead of 2
    let x = Array::ones((1_usize, 4, 4, 3)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.forward(&x, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// DepthwiseConv2D - channel independence (the core correctness invariant)

/// Each channel convolves only its own input: all-ones kernel gives 4.0, zero kernel gives 0.0
#[test]
fn depthwise_conv2d_channel_independence() {
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 3, 3, 2])).unwrap();

    // Weight layout is [kh, kw, channels, depth_multiplier], dm = 1 here
    // Channel 0 kernel = all 1s. Channel 1 kernel stays zero.
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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();

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
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 2])).unwrap();

    // Channel 0 kernel = [[1,0],[0,0]] (only top-left). Channel 1 kernel = all zeros.
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
    let mut ctx_a = Ctx::training();
    let out_a = conv.forward(&inp_a.into_dyn(), &mut ctx_a).unwrap();

    // Input B: channel 0 = 1s everywhere, channel 1 = 999s
    let mut inp_b = Array::zeros((1_usize, 3, 3, 2));
    for h in 0..3 {
        for ww in 0..3 {
            inp_b[[0, h, ww, 0]] = 1.0_f32;
            inp_b[[0, h, ww, 1]] = 999.0_f32;
        }
    }
    let mut ctx_b = Ctx::training();
    let out_b = conv.forward(&inp_b.into_dyn(), &mut ctx_b).unwrap();

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
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 1])).unwrap();

    // Kernel [[1,2],[3,4]] is laid out as [kh, kw, channels=1, dm=1]. Bias is 0.5.
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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();
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

// DepthwiseConv2D - a forward pass with an inference context equals one with a training context

#[test]
fn depthwise_conv2d_predict_equals_forward() {
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 4, 4, 2])).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();

    let mut train_ctx = Ctx::training();
    let fwd = conv.forward(&x, &mut train_ctx).unwrap();
    let mut infer_ctx = Ctx::inference();
    let pred = conv.forward(&x, &mut infer_ctx).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// DepthwiseConv2D - error paths

/// backward before forward must return ForwardPassNotRun
#[test]
fn depthwise_conv2d_backward_before_forward_errors() {
    let conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.backward(&grad, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong weight shape must return NeuralNetwork(WeightShape)
#[test]
fn depthwise_conv2d_set_weights_shape_mismatch_errors() {
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 4, 4, 2])).unwrap();
    // Layer expects weights [2,2,2,1] ([kh,kw,channels,dm]). This test supplies 3 channels.
    let wrong_w = Array4::<f32>::zeros((2, 2, 3, 1));
    let bias = Array1::<f32>::zeros(2);
    let err = conv.set_weights(wrong_w, bias).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );
}

// SeparableConv2D - constructor validation

/// Each invalid constructor argument returns the matching error: InvalidParameter for a zero
/// filter count, depth multiplier, or kernel dimension, and InvalidInput for a bad input_shape.
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
        let err = match want {
            Want::Param => {
                SeparableConv2D::new(filters, kernel, (1, 1), depth_multiplier, Linear::new())
                    .unwrap_err()
            }
            Want::Input => {
                let mut layer =
                    SeparableConv2D::new(filters, kernel, (1, 1), depth_multiplier, Linear::new())
                        .unwrap();
                layer.build(&Shape::known(&input_shape)).unwrap_err()
            }
        };
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
    let mut conv = SeparableConv2D::new(1, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 3, 3, 1)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 2, 1],
        "SeparableConv2D dm=1 shape mismatch"
    );
}

/// depth_multiplier=2, 2 input channels, 4 output filters: forward output shape is [1, 2, 2, 4]
#[test]
fn separable_conv2d_output_shape_dm2() {
    let mut conv = SeparableConv2D::new(4, (2, 2), (1, 1), 2, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let mut ctx = Ctx::training();
    let out = conv.forward_mut(&x, &mut ctx).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 2, 2, 4],
        "SeparableConv2D dm=2 output shape mismatch"
    );
}

/// A runtime input whose channel count differs from the built shape returns InvalidInput,
/// instead of reading past the end of the depthwise kernel.
///
/// The build step fixes the channel count, so a later mismatch is a build-shape disagreement
/// (`validate_built_input`) rather than the layer's own runtime check.
#[test]
fn separable_conv2d_forward_rejects_wrong_channels() {
    let mut conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 4, 4, 2])).unwrap();
    let x = Array::ones((1_usize, 4, 4, 3)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.forward(&x, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
    let mut infer_ctx = Ctx::inference();
    let err = conv.forward(&x, &mut infer_ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput from predict, got {err:?}"
    );
}

// SeparableConv2D - identity depthwise + identity pointwise reproduces input

/// Identity 1x1 depthwise and pointwise kernels with zero bias reproduce the input exactly
#[test]
fn separable_conv2d_identity_reproduces_input() {
    let mut conv = SeparableConv2D::new(1, (1, 1), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 1])).unwrap();

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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();
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
    let mut conv = SeparableConv2D::new(1, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 3, 3, 1])).unwrap();

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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();
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

/// `Same` padding zero-pads the depthwise stage: output is the zero-padded 3x3 box filter
/// over the input 1..=9.
#[test]
fn separable_conv2d_same_padding_zero_pads_depthwise() {
    let mut conv = SeparableConv2D::new(1, (3, 3), (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    conv.build(&Shape::known(&[1_usize, 3, 3, 1])).unwrap();

    // Depthwise is a 3x3 all-ones box filter. Pointwise is identity (scale by 1). Bias is 0.
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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();
    // `Same` keeps the spatial shape
    assert_eq!(out.shape(), &[1, 3, 3, 1]);

    // pad_total = (3-1)*1 + 3 - 3 = 2, so 1 zero row/column sits on each edge. Each entry is
    // the sum of the in-bounds members of its 3x3 neighborhood over [[1,2,3],[4,5,6],[7,8,9]].
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

// SeparableConv2D - a forward pass with an inference context equals one with a training context

#[test]
fn separable_conv2d_predict_equals_forward() {
    let mut conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 4, 4, 2])).unwrap();

    let x = Array::from_shape_vec(
        (1_usize, 4, 4, 2),
        (0..32).map(|v| 0.05 * v as f32 - 0.7).collect(),
    )
    .unwrap()
    .into_dyn();

    let mut train_ctx = Ctx::training();
    let fwd = conv.forward(&x, &mut train_ctx).unwrap();
    let mut infer_ctx = Ctx::inference();
    let pred = conv.forward(&x, &mut infer_ctx).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// SeparableConv2D - error paths

/// backward before forward must return ForwardPassNotRun
#[test]
fn separable_conv2d_backward_before_forward_errors() {
    let conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 3, 3, 2)).into_dyn();
    let mut ctx = Ctx::training();
    let err = conv.backward(&grad, &mut ctx).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::ForwardPassNotRun(_))),
        "expected ForwardPassNotRun, got {err:?}"
    );
}

/// set_weights with wrong depthwise shape must return NeuralNetwork(WeightShape)
#[test]
fn separable_conv2d_set_weights_shape_mismatch_errors() {
    let mut conv = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 4, 4, 2])).unwrap();
    // depthwise_weights should be [kh=2, kw=2, channels=2, dm=1]. This test supplies dm=3.
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

/// depth_multiplier=2 yields 2 scaled copies. Filter 0 reproduces the input, filter 1 doubles it.
#[test]
fn separable_conv2d_depth_multiplier_2_forward_values() {
    let mut conv = SeparableConv2D::new(2, (1, 1), (1, 1), 2, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1_usize, 2, 2, 1])).unwrap();

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

    let mut ctx = Ctx::training();
    let out = conv.forward(&x, &mut ctx).unwrap();
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
    let mut conv = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 40, 40, 1])).unwrap();
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
    let mut ctx = Ctx::training();
    let out = conv.forward(&input.into_dyn(), &mut ctx).unwrap();

    assert_eq!(out.shape(), &[1, 39, 39, 1]);

    // (i+j) + (i+j+1) + (i+1+j) + (i+1+j+1) = 4*(i+j) + 4
    let expect = |oh: usize, ow: usize| (4 * oh + 4 * ow + 4) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 4
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(0, 1), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], expect(1, 0), epsilon = 1e-4); // 8
    assert_abs_diff_eq!(out[[0, 38, 38, 0]], expect(38, 38), epsilon = 1e-4); // 308
}

// SeparableConv2D - large-input windowed sums

/// A [1,72,72,1] ramp goes through an all-ones 2x2 depthwise, then a scale-by-2 pointwise plus
/// bias 1. The final output is 8*oh + 8*ow + 9.
#[test]
fn separable_conv2d_large_input_windowed_sums() {
    let mut conv = SeparableConv2D::new(1, (2, 2), (1, 1), 1, Linear::new()).unwrap();
    conv.build(&Shape::known(&[1, 72, 72, 1])).unwrap();

    // Depthwise is an all-ones 2x2 kernel. Pointwise scales by 2. Bias is 1.0.
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
    let mut ctx = Ctx::training();
    let out = conv.forward(&input.into_dyn(), &mut ctx).unwrap();

    assert_eq!(out.shape(), &[1, 71, 71, 1]);

    // final out[0,oh,ow,0] = 8*oh + 8*ow + 9
    let expect = |oh: usize, ow: usize| (8 * oh + 8 * ow + 9) as f32;
    assert_abs_diff_eq!(out[[0, 0, 0, 0]], expect(0, 0), epsilon = 1e-4); // 9
    assert_abs_diff_eq!(out[[0, 0, 1, 0]], expect(0, 1), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 1, 0, 0]], expect(1, 0), epsilon = 1e-4); // 17
    assert_abs_diff_eq!(out[[0, 70, 70, 0]], expect(70, 70), epsilon = 1e-4); // 1129
}

// Dilation

/// Each of the 3 spatial axes carries its own dilation
///
/// A `(1, 2, 3)` dilation spans 2 cells of depth, 3 of height, and 4 of width, so a 5x5x5 input
/// leaves a 4x3x2 output. Each input cell holds `d * 100 + h * 10 + w`, so a swapped axis shows
/// up in the value at once. The first window sums the 8 cells at depth 0 and 1, height 0 and 2,
/// and width 0 and 3.
#[test]
fn conv3d_dilation_is_per_axis() {
    let mut layer = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((1, 2, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 5, 5, 1])).unwrap();
    layer
        .set_weights(Array5::from_elem((2, 2, 2, 1, 1), 1.0f32), Array1::zeros(1))
        .unwrap();

    let values: Vec<f32> = (0..5)
        .flat_map(|d| (0..5).flat_map(move |h| (0..5).map(move |w| (d * 100 + h * 10 + w) as f32)))
        .collect();
    let input = Array::from_shape_vec((1, 5, 5, 5, 1), values)
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 4, 3, 2, 1], "keff is 2 by 3 by 4");
    // 0 + 3 + 20 + 23 + 100 + 103 + 120 + 123
    assert_abs_diff_eq!(output[[0, 0, 0, 0, 0]], 492.0f32, epsilon = 1e-6f32);
    // 1 step of depth adds 100 to all 8 terms
    assert_abs_diff_eq!(output[[0, 1, 0, 0, 0]], 1292.0f32, epsilon = 1e-6f32);
    // 1 step of height adds 10 to all 8 terms
    assert_abs_diff_eq!(output[[0, 0, 1, 0, 0]], 572.0f32, epsilon = 1e-6f32);
    // 1 step of width adds 1 to all 8 terms
    assert_abs_diff_eq!(output[[0, 0, 0, 1, 0]], 500.0f32, epsilon = 1e-6f32);
}

/// The stride-with-dilation rule reads the maximum over the axes, not 1 axis at a time
///
/// A stride above 1 on the depth axis and a dilation above 1 on the width axis carry no common
/// axis, and the rule still fires.
#[test]
fn conv3d_rejects_a_stride_and_a_dilation_above_one_on_different_axes() {
    let mixed = Conv3D::new(1, (2, 2, 2), (2, 1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((1, 1, 3));
    assert!(
        matches!(mixed, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter for stride 2 with dilation 3"
    );

    // The same dilation at unit strides is accepted
    assert!(
        Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new())
            .unwrap()
            .with_dilation_rate((1, 1, 3))
            .is_ok()
    );
}

/// The window advances by the stride and the taps sit `dilation` apart. The 2 factors are
/// independent
///
/// Output `ow` reads `ow * stride + tap * dilation`. The wrong form
/// `(ow * stride + tap) * dilation` agrees only when the stride equals the dilation, and a
/// depthwise layer accepts any pair. A stride of 2 with a dilation of 3 tells them apart: the
/// windows here read `(0, 3)`, `(2, 5)`, and `(4, 7)` of a height-1 row.
#[test]
fn depthwise_conv2d_keeps_the_stride_and_the_dilation_independent() {
    let mut layer = DepthwiseConv2D::new((1, 2), (1, 2), Linear::new())
        .unwrap()
        .with_dilation_rate((1, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 1, 8, 1])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((1, 2, 1, 1), vec![1.0f32, 10.0]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

    let input = Array::from_shape_vec(
        (1, 1, 8, 1),
        (1..=8).map(|v| v as f32).collect::<Vec<f32>>(),
    )
    .unwrap()
    .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 1, 3, 1]);
    // x[0] + 10 * x[3], x[2] + 10 * x[5], x[4] + 10 * x[7]
    let expected = Array::from_shape_vec((1, 1, 3, 1), vec![41.0f32, 63.0, 85.0])
        .unwrap()
        .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// A dilated `Same` pass keeps the input size and still splits its padding with the extra cell
/// at the end
///
/// The depthwise stage spans 4 cells of width at dilation 3, so it needs 3 pad cells. 1 goes in
/// front and 2 go behind. The first output therefore reads only the second tap, and the last 2
/// outputs read only the first. Swapping the 2 halves gives a different, and detectably wrong,
/// result.
#[test]
fn separable_conv2d_dilated_same_padding_splits_with_the_extra_cell_at_the_end() {
    let mut layer = SeparableConv2D::new(1, (1, 2), (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((1, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 1, 8, 1])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((1, 2, 1, 1), vec![1.0f32, 10.0]).unwrap(),
            Array4::from_elem((1, 1, 1, 1), 1.0f32),
            Array1::zeros(1),
        )
        .unwrap();

    let input = Array::from_shape_vec(
        (1, 1, 8, 1),
        (1..=8).map(|v| v as f32).collect::<Vec<f32>>(),
    )
    .unwrap()
    .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();

    assert_eq!(output.shape(), &[1, 1, 8, 1], "Same keeps the width");
    let expected = Array::from_shape_vec(
        (1, 1, 8, 1),
        vec![30.0f32, 41.0, 52.0, 63.0, 74.0, 85.0, 6.0, 7.0],
    )
    .unwrap()
    .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
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

/// A Conv3D whose dilated kernel is longer than every input axis matches Keras under `Same`
///
/// The dilation is (2, 3, 3) on a (2, 2, 2) kernel. The taps then span (3, 4, 4) cells against
/// an input of (2, 3, 3). Keras 3.15.1 pads the missing cells on all 3 axes.
#[test]
fn conv3d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((2, 3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 2, 3, 3, 1])).unwrap();
    layer
        .set_weights(
            Array5::from_shape_vec((2, 2, 2, 1, 1), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(1)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 2, 3, 3, 1), ramp_input(18))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 2, 3, 3, 1]);
    assert_flat_close(
        &flat(&output),
        &[
            -1.75, 0.5, 0.875, -0.75, -0.75, -0.625, 0.0, -0.375, -0.25, -0.25, -0.5, -0.625, 0.25,
            1.25, 0.875, -0.5, 0.125, -0.25,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 2, 3, 3, 1), ramp_upstream(18))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.28125, -0.46875, -0.0625, 0.09375, -0.09375, 0.1875, 0.09375, 0.03125, 0.0, 0.15625,
            -0.09375, 0.1875, 0.03125, 0.09375, -0.0625, -0.09375, 0.09375, 0.28125,
        ],
        "grad_input",
    );
    let kernel_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    assert_flat_close(
        &flat(kernel_grad),
        &[
            -1.625, -0.3125, -0.5625, 1.25, -0.75, -0.4375, 0.0625, -0.75,
        ],
        "grad_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[1.5], "grad_bias");
}

/// A DepthwiseConv2D whose dilated kernel is longer than both input axes matches Keras under
/// `Same`
#[test]
fn depthwise_conv2d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = DepthwiseConv2D::new((2, 2), (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 2])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 2, 1), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 3, 3, 2), ramp_input(18))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            0.875, -1.25, -0.375, 0.25, -0.125, 0.75, -0.25, 0.25, 1.25, 1.0, 0.5, 0.5, 0.125,
            0.25, -1.0, -0.5, -1.75, 1.25,
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
            -0.28125, -0.3125, 0.28125, 0.0625, 0.03125, 0.0, -0.46875, 0.1875, 0.09375, -0.0625,
            -0.03125, 0.0, 0.03125, 0.1875, 0.15625, -0.1875, -0.28125, 0.09375,
        ],
        "grad_input",
    );
    let kernel_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    assert_flat_close(
        &flat(kernel_grad),
        &[0.0, -1.6875, -0.1875, -0.3125, 0.25, -0.375, -0.5625, -0.25],
        "grad_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[0.875, 0.625], "grad_bias");
}

/// A SeparableConv2D whose dilated depthwise kernel is longer than both input axes matches
/// Keras under `Same`
#[test]
fn separable_conv2d_dilated_kernel_longer_than_the_input_matches_keras_under_same() {
    let mut layer = SeparableConv2D::new(2, (2, 2), (1, 1), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 2])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 2, 1), ramp_kernel(8)).unwrap(),
            Array4::from_shape_vec((1, 1, 2, 2), ramp_pointwise(4)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 3, 3, 2), ramp_input(18))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            -1.375, -1.0625, -0.125, 0.3125, -0.375, 0.4375, -0.25, 0.25, -1.75, -0.125, -1.0, 0.0,
            -0.625, 0.0625, 0.5, 0.25, 1.25, 1.5,
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
            0.515625, -0.15625, -0.328125, 0.03125, -0.015625, 0.0, 0.328125, 0.09375, -0.046875,
            -0.03125, 0.078125, 0.0, -0.078125, 0.09375, -0.109375, -0.09375, 0.328125, 0.046875,
        ],
        "grad_input",
    );
    let depthwise_grad = ctx
        .grads()
        .get(ParamId::new(0, "depthwise_kernel"))
        .unwrap();
    assert_flat_close(
        &flat(depthwise_grad),
        &[
            0.625, -0.84375, 0.46875, -0.15625, -0.0625, -0.1875, 0.65625, -0.125,
        ],
        "grad_depthwise_weights",
    );
    let pointwise_grad = ctx
        .grads()
        .get(ParamId::new(0, "pointwise_kernel"))
        .unwrap();
    assert_flat_close(
        &flat(pointwise_grad),
        &[-0.3125, 0.84375, 0.46875, 0.84375],
        "grad_pointwise_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[0.875, 0.625], "grad_bias");
}

/// Conv3D dilated gradients under `Valid` match Keras 3.15.1
///
/// A dilation of 2 on all 3 axes doubles the tap spacing, so a backward pass that ignores the
/// dilation credits the wrong input cells. The stride stays at 1, which differs from the
/// dilation.
#[test]
fn conv3d_dilated_gradients_match_keras_under_valid() {
    let mut layer = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new())
        .unwrap()
        .with_dilation_rate((2, 2, 2))
        .unwrap();
    layer.build(&Shape::known(&[1, 4, 4, 4, 1])).unwrap();
    layer
        .set_weights(
            Array5::from_shape_vec((2, 2, 2, 1, 1), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(1)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 4, 4, 4, 1), ramp_input(64))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2, 1]);
    assert_flat_close(
        &flat(&output),
        &[1.375, 2.125, 1.0, -2.75, -1.25, -1.625, -0.5, 2.5],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 2, 2, 2, 1), ramp_upstream(8))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            0.28125, 0.09375, 0.1875, 0.0625, -0.09375, -0.28125, -0.0625, -0.1875, 0.09375,
            0.03125, 0.0, 0.0, -0.03125, -0.09375, 0.0, 0.0, -0.46875, 0.28125, -0.3125, 0.1875,
            0.09375, -0.09375, 0.0625, -0.0625, -0.15625, 0.09375, 0.0, 0.0, 0.03125, -0.03125,
            0.0, 0.0, -0.09375, -0.03125, -0.1875, -0.0625, 0.03125, 0.09375, 0.0625, 0.1875,
            -0.28125, -0.09375, 0.28125, 0.09375, 0.09375, 0.28125, -0.09375, -0.28125, 0.15625,
            -0.09375, 0.3125, -0.1875, -0.03125, 0.03125, -0.0625, 0.0625, 0.46875, -0.28125,
            -0.46875, 0.28125, -0.09375, 0.09375, 0.09375, -0.09375,
        ],
        "grad_input",
    );
    let kernel_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    assert_flat_close(
        &flat(kernel_grad),
        &[
            1.375, 0.5, -0.4375, 3.1875, -1.375, -1.125, -0.9375, -1.8125,
        ],
        "grad_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[0.25], "grad_bias");
}

/// Conv3D dilated gradients under `Same` match Keras 3.15.1
///
/// `Same` padding and dilation both move the cell a tap reads, so the backward pass must apply
/// the 2 together on all 3 axes.
#[test]
fn conv3d_dilated_gradients_match_keras_under_same() {
    let mut layer = Conv3D::new(1, (2, 2, 2), (1, 1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((2, 2, 2))
        .unwrap();
    layer.build(&Shape::known(&[1, 3, 3, 3, 1])).unwrap();
    layer
        .set_weights(
            Array5::from_shape_vec((2, 2, 2, 1, 1), ramp_kernel(8)).unwrap(),
            Array1::from_vec(ramp_bias(1)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 3, 3, 3, 1), ramp_input(27))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 3, 1]);
    assert_flat_close(
        &flat(&output),
        &[
            -0.25, -1.0, -0.25, -2.125, -2.0, 0.5, -0.25, -0.125, -0.25, -0.25, -0.875, -0.25,
            -1.375, -0.25, 1.25, -0.25, 0.0, -0.25, -0.25, -0.125, -0.25, 0.5, 1.5, 0.5, -0.25,
            -0.125, -0.25,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 3, 3, 1), ramp_upstream(27))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            -0.28125, -0.53125, -0.1875, 0.1875, 0.125, 0.0625, -0.09375, -0.15625, 0.0, 0.0625,
            -0.125, 0.25, 0.0625, 0.28125, 0.34375, 0.4375, -0.65625, -0.46875, 0.09375, 0.21875,
            0.1875, -0.3125, -0.71875, 0.21875, 0.28125, 0.375, -0.28125,
        ],
        "grad_input",
    );
    let kernel_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    assert_flat_close(
        &flat(kernel_grad),
        &[-2.25, -0.75, 1.875, 1.125, -2.375, -1.75, 1.625, 2.5],
        "grad_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[2.625], "grad_bias");
}

/// DepthwiseConv2D gradients at a stride of 2 and a dilation of 3 match Keras 3.15.1
///
/// The stride and the dilation differ, the padding is `Same`, and the depth multiplier is 2.
/// A backward pass that swaps the stride for the dilation, or that drops the leading pad, fails
/// here.
#[test]
fn depthwise_conv2d_stride_and_dilation_gradients_match_keras() {
    let mut layer = DepthwiseConv2D::new((2, 2), (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_depth_multiplier(2)
        .unwrap()
        .with_dilation_rate((3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 5, 2])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 2, 2), ramp_kernel(16)).unwrap(),
            Array1::from_vec(ramp_bias(4)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 5, 2), ramp_input(50))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 4]);
    assert_flat_close(
        &flat(&output),
        &[
            0.25, 1.0, -0.375, 0.5, -1.0, -0.875, 1.5, 1.875, -1.25, -0.25, 0.75, 0.75, 0.875, 2.0,
            3.0, 1.5, -1.125, -1.125, -0.375, 2.75, -0.625, -0.125, 0.25, 1.0, 0.125, 1.0, 2.25,
            -0.25, -0.875, -0.5, 0.125, 1.625, 1.25, 1.25, 1.125, 1.25,
        ],
        "output",
    );

    let upstream = Array::from_shape_vec((1, 3, 3, 4), ramp_upstream(36))
        .unwrap()
        .into_dyn();
    let grad_input = layer.backward(&upstream, &mut ctx).unwrap();
    assert_flat_close(
        &flat(&grad_input),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03125, -0.09375, 0.21875,
            0.75, 0.34375, -0.03125, 0.03125, -0.1875, 0.0, 0.0, -0.21875, 0.03125, -0.28125,
            -0.28125, -0.34375, -0.03125, 0.03125, 0.03125, 0.0, 0.0, -0.59375, 0.09375, -0.03125,
            -0.1875, -0.28125, -0.15625, 0.40625, -0.1875, 0.0, 0.0, 0.03125, 0.15625, 0.34375,
            -0.28125, 0.21875, 0.09375, 0.03125, -0.59375,
        ],
        "grad_input",
    );
    let kernel_grad = ctx.grads().get(ParamId::new(0, "kernel")).unwrap();
    assert_flat_close(
        &flat(kernel_grad),
        &[
            -0.5625, -0.6875, -1.0625, 1.1875, 0.875, -1.25, 0.0, -1.0, 1.1875, 1.5625, 1.125,
            0.875, -1.0, 1.125, -0.9375, 1.1875,
        ],
        "grad_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[1.375, 1.125, 0.875, 0.625], "grad_bias");
}

/// SeparableConv2D gradients at a stride of 2 and a dilation of 3 match Keras 3.15.1
#[test]
fn separable_conv2d_stride_and_dilation_gradients_match_keras() {
    let mut layer = SeparableConv2D::new(2, (2, 2), (2, 2), 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same)
        .with_dilation_rate((3, 3))
        .unwrap();
    layer.build(&Shape::known(&[1, 5, 5, 2])).unwrap();
    layer
        .set_weights(
            Array4::from_shape_vec((2, 2, 2, 1), ramp_kernel(8)).unwrap(),
            Array4::from_shape_vec((1, 1, 2, 2), ramp_pointwise(4)).unwrap(),
            Array1::from_vec(ramp_bias(2)),
        )
        .unwrap();

    let input = Array::from_shape_vec((1, 5, 5, 2), ramp_input(50))
        .unwrap()
        .into_dyn();
    let mut ctx = Ctx::training();
    let output = layer.forward(&input, &mut ctx).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3, 2]);
    assert_flat_close(
        &flat(&output),
        &[
            -1.0, -0.6875, 0.875, 1.3125, -0.75, -0.5, -1.625, 0.3125, -1.0, 0.25, 1.25, 0.25,
            0.125, 0.4375, -0.125, 0.0625, -1.75, -0.125,
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.515625, -0.15625,
            -0.015625, 0.0, -0.328125, 0.03125, 0.171875, 0.0, 0.0, 0.0, -0.078125, 0.09375,
            0.328125, 0.046875, -0.109375, -0.09375, -0.234375, -0.140625, 0.0, 0.0, 0.328125,
            0.09375, 0.078125, 0.0, -0.046875, -0.03125, 0.109375, 0.0, 0.0, 0.0, -0.171875,
            0.15625, 0.046875, -0.046875, 0.109375, -0.03125, -0.515625, -0.234375,
        ],
        "grad_input",
    );
    let depthwise_grad = ctx
        .grads()
        .get(ParamId::new(0, "depthwise_kernel"))
        .unwrap();
    assert_flat_close(
        &flat(depthwise_grad),
        &[
            0.65625, -0.40625, 1.375, 0.0625, -2.21875, 1.0, 1.375, -0.40625,
        ],
        "grad_depthwise_weights",
    );
    let pointwise_grad = ctx
        .grads()
        .get(ParamId::new(0, "pointwise_kernel"))
        .unwrap();
    assert_flat_close(
        &flat(pointwise_grad),
        &[0.25, 0.21875, 0.171875, 2.015625],
        "grad_pointwise_weights",
    );
    let bias_grad = ctx.grads().get(ParamId::new(0, "bias")).unwrap();
    assert_flat_close(&flat(bias_grad), &[0.875, 0.625], "grad_bias");
}

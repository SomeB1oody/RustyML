//! Integration tests for DepthwiseConv1D and SeparableConv1D.
//!
//! Tensors are channels-last (Keras): both layers take \[batch, length, channels\]. The depthwise
//! kernel is \[kernel_size, channels, depth_multiplier\] and the pointwise kernel is
//! \[1, channels * depth_multiplier, filters\].
//!
//! Expected values come from the mathematical definition or hand calculation, not from recording
//! layer output. Gradient checks live in tests/neural_network/gradient_check.rs.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array3};
use rustyml::neural_network::layers::TrainingParameters;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::depthwise_conv_1d::DepthwiseConv1D;
use rustyml::neural_network::layers::convolution::separable_conv_1d::SeparableConv1D;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use crate::common::assert_allclose;

/// A `[batch, length, channels]` tensor holding `values` in row-major order
fn seq(shape: [usize; 3], values: Vec<f32>) -> rustyml::neural_network::Tensor {
    Array::from_shape_vec(shape, values).unwrap().into_dyn()
}

/// A ramp of `n` distinct values, spread around 0 so no sign dominates
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|v| 0.1 * v as f32 - 0.5).collect()
}

/// The trainable parameter count of a layer
fn params_of(layer: &dyn Layer) -> usize {
    match layer.param_count() {
        TrainingParameters::Trainable(n) | TrainingParameters::NonTrainable(n) => n,
        TrainingParameters::NoTrainable => 0,
    }
}

// DepthwiseConv1D - constructor validation

/// Each invalid constructor argument returns the matching error: InvalidParameter for a zero
/// kernel size or stride, and InvalidInput for a bad input_shape.
#[test]
fn depthwise_conv1d_new_rejects_invalid_args() {
    enum Want {
        Param,
        Input,
    }
    // (label, kernel_size, input_shape, stride, expected error)
    type Case = (&'static str, usize, Vec<usize>, usize, Want);
    let cases: [Case; 5] = [
        ("kernel_size=0", 0, vec![1, 5, 2], 1, Want::Param),
        ("stride=0", 2, vec![1, 5, 2], 0, Want::Param),
        ("non-3D input_shape", 2, vec![1, 5, 2, 2], 1, Want::Input),
        ("zero channels", 2, vec![1, 5, 0], 1, Want::Input),
        (
            "input shorter than kernel",
            6,
            vec![1, 5, 2],
            1,
            Want::Input,
        ),
    ];
    for (label, kernel_size, input_shape, stride, want) in cases {
        let err =
            DepthwiseConv1D::new(kernel_size, input_shape, stride, Linear::new()).unwrap_err();
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
fn depthwise_conv1d_with_depth_multiplier_rejects_zero() {
    let err = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new())
        .unwrap()
        .with_depth_multiplier(0)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// A runtime input whose channel count differs from the declared one returns DimensionMismatch,
/// instead of a panic.
#[test]
fn depthwise_conv1d_forward_rejects_wrong_channels() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 5, 3)).into_dyn();
    let err = conv.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

/// A 4D input reaching a 1D layer returns InvalidInput rather than reading the wrong axes.
#[test]
fn depthwise_conv1d_forward_rejects_non_3d_input() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 5, 2, 1)).into_dyn();
    let err = conv.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// DepthwiseConv1D - channel independence (the core correctness invariant)

/// Changing 1 channel's input leaves every other channel's output untouched
///
/// This is what separates a depthwise convolution from a plain one. A plain `Conv1D` mixes all
/// input channels into every filter, so perturbing channel 0 would move all 3 outputs.
#[test]
fn depthwise_conv1d_cross_channel_no_bleed() {
    let mut conv = DepthwiseConv1D::new(3, vec![1, 6, 3], 1, Linear::new()).unwrap();
    // Distinct, non-zero weights everywhere, so a leak between channels cannot cancel
    let w = Array3::from_shape_vec((3, 3, 1), ramp(9)).unwrap();
    conv.set_weights(w, Array1::from_vec(vec![0.25, -0.5, 0.75]))
        .unwrap();

    let base = seq([1, 6, 3], ramp(18));
    let baseline = conv.predict(&base).unwrap();

    // Perturb every position of channel 1 only
    let mut bumped = base.clone();
    for t in 0..6 {
        bumped[[0, t, 1]] += 3.0;
    }
    let after = conv.predict(&bumped).unwrap();

    for t in 0..4 {
        assert_abs_diff_eq!(after[[0, t, 0]], baseline[[0, t, 0]], epsilon = 1e-6);
        assert_abs_diff_eq!(after[[0, t, 2]], baseline[[0, t, 2]], epsilon = 1e-6);
        assert!(
            (after[[0, t, 1]] - baseline[[0, t, 1]]).abs() > 1e-3,
            "channel 1 must respond to its own input at position {t}"
        );
    }
}

/// A depthwise convolution at `depth_multiplier = 1` equals a `Conv1D` whose kernel is 0 except
/// where the input channel and the filter index agree
///
/// The 2 layers share no code: `Conv1D` runs the im2col plus GEMM engine, while this layer runs
/// the direct loop nest. Agreement is therefore an independent check of the whole pass.
#[test]
fn depthwise_conv1d_matches_a_block_diagonal_conv1d() {
    let (length, channels, kernel) = (7usize, 3usize, 3usize);
    let taps = ramp(kernel * channels);

    let mut depthwise =
        DepthwiseConv1D::new(kernel, vec![2, length, channels], 2, Linear::new()).unwrap();
    let dw = Array3::from_shape_vec((kernel, channels, 1), taps.clone()).unwrap();
    let bias = Array1::from_vec(vec![0.5, -0.25, 1.0]);
    depthwise.set_weights(dw, bias.clone()).unwrap();

    // Conv1D kernel is [k, Cin, F]. Filter `f` reads only input channel `f`
    let mut plain = Conv1D::new(
        channels,
        kernel,
        vec![2, length, channels],
        2,
        Linear::new(),
    )
    .unwrap();
    let mut pw = Array3::<f32>::zeros((kernel, channels, channels));
    for t in 0..kernel {
        for c in 0..channels {
            pw[[t, c, c]] = taps[t * channels + c];
        }
    }
    plain.set_weights(pw, bias).unwrap();

    let x = seq([2, length, channels], ramp(2 * length * channels));
    assert_allclose(
        &depthwise.predict(&x).unwrap(),
        &plain.predict(&x).unwrap(),
        1e-5_f32,
    );
}

// DepthwiseConv1D - output length and padding

/// The output length follows `(in - k)/stride + 1` for Valid and `ceil(in/stride)` for Same
#[test]
fn depthwise_conv1d_output_length_rule() {
    // (label, length, kernel, stride, padding, expected output length)
    let cases: [(&str, usize, usize, usize, PaddingType, usize); 6] = [
        ("valid k3 s1", 10, 3, 1, PaddingType::Valid, 8),
        ("valid k3 s2", 10, 3, 2, PaddingType::Valid, 4),
        ("valid k=len", 6, 6, 1, PaddingType::Valid, 1),
        ("same s1", 10, 3, 1, PaddingType::Same, 10),
        ("same s2", 10, 3, 2, PaddingType::Same, 5),
        ("same s3", 10, 4, 3, PaddingType::Same, 4),
    ];
    for (label, length, kernel, stride, padding, want) in cases {
        let conv = DepthwiseConv1D::new(kernel, vec![2, length, 3], stride, Linear::new())
            .unwrap()
            .with_padding(padding);
        let x = Array::ones((2_usize, length, 3)).into_dyn();
        let out = conv.predict(&x).unwrap();
        assert_eq!(out.shape(), &[2, want, 3], "[{label}] output shape");
    }
}

/// `Same` padding puts `pad_total / 2` zeros on the leading edge and the remainder on the
/// trailing edge
///
/// An even kernel over an equal-length input makes the split visible. A width-4 kernel over
/// length 4 needs 3 pad cells. 1 goes in front and 2 go behind. Swapping the 2 halves would
/// give a different, and detectably wrong, output.
#[test]
fn depthwise_conv1d_same_padding_splits_with_the_extra_cell_at_the_end() {
    let mut conv = DepthwiseConv1D::new(4, vec![1, 4, 1], 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    conv.set_weights(Array3::ones((4, 1, 1)), Array1::zeros(1))
        .unwrap();

    let x = seq([1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let out = conv.predict(&x).unwrap();
    assert_eq!(out.shape(), &[1, 4, 1]);

    // pad_total = (4-1)*1 + 4 - 4 = 3, so 1 zero leads and 2 trail. The windows over
    // [0, 1, 2, 3, 4, 0, 0] are 0+1+2+3, 1+2+3+4, 2+3+4+0, and 3+4+0+0
    assert_eq!(
        out.iter().copied().collect::<Vec<f32>>(),
        vec![6.0, 10.0, 9.0, 7.0]
    );
}

// DepthwiseConv1D - depth multiplier

/// `depth_multiplier` widens the kernel, the bias, and the output, and reports the widened
/// channel count in `output_shape`
///
/// `output_shape` feeds `summary()`. It must report `channels * depth_multiplier`, not the
/// input channel count.
#[test]
fn depthwise_conv1d_depth_multiplier_widens_every_shape() {
    let mut conv = DepthwiseConv1D::new(3, vec![2, 10, 4], 1, Linear::new())
        .unwrap()
        .with_depth_multiplier(3)
        .unwrap();

    // Kernel [k, C, dm] = [3, 4, 3] and bias [C*dm] = [12]
    assert_eq!(params_of(&conv), 3 * 4 * 3 + 12);

    let x = Array::ones((2_usize, 10, 4)).into_dyn();
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 8, 12], "output carries C * dm channels");
    assert_eq!(conv.output_shape(), "(2, 8, 12)");
}

// DepthwiseConv1D - the remaining contract

/// `predict` in eval mode returns the same values as `forward`, and writes no caches
#[test]
fn depthwise_conv1d_predict_equals_forward() {
    let mut conv = DepthwiseConv1D::new(3, vec![2, 9, 2], 2, Linear::new()).unwrap();
    let x = seq([2, 9, 2], ramp(36));

    let predicted = conv.predict(&x).unwrap();
    let forwarded = conv.forward(&x).unwrap();
    assert_allclose(&predicted, &forwarded, 0.0_f32);

    // `predict` left no cache, so a second `predict` cannot enable `backward`
    let mut fresh = DepthwiseConv1D::new(3, vec![2, 9, 2], 2, Linear::new()).unwrap();
    fresh.predict(&x).unwrap();
    assert!(fresh.backward(&predicted).is_err());
}

/// `backward` before `forward` returns an error rather than reading an empty cache
#[test]
fn depthwise_conv1d_backward_before_forward_errors() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 4, 2)).into_dyn();
    assert!(conv.backward(&grad).is_err());
}

/// A rejected input leaves no partial cache behind
#[test]
fn depthwise_conv1d_rejected_forward_leaves_no_cache() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
    let wrong = Array::ones((1_usize, 5, 3)).into_dyn();
    assert!(conv.forward(&wrong).is_err());

    let grad = Array::ones((1_usize, 4, 2)).into_dyn();
    assert!(
        conv.backward(&grad).is_err(),
        "a rejected forward must not enable backward"
    );
}

/// `set_weights` rejects an array whose shape does not match the layer
#[test]
fn depthwise_conv1d_set_weights_shape_mismatch_errors() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();

    let err = conv
        .set_weights(Array3::zeros((3, 2, 1)), Array1::zeros(2))
        .unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape, got {err:?}"
    );

    let err = conv
        .set_weights(Array3::zeros((2, 2, 1)), Array1::zeros(5))
        .unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape for the bias, got {err:?}"
    );
}

/// Forward output and input gradient return in C order even from a strided input
#[test]
fn depthwise_conv1d_emits_c_order_tensors_from_a_strided_input() {
    // Permuting the length and channel axes makes the array non-contiguous
    let strided = Array::from_shape_vec((1_usize, 6, 4), ramp(24))
        .unwrap()
        .permuted_axes([0, 2, 1])
        .into_dyn();
    assert!(!strided.is_standard_layout(), "the fixture must be strided");
    assert_eq!(strided.shape(), &[1, 4, 6]);

    let mut conv = DepthwiseConv1D::new(2, vec![1, 4, 6], 1, Linear::new())
        .unwrap()
        .with_random_state(7);
    let out = conv.forward(&strided).unwrap();
    assert!(out.is_standard_layout(), "forward output must be C order");

    let grad_seed = Array::from_shape_vec((1_usize, 3, 6), ramp(18))
        .unwrap()
        .into_dyn();
    let grad = conv.backward(&grad_seed).unwrap();
    assert!(grad.is_standard_layout(), "input gradient must be C order");

    // The values must not depend on the input layout either
    let repacked = strided.as_standard_layout().into_owned();
    let mut twin = DepthwiseConv1D::new(2, vec![1, 4, 6], 1, Linear::new())
        .unwrap()
        .with_random_state(7);
    assert_allclose(&twin.forward(&repacked).unwrap(), &out, 0.0_f32);
    assert_allclose(&twin.backward(&grad_seed).unwrap(), &grad, 0.0_f32);
}

/// The parallel branch and the serial branch return the same numbers
///
/// The batch clears the naive-convolution FLOPs gate, while 1 sample on its own stays under it.
/// Running the batch and then each sample alone therefore exercises both branches.
#[test]
fn depthwise_conv1d_parallel_path_matches_the_serial_path() {
    let (samples, length, channels, kernel) = (4usize, 2048usize, 16usize, 8usize);
    let out_length = length - kernel + 1;
    let gate = rustyml::tuning::conv::get_naive_parallel_min_flops();
    let flops = 2 * samples * out_length * channels * kernel;
    assert!(flops >= gate, "the whole batch must clear the gate");
    assert!(flops / samples < gate, "1 sample must stay under the gate");

    let data: Vec<f32> = (0..samples * length * channels)
        .map(|k| (k % 17) as f32 - 8.0)
        .collect();
    let grad_data: Vec<f32> = (0..samples * out_length * channels)
        .map(|k| (k % 11) as f32 - 5.0)
        .collect();

    let mut conv = DepthwiseConv1D::new(kernel, vec![samples, length, channels], 1, Linear::new())
        .unwrap()
        .with_random_state(11);

    let batched_out = conv
        .forward(&seq([samples, length, channels], data.clone()))
        .unwrap();
    let batched_grad = conv
        .backward(&seq([samples, out_length, channels], grad_data.clone()))
        .unwrap();

    let in_stride = length * channels;
    let out_stride = out_length * channels;
    for sample in 0..samples {
        let one_out = conv
            .forward(&seq(
                [1, length, channels],
                data[sample * in_stride..(sample + 1) * in_stride].to_vec(),
            ))
            .unwrap();
        let one_grad = conv
            .backward(&seq(
                [1, out_length, channels],
                grad_data[sample * out_stride..(sample + 1) * out_stride].to_vec(),
            ))
            .unwrap();

        assert_eq!(
            one_out.as_slice().unwrap(),
            &batched_out.as_slice().unwrap()[sample * out_stride..(sample + 1) * out_stride],
            "sample {sample} forward"
        );
        assert_eq!(
            one_grad.as_slice().unwrap(),
            &batched_grad.as_slice().unwrap()[sample * in_stride..(sample + 1) * in_stride],
            "sample {sample} input gradient"
        );
    }
}

// SeparableConv1D - constructor validation

/// Each invalid constructor argument returns the matching error.
#[test]
fn separable_conv1d_new_rejects_invalid_args() {
    enum Want {
        Param,
        Input,
    }
    // (label, filters, kernel_size, input_shape, stride, depth_multiplier, expected error)
    type Case = (&'static str, usize, usize, Vec<usize>, usize, usize, Want);
    let cases: [Case; 6] = [
        ("filters=0", 0, 2, vec![1, 5, 2], 1, 1, Want::Param),
        ("kernel_size=0", 2, 0, vec![1, 5, 2], 1, 1, Want::Param),
        ("stride=0", 2, 2, vec![1, 5, 2], 0, 1, Want::Param),
        ("depth_multiplier=0", 2, 2, vec![1, 5, 2], 1, 0, Want::Param),
        ("non-3D input_shape", 2, 2, vec![1, 5], 1, 1, Want::Input),
        (
            "input shorter than kernel",
            2,
            6,
            vec![1, 5, 2],
            1,
            1,
            Want::Input,
        ),
    ];
    for (label, filters, kernel_size, input_shape, stride, dm, want) in cases {
        let err =
            SeparableConv1D::new(filters, kernel_size, input_shape, stride, dm, Linear::new())
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

/// A 4D input reaching a 1D layer returns InvalidInput rather than reading the wrong axes.
#[test]
fn separable_conv1d_forward_rejects_non_3d_input() {
    let mut conv = SeparableConv1D::new(2, 2, vec![1, 5, 2], 1, 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 5, 2, 1)).into_dyn();
    let err = conv.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// A runtime input whose channel count differs from the declared one returns DimensionMismatch,
/// instead of reading past the end of the depthwise kernel.
#[test]
fn separable_conv1d_forward_rejects_wrong_channels() {
    let mut conv = SeparableConv1D::new(2, 2, vec![1, 5, 2], 1, 1, Linear::new()).unwrap();
    let x = Array::ones((1_usize, 5, 3)).into_dyn();
    let err = conv.forward(&x).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
    // `predict` guards the same way, so neither entry point can reach the kernel
    let err = conv.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch from predict, got {err:?}"
    );
}

// SeparableConv1D - shapes and parameter counts

/// The output carries `filters` channels regardless of `depth_multiplier`, which only widens
/// the intermediate stage
#[test]
fn separable_conv1d_output_shape_and_param_count() {
    // (depth_multiplier, expected parameter count). The depthwise kernel is [k, C, dm], the
    // pointwise kernel is [1, C*dm, F], and the bias is [F]
    for (dm, want_params) in [(1usize, 9 + 3 * 5 + 5), (2, 3 * 3 * 2 + 6 * 5 + 5)] {
        let mut conv = SeparableConv1D::new(5, 3, vec![2, 10, 3], 1, dm, Linear::new()).unwrap();
        assert_eq!(params_of(&conv), want_params, "dm {dm} parameter count");

        let out = conv
            .forward(&Array::ones((2_usize, 10, 3)).into_dyn())
            .unwrap();
        assert_eq!(out.shape(), &[2, 8, 5], "dm {dm} output shape");
        assert_eq!(conv.output_shape(), "(2, 8, 5)");
    }
}

// SeparableConv1D - known-value forward passes

/// Identity depthwise and pointwise 1-tap kernels with zero bias reproduce the input exactly
#[test]
fn separable_conv1d_identity_reproduces_input() {
    let mut conv = SeparableConv1D::new(1, 1, vec![1, 5, 1], 1, 1, Linear::new()).unwrap();
    conv.set_weights(
        Array3::ones((1, 1, 1)),
        Array3::ones((1, 1, 1)),
        Array1::zeros(1),
    )
    .unwrap();

    let x = seq([1, 5, 1], ramp(5));
    let out = conv.forward(&x).unwrap();
    assert_allclose(&out, &x, 1e-6_f32);
}

/// Depthwise `[1, 0, 1]` then pointwise scale-by-2 plus bias 1 over 1..=5 matches by hand
#[test]
fn separable_conv1d_known_weight_forward_values() {
    let mut conv = SeparableConv1D::new(1, 3, vec![1, 5, 1], 1, 1, Linear::new()).unwrap();

    // depthwise [k, C, dm]: taps 1, 0, 1 select position i and i+2
    let dw = Array3::from_shape_vec((3, 1, 1), vec![1.0, 0.0, 1.0]).unwrap();
    // pointwise [1, C*dm, F]: scale by 2
    let pw = Array3::from_shape_vec((1, 1, 1), vec![2.0]).unwrap();
    conv.set_weights(dw, pw, Array1::from_elem(1, 1.0)).unwrap();

    let x = seq([1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 1]);

    // depthwise sums are 1+3, 2+4, 3+5, then scaled by 2 with bias 1
    assert_eq!(
        out.iter().copied().collect::<Vec<f32>>(),
        vec![9.0, 13.0, 17.0]
    );
}

/// `Same` padding zero-pads the depthwise stage of the separable layer too
#[test]
fn separable_conv1d_same_padding_zero_pads_depthwise() {
    let mut conv = SeparableConv1D::new(1, 3, vec![1, 5, 1], 1, 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    conv.set_weights(
        Array3::ones((3, 1, 1)),
        Array3::ones((1, 1, 1)),
        Array1::zeros(1),
    )
    .unwrap();

    let x = seq([1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 5, 1]);

    // pad_total = 2, so 1 zero sits on each edge. Each entry sums its in-bounds neighborhood
    assert_eq!(
        out.iter().copied().collect::<Vec<f32>>(),
        vec![3.0, 6.0, 9.0, 12.0, 9.0]
    );
}

/// The 2 stages agree on the intermediate channel order at `depth_multiplier = 2`
///
/// Each `(channel, multiplier)` pair gets a distinct power of 10 and each intermediate channel a
/// distinct power of 2. Any transposition of either stage changes the total.
#[test]
fn separable_conv1d_depth_multiplier_2_forward_values() {
    let mut conv = SeparableConv1D::new(1, 1, vec![1, 2, 2], 1, 2, Linear::new()).unwrap();
    let dw = Array3::from_shape_vec((1, 2, 2), vec![1.0, 10.0, 100.0, 1000.0]).unwrap();
    let pw = Array3::from_shape_vec((1, 4, 1), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
    conv.set_weights(dw, pw, Array1::zeros(1)).unwrap();

    // 2 positions: [2, 3] then [1, 1]
    let x = seq([1, 2, 2], vec![2.0, 3.0, 1.0, 1.0]);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 1]);

    // position 0: 2*1*1 + 2*10*2 + 3*100*4 + 3*1000*8 = 25242
    // position 1: 1*1*1 + 1*10*2 + 1*100*4 + 1*1000*8 = 8421
    assert_eq!(
        out.iter().copied().collect::<Vec<f32>>(),
        vec![25242.0, 8421.0]
    );
}

/// The fused layer equals a `DepthwiseConv1D` followed by a 1-tap `Conv1D`
///
/// This is the definition of a separable convolution, assembled from 2 other layers. It checks
/// the stage order, the intermediate channel order, and which stage carries the bias.
#[test]
fn separable_conv1d_matches_a_depthwise_then_pointwise_stack() {
    let (length, channels, filters, kernel, dm) = (9usize, 3usize, 4usize, 3usize, 2usize);
    let dw_taps = ramp(kernel * channels * dm);
    let pw_taps: Vec<f32> = (0..channels * dm * filters)
        .map(|v| 0.05 * v as f32 - 0.3)
        .collect();
    let bias: Vec<f32> = (0..filters).map(|v| 0.2 * v as f32 - 0.3).collect();

    let mut fused = SeparableConv1D::new(
        filters,
        kernel,
        vec![2, length, channels],
        2,
        dm,
        Linear::new(),
    )
    .unwrap()
    .with_padding(PaddingType::Same);
    fused
        .set_weights(
            Array3::from_shape_vec((kernel, channels, dm), dw_taps.clone()).unwrap(),
            Array3::from_shape_vec((1, channels * dm, filters), pw_taps.clone()).unwrap(),
            Array1::from_vec(bias.clone()),
        )
        .unwrap();

    // Stage 1 alone. It carries no bias, so this stack puts a zero bias on it
    let mut stage_1 = DepthwiseConv1D::new(kernel, vec![2, length, channels], 2, Linear::new())
        .unwrap()
        .with_depth_multiplier(dm)
        .unwrap()
        .with_padding(PaddingType::Same);
    stage_1
        .set_weights(
            Array3::from_shape_vec((kernel, channels, dm), dw_taps).unwrap(),
            Array1::zeros(channels * dm),
        )
        .unwrap();

    // Stage 2 is a 1-tap Conv1D over the widened channels, and it carries the bias
    let out_length = length.div_ceil(2);
    let mut stage_2 = Conv1D::new(
        filters,
        1,
        vec![2, out_length, channels * dm],
        1,
        Linear::new(),
    )
    .unwrap();
    stage_2
        .set_weights(
            Array3::from_shape_vec((1, channels * dm, filters), pw_taps).unwrap(),
            Array1::from_vec(bias),
        )
        .unwrap();

    let x = seq([2, length, channels], ramp(2 * length * channels));
    let stacked = stage_2.predict(&stage_1.predict(&x).unwrap()).unwrap();
    assert_allclose(&fused.predict(&x).unwrap(), &stacked, 1e-5_f32);
}

// SeparableConv1D - the remaining contract

/// `predict` in eval mode returns the same values as `forward`
#[test]
fn separable_conv1d_predict_equals_forward() {
    let mut conv = SeparableConv1D::new(3, 3, vec![2, 9, 2], 2, 2, Linear::new()).unwrap();
    let x = seq([2, 9, 2], ramp(36));
    let predicted = conv.predict(&x).unwrap();
    let forwarded = conv.forward(&x).unwrap();
    assert_allclose(&predicted, &forwarded, 0.0_f32);
}

/// `backward` before `forward` returns an error rather than reading an empty cache
#[test]
fn separable_conv1d_backward_before_forward_errors() {
    let mut conv = SeparableConv1D::new(2, 2, vec![1, 5, 2], 1, 1, Linear::new()).unwrap();
    let grad = Array::ones((1_usize, 4, 2)).into_dyn();
    assert!(conv.backward(&grad).is_err());
}

/// `set_weights` rejects any of the 3 arrays whose shape does not match the layer
#[test]
fn separable_conv1d_set_weights_shape_mismatch_errors() {
    let mut conv = SeparableConv1D::new(2, 2, vec![1, 5, 2], 1, 1, Linear::new()).unwrap();
    // (label, depthwise shape, pointwise shape, bias length)
    type Shape = (usize, usize, usize);
    type Case = (&'static str, Shape, Shape, usize);
    let cases: [Case; 3] = [
        ("depthwise", (3, 2, 1), (1, 2, 2), 2),
        ("pointwise", (2, 2, 1), (1, 3, 2), 2),
        ("bias", (2, 2, 1), (1, 2, 2), 5),
    ];
    for (label, dw, pw, bias) in cases {
        let err = conv
            .set_weights(Array3::zeros(dw), Array3::zeros(pw), Array1::zeros(bias))
            .unwrap_err();
        assert!(
            matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
            "[{label}] expected WeightShape, got {err:?}"
        );
    }
}

/// Forward output and input gradient return in C order even from a strided input
#[test]
fn separable_conv1d_emits_c_order_tensors_from_a_strided_input() {
    let strided = Array::from_shape_vec((1_usize, 6, 4), ramp(24))
        .unwrap()
        .permuted_axes([0, 2, 1])
        .into_dyn();
    assert!(!strided.is_standard_layout(), "the fixture must be strided");
    assert_eq!(strided.shape(), &[1, 4, 6]);

    let mut conv = SeparableConv1D::new(2, 2, vec![1, 4, 6], 1, 1, Linear::new())
        .unwrap()
        .with_random_state(3);
    let out = conv.forward(&strided).unwrap();
    assert!(out.is_standard_layout(), "forward output must be C order");

    let grad_seed = Array::from_shape_vec((1_usize, 3, 2), ramp(6))
        .unwrap()
        .into_dyn();
    let grad = conv.backward(&grad_seed).unwrap();
    assert!(grad.is_standard_layout(), "input gradient must be C order");

    let repacked = strided.as_standard_layout().into_owned();
    let mut twin = SeparableConv1D::new(2, 2, vec![1, 4, 6], 1, 1, Linear::new())
        .unwrap()
        .with_random_state(3);
    assert_allclose(&twin.forward(&repacked).unwrap(), &out, 0.0_f32);
    assert_allclose(&twin.backward(&grad_seed).unwrap(), &grad, 0.0_f32);
}

// Weight containers

/// `get_weights` returns the matching variant, with the shapes the layer declares
#[test]
fn conv_1d_variants_report_their_weight_containers() {
    let depthwise = DepthwiseConv1D::new(3, vec![1, 10, 4], 1, Linear::new())
        .unwrap()
        .with_depth_multiplier(2)
        .unwrap();
    let LayerWeight::DepthwiseConv1D(dw) = depthwise.get_weights() else {
        panic!("DepthwiseConv1D must report its own variant");
    };
    assert_eq!(dw.weight.shape(), &[3, 4, 2]);
    assert_eq!(dw.bias.len(), 8);
    assert_eq!(depthwise.layer_type(), "DepthwiseConv1D");

    let separable = SeparableConv1D::new(5, 3, vec![1, 10, 4], 1, 2, Linear::new()).unwrap();
    let LayerWeight::SeparableConv1D(sw) = separable.get_weights() else {
        panic!("SeparableConv1D must report its own variant");
    };
    assert_eq!(sw.depthwise_weight.shape(), &[3, 4, 2]);
    assert_eq!(sw.pointwise_weight.shape(), &[1, 8, 5]);
    assert_eq!(sw.bias.len(), 5);
    assert_eq!(separable.layer_type(), "SeparableConv1D");
}

/// The bias skips decoupled weight decay while both kernels take it
#[test]
fn separable_conv1d_bias_is_exempt_from_weight_decay() {
    let mut conv = SeparableConv1D::new(2, 2, vec![1, 5, 2], 1, 1, Linear::new()).unwrap();
    let x = seq([1, 5, 2], ramp(10));
    conv.forward(&x).unwrap();
    conv.backward(&Array::ones((1_usize, 4, 2)).into_dyn())
        .unwrap();

    let params = conv.parameters();
    assert_eq!(params.len(), 3, "depthwise, pointwise, and bias");
    assert!(params[0].decays, "the depthwise kernel decays");
    assert!(params[1].decays, "the pointwise kernel decays");
    assert!(!params[2].decays, "the bias does not decay");
}

/// The depthwise bias skips decoupled weight decay while the kernel takes it
#[test]
fn depthwise_conv1d_bias_is_exempt_from_weight_decay() {
    let mut conv = DepthwiseConv1D::new(2, vec![1, 5, 2], 1, Linear::new()).unwrap();
    let x = seq([1, 5, 2], ramp(10));
    conv.forward(&x).unwrap();
    conv.backward(&Array::ones((1_usize, 4, 2)).into_dyn())
        .unwrap();

    let params = conv.parameters();
    assert_eq!(params.len(), 2, "kernel and bias");
    assert!(params[0].decays, "the kernel decays");
    assert!(!params[1].decays, "the bias does not decay");
}

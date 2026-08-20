//! Integration tests for Conv1DTranspose, Conv2DTranspose, and Conv3DTranspose forward values,
//! shapes, error paths, param counts, set_weights, and predict==forward equivalence.
//!
//! Layout is channels-last: Conv1DTranspose sees \[batch, length, channels\] and Conv2DTranspose
//! sees \[batch, height, width, channels\]. A transposed kernel is \[k, F, Cin\] and
//! \[kh, kw, F, Cin\]. The filter axis comes before the input-channel axis, which is the reverse
//! of the plain convolution kernel. Every bias is rank-1 of length `filters`.
//!
//! Expected values come from the scatter definition with known weights, worked out in a comment
//! next to each one, not from recorded implementation output. Every pinned number was also
//! confirmed against Keras 3.15 on the jax backend. Gradient values live in gradient_check.rs,
//! and this file does not duplicate them.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array3, Array4, Array5, IxDyn};
use rustyml::error::Error;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d_transpose::Conv1DTranspose;
use rustyml::neural_network::layers::convolution::conv_2d_transpose::Conv2DTranspose;
use rustyml::neural_network::layers::convolution::conv_3d_transpose::Conv3DTranspose;
use rustyml::neural_network::layers::layer_weight::LayerWeight;
use rustyml::neural_network::layers::{Dense, Flatten, TrainingParameters};
use rustyml::neural_network::losses::MeanSquaredError;
use rustyml::neural_network::optimizers::SGD;
use rustyml::neural_network::sequential::Sequential;
use rustyml::neural_network::traits::Layer;
use rustyml::neural_network::{NnError, Tensor};

use super::common::assert_allclose;

// helpers

/// A rank-3 tensor from a flat row-major list
fn t3(shape: (usize, usize, usize), data: Vec<f32>) -> Tensor {
    Array::from_shape_vec(shape, data).unwrap().into_dyn()
}

/// A rank-4 tensor from a flat row-major list
fn t4(shape: (usize, usize, usize, usize), data: Vec<f32>) -> Tensor {
    Array::from_shape_vec(shape, data).unwrap().into_dyn()
}

/// A tensor of distinct values, so a moved axis changes the result
fn ramp_of(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|v| 0.25 * v as f32 - 1.0).collect();
    Array::from_shape_vec(IxDyn(shape), data).unwrap()
}

// Conv1DTranspose - forward with known weights

/// Every input position writes its value times the whole kernel, and the writes accumulate where
/// they overlap
#[test]
fn conv1d_transpose_scatters_each_input_across_the_kernel() {
    let mut layer = Conv1DTranspose::new(1, 2, vec![1, 2, 1], 1, Linear::new()).unwrap();
    // Weight shape [kernel=2, filters=1, channels=1]. The 2 taps in order are 2 and 3
    let weights = Array3::from_shape_vec((2, 1, 1), vec![2.0f32, 3.0]).unwrap();
    let bias = Array1::from_vec(vec![0.5f32]);
    layer.set_weights(weights, bias).unwrap();

    let input = t3((1, 2, 1), vec![1.0, 2.0]);
    let output = layer.forward(&input).unwrap();

    // Valid at stride 1 gives 2 * 1 + (2 - 1) = 3 positions. out[o] = sum over i + k == o.
    // out[0] = 1*2 = 2. out[1] = 1*3 + 2*2 = 7. out[2] = 2*3 = 6. The bias adds 0.5.
    assert_eq!(output.shape(), &[1, 3, 1]);
    let expected = t3((1, 3, 1), vec![2.5f32, 7.5, 6.5]);
    assert_allclose(&output, &expected, 1e-6f32);
}

/// A stride wider than the kernel leaves output positions that no input reaches, and those hold
/// exactly the bias
#[test]
fn conv1d_transpose_gap_positions_hold_only_the_bias() {
    let mut layer = Conv1DTranspose::new(1, 1, vec![1, 2, 1], 3, Linear::new()).unwrap();
    let weights = Array3::from_shape_vec((1, 1, 1), vec![10.0f32]).unwrap();
    let bias = Array1::from_vec(vec![0.25f32]);
    layer.set_weights(weights, bias).unwrap();

    let input = t3((1, 2, 1), vec![1.0, 2.0]);
    let output = layer.forward(&input).unwrap();

    // Valid with kernel 1 below stride 3 gives 2 * 3 + max(1 - 3, 0) = 6 positions. Only
    // positions 0 and 3 receive a write
    assert_eq!(output.shape(), &[1, 6, 1]);
    let expected = t3((1, 6, 1), vec![10.25f32, 0.25, 0.25, 20.25, 0.25, 0.25]);
    assert_allclose(&output, &expected, 1e-6f32);
}

/// `Same` padding at stride 1 keeps the length and drops the writes that fall past both ends
#[test]
fn conv1d_transpose_same_padding_keeps_the_length() {
    let mut layer = Conv1DTranspose::new(1, 3, vec![1, 3, 1], 1, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    // Taps in order are 2, 3, 5
    let weights = Array3::from_shape_vec((3, 1, 1), vec![2.0f32, 3.0, 5.0]).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = t3((1, 3, 1), vec![1.0, 2.0, 3.0]);
    let output = layer.forward(&input).unwrap();

    // Same at stride 1 gives 3 * 1 = 3 positions, and pad_before = (3 - 1) / 2 = 1.
    // So out[o] = sum over i + k == o + 1.
    // out[0] = 1*3 + 2*2 = 7. out[1] = 1*5 + 2*3 + 3*2 = 17. out[2] = 2*5 + 3*3 = 19.
    assert_eq!(output.shape(), &[1, 3, 1]);
    let expected = t3((1, 3, 1), vec![7.0f32, 17.0, 19.0]);
    assert_allclose(&output, &expected, 1e-6f32);
}

/// `Same` padding at stride 2 doubles the length and crops only the trailing overhang
#[test]
fn conv1d_transpose_same_padding_at_stride_2_crops_the_tail() {
    let mut layer = Conv1DTranspose::new(1, 3, vec![1, 2, 1], 2, Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let weights = Array3::from_shape_vec((3, 1, 1), vec![2.0f32, 3.0, 5.0]).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = t3((1, 2, 1), vec![1.0, 2.0]);
    let output = layer.forward(&input).unwrap();

    // Same at stride 2 gives 2 * 2 = 4 positions, and pad_before = (3 - 2) / 2 = 0.
    // So out[o] = sum over 2*i + k == o. Input 0 writes 2, 3, 5 at 0, 1, 2, and input 1 writes
    // 4, 6, 10 at 2, 3, 4. Position 4 falls past the output and drops.
    assert_eq!(output.shape(), &[1, 4, 1]);
    let expected = t3((1, 4, 1), vec![2.0f32, 3.0, 9.0, 6.0]);
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Both input channels reach every output position, so the channel axis cannot be read anywhere
/// but last
#[test]
fn conv1d_transpose_two_input_channels_cross_channel_sum() {
    let mut layer = Conv1DTranspose::new(1, 2, vec![1, 2, 2], 1, Linear::new()).unwrap();
    // [kernel=2, filters=1, channels=2]: tap 0 weighs the 2 channels 1 and 2, tap 1 weighs them
    // 3 and 4. The per-channel weights and the per-channel data both differ, so reading the
    // channel axis anywhere but last changes every value
    let weights = Array3::from_shape_vec((2, 1, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = t3((1, 2, 2), vec![1.0, 10.0, 2.0, 20.0]);
    let output = layer.forward(&input).unwrap();

    // out[0] = 1*1 + 10*2 = 21. out[1] = (1*3 + 10*4) + (2*1 + 20*2) = 43 + 42 = 85.
    // out[2] = 2*3 + 20*4 = 86.
    assert_eq!(output.shape(), &[1, 3, 1]);
    let expected = t3((1, 3, 1), vec![21.0f32, 85.0, 86.0]);
    assert_allclose(&output, &expected, 1e-6f32);
}

/// The filter axis is the fastest-varying output axis, so a second filter lands interleaved with
/// the first and not in a separate plane
#[test]
fn conv1d_transpose_filter_axis_is_innermost() {
    let mut layer = Conv1DTranspose::new(2, 1, vec![1, 1, 1], 1, Linear::new()).unwrap();
    // [kernel=1, filters=2, channels=1]: filter 0 weighs 1 and filter 1 weighs 2
    let weights = Array3::from_shape_vec((1, 2, 1), vec![1.0f32, 2.0]).unwrap();
    layer.set_weights(weights, Array1::zeros(2)).unwrap();

    let output = layer.forward(&t3((1, 1, 1), vec![3.0])).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2]);
    let expected = t3((1, 1, 2), vec![3.0f32, 6.0]);
    assert_allclose(&output, &expected, 1e-6f32);
}

// Conv1DTranspose - output length rules

/// Valid output length is `input * stride + max(kernel - stride, 0)` across parameterized cases
#[test]
fn conv1d_transpose_valid_output_length_cases() {
    // (input_len, kernel, stride, expected_out_len)
    let cases = [
        (4usize, 3usize, 1usize, 6usize),
        (4, 3, 2, 9),
        (4, 1, 1, 4),
        (4, 2, 3, 12),
        (1, 5, 2, 5),
        (3, 4, 4, 12),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut layer =
            Conv1DTranspose::new(1, kernel, vec![1, len, 1], stride, Linear::new()).unwrap();
        let x = Array::ones((1, len, 1)).into_dyn();
        let out = layer.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, expected, 1],
            "Conv1DTranspose Valid: input_len={len}, kernel={kernel}, stride={stride}"
        );
    }
}

/// Same output length is `input * stride`, whatever the kernel size
#[test]
fn conv1d_transpose_same_output_length_cases() {
    // (input_len, kernel, stride, expected_out_len)
    let cases = [
        (4usize, 3usize, 1usize, 4usize),
        (4, 3, 2, 8),
        (4, 7, 2, 8),
        (5, 1, 3, 15),
        (1, 4, 1, 1),
    ];
    for (len, kernel, stride, expected) in cases {
        let mut layer = Conv1DTranspose::new(1, kernel, vec![1, len, 1], stride, Linear::new())
            .unwrap()
            .with_padding(PaddingType::Same);
        let x = Array::ones((1, len, 1)).into_dyn();
        let out = layer.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[1, expected, 1],
            "Conv1DTranspose Same: input_len={len}, kernel={kernel}, stride={stride}"
        );
    }
}

// Conv1DTranspose - Layer trait surface

/// param_count is `kernel * filters * channels + filters`
#[test]
fn conv1d_transpose_param_count_formula() {
    let layer = Conv1DTranspose::new(4, 3, vec![1, 5, 2], 1, Linear::new()).unwrap();
    // 3 * 4 * 2 kernel values plus 4 biases
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(28));
}

/// get_weights returns the Conv1DTranspose variant, with the filter axis before the channel axis
#[test]
fn conv1d_transpose_get_weights_correct_shapes() {
    let layer = Conv1DTranspose::new(4, 3, vec![1, 5, 2], 1, Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv1DTranspose(w) => {
            assert_eq!(w.weight.shape(), &[3, 4, 2]);
            assert_eq!(w.bias.shape(), &[4]);
        }
        _other => panic!("expected LayerWeight::Conv1DTranspose variant"),
    }
}

/// layer_type and output_shape report the transposed geometry
#[test]
fn conv1d_transpose_reports_its_type_and_output_shape() {
    let layer = Conv1DTranspose::new(3, 3, vec![2, 5, 1], 2, Linear::new()).unwrap();
    assert_eq!(layer.layer_type(), "Conv1DTranspose");
    // Valid: 5 * 2 + (3 - 2) = 11
    assert_eq!(layer.output_shape(), "(2, 11, 3)");

    let same = layer.with_padding(PaddingType::Same);
    // Same: 5 * 2 = 10
    assert_eq!(same.output_shape(), "(2, 10, 3)");
}

/// predict returns the same values as forward for a deterministic layer
#[test]
fn conv1d_transpose_predict_equals_forward() {
    let mut layer = Conv1DTranspose::new(2, 3, vec![1, 4, 2], 2, Linear::new())
        .unwrap()
        .with_random_state(7);
    let x = ramp_of(&[1, 4, 2]);
    let forward_output = layer.forward(&x).unwrap();
    let predict_output = layer.predict(&x).unwrap();
    assert_allclose(&predict_output, &forward_output, 0.0f32);
}

// Conv2DTranspose - forward with known weights

/// A 2x2 input under a 2x2 kernel at stride 1 grows to 3x3, and the 4 windows overlap in the
/// middle
#[test]
fn conv2d_transpose_overlapping_windows_add_up() {
    let mut layer =
        Conv2DTranspose::new(1, (2, 2), vec![1, 2, 2, 1], (1, 1), Linear::new()).unwrap();
    // [kh=2, kw=2, filters=1, channels=1]: taps in (kh, kw) order are 1, 2, 3, 4
    let weights = Array4::from_shape_vec((2, 2, 1, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = t4((1, 2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]);
    let output = layer.forward(&input).unwrap();

    // out[oh, ow] = sum over ih + kh == oh and iw + kw == ow of x[ih, iw] * w[kh, kw].
    // out[0,0] = 1*1 = 1. out[0,1] = 1*2 + 2*1 = 4. out[0,2] = 2*2 = 4.
    // out[1,0] = 1*3 + 3*1 = 6. out[1,1] = 1*4 + 2*3 + 3*2 + 4*1 = 20. out[1,2] = 2*4 + 4*2 = 16.
    // out[2,0] = 3*3 = 9. out[2,1] = 3*4 + 4*3 = 24. out[2,2] = 4*4 = 16.
    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    let expected = t4(
        (1, 3, 3, 1),
        vec![1.0f32, 4.0, 4.0, 6.0, 20.0, 16.0, 9.0, 24.0, 16.0],
    );
    assert_allclose(&output, &expected, 1e-6f32);
}

/// A 1x1 input under a 3x3 kernel is a normal first decoder step, and a plain Conv2D refuses
/// exactly this shape
#[test]
fn conv2d_transpose_grows_a_single_pixel_into_the_whole_kernel() {
    let mut layer =
        Conv2DTranspose::new(1, (3, 3), vec![1, 1, 1, 1], (2, 2), Linear::new()).unwrap();
    let taps: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let weights = Array4::from_shape_vec((3, 3, 1, 1), taps.clone()).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let output = layer.forward(&t4((1, 1, 1, 1), vec![2.0])).unwrap();

    // Valid: 1 * 2 + (3 - 2) = 3 on each axis. The single input scales the whole kernel
    assert_eq!(output.shape(), &[1, 3, 3, 1]);
    let expected = t4((1, 3, 3, 1), taps.iter().map(|v| 2.0 * v).collect());
    assert_allclose(&output, &expected, 1e-6f32);

    // A plain Conv2D cannot be built on this input at all
    let refused = rustyml::neural_network::layers::convolution::conv_2d::Conv2D::new(
        1,
        (3, 3),
        vec![1, 1, 1, 1],
        (1, 1),
        Linear::new(),
    );
    assert!(
        matches!(refused, Err(Error::InvalidInput(_))),
        "a plain Conv2D must reject an input below its kernel size, got {refused:?}"
    );
}

/// `Same` padding at stride 2 doubles both axes and crops the overhang off the trailing edges
#[test]
fn conv2d_transpose_same_padding_doubles_both_axes() {
    let mut layer = Conv2DTranspose::new(1, (3, 3), vec![1, 2, 2, 1], (2, 2), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let taps: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let weights = Array4::from_shape_vec((3, 3, 1, 1), taps).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = t4((1, 2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]);
    let output = layer.forward(&input).unwrap();

    // Same at stride 2 gives 2 * 2 = 4 on each axis, and pad_before = (3 - 2) / 2 = 0.
    // So out[oh, ow] = sum over 2*ih + kh == oh and 2*iw + kw == ow.
    // out[0,0] = 1*w[0,0] = 1.
    // out[2,2] = 1*w[2,2] + 2*w[2,0] + 3*w[0,2] + 4*w[0,0] = 9 + 14 + 9 + 4 = 36.
    // out[3,3] = 4*w[1,1] = 20.
    assert_eq!(output.shape(), &[1, 4, 4, 1]);
    let expected = t4(
        (1, 4, 4, 1),
        vec![
            1.0f32, 2.0, 5.0, 4.0, //
            4.0, 5.0, 14.0, 10.0, //
            10.0, 14.0, 36.0, 24.0, //
            12.0, 15.0, 34.0, 20.0,
        ],
    );
    assert_allclose(&output, &expected, 1e-6f32);
}

// Conv2DTranspose - output shape rules

/// Each axis applies its own rule, so a mixed kernel and a mixed stride do not interfere
#[test]
fn conv2d_transpose_output_shape_cases() {
    // (H, W, kh, kw, sh, sw, padding, out_h, out_w)
    let cases = [
        (
            3usize, 4usize, 3usize, 3usize, 1usize, 1usize, false, 5usize, 6usize,
        ),
        (3, 4, 3, 3, 2, 2, false, 7, 9),
        (3, 4, 2, 3, 2, 1, false, 6, 6),
        (3, 4, 4, 2, 1, 3, false, 6, 12),
        (3, 4, 3, 3, 1, 1, true, 3, 4),
        (3, 4, 3, 3, 2, 2, true, 6, 8),
        (3, 4, 5, 3, 3, 1, true, 9, 4),
        (3, 4, 1, 1, 2, 2, true, 6, 8),
    ];
    for (h, w, kh, kw, sh, sw, same, out_h, out_w) in cases {
        let mut layer =
            Conv2DTranspose::new(2, (kh, kw), vec![1, h, w, 1], (sh, sw), Linear::new()).unwrap();
        if same {
            layer = layer.with_padding(PaddingType::Same);
        }
        let out = layer
            .forward(&Array::ones((1, h, w, 1)).into_dyn())
            .unwrap();
        assert_eq!(
            out.shape(),
            &[1, out_h, out_w, 2],
            "Conv2DTranspose: {h}x{w}, kernel {kh}x{kw}, stride {sh}x{sw}, same={same}"
        );
    }
}

// Conv2DTranspose - Layer trait surface

/// param_count is `kh * kw * filters * channels + filters`
#[test]
fn conv2d_transpose_param_count_formula() {
    let layer = Conv2DTranspose::new(2, (3, 3), vec![1, 5, 5, 3], (1, 1), Linear::new()).unwrap();
    // 3 * 3 * 2 * 3 kernel values plus 2 biases
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(56));
}

/// get_weights returns the Conv2DTranspose variant with shape \[kh, kw, filters, channels\]
#[test]
fn conv2d_transpose_get_weights_correct_shapes() {
    let layer = Conv2DTranspose::new(4, (3, 3), vec![1, 8, 8, 2], (1, 1), Linear::new()).unwrap();
    match layer.get_weights() {
        LayerWeight::Conv2DTranspose(w) => {
            assert_eq!(w.weight.shape(), &[3, 3, 4, 2]);
            assert_eq!(w.bias.shape(), &[4]);
        }
        _other => panic!("expected LayerWeight::Conv2DTranspose variant"),
    }
}

/// layer_type and output_shape report the transposed geometry
#[test]
fn conv2d_transpose_reports_its_type_and_output_shape() {
    let layer = Conv2DTranspose::new(3, (3, 2), vec![2, 5, 5, 1], (2, 1), Linear::new()).unwrap();
    assert_eq!(layer.layer_type(), "Conv2DTranspose");
    // Valid: 5 * 2 + (3 - 2) = 11 on the height. 5 * 1 + (2 - 1) = 6 on the width.
    assert_eq!(layer.output_shape(), "(2, 11, 6, 3)");
}

/// with_random_state makes the initial kernel reproducible and leaves the bias at 0
#[test]
fn conv2d_transpose_with_random_state_is_reproducible() {
    let first = Conv2DTranspose::new(2, (3, 3), vec![1, 4, 4, 2], (1, 1), Linear::new())
        .unwrap()
        .with_random_state(19);
    let second = Conv2DTranspose::new(2, (3, 3), vec![1, 4, 4, 2], (1, 1), Linear::new())
        .unwrap()
        .with_random_state(19);

    match (first.get_weights(), second.get_weights()) {
        (LayerWeight::Conv2DTranspose(a), LayerWeight::Conv2DTranspose(b)) => {
            assert_eq!(a.weight.as_ref(), b.weight.as_ref());
            assert!(a.bias.iter().all(|&v| v == 0.0), "the bias must start at 0");
            // Xavier bound for 3*3 taps at 2 channels and 2 filters is sqrt(6 / 36)
            let bound = (6.0f32 / 36.0).sqrt();
            assert!(
                a.weight.iter().all(|&v| v.abs() <= bound),
                "every weight must stay inside the Xavier bound {bound}"
            );
        }
        _other => panic!("expected LayerWeight::Conv2DTranspose variants"),
    }
}

// Conv3DTranspose

/// A single voxel under a 2x2x2 kernel scales the whole kernel into a 2x2x2 block
#[test]
fn conv3d_transpose_grows_a_single_voxel_into_the_whole_kernel() {
    let mut layer =
        Conv3DTranspose::new(1, (2, 2, 2), vec![1, 1, 1, 1, 1], (1, 1, 1), Linear::new()).unwrap();
    let taps: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let weights = Array5::from_shape_vec((2, 2, 2, 1, 1), taps.clone()).unwrap();
    layer.set_weights(weights, Array1::zeros(1)).unwrap();

    let input = Array::from_shape_vec((1, 1, 1, 1, 1), vec![3.0f32])
        .unwrap()
        .into_dyn();
    let output = layer.forward(&input).unwrap();

    // Valid: 1 * 1 + (2 - 1) = 2 on each of the 3 axes
    assert_eq!(output.shape(), &[1, 2, 2, 2, 1]);
    let expected = Array::from_shape_vec(
        (1, 2, 2, 2, 1),
        taps.iter().map(|v| 3.0 * v).collect::<Vec<f32>>(),
    )
    .unwrap()
    .into_dyn();
    assert_allclose(&output, &expected, 1e-6f32);
}

/// Depth, height, and width each apply the rule on their own
#[test]
fn conv3d_transpose_output_shape_cases() {
    // (D, H, W, kd, kh, kw, sd, sh, sw, padding, out_d, out_h, out_w)
    let cases = [
        (
            2usize, 3usize, 2usize, 2usize, 2usize, 2usize, 1usize, 1usize, 1usize, false, 3usize,
            4usize, 3usize,
        ),
        (2, 3, 2, 3, 3, 3, 2, 2, 2, false, 5, 7, 5),
        (2, 3, 2, 2, 3, 1, 1, 2, 3, false, 3, 7, 6),
        (2, 3, 2, 3, 3, 3, 1, 1, 1, true, 2, 3, 2),
        (2, 3, 2, 2, 2, 2, 2, 2, 2, true, 4, 6, 4),
        (2, 3, 2, 3, 2, 4, 2, 1, 2, true, 4, 3, 4),
    ];
    for (d, h, w, kd, kh, kw, sd, sh, sw, same, od, oh, ow) in cases {
        let mut layer = Conv3DTranspose::new(
            2,
            (kd, kh, kw),
            vec![1, d, h, w, 1],
            (sd, sh, sw),
            Linear::new(),
        )
        .unwrap();
        if same {
            layer = layer.with_padding(PaddingType::Same);
        }
        let out = layer
            .forward(&Array::ones((1, d, h, w, 1)).into_dyn())
            .unwrap();
        assert_eq!(
            out.shape(),
            &[1, od, oh, ow, 2],
            "Conv3DTranspose: {d}x{h}x{w}, kernel {kd}x{kh}x{kw}, stride {sd}x{sh}x{sw}, same={same}"
        );
    }
}

/// get_weights returns the Conv3DTranspose variant, and param_count matches its extent
#[test]
fn conv3d_transpose_weights_and_param_count() {
    let layer =
        Conv3DTranspose::new(2, (2, 3, 2), vec![1, 3, 3, 3, 2], (1, 1, 1), Linear::new()).unwrap();
    assert_eq!(layer.layer_type(), "Conv3DTranspose");
    // 2 * 3 * 2 taps at 2 filters and 2 channels, plus 2 biases
    assert_eq!(layer.param_count(), TrainingParameters::Trainable(50));
    match layer.get_weights() {
        LayerWeight::Conv3DTranspose(w) => {
            assert_eq!(w.weight.shape(), &[2, 3, 2, 2, 2]);
            assert_eq!(w.bias.shape(), &[2]);
        }
        _other => panic!("expected LayerWeight::Conv3DTranspose variant"),
    }
}

// Round trip against the plain convolution

/// A transposed convolution returns to the shape the matching convolution consumed
///
/// The 2 layers share the kernel size, the stride, and the padding mode. The round trip is exact
/// when the convolution kept every input position. `(in - kernel)` is divisible by the stride
/// under `Valid`, and `in` is divisible by the stride under `Same`. The next test covers what
/// happens when it is not
#[test]
fn conv2d_transpose_returns_to_the_shape_the_convolution_consumed() {
    use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;

    // (H, W, kh, kw, sh, sw, same)
    let cases = [
        (8usize, 8usize, 3usize, 3usize, 1usize, 1usize, false),
        (9, 9, 3, 3, 2, 2, false),
        (9, 7, 4, 2, 1, 1, false),
        (8, 8, 3, 3, 1, 1, true),
        (8, 8, 3, 3, 2, 2, true),
        (10, 8, 4, 4, 2, 2, true),
    ];
    for (h, w, kh, kw, sh, sw, same) in cases {
        let padding = if same {
            PaddingType::Same
        } else {
            PaddingType::Valid
        };
        let mut down = Conv2D::new(2, (kh, kw), vec![1, h, w, 1], (sh, sw), Linear::new())
            .unwrap()
            .with_padding(padding);
        let small = down.forward(&Array::ones((1, h, w, 1)).into_dyn()).unwrap();

        let mut up =
            Conv2DTranspose::new(1, (kh, kw), small.shape().to_vec(), (sh, sw), Linear::new())
                .unwrap()
                .with_padding(padding);
        let back = up.forward(&small).unwrap();

        assert_eq!(
            back.shape(),
            &[1, h, w, 1],
            "round trip at {h}x{w}, kernel {kh}x{kw}, stride {sh}x{sw}, same={same}"
        );
    }
}

/// A stride that does not divide the input makes the round trip short by the remainder
///
/// The convolution drops the trailing positions its last window cannot reach, so the transposed
/// convolution has nothing to rebuild them from. An 8-wide input at kernel 3 and stride 2 gives 3
/// output positions, and 3 positions rebuild only 7. This is the case an `output_padding`
/// argument exists to cover, and this crate has no such argument
#[test]
fn conv2d_transpose_falls_short_when_the_stride_does_not_divide() {
    use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;

    let mut down = Conv2D::new(1, (3, 3), vec![1, 8, 8, 1], (2, 2), Linear::new()).unwrap();
    let small = down.forward(&Array::ones((1, 8, 8, 1)).into_dyn()).unwrap();
    assert_eq!(small.shape(), &[1, 3, 3, 1]);

    let mut up =
        Conv2DTranspose::new(1, (3, 3), small.shape().to_vec(), (2, 2), Linear::new()).unwrap();
    let back = up.forward(&small).unwrap();
    assert_eq!(back.shape(), &[1, 7, 7, 1]);
}

/// The transposed forward pass IS the plain convolution's input gradient
///
/// This is the identity that defines the layer. Give a `Conv2D` and a `Conv2DTranspose` the same
/// flat kernel, the same stride, and the same padding mode. Make the transposed layer read the
/// convolution's `filters` channels and write its `channels` filters. Then
/// `Conv2D::backward(g)` and `Conv2DTranspose::forward(g)` must agree to the last bit. The 2
/// kernels need no permutation. `[kh, kw, Cin, F]` for the plain layer and
/// `[kh, kw, filters, channels]` for the transposed one are the same block once the roles swap.
///
/// A finite-difference check cannot catch a transposed axis, because it compares a layer against
/// itself. This test compares the layer against an independently written implementation of the
/// same map, so a moved axis fails here
#[test]
fn conv2d_transpose_forward_equals_the_convolution_input_gradient() {
    use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;

    // (H, W, kh, kw, sh, sw, Cin, F, same)
    let cases = [
        (
            8usize, 8usize, 3usize, 3usize, 1usize, 1usize, 2usize, 3usize, false,
        ),
        (9, 9, 3, 3, 2, 2, 2, 3, false),
        (9, 7, 4, 2, 1, 1, 3, 2, false),
        (8, 8, 3, 3, 1, 1, 2, 3, true),
        (8, 8, 3, 3, 2, 2, 2, 3, true),
        (9, 6, 5, 3, 3, 3, 2, 2, true),
    ];
    for (h, w, kh, kw, sh, sw, cin, f, same) in cases {
        let padding = if same {
            PaddingType::Same
        } else {
            PaddingType::Valid
        };
        let label = format!("{h}x{w} kernel {kh}x{kw} stride {sh}x{sw} same={same}");

        let taps: Vec<f32> = (0..kh * kw * cin * f)
            .map(|v| v as f32 * 0.0625 - 1.5)
            .collect();

        let mut down = Conv2D::new(f, (kh, kw), vec![1, h, w, cin], (sh, sw), Linear::new())
            .unwrap()
            .with_padding(padding);
        down.set_weights(
            Array4::from_shape_vec((kh, kw, cin, f), taps.clone()).unwrap(),
            Array1::zeros(f),
        )
        .unwrap();

        let small = down.forward(&ramp_of(&[1, h, w, cin])).unwrap();
        let grad = ramp_of(small.shape());
        let from_backward = down.backward(&grad).unwrap();

        // The transposed layer reads `f` channels and writes `cin` filters, so its kernel is the
        // very same flat block
        let mut up = Conv2DTranspose::new(
            cin,
            (kh, kw),
            small.shape().to_vec(),
            (sh, sw),
            Linear::new(),
        )
        .unwrap()
        .with_padding(padding);
        up.set_weights(
            Array4::from_shape_vec((kh, kw, cin, f), taps).unwrap(),
            Array1::zeros(cin),
        )
        .unwrap();
        let from_forward = up.forward(&grad).unwrap();

        assert_eq!(
            from_forward.shape(),
            from_backward.shape(),
            "[{label}] shape"
        );
        assert_allclose(&from_forward, &from_backward, 0.0f32);
    }
}

// Error paths

/// Each invalid scalar hyperparameter makes the constructor return InvalidParameter
#[test]
fn conv2d_transpose_invalid_scalar_param_errors() {
    // (filters, kernel, strides, label)
    let cases = [
        (0usize, (3usize, 3usize), (1usize, 1usize), "filters=0"),
        (1, (0, 3), (1, 1), "kernel height=0"),
        (1, (3, 0), (1, 1), "kernel width=0"),
        (1, (3, 3), (0, 1), "stride height=0"),
        (1, (3, 3), (1, 0), "stride width=0"),
    ];
    for (filters, kernel, strides, label) in cases {
        let result =
            Conv2DTranspose::new(filters, kernel, vec![1, 5, 5, 1], strides, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {label}, got {result:?}"
        );
    }
}

/// A wrong rank, or a 0 anywhere in the shape, makes the constructor return InvalidInput
#[test]
fn conv2d_transpose_invalid_input_shape_errors() {
    let cases = [
        (vec![5usize, 5, 1], "rank 3"),
        (vec![1, 5, 5, 5, 1], "rank 5"),
        (vec![1, 5, 5, 0], "0 channels"),
        (vec![1, 0, 5, 1], "0 height"),
    ];
    for (shape, label) in cases {
        let result = Conv2DTranspose::new(1, (3, 3), shape, (1, 1), Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "expected InvalidInput for {label}, got {result:?}"
        );
    }
}

/// An input smaller than the kernel is accepted, unlike a plain convolution
#[test]
fn conv2d_transpose_accepts_an_input_below_the_kernel_size() {
    let result = Conv2DTranspose::new(1, (5, 5), vec![1, 1, 1, 1], (1, 1), Linear::new());
    assert!(
        result.is_ok(),
        "a transposed convolution must accept an input below its kernel size, got {result:?}"
    );
}

/// forward with a tensor of the wrong rank returns InvalidInput
#[test]
fn conv2d_transpose_forward_wrong_ndim_errors() {
    let mut layer =
        Conv2DTranspose::new(1, (3, 3), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let result = layer.forward(&t3((1, 4, 4), vec![0.0; 16]));
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

/// forward with a channel count the kernel was not built for returns InvalidInput
#[test]
fn conv2d_transpose_forward_wrong_channel_count_errors() {
    let mut layer =
        Conv2DTranspose::new(1, (3, 3), vec![1, 4, 4, 2], (1, 1), Linear::new()).unwrap();
    let result = layer.forward(&Array::ones((1, 4, 4, 3)).into_dyn());
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

/// forward with a 0-length spatial axis returns InvalidInput instead of underflowing the padding
/// math
#[test]
fn conv2d_transpose_forward_empty_spatial_axis_errors() {
    let mut layer = Conv2DTranspose::new(1, (3, 3), vec![1, 4, 4, 1], (1, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let result = layer.forward(&Tensor::zeros(IxDyn(&[1, 0, 4, 1])));
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

/// backward before forward returns ForwardPassNotRun
#[test]
fn conv2d_transpose_backward_before_forward_errors() {
    let mut layer =
        Conv2DTranspose::new(1, (3, 3), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    let result = layer.backward(&Array::ones((1, 6, 6, 1)).into_dyn());
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {result:?}"
    );
}

/// backward with a gradient of the wrong shape returns ShapeMismatch
#[test]
fn conv2d_transpose_backward_checks_the_gradient_shape() {
    let mut layer =
        Conv2DTranspose::new(1, (3, 3), vec![1, 4, 4, 1], (1, 1), Linear::new()).unwrap();
    layer
        .forward(&Array::ones((1, 4, 4, 1)).into_dyn())
        .unwrap();
    // The forward output is 6x6, so a 5x5 gradient cannot belong to it
    let result = layer.backward(&Array::ones((1, 5, 5, 1)).into_dyn());
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {result:?}"
    );
}

/// set_weights rejects a kernel or a bias of the wrong shape
#[test]
fn conv2d_transpose_set_weights_shape_errors() {
    let mut layer =
        Conv2DTranspose::new(2, (3, 3), vec![1, 4, 4, 2], (1, 1), Linear::new()).unwrap();

    // The kernel is [3, 3, filters=2, channels=2]. The plain Conv2D order would be [3, 3, 2, 2]
    // too, so this case uses a shape that neither order accepts
    let wrong_weights = Array4::zeros((3, 3, 2, 3));
    let result = layer.set_weights(wrong_weights, Array1::zeros(2));
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape for the kernel, got {result:?}"
    );

    let result = layer.set_weights(Array4::zeros((3, 3, 2, 2)), Array1::zeros(3));
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape for the bias, got {result:?}"
    );
}

/// set_weights refuses a kernel laid out for the matching plain convolution
///
/// A `Conv2D` over 3 channels with 2 filters carries a \[3, 3, 3, 2\] kernel. The transposed
/// layer over the same channels and filters carries \[3, 3, 2, 3\]. The 2 shapes differ whenever
/// the counts differ, so an accidental swap is caught instead of silently transposing the map
#[test]
fn conv2d_transpose_set_weights_refuses_the_plain_kernel_order() {
    let mut layer =
        Conv2DTranspose::new(2, (3, 3), vec![1, 4, 4, 3], (1, 1), Linear::new()).unwrap();

    // The plain Conv2D order is [kh, kw, channels, filters]
    let plain_order = Array4::zeros((3, 3, 3, 2));
    let result = layer.set_weights(plain_order, Array1::zeros(2));
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape for the plain kernel order, got {result:?}"
    );

    // The transposed order is [kh, kw, filters, channels]
    assert!(
        layer
            .set_weights(Array4::zeros((3, 3, 2, 3)), Array1::zeros(2))
            .is_ok(),
        "the transposed kernel order must be accepted"
    );
}

/// The layer exposes no parameter before a backward pass has produced a gradient
#[test]
fn conv2d_transpose_exposes_no_parameters_before_backward() {
    let mut layer =
        Conv2DTranspose::new(2, (2, 2), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();
    assert!(
        layer.parameters().is_empty(),
        "a fresh layer has no gradient"
    );

    let x = ramp_of(&[1, 3, 3, 1]);
    let out = layer.forward(&x).unwrap();
    assert!(
        layer.parameters().is_empty(),
        "a forward pass alone produces no gradient"
    );

    layer.backward(&Tensor::ones(out.raw_dim())).unwrap();
    assert_eq!(
        layer.parameters().len(),
        2,
        "after backward the kernel and the bias are both exposed"
    );
}

/// Every rank rejects a tensor of the wrong rank, on both forward and predict
///
/// Each layer hand-writes its own rank literal, so a rank-2 test alone leaves the 1D and 3D
/// guards unpinned
#[test]
fn conv_transpose_layers_reject_a_tensor_of_the_wrong_rank() {
    let mut one = Conv1DTranspose::new(1, 2, vec![1, 3, 1], 1, Linear::new()).unwrap();
    let mut two = Conv2DTranspose::new(1, (2, 2), vec![1, 3, 3, 1], (1, 1), Linear::new()).unwrap();
    let mut three =
        Conv3DTranspose::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();

    // Every layer gets a tensor of a rank it does not accept
    let rank_4 = ramp_of(&[1, 3, 3, 3]);
    let rank_3 = ramp_of(&[1, 3, 3]);
    for (label, forward, predict) in [
        (
            "Conv1DTranspose",
            one.forward(&rank_4),
            one.predict(&rank_4),
        ),
        (
            "Conv2DTranspose",
            two.forward(&rank_3),
            two.predict(&rank_3),
        ),
        (
            "Conv3DTranspose",
            three.forward(&rank_4),
            three.predict(&rank_4),
        ),
    ] {
        assert!(
            matches!(forward, Err(Error::InvalidInput(_))),
            "[{label}] forward must reject a tensor of the wrong rank, got {forward:?}"
        );
        assert!(
            matches!(predict, Err(Error::InvalidInput(_))),
            "[{label}] predict must reject a tensor of the wrong rank, got {predict:?}"
        );
    }
}

/// Every rank reports ForwardPassNotRun, and names itself
#[test]
fn conv_transpose_layers_reject_backward_before_forward() {
    let mut one = Conv1DTranspose::new(1, 2, vec![1, 3, 1], 1, Linear::new()).unwrap();
    let mut three =
        Conv3DTranspose::new(1, (2, 2, 2), vec![1, 3, 3, 3, 1], (1, 1, 1), Linear::new()).unwrap();

    for (expected, result) in [
        ("Conv1DTranspose", one.backward(&ramp_of(&[1, 4, 1]))),
        (
            "Conv3DTranspose",
            three.backward(&ramp_of(&[1, 4, 4, 4, 1])),
        ),
    ] {
        match result {
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(name))) => {
                assert_eq!(
                    name, expected,
                    "the error must name the layer that raised it"
                );
            }
            other => panic!("[{expected}] expected ForwardPassNotRun, got {other:?}"),
        }
    }
}

/// predict returns the same values as forward at every rank
///
/// A save and load round trip cannot pin this, because both of its sides use `predict`
#[test]
fn conv_transpose_layers_predict_equals_forward() {
    let mut two = Conv2DTranspose::new(2, (3, 2), vec![1, 3, 4, 2], (2, 1), Linear::new())
        .unwrap()
        .with_random_state(23);
    let x = ramp_of(&[1, 3, 4, 2]);
    assert_allclose(&two.predict(&x).unwrap(), &two.forward(&x).unwrap(), 0.0f32);

    let mut three =
        Conv3DTranspose::new(2, (2, 2, 3), vec![1, 2, 3, 2, 2], (1, 2, 1), Linear::new())
            .unwrap()
            .with_random_state(23);
    let x = ramp_of(&[1, 2, 3, 2, 2]);
    assert_allclose(
        &three.predict(&x).unwrap(),
        &three.forward(&x).unwrap(),
        0.0f32,
    );
}

/// Conv3DTranspose seeds its kernel from the channel axis, and reports its axes in order
///
/// The channel count, the 3 kernel extents, and the filter count are all different here. An
/// index that reads the wrong axis cannot coincide with the right one
#[test]
fn conv3d_transpose_with_random_state_is_reproducible() {
    let make = || {
        Conv3DTranspose::new(3, (2, 4, 5), vec![1, 6, 7, 8, 2], (1, 1, 1), Linear::new())
            .unwrap()
            .with_random_state(31)
    };
    let first = make();
    let second = make();

    match (first.get_weights(), second.get_weights()) {
        (LayerWeight::Conv3DTranspose(a), LayerWeight::Conv3DTranspose(b)) => {
            assert_eq!(a.weight.shape(), &[2, 4, 5, 3, 2]);
            assert_eq!(a.weight.as_ref(), b.weight.as_ref());
            // Xavier bound for 2*4*5 taps at 2 channels and 3 filters is sqrt(6 / (40 * 5))
            let bound = (6.0f32 / 200.0).sqrt();
            assert!(
                a.weight.iter().all(|&v| v.abs() <= bound),
                "every weight must stay inside the Xavier bound {bound}"
            );
        }
        _other => panic!("expected LayerWeight::Conv3DTranspose variants"),
    }

    // Valid: 6 + 1, 7 + 3, 8 + 4 at stride 1, so the axes must not be reordered
    assert_eq!(first.output_shape(), "(1, 7, 10, 12, 3)");
}

// Layout

/// An input that is not in C order gives the same values as the same data in C order. The
/// output is in C order either way
#[test]
fn conv2d_transpose_accepts_an_input_that_is_not_in_c_order() {
    let base = ramp_of(&[2, 3, 4, 2]);
    // A permuted view shares the buffer under reordered strides, so it is not in C order
    let permuted = base.view().permuted_axes(IxDyn(&[0, 2, 1, 3])).to_owned();
    assert!(!permuted.is_standard_layout());

    let make = || {
        Conv2DTranspose::new(2, (3, 3), vec![2, 4, 3, 2], (2, 2), Linear::new())
            .unwrap()
            .with_random_state(5)
    };
    let mut layer = make();
    let from_view = layer.forward(&permuted).unwrap();

    let mut contiguous = Tensor::zeros(permuted.raw_dim());
    contiguous.assign(&permuted);
    let mut twin = make();
    let from_contiguous = twin.forward(&contiguous).unwrap();

    assert_eq!(from_view, from_contiguous);
    assert!(from_view.is_standard_layout());
}

/// Every layer of the family emits a gradient in C order
#[test]
fn conv_transpose_layers_emit_gradients_in_c_order() {
    let mut first = Conv1DTranspose::new(2, 3, vec![1, 4, 2], 2, Linear::new()).unwrap();
    let out = first.forward(&ramp_of(&[1, 4, 2])).unwrap();
    assert!(first.backward(&out).unwrap().is_standard_layout());

    let mut second = Conv2DTranspose::new(2, (3, 2), vec![1, 3, 4, 2], (2, 1), Linear::new())
        .unwrap()
        .with_padding(PaddingType::Same);
    let out = second.forward(&ramp_of(&[1, 3, 4, 2])).unwrap();
    assert!(second.backward(&out).unwrap().is_standard_layout());

    let mut third =
        Conv3DTranspose::new(2, (2, 2, 2), vec![1, 2, 2, 2, 1], (2, 2, 2), Linear::new()).unwrap();
    let out = third.forward(&ramp_of(&[1, 2, 2, 2, 1])).unwrap();
    assert!(third.backward(&out).unwrap().is_standard_layout());
}

/// A gradient that is not in C order is accepted, because the pass settles the layout first
#[test]
fn conv2d_transpose_backward_accepts_a_gradient_that_is_not_in_c_order() {
    let make = || {
        Conv2DTranspose::new(2, (2, 2), vec![1, 2, 3, 2], (1, 1), Linear::new())
            .unwrap()
            .with_random_state(13)
    };
    let x = ramp_of(&[1, 2, 3, 2]);

    // The forward output is [1, 3, 4, 2]. Building the gradient as [1, 4, 3, 2] and permuting the
    // 2 spatial axes gives that shape under reordered strides
    let source = ramp_of(&[1, 4, 3, 2]);
    let permuted = source.view().permuted_axes(IxDyn(&[0, 2, 1, 3])).to_owned();
    assert!(!permuted.is_standard_layout());
    let mut contiguous = Tensor::zeros(permuted.raw_dim());
    contiguous.assign(&permuted);

    let mut layer = make();
    layer.forward(&x).unwrap();
    let from_view = layer.backward(&permuted).unwrap();

    let mut twin = make();
    twin.forward(&x).unwrap();
    let from_contiguous = twin.backward(&contiguous).unwrap();

    assert_eq!(from_view, from_contiguous);
}

// Serial and parallel agreement

/// The parallel path gives the same bits as the serial path
///
/// Batch items never mix, so 1 call over a batch of 4 must equal 4 calls over a batch of 1. The
/// whole clears the engine's estimated-FLOPs gate, and each single item stays under it, so the
/// 2 sides really do take the 2 branches
#[test]
fn conv2d_transpose_parallel_path_matches_the_serial_path() {
    let gate = rustyml::tuning::conv::get_parallel_min_flops();
    let (batch, side, cin, filters, k) = (4usize, 48usize, 4usize, 8usize, 3usize);
    // Forward FLOPs are 2 * batch * in_plane * Cin * k_plane * F
    let whole = 2 * batch * side * side * cin * k * k * filters;
    assert!(
        whole >= gate,
        "the fixture no longer clears the engine's parallel gate: {whole} FLOPs against {gate}"
    );
    assert!(
        whole / batch < gate,
        "1 item must stay under the gate: {} FLOPs against {gate}",
        whole / batch
    );

    let x = ramp_of(&[batch, side, side, cin]);
    let make = || {
        Conv2DTranspose::new(
            filters,
            (k, k),
            vec![batch, side, side, cin],
            (2, 2),
            Linear::new(),
        )
        .unwrap()
        .with_random_state(29)
    };

    let mut parallel = make();
    let whole_out = parallel.forward(&x).unwrap();

    for b in 0..batch {
        let item = x
            .slice(ndarray::s![b..b + 1, .., .., ..])
            .to_owned()
            .into_dyn();
        let mut serial = make();
        let one_out = serial.forward(&item).unwrap();
        let expected = whole_out
            .slice(ndarray::s![b..b + 1, .., .., ..])
            .to_owned()
            .into_dyn();
        assert_eq!(
            one_out, expected,
            "batch item {b} differs between the paths"
        );
    }
}

/// Above the gate the weight gradient is still the plain count of the taps that landed
///
/// Every weight, every input, and every upstream gradient is 1, and the stride equals the
/// kernel. Each (tap, filter, channel) triple then receives exactly 1 contribution per input
/// position of every batch item. So each weight gradient is `batch * in_plane`, each bias
/// gradient is `batch * out_plane`, and each input gradient is `k_plane * filters`
#[test]
fn conv2d_transpose_parallel_gradient_counts_are_constant() {
    let gate = rustyml::tuning::conv::get_parallel_min_flops();
    let (batch, side, cin, filters, k) = (32usize, 16usize, 4usize, 8usize, 3usize);
    let in_plane = side * side;
    // The backward pass runs 2 GEMMs per item, so 4 * batch * in_plane * Cin * k_plane * F
    let flops = 4 * batch * in_plane * cin * k * k * filters;
    assert!(
        flops >= gate,
        "the fixture no longer clears the engine's parallel gate: {flops} FLOPs against {gate}"
    );

    let mut layer = Conv2DTranspose::new(
        filters,
        (k, k),
        vec![batch, side, side, cin],
        (k, k),
        Linear::new(),
    )
    .unwrap();
    layer
        .set_weights(Array4::ones((k, k, filters, cin)), Array1::zeros(filters))
        .unwrap();

    let x = Array::ones((batch, side, side, cin)).into_dyn();
    let out = layer.forward(&x).unwrap();
    // A stride equal to the kernel makes the windows disjoint. Each output position then collects
    // exactly 1 tap from 1 input position, over all `cin` channels
    let out_side = side * k;
    assert_eq!(out.shape(), &[batch, out_side, out_side, filters]);
    let (lo, hi) = out
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert_abs_diff_eq!(lo, cin as f32, epsilon = 1e-4f32);
    assert_abs_diff_eq!(hi, cin as f32, epsilon = 1e-4f32);

    let grad_in = layer.backward(&Tensor::ones(out.raw_dim())).unwrap();
    for &v in grad_in.iter() {
        assert_abs_diff_eq!(v, (k * k * filters) as f32, epsilon = 1e-3f32);
    }

    let params = layer.parameters();
    assert_eq!(params[0].grad.len(), k * k * filters * cin);
    for &v in params[0].grad.iter() {
        assert_abs_diff_eq!(v, (batch * in_plane) as f32, epsilon = 1e-1f32);
    }
    for &v in params[1].grad.iter() {
        assert_abs_diff_eq!(v, (batch * out_side * out_side) as f32, epsilon = 1e-1f32);
    }
}

// Model integration

/// A transposed convolution trains inside a model, and the loss falls
#[test]
fn conv2d_transpose_trains_inside_a_sequential_model() {
    let x = ramp_of(&[2, 3, 3, 1]);
    let y = Array::from_shape_vec((2, 1), vec![1.0f32, -1.0])
        .unwrap()
        .into_dyn();

    let mut model = Sequential::new();
    model
        .add(
            Conv2DTranspose::new(2, (3, 3), vec![2, 3, 3, 1], (1, 1), Linear::new())
                .unwrap()
                .with_random_state(17),
        )
        .add(Flatten::new(vec![2, 5, 5, 2]).unwrap())
        .add(Dense::new(50, 1, Linear::new()).unwrap())
        .compile(
            SGD::new(0.002, 0.0, false, 0.0).unwrap(),
            MeanSquaredError::new(),
        );

    let history = model.fit(&x, &y, 25).unwrap();
    let losses = history.loss();
    assert!(
        losses[losses.len() - 1] < losses[0],
        "the loss must fall: {} then {}",
        losses[0],
        losses[losses.len() - 1]
    );
}

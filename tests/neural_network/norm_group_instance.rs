//! Integration tests for GroupNormalization and InstanceNormalization: forward
//! values, error paths, mode behavior, eval-mode backward passthrough, and
//! predict == forward. Expected values come from the mathematical definition;
//! gradient correctness lives in gradient_check.rs

use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::traits::Layer;

use crate::common::assert_allclose;

// Helpers

/// Build a 1-D Tensor (parameter vector) from a slice
fn param1d(vals: &[f32]) -> Tensor {
    Array::from_shape_vec(vec![vals.len()], vals.to_vec())
        .unwrap()
        .into_dyn()
}

// GroupNormalization - forward value tests

/// GN with num_groups=1 folds all channels into one group, normalizing across the
/// whole (channels * spatial) volume
#[test]
fn group_norm_single_group_forward_values() {
    // shape: [batch=1, channels=2, spatial=3], channel_axis=1
    let mut gn = GroupNormalization::new(vec![1, 2, 3], 1, 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // mean=3.5, var=17.5/6, std=sqrt(var+eps)
    let std_val = 1.707_828_f32;
    let expected_flat = vec![
        -2.5 / std_val,
        -1.5 / std_val,
        -0.5 / std_val,
        0.5 / std_val,
        1.5 / std_val,
        2.5 / std_val,
    ];
    let expected = Array::from_shape_vec((1, 2, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN with num_groups=2 normalizes channels {0,1} and {2,3} as separate groups
#[test]
fn group_norm_two_groups_forward_values() {
    let mut gn = GroupNormalization::new(vec![1, 4, 2], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // group_size=4, var=1.25; both groups share centered=[-1.5,-0.5,0.5,1.5]
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let expected_flat = vec![
        -1.5 / std_val, // ch0 sp0
        -0.5 / std_val, // ch0 sp1
        0.5 / std_val,  // ch1 sp0
        1.5 / std_val,  // ch1 sp1
        -1.5 / std_val, // ch2 sp0
        -0.5 / std_val, // ch2 sp1
        0.5 / std_val,  // ch3 sp0
        1.5 / std_val,  // ch3 sp1
    ];
    let expected = Array::from_shape_vec((1, 4, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN normalizes each batch sample independently across two groups
#[test]
fn group_norm_two_batches_forward_values() {
    let mut gn = GroupNormalization::new(vec![2, 4, 3], 2, 1, 1e-5).unwrap();

    // batch0: channels [1..6, 7..12], batch1: channels [2..7, 8..13]
    let input = Array::from_shape_vec(
        (2, 4, 3),
        vec![
            // batch 0
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
            10.0, 11.0, 12.0, // ch3
            // batch 1
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let output = gn.forward(&input).unwrap();

    // All four groups share var=17.5/6 with offsets [-2.5,-1.5,-0.5,0.5,1.5,2.5]
    let std_val = (17.5_f32 / 6.0 + 1e-5).sqrt();
    let n = [-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5].map(|c| c / std_val);

    // Each group contributes 6 elements: ch_lo gets n[0..3], ch_hi gets n[3..6]
    let expected_flat = vec![
        // batch 0
        n[0], n[1], n[2], // ch0
        n[3], n[4], n[5], // ch1
        n[0], n[1], n[2], // ch2
        n[3], n[4], n[5], // ch3
        // batch 1 (same variance, same norms)
        n[0], n[1], n[2], n[3], n[4], n[5], n[0], n[1], n[2], n[3], n[4], n[5],
    ];
    let expected = Array::from_shape_vec((2, 4, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN applies custom per-channel gamma and beta after normalization
#[test]
fn group_norm_custom_gamma_beta_forward_values() {
    let mut gn = GroupNormalization::new(vec![1, 4, 2], 2, 1, 1e-5).unwrap();

    gn.set_weights(
        param1d(&[2.0, 3.0, 4.0, 5.0]),
        param1d(&[1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();

    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // group_size=4, var=1.25; each group's centers are [-1.5,-0.5,0.5,1.5]
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // output[0, ch, sp] = gamma[ch] * x_norm + beta[ch]
    let expected_flat = vec![
        c[0] * 2.0 + 1.0,
        c[1] * 2.0 + 1.0, // ch0
        c[2] * 3.0 + 2.0,
        c[3] * 3.0 + 2.0, // ch1
        c[0] * 4.0 + 3.0,
        c[1] * 4.0 + 3.0, // ch2
        c[2] * 5.0 + 4.0,
        c[3] * 5.0 + 4.0, // ch3
    ];
    let expected = Array::from_shape_vec((1, 4, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// A constant input has zero variance, so GN output is all zero (x_norm=0, beta=0)
#[test]
fn group_norm_constant_input_yields_zero_output() {
    let mut gn = GroupNormalization::new(vec![1, 3, 4], 1, 1, 1e-5).unwrap();

    let input = Array::from_elem((1, 3, 4), 5.0_f32).into_dyn();
    let output = gn.forward(&input).unwrap();

    let expected = Array::zeros((1, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// GN with channel_axis=2 yields the same normalized values as channel_axis=1,
/// reordered to channels-last layout
#[test]
fn group_norm_channels_last_vs_channels_first() {
    let std_val: f32 = (17.5 / 6.0_f32 + 1e-5).sqrt();

    // channels-first [1, 2, 3], channel_axis=1
    let mut gn_cf = GroupNormalization::new(vec![1, 2, 3], 1, 1, 1e-5).unwrap();
    let input_cf = Array::from_shape_vec((1, 2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let out_cf = gn_cf.forward(&input_cf).unwrap();

    // channels-last [1, 3, 2], channel_axis=2, layout [b, spatial, channel]
    let mut gn_cl = GroupNormalization::new(vec![1, 3, 2], 1, 2, 1e-5).unwrap();
    let input_cl = Array::from_shape_vec((1, 3, 2), vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap()
        .into_dyn();
    let out_cl = gn_cl.forward(&input_cl).unwrap();

    // n(v) = (v - 3.5) / std_val
    let n = |v: f32| (v - 3.5) / std_val;

    // channels-last output shape [1,3,2], indexed [b, sp, ch]
    let expected_cl = Array::from_shape_vec(
        (1, 3, 2),
        vec![n(1.0), n(4.0), n(2.0), n(5.0), n(3.0), n(6.0)],
    )
    .unwrap()
    .into_dyn();

    let expected_cf = Array::from_shape_vec(
        (1, 2, 3),
        vec![n(1.0), n(2.0), n(3.0), n(4.0), n(5.0), n(6.0)],
    )
    .unwrap()
    .into_dyn();

    assert_allclose(&out_cf, &expected_cf, 1e-5_f32);
    assert_allclose(&out_cl, &expected_cl, 1e-5_f32);
}

// GroupNormalization - predict == forward in eval mode

/// `predict` matches `forward`: GN always computes from-data statistics, with no
/// running mean/var and no mode dependence
#[test]
fn group_norm_predict_equals_forward() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    gn.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| 0.5 * v as f32 - 3.75).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let out_fwd = gn.forward(&input).unwrap();
    let out_pred = gn.predict(&input).unwrap();

    assert_allclose(&out_pred, &out_fwd, 1e-6_f32);
}

// GroupNormalization - constructor / forward error paths

/// Constructor rejects each invalid hyperparameter with InvalidParameter. Each row
/// makes exactly one argument bad: num_groups (zero), epsilon (zero / negative),
/// or channel_axis (zero = batch axis, or out of bounds for a 3-D input shape).
#[test]
fn group_norm_constructor_invalid_parameter_errors() {
    // (num_groups, channel_axis, epsilon, description)
    let cases = [
        (0, 1, 1e-5_f32, "num_groups=0"),
        (2, 1, 0.0_f32, "epsilon=0.0"),
        (2, 1, -1e-5_f32, "epsilon=-1e-5"),
        (2, 0, 1e-5_f32, "channel_axis=0 (batch axis)"),
        (2, 3, 1e-5_f32, "channel_axis=3 out of bounds"),
    ];

    for (num_groups, channel_axis, epsilon, desc) in cases {
        let err =
            GroupNormalization::new(vec![1, 4, 4], num_groups, channel_axis, epsilon).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for {}, got {:?}",
            desc,
            err
        );
    }
}

#[test]
fn group_norm_error_empty_input_shape() {
    let err = GroupNormalization::new(vec![], 2, 1, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// forward() returns InvalidParameter when num_groups does not divide num_channels
#[test]
fn group_norm_error_channels_not_divisible_by_groups_at_forward() {
    // 3 channels, 2 groups: 3 % 2 != 0
    let mut gn = GroupNormalization::new(vec![1, 3, 4], 2, 1, 1e-5).unwrap();
    let input = Array::ones((1, 3, 4)).into_dyn();
    let err = gn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for non-divisible channels/groups, got {:?}",
        err
    );
}

/// `backward` before `forward` returns `NnError::ForwardPassNotRun`
#[test]
fn group_norm_error_backward_before_forward() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    let grad = Array::ones((1, 4, 4)).into_dyn();
    let err = gn.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("GroupNormalization"))
        ),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

/// `set_weights` with a mismatched gamma shape fails with NnError::WeightShape
#[test]
fn group_norm_set_weights_shape_mismatch() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    // gamma should have shape [4] but shape [3] is passed
    let bad_gamma = param1d(&[1.0, 1.0, 1.0]);
    let beta = param1d(&[0.0, 0.0, 0.0, 0.0]);
    let err = gn.set_weights(bad_gamma, beta).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got {:?}",
        err
    );
}

// InstanceNormalization - forward value tests

/// IN normalizes each (batch, channel) pair independently over the spatial dimension
#[test]
fn instance_norm_forward_values() {
    let mut inn = InstanceNormalization::new(vec![1, 2, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    let std_val = (1.25_f32 + 1e-5).sqrt();
    // both channels have var=1.25, centered values [-1.5,-0.5,0.5,1.5]
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);
    let expected_flat = vec![c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3]];
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// IN applies the per-channel affine scale-shift step (gamma, beta)
#[test]
fn instance_norm_custom_gamma_beta_forward_values() {
    let mut inn = InstanceNormalization::new(vec![1, 2, 4], 1, 1e-5).unwrap();
    inn.set_weights(param1d(&[2.0, 3.0]), param1d(&[0.5, -0.5]))
        .unwrap();

    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // ch0: gamma=2, beta=0.5
    let ch0: Vec<f32> = c.iter().map(|&v| v * 2.0 + 0.5).collect();
    // ch1: gamma=3, beta=-0.5
    let ch1: Vec<f32> = c.iter().map(|&v| v * 3.0 + (-0.5)).collect();

    let expected_flat: Vec<f32> = ch0.into_iter().chain(ch1).collect();
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// IN normalizes each (batch, channel) instance independently; here every instance
/// is a ternary ramp with variance 2/3
#[test]
fn instance_norm_multiple_batches_forward_values() {
    // Each channel per batch-item is a ternary [-1,0,1] centered on a different offset
    let input = Array::from_shape_vec(
        (2, 3, 3),
        vec![
            // batch 0
            0.0_f32, 1.0, 2.0, // ch0: mean=1
            3.0, 4.0, 5.0, // ch1: mean=4
            -1.0, 0.0, 1.0, // ch2: mean=0
            // batch 1
            10.0, 11.0, 12.0, // ch0: mean=11
            -5.0, -4.0, -3.0, // ch1: mean=-4
            100.0, 101.0, 102.0, // ch2: mean=101
        ],
    )
    .unwrap()
    .into_dyn();

    let mut inn = InstanceNormalization::new(vec![2, 3, 3], 1, 1e-5).unwrap();
    let output = inn.forward(&input).unwrap();

    // var = ((-1)^2 + 0^2 + 1^2) / 3 = 2/3
    let var = 2.0_f32 / 3.0;
    let std_val = (var + 1e-5).sqrt();
    let x_norm = [-1.0_f32 / std_val, 0.0, 1.0 / std_val];

    // All 6 instances produce the same x_norm triple
    let expected_flat: Vec<f32> = std::iter::repeat_n(x_norm, 6).flatten().collect();
    let expected = Array::from_shape_vec((2, 3, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// Constant input gives zero output (var=0, x_norm=0, beta=0)
#[test]
fn instance_norm_constant_input_yields_zero_output() {
    let mut inn = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();
    let input = Array::from_elem((2, 3, 4), 7.0_f32).into_dyn();
    let output = inn.forward(&input).unwrap();
    let expected = Array::zeros((2, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// IN with channel_axis=2 gives the same normalized values as the channels-first
/// computation, stored in channels-last layout
#[test]
fn instance_norm_channels_last_layout() {
    let mut inn = InstanceNormalization::new(vec![1, 4, 2], 2, 1e-5).unwrap();

    // channels-last: [b, sp, ch]
    let input_cl =
        Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
            .unwrap()
            .into_dyn();

    let output = inn.forward(&input_cl).unwrap();

    let std_val = (1.25_f32 + 1e-5).sqrt();
    // both channels share centered values [-1.5,-0.5,0.5,1.5]
    let n = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // output shape [1,4,2]: [0, sp, 0] comes from ch0, [0, sp, 1] from ch1
    let expected_flat = vec![
        n[0], n[0], // sp0: ch0=n[0], ch1=n[0]
        n[1], n[1], // sp1
        n[2], n[2], // sp2
        n[3], n[3], // sp3
    ];
    let expected = Array::from_shape_vec((1, 4, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

// Equivalence: GroupNorm(num_groups == num_channels) == InstanceNorm

/// GroupNorm with num_groups == num_channels matches InstanceNorm: each channel of
/// each sample is normalized independently
#[test]
fn group_norm_full_groups_equals_instance_norm() {
    // Input: ch0=[1,2,3,4], ch1=[5,6,7,8], ch2=[10,11,12,13]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let input = Array::from_shape_vec((1, 3, 4), data.clone())
        .unwrap()
        .into_dyn();

    let mut gn = GroupNormalization::new(vec![1, 3, 4], 3, 1, 1e-5).unwrap();
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();

    let out_gn = gn.forward(&input).unwrap();
    let out_in = inn.forward(&input).unwrap();

    // The outputs should be numerically identical (same algorithm)
    assert_allclose(&out_gn, &out_in, 1e-6_f32);

    // Cross-check against hand-computed values: all 3 channels share var=1.25
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);
    let expected_flat = vec![
        c[0], c[1], c[2], c[3], // ch0
        c[0], c[1], c[2], c[3], // ch1
        c[0], c[1], c[2], c[3], // ch2
    ];
    let expected = Array::from_shape_vec((1, 3, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&out_gn, &expected, 1e-5_f32);
    assert_allclose(&out_in, &expected, 1e-5_f32);
}

/// GroupNorm and InstanceNorm stay equivalent under custom gamma/beta, confirming
/// the affine step matches too
#[test]
fn group_norm_full_groups_equals_instance_norm_with_affine() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Array::from_shape_vec((1, 2, 4), data).unwrap().into_dyn();

    let gamma = param1d(&[2.0, 0.5]);
    let beta = param1d(&[1.0, -1.0]);

    let mut gn = GroupNormalization::new(vec![1, 2, 4], 2, 1, 1e-5).unwrap();
    gn.set_weights(gamma.clone(), beta.clone()).unwrap();

    let mut inn = InstanceNormalization::new(vec![1, 2, 4], 1, 1e-5).unwrap();
    inn.set_weights(gamma, beta).unwrap();

    let out_gn = gn.forward(&input).unwrap();
    let out_in = inn.forward(&input).unwrap();

    assert_allclose(&out_gn, &out_in, 1e-6_f32);
}

// InstanceNormalization - predict == forward in eval mode

#[test]
fn instance_norm_predict_equals_forward() {
    let mut inn = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();
    inn.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec(
        (2, 3, 4),
        (0..24).map(|v| 0.5 * v as f32 - 5.75).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();

    let out_fwd = inn.forward(&input).unwrap();
    let out_pred = inn.predict(&input).unwrap();

    assert_allclose(&out_pred, &out_fwd, 1e-6_f32);
}

/// predict() matches forward() in TRAINING mode too, since statistics are always
/// recomputed from the input regardless of mode
#[test]
fn instance_norm_predict_equals_forward_training_mode() {
    let mut inn = InstanceNormalization::new(vec![1, 2, 4], 1, 1e-5).unwrap();
    inn.set_training_if_mode_dependent(true);

    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let out_fwd = inn.forward(&input).unwrap();
    let out_pred = inn.predict(&input).unwrap();

    assert_allclose(&out_pred, &out_fwd, 1e-6_f32);
}

// InstanceNormalization - constructor / forward error paths

/// Constructor rejects each invalid hyperparameter with InvalidParameter. Each row
/// makes exactly one argument bad: epsilon (zero / negative) or channel_axis
/// (zero = batch axis, or out of bounds for a 3-D input shape).
#[test]
fn instance_norm_constructor_invalid_parameter_errors() {
    // (channel_axis, epsilon, description)
    let cases = [
        (1, 0.0_f32, "epsilon=0.0"),
        (1, -1e-3_f32, "epsilon=-1e-3"),
        (0, 1e-5_f32, "channel_axis=0 (batch axis)"),
        (3, 1e-5_f32, "channel_axis=3 out of bounds"),
    ];

    for (channel_axis, epsilon, desc) in cases {
        let err = InstanceNormalization::new(vec![1, 3, 4], channel_axis, epsilon).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for {}, got {:?}",
            desc,
            err
        );
    }
}

#[test]
fn instance_norm_error_empty_input_shape() {
    let err = InstanceNormalization::new(vec![], 1, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// `backward` before `forward` returns `NnError::ForwardPassNotRun`
#[test]
fn instance_norm_error_backward_before_forward() {
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();
    let grad = Array::ones((1, 3, 4)).into_dyn();
    let err = inn.backward(&grad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::NeuralNetwork(NnError::ForwardPassNotRun("InstanceNormalization"))
        ),
        "expected ForwardPassNotRun, got {:?}",
        err
    );
}

/// `set_weights` with a mismatched gamma shape fails with NnError::WeightShape
#[test]
fn instance_norm_set_weights_shape_mismatch() {
    let mut inn = InstanceNormalization::new(vec![1, 4, 4], 1, 1e-5).unwrap();
    // gamma expects shape [4] but shape [3] is passed
    let bad_gamma = param1d(&[1.0, 1.0, 1.0]);
    let beta = param1d(&[0.0, 0.0, 0.0, 0.0]);
    let err = inn.set_weights(bad_gamma, beta).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got {:?}",
        err
    );
}

// GroupNormalization - output shape is identical to input shape

#[test]
fn group_norm_output_shape_matches_input() {
    let mut gn = GroupNormalization::new(vec![2, 6, 5], 3, 1, 1e-5).unwrap();
    let input = Array::ones((2, 6, 5)).into_dyn();
    let output = gn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 6, 5]);
}

// InstanceNormalization - output shape is identical to input shape

#[test]
fn instance_norm_output_shape_matches_input() {
    let mut inn = InstanceNormalization::new(vec![2, 4, 6], 1, 1e-5).unwrap();
    let input = Array::ones((2, 4, 6)).into_dyn();
    let output = inn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4, 6]);
}
// EVAL-mode backward: gradient passes through unchanged

/// GroupNormalization::backward in EVAL mode returns grad_output unchanged (bit-exact
/// copy, compared with eps=0)
#[test]
fn group_norm_backward_eval_mode_passes_gradient_through() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    gn.set_training_if_mode_dependent(false);

    // Forward in eval mode (still computes from-data stats), irrelevant to passthrough
    let input = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| 0.5 * v as f32 - 3.75).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    gn.forward(&input).unwrap();

    // Distinct per-element gradient
    let grad = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| v as f32 - 7.5).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    let grad_input = gn.backward(&grad).unwrap();

    assert_allclose(&grad_input, &grad, 0.0_f32);
}

/// InstanceNormalization::backward in EVAL mode returns grad_output unchanged
/// (bit-exact copy, asserted with exact equality)
#[test]
fn instance_norm_backward_eval_mode_passes_gradient_through() {
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();
    inn.set_training_if_mode_dependent(false);

    let input = Array::from_shape_vec(
        (1, 3, 4),
        (0..12).map(|v| 0.5 * v as f32 - 2.75).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    inn.forward(&input).unwrap();

    let grad = Array::from_shape_vec(
        (1, 3, 4),
        (0..12).map(|v| v as f32 - 5.5).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    let grad_input = inn.backward(&grad).unwrap();

    assert_allclose(&grad_input, &grad, 0.0_f32);
}

// Forward with <3D input must error (min-ndim guard)

/// GroupNormalization::forward with a 2-D input is rejected with InvalidInput by the
/// min-ndim guard
#[test]
fn group_norm_forward_below_3d_input_errors() {
    let mut gn = GroupNormalization::new(vec![4, 8], 2, 1, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn();
    let err = gn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for <3D input, got {:?}",
        err
    );
}

/// InstanceNormalization::forward with a 2-D input is rejected with InvalidInput by
/// the min-ndim guard
#[test]
fn instance_norm_forward_below_3d_input_errors() {
    let mut inn = InstanceNormalization::new(vec![4, 8], 1, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn();
    let err = inn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for <3D input, got {:?}",
        err
    );
}

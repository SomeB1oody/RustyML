//! Integration tests for GroupNormalization and InstanceNormalization: forward
//! values, error paths, mode behavior, eval-mode backward passthrough, and
//! predict == forward. Expected values come from the mathematical definition.
//! Gradient correctness lives in gradient_check.rs.
//!
//! Both layers read the crate's channels-last layout: an input is
//! `[batch, spatial..., channels]`, and the channel axis is always the trailing one.
//! A channel group is a contiguous sub-run of each position's channel vector. Statistics are
//! population statistics (divide by n, not n-1), taken per (sample, group) over that group's
//! channels across every spatial position.

use ndarray::Array;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use crate::common::assert_allclose;

// Helpers

/// Build a 1-D Tensor (parameter vector) from a slice
fn param1d(vals: &[f32]) -> Tensor {
    Array::from_shape_vec(vec![vals.len()], vals.to_vec())
        .unwrap()
        .into_dyn()
}

// GroupNormalization - forward value tests

/// GN with num_groups=1 folds all channels into 1 group, normalizing across the
/// whole (positions * channels) volume
#[test]
fn group_norm_single_group_forward_values() {
    // shape [batch=1, positions=3, channels=2]. 1 group spans both channels.
    let mut gn = GroupNormalization::new(vec![1, 3, 2], 1, 1e-5).unwrap();

    // Channels-last row-major: position p holds [channel0, channel1]
    let input = Array::from_shape_vec((1, 3, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // The single group holds all 6 values {1..6}: mean 3.5.
    // Population variance is (2.5^2 + 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 + 2.5^2) / 6 = 17.5 / 6.
    let std_val = (17.5_f32 / 6.0 + 1e-5).sqrt();
    let expected_flat = vec![
        -2.5 / std_val,
        -1.5 / std_val,
        -0.5 / std_val,
        0.5 / std_val,
        1.5 / std_val,
        2.5 / std_val,
    ];
    let expected = Array::from_shape_vec((1, 3, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN with num_groups=2 normalizes trailing-axis channels {0,1} and {2,3} as separate groups
#[test]
fn group_norm_two_groups_forward_values() {
    // shape [batch=1, positions=2, channels=4]. 2 groups of 2 channels each.
    let mut gn = GroupNormalization::new(vec![1, 2, 4], 2, 1e-5).unwrap();

    // position0 = [1, 2, 5, 6], position1 = [3, 4, 7, 8].
    // Group 0 (channels 0-1) holds {1, 2, 3, 4}, and group 1 (channels 2-3) holds {5, 6, 7, 8}.
    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // Group 0: mean 2.5. Group 1: mean 6.5. Both have the same spread.
    // Population variance is (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4 = 1.25, and both center to
    // [-1.5,-0.5,0.5,1.5]
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let expected_flat = vec![
        -1.5 / std_val, // pos0 ch0: 1 - 2.5
        -0.5 / std_val, // pos0 ch1: 2 - 2.5
        -1.5 / std_val, // pos0 ch2: 5 - 6.5
        -0.5 / std_val, // pos0 ch3: 6 - 6.5
        0.5 / std_val,  // pos1 ch0: 3 - 2.5
        1.5 / std_val,  // pos1 ch1: 4 - 2.5
        0.5 / std_val,  // pos1 ch2: 7 - 6.5
        1.5 / std_val,  // pos1 ch3: 8 - 6.5
    ];
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN normalizes each batch sample independently across 2 groups
#[test]
fn group_norm_two_batches_forward_values() {
    // shape [batch=2, positions=3, channels=4]. 2 groups of 2 channels each.
    let mut gn = GroupNormalization::new(vec![2, 3, 4], 2, 1e-5).unwrap();

    // Per channel: batch0 has ch0=[1,2,3], ch1=[4,5,6], ch2=[7,8,9], ch3=[10,11,12].
    // batch1 is the same, plus 1. Channels-last stores position p as
    // [ch0[p], ch1[p], ch2[p], ch3[p]]
    let input = Array::from_shape_vec(
        (2, 3, 4),
        vec![
            // batch 0
            1.0, 4.0, 7.0, 10.0, // position 0
            2.0, 5.0, 8.0, 11.0, // position 1
            3.0, 6.0, 9.0, 12.0, // position 2
            // batch 1 (every value increased by 1)
            2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0, 4.0, 7.0, 10.0, 13.0,
        ],
    )
    .unwrap()
    .into_dyn();

    let output = gn.forward(&input).unwrap();

    // Group 0 of batch 0 is channels 0-1 = {1,2,3,4,5,6} (mean 3.5).
    // Group 1 is channels 2-3 = {7..12} (mean 9.5).
    // Both, and both groups of batch 1, share the population variance.
    // (2.5^2 + 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 + 2.5^2) / 6 = 17.5 / 6.
    let std_val = (17.5_f32 / 6.0 + 1e-5).sqrt();
    let n = [-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5].map(|c| c / std_val);

    // batch 0, position 0 = [1-3.5, 4-3.5, 7-9.5, 10-9.5] = [-2.5, 0.5, -2.5, 0.5]
    let expected_flat = vec![
        // batch 0
        n[0], n[3], n[0], n[3], // position 0
        n[1], n[4], n[1], n[4], // position 1
        n[2], n[5], n[2], n[5], // position 2
        // batch 1 (uniformly shifted input, so identical normalized values)
        n[0], n[3], n[0], n[3], n[1], n[4], n[1], n[4], n[2], n[5], n[2], n[5],
    ];
    let expected = Array::from_shape_vec((2, 3, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// GN applies custom per-channel gamma and beta after normalization
#[test]
fn group_norm_custom_gamma_beta_forward_values() {
    let mut gn = GroupNormalization::new(vec![1, 2, 4], 2, 1e-5).unwrap();

    gn.set_weights(
        param1d(&[2.0, 3.0, 4.0, 5.0]),
        param1d(&[1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();

    // Same buffer as `group_norm_two_groups_forward_values`
    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // Both groups have variance 1.25 and center to [-1.5,-0.5,0.5,1.5]
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // output[0, p, ch] = gamma[ch] * x_norm[0, p, ch] + beta[ch]
    let expected_flat = vec![
        c[0] * 2.0 + 1.0, // pos0 ch0
        c[1] * 3.0 + 2.0, // pos0 ch1
        c[0] * 4.0 + 3.0, // pos0 ch2
        c[1] * 5.0 + 4.0, // pos0 ch3
        c[2] * 2.0 + 1.0, // pos1 ch0
        c[3] * 3.0 + 2.0, // pos1 ch1
        c[2] * 4.0 + 3.0, // pos1 ch2
        c[3] * 5.0 + 4.0, // pos1 ch3
    ];
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// A constant input has zero variance, so GN output is all zero (x_norm=0, beta=0)
#[test]
fn group_norm_constant_input_yields_zero_output() {
    // [batch=1, positions=3, channels=4], 1 group over all 4 channels
    let mut gn = GroupNormalization::new(vec![1, 3, 4], 1, 1e-5).unwrap();

    let input = Array::from_elem((1, 3, 4), 5.0_f32).into_dyn();
    let output = gn.forward(&input).unwrap();

    let expected = Array::zeros((1, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// The group boundary runs across the trailing (channel) axis. This test uses 2 groups of
/// wildly different scales, so a wrong split axis would blend them and change the results.
#[test]
fn group_norm_channel_axis_is_last() {
    // [batch=1, positions=2, channels=4], 2 groups of 2 channels
    let mut gn = GroupNormalization::new(vec![1, 2, 4], 2, 1e-5).unwrap();

    // position0 = [2, 4, 100, 200], position1 = [6, 8, 300, 400]
    let input = Array::from_shape_vec(
        (1, 2, 4),
        vec![2.0_f32, 4.0, 100.0, 200.0, 6.0, 8.0, 300.0, 400.0],
    )
    .unwrap()
    .into_dyn();

    let output = gn.forward(&input).unwrap();

    // Group 0 (channels 0-1) = {2, 4, 6, 8}: mean 5, variance (9 + 1 + 1 + 9) / 4 = 5
    let inv0 = 1.0 / (5.0_f32 + 1e-5).sqrt();
    // Group 1 (channels 2-3) = {100, 200, 300, 400}: mean 250.
    // Variance is (150^2 + 50^2 + 50^2 + 150^2) / 4 = 50000 / 4 = 12500.
    let inv1 = 1.0 / (12500.0_f32 + 1e-5).sqrt();

    let expected_flat = vec![
        -3.0 * inv0,   // 2 - 5
        -inv0,         // 4 - 5
        -150.0 * inv1, // 100 - 250
        -50.0 * inv1,  // 200 - 250
        1.0 * inv0,    // 6 - 5
        3.0 * inv0,    // 8 - 5
        50.0 * inv1,   // 300 - 250
        150.0 * inv1,  // 400 - 250
    ];
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

// GroupNormalization - predict == forward in eval mode

/// `predict` matches `forward`: GN always computes from-data statistics, with no
/// running mean/var and no mode dependence
#[test]
fn group_norm_predict_equals_forward() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1e-5).unwrap();
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
/// makes exactly 1 argument bad: num_groups (zero) or epsilon (zero or negative)
#[test]
fn group_norm_constructor_invalid_parameter_errors() {
    // (num_groups, epsilon, description)
    let cases = [
        (0, 1e-5_f32, "num_groups=0"),
        (2, 0.0_f32, "epsilon=0.0"),
        (2, -1e-5_f32, "epsilon=-1e-5"),
    ];

    for (num_groups, epsilon, desc) in cases {
        let err = GroupNormalization::new(vec![1, 4, 4], num_groups, epsilon).unwrap_err();
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
    let err = GroupNormalization::new(vec![], 2, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// forward() returns InvalidParameter when num_groups does not divide num_channels
#[test]
fn group_norm_error_channels_not_divisible_by_groups_at_forward() {
    // Trailing axis is the channel axis: 3 channels, 2 groups, and 3 % 2 != 0
    let mut gn = GroupNormalization::new(vec![1, 4, 3], 2, 1e-5).unwrap();
    let input = Array::ones((1, 4, 3)).into_dyn();
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
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1e-5).unwrap();
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
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1e-5).unwrap();
    // gamma should have shape [4] (the trailing channel axis) but shape [3] is passed
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

/// IN normalizes each (batch, channel) pair independently over the spatial positions
#[test]
fn instance_norm_forward_values() {
    // [batch=1, positions=4, channels=2]
    let mut inn = InstanceNormalization::new(vec![1, 4, 2], 1e-5).unwrap();

    // Channels-last: position p holds [ch0[p], ch1[p]], laying down ch0 = [1,2,3,4] and
    // ch1 = [5,6,7,8]
    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    // ch0 mean 2.5, ch1 mean 6.5. Both have population variance
    // (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4 = 1.25 and center to [-1.5,-0.5,0.5,1.5]
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);
    let expected_flat = vec![c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3]];
    let expected = Array::from_shape_vec((1, 4, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// IN applies the per-channel affine scale-shift step (gamma, beta)
#[test]
fn instance_norm_custom_gamma_beta_forward_values() {
    let mut inn = InstanceNormalization::new(vec![1, 4, 2], 1e-5).unwrap();
    inn.set_weights(param1d(&[2.0, 3.0]), param1d(&[0.5, -0.5]))
        .unwrap();

    // Same buffer as `instance_norm_forward_values`: ch0 = [1,2,3,4], ch1 = [5,6,7,8]
    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // Each position contributes [ch0: gamma=2, beta=0.5, ch1: gamma=3, beta=-0.5]
    let expected_flat: Vec<f32> = c
        .iter()
        .flat_map(|&v| [v * 2.0 + 0.5, v * 3.0 - 0.5])
        .collect();
    let expected = Array::from_shape_vec((1, 4, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// IN normalizes each (batch, channel) instance independently. Here every instance
/// is a ternary ramp with variance 2/3
#[test]
fn instance_norm_multiple_batches_forward_values() {
    // [batch=2, positions=3, channels=3]. Per channel the instances are
    // batch0: ch0=[0,1,2], ch1=[3,4,5], ch2=[-1,0,1]
    // batch1: ch0=[10,11,12], ch1=[-5,-4,-3], ch2=[100,101,102]
    // Channels-last stores position p as [ch0[p], ch1[p], ch2[p]]
    let input = Array::from_shape_vec(
        (2, 3, 3),
        vec![
            // batch 0
            0.0_f32, 3.0, -1.0, // position 0
            1.0, 4.0, 0.0, // position 1
            2.0, 5.0, 1.0, // position 2
            // batch 1
            10.0, -5.0, 100.0, // position 0
            11.0, -4.0, 101.0, // position 1
            12.0, -3.0, 102.0, // position 2
        ],
    )
    .unwrap()
    .into_dyn();

    let mut inn = InstanceNormalization::new(vec![2, 3, 3], 1e-5).unwrap();
    let output = inn.forward(&input).unwrap();

    // Every instance is a unit-step ramp of 3 values, so var = ((-1)^2 + 0^2 + 1^2) / 3 = 2/3
    let var = 2.0_f32 / 3.0;
    let std_val = (var + 1e-5).sqrt();
    let a = 1.0 / std_val;

    // At position 0 every channel sits 1 step below its mean, and at position 1 exactly on it.
    // At position 2 every channel sits 1 step above its mean, in both batches.
    let expected_flat: Vec<f32> = vec![
        -a, -a, -a, // batch 0, position 0
        0.0, 0.0, 0.0, // batch 0, position 1
        a, a, a, // batch 0, position 2
        -a, -a, -a, // batch 1, position 0
        0.0, 0.0, 0.0, // batch 1, position 1
        a, a, a, // batch 1, position 2
    ];
    let expected = Array::from_shape_vec((2, 3, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// Constant input gives zero output (var=0, x_norm=0, beta=0)
#[test]
fn instance_norm_constant_input_yields_zero_output() {
    let mut inn = InstanceNormalization::new(vec![2, 3, 4], 1e-5).unwrap();
    let input = Array::from_elem((2, 3, 4), 7.0_f32).into_dyn();
    let output = inn.forward(&input).unwrap();
    let expected = Array::zeros((2, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// IN takes its instances along the trailing (channel) axis. This test uses 2 channels of
/// very different spreads, so reading the axis anywhere else would mix them and change the results.
#[test]
fn instance_norm_channel_axis_is_last() {
    // [batch=1, positions=3, channels=2]: ch0 = [1,2,3], ch1 = [10,20,30]
    let mut inn = InstanceNormalization::new(vec![1, 3, 2], 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 3, 2), vec![1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    // ch0: mean 2, variance (1 + 0 + 1) / 3 = 2/3
    let inv0 = 1.0 / (2.0_f32 / 3.0 + 1e-5).sqrt();
    // ch1: mean 20, variance (100 + 0 + 100) / 3 = 200/3
    let inv1 = 1.0 / (200.0_f32 / 3.0 + 1e-5).sqrt();

    let expected_flat = vec![
        -inv0,        // pos0 ch0: 1 - 2
        -10.0 * inv1, // pos0 ch1: 10 - 20
        0.0,          // pos1 ch0: 2 - 2
        0.0,          // pos1 ch1: 20 - 20
        1.0 * inv0,   // pos2 ch0: 3 - 2
        10.0 * inv1,  // pos2 ch1: 30 - 20
    ];
    let expected = Array::from_shape_vec((1, 3, 2), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

// Equivalence: GroupNorm(num_groups == num_channels) == InstanceNorm

/// GroupNorm with num_groups == num_channels matches InstanceNorm: each channel of
/// each sample is normalized independently
#[test]
fn group_norm_full_groups_equals_instance_norm() {
    // [batch=1, positions=4, channels=3]: ch0=[1,2,3,4], ch1=[5,6,7,8], ch2=[10,11,12,13]
    let data: Vec<f32> = vec![
        1.0, 5.0, 10.0, // position 0
        2.0, 6.0, 11.0, // position 1
        3.0, 7.0, 12.0, // position 2
        4.0, 8.0, 13.0, // position 3
    ];
    let input = Array::from_shape_vec((1, 4, 3), data).unwrap().into_dyn();

    let mut gn = GroupNormalization::new(vec![1, 4, 3], 3, 1e-5).unwrap();
    let mut inn = InstanceNormalization::new(vec![1, 4, 3], 1e-5).unwrap();

    let out_gn = gn.forward(&input).unwrap();
    let out_in = inn.forward(&input).unwrap();

    // The outputs should be numerically identical (same algorithm)
    assert_allclose(&out_gn, &out_in, 1e-6_f32);

    // Cross-check against hand-computed values: each channel is a unit ramp of 4 values.
    // All 3 share population variance (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4 = 1.25.
    let std_val = (1.25_f32 + 1e-5).sqrt();
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);
    let expected_flat = vec![
        c[0], c[0], c[0], // position 0
        c[1], c[1], c[1], // position 1
        c[2], c[2], c[2], // position 2
        c[3], c[3], c[3], // position 3
    ];
    let expected = Array::from_shape_vec((1, 4, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&out_gn, &expected, 1e-5_f32);
    assert_allclose(&out_in, &expected, 1e-5_f32);
}

/// GroupNorm and InstanceNorm stay equivalent under custom gamma/beta, confirming
/// the affine step matches too
#[test]
fn group_norm_full_groups_equals_instance_norm_with_affine() {
    // [batch=1, positions=4, channels=2]: ch0=[1,2,3,4], ch1=[5,6,7,8]
    let data: Vec<f32> = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
    let input = Array::from_shape_vec((1, 4, 2), data).unwrap().into_dyn();

    let gamma = param1d(&[2.0, 0.5]);
    let beta = param1d(&[1.0, -1.0]);

    let mut gn = GroupNormalization::new(vec![1, 4, 2], 2, 1e-5).unwrap();
    gn.set_weights(gamma.clone(), beta.clone()).unwrap();

    let mut inn = InstanceNormalization::new(vec![1, 4, 2], 1e-5).unwrap();
    inn.set_weights(gamma, beta).unwrap();

    let out_gn = gn.forward(&input).unwrap();
    let out_in = inn.forward(&input).unwrap();

    assert_allclose(&out_gn, &out_in, 1e-6_f32);
}

// InstanceNormalization - predict == forward in eval mode

#[test]
fn instance_norm_predict_equals_forward() {
    let mut inn = InstanceNormalization::new(vec![2, 3, 4], 1e-5).unwrap();
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
    let mut inn = InstanceNormalization::new(vec![1, 4, 2], 1e-5).unwrap();
    inn.set_training_if_mode_dependent(true);

    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
        .unwrap()
        .into_dyn();

    let out_fwd = inn.forward(&input).unwrap();
    let out_pred = inn.predict(&input).unwrap();

    assert_allclose(&out_pred, &out_fwd, 1e-6_f32);
}

// InstanceNormalization - constructor / forward error paths

/// Constructor rejects a non-positive epsilon (zero / negative) with InvalidParameter
#[test]
fn instance_norm_constructor_invalid_parameter_errors() {
    // (epsilon, description)
    let cases = [(0.0_f32, "epsilon=0.0"), (-1e-3_f32, "epsilon=-1e-3")];

    for (epsilon, desc) in cases {
        let err = InstanceNormalization::new(vec![1, 3, 4], epsilon).unwrap_err();
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
    let err = InstanceNormalization::new(vec![], 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// `backward` before `forward` returns `NnError::ForwardPassNotRun`
#[test]
fn instance_norm_error_backward_before_forward() {
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1e-5).unwrap();
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
    let mut inn = InstanceNormalization::new(vec![1, 4, 4], 1e-5).unwrap();
    // gamma expects shape [4] (the trailing channel axis) but shape [3] is passed
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
    // [batch=2, positions=5, channels=6] split into 3 groups of 2 channels
    let mut gn = GroupNormalization::new(vec![2, 5, 6], 3, 1e-5).unwrap();
    let input = Array::ones((2, 5, 6)).into_dyn();
    let output = gn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 5, 6]);
}

// InstanceNormalization - output shape is identical to input shape

#[test]
fn instance_norm_output_shape_matches_input() {
    let mut inn = InstanceNormalization::new(vec![2, 4, 6], 1e-5).unwrap();
    let input = Array::ones((2, 4, 6)).into_dyn();
    let output = inn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4, 6]);
}
// EVAL-mode backward: gradient passes through unchanged

/// GroupNormalization::backward in EVAL mode returns grad_output unchanged (bit-exact
/// copy, compared with eps=0)
#[test]
fn group_norm_backward_eval_mode_passes_gradient_through() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1e-5).unwrap();
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
    let mut inn = InstanceNormalization::new(vec![1, 3, 4], 1e-5).unwrap();
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
    let mut gn = GroupNormalization::new(vec![4, 8], 2, 1e-5).unwrap();
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
    let mut inn = InstanceNormalization::new(vec![4, 8], 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn();
    let err = inn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for <3D input, got {:?}",
        err
    );
}

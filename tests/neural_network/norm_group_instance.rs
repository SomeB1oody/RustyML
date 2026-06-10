//! Integration tests for GroupNormalization and InstanceNormalization.
//!
//! All expected values are derived from the mathematical definition (mean, variance,
//! normalization, affine transform). No values were obtained by running the
//! implementation and recording its output.
//!
//! Backward/gradient correctness is already covered by gradient_check.rs, so
//! only forward values, error paths, mode behaviour, and predict==forward are tested here.

use ndarray::Array;
use rustyml::error::{Error, NnError};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::traits::Layer;

use crate::common::assert_allclose;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a 1-D Tensor (parameter vector) from a slice.
fn param1d(vals: &[f32]) -> Tensor {
    Array::from_shape_vec(vec![vals.len()], vals.to_vec())
        .unwrap()
        .into_dyn()
}

// ═══════════════════════════════════════════════════════════════════════════
// GroupNormalization — forward value tests
// ═══════════════════════════════════════════════════════════════════════════

/// GN with num_groups=1 folds all channels into a single group:
/// mean and variance are computed across the entire (channels × spatial) volume.
///
/// Hand derivation (batch=1, channels=2, spatial=3, num_groups=1, eps=1e-5):
///   input (channels-first) = [1, 2, 3, 4, 5, 6]
///   group_size = 2 × 3 = 6
///   mean = 21 / 6 = 3.5
///   var  = (2.5² + 1.5² + 0.5² + 0.5² + 1.5² + 2.5²) / 6
///        = 17.5 / 6 ≈ 2.91667
///   std  = sqrt(2.91667 + 1e-5) ≈ 1.70783
///   x_norm = (input − 3.5) / std   (same formula for every element)
///   output = gamma(=1) × x_norm + beta(=0) = x_norm
#[test]
fn group_norm_single_group_forward_values() {
    // shape: [batch=1, channels=2, spatial=3], channel_axis=1
    let mut gn = GroupNormalization::new(vec![1, 2, 3], 1, 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // Derived by hand (see doc comment above).
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

/// GN with num_groups=2, batch=1, channels=4, spatial=2.
///
/// Group 0 covers channels {0,1}, group 1 covers channels {2,3}.
/// Input (channels-first):
///   ch0=[1,2], ch1=[3,4] → group0 = [1,2,3,4]
///   ch2=[5,6], ch3=[7,8] → group1 = [5,6,7,8]
///
/// For each group (group_size = 4, mean offset differs but var is the same):
///   group0: mean=2.5, var=1.25, std≈1.11804, x_norm=[-1.34164,-0.44721,0.44721,1.34164]
///   group1: mean=6.5, var=1.25, std≈1.11804, x_norm=[-1.34164,-0.44721,0.44721,1.34164]
///
/// gamma=1, beta=0  →  output = x_norm (re-arranged to channels-first shape).
#[test]
fn group_norm_two_groups_forward_values() {
    let mut gn = GroupNormalization::new(vec![1, 4, 2], 2, 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 4, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = gn.forward(&input).unwrap();

    // x_norm for every element in its 4-element group:
    // group_size=4, var=1.25, std=sqrt(1.25+eps)≈1.11804
    let std_val = (1.25_f32 + 1e-5).sqrt();
    // group0 x_norm: element flat indices 0..3  →  centered=[-1.5,-0.5,0.5,1.5]
    // group1 x_norm: element flat indices 4..7  →  same centered values
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

/// GN with multiple batches: each batch sample is normalised independently.
///
/// batch=2, channels=4, spatial=3, num_groups=2, gamma=1, beta=0.
///   batch0 group0: [1,2,3,4,5,6], mean=3.5, var=17.5/6≈2.9167, std≈1.70783
///   batch0 group1: [7,8,9,10,11,12], mean=9.5, same var/std
///   batch1 group0: [2,3,4,5,6,7], mean=4.5, same var/std
///   batch1 group1: [8,9,10,11,12,13], mean=10.5, same var/std
///
/// All groups share the same variance (arithmetic ramp), so std is identical.
/// x_norm values are the same pattern (centered / std) for every group.
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

    // All four groups share var=17.5/6≈2.91667, std≈1.70783.
    // centered/std for each group uses offsets [-2.5,-1.5,-0.5,0.5,1.5,2.5] (same arithmetic ramp).
    let std_val = (17.5_f32 / 6.0 + 1e-5).sqrt();
    let n = [-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5].map(|c| c / std_val);

    // Each group contributes 6 normalised elements: ch_lo gets n[0..3], ch_hi gets n[3..6].
    let expected_flat = vec![
        // batch 0
        n[0], n[1], n[2], // ch0
        n[3], n[4], n[5], // ch1
        n[0], n[1], n[2], // ch2
        n[3], n[4], n[5], // ch3
        // batch 1 (same variance → same norms)
        n[0], n[1], n[2], n[3], n[4], n[5], n[0], n[1], n[2], n[3], n[4], n[5],
    ];
    let expected = Array::from_shape_vec((2, 4, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// Verify that a custom gamma and beta are applied correctly after normalisation.
///
/// Same setup as the two-group test (batch=1, channels=4, spatial=2, groups=2).
/// gamma=[2,3,4,5], beta=[1,2,3,4].
///
/// With each group's x_norm computed as before, output[ch] = gamma[ch]*x_norm[ch] + beta[ch].
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

    // group_size=4, var=1.25, std=sqrt(1.25+eps)
    let std_val = (1.25_f32 + 1e-5).sqrt();
    // Within group0, the 4 elements have centers: ch0 sp0→-1.5, ch0 sp1→-0.5, ch1 sp0→0.5, ch1 sp1→1.5
    // within group1: same pattern.
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // output[0, ch, sp] = gamma[ch] * x_norm + beta[ch]
    // ch0 (gamma=2, beta=1): c[0]*2+1, c[1]*2+1
    // ch1 (gamma=3, beta=2): c[2]*3+2, c[3]*3+2
    // ch2 (gamma=4, beta=3): c[0]*4+3, c[1]*4+3
    // ch3 (gamma=5, beta=4): c[2]*5+4, c[3]*5+4
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

/// A constant input (all elements equal) has zero variance after centering.
/// x_centered = 0 for every element, so x_norm = 0 / sqrt(eps), output = beta = 0.
#[test]
fn group_norm_constant_input_yields_zero_output() {
    let mut gn = GroupNormalization::new(vec![1, 3, 4], 1, 1, 1e-5).unwrap();

    let input = Array::from_elem((1, 3, 4), 5.0_f32).into_dyn();
    let output = gn.forward(&input).unwrap();

    let expected = Array::zeros((1, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// Channel-last layout (channel_axis=2) must produce the same normalised values as
/// the equivalent channel-first layout (channel_axis=1), just stored in channels-last order.
///
/// channels-first input  (shape [1,2,3]): data [1,2,3, 4,5,6]
/// channels-last  input  (shape [1,3,2]): same channel values, stored [1,4, 2,5, 3,6]
///   (each spatial step contains all channels).
///
/// After GroupNorm (1 group, gamma=1, beta=0) the output should contain the same
/// x_norm values re-ordered according to the respective layout.
#[test]
fn group_norm_channels_last_vs_channels_first() {
    let std_val: f32 = (17.5 / 6.0_f32 + 1e-5).sqrt();

    // channels-first [1, 2, 3], channel_axis=1
    let mut gn_cf = GroupNormalization::new(vec![1, 2, 3], 1, 1, 1e-5).unwrap();
    let input_cf = Array::from_shape_vec((1, 2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let out_cf = gn_cf.forward(&input_cf).unwrap();

    // channels-last [1, 3, 2], channel_axis=2
    let mut gn_cl = GroupNormalization::new(vec![1, 3, 2], 1, 2, 1e-5).unwrap();
    // layout [b, spatial, channel]: [1,4, 2,5, 3,6]
    let input_cl = Array::from_shape_vec((1, 3, 2), vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap()
        .into_dyn();
    let out_cl = gn_cl.forward(&input_cl).unwrap();

    // channels-first output flat: n(1), n(2), n(3), n(4), n(5), n(6)
    // where n(v) = (v - 3.5) / std_val
    let n = |v: f32| (v - 3.5) / std_val;

    // channels-last output shape [1,3,2]: [b, sp, ch]
    // [0,0,0]=n(1), [0,0,1]=n(4), [0,1,0]=n(2), [0,1,1]=n(5), [0,2,0]=n(3), [0,2,1]=n(6)
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

// ═══════════════════════════════════════════════════════════════════════════
// GroupNormalization — predict == forward in eval mode
// ═══════════════════════════════════════════════════════════════════════════

/// `predict` must produce the same output as `forward` (both compute from-data statistics
/// for GN — there is no running mean/var — and the layer is mode-independent for the output).
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

// ═══════════════════════════════════════════════════════════════════════════
// GroupNormalization — constructor / forward error paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn group_norm_error_num_groups_zero() {
    let err = GroupNormalization::new(vec![1, 4, 4], 0, 1, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn group_norm_error_epsilon_zero() {
    let err = GroupNormalization::new(vec![1, 4, 4], 2, 1, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn group_norm_error_epsilon_negative() {
    let err = GroupNormalization::new(vec![1, 4, 4], 2, 1, -1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn group_norm_error_channel_axis_zero() {
    // axis 0 is the batch axis — not permitted
    let err = GroupNormalization::new(vec![1, 4, 4], 2, 0, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn group_norm_error_channel_axis_out_of_bounds() {
    // input_shape has 3 dims (ndim=3), so valid axes are 1 and 2
    let err = GroupNormalization::new(vec![1, 4, 4], 2, 3, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
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

/// num_groups must divide num_channels evenly; if not, forward() returns InvalidParameter.
#[test]
fn group_norm_error_channels_not_divisible_by_groups_at_forward() {
    // 3 channels, 2 groups → 3 % 2 ≠ 0
    let mut gn = GroupNormalization::new(vec![1, 3, 4], 2, 1, 1e-5).unwrap();
    let input = Array::ones((1, 3, 4)).into_dyn();
    let err = gn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for non-divisible channels/groups, got {:?}",
        err
    );
}

/// `backward` before `forward` must return `NnError::ForwardPassNotRun`.
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

/// `set_weights` with a mismatched gamma shape must fail with an NnError::WeightShape.
#[test]
fn group_norm_set_weights_shape_mismatch() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    // gamma should have shape [4] but we pass shape [3]
    let bad_gamma = param1d(&[1.0, 1.0, 1.0]);
    let beta = param1d(&[0.0, 0.0, 0.0, 0.0]);
    let err = gn.set_weights(bad_gamma, beta).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got {:?}",
        err
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// InstanceNormalization — forward value tests
// ═══════════════════════════════════════════════════════════════════════════

/// IN normalises each (batch, channel) pair independently over the spatial dimension.
///
/// Hand derivation (batch=1, channels=2, spatial=4, channel_axis=1, eps=1e-5):
///   ch0=[1,2,3,4]: mean=2.5, var=1.25, std=sqrt(1.25+eps)≈1.11804
///   ch1=[5,6,7,8]: mean=6.5, var=1.25, std≈1.11804   (same variance)
///
/// x_norm(ch0) = [-1.34164, -0.44721, 0.44721, 1.34164]
/// x_norm(ch1) = same pattern (shifted mean; same var)
/// output = x_norm  (gamma=1, beta=0)
#[test]
fn instance_norm_forward_values() {
    let mut inn = InstanceNormalization::new(vec![1, 2, 4], 1, 1e-5).unwrap();

    let input = Array::from_shape_vec((1, 2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap()
        .into_dyn();

    let output = inn.forward(&input).unwrap();

    let std_val = (1.25_f32 + 1e-5).sqrt();
    // both channels have var=1.25; their centered values are always [-1.5,-0.5,0.5,1.5]
    let c = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);
    let expected_flat = vec![c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3]];
    let expected = Array::from_shape_vec((1, 2, 4), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// Verify the affine scale-shift step: gamma and beta per channel.
///
/// Same setup as above; gamma=[2, 3], beta=[0.5, -0.5].
/// output[ch] = gamma[ch] * x_norm[ch] + beta[ch].
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

/// Multiple batches: each (batch, channel) instance is independent.
///
/// batch=2, channels=3, spatial=3, channel_axis=1.
/// The 6 instances each have their own mean/var; we pick values so that the variance
/// is always 2/3 = 0.6667 (symmetric ternary: mean=0, vals=[-1,0,1]).
///
///   Every instance: input=[mean-1, mean, mean+1], var=2/3, std=sqrt(2/3+eps)
///   x_norm = [-1, 0, 1] / std
///   output = x_norm  (default gamma=1, beta=0)
#[test]
fn instance_norm_multiple_batches_forward_values() {
    // Each channel per batch-item is a ternary [-1,0,1] centred on a different offset.
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

    // var = ((−1)² + 0² + 1²) / 3 = 2/3
    let var = 2.0_f32 / 3.0;
    let std_val = (var + 1e-5).sqrt();
    let x_norm = [-1.0_f32 / std_val, 0.0, 1.0 / std_val];

    // All 6 instances produce the same x_norm triple.
    let expected_flat: Vec<f32> = std::iter::repeat_n(x_norm, 6).flatten().collect();
    let expected = Array::from_shape_vec((2, 3, 3), expected_flat)
        .unwrap()
        .into_dyn();

    assert_allclose(&output, &expected, 1e-5_f32);
}

/// Constant input → zero output (var=0, x_norm=0, output=beta=0).
#[test]
fn instance_norm_constant_input_yields_zero_output() {
    let mut inn = InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap();
    let input = Array::from_elem((2, 3, 4), 7.0_f32).into_dyn();
    let output = inn.forward(&input).unwrap();
    let expected = Array::zeros((2, 3, 4)).into_dyn();
    assert_allclose(&output, &expected, 1e-6_f32);
}

/// Channels-last layout: IN with channel_axis=2 must give the same normalised
/// values as the equivalent channels-first computation, stored in the returned
/// channels-last tensor.
///
/// Input (channels-last, shape [1,4,2], channel_axis=2):
///   flat = [1,5, 2,6, 3,7, 4,8]  ← (sp0: ch0=1,ch1=5), (sp1: ch0=2,ch1=6), …
/// Equivalent channels-first input: ch0=[1,2,3,4], ch1=[5,6,7,8].
/// x_norm(ch0) = x_norm(ch1) = [-1.34164,-0.44721,0.44721,1.34164] (var=1.25).
/// Output (channels-last, shape [1,4,2]):
///   flat = [norm0(1),norm1(5), norm0(2),norm1(6), …]
///         = [norm0[0],norm1[0], norm0[1],norm1[1], norm0[2],norm1[2], norm0[3],norm1[3]]
///   Since norm0 == norm1 this is simply pairs of equal values.
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
    // centered for ch0 = [-1.5,-0.5,0.5,1.5] (mean=2.5)
    // centered for ch1 = [-1.5,-0.5,0.5,1.5] (mean=6.5, same variance)
    let n = [-1.5_f32, -0.5, 0.5, 1.5].map(|v| v / std_val);

    // output shape [1,4,2]: [0, sp, 0] comes from ch0, [0, sp, 1] from ch1.
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

// ═══════════════════════════════════════════════════════════════════════════
// POSITIVE EQUIVALENCE: GroupNorm(num_groups == num_channels) == InstanceNorm
// ═══════════════════════════════════════════════════════════════════════════

/// When `num_groups == num_channels`, GroupNorm normalises each channel of each
/// sample independently — identical to InstanceNorm.
///
/// batch=1, channels=3, spatial=4, channel_axis=1.
/// Both layers must produce identical floating-point outputs.
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

    // The outputs should be numerically identical (same algorithm).
    assert_allclose(&out_gn, &out_in, 1e-6_f32);

    // Cross-check against hand-computed values for all 3 channels.
    // ch0: mean=2.5, ch1: mean=6.5, ch2: mean=11.5 — all same var=1.25
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

/// Extended equivalence check with custom gamma/beta (not just the defaults),
/// to confirm the affine step is also identical.
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

// ═══════════════════════════════════════════════════════════════════════════
// InstanceNormalization — predict == forward in eval mode
// ═══════════════════════════════════════════════════════════════════════════

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

/// predict() must also match forward() when the layer is in TRAINING mode.
/// Both GroupNorm and InstanceNorm always re-compute statistics from the
/// input, so mode does not affect the output values.
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

// ═══════════════════════════════════════════════════════════════════════════
// InstanceNormalization — constructor / forward error paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn instance_norm_error_epsilon_zero() {
    let err = InstanceNormalization::new(vec![1, 3, 4], 1, 0.0).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn instance_norm_error_epsilon_negative() {
    let err = InstanceNormalization::new(vec![1, 3, 4], 1, -1e-3).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn instance_norm_error_channel_axis_zero() {
    let err = InstanceNormalization::new(vec![1, 3, 4], 0, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn instance_norm_error_channel_axis_out_of_bounds() {
    // 3-D input: valid channel axes are 1 and 2
    let err = InstanceNormalization::new(vec![1, 3, 4], 3, 1e-5).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
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

/// `backward` before `forward` must return `NnError::ForwardPassNotRun`.
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

/// `set_weights` with a mismatched gamma shape must fail with NnError::WeightShape.
#[test]
fn instance_norm_set_weights_shape_mismatch() {
    let mut inn = InstanceNormalization::new(vec![1, 4, 4], 1, 1e-5).unwrap();
    // gamma expects shape [4]; we pass [3]
    let bad_gamma = param1d(&[1.0, 1.0, 1.0]);
    let beta = param1d(&[0.0, 0.0, 0.0, 0.0]);
    let err = inn.set_weights(bad_gamma, beta).unwrap_err();
    assert!(
        matches!(err, Error::NeuralNetwork(NnError::WeightShape { .. })),
        "expected WeightShape error, got {:?}",
        err
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// GroupNormalization — output shape is identical to input shape
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn group_norm_output_shape_matches_input() {
    let mut gn = GroupNormalization::new(vec![2, 6, 5], 3, 1, 1e-5).unwrap();
    let input = Array::ones((2, 6, 5)).into_dyn();
    let output = gn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 6, 5]);
}

// ═══════════════════════════════════════════════════════════════════════════
// InstanceNormalization — output shape is identical to input shape
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn instance_norm_output_shape_matches_input() {
    let mut inn = InstanceNormalization::new(vec![2, 4, 6], 1, 1e-5).unwrap();
    let input = Array::ones((2, 4, 6)).into_dyn();
    let output = inn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4, 6]);
}
// ═══════════════════════════════════════════════════════════════════════════
// EVAL-mode backward: gradient passes through unchanged
// ═══════════════════════════════════════════════════════════════════════════

/// GroupNormalization::backward in EVAL mode returns grad_output unchanged.
///
/// Derivation:
///   With training == false, backward takes the early return
///   `if !self.training { return Ok(grad_output.clone()) }` (GN source 669-673)
///   BEFORE any channels-first permutation or cache read, so the returned
///   input-gradient is a bit-exact copy of grad_output (the layer contributes no
///   Jacobian during inference). We use all-distinct grad values and compare with eps=0.
#[test]
fn group_norm_backward_eval_mode_passes_gradient_through() {
    let mut gn = GroupNormalization::new(vec![1, 4, 4], 2, 1, 1e-5).unwrap();
    gn.set_training_if_mode_dependent(false);

    // Forward in eval mode (still computes from-data stats); irrelevant to the passthrough.
    let input = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| 0.5 * v as f32 - 3.75).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    gn.forward(&input).unwrap();

    // Distinct per-element gradient.
    let grad = Array::from_shape_vec(
        (1, 4, 4),
        (0..16).map(|v| v as f32 - 7.5).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_dyn();
    let grad_input = gn.backward(&grad).unwrap();

    assert_allclose(&grad_input, &grad, 0.0_f32);
}

/// InstanceNormalization::backward in EVAL mode returns grad_output unchanged.
///
/// Derivation:
///   With training == false, backward takes the early return
///   `if !self.training { return Ok(grad_output.clone()) }` (IN source 539-543),
///   so the input-gradient is a bit-exact copy of grad_output. We assert exact equality.
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

// ═══════════════════════════════════════════════════════════════════════════
// Forward with <3D input must error (min-ndim guard)
// ═══════════════════════════════════════════════════════════════════════════

/// GroupNormalization::forward with a 2-D input is rejected with InvalidInput.
///
/// Derivation:
///   The constructor accepts a 2-D input_shape [4, 8] (channel_axis=1 is valid for ndim=2:
///   1 != 0 and 1 < 2; param_shape becomes [8]). At forward time, with a matching 2-D input
///   [4, 8], `validate_input_shape` passes, but `validate_min_input_ndim(2, 3, "Group
///   normalization")` (GN source 177) fires because 2 < 3, returning
///   Error::invalid_input("Group normalization expects at least 3D input, got 2D")
///   => Error::InvalidInput. (This check precedes validate_num_groups at line 189, so
///   num_groups=2 is moot.)
#[test]
fn group_norm_forward_below_3d_input_errors() {
    let mut gn = GroupNormalization::new(vec![4, 8], 2, 1, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn(); // ndim = 2
    let err = gn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for <3D input, got {:?}",
        err
    );
}

/// InstanceNormalization::forward with a 2-D input is rejected with InvalidInput.
///
/// Derivation:
///   The constructor accepts a 2-D input_shape [4, 8] (channel_axis=1 valid for ndim=2;
///   param_shape becomes [8]). At forward time, with a matching 2-D input [4, 8],
///   `validate_min_input_ndim(2, 3, "Instance normalization")` (IN source 150) fires
///   because 2 < 3, returning
///   Error::invalid_input("Instance normalization expects at least 3D input, got 2D")
///   => Error::InvalidInput.
#[test]
fn instance_norm_forward_below_3d_input_errors() {
    let mut inn = InstanceNormalization::new(vec![4, 8], 1, 1e-5).unwrap();
    let input = Array::ones((4, 8)).into_dyn(); // ndim = 2
    let err = inn.forward(&input).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for <3D input, got {:?}",
        err
    );
}

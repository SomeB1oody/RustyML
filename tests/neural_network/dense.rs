//! Integration tests for the Dense and Flatten layers.
//!
//! Covers forward values, error paths, parameter counts, and the named weight shapes.
//! `gradient_check.rs` covers gradient values. This file does not duplicate them.

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array2, Array3, Array4};
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::linear::Linear;
use rustyml::neural_network::layers::activation::relu::ReLU;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::traits::Layer;
use rustyml::{error::Error, neural_network::NnError};

use super::common::{GateGuard, assert_allclose, named};

// helpers

/// Build a 2D Tensor from row-major data
fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Array2::from_shape_vec((rows, cols), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 3D Tensor from row-major data
fn t3(a: usize, b: usize, c: usize, data: Vec<f32>) -> Tensor {
    Array3::from_shape_vec((a, b, c), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a 4D Tensor from row-major data
fn t4(a: usize, b: usize, c: usize, d: usize, data: Vec<f32>) -> Tensor {
    Array4::from_shape_vec((a, b, c, d), data)
        .expect("shape/data mismatch")
        .into_dyn()
}

/// Build a Dense(2 -> 2, Linear) with row-major weight matrix (2x2) and bias (1x2)
fn dense_2x2_with_weights(w_flat: Vec<f32>, b_flat: Vec<f32>) -> Dense {
    let mut d = Dense::new(2, 2, Linear::new()).unwrap();
    let w = Array2::from_shape_vec((2, 2), w_flat).unwrap();
    let b = Array2::from_shape_vec((1, 2), b_flat).unwrap();
    d.set_weights(w, b).unwrap();
    d
}

// Dense: constructor validation

#[test]
fn dense_new_rejects_zero_dim() {
    // Each row zeroes one of input_dim or units and expects InvalidParameter.
    // `which` names the offending argument in the failure message.
    let cases: [(usize, usize, &str); 2] = [(0, 4, "input_dim=0"), (4, 0, "units=0")];
    for (input_dim, units, which) in cases {
        let result = Dense::new(input_dim, units, Linear::new());
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for {}, got {:?}",
            which,
            result
        );
    }
}

#[test]
fn dense_new_accepts_valid_dims() {
    Dense::new(3, 5, Linear::new()).unwrap();
}

// Dense: param_count

/// param_count = input_dim * units + units (weights + bias elements)
#[test]
fn dense_param_count_2x2() {
    use rustyml::neural_network::layers::ParamCounts;
    let d = Dense::new(2, 2, Linear::new()).unwrap();
    // 2*2 weights + 2 bias = 6
    assert_eq!(d.param_count(), ParamCounts::trainable(6));
}

#[test]
fn dense_param_count_3x5() {
    use rustyml::neural_network::layers::ParamCounts;
    let d = Dense::new(3, 5, Linear::new()).unwrap();
    // 3*5 weights + 5 bias = 20
    assert_eq!(d.param_count(), ParamCounts::trainable(20));
}

// Dense: forward, identity weight gives output equal to input

/// Identity weight with zero bias and Linear activation passes input through unchanged
#[test]
fn dense_forward_identity_weight_output_equals_input() {
    let mut d = dense_2x2_with_weights(
        vec![1.0, 0.0, 0.0, 1.0], // identity
        vec![0.0, 0.0],
    );
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_allclose(&out, &expected, 1e-6_f32);
}

// Dense: forward, known non-identity weight and bias

/// Forward computes X*W + b with a diagonal weight and nonzero bias under Linear activation
#[test]
fn dense_forward_known_weights_and_bias() {
    let mut d = dense_2x2_with_weights(
        vec![2.0, 0.0, 0.0, 3.0], // W = diag(2,3)
        vec![1.0, 2.0],           // b = [1, 2]
    );
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![3.0, 8.0, 7.0, 14.0]);
    assert_allclose(&out, &expected, 1e-5_f32);
}

// Dense: forward, ReLU activation zeroes negative pre-activations

/// ReLU activation clamps negative pre-activations to zero
#[test]
fn dense_forward_relu_zeroes_negative_preactivations() {
    let mut d = Dense::new(2, 2, ReLU::new()).unwrap();
    let w = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 1.0, -1.0]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    d.set_weights(w, b).unwrap();

    let x = t2(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(2, 2, vec![3.0, 0.0, 4.0, 0.0]);
    assert_allclose(&out, &expected, 1e-6_f32);
}

// Dense: forward, single batch item (batch_size equal to 1)

/// Forward on a 3 -> 2 Linear layer with a single-row input
#[test]
fn dense_forward_3_to_2_linear_single_row() {
    let mut d = Dense::new(3, 2, Linear::new()).unwrap();
    let w = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    d.set_weights(w, b).unwrap();

    let x = t2(1, 3, vec![1.0, 2.0, 3.0]);
    let out = d.forward(&x).unwrap();
    let expected = t2(1, 2, vec![22.0, 29.0]);
    assert_allclose(&out, &expected, 1e-5_f32);
}

// Dense: predict equal to forward in eval mode (no activation side effects)

/// Dense has no mode-dependent behavior, so predict and forward produce identical outputs
#[test]
fn dense_predict_equals_forward() {
    let mut d = dense_2x2_with_weights(vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -0.5]);
    let x = t2(2, 2, vec![1.0, -1.0, 0.5, 2.0]);

    let fwd = d.forward(&x).unwrap();
    let pred = d.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// Dense: input of rank 3 or more, where the last axis is the only contracted axis

/// Dense(2 -> 3, Linear) with the fixed kernel [[1, 2, 3], [4, 5, 6]] and the bias
/// [0.5, -0.5, 1.0]. Shared by the cases with an input of rank 3 or more
fn dense_2_to_3_ramp() -> Dense {
    let mut d = Dense::new(2, 3, Linear::new()).unwrap();
    let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((1, 3), vec![0.5, -0.5, 1.0]).unwrap();
    d.set_weights(w, b).unwrap();
    d
}

/// The rank-3 forward pass gives every leading position the same kernel
///
/// The layer contracts the last axis only, so a Dense(3) over a [2, 3, 2] input keeps the
/// [2, 3] kernel and returns [2, 3, 3]. A per-timestep kernel would need 3 times the weights
#[test]
fn dense_forward_rank_3_shares_1_kernel_over_the_leading_axes() {
    let mut d = dense_2_to_3_ramp();
    let x = t3(2, 3, 2, (1..=12).map(|v| v as f32).collect());

    let out = d.forward(&x).unwrap();

    // Row [a, b] gives [a + 4b + 0.5, 2a + 5b - 0.5, 3a + 6b + 1]
    let expected = t3(
        2,
        3,
        3,
        vec![
            9.5, 11.5, 16.0, 19.5, 25.5, 34.0, 29.5, 39.5, 52.0, 39.5, 53.5, 70.0, 49.5, 67.5,
            88.0, 59.5, 81.5, 106.0,
        ],
    );
    assert_allclose(&out, &expected, 1e-4_f32);
}

/// The rank-4 result equals the result of the same values given as rank 2
///
/// The leading axes fold into 1 row axis, so both calls run the very same matrix product.
/// The values are therefore bit-identical, not merely close
#[test]
fn dense_forward_rank_4_equals_the_folded_rank_2_result() {
    let data: Vec<f32> = (0..24).map(|v| v as f32 * 0.25 - 3.0).collect();
    let x4 = t4(1, 2, 6, 2, data.clone());
    let x2 = t2(12, 2, data);

    let out4 = dense_2_to_3_ramp().forward(&x4).unwrap();
    let out2 = dense_2_to_3_ramp().forward(&x2).unwrap();

    assert_eq!(
        out4.shape(),
        &[1, 2, 6, 3],
        "the rank must survive the fold"
    );
    let flat4: Vec<f32> = out4.iter().cloned().collect();
    let flat2: Vec<f32> = out2.iter().cloned().collect();
    assert_eq!(flat4, flat2, "1 fold, so 1 product, so the same bits");
}

/// The rank-3 backward pass gives the 3 gradients of the folded product
///
/// The bias gradient sums over every axis except the last one, so it stays [1, units]. The
/// weight gradient folds both operands to 2D first, so it stays [input_dim, units]. Only the
/// input gradient goes back to the rank of the input
#[test]
fn dense_backward_rank_3_produces_the_3_gradients_of_the_fold() {
    let mut d = dense_2_to_3_ramp();
    let x = t3(2, 3, 2, (1..=12).map(|v| v as f32).collect());
    d.forward(&x).unwrap();

    // A gradient that differs in every position pins the orientation of both products
    let grad_output = t3(2, 3, 3, (1..=18).map(|v| v as f32).collect());
    let grad_input = d.backward(&grad_output).unwrap();

    // grad_input[r] = G[r] * W^T, back at the rank of the input
    let expected_input = t3(
        2,
        3,
        2,
        vec![
            14.0, 32.0, 32.0, 77.0, 50.0, 122.0, 68.0, 167.0, 86.0, 212.0, 104.0, 257.0,
        ],
    );
    assert_eq!(grad_input.shape(), x.shape());
    assert_allclose(&grad_input, &expected_input, 1e-3_f32);

    let params = d.parameters();
    // grad_weight = X2^T * G2, summed over all 6 folded rows
    let expected_weight = [411.0_f32, 447.0, 483.0, 462.0, 504.0, 546.0];
    // grad_bias sums the 6 folded rows, 1 sum for each unit
    let expected_bias = [51.0_f32, 57.0, 63.0];
    assert_eq!(params[0].grad.len(), expected_weight.len());
    assert_eq!(params[1].grad.len(), expected_bias.len());
    for (got, want) in params[0].grad.iter().zip(expected_weight.iter()) {
        assert_abs_diff_eq!(*got, *want, epsilon = 1e-2);
    }
    for (got, want) in params[1].grad.iter().zip(expected_bias.iter()) {
        assert_abs_diff_eq!(*got, *want, epsilon = 1e-3);
    }
}

/// A rank-3 input that is not in C order gives the same values as its C-order copy
///
/// `permuted_axes` reorders the strides only, and `to_owned` keeps them. The fold must then
/// copy the values in logical order instead of reading the buffer as it lies
#[test]
fn dense_forward_rank_3_accepts_an_input_that_is_not_in_c_order() {
    use ndarray::IxDyn;

    let base = t3(2, 3, 2, (1..=12).map(|v| v as f32).collect());
    let permuted: Tensor = base.view().permuted_axes(IxDyn(&[1, 0, 2])).to_owned();
    assert!(
        !permuted.is_standard_layout(),
        "the test input must not be in C order"
    );
    let c_order: Tensor = permuted.as_standard_layout().into_owned();

    let out = dense_2_to_3_ramp().forward(&permuted).unwrap();
    let want = dense_2_to_3_ramp().forward(&c_order).unwrap();

    assert_eq!(out.shape(), &[3, 2, 3]);
    assert_allclose(&out, &want, 1e-4_f32);
}

/// The rank-3 forward pass matches a reference that runs 1 slice at a time
///
/// 1 product over the folded rows sums the same terms as 1 product for each slice, but the
/// backend blocks the folded product differently. The 2 results agree to about 1 part in a
/// million, so this comparison needs a tolerance
#[test]
fn dense_forward_rank_3_matches_a_per_slice_reference() {
    let (batch, steps, features, units) = (3, 5, 8, 6);
    let mut d = Dense::new(features, units, Linear::new()).unwrap();
    let w: Vec<f32> = (0..features * units)
        .map(|i| (i % 7) as f32 * 0.37 - 1.1)
        .collect();
    let b: Vec<f32> = (0..units).map(|i| i as f32 * 0.11 - 0.3).collect();
    d.set_weights(
        Array2::from_shape_vec((features, units), w.clone()).unwrap(),
        Array2::from_shape_vec((1, units), b.clone()).unwrap(),
    )
    .unwrap();

    let data: Vec<f32> = (0..batch * steps * features)
        .map(|i| (i % 11) as f32 * 0.29 - 1.6)
        .collect();
    let x = t3(batch, steps, features, data.clone());
    let out = d.forward(&x).unwrap();
    assert_eq!(out.shape(), &[batch, steps, units]);

    let mut reference = Dense::new(features, units, Linear::new()).unwrap();
    reference
        .set_weights(
            Array2::from_shape_vec((features, units), w).unwrap(),
            Array2::from_shape_vec((1, units), b).unwrap(),
        )
        .unwrap();

    for row in 0..batch * steps {
        let slice = data[row * features..(row + 1) * features].to_vec();
        let want = reference.predict(&t2(1, features, slice)).unwrap();
        for unit in 0..units {
            let got = out.as_slice().expect("C order")[row * units + unit];
            let expected = want.as_slice().expect("C order")[unit];
            assert_abs_diff_eq!(got, expected, epsilon = 1e-4);
        }
    }
}

/// `predict` gives the same rank-3 result as `forward`, and writes no cache
#[test]
fn dense_predict_equals_forward_rank_3() {
    let mut d = dense_2_to_3_ramp();
    let x = t3(2, 3, 2, (1..=12).map(|v| v as f32).collect());

    let predicted = d.predict(&x).unwrap();
    let forwarded = d.forward(&x).unwrap();

    assert_eq!(predicted.shape(), &[2, 3, 3]);
    assert_allclose(&predicted, &forwarded, 1e-6_f32);
}

/// A rank-3 Softmax output normalizes over the units, 1 lane at a time
///
/// The fold keeps each last-axis lane whole, so the softmax never mixes 2 timesteps. The
/// rank-3 result is also bit-identical to the same values given as rank 2
#[test]
fn dense_rank_3_softmax_normalizes_each_last_axis_lane() {
    use ndarray::Axis;
    use rustyml::neural_network::layers::activation::softmax::Softmax;

    let data: Vec<f32> = (0..12).map(|v| v as f32 * 0.5 - 2.0).collect();
    let mut d = Dense::new(2, 3, Softmax::new()).unwrap();
    let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((1, 3), vec![0.5, -0.5, 1.0]).unwrap();
    d.set_weights(w.clone(), b.clone()).unwrap();

    let out = d.forward(&t3(2, 3, 2, data.clone())).unwrap();
    assert_eq!(out.shape(), &[2, 3, 3]);
    for lane in out.lanes(Axis(2)) {
        assert_abs_diff_eq!(lane.sum(), 1.0_f32, epsilon = 1e-6);
    }

    let mut flat = Dense::new(2, 3, Softmax::new()).unwrap();
    flat.set_weights(w, b).unwrap();
    let want = flat.forward(&t2(6, 2, data)).unwrap();
    let got_values: Vec<f32> = out.iter().cloned().collect();
    let want_values: Vec<f32> = want.iter().cloned().collect();
    assert_eq!(got_values, want_values, "the fold keeps every lane whole");
}

/// `output_shape` reports the rank of the last input it saw
///
/// Dense(4) after an input of shape (batch, 5, 7) reports "(None, 5, 4)". Before the first
/// forward pass only the unit count is known
#[test]
fn dense_output_shape_reports_the_real_rank() {
    use ndarray::{ArrayD, IxDyn};

    let mut d = Dense::new(7, 4, Linear::new()).unwrap();
    assert_eq!(d.output_shape(), "(None, 4)");

    d.forward(&ArrayD::zeros(IxDyn(&[2, 7]))).unwrap();
    assert_eq!(d.output_shape(), "(None, 4)");

    d.forward(&ArrayD::zeros(IxDyn(&[2, 5, 7]))).unwrap();
    assert_eq!(d.output_shape(), "(None, 5, 4)");

    d.forward(&ArrayD::zeros(IxDyn(&[3, 2, 5, 7]))).unwrap();
    assert_eq!(d.output_shape(), "(None, 2, 5, 4)");
}

/// A rank-3 Dense gives the same values on both sides of every tuning gate
///
/// The fold changes the row count that the gated activation pass sees, and the parallel
/// branch splits that row count into tasks. Both branches must write the same bits
#[test]
fn dense_rank_3_matches_across_the_tuning_gates() {
    use rustyml::neural_network::layers::activation::softmax::Softmax;

    let data: Vec<f32> = (0..24).map(|v| v as f32 * 0.375 - 4.0).collect();
    let x = t3(4, 3, 2, data);
    let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec((1, 3), vec![0.5, -0.5, 1.0]).unwrap();

    let run = |value: usize| {
        let _gates = GateGuard::set_all(value).with_split_cap(1);
        let mut d = Dense::new(2, 3, Softmax::new()).unwrap();
        d.set_weights(w.clone(), b.clone()).unwrap();
        let out = d.forward(&x).unwrap();
        let grad = d.backward(&Tensor::ones(out.raw_dim())).unwrap();
        let params = d.parameters();
        (
            out.iter().cloned().collect::<Vec<f32>>(),
            grad.iter().cloned().collect::<Vec<f32>>(),
            params[0].grad.to_vec(),
            params[1].grad.to_vec(),
        )
    };

    let serial = run(usize::MAX);
    let parallel = run(0);
    assert_eq!(serial, parallel, "the branch must not move the values");
}

// Dense: error paths

#[test]
fn dense_forward_rejects_rank_1_input() {
    let mut d = Dense::new(3, 2, Linear::new()).unwrap();
    let x = Array::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
    let result = d.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 1D input, got {:?}",
        result
    );
}

/// A last axis that differs from `input_dim` returns InvalidInput
///
/// The leading axes fold into the row axis, so a wrong last axis can still leave a valid
/// element count. Without this check a 2-feature layer would fold [2, 4, 7] into [28, 2] and
/// give a silently wrong result
#[test]
fn dense_rejects_a_last_axis_that_is_not_the_input_dim() {
    let mut d = Dense::new(2, 2, Linear::new()).unwrap();

    let rank_3 = t3(2, 4, 7, vec![0.5; 56]);
    let result = d.forward(&rank_3);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-3 last axis of 7, got {:?}",
        result
    );

    let rank_2 = t2(2, 7, vec![0.5; 14]);
    let result = d.forward(&rank_2);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for a rank-2 last axis of 7, got {:?}",
        result
    );

    let result = d.predict(&rank_3);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput from predict, got {:?}",
        result
    );
}

/// A gradient of the wrong rank returns ShapeMismatch, even where it folds to the right
/// matrix
#[test]
fn dense_backward_rank_3_rejects_a_folded_gradient() {
    let mut d = dense_2_to_3_ramp();
    d.forward(&t3(2, 3, 2, (1..=12).map(|v| v as f32).collect()))
        .unwrap();

    // [6, 3] holds the same 18 values as the cached [2, 3, 3] output, but it is not that shape
    let folded = t2(6, 3, (1..=18).map(|v| v as f32).collect());
    let result = d.backward(&folded);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

#[test]
fn dense_backward_before_forward_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new()).unwrap();
    // no forward called yet
    let grad = t2(1, 2, vec![1.0, 1.0]);
    let result = d.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

/// A wrong-shaped upstream gradient returns ShapeMismatch instead of panicking
#[test]
fn dense_backward_wrong_grad_shape_returns_err() {
    let mut d = Dense::new(3, 2, Linear::new()).unwrap();
    // Valid forward establishes the cached 2D output of shape [1, 2]
    let x = t2(1, 3, vec![1.0, 2.0, 3.0]);
    d.forward(&x).unwrap();
    // Feed a 3D gradient: backward must reject it, not panic
    let bad_grad = Array::from_shape_vec((1, 2, 1), vec![1.0_f32, 1.0])
        .unwrap()
        .into_dyn();
    let result = d.backward(&bad_grad);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {:?}",
        result
    );
}

#[test]
fn dense_set_weights_wrong_weight_shape_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new()).unwrap();
    // The correct shape is (2, 2). This test uses (3, 2), which must fail.
    let w_bad = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let b_ok = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let result = d.set_weights(w_bad, b_ok);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

#[test]
fn dense_set_weights_wrong_bias_shape_returns_err() {
    let mut d = Dense::new(2, 2, Linear::new()).unwrap();
    let w_ok = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    // The correct bias shape is (1, 2). This test uses (1, 3), which must fail.
    let b_bad = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let result = d.set_weights(w_ok, b_bad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::WeightShape { .. }))
        ),
        "expected WeightShape error, got {:?}",
        result
    );
}

// Dense: the named weights carry the shapes the layer declares

#[test]
fn dense_weights_carry_the_declared_shapes() {
    let mut d = Dense::new(3, 4, Linear::new()).unwrap();
    // inject known weights so exact values can be asserted too
    let w = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let b = Array2::from_shape_vec((1, 4), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    d.set_weights(w.clone(), b.clone()).unwrap();

    let kernel = named(&d, "kernel");
    let bias = named(&d, "bias");
    assert_eq!(kernel.shape(), &[3, 4]);
    assert_eq!(bias.shape(), &[1, 4]);
    // spot-check values
    assert_abs_diff_eq!(kernel[[0, 0]], 1.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(kernel[[2, 3]], 12.0_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(bias[[0, 1]], 0.2_f32, epsilon = 1e-6);
}

// Dense: backward restores the correct grad shape after forward

/// backward returns a gradient matching the input shape (values covered by gradient_check.rs)
#[test]
fn dense_backward_output_shape_matches_input() {
    let mut d = Dense::new(2, 3, Linear::new()).unwrap();
    let x = t2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = d.forward(&x).unwrap();
    let ones = Tensor::ones(out.raw_dim());
    let grad = d.backward(&ones).unwrap();
    assert_eq!(
        grad.shape(),
        x.shape(),
        "backward shape must match input shape"
    );
}

// Dense: layer_type string

#[test]
fn dense_layer_type_is_dense() {
    let d = Dense::new(2, 2, Linear::new()).unwrap();
    assert_eq!(d.layer_type(), "Dense");
}

// Flatten: constructor validation

#[test]
fn flatten_new_rejects_fewer_than_2_dims() {
    let result = Flatten::new(vec![4]);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 1-D input_shape, got {:?}",
        result
    );
}

#[test]
fn flatten_new_rejects_zero_dim() {
    let result = Flatten::new(vec![2, 0, 4]);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for zero dim, got {:?}",
        result
    );
}

#[test]
fn flatten_new_accepts_valid_3d_shape() {
    Flatten::new(vec![2, 3, 4]).unwrap();
}

// Flatten: forward values, 3D input

/// Flatten of [2, 3, 4] yields [2, 12] with values preserved in row-major order
#[test]
fn flatten_forward_3d_correct_shape_and_values() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 12]);

    // row 0 should be 0..12, row 1 should be 12..24
    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// Flatten: forward values, 4D input

/// Flatten of [2, 2, 3, 4] yields [2, 24] with values preserved in row-major order
#[test]
fn flatten_forward_4d_correct_shape_and_values() {
    let data: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let x = t4(2, 2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 24]);

    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// Flatten: forward values, 5D input

/// Flatten of [2, 2, 2, 3, 4] yields [2, 48] with values preserved
#[test]
fn flatten_forward_5d_correct_shape_and_values() {
    use ndarray::Array5;
    let data: Vec<f32> = (0..96).map(|v| v as f32).collect();
    let x = Array5::from_shape_vec((2, 2, 2, 3, 4), data.clone())
        .unwrap()
        .into_dyn();

    let mut fl = Flatten::new(vec![2, 2, 2, 3, 4]).unwrap();
    let out = fl.forward(&x).unwrap();

    assert_eq!(out.shape(), &[2, 48]);

    let out_slice = out.as_slice().expect("output not contiguous");
    for (i, &val) in out_slice.iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32, epsilon = 1e-6);
    }
}

// Flatten: backward restores the original shape and values

/// backward returns the original 3D shape with gradient values matching grad_flat reshaped
#[test]
fn flatten_backward_restores_3d_shape_and_values() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let _out = fl.forward(&x).unwrap();

    let grad_flat_data: Vec<f32> = (0..24).map(|v| (v as f32) * 2.0).collect();
    let grad_flat = t2(2, 12, grad_flat_data.clone());

    let grad_input = fl.backward(&grad_flat).unwrap();

    assert_eq!(grad_input.shape(), &[2, 3, 4]);

    let gs = grad_input.as_slice().expect("grad not contiguous");
    for (i, &val) in gs.iter().enumerate() {
        assert_abs_diff_eq!(val, (i as f32) * 2.0, epsilon = 1e-6);
    }
}

/// backward restores the original 4D shape with gradient values matching grad_flat reshaped
#[test]
fn flatten_backward_restores_4d_shape_and_values() {
    let data: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let x = t4(2, 2, 3, 4, data.clone());

    let mut fl = Flatten::new(vec![2, 2, 3, 4]).unwrap();
    let _out = fl.forward(&x).unwrap();

    let grad_flat_data: Vec<f32> = (0..48).map(|v| -(v as f32)).collect();
    let grad_flat = t2(2, 24, grad_flat_data.clone());

    let grad_input = fl.backward(&grad_flat).unwrap();

    assert_eq!(grad_input.shape(), &[2, 2, 3, 4]);

    let gs = grad_input.as_slice().expect("grad not contiguous");
    for (i, &val) in gs.iter().enumerate() {
        assert_abs_diff_eq!(val, -(i as f32), epsilon = 1e-6);
    }
}

// Flatten: error paths

#[test]
fn flatten_forward_rejects_2d_input() {
    let mut fl = Flatten::new(vec![2, 3]).unwrap();
    let x = t2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = fl.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 2D input, got {:?}",
        result
    );
}

#[test]
fn flatten_forward_rejects_6d_input() {
    use ndarray::ArrayD;
    let mut fl = Flatten::new(vec![1, 2, 2, 2, 2]).unwrap();
    // build a 6D tensor manually
    let x: Tensor = ArrayD::zeros(vec![1, 2, 2, 2, 2, 2]);
    let result = fl.forward(&x);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for 6D input, got {:?}",
        result
    );
}

#[test]
fn flatten_backward_before_forward_returns_err() {
    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    // No forward call happened yet, so the layer's input_cache is None.
    let grad = t2(2, 12, vec![0.0_f32; 24]);
    let result = fl.backward(&grad);
    assert!(
        matches!(
            result,
            Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(_)))
        ),
        "expected ForwardPassNotRun, got {:?}",
        result
    );
}

// Flatten: predict equal to forward (no training-mode difference)

#[test]
fn flatten_predict_equals_forward() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = t3(2, 3, 4, data);

    let mut fl = Flatten::new(vec![2, 3, 4]).unwrap();
    let fwd = fl.forward(&x).unwrap();
    let pred = fl.predict(&x).unwrap();
    assert_allclose(&fwd, &pred, 1e-6_f32);
}

// Flatten: the named weight list is empty (no trainable parameters)

#[test]
fn flatten_weights_is_empty() {
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert!(fl.weights().is_empty(), "Flatten must expose no weight");
}

// Flatten: param_count is NoTrainable

#[test]
fn flatten_param_count_is_no_trainable() {
    use rustyml::neural_network::layers::ParamCounts;
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert_eq!(fl.param_count(), ParamCounts::none());
}

// Flatten: layer_type string

#[test]
fn flatten_layer_type_is_flatten() {
    let fl = Flatten::new(vec![2, 3, 4]).unwrap();
    assert_eq!(fl.layer_type(), "Flatten");
}
